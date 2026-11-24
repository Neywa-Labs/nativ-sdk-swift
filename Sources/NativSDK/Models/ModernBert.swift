import Foundation
import MLX
import MLXNN

struct ModernBertConfiguration: Decodable {
    let vocabularySize: Int
    let hiddenSize: Int
    let hiddenLayers: Int
    let intermediateSize: Int
    let attentionHeads: Int
    let maximumLength: Int
    let normEpsilon: Float
    let normBias: Bool
    let attentionBias: Bool
    let mlpBias: Bool
    let globalTheta: Float
    let localTheta: Float
    let globalEvery: Int
    let localWindow: Int
    enum CodingKeys: String, CodingKey {
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case maximumLength = "max_position_embeddings"
        case normEpsilon = "norm_eps"
        case normBias = "norm_bias"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case globalTheta = "global_rope_theta"
        case localTheta = "local_rope_theta"
        case globalEvery = "global_attn_every_n_layers"
        case localWindow = "local_attention"
        case ropeParameters = "rope_parameters"
    }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        maximumLength = try c.decodeIfPresent(Int.self, forKey: .maximumLength) ?? 8192
        normEpsilon = try c.decodeIfPresent(Float.self, forKey: .normEpsilon) ?? 1e-5
        normBias = try c.decodeIfPresent(Bool.self, forKey: .normBias) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        let rope = try c.decodeIfPresent(
            [String: [String: JSONValue]].self, forKey: .ropeParameters)
        func theta(_ kind: String) -> Float? {
            switch rope?[kind]?["rope_theta"] {
            case .double(let value): return Float(value)
            case .int(let value): return Float(value)
            default: return nil
            }
        }
        globalTheta =
            try theta("full_attention") ?? c.decodeIfPresent(Float.self, forKey: .globalTheta)
            ?? 160000
        localTheta =
            try theta("sliding_attention") ?? c.decodeIfPresent(Float.self, forKey: .localTheta)
            ?? 10000
        globalEvery = try c.decodeIfPresent(Int.self, forKey: .globalEvery) ?? 3
        localWindow = try c.decodeIfPresent(Int.self, forKey: .localWindow) ?? 128
        guard vocabularySize > 0, hiddenSize > 0, attentionHeads > 0,
            hiddenSize % attentionHeads == 0,
            (hiddenSize / attentionHeads) % 2 == 0, hiddenLayers > 0, intermediateSize > 0,
            maximumLength > 0, globalEvery > 0, localWindow >= 0,
            normEpsilon.isFinite, normEpsilon > 0,
            globalTheta.isFinite, globalTheta > 0, localTheta.isFinite, localTheta > 0
        else {
            throw ModelExecutionError.invalidOptions("Invalid ModernBERT configuration")
        }
    }
}

private enum ModernBert {
    final class Embeddings: Module {
        @ModuleInfo(key: "tok_embeddings") var tokens: Embedding
        let norm: LayerNorm
        init(_ c: ModernBertConfiguration) {
            _tokens.wrappedValue = Embedding(
                embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
            norm = LayerNorm(dimensions: c.hiddenSize, eps: c.normEpsilon, bias: c.normBias)
        }
        func callAsFunction(_ ids: MLXArray) -> MLXArray { norm(tokens(ids)) }
    }
    final class Attention: Module {
        @ModuleInfo(key: "Wqkv") var qkv: Linear
        @ModuleInfo(key: "Wo") var output: Linear
        let rope: RoPE
        let heads: Int
        let dimensions: Int
        let local: Bool
        init(_ c: ModernBertConfiguration, index: Int) {
            heads = c.attentionHeads
            dimensions = c.hiddenSize / heads
            local = index % c.globalEvery != 0
            _qkv.wrappedValue = Linear(c.hiddenSize, c.hiddenSize * 3, bias: c.attentionBias)
            _output.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: c.attentionBias)
            rope = RoPE(
                dimensions: dimensions, traditional: false,
                base: local ? c.localTheta : c.globalTheta)
        }
        func callAsFunction(_ x: MLXArray, global: MLXArray, sliding: MLXArray) -> MLXArray {
            let parts = qkv(x).reshaped(x.dim(0), -1, 3, heads, dimensions).transposed(
                0, 3, 2, 1, 4
            ).split(parts: 3, axis: 2)
            let q = rope(parts[0].squeezed(axis: 2))
            let k = rope(parts[1].squeezed(axis: 2))
            let v = parts[2].squeezed(axis: 2)
            let h = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v,
                scale: pow(Float(dimensions), -0.5), mask: .array(local ? sliding : global))
            return output(h.transposed(0, 2, 1, 3).reshaped(x.dim(0), x.dim(1), -1))
        }
    }
    final class MLP: Module {
        @ModuleInfo(key: "Wi") var input: Linear
        @ModuleInfo(key: "Wo") var output: Linear
        init(_ c: ModernBertConfiguration) {
            _input.wrappedValue = Linear(c.hiddenSize, c.intermediateSize * 2, bias: c.mlpBias)
            _output.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: c.mlpBias)
        }
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let parts = input(x).split(parts: 2, axis: -1)
            return output(gelu(parts[0]) * parts[1])
        }
    }
    final class Layer: Module {
        @ModuleInfo(key: "attn_norm") var attentionNorm: LayerNorm?
        let attn: Attention
        @ModuleInfo(key: "mlp_norm") var mlpNorm: LayerNorm
        let mlp: MLP
        init(_ c: ModernBertConfiguration, index: Int) {
            if index > 0 {
                _attentionNorm.wrappedValue = LayerNorm(
                    dimensions: c.hiddenSize, eps: c.normEpsilon, bias: c.normBias)
            }
            attn = Attention(c, index: index)
            _mlpNorm.wrappedValue = LayerNorm(
                dimensions: c.hiddenSize, eps: c.normEpsilon, bias: c.normBias)
            mlp = MLP(c)
        }
        func callAsFunction(_ x: MLXArray, global: MLXArray, sliding: MLXArray) -> MLXArray {
            let h = x + attn(attentionNorm?(x) ?? x, global: global, sliding: sliding)
            return h + mlp(mlpNorm(h))
        }
    }
}

final class ModernBertEmbeddingModel: Module, EmbeddingModel {
    private let embeddings: ModernBert.Embeddings
    private let layers: [ModernBert.Layer]
    @ModuleInfo(key: "final_norm") var norm: LayerNorm
    let config: ModernBertConfiguration
    let pooling: EmbeddingPooling
    var maximumSequenceLength: Int { config.maximumLength }
    init(_ config: ModernBertConfiguration, pooling: EmbeddingPooling) {
        self.config = config
        self.pooling = pooling
        embeddings = ModernBert.Embeddings(config)
        layers = (0 ..< config.hiddenLayers).map { ModernBert.Layer(config, index: $0) }
        _norm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.normEpsilon, bias: config.normBias)
    }
    func embed(_ input: LMInput) throws -> EmbeddingOutput {
        let (ids, valid) = try embeddingTokens(
            input, vocabularySize: config.vocabularySize, maximumLength: maximumSequenceLength)
        return .pooled(
            normalizeEmbeddings(pooling.apply(try encode(ids, valid: valid), mask: valid)))
    }
    /// Unpooled encoder states, shared with decision models using a ModernBERT backbone.
    func encode(_ ids: MLXArray, valid: MLXArray) throws -> MLXArray {
        var h = embeddings(ids)
        let global = embeddingAttentionMask(valid)
        let positions = arange(ids.dim(1))
        let window =
            abs(positions.expandedDimensions(axis: 1) - positions) .<= config.localWindow / 2
        let sliding = global .&& window.reshaped(1, 1, ids.dim(1), ids.dim(1))
        for layer in layers {
            try Task.checkCancellation()
            h = layer(h, global: global, sliding: sliding)
        }
        return norm(h)
    }
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            let key = key.hasPrefix("model.") ? String(key.dropFirst(6)) : key
            if !["head.", "decoder.", "classifier.", "pooler."].contains(where: key.hasPrefix) {
                result[key] = value
            }
        }
        return result
    }
}
