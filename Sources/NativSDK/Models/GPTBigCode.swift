import Foundation
import MLX
import MLXNN

class GPTBigCodeAttention: Module {
    let dimensions: Int
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let kvDimensions: Int
    let scale: Float

    @ModuleInfo(key: "c_attn") var cAttn: Linear
    @ModuleInfo(key: "c_proj") var cProj: Linear

    init(_ config: GPTBigCodeConfiguration) {
        self.dimensions = config.embeddingDimensions
        self.heads = config.heads
        // Recomputed from `multi_query` rather than read from num_key_value_heads, which is
        // what upstream's forward does; the config field only sizes the cache.
        self.kvHeads = config.multiQuery ? 1 : config.heads
        self.headDimensions = config.embeddingDimensions / config.heads
        self.kvDimensions = kvHeads * headDimensions
        self.scale = pow(Float(headDimensions), -0.5)

        _cAttn.wrappedValue = Linear(
            dimensions, dimensions + 2 * kvDimensions, bias: config.attentionBias)
        _cProj.wrappedValue = Linear(dimensions, dimensions, bias: config.attentionBias)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let fused = cAttn(x)
        let queries = fused[0..., 0..., ..<dimensions]
            .reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        let keys = fused[0..., 0..., dimensions ..< (dimensions + kvDimensions)]
            .reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        let values = fused[0..., 0..., (dimensions + kvDimensions)...]
            .reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return cProj(output)
    }
}

class GPTBigCodeMLP: Module, UnaryLayer {
    @ModuleInfo(key: "c_fc") var cFC: Linear
    @ModuleInfo(key: "c_proj") var cProj: Linear

    init(_ config: GPTBigCodeConfiguration) {
        _cFC.wrappedValue = Linear(
            config.embeddingDimensions, config.innerDimensions, bias: config.mlpBias)
        _cProj.wrappedValue = Linear(
            config.innerDimensions, config.embeddingDimensions, bias: config.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Exact gelu, not the tanh approximation gpt2 uses in this same package.
        cProj(gelu(cFC(x)))
    }
}

class GPTBigCodeBlock: Module {
    @ModuleInfo(key: "attn") var attention: GPTBigCodeAttention
    let mlp: GPTBigCodeMLP

    @ModuleInfo(key: "ln_1") var ln1: LayerNorm
    @ModuleInfo(key: "ln_2") var ln2: LayerNorm

    init(_ config: GPTBigCodeConfiguration) {
        _attention.wrappedValue = GPTBigCodeAttention(config)
        self.mlp = GPTBigCodeMLP(config)
        _ln1.wrappedValue = LayerNorm(
            dimensions: config.embeddingDimensions, eps: config.layerNormEpsilon)
        _ln2.wrappedValue = LayerNorm(
            dimensions: config.embeddingDimensions, eps: config.layerNormEpsilon)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(ln1(x), mask: mask, cache: cache)
        return h + mlp(ln2(h))
    }
}

public class GPTBigCodeTransformer: Module {
    @ModuleInfo(key: "wte") var wte: Embedding
    @ModuleInfo(key: "wpe") var wpe: Embedding

    let h: [GPTBigCodeBlock]

    @ModuleInfo(key: "ln_f") var lnF: LayerNorm

    public init(_ config: GPTBigCodeConfiguration) {
        precondition(config.vocabularySize > 0)
        _wte.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.embeddingDimensions)
        _wpe.wrappedValue = Embedding(
            embeddingCount: config.positions, dimensions: config.embeddingDimensions)
        self.h = (0 ..< config.layers).map { _ in GPTBigCodeBlock(config) }
        _lnF.wrappedValue = LayerNorm(
            dimensions: config.embeddingDimensions, eps: config.layerNormEpsilon)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let L = inputs.dim(-1)
        var hidden = wte(inputs)

        // Positions are looked up, so a warm cache has to supply the offset or every
        // generated token embeds as if it were the first.
        let offset = cache?.first?.offset ?? 0
        let mask = createAttentionMask(h: hidden, cache: cache?.first)
        hidden = hidden + wpe(MLXArray(offset ..< offset + L))

        for (i, layer) in h.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[i])
        }

        return lnF(hidden)
    }
}

public class GPTBigCodeLanguageModel: Module {
    public let transformer: GPTBigCodeTransformer

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(_ config: GPTBigCodeConfiguration) {
        self.transformer = GPTBigCodeTransformer(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                config.embeddingDimensions, config.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = transformer(inputs, cache: cache)
        if let lmHead {
            return lmHead(hidden)
        }
        return transformer.wte.asLinear(hidden)
    }
}

public class GPTBigCodeModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: GPTBigCodeLanguageModel

    let configuration: GPTBigCodeConfiguration

    public init(_ config: GPTBigCodeConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.layers).map { _ in config.resolvedKVHeads }
        _languageModel.wrappedValue = GPTBigCodeLanguageModel(config)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            var name = key
            if name.hasPrefix("language_model.") {
                name = String(name.dropFirst("language_model.".count))
            }
            if configuration.tieWordEmbeddings, name.hasPrefix("lm_head.") {
                continue
            }
            sanitized["language_model." + name] = value
        }
        return sanitized
    }
}

public struct GPTBigCodeConfiguration: Codable, Sendable {
    var modelType: String
    var embeddingDimensions: Int
    var layers: Int
    var innerDimensions: Int
    var heads: Int
    var positions: Int
    var layerNormEpsilon: Float
    var vocabularySize: Int
    var kvHeads: Int?
    var multiQuery: Bool
    var attentionBias: Bool
    var mlpBias: Bool
    var tieWordEmbeddings: Bool

    /// One key-value head under `multi_query`, otherwise as many as there are heads.
    var resolvedKVHeads: Int {
        kvHeads ?? (multiQuery ? 1 : heads)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case embeddingDimensions = "n_embd"
        case layers = "n_layer"
        case innerDimensions = "n_inner"
        case heads = "n_head"
        case positions = "n_positions"
        case layerNormEpsilon = "layer_norm_epsilon"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case multiQuery = "multi_query"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gpt_bigcode"
        embeddingDimensions = try c.decode(Int.self, forKey: .embeddingDimensions)
        layers = try c.decode(Int.self, forKey: .layers)
        innerDimensions = try c.decode(Int.self, forKey: .innerDimensions)
        heads = try c.decode(Int.self, forKey: .heads)
        positions = try c.decode(Int.self, forKey: .positions)
        layerNormEpsilon = try c.decode(Float.self, forKey: .layerNormEpsilon)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads)
        multiQuery = try c.decodeIfPresent(Bool.self, forKey: .multiQuery) ?? true
        // Both bias flags default true here, which is the opposite of most of this zoo.
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? true
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? true
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true

        guard embeddingDimensions % heads == 0 else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: c.codingPath,
                    debugDescription:
                        "n_embd \(embeddingDimensions) must be divisible by n_head \(heads)"))
        }
    }
}

extension GPTBigCodeModel: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.transformer.h
    }
}
