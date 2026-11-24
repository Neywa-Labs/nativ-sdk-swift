import Foundation
import MLX
import MLXNN

class ExaoneAttentionModule: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "out_proj") var wo: Linear

    let rope: RoPELayer

    init(_ config: ExaoneConfiguration) {
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.headDimensions = config.resolvedHeadDimensions
        self.scale = pow(Float(headDimensions), -0.5)

        _wq.wrappedValue = Linear(
            config.hiddenSize, heads * headDimensions, bias: config.attentionBias)
        _wk.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDimensions, bias: config.attentionBias)
        _wv.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDimensions, bias: config.attentionBias)
        _wo.wrappedValue = Linear(
            heads * headDimensions, config.hiddenSize, bias: config.attentionBias)

        self.rope = initializeRope(
            dims: headDimensions,
            base: config.ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

/// Holds the attention and nothing else, so the published `attn.attention.` path binds.
class ExaoneAttention: Module {
    @ModuleInfo(key: "attention") var attention: ExaoneAttentionModule

    init(_ config: ExaoneConfiguration) {
        _attention.wrappedValue = ExaoneAttentionModule(config)
    }
}

class ExaoneMLP: Module, UnaryLayer {
    @ModuleInfo(key: "c_fc_0") var gate: Linear
    @ModuleInfo(key: "c_fc_1") var up: Linear
    @ModuleInfo(key: "c_proj") var down: Linear

    init(_ config: ExaoneConfiguration) {
        _gate.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        _up.wrappedValue = Linear(
            config.hiddenSize, config.intermediateSize, bias: config.mlpBias)
        _down.wrappedValue = Linear(
            config.intermediateSize, config.hiddenSize, bias: config.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class ExaoneBlock: Module {
    @ModuleInfo(key: "ln_1") var ln1: RMSNorm
    @ModuleInfo(key: "attn") var attn: ExaoneAttention
    @ModuleInfo(key: "ln_2") var ln2: RMSNorm
    let mlp: ExaoneMLP

    init(_ config: ExaoneConfiguration) {
        _ln1.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
        _attn.wrappedValue = ExaoneAttention(config)
        _ln2.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
        self.mlp = ExaoneMLP(config)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attn.attention(ln1(x), mask: mask, cache: cache)
        return h + mlp(ln2(h))
    }
}

public class ExaoneTransformer: Module {
    @ModuleInfo(key: "wte") var wte: Embedding

    let h: [ExaoneBlock]

    @ModuleInfo(key: "ln_f") var lnF: RMSNorm

    public init(_ config: ExaoneConfiguration) {
        precondition(config.vocabularySize > 0)
        _wte.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.h = (0 ..< config.layers).map { _ in ExaoneBlock(config) }
        _lnF.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden = wte(inputs)
        let mask = createAttentionMask(h: hidden, cache: cache?.first)

        for (i, layer) in h.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[i])
        }

        return lnF(hidden)
    }
}

public class ExaoneLanguageModel: Module {
    public let transformer: ExaoneTransformer

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: ExaoneConfiguration

    public init(_ config: ExaoneConfiguration) {
        self.configuration = config
        self.transformer = ExaoneTransformer(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = transformer(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return transformer.wte.asLinear(out)
    }
}

public class ExaoneModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: ExaoneLanguageModel

    let configuration: ExaoneConfiguration

    public init(_ config: ExaoneConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.layers).map { _ in config.kvHeads }
        _languageModel.wrappedValue = ExaoneLanguageModel(config)
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
            // `tie_word_embeddings` defaults to true here, unlike most of the zoo, so a
            // checkpoint that ships a head this model does not build must not be bound.
            if configuration.tieWordEmbeddings, name.hasPrefix("lm_head.") {
                continue
            }
            sanitized["language_model." + name] = value
        }
        return sanitized
    }
}

public struct ExaoneConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var layers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var vocabularySize: Int
    var ropeTheta: Float
    var layerNormEpsilon: Float
    var kvHeads: Int
    var headDimensions: Int?
    var maxPositionEmbeddings: Int?
    var ropeTraditional: Bool
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool
    var attentionBias: Bool
    var mlpBias: Bool

    /// `head_dim` is optional upstream and falls back to the even split.
    var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case layers = "num_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case layerNormEpsilon = "layer_norm_epsilon"
        case kvHeads = "num_key_value_heads"
        case headDimensions = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "exaone"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        layers = try c.decode(Int.self, forKey: .layers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        layerNormEpsilon = try c.decode(Float.self, forKey: .layerNormEpsilon)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDimensions = try c.decodeIfPresent(Int.self, forKey: .headDimensions)
        maxPositionEmbeddings = try c.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        // Defaults to true upstream, which is the opposite of most of this zoo.
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
    }
}

extension ExaoneModel: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.transformer.h
    }
}
