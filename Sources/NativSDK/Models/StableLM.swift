import Foundation
import MLX
import MLXNN

/// One LayerNorm weight per head, applied to a normalization that carries no affine term.
///
/// The weights are stored as a list so a published checkpoint's `norms.<i>.weight` keys
/// bind, and stacked at call time to `[heads, headDim]`, which broadcasts over the
/// `[batch, length, heads, headDim]` layout this is applied in -- before the transpose, so
/// each head is normalized over its own dimensions rather than across the whole projection.
class StableLMPerHeadLayerNorm: Module {
    let norms: [LayerNorm]
    let eps: Float

    init(headDimensions: Int, heads: Int, eps: Float) {
        self.norms = (0 ..< heads).map { _ in
            LayerNorm(dimensions: headDimensions, eps: eps, affine: true, bias: false)
        }
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let weight = MLX.stacked(norms.map { $0.weight! })
        return weight * MLXFast.layerNorm(x, weight: nil, bias: nil, eps: eps)
    }
}

class StableLMAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_layernorm") var qNorm: StableLMPerHeadLayerNorm?
    @ModuleInfo(key: "k_layernorm") var kNorm: StableLMPerHeadLayerNorm?

    let rope: RoPE

    /// How many of each head's dimensions rotate. The rest carry no position at all.
    let rotaryDimensions: Int

    init(_ config: StableLMConfiguration) {
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.headDimensions = config.hiddenSize / config.attentionHeads
        self.scale = pow(Float(headDimensions), -0.5)

        _wq.wrappedValue = Linear(
            config.hiddenSize, heads * headDimensions, bias: config.useQKVBias)
        _wk.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDimensions, bias: config.useQKVBias)
        _wv.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDimensions, bias: config.useQKVBias)
        _wo.wrappedValue = Linear(heads * headDimensions, config.hiddenSize, bias: false)

        self.rotaryDimensions = Int(config.partialRotaryFactor * Float(headDimensions))
        self.rope = RoPE(
            dimensions: rotaryDimensions, traditional: false, base: config.ropeTheta)

        if config.qkLayerNorm {
            _qNorm.wrappedValue = StableLMPerHeadLayerNorm(
                headDimensions: headDimensions, heads: heads, eps: config.layerNormEps)
            _kNorm.wrappedValue = StableLMPerHeadLayerNorm(
                headDimensions: headDimensions, heads: kvHeads, eps: config.layerNormEps)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x).reshaped(B, L, heads, -1)
        var keys = wk(x).reshaped(B, L, kvHeads, -1)
        if let qNorm, let kNorm {
            queries = qNorm(queries)
            keys = kNorm(keys)
        }
        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        // Upstream promotes both to float32 unconditionally here, not just for float16.
        let output = attentionWithCacheUpdate(
            queries: queries.asType(.float32),
            keys: keys.asType(.float32),
            values: values,
            cache: cache, scale: scale, mask: mask
        )
        .asType(values.dtype)
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

class StableLMBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: StableLMAttention
    let mlp: SwiGLUMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: LayerNorm?

    let useParallelResidual: Bool

    init(_ config: StableLMConfiguration) {
        _attention.wrappedValue = StableLMAttention(config)
        self.mlp = SwiGLUMLP(
            inputDimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        _inputLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.useParallelResidual = config.useParallelResidual
        if !config.useParallelResidual {
            _postAttentionLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let normalized = inputLayerNorm(x)
        let attended = attention(normalized, mask: mask, cache: cache)

        // Parallel feeds the MLP the same norm attention saw and sums both into the
        // residual; sequential runs it over the attention's output through a second norm.
        // The two share every weight except that norm, so the wrong one loads and answers.
        if useParallelResidual {
            return x + attended + mlp(normalized)
        }
        let h = x + attended
        guard let postAttentionLayerNorm else { return h }
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class StableLMModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [StableLMBlock]
    let norm: LayerNorm

    public init(_ config: StableLMConfiguration) {
        precondition(config.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in StableLMBlock(config) }
        self.norm = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class StableLMLanguageModel: Module {
    public let model: StableLMModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: StableLMConfiguration) {
        self.model = StableLMModelInner(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }
}

public class StableLMModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: StableLMLanguageModel

    public init(_ config: StableLMConfiguration) {
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        _languageModel.wrappedValue = StableLMLanguageModel(config)
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
            sanitized["language_model." + name] = value
        }
        return sanitized
    }
}

public struct StableLMConfiguration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var attentionHeads: Int
    var hiddenLayers: Int
    var kvHeads: Int
    var intermediateSize: Int
    var ropeTheta: Float
    var useQKVBias: Bool
    var partialRotaryFactor: Float
    var layerNormEps: Float
    var useParallelResidual: Bool
    var qkLayerNorm: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case attentionHeads = "num_attention_heads"
        case hiddenLayers = "num_hidden_layers"
        case kvHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case ropeTheta = "rope_theta"
        case useQKVBias = "use_qkv_bias"
        case partialRotaryFactor = "partial_rotary_factor"
        case layerNormEps = "layer_norm_eps"
        case useParallelResidual = "use_parallel_residual"
        case qkLayerNorm = "qk_layernorm"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "stablelm"
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        useQKVBias = try c.decode(Bool.self, forKey: .useQKVBias)
        partialRotaryFactor = try c.decode(Float.self, forKey: .partialRotaryFactor)
        layerNormEps = try c.decode(Float.self, forKey: .layerNormEps)
        useParallelResidual =
            try c.decodeIfPresent(Bool.self, forKey: .useParallelResidual) ?? false
        qkLayerNorm = try c.decodeIfPresent(Bool.self, forKey: .qkLayerNorm) ?? false

        guard hiddenSize % attentionHeads == 0 else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: c.codingPath,
                    debugDescription:
                        "hidden_size \(hiddenSize) must be divisible by num_attention_heads "
                        + "\(attentionHeads)"))
        }
    }
}

extension StableLMModel: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
