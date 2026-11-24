import Foundation
import MLX
import MLXNN

class Cohere2Attention: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let scale: Float

    /// Sliding layers rope; the global layers this pattern leaves are NoPE.
    let useRotary: Bool

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    init(_ config: Cohere2Configuration, layerIndex: Int) {
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.headDimensions = config.headDim
        self.scale = pow(Float(config.headDim), -0.5)
        self.useRotary = config.isSlidingLayer(layerIndex)

        _wq.wrappedValue = Linear(
            config.hiddenSize, heads * headDimensions, bias: config.attentionBias)
        _wk.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDimensions, bias: config.attentionBias)
        _wv.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDimensions, bias: config.attentionBias)
        _wo.wrappedValue = Linear(
            heads * headDimensions, config.hiddenSize, bias: config.attentionBias)

        self.rope = RoPE(dimensions: headDimensions, traditional: true, base: config.ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        if useRotary {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }

        // Float16 queries need float32 attention to avoid exponent overflow.
        let promote = queries.dtype == .float16
        let output = attentionWithCacheUpdate(
            queries: promote ? queries.asType(.float32) : queries,
            keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .asType(queries.dtype)
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

class Cohere2Block: Module {
    @ModuleInfo(key: "self_attn") var attention: Cohere2Attention
    let mlp: SwiGLUMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm

    init(_ config: Cohere2Configuration, layerIndex: Int) {
        _attention.wrappedValue = Cohere2Attention(config, layerIndex: layerIndex)
        self.mlp = SwiGLUMLP(
            inputDimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        _inputLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps, affine: true,
            bias: config.layerNormBias)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        // Attention and MLP run in parallel from the same normalized input.
        let h = inputLayerNorm(x)
        return attention(h, mask: mask, cache: cache) + mlp(h) + x
    }
}

public class Cohere2ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [Cohere2Block]
    let norm: LayerNorm
    let config: Cohere2Configuration

    public init(_ config: Cohere2Configuration) {
        precondition(config.vocabularySize > 0)
        self.config = config

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { Cohere2Block(config, layerIndex: $0) }
        self.norm = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps, affine: true,
            bias: config.layerNormBias)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            let mask = createAttentionMask(
                h: h, cache: layerCache,
                windowSize: config.isSlidingLayer(i) ? config.slidingWindow : nil)
            h = layer(h, mask: mask, cache: layerCache)
        }

        return norm(h)
    }
}

public class Cohere2LanguageModel: Module {
    public let model: Cohere2ModelInner
    let config: Cohere2Configuration

    public init(_ config: Cohere2Configuration) {
        self.config = config
        self.model = Cohere2ModelInner(config)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model.embedTokens.asLinear(model(inputs, cache: cache)) * config.logitScale
    }
}

public class Cohere2Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: Cohere2LanguageModel

    let config: Cohere2Configuration

    public init(_ config: Cohere2Configuration) {
        self.config = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        _languageModel.wrappedValue = Cohere2LanguageModel(config)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    /// Sliding layers retain exactly `sliding_window` positions, with no pinned prefix.
    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try (0 ..< config.hiddenLayers).map { layerIndex in
            try makeHybridAttentionKVCache(
                configuration: configuration,
                slidingWindow: config.slidingWindow,
                usesSlidingWindow: config.isSlidingLayer(layerIndex))
        }
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

public struct Cohere2Configuration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var headDim: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var ropeTheta: Float
    var vocabularySize: Int
    var layerNormEps: Float
    var logitScale: Float
    var attentionBias: Bool
    var layerNormBias: Bool
    var slidingWindow: Int
    var slidingWindowPattern: Int

    /// Whether layer `index` attends through the sliding window.
    ///
    /// Every `sliding_window_pattern`-th layer is global; the rest slide. The same
    /// predicate decides both the mask and whether the layer ropes at all, which is why it
    /// lives here rather than being restated in each.
    func isSlidingLayer(_ index: Int) -> Bool {
        (index + 1) % slidingWindowPattern != 0
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case headDim = "head_dim"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case vocabularySize = "vocab_size"
        case layerNormEps = "layer_norm_eps"
        case logitScale = "logit_scale"
        case attentionBias = "attention_bias"
        case layerNormBias = "layer_norm_bias"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "cohere2"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 50000
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 256_000
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
        logitScale = try c.decodeIfPresent(Float.self, forKey: .logitScale) ?? 0.0625
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        layerNormBias = try c.decodeIfPresent(Bool.self, forKey: .layerNormBias) ?? false
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 4096
        slidingWindowPattern =
            try c.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 4

        guard headDim * attentionHeads == hiddenSize else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: c.codingPath,
                    debugDescription:
                        "hidden_size \(hiddenSize) must equal head_dim \(headDim) times "
                        + "num_attention_heads \(attentionHeads)"))
        }
    }
}

extension Cohere2Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
