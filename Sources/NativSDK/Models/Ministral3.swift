import Foundation
import MLX
import MLXNN

/// Llama 4's position-dependent query temperature.
///
/// `1 + beta * log(1 + floor(position / window))` per position, shaped to broadcast over
/// `[batch, heads, length, dim]`. It is a step function, not a smooth one: every position
/// inside the first window gets exactly one, which is why a short-prompt test has to reach
/// past the window to see anything at all.
func llama4AttentionScale(
    size: Int, offset: MLXArray, beta: Float, maxPositionEmbeddings: Int
) -> MLXArray {
    let positions =
        offset.ndim > 0
        ? MLXArray(0 ..< size).asType(.float32)[.newAxis, 0...] + offset[0..., .newAxis]
        : MLXArray(0 ..< size).asType(.float32) + offset

    let scaling =
        1 + beta * MLX.log(1 + MLX.floor(positions / Float(maxPositionEmbeddings)))

    // A batched offset gives one row of positions per row of the batch.
    return scaling.ndim == 2
        ? scaling[0..., .newAxis, 0..., .newAxis]
        : scaling[0..., .newAxis]
}

class Ministral3Attention: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer

    init(_ args: Ministral3Configuration) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDimensions = args.resolvedHeadDimensions
        self.scale = pow(Float(headDimensions), -0.5)

        _wq.wrappedValue = Linear(args.hiddenSize, heads * headDimensions, bias: false)
        _wk.wrappedValue = Linear(args.hiddenSize, kvHeads * headDimensions, bias: false)
        _wv.wrappedValue = Linear(args.hiddenSize, kvHeads * headDimensions, bias: false)
        _wo.wrappedValue = Linear(heads * headDimensions, args.hiddenSize, bias: false)

        self.rope = initializeRope(
            dims: headDimensions,
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, attentionScale: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        // After the rotation, before the attention.
        queries = queries * attentionScale

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

class Ministral3Block: Module {
    @ModuleInfo(key: "self_attn") var attention: Ministral3Attention
    let mlp: SwiGLUMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let usesSliding: Bool

    init(_ args: Ministral3Configuration, usesSliding: Bool) {
        self.usesSliding = usesSliding
        _attention.wrappedValue = Ministral3Attention(args)
        self.mlp = SwiGLUMLP(
            inputDimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, attentionScale: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h =
            x
            + attention(
                inputLayerNorm(x), attentionScale: attentionScale, mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class Ministral3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [Ministral3Block]
    let norm: RMSNorm

    let configuration: Ministral3Configuration

    public init(_ args: Ministral3Configuration) {
        precondition(args.vocabularySize > 0)
        self.configuration = args
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = args.resolvedLayerTypes.map {
            Ministral3Block(args, usesSliding: $0 == "sliding_attention")
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        encode(inputs, cache: cache, penultimate: false, useQueryTemperature: true)
    }

    func encode(
        _ inputs: MLXArray, cache: [KVCache]? = nil,
        penultimate: Bool, useQueryTemperature: Bool
    ) -> MLXArray {
        var h = embedTokens(inputs)

        // Two masks, each built from a cache of its own kind. Building one from the wrong
        // kind of cache gives it the wrong offset once the two diverge in length.
        let firstFull = layers.firstIndex { !$0.usesSliding }
        let firstSliding = layers.firstIndex { $0.usesSliding }
        let fullMask =
            firstFull.map { createAttentionMask(h: h, cache: cache?[$0]) } ?? .none
        let slidingMask =
            firstSliding.map {
                createAttentionMask(
                    h: h, cache: cache?[$0], windowSize: configuration.slidingWindow)
            } ?? .none

        let attentionScale =
            useQueryTemperature
            ? llama4AttentionScale(
                size: inputs.dim(-1), offset: MLXArray(cache?.first?.offset ?? 0),
                beta: configuration.llama4ScalingBeta,
                maxPositionEmbeddings: configuration.originalMaxPositionEmbeddings
            ).asType(h.dtype)
            : MLXArray(1, dtype: h.dtype)

        let layerCount = layers.count - (penultimate ? 1 : 0)
        for (i, layer) in layers.prefix(layerCount).enumerated() {
            h = layer(
                h, attentionScale: attentionScale,
                mask: layer.usesSliding ? slidingMask : fullMask, cache: cache?[i])
        }

        return penultimate ? h : norm(h)
    }
}

public class Ministral3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Ministral3ModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: Ministral3Configuration

    public init(_ args: Ministral3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = args.resolvedLayerTypes.map { _ in args.kvHeads }
        self.model = Ministral3ModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    /// A sliding layer needs the rotating cache, or its window is not a window.
    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try model.layers.map { layer in
            try makeHybridAttentionKVCache(
                configuration: configuration,
                slidingWindow: self.configuration.slidingWindow,
                usesSlidingWindow: layer.usesSliding)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        guard configuration.tieWordEmbeddings else { return weights }
        return weights.filter { $0.key != "lm_head.weight" }
    }
}

public struct Ministral3Configuration: Decodable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var headDimensions: Int?
    var maxPositionEmbeddings: Int?
    var kvHeads: Int
    var tieWordEmbeddings: Bool
    var layerTypes: [String]?
    var slidingWindow: Int?

    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var llama4ScalingBeta: Float
    var originalMaxPositionEmbeddings: Int

    var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    /// Unspecified layer types default to full attention.
    var resolvedLayerTypes: [String] {
        layerTypes ?? Array(repeating: "full_attention", count: hiddenLayers)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case headDimensions = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case kvHeads = "num_key_value_heads"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
        case layerTypes = "layer_types"
        case slidingWindow = "sliding_window"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "ministral3"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        headDimensions = try c.decodeIfPresent(Int.self, forKey: .headDimensions)
        maxPositionEmbeddings = try c.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)

        // `rope_parameters` carries the base beside the scaling and the two Llama 4
        // temperature terms, all of which the attention needs. Upstream indexes it
        // directly, so a config lacking them cannot run and is rejected here rather than
        // silently scaling by one.
        let parameters =
            try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters) ?? [:]

        func number(_ key: String) -> Float? {
            switch parameters[key] {
            case .float(let value): value
            case .int(let value): Float(value)
            default: nil
            }
        }

        ropeTheta = number("rope_theta") ?? 10000
        guard let beta = number("llama_4_scaling_beta"),
            let original = number("original_max_position_embeddings")
        else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: c.codingPath,
                    debugDescription:
                        "rope_parameters must carry 'llama_4_scaling_beta' and "
                        + "'original_max_position_embeddings'; this family's attention "
                        + "scaling is not optional"))
        }
        llama4ScalingBeta = beta
        originalMaxPositionEmbeddings = Int(original)

        var scaling = parameters
        scaling.removeValue(forKey: "rope_theta")
        let kind = scaling["rope_type"] ?? scaling["type"]
        let isDefault: Bool =
            if case .string(let name) = kind { name == "default" } else { kind == nil }
        ropeScaling = isDefault ? nil : scaling
    }
}

extension Ministral3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
