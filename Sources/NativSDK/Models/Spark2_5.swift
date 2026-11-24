import Foundation
import MLX
import MLXNN

private let slidingLayerType = "sliding_attention"

class Spark2_5Attention: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let queryDimensions: Int
    let keyValueDimensions: Int
    let scale: Float
    let gateMode: String

    @ModuleInfo(key: "q_k_v_proj") var qkv: Linear
    @ModuleInfo(key: "out_proj") var wo: Linear
    @ModuleInfo(key: "g_proj") var gateProjection: Linear?

    let rope: RoPE

    init(_ args: Spark2_5Configuration, layerType: String) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDimensions = args.headDimensions
        self.queryDimensions = args.attentionHeads * args.headDimensions
        self.keyValueDimensions = args.kvHeads * args.headDimensions
        self.scale = pow(Float(args.headDimensions), -0.5)
        self.gateMode = args.gateAttentionActivationMode

        _qkv.wrappedValue = Linear(
            args.hiddenSize, queryDimensions + 2 * keyValueDimensions,
            bias: args.attentionBias)
        _wo.wrappedValue = Linear(queryDimensions, args.hiddenSize, bias: args.attentionBias)
        if args.headwiseAttentionOutputGate {
            _gateProjection.wrappedValue = Linear(
                args.hiddenSize, args.attentionHeads, bias: args.attentionBias)
        }

        let (ropeDimensions, ropeBase) = args.ropeFor(layerType: layerType)
        self.rope = RoPE(dimensions: ropeDimensions, traditional: false, base: ropeBase)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let fused = qkv(x)
        var queries = fused[.ellipsis, ..<queryDimensions]
        var keys = fused[.ellipsis, queryDimensions ..< (queryDimensions + keyValueDimensions)]
        let values = fused[.ellipsis, (queryDimensions + keyValueDimensions)...]

        queries = queries.reshaped(B, L, heads, headDimensions).transposed(0, 2, 1, 3)
        keys = keys.reshaped(B, L, kvHeads, headDimensions).transposed(0, 2, 1, 3)
        let reshapedValues = values.reshaped(B, L, kvHeads, headDimensions).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        var output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: reshapedValues,
            cache: cache, scale: scale, mask: mask
        )

        if let gateProjection {
            // The gate is drawn from the block input, not from the attention output, and it
            // is one scalar per head rather than per channel. Computed in float32 because
            // the sigmoid saturates in bf16 well before the logit does.
            var gate = gateProjection(x).reshaped(B, L, heads, 1).transposed(0, 2, 1, 3)
                .asType(.float32)
            gate = gateMode == "sigmoid" ? sigmoid(gate) : silu(gate)
            output = output * gate.asType(output.dtype)
        }

        return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

class Spark2_5MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ args: Spark2_5Configuration) {
        _gateProj.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
        _upProj.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
        _downProj.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: args.mlpBias)
    }

    /// GeGLU, not SwiGLU: the gate branch goes through gelu.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(gelu(gateProj(x)) * upProj(x))
    }
}

class Spark2_5Block: Module {
    @ModuleInfo(key: "self_attn") var attention: Spark2_5Attention
    @ModuleInfo(key: "mlp") var mlp: Spark2_5MLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let usesSliding: Bool

    init(_ args: Spark2_5Configuration, layerType: String) {
        self.usesSliding = layerType == slidingLayerType
        _attention.wrappedValue = Spark2_5Attention(args, layerType: layerType)
        _mlp.wrappedValue = Spark2_5MLP(args)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class Spark2_5ModelInner: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding

    let layers: [Spark2_5Block]
    let norm: RMSNorm

    let configuration: Spark2_5Configuration

    public init(_ args: Spark2_5Configuration) {
        self.configuration = args
        _embedding.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = args.layerTypes.map { Spark2_5Block(args, layerType: $0) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedding(inputs)

        // One mask per layer *kind*, each built from a cache of that kind: the sliding
        // layers hold a rotating cache whose offset does not track the full layers'.
        let firstFull = layers.firstIndex { !$0.usesSliding }
        let firstSliding = layers.firstIndex { $0.usesSliding }
        let fullMask = firstFull.map { createAttentionMask(h: h, cache: cache?[$0]) } ?? .none
        let slidingMask =
            firstSliding.map {
                createAttentionMask(
                    h: h, cache: cache?[$0], windowSize: configuration.slidingWindow)
            } ?? .none

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: layer.usesSliding ? slidingMask : fullMask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class Spark2_5Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Spark2_5ModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: Spark2_5Configuration

    public init(_ args: Spark2_5Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = args.layerTypes.map { _ in args.kvHeads }
        self.model = Spark2_5ModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        guard let lmHead else {
            return model.embedding.asLinear(out)
        }
        return lmHead(out)
    }

    /// A sliding layer never attends beyond its window, so its cache is bounded rather than
    /// grown to the full context; the full-attention layers keep an unbounded one.
    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try model.layers.map { layer in
            try makeHybridAttentionKVCache(
                configuration: configuration,
                slidingWindow: self.configuration.slidingWindow,
                usesSlidingWindow: layer.usesSliding)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights
        if configuration.tieWordEmbeddings {
            sanitized = sanitized.filter { !$0.key.hasPrefix("lm_head.") }
        }
        return sanitized
    }
}

public struct Spark2_5Configuration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDimensions: Int
    var vocabularySize: Int
    var rmsNormEps: Float
    var maxPositionEmbeddings: Int
    var slidingWindow: Int
    var tieWordEmbeddings: Bool
    var attentionBias: Bool
    var mlpBias: Bool
    var headwiseAttentionOutputGate: Bool
    var gateAttentionActivationMode: String
    var layerTypes: [String]
    var ropeParameters: [String: [String: StringOrNumber]]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDimensions = "head_dim"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case headwiseAttentionOutputGate = "headwise_attn_output_gate"
        case gateAttentionActivationMode = "gate_attn_act_mode"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Swift.Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "spark2_5"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2560
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 36
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 10240
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        headDimensions = try c.decodeIfPresent(Int.self, forKey: .headDimensions) ?? 256
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 131_072
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 1_048_576
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        headwiseAttentionOutputGate =
            try c.decodeIfPresent(Bool.self, forKey: .headwiseAttentionOutputGate) ?? true
        gateAttentionActivationMode =
            try c.decodeIfPresent(String.self, forKey: .gateAttentionActivationMode) ?? "sigmoid"
        ropeParameters =
            try c.decodeIfPresent([String: [String: StringOrNumber]].self, forKey: .ropeParameters)
            ?? [:]
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: hiddenLayers)
    }

    /// Resolve each layer kind's rotary width and base, falling back to full attention.
    func ropeFor(layerType: String) -> (dimensions: Int, base: Float) {
        let parameters = ropeParameters[layerType] ?? ropeParameters["full_attention"] ?? [:]
        let partial = parameters["partial_rotary_factor"]?.asFloat() ?? 1.0
        let base = parameters["rope_theta"]?.asFloat() ?? 10000.0
        return (Swift.max(2, Int(Float(headDimensions) * partial)), base)
    }
}

extension Spark2_5Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
