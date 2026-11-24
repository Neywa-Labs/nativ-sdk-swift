import Foundation
import MLX
import MLXNN

class Internlm3Attention: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: Internlm2DynamicNTKScalingRoPE

    init(_ args: Internlm3Configuration) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDimensions = args.hiddenSize / args.attentionHeads
        self.scale = pow(Float(headDimensions), -0.5)

        _wq.wrappedValue = Linear(
            args.hiddenSize, heads * headDimensions, bias: args.qkvBias)
        _wk.wrappedValue = Linear(
            args.hiddenSize, kvHeads * headDimensions, bias: args.qkvBias)
        _wv.wrappedValue = Linear(
            args.hiddenSize, kvHeads * headDimensions, bias: args.qkvBias)
        _wo.wrappedValue = Linear(
            heads * headDimensions, args.hiddenSize, bias: args.qkvBias)

        // Linear scaling divides positions; dynamic scaling adjusts the rotary base.
        let ropeScale: Float =
            if let scaling = args.ropeScaling,
                case .string("linear") = scaling["rope_type"],
                let factor = scaling["factor"]?.asFloat()
            {
                1 / factor
            } else {
                2
            }

        self.rope = Internlm2DynamicNTKScalingRoPE(
            dims: headDimensions,
            maxPositionEmbeddings: args.maxPositionEmbeddings,
            traditional: args.ropeTraditional,
            base: args.ropeTheta,
            scale: ropeScale)
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

class Internlm3Block: Module {
    @ModuleInfo(key: "self_attn") var attention: Internlm3Attention
    let mlp: SwiGLUMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: Internlm3Configuration) {
        _attention.wrappedValue = Internlm3Attention(args)
        self.mlp = SwiGLUMLP(
            inputDimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize,
            bias: args.bias)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
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

public class Internlm3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [Internlm3Block]
    let norm: RMSNorm

    public init(_ args: Internlm3Configuration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { _ in Internlm3Block(args) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
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

public class Internlm3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Internlm3ModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: Internlm3Configuration

    public init(_ args: Internlm3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Internlm3ModelInner(args)
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

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // The rotary frequencies are recomputed from the base, and the base is recomputed
        // per call, so a checkpoint's precomputed copy binds to nothing.
        var sanitized = weights.filter { !$0.key.contains("rotary_emb.inv_freq") }
        if configuration.tieWordEmbeddings {
            sanitized = sanitized.filter { !$0.key.hasPrefix("lm_head.") }
        }
        return sanitized
    }
}

public struct Internlm3Configuration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var bias: Bool
    var qkvBias: Bool
    var maxPositionEmbeddings: Int
    var kvHeads: Int
    var ropeTheta: Float
    var ropeTraditional: Bool
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case bias
        case qkvBias = "qkv_bias"
        case maxPositionEmbeddings = "max_position_embeddings"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "internlm3"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        bias = try c.decodeIfPresent(Bool.self, forKey: .bias) ?? false
        qkvBias = try c.decodeIfPresent(Bool.self, forKey: .qkvBias) ?? false
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        // Upstream raises on both of these in __post_init__ rather than ignoring them.
        if let scaling = ropeScaling {
            guard scaling["factor"] != nil, let type = scaling["rope_type"] else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: c.codingPath,
                        debugDescription: "rope_scaling must contain 'factor' and 'rope_type'"))
            }
            guard case .string(let name) = type, name == "linear" || name == "dynamic" else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: c.codingPath,
                        debugDescription:
                            "rope_scaling 'rope_type' supports only 'linear' or 'dynamic'"))
            }
        }
    }
}

extension Internlm3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
