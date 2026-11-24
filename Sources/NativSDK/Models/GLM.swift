import Foundation
import MLX
import MLXNN

class GLMAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    init(_ args: GLMConfiguration) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDimensions = args.headDimensions
        self.scale = pow(Float(args.headDimensions), -0.5)

        _wq.wrappedValue = Linear(
            args.hiddenSize, heads * headDimensions, bias: args.attentionBias)
        _wk.wrappedValue = Linear(
            args.hiddenSize, kvHeads * headDimensions, bias: args.attentionBias)
        _wv.wrappedValue = Linear(
            args.hiddenSize, kvHeads * headDimensions, bias: args.attentionBias)
        // Never biased, whatever attention_bias says.
        _wo.wrappedValue = Linear(heads * headDimensions, args.hiddenSize, bias: false)

        self.rope = RoPE(
            dimensions: headDimensions, traditional: true, base: args.ropeTheta)
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

class GLMMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_up_proj") var gateUp: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ args: GLMConfiguration) {
        _gateUp.wrappedValue = Linear(
            args.hiddenSize, 2 * args.intermediateSize, bias: false)
        _down.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // The gate is the first half. Reading it the other way round loads the same weight
        // and changes every activation.
        let halves = split(gateUp(x), parts: 2, axis: -1)
        return down(silu(halves[0]) * halves[1])
    }
}

class GLMBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: GLMAttention
    let mlp: GLMMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: GLMConfiguration) {
        _attention.wrappedValue = GLMAttention(args)
        self.mlp = GLMMLP(args)
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

public class GLMModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [GLMBlock]
    let norm: RMSNorm

    public init(_ args: GLMConfiguration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { _ in GLMBlock(args) }
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

public class GLMModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: GLMModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: GLMConfiguration

    public init(_ args: GLMConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = GLMModelInner(args)
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
        guard configuration.tieWordEmbeddings else { return weights }
        return weights.filter { $0.key != "lm_head.weight" }
    }
}

public struct GLMConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var headDimensions: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int?
    var attentionBias: Bool
    var ropeTheta: Float
    var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case headDimensions = "head_dim"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case ropeTheta = "rope_theta"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "glm"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        headDimensions = try c.decode(Int.self, forKey: .headDimensions)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        maxPositionEmbeddings = try c.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }
}

extension GLMModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
