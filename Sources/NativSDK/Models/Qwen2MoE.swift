import Foundation
import MLX
import MLXNN

class Qwen2MoEAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer

    init(_ args: Qwen2MoEConfiguration) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDimensions = args.hiddenSize / args.attentionHeads
        self.scale = pow(Float(headDimensions), -0.5)

        // q, k and v are always biased here and the output projection never is.
        _wq.wrappedValue = Linear(args.hiddenSize, heads * headDimensions, bias: true)
        _wk.wrappedValue = Linear(args.hiddenSize, kvHeads * headDimensions, bias: true)
        _wv.wrappedValue = Linear(args.hiddenSize, kvHeads * headDimensions, bias: true)
        _wo.wrappedValue = Linear(heads * headDimensions, args.hiddenSize, bias: false)

        self.rope = initializeRope(
            dims: headDimensions,
            base: args.ropeTheta,
            traditional: args.ropeTraditional,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: nil)
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

class Qwen2MoESparseBlock: Module {
    let topK: Int

    @ModuleInfo var gate: Linear

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: SwiGLUMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    init(_ args: Qwen2MoEConfiguration) {
        self.topK = args.expertsPerToken

        _gate.wrappedValue = Linear(args.hiddenSize, args.experts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
            numExperts: args.experts)
        _sharedExpert.wrappedValue = SwiGLUMLP(
            inputDimensions: args.hiddenSize,
            hiddenDimensions: args.sharedExpertIntermediateSize)
        _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // All the logits are softmaxed, then the top few of the resulting distribution are
        // taken. The discarded experts keep their share of the mass, so the kept weights
        // sum to less than one -- selecting first and softmaxing the selection, as phixtral
        // does, would renormalize them to one and change every token's mixture.
        let gates = softmax(gate(x), axis: -1, precise: true)

        let indices = MLX.stopGradient(
            MLX.argPartition(-gates, kth: topK - 1, axis: -1)
        )[.ellipsis, ..<topK]
        let scores = takeAlong(gates, indices, axis: -1)

        let routed = (switchMLP(x, indices) * scores[.ellipsis, .newAxis]).sum(axis: -2)
        let shared = sigmoid(sharedExpertGate(x)) * sharedExpert(x)
        return routed + shared
    }
}

class Qwen2MoEBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen2MoEAttention
    let mlp: Qwen2MoESparseBlock

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: Qwen2MoEConfiguration) {
        _attention.wrappedValue = Qwen2MoEAttention(args)
        self.mlp = Qwen2MoESparseBlock(args)
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

public class Qwen2MoEModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [Qwen2MoEBlock]
    let norm: RMSNorm

    public init(_ args: Qwen2MoEConfiguration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { _ in Qwen2MoEBlock(args) }
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

public class Qwen2MoEModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Qwen2MoEModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: Qwen2MoEConfiguration

    public init(_ args: Qwen2MoEConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Qwen2MoEModelInner(args)
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

    /// Stacks each layer's separately shipped experts onto a leading axis.
    ///
    /// A checkpoint already in the switched form is left alone, which is what the guard on
    /// the first layer's first expert is for.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        guard weights["model.layers.0.mlp.experts.0.up_proj.weight"] != nil else {
            return weights
        }

        var sanitized = weights
        for layer in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layer)"
            for projection in ["up_proj", "down_proj", "gate_proj"] {
                for field in ["weight", "scales", "biases"] {
                    guard sanitized["\(prefix).mlp.experts.0.\(projection).\(field)"] != nil
                    else { continue }

                    var stacked: [MLXArray] = []
                    for expert in 0 ..< configuration.experts {
                        let key = "\(prefix).mlp.experts.\(expert).\(projection).\(field)"
                        guard let value = sanitized.removeValue(forKey: key) else { break }
                        stacked.append(value)
                    }
                    guard stacked.count == configuration.experts else { continue }
                    sanitized["\(prefix).mlp.switch_mlp.\(projection).\(field)"] =
                        MLX.stacked(stacked)
                }
            }
        }
        return sanitized
    }
}

public struct Qwen2MoEConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var expertsPerToken: Int
    var experts: Int
    var moeIntermediateSize: Int
    var sharedExpertIntermediateSize: Int
    var rmsNormEps: Float
    var vocabularySize: Int
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
        case expertsPerToken = "num_experts_per_tok"
        case experts = "num_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen2_moe"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        expertsPerToken = try c.decode(Int.self, forKey: .expertsPerToken)
        experts = try c.decode(Int.self, forKey: .experts)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        sharedExpertIntermediateSize = try c.decode(
            Int.self, forKey: .sharedExpertIntermediateSize)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        // Upstream raises rather than ignoring a malformed scaling, and requires `type`
        // specifically rather than accepting `rope_type`.
        if let scaling = ropeScaling {
            guard scaling["factor"] != nil, let kind = scaling["type"] else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: c.codingPath,
                        debugDescription: "rope_scaling must contain 'factor' and 'type'"))
            }
            guard case .string("linear") = kind else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: c.codingPath,
                        debugDescription: "rope_scaling 'type' supports only 'linear'"))
            }
        }
    }
}

extension Qwen2MoEModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
