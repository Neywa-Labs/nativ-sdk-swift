import Foundation
import MLX
import MLXNN

class KlearAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    init(_ args: KlearConfiguration) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDimensions = args.hiddenSize / args.attentionHeads
        self.scale = pow(Float(headDimensions), -0.5)

        _wq.wrappedValue = Linear(
            args.hiddenSize, heads * headDimensions, bias: args.attentionBias)
        _wk.wrappedValue = Linear(
            args.hiddenSize, kvHeads * headDimensions, bias: args.attentionBias)
        _wv.wrappedValue = Linear(
            args.hiddenSize, kvHeads * headDimensions, bias: args.attentionBias)
        _wo.wrappedValue = Linear(
            heads * headDimensions, args.hiddenSize, bias: args.attentionBias)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDimensions, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDimensions, eps: args.rmsNormEps)

        self.rope = RoPE(
            dimensions: headDimensions, traditional: false, base: args.ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(wq(x).reshaped(B, L, heads, -1)).transposed(0, 2, 1, 3)
        var keys = kNorm(wk(x).reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
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

class KlearSparseMoeBlock: Module, UnaryLayer {
    let topK: Int
    let normalizesTopK: Bool

    @ModuleInfo var gate: Linear
    @ModuleInfo var experts: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: SwiGLUMLP
    @ModuleInfo var coefficient: Linear

    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray

    init(_ args: KlearConfiguration) {
        self.topK = args.expertsPerToken
        self.normalizesTopK = args.normalizeTopKProbabilities

        _gate.wrappedValue = Linear(args.hiddenSize, args.experts, bias: false)
        _experts.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
            numExperts: args.experts)
        _sharedExperts.wrappedValue = SwiGLUMLP(
            inputDimensions: args.hiddenSize,
            hiddenDimensions: args.moeIntermediateSize * args.sharedExperts)
        // Two outputs: one weight for the routed branch, one for the shared branch.
        _coefficient.wrappedValue = Linear(args.hiddenSize, 2)
        _expertBias.wrappedValue = MLXArray.zeros([args.experts], dtype: .float32)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Sigmoid, not softmax: the weights are independent and do not sum to one.
        let routingWeights = sigmoid(gate(x).asType(.float32))

        // The bias shifts *selection* and nothing else. Taking the biased values as the
        // scores would let a bias term change how much an expert counts, which is not what
        // it is for.
        let biased = routingWeights + expertBias.reshaped(1, 1, -1)
        let indices = MLX.argPartition(-biased, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]

        var scores = takeAlong(routingWeights, indices, axis: -1)
        if normalizesTopK {
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }
        scores = scores.asType(x.dtype)

        let routed = (experts(x, indices) * scores[.ellipsis, .newAxis]).sum(axis: -2)
        let shared = sharedExperts(x)

        // A learned mixture of the two branches, not a sum.
        let coefficients = softmax(coefficient(x), axis: -1, precise: true)
        return routed * coefficients[.ellipsis, ..<1] + shared * coefficients[.ellipsis, 1...]
    }
}

class KlearBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: KlearAttention
    @ModuleInfo(key: "mlp") var mlp: UnaryLayer

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let isMixture: Bool

    init(_ args: KlearConfiguration, layerIndex: Int) {
        _attention.wrappedValue = KlearAttention(args)
        self.isMixture = args.isMixtureLayer(layerIndex)
        _mlp.wrappedValue =
            isMixture
            ? KlearSparseMoeBlock(args)
            : SwiGLUMLP(
                inputDimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize)
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

public class KlearModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [KlearBlock]
    let norm: RMSNorm

    public init(_ args: KlearConfiguration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { KlearBlock(args, layerIndex: $0) }
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

public class KlearModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: KlearModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let configuration: KlearConfiguration

    public init(_ args: KlearConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = KlearModelInner(args)
        _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    /// Stacks each mixture layer's experts under `mlp.experts`, skipping dense layers.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights
        for layer in 0 ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layer).mlp.experts"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                guard sanitized["\(prefix).0.\(projection).weight"] != nil else { continue }

                var stacked: [MLXArray] = []
                for expert in 0 ..< configuration.experts {
                    guard
                        let value = sanitized.removeValue(
                            forKey: "\(prefix).\(expert).\(projection).weight")
                    else { break }
                    stacked.append(value)
                }
                guard stacked.count == configuration.experts else { continue }
                sanitized["\(prefix).\(projection).weight"] = MLX.stacked(stacked)
            }
        }
        return sanitized
    }
}

public struct KlearConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var attentionBias: Bool
    var mlpOnlyLayers: [Int]
    var experts: Int
    var expertsPerToken: Int
    var decoderSparseStep: Int
    var sharedExperts: Int
    var moeIntermediateSize: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var ropeTheta: Float
    var maxPositionEmbeddings: Int
    var normalizeTopKProbabilities: Bool

    /// Whether layer `index` gets a mixture.
    ///
    /// Note the `+ 1` in the step: the sparse layers are the ones whose *one-based* index
    /// divides, which is off by one from deepseek's plain modulo on the same-looking field.
    func isMixtureLayer(_ index: Int) -> Bool {
        !mlpOnlyLayers.contains(index) && experts > 0
            && (index + 1) % decoderSparseStep == 0
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case attentionBias = "attention_bias"
        case mlpOnlyLayers = "mlp_only_layers"
        case experts = "num_experts"
        case expertsPerToken = "num_experts_per_tok"
        case decoderSparseStep = "decoder_sparse_step"
        case sharedExperts = "n_shared_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normalizeTopKProbabilities = "norm_topk_prob"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "klear"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        attentionBias = try c.decode(Bool.self, forKey: .attentionBias)
        mlpOnlyLayers = try c.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? []
        experts = try c.decode(Int.self, forKey: .experts)
        expertsPerToken = try c.decode(Int.self, forKey: .expertsPerToken)
        decoderSparseStep = try c.decode(Int.self, forKey: .decoderSparseStep)
        sharedExperts = try c.decode(Int.self, forKey: .sharedExperts)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        normalizeTopKProbabilities = try c.decode(
            Bool.self, forKey: .normalizeTopKProbabilities)
    }
}

extension KlearModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
