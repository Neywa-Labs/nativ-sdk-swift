import Foundation
import MLX
import MLXNN

class Dots1Attention: Module {
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

    let rope: RoPELayer

    init(_ args: Dots1Configuration) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDimensions = args.resolvedHeadDimensions
        self.scale = pow(Float(headDimensions), -0.5)

        _wq.wrappedValue = Linear(args.hiddenSize, heads * headDimensions, bias: false)
        _wk.wrappedValue = Linear(args.hiddenSize, kvHeads * headDimensions, bias: false)
        _wv.wrappedValue = Linear(args.hiddenSize, kvHeads * headDimensions, bias: false)
        _wo.wrappedValue = Linear(heads * headDimensions, args.hiddenSize, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDimensions, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDimensions, eps: args.rmsNormEps)

        self.rope = initializeRope(
            dims: headDimensions,
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)
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

/// The router: two bare parameters, no `Linear` around either.
class Dots1TopKRouter: Module {
    let topK: Int
    let groups: Int
    let topGroups: Int
    let routedScalingFactor: Float
    let normalizesTopK: Bool

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    init(_ args: Dots1Configuration) {
        self.topK = args.expertsPerToken
        self.groups = args.groups
        self.topGroups = args.topGroups
        self.routedScalingFactor = args.routedScalingFactor
        self.normalizesTopK = args.normalizeTopKProbabilities

        _weight.wrappedValue = MLXArray.zeros([args.routedExperts, args.hiddenSize])
        _correctionBias.wrappedValue = MLXArray.zeros([args.routedExperts])
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        groupedExpertSelection(
            gates: matmul(x, weight.transposed()),
            correctionBias: correctionBias,
            topK: topK,
            groups: groups,
            topGroups: topGroups,
            routedScalingFactor: routedScalingFactor,
            normalizeTopK: normalizesTopK)
    }
}

class Dots1MoE: Module, UnaryLayer {
    @ModuleInfo var experts: SwitchGLU
    let gate: Dots1TopKRouter
    @ModuleInfo(key: "shared_experts") var sharedExperts: SwiGLUMLP?

    init(_ args: Dots1Configuration) {
        _experts.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
            numExperts: args.routedExperts)
        self.gate = Dots1TopKRouter(args)
        if args.sharedExperts > 0 {
            _sharedExperts.wrappedValue = SwiGLUMLP(
                inputDimensions: args.hiddenSize,
                hiddenDimensions: args.moeIntermediateSize * args.sharedExperts,
                bias: args.mlpBias)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = (experts(x, indices) * scores[.ellipsis, .newAxis]).sum(axis: -2)
            .asType(x.dtype)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

class Dots1Block: Module {
    @ModuleInfo(key: "self_attn") var attention: Dots1Attention
    @ModuleInfo(key: "mlp") var mlp: UnaryLayer

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let isMixture: Bool

    init(_ args: Dots1Configuration, layerIndex: Int) {
        _attention.wrappedValue = Dots1Attention(args)
        // A single threshold, with no stride: dense below it, mixture at and above.
        self.isMixture = layerIndex >= args.firstKDenseReplace
        _mlp.wrappedValue =
            isMixture
            ? Dots1MoE(args)
            : SwiGLUMLP(
                inputDimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize, bias: args.mlpBias)
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

public class Dots1ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [Dots1Block]
    let norm: RMSNorm

    public init(_ args: Dots1Configuration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { Dots1Block(args, layerIndex: $0) }
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

public class Dots1Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Dots1ModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: Dots1Configuration

    public init(_ args: Dots1Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = Dots1ModelInner(args)
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

    /// Stacks the mixture layers' experts under `mlp.experts`, and drops the precomputed
    /// rotary frequencies, which are recomputed from the base.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        for layer in configuration.firstKDenseReplace ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layer).mlp.experts"
            for projection in ["gate_proj", "down_proj", "up_proj"] {
                for field in ["weight", "scales", "biases"] {
                    guard sanitized["\(prefix).0.\(projection).\(field)"] != nil
                    else { continue }

                    var stacked: [MLXArray] = []
                    for expert in 0 ..< configuration.routedExperts {
                        guard
                            let value = sanitized.removeValue(
                                forKey: "\(prefix).\(expert).\(projection).\(field)")
                        else { break }
                        stacked.append(value)
                    }
                    guard stacked.count == configuration.routedExperts else { continue }
                    sanitized["\(prefix).\(projection).\(field)"] = MLX.stacked(stacked)
                }
            }
        }

        if configuration.tieWordEmbeddings {
            sanitized = sanitized.filter { $0.key != "lm_head.weight" }
        }
        return sanitized.filter { !$0.key.contains("rotary_emb.inv_freq") }
    }
}

public struct Dots1Configuration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var maxPositionEmbeddings: Int?
    var kvHeads: Int
    var firstKDenseReplace: Int
    var moeIntermediateSize: Int
    var routedExperts: Int
    var sharedExperts: Int
    var normalizeTopKProbabilities: Bool
    var expertsPerToken: Int
    var ropeTheta: Float
    var routedScalingFactor: Float
    var headDimensions: Int?
    var scoringFunction: String
    var groups: Int
    var topGroups: Int
    var attentionBias: Bool
    var mlpBias: Bool
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool

    var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case kvHeads = "num_key_value_heads"
        case firstKDenseReplace = "first_k_dense_replace"
        case moeIntermediateSize = "moe_intermediate_size"
        case routedExperts = "n_routed_experts"
        case sharedExperts = "n_shared_experts"
        case normalizeTopKProbabilities = "norm_topk_prob"
        case expertsPerToken = "num_experts_per_tok"
        case ropeTheta = "rope_theta"
        case routedScalingFactor = "routed_scaling_factor"
        case headDimensions = "head_dim"
        case scoringFunction = "scoring_func"
        case groups = "n_group"
        case topGroups = "topk_group"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "dots1"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        maxPositionEmbeddings = try c.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        firstKDenseReplace = try c.decode(Int.self, forKey: .firstKDenseReplace)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        routedExperts = try c.decode(Int.self, forKey: .routedExperts)
        sharedExperts = try c.decode(Int.self, forKey: .sharedExperts)
        normalizeTopKProbabilities = try c.decode(
            Bool.self, forKey: .normalizeTopKProbabilities)
        expertsPerToken = try c.decode(Int.self, forKey: .expertsPerToken)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        routedScalingFactor = try c.decode(Float.self, forKey: .routedScalingFactor)
        headDimensions = try c.decodeIfPresent(Int.self, forKey: .headDimensions)
        scoringFunction =
            try c.decodeIfPresent(String.self, forKey: .scoringFunction) ?? "noaux_tc"
        groups = try c.decodeIfPresent(Int.self, forKey: .groups) ?? 1
        topGroups = try c.decodeIfPresent(Int.self, forKey: .topGroups) ?? 1
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        // The two-stage router is the only scoring this family implements, so a config
        // asking for another one is refused rather than quietly routed the one way.
        guard scoringFunction == "noaux_tc" else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: c.codingPath,
                    debugDescription:
                        "scoring_func '\(scoringFunction)' is not supported; dots1 routes "
                        + "through the grouped noaux_tc selection"))
        }
    }
}

extension Dots1Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
