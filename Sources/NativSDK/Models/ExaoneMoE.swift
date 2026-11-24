import Foundation
import MLX
import MLXNN

class ExaoneMoEGate: Module {

    let topK: Int
    let groups: Int
    let topGroups: Int
    let routedScalingFactor: Float
    let normalizeTopK: Bool

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    init(_ args: ExaoneMoEConfiguration) {
        self.topK = args.expertsPerToken
        self.groups = args.groups
        self.topGroups = args.topGroups
        self.routedScalingFactor = args.routedScalingFactor
        self.normalizeTopK = args.normalizeTopKProbabilities

        _weight.wrappedValue = MLXArray.zeros([args.experts, args.hiddenSize])
        _correctionBias.wrappedValue = MLXArray.zeros([args.experts])
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        groupedExpertSelection(
            gates: matmul(x, weight.T),
            correctionBias: correctionBias,
            topK: topK,
            groups: groups,
            topGroups: topGroups,
            routedScalingFactor: routedScalingFactor,
            normalizeTopK: normalizeTopK)
    }
}

class ExaoneMoE: Module, UnaryLayer {

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "gate") var gate: ExaoneMoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: SwiGLUMLP?

    init(_ args: ExaoneMoEConfiguration) {
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
            numExperts: args.experts)
        _gate.wrappedValue = ExaoneMoEGate(args)
        if args.sharedExperts > 0 {
            _sharedExperts.wrappedValue = SwiGLUMLP(
                inputDimensions: args.hiddenSize,
                hiddenDimensions: args.moeIntermediateSize * args.sharedExperts)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        let routed = switchMLP(x, indices)
        var y = (routed * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(routed.dtype)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

class ExaoneMoEAttention: Module {

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    let isSliding: Bool

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer?

    init(_ args: ExaoneMoEConfiguration, layerIndex: Int) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDim = args.headDimensions
        self.scale = pow(Float(args.headDimensions), -0.5)

        self.isSliding = args.layerTypes[layerIndex] == "sliding_attention"

        _wq.wrappedValue = Linear(args.hiddenSize, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(args.hiddenSize, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(args.hiddenSize, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, args.hiddenSize, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        let uniformlyGlobal = !args.layerTypes.contains("sliding_attention")
        self.rope =
            (isSliding || uniformlyGlobal)
            ? initializeRope(
                dims: headDim, base: args.ropeTheta, traditional: false,
                scalingConfig: args.ropeScaling,
                maxPositionEmbeddings: args.maxPositionEmbeddings)
            : nil
    }

    var usesRope: Bool { rope != nil }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(wq(x).reshaped(B, L, heads, -1)).transposed(0, 2, 1, 3)
        var keys = kNorm(wk(x).reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        if let rope {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache,
            scale: scale, mask: mask)

        return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

class ExaoneMoEBlock: Module {

    @ModuleInfo(key: "self_attn") var attention: ExaoneMoEAttention
    @ModuleInfo(key: "mlp") var mlp: UnaryLayer

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    var isSliding: Bool { attention.isSliding }

    init(_ args: ExaoneMoEConfiguration, layerIndex: Int) {
        _attention.wrappedValue = ExaoneMoEAttention(args, layerIndex: layerIndex)
        _mlp.wrappedValue =
            args.isMoELayer[layerIndex]
            ? ExaoneMoE(args)
            : SwiGLUMLP(
                inputDimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
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

public class ExaoneMoEModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [ExaoneMoEBlock]
    let norm: RMSNorm

    let windowSize: Int

    public init(_ args: ExaoneMoEConfiguration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map {
            ExaoneMoEBlock(args, layerIndex: $0)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self.windowSize = args.slidingWindow
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let globalIndex = layers.firstIndex { !$0.isSliding } ?? 0
        let slidingIndex = layers.firstIndex { $0.isSliding } ?? 0
        let globalMask = createAttentionMask(h: h, cache: cache?[globalIndex])
        let slidingMask = createAttentionMask(
            h: h, cache: cache?[slidingIndex], windowSize: windowSize)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: layer.isSliding ? slidingMask : globalMask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class ExaoneMoEModel: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: ExaoneMoEModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: ExaoneMoEConfiguration

    public init(_ args: ExaoneMoEConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.model = ExaoneMoEModelInner(args)
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

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try model.layers.map { layer in
            try makeHybridAttentionKVCache(
                configuration: configuration,
                slidingWindow: self.configuration.slidingWindow, usesSlidingWindow: layer.isSliding)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights.filter { !$0.key.hasPrefix("mtp.") }

        for layer in 0 ..< configuration.hiddenLayers
        where configuration.isMoELayer[layer] {
            let prefix = "model.layers.\(layer)"

            if let bias = sanitized.removeValue(
                forKey: "\(prefix).mlp.e_score_correction_bias")
            {
                sanitized["\(prefix).mlp.gate.e_score_correction_bias"] = bias
            }

            for projection in ["gate_proj", "down_proj", "up_proj"] {
                for kind in ["weight", "scales", "biases"] {
                    let keys = (0 ..< configuration.experts).map {
                        "\(prefix).mlp.experts.\($0).\(projection).\(kind)"
                    }
                    guard let first = keys.first, let last = keys.last,
                        sanitized[first] != nil, sanitized[last] != nil
                    else { continue }
                    let stacked = keys.compactMap { sanitized.removeValue(forKey: $0) }
                    guard stacked.count == configuration.experts else { continue }
                    sanitized["\(prefix).mlp.switch_mlp.\(projection).\(kind)"] =
                        MLX.stacked(stacked)
                }
            }
        }

        if configuration.tieWordEmbeddings {
            sanitized.removeValue(forKey: "lm_head.weight")
        }
        return sanitized
    }
}

public struct ExaoneMoEConfiguration: Decodable, Sendable {

    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var headDimensions: Int
    var experts: Int
    var expertsPerToken: Int
    var sharedExperts: Int
    var rmsNormEps: Float
    var maxPositionEmbeddings: Int
    var slidingWindow: Int
    var layerTypes: [String]
    var isMoELayer: [Bool]
    var groups: Int
    var topGroups: Int
    var routedScalingFactor: Float
    var normalizeTopKProbabilities: Bool
    var scoringFunction: String
    var topKMethod: String
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDimensions = "head_dim"
        case experts = "num_experts"
        case expertsPerToken = "num_experts_per_tok"
        case sharedExperts = "num_shared_experts"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case isMoELayer = "is_moe_layer"
        case groups = "n_group"
        case topGroups = "topk_group"
        case routedScalingFactor = "routed_scaling_factor"
        case normalizeTopKProbabilities = "norm_topk_prob"
        case scoringFunction = "scoring_func"
        case topKMethod = "topk_method"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "exaone_moe"
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDimensions = try c.decode(Int.self, forKey: .headDimensions)
        experts = try c.decode(Int.self, forKey: .experts)
        expertsPerToken = try c.decode(Int.self, forKey: .expertsPerToken)
        sharedExperts = try c.decodeIfPresent(Int.self, forKey: .sharedExperts) ?? 0
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
        isMoELayer = try c.decode([Bool].self, forKey: .isMoELayer)
        groups = try c.decodeIfPresent(Int.self, forKey: .groups) ?? 1
        topGroups = try c.decodeIfPresent(Int.self, forKey: .topGroups) ?? 1
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 2.5
        normalizeTopKProbabilities =
            try c.decodeIfPresent(Bool.self, forKey: .normalizeTopKProbabilities) ?? true
        scoringFunction =
            try c.decodeIfPresent(String.self, forKey: .scoringFunction) ?? "sigmoid"
        topKMethod = try c.decodeIfPresent(String.self, forKey: .topKMethod) ?? "noaux_tc"
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        let parameters = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        let flat = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        ropeTheta = parameters?["rope_theta"]?.asFloat() ?? flat

        precondition(topKMethod == "noaux_tc", "unsupported topk method \(topKMethod)")
    }
}

extension ExaoneMoEModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
