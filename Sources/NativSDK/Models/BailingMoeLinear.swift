import Foundation
import MLX
import MLXNN

/// The gated linear recurrence `h_t = h_{t-1} * exp(decay) + k_t^T v_t`, `y_t = q_t h_t`.
///
/// `decay` is one value per head. The query is scaled once, before the loop.
func bailingRecurrentGLA(
    q: MLXArray, k: MLXArray, v: MLXArray, decay: MLXArray, scale: Float,
    state: MLXArray?
) -> (MLXArray, MLXArray) {
    let decayFactor = exp(decay)[0..., .newAxis, .newAxis].asType(q.dtype)
    let scaled = q * scale

    var h = state
    var outputs: [MLXArray] = []
    outputs.reserveCapacity(q.dim(2))

    for t in 0 ..< q.dim(2) {
        let kt = k[0..., 0..., t ..< (t + 1)]
        let vt = v[0..., 0..., t ..< (t + 1)]
        let update = matmul(kt.transposed(0, 1, 3, 2), vt)
        h = h.map { $0 * decayFactor + update } ?? update
        outputs.append(matmul(scaled[0..., 0..., t ..< (t + 1)], h!))
    }

    return (concatenated(outputs, axis: 2), h!)
}

/// Normalizes within `groups` slices of the last axis without a weight, then applies one
/// weight across the whole axis.
class BailingGroupRMSNorm: Module, UnaryLayer {

    @ModuleInfo var weight: MLXArray
    let groups: Int
    let eps: Float

    init(dimensions: Int, eps: Float, groups: Int) {
        self.groups = groups
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let grouped = unflatten(x, axis: -1, shape: [groups, -1])
        let normalized = MLXFast.rmsNorm(grouped, weight: MLXArray.mlxNone, eps: eps)
        return weight * flattened(normalized, start: -2)
    }
}

/// The two shapes the `attention` slot can take.
///
/// They share one checkpoint key, so they share one property: two `@ModuleInfo` wrappers
/// with the same key do *not* both publish -- the first declared wins and the other's
/// weights silently never load.
protocol BailingAttentionLayer: Module {
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        offset: Int
    ) -> MLXArray
}

class BailingAttention: Module, BailingAttentionLayer {

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let usesQKNorm: Bool

    @ModuleInfo(key: "query_key_value") var qkv: Linear
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "query_layernorm") var queryNorm: RMSNorm?
    @ModuleInfo(key: "key_layernorm") var keyNorm: RMSNorm?

    let rope: RoPELayer

    init(_ args: BailingMoeLinearConfiguration) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDim = args.headDimensions
        self.scale = pow(Float(headDim), -0.5)
        self.usesQKNorm = args.usesQKNorm

        _qkv.wrappedValue = Linear(
            args.hiddenSize, (heads + 2 * kvHeads) * headDim, bias: args.usesQKVBias)
        _dense.wrappedValue = Linear(
            heads * headDim, args.hiddenSize, bias: args.usesBias)

        if usesQKNorm {
            _queryNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
            _keyNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        }

        self.rope = initializeRope(
            dims: Int(Float(headDim) * args.partialRotaryFactor),
            base: args.ropeTheta, traditional: args.ropeTraditional,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        offset: Int
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let querySize = heads * headDim
        let keyValueSize = kvHeads * headDim
        let parts = split(
            qkv(x), indices: [querySize, querySize + keyValueSize], axis: -1)

        var queries = parts[0].reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        var keys = parts[1].reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        let values = parts[2].reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

        if let queryNorm, let keyNorm {
            queries = queryNorm(queries)
            keys = keyNorm(keys)
        }

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache,
            scale: scale, mask: mask)

        return dense(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

class BailingLinearAttention: Module, BailingAttentionLayer {

    let heads: Int
    let headDim: Int
    let scale: Float
    let usesQKNorm: Bool

    /// Held as plain floats: an `MLXArray` stored property would be discovered as a
    /// parameter and add a key no checkpoint fills.
    let decaySlopes: [Float]

    @ModuleInfo(key: "query_key_value") var qkv: Linear
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear
    @ModuleInfo(key: "g_norm") var gNorm: BailingGroupRMSNorm
    @ModuleInfo(key: "query_layernorm") var queryNorm: RMSNorm?
    @ModuleInfo(key: "key_layernorm") var keyNorm: RMSNorm?

    let rope: RoPELayer

    init(_ args: BailingMoeLinearConfiguration, layerIndex: Int) {
        self.heads = args.attentionHeads
        self.headDim = args.hiddenSize / args.attentionHeads
        self.scale = pow(Float(headDim), -0.5)
        self.usesQKNorm = args.usesQKNorm

        _qkv.wrappedValue = Linear(
            args.hiddenSize, 3 * heads * headDim, bias: args.usesQKVBias)
        _dense.wrappedValue = Linear(
            heads * headDim, args.hiddenSize, bias: args.usesBias)
        _gProj.wrappedValue = Linear(args.hiddenSize, heads * headDim, bias: false)
        _gNorm.wrappedValue = BailingGroupRMSNorm(
            dimensions: heads * headDim, eps: args.rmsNormEps,
            groups: args.groupNormSize)

        if usesQKNorm {
            _queryNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
            _keyNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        }

        self.rope = initializeRope(
            dims: Int(Float(headDim) * args.partialRotaryFactor),
            base: args.ropeTheta, traditional: args.ropeTraditional,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)

        self.decaySlopes = Self.slopes(
            heads: args.attentionHeads, layers: args.hiddenLayers,
            layerIndex: layerIndex)
        super.init()
    }

    /// ALiBi slopes, negated and scaled down as the layer index rises.
    ///
    /// Layers 0 and 1 share a factor -- the index is clamped at one before the subtraction
    /// -- and the `1e-5` keeps the last layer's factor off zero.
    static func slopes(heads: Int, layers: Int, layerIndex: Int) -> [Float] {
        func powerOfTwoSlopes(_ n: Int) -> [Float] {
            let step = pow(2.0, -pow(2.0, -(log2(Double(n)) - 3)))
            return (0 ..< n).map { Float(pow(step, Double($0 + 1))) }
        }

        var base: [Float]
        if log2(Double(heads)).rounded() == log2(Double(heads)) {
            base = powerOfTwoSlopes(heads)
        } else {
            let closest = Int(pow(2.0, floor(log2(Double(heads)))))
            let extra = powerOfTwoSlopes(2 * closest)
                .enumerated().filter { $0.offset % 2 == 0 }.map(\.element)
            base = powerOfTwoSlopes(closest) + extra.prefix(heads - closest)
        }

        let denominator = Float(max(1, layers - 1))
        let position = Float(max(0, layerIndex - 1))
        let factor = 1 - (position / denominator) + 1e-5
        return base.map { -$0 * factor }
    }

    /// `mask` is accepted to satisfy the shared slot and never read, matching upstream --
    /// which builds one for these layers and discards it.
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        offset: Int
    ) -> MLXArray {
        let slots = cache as? ArraysCache
        let (B, L) = (x.dim(0), x.dim(1))

        let mixed = qkv(x).reshaped(B, L, 3 * heads, headDim)
        let parts = split(mixed, indices: [heads, 2 * heads], axis: 2)

        var queries = parts[0].transposed(0, 2, 1, 3)
        var keys = parts[1].transposed(0, 2, 1, 3)
        let values = parts[2].transposed(0, 2, 1, 3)

        if let queryNorm, let keyNorm {
            queries = queryNorm(queries)
            keys = keyNorm(keys)
        }

        queries = applyRotaryPosition(rope, to: queries, offset: .scalar(offset))
        keys = applyRotaryPosition(rope, to: keys, offset: .scalar(offset))

        let (recurred, state) = bailingRecurrentGLA(
            q: queries, k: keys, v: values, decay: MLXArray(decaySlopes),
            scale: scale, state: slots?[0])
        slots?[0] = state

        let output = recurred.transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return dense(gNorm(output) * sigmoid(gProj(x)))
    }
}

class BailingGate: Module {

    let topK: Int
    let groups: Int
    let topGroups: Int
    let routedScalingFactor: Float
    let normalizeTopK: Bool
    let scoring: ExpertScoring
    let experts: Int

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ParameterInfo(key: "expert_bias") var expertBias: MLXArray?

    init(_ args: BailingMoeLinearConfiguration) {
        self.topK = args.expertsPerToken
        self.groups = args.groups
        self.topGroups = args.topGroups
        self.routedScalingFactor = args.routedScalingFactor
        self.normalizeTopK = args.normalizeTopKProbabilities
        self.scoring = args.scoreFunction == "sigmoid" ? .sigmoid : .softmax
        self.experts = args.experts

        _gateProj.wrappedValue = Linear(args.hiddenSize, args.experts, bias: false)
        if args.routerExpertBias {
            _expertBias.wrappedValue = MLXArray.zeros([args.experts])
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        groupedExpertSelection(
            gates: gateProj(x),
            correctionBias: expertBias ?? MLXArray.zeros([experts]),
            topK: topK,
            groups: groups,
            topGroups: topGroups,
            routedScalingFactor: routedScalingFactor,
            normalizeTopK: normalizeTopK,
            scoring: scoring)
    }
}

class BailingSparseMoeBlock: Module, UnaryLayer {

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "gate") var gate: BailingGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: SwiGLUMLP?

    init(_ args: BailingMoeLinearConfiguration) {
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
            numExperts: args.experts, bias: args.usesBias)
        _gate.wrappedValue = BailingGate(args)
        if args.sharedExperts > 0, args.routerSharedExpert {
            let width = args.moeSharedExpertIntermediateSize ?? args.moeIntermediateSize
            _sharedExperts.wrappedValue = SwiGLUMLP(
                inputDimensions: args.hiddenSize,
                hiddenDimensions: width * args.sharedExperts, bias: args.usesBias)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var out = (switchMLP(x, indices) * scores[.ellipsis, .newAxis]).sum(axis: -2)
        if let sharedExperts {
            out = out + sharedExperts(x)
        }
        return out
    }
}

class BailingBlock: Module {

    @ModuleInfo(key: "attention") var attention: BailingAttentionLayer
    @ModuleInfo(key: "mlp") var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let isGlobal: Bool

    init(_ args: BailingMoeLinearConfiguration, layerIndex: Int) {
        self.isGlobal = args.isGlobal(at: layerIndex)

        _attention.wrappedValue =
            isGlobal
            ? BailingAttention(args)
            : BailingLinearAttention(args, layerIndex: layerIndex)

        _mlp.wrappedValue =
            layerIndex >= args.firstDenseLayers
            ? BailingSparseMoeBlock(args)
            : SwiGLUMLP(
                inputDimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize, bias: args.usesBias)

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        offset: Int
    ) -> MLXArray {
        let h =
            x + attention(inputLayerNorm(x), mask: mask, cache: cache, offset: offset)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class BailingMoeLinearModelInner: Module {

    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding

    let layers: [BailingBlock]
    let norm: RMSNorm

    /// Cache index the rotary offset comes from: the first layer of a group that keeps a
    /// key/value cache. The linear layers have none of their own.
    let attentionIndex: Int

    public init(_ args: BailingMoeLinearConfiguration) {
        precondition(args.vocabularySize > 0)
        _wordEmbeddings.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map {
            BailingBlock(args, layerIndex: $0)
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self.attentionIndex = args.layerGroupSize - 1
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = wordEmbeddings(inputs)

        let reference = cache?[min(attentionIndex, (cache?.count ?? 1) - 1)]
        let mask = createAttentionMask(h: h, cache: reference)
        let offset = reference?.offset ?? 0

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i], offset: offset)
        }

        return norm(h)
    }
}

public class BailingMoeLinearModel: Module, LLMModel {

    public let vocabularySize: Int

    public let model: BailingMoeLinearModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: BailingMoeLinearConfiguration

    public init(_ args: BailingMoeLinearConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.model = BailingMoeLinearModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.wordEmbeddings.asLinear(out)
    }

    /// A key/value cache for the attention layers, a single array slot for the recurrent
    /// state of each linear one.
    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        try model.layers.map {
            $0.isGlobal
                ? try makeAttentionKVCache(configuration: cacheConfiguration) : ArraysCache(size: 1)
        }
    }

    /// Stacks the experts, moves the router weight under `gate_proj`, and -- when the head
    /// is untied and `norm_head` is set -- normalizes its columns.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        if configuration.tieWordEmbeddings {
            sanitized.removeValue(forKey: "lm_head.weight")
        } else if configuration.normalizeHead, let head = sanitized["lm_head.weight"] {
            let dtype = head.dtype
            let norms =
                sqrt(head.asType(.float32).square().sum(axis: 0, keepDims: true)) + 1e-7
            sanitized["lm_head.weight"] = (head / norms).asType(dtype)
        }

        for layer in configuration.firstDenseLayers ..< configuration.hiddenLayers {
            let prefix = "model.layers.\(layer)"
            for projection in ["gate_proj", "down_proj", "up_proj"] {
                for kind in ["weight", "scales", "biases"] {
                    let keys = (0 ..< configuration.experts).map {
                        "\(prefix).mlp.experts.\($0).\(projection).\(kind)"
                    }
                    guard let first = keys.first, sanitized[first] != nil else { continue }
                    let stacked = keys.compactMap { sanitized.removeValue(forKey: $0) }
                    guard stacked.count == configuration.experts else { continue }
                    sanitized["\(prefix).mlp.switch_mlp.\(projection).\(kind)"] =
                        MLX.stacked(stacked)
                }
            }

            for suffix in ["weight", "bias"] {
                if let value = sanitized.removeValue(
                    forKey: "\(prefix).mlp.gate.\(suffix)")
                {
                    sanitized["\(prefix).mlp.gate.gate_proj.\(suffix)"] = value
                }
            }
        }

        return sanitized
    }
}

public struct BailingMoeLinearConfiguration: Decodable, Sendable {

    var modelType: String
    var hiddenSize: Int
    var intermediateSize: Int
    var maxPositionEmbeddings: Int
    var moeIntermediateSize: Int
    var experts: Int
    var sharedExperts: Int
    var normalizeTopKProbabilities: Bool
    var attentionHeads: Int
    var expertsPerToken: Int
    var hiddenLayers: Int
    var kvHeads: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var vocabularySize: Int
    var firstDenseLayers: Int
    var layerGroupSize: Int
    var groupNormSize: Int
    var ropeScaling: [String: StringOrNumber]?
    var ropeTraditional: Bool
    var usesBias: Bool
    var usesQKVBias: Bool
    var normalizeHead: Bool
    var usesQKNorm: Bool
    var tieWordEmbeddings: Bool
    var partialRotaryFactor: Float
    var routerExpertBias: Bool
    var routedScalingFactor: Float
    var scoreFunction: String
    var groups: Int
    var topGroups: Int
    var moeSharedExpertIntermediateSize: Int?
    var routerSharedExpert: Bool
    var headDimensionsOverride: Int?

    var headDimensions: Int {
        headDimensionsOverride ?? (hiddenSize / attentionHeads)
    }

    /// The last layer of each complete group attends normally, as does anything trailing
    /// past the last complete group.
    func isGlobal(at layer: Int) -> Bool {
        let complete = hiddenLayers / layerGroupSize * layerGroupSize
        return (layer + 1) % layerGroupSize == 0 || layer >= complete
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case moeIntermediateSize = "moe_intermediate_size"
        case experts = "num_experts"
        case sharedExperts = "num_shared_experts"
        case normalizeTopKProbabilities = "norm_topk_prob"
        case attentionHeads = "num_attention_heads"
        case expertsPerToken = "num_experts_per_tok"
        case hiddenLayers = "num_hidden_layers"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case vocabularySize = "vocab_size"
        case firstDenseLayers = "first_k_dense_replace"
        case layerGroupSize = "layer_group_size"
        case groupNormSize = "group_norm_size"
        case ropeScaling = "rope_scaling"
        case ropeTraditional = "rope_traditional"
        case usesBias = "use_bias"
        case usesQKVBias = "use_qkv_bias"
        case normalizeHead = "norm_head"
        case usesQKNorm = "use_qk_norm"
        case tieWordEmbeddings = "tie_word_embeddings"
        case partialRotaryFactor = "partial_rotary_factor"
        case routerExpertBias = "moe_router_enable_expert_bias"
        case routedScalingFactor = "routed_scaling_factor"
        case scoreFunction = "score_function"
        case groups = "n_group"
        case topGroups = "topk_group"
        case moeSharedExpertIntermediateSize = "moe_shared_expert_intermediate_size"
        case routerSharedExpert = "moe_router_enable_shared_expert"
        case headDimensionsOverride = "head_dim"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType) ?? "bailing_moe_linear"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        experts = try c.decode(Int.self, forKey: .experts)
        sharedExperts = try c.decode(Int.self, forKey: .sharedExperts)
        normalizeTopKProbabilities = try c.decode(
            Bool.self, forKey: .normalizeTopKProbabilities)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        expertsPerToken = try c.decode(Int.self, forKey: .expertsPerToken)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        firstDenseLayers = try c.decode(Int.self, forKey: .firstDenseLayers)
        layerGroupSize = try c.decode(Int.self, forKey: .layerGroupSize)
        groupNormSize = try c.decode(Int.self, forKey: .groupNormSize)
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        usesBias = try c.decodeIfPresent(Bool.self, forKey: .usesBias) ?? false
        usesQKVBias = try c.decodeIfPresent(Bool.self, forKey: .usesQKVBias) ?? false
        normalizeHead = try c.decodeIfPresent(Bool.self, forKey: .normalizeHead) ?? false
        usesQKNorm = try c.decodeIfPresent(Bool.self, forKey: .usesQKNorm) ?? false
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        partialRotaryFactor =
            try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 1
        routerExpertBias =
            try c.decodeIfPresent(Bool.self, forKey: .routerExpertBias) ?? false
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1
        scoreFunction =
            try c.decodeIfPresent(String.self, forKey: .scoreFunction) ?? "softmax"
        groups = try c.decodeIfPresent(Int.self, forKey: .groups) ?? 1
        topGroups = try c.decodeIfPresent(Int.self, forKey: .topGroups) ?? 4
        moeSharedExpertIntermediateSize = try c.decodeIfPresent(
            Int.self, forKey: .moeSharedExpertIntermediateSize)
        routerSharedExpert =
            try c.decodeIfPresent(Bool.self, forKey: .routerSharedExpert) ?? true
        headDimensionsOverride = try c.decodeIfPresent(
            Int.self, forKey: .headDimensionsOverride)
    }
}

extension BailingMoeLinearModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
