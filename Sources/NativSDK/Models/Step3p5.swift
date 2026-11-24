import Foundation
import MLX
import MLXNN

/// `clip(silu(gate), max: limit) * clip(up, -limit, limit)`.
///
/// The gate is bounded above only; the up branch is bounded on both sides.
func step3p5ClampedSwiGLU(gate: MLXArray, up: MLXArray, limit: Float) -> MLXArray {
    clip(silu(gate), max: limit) * clip(up, min: -limit, max: limit)
}

/// RMS normalization that uses its weight directly, with no implicit `1 +`.
class Step3p5RMSNorm: Module, UnaryLayer {

    @ModuleInfo var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

class Step3p5MLP: Module, UnaryLayer {

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    let limit: Float?

    init(_ args: Step3p5Configuration, intermediateSize: Int, swiGLULimit: Float) {
        _gateProj.wrappedValue = Linear(args.hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(args.hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, args.hiddenSize, bias: false)
        self.limit = swiGLULimit > 0 ? swiGLULimit : nil
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = gateProj(x)
        let up = upProj(x)
        if let limit {
            return downProj(step3p5ClampedSwiGLU(gate: gate, up: up, limit: limit))
        }
        return downProj(compiledSiluProduct(gate, up))
    }
}

class Step3p5MoEGate: Module {

    let topK: Int
    let routedScalingFactor: Float
    let normalizeTopK: Bool

    @ModuleInfo(key: "gate") var gate: Linear
    @ParameterInfo(key: "router_bias") var routerBias: MLXArray

    init(_ args: Step3p5Configuration) {
        self.topK = args.moeTopK
        self.routedScalingFactor = args.moeRouterScalingFactor
        self.normalizeTopK = args.normalizeExpertWeight

        _gate.wrappedValue = Linear(args.hiddenSize, args.moeExperts, bias: false)
        _routerBias.wrappedValue = MLXArray.zeros([args.moeExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        groupedExpertSelection(
            gates: gate(x),
            correctionBias: routerBias,
            topK: topK,
            groups: 1,
            topGroups: 1,
            routedScalingFactor: routedScalingFactor,
            normalizeTopK: normalizeTopK,
            scoring: .sigmoid)
    }
}

class Step3p5MoE: Module, UnaryLayer {

    @ModuleInfo(key: "gate") var gate: Step3p5MoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "share_expert") var shareExpert: Step3p5MLP

    init(_ args: Step3p5Configuration, layerIndex: Int) {
        let routedLimit = args.swiGLULimit(at: layerIndex)
        let sharedLimit = args.sharedSwiGLULimit(at: layerIndex)

        _gate.wrappedValue = Step3p5MoEGate(args)
        if routedLimit > 0 {
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
                numExperts: args.moeExperts,
                activationProduct: { gate, up in
                    step3p5ClampedSwiGLU(gate: gate, up: up, limit: routedLimit)
                })
        } else {
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
                numExperts: args.moeExperts)
        }
        _shareExpert.wrappedValue = Step3p5MLP(
            args, intermediateSize: args.shareExpertDimensions, swiGLULimit: sharedLimit)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, weights) = gate(x)
        let routed = switchMLP(x, indices)
        let combined = (routed * weights[.ellipsis, .newAxis]).sum(axis: -2)
            .asType(routed.dtype)
        return combined + shareExpert(x)
    }
}

class Step3p5Attention: Module {

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let isSliding: Bool
    let usesHeadWiseGate: Bool

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ModuleInfo(key: "g_proj") var wg: Linear?

    @ModuleInfo(key: "q_norm") var qNorm: Step3p5RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Step3p5RMSNorm

    let rope: RoPELayer

    init(_ args: Step3p5Configuration, layerIndex: Int) {
        self.isSliding = args.isSliding(at: layerIndex)

        if isSliding, let other = args.attentionOtherSetting {
            self.heads = other.heads
            self.kvHeads = other.kvHeads
        } else {
            self.heads = args.attentionHeads
            self.kvHeads = args.attentionGroups
        }

        self.headDim = args.headDimensions
        self.scale = pow(Float(args.headDimensions), -0.5)
        self.usesHeadWiseGate = args.usesHeadWiseAttentionGate

        _wq.wrappedValue = Linear(args.hiddenSize, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(args.hiddenSize, kvHeads * headDim, bias: false)
        _wv.wrappedValue = Linear(args.hiddenSize, kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, args.hiddenSize, bias: false)
        if usesHeadWiseGate {
            _wg.wrappedValue = Linear(args.hiddenSize, heads, bias: false)
        }

        _qNorm.wrappedValue = Step3p5RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _kNorm.wrappedValue = Step3p5RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

        self.rope = initializeRope(
            dims: Int(Float(headDim) * args.partialRotaryFactor(at: layerIndex)),
            base: args.ropeTheta(at: layerIndex),
            traditional: false,
            scalingConfig: args.ropeScaling(at: layerIndex),
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

        var output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache,
            scale: scale, mask: mask
        ).transposed(0, 2, 1, 3)

        if let wg {
            output = output * sigmoid(wg(x))[.ellipsis, .newAxis]
        }

        return wo(output.reshaped(B, L, -1))
    }
}

class Step3p5Block: Module {

    @ModuleInfo(key: "self_attn") var attention: Step3p5Attention
    @ModuleInfo(key: "mlp") var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Step3p5RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Step3p5RMSNorm

    let isMoELayer: Bool

    var isSliding: Bool { attention.isSliding }

    init(_ args: Step3p5Configuration, layerIndex: Int) {
        _attention.wrappedValue = Step3p5Attention(args, layerIndex: layerIndex)
        self.isMoELayer = args.isMoELayer(at: layerIndex)

        if isMoELayer {
            _mlp.wrappedValue = Step3p5MoE(args, layerIndex: layerIndex)
        } else {
            _mlp.wrappedValue = Step3p5MLP(
                args, intermediateSize: args.intermediateSize,
                swiGLULimit: args.sharedSwiGLULimit(at: layerIndex))
        }

        _inputLayerNorm.wrappedValue = Step3p5RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = Step3p5RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class Step3p5ModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [Step3p5Block]
    let norm: Step3p5RMSNorm

    let slidingWindow: Int

    public init(_ args: Step3p5Configuration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { Step3p5Block(args, layerIndex: $0) }
        self.norm = Step3p5RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self.slidingWindow = args.slidingWindow
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let firstFull = layers.firstIndex { !$0.isSliding }
        let firstSliding = layers.firstIndex { $0.isSliding }
        let fullMask =
            firstFull.map { createAttentionMask(h: h, cache: cache?[$0]) } ?? .none
        let slidingMask =
            firstSliding.map {
                createAttentionMask(h: h, cache: cache?[$0], windowSize: slidingWindow)
            } ?? .none

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: layer.isSliding ? slidingMask : fullMask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class Step3p5Model: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Step3p5ModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let configuration: Step3p5Configuration

    public init(_ args: Step3p5Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.model = Step3p5ModelInner(args)
        self.kvHeads = model.layers.map(\.attention.kvHeads)
        _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        try model.layers.map { layer in
            layer.isSliding
                ? try makeSlidingWindowKVCache(
                    configuration: cacheConfiguration, window: configuration.slidingWindow)
                : try makeAttentionKVCache(configuration: cacheConfiguration)
        }
    }

    /// Remaps the pre-conversion `.moe.` names, drops the extra prediction head and any
    /// layer past `num_hidden_layers`, and -- only for a checkpoint that still needed
    /// remapping -- adds one to every norm weight.
    ///
    /// That last step is what makes a zero-centered weight mean the same thing as an
    /// already-converted one. Applying it to a converted checkpoint, or skipping it on a
    /// vanilla one, scales every normalization in the model by about one.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let remappings = [
            (".moe.gate_proj.", ".mlp.switch_mlp.gate_proj."),
            (".moe.up_proj.", ".mlp.switch_mlp.up_proj."),
            (".moe.down_proj.", ".mlp.switch_mlp.down_proj."),
            (".moe.gate.", ".mlp.gate.gate."),
            (".moe.router_bias", ".mlp.gate.router_bias"),
            (".share_expert.", ".mlp.share_expert."),
        ]

        let isVanilla = weights.keys.contains { key in
            remappings.contains { key.contains($0.0) && !key.contains($0.1) }
        }

        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains(".mtp") {
                continue
            }
            if let range = key.range(of: "model.layers.") {
                let digits = key[range.upperBound...].prefix { $0.isNumber }
                if let index = Int(digits), index >= configuration.hiddenLayers {
                    continue
                }
            }

            var name = key
            for (source, destination) in remappings
            where name.contains(source) && !name.contains(destination) {
                name = name.replacingOccurrences(of: source, with: destination)
                break
            }

            if isVanilla, name.hasSuffix(".weight"), name.contains("norm") {
                sanitized[name] = value + 1
            } else {
                sanitized[name] = value
            }
        }
        return sanitized
    }
}

public struct Step3p5Configuration: Decodable, Sendable {

    /// The head counts a sliding layer uses instead of the model's.
    public struct OtherAttentionSetting: Decodable, Sendable {
        var heads: Int
        var kvHeads: Int

        enum CodingKeys: String, CodingKey {
            case heads = "num_attention_heads"
            case kvHeads = "num_attention_groups"
        }
    }

    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var vocabularySize: Int
    var attentionHeads: Int
    var attentionGroups: Int
    var headDimensions: Int
    var intermediateSize: Int
    var rmsNormEps: Float
    var ropeThetas: [Float]
    var ropeScalingConfig: [String: StringOrNumber]?
    var maxPositionEmbeddings: Int
    var slidingWindow: Int
    var layerTypes: [String]
    var yarnOnlyTypes: [String]
    var partialRotaryFactors: [Float]
    var attentionOtherSetting: OtherAttentionSetting?
    var usesHeadWiseAttentionGate: Bool
    var moeExperts: Int
    var moeTopK: Int
    var moeIntermediateSize: Int
    var shareExpertDimensions: Int
    var moeLayersEnum: String?
    var moeRouterScalingFactor: Float
    var normalizeExpertWeight: Bool
    var swiGLULimits: [Float]
    var swiGLULimitsShared: [Float]
    var tieWordEmbeddings: Bool

    /// With no `layer_types`, the *even* layers slide.
    func isSliding(at layer: Int) -> Bool {
        layerTypes.isEmpty ? layer % 2 == 0 : layerTypes[layer] == "sliding_attention"
    }

    /// Absent `moe_layers_enum`, every layer but layer 0 is a mixture.
    func isMoELayer(at layer: Int) -> Bool {
        guard let moeLayersEnum, !moeLayersEnum.isEmpty else { return layer >= 1 }
        return moeLayersEnum.split(separator: ",")
            .compactMap { Int($0.trimmingCharacters(in: .whitespaces)) }
            .contains(layer)
    }

    func ropeTheta(at layer: Int) -> Float {
        ropeThetas.count > 1 ? ropeThetas[layer] : (ropeThetas.first ?? 10_000)
    }

    func partialRotaryFactor(at layer: Int) -> Float {
        layer < partialRotaryFactors.count ? partialRotaryFactors[layer] : 1
    }

    /// Rotary scaling applies only to the layer types `yarn_only_types` names.
    func ropeScaling(at layer: Int) -> [String: StringOrNumber]? {
        guard !yarnOnlyTypes.isEmpty else { return ropeScalingConfig }
        let type = layerTypes.isEmpty ? "full_attention" : layerTypes[layer]
        return yarnOnlyTypes.contains(type) ? ropeScalingConfig : nil
    }

    func swiGLULimit(at layer: Int) -> Float {
        layer < swiGLULimits.count ? swiGLULimits[layer] : 0
    }

    func sharedSwiGLULimit(at layer: Int) -> Float {
        layer < swiGLULimitsShared.count ? swiGLULimitsShared[layer] : 0
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case vocabularySize = "vocab_size"
        case attentionHeads = "num_attention_heads"
        case attentionGroups = "num_attention_groups"
        case headDimensions = "head_dim"
        case intermediateSize = "intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case yarnOnlyTypes = "yarn_only_types"
        case partialRotaryFactors = "partial_rotary_factors"
        case attentionOtherSetting = "attention_other_setting"
        case usesHeadWiseAttentionGate = "use_head_wise_attn_gate"
        case moeExperts = "moe_num_experts"
        case moeTopK = "moe_top_k"
        case moeIntermediateSize = "moe_intermediate_size"
        case shareExpertDimensions = "share_expert_dim"
        case moeLayersEnum = "moe_layers_enum"
        case moeRouterScalingFactor = "moe_router_scaling_factor"
        case normalizeExpertWeight = "norm_expert_weight"
        case swiGLULimits = "swiglu_limits"
        case swiGLULimitsShared = "swiglu_limits_shared"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "step3p5"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        attentionGroups = try c.decode(Int.self, forKey: .attentionGroups)
        headDimensions = try c.decode(Int.self, forKey: .headDimensions)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5

        if let single = try? c.decodeIfPresent(Float.self, forKey: .ropeTheta) {
            ropeThetas = [single]
        } else {
            ropeThetas = try c.decodeIfPresent([Float].self, forKey: .ropeTheta) ?? [10_000]
        }

        ropeScalingConfig = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262_144
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes) ?? []
        yarnOnlyTypes = try c.decodeIfPresent([String].self, forKey: .yarnOnlyTypes) ?? []
        partialRotaryFactors =
            try c.decodeIfPresent([Float?].self, forKey: .partialRotaryFactors)?
            .map { $0 ?? 1 } ?? []
        attentionOtherSetting = try c.decodeIfPresent(
            OtherAttentionSetting.self, forKey: .attentionOtherSetting)
        usesHeadWiseAttentionGate =
            try c.decodeIfPresent(Bool.self, forKey: .usesHeadWiseAttentionGate) ?? true
        moeExperts = try c.decodeIfPresent(Int.self, forKey: .moeExperts) ?? 288
        moeTopK = try c.decodeIfPresent(Int.self, forKey: .moeTopK) ?? 8
        moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 1280
        shareExpertDimensions =
            try c.decodeIfPresent(Int.self, forKey: .shareExpertDimensions) ?? 1280
        moeLayersEnum = try c.decodeIfPresent(String.self, forKey: .moeLayersEnum)
        moeRouterScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .moeRouterScalingFactor) ?? 3
        normalizeExpertWeight =
            try c.decodeIfPresent(Bool.self, forKey: .normalizeExpertWeight) ?? true
        swiGLULimits =
            try c.decodeIfPresent([Float?].self, forKey: .swiGLULimits)?.map { $0 ?? 0 } ?? []
        swiGLULimitsShared =
            try c.decodeIfPresent([Float?].self, forKey: .swiGLULimitsShared)?
            .map { $0 ?? 0 } ?? []
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

extension Step3p5Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
