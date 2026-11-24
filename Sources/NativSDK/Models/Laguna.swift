import Foundation
import MLX
import MLXNN

private struct LagunaDynamicCodingKey: CodingKey {
    let stringValue: String
    let intValue: Int?

    init?(stringValue: String) {
        self.stringValue = stringValue
        intValue = nil
    }

    init?(intValue: Int) {
        stringValue = "\(intValue)"
        self.intValue = intValue
    }
}

private struct LagunaRopeParameters: Codable {
    var values: [String: StringOrNumber]
    var nested: [String: [String: StringOrNumber]]

    init(values: [String: StringOrNumber] = [:], nested: [String: [String: StringOrNumber]] = [:]) {
        self.values = values
        self.nested = nested
    }

    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: LagunaDynamicCodingKey.self)
        var values: [String: StringOrNumber] = [:]
        var nested: [String: [String: StringOrNumber]] = [:]

        for key in c.allKeys {
            if let scalar = try? c.decode(StringOrNumber.self, forKey: key) {
                values[key.stringValue] = scalar
            } else if let dictionary = try? c.decode(
                [String: StringOrNumber].self,
                forKey: key,
            ) {
                nested[key.stringValue] = dictionary
            }
        }

        self.values = values
        self.nested = nested
    }

    func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: LagunaDynamicCodingKey.self)
        for (key, value) in values {
            try c.encode(value, forKey: LagunaDynamicCodingKey(stringValue: key)!)
        }
        for (key, value) in nested {
            try c.encode(value, forKey: LagunaDynamicCodingKey(stringValue: key)!)
        }
    }

    func parameters(
        for layerType: String,
        ropeTheta: Float,
        partialRotaryFactor: Float?,
    )
        -> [String: StringOrNumber]
    {
        var result = values
        if let layerValues = nested[layerType] {
            for (key, value) in layerValues {
                result[key] = value
            }
        }
        result["rope_type"] = result["rope_type"] ?? .string("default")
        result["rope_theta"] = result["rope_theta"] ?? .float(ropeTheta)
        if let partialRotaryFactor {
            result["partial_rotary_factor"] =
                result["partial_rotary_factor"] ?? .float(partialRotaryFactor)
        }
        return result
    }
}

public struct LagunaConfiguration: Codable, Sendable {
    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let maxPositionEmbeddings: Int
    public let rmsNormEps: Float
    public let qkvBias: Bool
    public let attentionBias: Bool
    public let gating: String
    public let tieWordEmbeddings: Bool
    public let ropeTheta: Float
    private let ropeParameters: LagunaRopeParameters?
    public let partialRotaryFactor: Float?
    public let slidingWindow: Int?
    public let layerTypes: [String]
    public let attentionHeadsPerLayer: [Int]
    public let swaAttentionSinkEnabled: Bool
    public let numExperts: Int
    public let numExpertsPerToken: Int
    public let moeIntermediateSize: Int
    public let sharedExpertIntermediateSize: Int
    public let normTopkProb: Bool
    public let decoderSparseStep: Int
    public let mlpOnlyLayers: [Int]
    public let mlpLayerTypes: [String]?
    public let moeRoutedScalingFactor: Float
    public let moeApplyRouterWeightOnInput: Bool
    public let moeRouterLogitSoftcapping: Float
    public let moeRouterUseSigmoid: Bool
    public let bosTokenID: Int?
    public let eosTokenID: IntOrIntArray?
    public let padTokenID: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case qkvBias = "qkv_bias"
        case attentionBias = "attention_bias"
        case gating
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case ropeScaling = "rope_scaling"
        case partialRotaryFactor = "partial_rotary_factor"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case attentionHeadsPerLayer = "num_attention_heads_per_layer"
        case swaAttentionSinkEnabled = "swa_attention_sink_enabled"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case decoderSparseStep = "decoder_sparse_step"
        case mlpOnlyLayers = "mlp_only_layers"
        case mlpLayerTypes = "mlp_layer_types"
        case moeRoutedScalingFactor = "moe_routed_scaling_factor"
        case moeApplyRouterWeightOnInput = "moe_apply_router_weight_on_input"
        case moeRouterLogitSoftcapping = "moe_router_logit_softcapping"
        case moeRouterUseSigmoid = "moe_router_use_sigmoid"
        case bosTokenID = "bos_token_id"
        case eosTokenID = "eos_token_id"
        case padTokenID = "pad_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "laguna"
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDim = try c.decode(Int.self, forKey: .headDim)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        qkvBias = try c.decodeIfPresent(Bool.self, forKey: .qkvBias) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        if let gatingBool = try? c.decode(Bool.self, forKey: .gating) {
            gating = gatingBool ? "per-head" : "none"
        } else {
            gating = try c.decodeIfPresent(String.self, forKey: .gating) ?? "per-head"
        }
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000
        let decodedRopeParameters = try c.decodeIfPresent(
            LagunaRopeParameters.self,
            forKey: .ropeParameters,
        )
        let decodedRopeScaling = try c.decodeIfPresent(
            LagunaRopeParameters.self,
            forKey: .ropeScaling,
        )
        ropeParameters = decodedRopeParameters ?? decodedRopeScaling
        partialRotaryFactor = try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
        layerTypes =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? Array(repeating: "full_attention", count: hiddenLayers)
        attentionHeadsPerLayer =
            try c.decodeIfPresent([Int].self, forKey: .attentionHeadsPerLayer)
            ?? Array(repeating: attentionHeads, count: hiddenLayers)
        swaAttentionSinkEnabled =
            try c.decodeIfPresent(Bool.self, forKey: .swaAttentionSinkEnabled) ?? false
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
        numExpertsPerToken = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 0
        moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
        sharedExpertIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        decoderSparseStep = try c.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
        mlpLayerTypes = try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes)
        if let mlpLayerTypes {
            mlpOnlyLayers = mlpLayerTypes.enumerated().compactMap { index, type in
                type == "dense" ? index : nil
            }
        } else {
            mlpOnlyLayers = try c.decodeIfPresent([Int].self, forKey: .mlpOnlyLayers) ?? [0]
        }
        moeRoutedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .moeRoutedScalingFactor) ?? 1.0
        moeApplyRouterWeightOnInput =
            try c.decodeIfPresent(Bool.self, forKey: .moeApplyRouterWeightOnInput) ?? false
        moeRouterLogitSoftcapping =
            try c.decodeIfPresent(Float.self, forKey: .moeRouterLogitSoftcapping) ?? 0
        moeRouterUseSigmoid =
            try c.decodeIfPresent(Bool.self, forKey: .moeRouterUseSigmoid) ?? true
        bosTokenID = try c.decodeIfPresent(Int.self, forKey: .bosTokenID)
        eosTokenID = try c.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenID)
        padTokenID = try c.decodeIfPresent(Int.self, forKey: .padTokenID)

        precondition(layerTypes.count == hiddenLayers, "layer_types must match num_hidden_layers")
        precondition(
            attentionHeadsPerLayer.count == hiddenLayers,
            "num_attention_heads_per_layer must match num_hidden_layers",
        )
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(vocabularySize, forKey: .vocabularySize)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(qkvBias, forKey: .qkvBias)
        try c.encode(attentionBias, forKey: .attentionBias)
        try c.encode(gating, forKey: .gating)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encodeIfPresent(ropeParameters, forKey: .ropeParameters)
        try c.encodeIfPresent(partialRotaryFactor, forKey: .partialRotaryFactor)
        try c.encodeIfPresent(slidingWindow, forKey: .slidingWindow)
        try c.encode(layerTypes, forKey: .layerTypes)
        try c.encode(attentionHeadsPerLayer, forKey: .attentionHeadsPerLayer)
        try c.encode(swaAttentionSinkEnabled, forKey: .swaAttentionSinkEnabled)
        try c.encode(numExperts, forKey: .numExperts)
        try c.encode(numExpertsPerToken, forKey: .numExpertsPerToken)
        try c.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try c.encode(sharedExpertIntermediateSize, forKey: .sharedExpertIntermediateSize)
        try c.encode(normTopkProb, forKey: .normTopkProb)
        try c.encode(decoderSparseStep, forKey: .decoderSparseStep)
        try c.encode(mlpOnlyLayers, forKey: .mlpOnlyLayers)
        try c.encodeIfPresent(mlpLayerTypes, forKey: .mlpLayerTypes)
        try c.encode(moeRoutedScalingFactor, forKey: .moeRoutedScalingFactor)
        try c.encode(moeApplyRouterWeightOnInput, forKey: .moeApplyRouterWeightOnInput)
        try c.encode(moeRouterLogitSoftcapping, forKey: .moeRouterLogitSoftcapping)
        try c.encode(moeRouterUseSigmoid, forKey: .moeRouterUseSigmoid)
        try c.encodeIfPresent(bosTokenID, forKey: .bosTokenID)
        try c.encodeIfPresent(eosTokenID, forKey: .eosTokenID)
        try c.encodeIfPresent(padTokenID, forKey: .padTokenID)
    }

    func isSlidingLayer(_ layerIndex: Int) -> Bool {
        layerTypes[layerIndex] == "sliding_attention"
    }

    func usesSparseMLP(_ layerIndex: Int) -> Bool {
        !mlpOnlyLayers.contains(layerIndex)
            && numExperts > 0
            && (layerIndex + 1) % decoderSparseStep == 0
    }

    func ropeParameters(for layerType: String) -> [String: StringOrNumber] {
        (ropeParameters ?? LagunaRopeParameters()).parameters(
            for: layerType,
            ropeTheta: ropeTheta,
            partialRotaryFactor: partialRotaryFactor,
        )
    }
}

private final class LagunaMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

private final class LagunaTopKRouter: Module {
    let config: LagunaConfiguration

    @ModuleInfo var proj: Linear
    @ModuleInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

    init(_ config: LagunaConfiguration) {
        self.config = config
        _proj.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        _eScoreCorrectionBias.wrappedValue = zeros([config.numExperts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
        var logits = proj(x).asType(.float32)
        if config.moeRouterLogitSoftcapping > 0 {
            let cap = config.moeRouterLogitSoftcapping
            logits = tanh(logits / cap) * cap
        }

        let scores =
            config.moeRouterUseSigmoid
            ? sigmoid(logits)
            : softmax(logits, axis: -1, precise: true)
        let corrected = scores + eScoreCorrectionBias.asType(scores.dtype)
        let indices = argPartition(-corrected, kth: config.numExpertsPerToken - 1, axis: -1)[
            .ellipsis, ..<config.numExpertsPerToken,
        ]
        var weights = takeAlong(scores, indices, axis: -1)
        if config.normTopkProb {
            weights = weights / weights.sum(axis: -1, keepDims: true)
        }
        return (indices, weights.asType(x.dtype))
    }
}

private final class LagunaSparseMoEBlock: Module, UnaryLayer {
    let config: LagunaConfiguration

    @ModuleInfo var gate: LagunaTopKRouter
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: LagunaMLP

    init(_ config: LagunaConfiguration) {
        precondition(
            !config.moeApplyRouterWeightOnInput,
            "Laguna moe_apply_router_weight_on_input is not supported.",
        )
        self.config = config
        _gate.wrappedValue = LagunaTopKRouter(config)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.numExperts,
        )
        _sharedExpert.wrappedValue = LagunaMLP(
            dimensions: config.hiddenSize,
            hiddenDimensions: config.sharedExpertIntermediateSize,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = (switchMLP(x, indices) * expandedDimensions(scores, axis: -1)).sum(axis: -2)
        if config.moeRoutedScalingFactor != 1.0 {
            y = y * config.moeRoutedScalingFactor
        }
        return y + sharedExpert(x)
    }
}

private final class LagunaAttention: Module {
    let config: LagunaConfiguration
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float
    let gatePerHead: Bool
    let useGating: Bool
    let isSliding: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "g_proj") var gProj: Linear?
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let rope: RoPELayer

    init(_ config: LagunaConfiguration, layerIndex: Int) {
        self.config = config
        nHeads = config.attentionHeadsPerLayer[layerIndex]
        nKVHeads = config.kvHeads
        headDim = config.headDim
        scale = pow(Float(headDim), -0.5)
        gatePerHead = config.gating == "per-head"
        useGating = config.gating != "none" && config.gating != "false"
        isSliding = config.isSlidingLayer(layerIndex)

        precondition(
            !isSliding || !config.swaAttentionSinkEnabled,
            "Laguna sliding-window attention sinks are not supported.",
        )

        _qProj.wrappedValue = Linear(
            config.hiddenSize,
            nHeads * headDim,
            bias: config.qkvBias,
        )
        _kProj.wrappedValue = Linear(
            config.hiddenSize,
            nKVHeads * headDim,
            bias: config.qkvBias,
        )
        _vProj.wrappedValue = Linear(
            config.hiddenSize,
            nKVHeads * headDim,
            bias: config.qkvBias,
        )
        _oProj.wrappedValue = Linear(
            nHeads * headDim,
            config.hiddenSize,
            bias: config.attentionBias,
        )
        if useGating {
            _gProj.wrappedValue = Linear(
                config.hiddenSize,
                gatePerHead ? nHeads : nHeads * headDim,
                bias: false,
            )
        }
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        let layerType = isSliding ? "sliding_attention" : "full_attention"
        let ropeConfig = config.ropeParameters(for: layerType)
        let partial =
            ropeConfig["partial_rotary_factor"]?.asFloat()
            ?? config.partialRotaryFactor
            ?? 1.0
        rope = initializeRope(
            dims: Int(Float(headDim) * partial),
            base: ropeConfig["rope_theta"]?.asFloat() ?? config.ropeTheta,
            traditional: false,
            scalingConfig: ropeConfig,
            maxPositionEmbeddings: config.maxPositionEmbeddings,
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
    )
        -> MLXArray
    {
        let (batchSize, sequenceLength) = (x.dim(0), x.dim(1))
        var queries = qProj(x)
        var keys = kProj(x)
        var values = vProj(x)

        queries = qNorm(queries.reshaped(batchSize, sequenceLength, nHeads, headDim))
            .transposed(0, 2, 1, 3)
        keys = kNorm(keys.reshaped(batchSize, sequenceLength, nKVHeads, headDim))
            .transposed(0, 2, 1, 3)
        values = values.reshaped(batchSize, sequenceLength, nKVHeads, headDim)
            .transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, offset: cache?.ropeOffset)
        keys = applyRotaryPosition(rope, to: keys, offset: cache?.ropeOffset)

        var output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask,
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, sequenceLength, -1)

        if let gProj {
            let gate = softplus(gProj(x).asType(.float32)).asType(output.dtype)
            if gatePerHead {
                let shape = output.shape
                output =
                    (output.reshaped(batchSize, sequenceLength, nHeads, headDim)
                    * expandedDimensions(gate, axis: -1))
                    .reshaped(shape)
            } else {
                output = output * gate
            }
        }

        return oProj(output)
    }
}

private final class LagunaDecoderLayer: Module {
    let attentionType: String

    @ModuleInfo(key: "self_attn") var attention: LagunaAttention
    @ModuleInfo var mlp: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: LagunaConfiguration, layerIndex: Int) {
        attentionType = config.layerTypes[layerIndex]
        _attention.wrappedValue = LagunaAttention(config, layerIndex: layerIndex)
        if config.usesSparseMLP(layerIndex) {
            _mlp.wrappedValue = LagunaSparseMoEBlock(config)
        } else {
            _mlp.wrappedValue = LagunaMLP(
                dimensions: config.hiddenSize,
                hiddenDimensions: config.intermediateSize,
            )
        }
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
    )
        -> MLXArray
    {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

private final class LagunaModelInner: Module {
    let config: LagunaConfiguration
    let fullAttentionIndex: Int
    let slidingAttentionIndex: Int?

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [LagunaDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ config: LagunaConfiguration) {
        self.config = config
        fullAttentionIndex = config.layerTypes.firstIndex(of: "full_attention") ?? 0
        slidingAttentionIndex = config.layerTypes.firstIndex(of: "sliding_attention")
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize,
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            LagunaDecoderLayer(config, layerIndex: $0)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        inputEmbeddings: MLXArray? = nil,
        hiddenSink: ((Int, MLXArray) -> Void)? = nil,
    )
        -> MLXArray
    {
        var h = inputEmbeddings ?? embedTokens(inputs)
        let fullMask = createAttentionMask(h: h, cache: cache?[fullAttentionIndex])
        let slidingMask =
            slidingAttentionIndex.map {
                createAttentionMask(h: h, cache: cache?[$0], windowSize: config.slidingWindow)
            } ?? .none

        for (i, layer) in layers.enumerated() {
            let mask = layer.attentionType == "sliding_attention" ? slidingMask : fullMask
            h = layer(h, mask: mask, cache: cache?[i])
            hiddenSink?(i, h)
        }

        return norm(h)
    }
}

private final class LagunaLanguageModel: Module {
    let config: LagunaConfiguration

    @ModuleInfo var model: LagunaModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    init(_ config: LagunaConfiguration) {
        self.config = config
        _model.wrappedValue = LagunaModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let out = model(inputs, cache: cache)
        if config.tieWordEmbeddings {
            return model.embedTokens.asLinear(out)
        }
        return lmHead!(out)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        if config.tieWordEmbeddings {
            sanitized.removeValue(forKey: "lm_head.weight")
            sanitized.removeValue(forKey: "language_model.lm_head.weight")
        }

        for basePrefix in ["", "language_model."] {
            for layerIndex in 0 ..< config.hiddenLayers {
                let mlpPrefix = "\(basePrefix)model.layers.\(layerIndex).mlp"
                if let gateWeight = sanitized.removeValue(forKey: "\(mlpPrefix).gate.weight") {
                    sanitized["\(mlpPrefix).gate.proj.weight"] = gateWeight
                }
                if let legacyBias = sanitized.removeValue(
                    forKey: "\(mlpPrefix).experts.e_score_correction_bias",
                ) {
                    sanitized["\(mlpPrefix).gate.e_score_correction_bias"] = legacyBias
                }

                for proj in ["gate_proj", "up_proj", "down_proj"] {
                    for suffix in ["weight", "scales", "biases"] {
                        let firstKey = "\(mlpPrefix).experts.0.\(proj).\(suffix)"
                        guard sanitized[firstKey] != nil else {
                            continue
                        }
                        let tensors = (0 ..< config.numExperts).map { expert in
                            sanitized.removeValue(
                                forKey: "\(mlpPrefix).experts.\(expert).\(proj).\(suffix)",
                            )!
                        }
                        sanitized["\(mlpPrefix).switch_mlp.\(proj).\(suffix)"] =
                            MLX.stacked(tensors)
                    }
                }
            }
        }

        for key in Array(sanitized.keys)
        where key.contains("rotary_emb.inv_freq")
            || key.hasSuffix(".self_attn.k_scale")
            || key.hasSuffix(".self_attn.v_scale")
        {
            sanitized.removeValue(forKey: key)
        }

        return sanitized
    }
}

public final class Laguna: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let config: LagunaConfiguration

    @ModuleInfo(key: "language_model") fileprivate var languageModel: LagunaLanguageModel

    public init(_ config: LagunaConfiguration) {
        self.config = config
        vocabularySize = config.vocabularySize
        kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        _languageModel.wrappedValue = LagunaLanguageModel(config)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try (0 ..< config.hiddenLayers).map { layerIndex in
            if config.isSlidingLayer(layerIndex), let slidingWindow = config.slidingWindow {
                return RotatingKVCache(maxSize: slidingWindow)
            }
            return try makeAttentionKVCache(configuration: configuration)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        languageModel.sanitize(weights: weights).reduce(into: [:]) { result, item in
            let key = item.key
            if key.hasPrefix("language_model.") {
                result[key] = item.value
            } else if key.hasPrefix("model.") || key.hasPrefix("lm_head.") {
                result["language_model.\(key)"] = item.value
            } else {
                result[key] = item.value
            }
        }
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

extension Laguna: DFlashTarget {
    var draftHiddenSize: Int { config.hiddenSize }
    var draftLayerCount: Int { config.hiddenLayers }
    var draftEmbedding: Embedding { languageModel.model.embedTokens }
    func draftReadout(_ hidden: MLXArray) -> MLXArray {
        languageModel.lmHead.map { $0(hidden) } ?? draftEmbedding.asLinear(hidden)
    }
    func draftForward(
        _ tokens: MLXArray, cache: [KVCache], capture: [Int], verify: Bool,
        inputEmbeddings: MLXArray? = nil, recordTransaction: Bool = false,
        state: LMOutput.State? = nil, positionIds: MLXArray? = nil
    ) throws -> DFlashTargetOutput {
        let transaction =
            try verify || recordTransaction
            ? AttentionDraftTransaction(cache, length: tokens.dim(1)) : nil
        do {
            try Task.checkCancellation()
            let selected = Set(capture)
            var features: [MLXArray] = []
            let hidden = languageModel.model(
                tokens, cache: transaction?.caches ?? cache, inputEmbeddings: inputEmbeddings,
                hiddenSink: { index, h in
                    if selected.contains(index) { features.append(h) }
                })
            return DFlashTargetOutput(
                logits: draftReadout(hidden), features: concatenated(features, axis: -1),
                commit: transaction.map { value in { try value.commit($0) } })
        } catch {
            try transaction?.commit(0)
            throw error
        }
    }
}
