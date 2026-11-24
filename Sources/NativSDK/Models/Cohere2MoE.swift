import Foundation
import MLX
import MLXNN

public struct Cohere2MoEConfiguration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let headDim: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let ropeTheta: Float
    public let vocabularySize: Int
    public let layerNormEps: Float
    public let logitScale: Float
    public let attentionBias: Bool
    public let layerNormBias: Bool
    public let slidingWindow: Int
    public let slidingWindowPattern: Int
    public let numExperts: Int
    public let numExpertsPerToken: Int
    public let normTopkProb: Bool
    public let moeNumSharedExperts: Int
    public let moeGateActivation: String
    public let sharedExpertCombinationStrategy: String
    public let rmsNormEps: Float?
    public let firstKDenseReplace: Int
    public let prefixDenseIntermediateSize: Int
    public let prefixDenseSlidingWindowPattern: Int
    public let layerTypes: [String]?
    public let bosTokenID: Int?
    public let eosTokenID: IntOrIntArray?
    public let padTokenID: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case headDim = "head_dim"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case ropeTheta = "rope_theta"
        case vocabularySize = "vocab_size"
        case layerNormEps = "layer_norm_eps"
        case logitScale = "logit_scale"
        case attentionBias = "attention_bias"
        case layerNormBias = "layer_norm_bias"
        case slidingWindow = "sliding_window"
        case slidingWindowPattern = "sliding_window_pattern"
        case numExperts = "num_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case normTopkProb = "norm_topk_prob"
        case numSharedExperts = "num_shared_experts"
        case moeNumSharedExperts = "moe_num_shared_experts"
        case moeGateActivation = "moe_gate_act"
        case expertSelectionFunction = "expert_selection_fn"
        case sharedExpertCombinationStrategy = "shared_expert_combination_strategy"
        case rmsNormEps = "rms_norm_eps"
        case firstKDenseReplace = "first_k_dense_replace"
        case prefixDenseIntermediateSize = "prefix_dense_intermediate_size"
        case prefixDenseSlidingWindowPattern = "prefix_dense_sliding_window_pattern"
        case layerTypes = "layer_types"
        case bosTokenID = "bos_token_id"
        case eosTokenID = "eos_token_id"
        case padTokenID = "pad_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "cohere2_moe"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 36
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 1024
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 64
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 50000
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 256_000
        layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
        logitScale = try c.decodeIfPresent(Float.self, forKey: .logitScale) ?? 0.0625
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        layerNormBias = try c.decodeIfPresent(Bool.self, forKey: .layerNormBias) ?? false
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 4096
        slidingWindowPattern = try c.decodeIfPresent(Int.self, forKey: .slidingWindowPattern) ?? 4
        numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 128
        numExpertsPerToken = try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 8
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        let sharedExperts = try c.decodeIfPresent(Int.self, forKey: .numSharedExperts)
        let configuredSharedExperts = try c.decodeIfPresent(
            Int.self,
            forKey: .moeNumSharedExperts,
        )
        moeNumSharedExperts = sharedExperts ?? configuredSharedExperts ?? 4
        let expertSelectionFunction = try c.decodeIfPresent(
            String.self, forKey: .expertSelectionFunction,
        )
        let configuredGateActivation = try c.decodeIfPresent(
            String.self,
            forKey: .moeGateActivation,
        )
        moeGateActivation = expertSelectionFunction ?? configuredGateActivation ?? "sigmoid"
        sharedExpertCombinationStrategy =
            try c.decodeIfPresent(String.self, forKey: .sharedExpertCombinationStrategy)
            ?? "average"
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps)
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        prefixDenseIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .prefixDenseIntermediateSize)
            ?? intermediateSize
        prefixDenseSlidingWindowPattern =
            try c.decodeIfPresent(Int.self, forKey: .prefixDenseSlidingWindowPattern) ?? 1
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
        bosTokenID = try c.decodeIfPresent(Int.self, forKey: .bosTokenID)
        eosTokenID = try c.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenID)
        padTokenID = try c.decodeIfPresent(Int.self, forKey: .padTokenID)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encode(vocabularySize, forKey: .vocabularySize)
        try c.encode(layerNormEps, forKey: .layerNormEps)
        try c.encode(logitScale, forKey: .logitScale)
        try c.encode(attentionBias, forKey: .attentionBias)
        try c.encode(layerNormBias, forKey: .layerNormBias)
        try c.encode(slidingWindow, forKey: .slidingWindow)
        try c.encode(slidingWindowPattern, forKey: .slidingWindowPattern)
        try c.encode(numExperts, forKey: .numExperts)
        try c.encode(numExpertsPerToken, forKey: .numExpertsPerToken)
        try c.encode(normTopkProb, forKey: .normTopkProb)
        try c.encode(moeNumSharedExperts, forKey: .moeNumSharedExperts)
        try c.encode(moeGateActivation, forKey: .moeGateActivation)
        try c.encode(sharedExpertCombinationStrategy, forKey: .sharedExpertCombinationStrategy)
        try c.encodeIfPresent(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(firstKDenseReplace, forKey: .firstKDenseReplace)
        try c.encode(prefixDenseIntermediateSize, forKey: .prefixDenseIntermediateSize)
        try c.encode(prefixDenseSlidingWindowPattern, forKey: .prefixDenseSlidingWindowPattern)
        try c.encodeIfPresent(layerTypes, forKey: .layerTypes)
        try c.encodeIfPresent(bosTokenID, forKey: .bosTokenID)
        try c.encodeIfPresent(eosTokenID, forKey: .eosTokenID)
        try c.encodeIfPresent(padTokenID, forKey: .padTokenID)
    }

    func isPrefixDenseLayer(_ layerIndex: Int) -> Bool {
        layerIndex < firstKDenseReplace
    }

    func isSlidingLayer(_ layerIndex: Int) -> Bool {
        if isPrefixDenseLayer(layerIndex) {
            return false
        }
        if let layerTypes {
            return layerTypes[layerIndex] == "sliding_attention"
        }
        return (layerIndex + 1) % slidingWindowPattern != 0
    }
}

private func cohere2MoEMakeNorm(_ config: Cohere2MoEConfiguration) -> Module & UnaryLayer {
    if let eps = config.rmsNormEps {
        return RMSNorm(dimensions: config.hiddenSize, eps: eps)
    }
    return LayerNorm(
        dimensions: config.hiddenSize,
        eps: config.layerNormEps,
        affine: true,
        bias: config.layerNormBias,
    )
}

private final class Cohere2MoEAttention: Module {
    let config: Cohere2MoEConfiguration
    let layerIndex: Int
    let scale: Float
    let useRotary: Bool

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    let rope: RoPE

    init(_ config: Cohere2MoEConfiguration, layerIndex: Int) {
        self.config = config
        self.layerIndex = layerIndex
        scale = pow(Float(config.headDim), -0.5)
        useRotary =
            config.isSlidingLayer(layerIndex)
            || (config.isPrefixDenseLayer(layerIndex)
                && config.prefixDenseSlidingWindowPattern == 1)

        _wq.wrappedValue = Linear(
            config.hiddenSize,
            config.attentionHeads * config.headDim,
            bias: config.attentionBias,
        )
        _wk.wrappedValue = Linear(
            config.hiddenSize,
            config.kvHeads * config.headDim,
            bias: config.attentionBias,
        )
        _wv.wrappedValue = Linear(
            config.hiddenSize,
            config.kvHeads * config.headDim,
            bias: config.attentionBias,
        )
        _wo.wrappedValue = Linear(
            config.attentionHeads * config.headDim,
            config.hiddenSize,
            bias: config.attentionBias,
        )
        rope = RoPE(dimensions: config.headDim, traditional: true, base: config.ropeTheta)
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
        var queries = wq(x)
        var keys = wk(x)
        var values = wv(x)

        queries = queries.reshaped(batchSize, sequenceLength, config.attentionHeads, -1)
            .transposed(0, 2, 1, 3)
        keys = keys.reshaped(batchSize, sequenceLength, config.kvHeads, -1)
            .transposed(0, 2, 1, 3)
        values = values.reshaped(batchSize, sequenceLength, config.kvHeads, -1)
            .transposed(0, 2, 1, 3)

        if useRotary {
            queries = applyRotaryPosition(rope, to: queries, offset: cache?.ropeOffset)
            keys = applyRotaryPosition(rope, to: keys, offset: cache?.ropeOffset)
        }

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask,
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, sequenceLength, -1)

        return wo(output)
    }
}

private final class Cohere2MoEMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

private final class Cohere2MoESparseBlock: Module, UnaryLayer {
    let config: Cohere2MoEConfiguration

    @ModuleInfo var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: Cohere2MoEMLP?

    init(_ config: Cohere2MoEConfiguration) {
        self.config = config
        _gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.intermediateSize,
            numExperts: config.numExperts,
        )
        if config.moeNumSharedExperts > 0 {
            _sharedExperts.wrappedValue = Cohere2MoEMLP(
                dimensions: config.hiddenSize,
                hiddenDimensions: config.intermediateSize * config.moeNumSharedExperts,
            )
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let logits = gate(x).asType(.float32)
        let gates: MLXArray
        switch config.moeGateActivation {
        case "softmax":
            gates = softmax(logits, axis: -1, precise: true)
        case "sigmoid":
            gates = sigmoid(logits)
        default:
            preconditionFailure(
                "Unsupported Cohere2 MoE gate activation: \(config.moeGateActivation)")
        }

        let indices = argPartition(-gates, kth: config.numExpertsPerToken - 1, axis: -1)[
            .ellipsis, ..<config.numExpertsPerToken,
        ]
        var scores = takeAlong(gates, indices, axis: -1)
        if config.normTopkProb {
            scores = scores / maximum(scores.sum(axis: -1, keepDims: true), 1e-12)
        }

        var output =
            (switchMLP(x, indices)
            * expandedDimensions(scores.asType(x.dtype), axis: -1))
            .sum(axis: -2)
        if let sharedExperts {
            if config.sharedExpertCombinationStrategy == "average" {
                output = (output + sharedExperts(x)) / 2
            } else {
                output = output + sharedExperts(x)
            }
        }
        return output
    }
}

private final class Cohere2MoEDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Cohere2MoEAttention
    @ModuleInfo var mlp: Module & UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Module & UnaryLayer

    init(_ config: Cohere2MoEConfiguration, layerIndex: Int) {
        _attention.wrappedValue = Cohere2MoEAttention(config, layerIndex: layerIndex)
        if config.isPrefixDenseLayer(layerIndex) {
            _mlp.wrappedValue = Cohere2MoEMLP(
                dimensions: config.hiddenSize,
                hiddenDimensions: config.prefixDenseIntermediateSize,
            )
        } else {
            _mlp.wrappedValue = Cohere2MoESparseBlock(config)
        }
        _inputLayerNorm.wrappedValue = cohere2MoEMakeNorm(config)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?,
    )
        -> MLXArray
    {
        let h = inputLayerNorm(x)
        let attnH = attention(h, mask: mask, cache: cache)
        let ffH = mlp(h)
        return attnH + ffH + x
    }
}

private final class Cohere2MoEModelInner: Module {
    let config: Cohere2MoEConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Cohere2MoEDecoderLayer]
    @ModuleInfo var norm: Module & UnaryLayer

    init(_ config: Cohere2MoEConfiguration) {
        self.config = config
        precondition(config.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize,
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            Cohere2MoEDecoderLayer(config, layerIndex: $0)
        }
        _norm.wrappedValue = cohere2MoEMakeNorm(config)
        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        inputEmbeddings: MLXArray? = nil,
    )
        -> MLXArray
    {
        var h = inputEmbeddings ?? embedTokens(inputs)

        for (i, layer) in layers.enumerated() {
            let layerCache = cache?[i]
            let mask = createAttentionMask(
                h: h,
                cache: layerCache,
                windowSize: config.isSlidingLayer(i) ? config.slidingWindow : nil,
            )
            h = layer(h, mask: mask, cache: layerCache)
        }

        return norm(h)
    }
}

private final class Cohere2MoELanguageModel: Module {
    let config: Cohere2MoEConfiguration
    @ModuleInfo var model: Cohere2MoEModelInner

    init(_ config: Cohere2MoEConfiguration) {
        self.config = config
        _model.wrappedValue = Cohere2MoEModelInner(config)
        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        inputEmbeddings: MLXArray? = nil,
    )
        -> MLXArray
    {
        let out = model(inputs, cache: cache, inputEmbeddings: inputEmbeddings)
        return model.embedTokens.asLinear(out) * config.logitScale
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        for layerIndex in 0 ..< config.hiddenLayers where !config.isPrefixDenseLayer(layerIndex) {
            for basePrefix in ["", "language_model."] {
                let layerPrefix = "\(basePrefix)model.layers.\(layerIndex)"
                let expertPrefix = "\(layerPrefix).mlp.experts"
                for name in ["up_proj", "down_proj", "gate_proj"] {
                    for suffix in ["weight", "scales", "biases", "weight_global_scale"] {
                        let firstKey = "\(expertPrefix).0.\(name).\(suffix)"
                        guard sanitized[firstKey] != nil else {
                            continue
                        }
                        let keys = (0 ..< config.numExperts).map { expert in
                            "\(expertPrefix).\(expert).\(name).\(suffix)"
                        }
                        let tensors = keys.compactMap { sanitized[$0] }
                        // Leave incomplete or incompatible groups for strict loading to
                        // reject, rather than trapping while stacking malformed sources.
                        guard tensors.count == config.numExperts,
                            tensors.allSatisfy({
                                $0.shape == tensors[0].shape && $0.dtype == tensors[0].dtype
                            })
                        else { continue }
                        for key in keys { sanitized.removeValue(forKey: key) }
                        sanitized["\(layerPrefix).mlp.switch_mlp.\(name).\(suffix)"] =
                            MLX.stacked(tensors)
                    }
                }
            }
        }

        for key in Array(sanitized.keys) {
            if key.contains("rotary_emb.inv_freq") {
                sanitized.removeValue(forKey: key)
            } else if key.hasSuffix(".bias") {
                if key.contains(".mlp.") {
                    sanitized.removeValue(forKey: key)
                } else if key.contains(".self_attn."), !config.attentionBias {
                    sanitized.removeValue(forKey: key)
                } else if key.localizedCaseInsensitiveContains("layernorm"),
                    !config.layerNormBias
                {
                    sanitized.removeValue(forKey: key)
                }
            }
        }

        return sanitized
    }
}

public final class Cohere2MoE: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let config: Cohere2MoEConfiguration

    @ModuleInfo(key: "language_model") fileprivate var languageModel: Cohere2MoELanguageModel

    public init(_ config: Cohere2MoEConfiguration) {
        self.config = config
        vocabularySize = config.vocabularySize
        kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        _languageModel.wrappedValue = Cohere2MoELanguageModel(config)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try (0 ..< config.hiddenLayers).map { layerIndex in
            if config.isSlidingLayer(layerIndex) {
                return try makeSlidingWindowKVCache(
                    configuration: configuration, window: config.slidingWindow)
            }
            return try makeAttentionKVCache(configuration: configuration)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        languageModel.sanitize(weights: weights).reduce(into: [:]) { result, item in
            let key = item.key
            if key.hasPrefix("language_model.") {
                result[key] = item.value
            } else if key.hasPrefix("model.") {
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

extension Cohere2MoE: SeparateExpertWeightsProviding {
    func separateExpertWeightNames(for path: String, expert: Int) -> [String] {
        guard path.hasPrefix("language_model."), path.contains(".mlp.switch_mlp."),
            let suffix = path.split(separator: ".").last,
            ["weight", "scales", "biases", "weight_global_scale"].contains(String(suffix))
        else { return [] }
        let key = path.replacingOccurrences(of: ".switch_mlp.", with: ".experts.\(expert).")
        return [key, String(key.dropFirst("language_model.".count))]
    }
}
