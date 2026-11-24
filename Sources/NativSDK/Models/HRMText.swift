import Foundation
import MLX
import MLXNN

public struct HRMTextConfiguration: Codable, Sendable {
    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let layersPerStack: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let maxPositionEmbeddings: Int
    public let initializerRange: Float
    public let rmsNormEps: Float
    public let tieWordEmbeddings: Bool
    public let ropeTheta: Float
    public let attentionBias: Bool
    public let mlpBias: Bool
    public let highCycles: Int
    public let lowCycles: Int
    public let lowBackpropCycles: [Int]
    public let embeddingScale: Float
    public let prefixLM: Bool
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
        case initializerRange = "initializer_range"
        case rmsNormEps = "rms_norm_eps"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeTheta = "rope_theta"
        case ropeParameters = "rope_parameters"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case highCycles = "H_cycles"
        case lowCycles = "L_cycles"
        case lowBackpropCycles = "L_bp_cycles"
        case embeddingScale = "embedding_scale"
        case prefixLM = "prefix_lm"
        case layersPerStack = "num_layers_per_stack"
        case bosTokenID = "bos_token_id"
        case eosTokenID = "eos_token_id"
        case padTokenID = "pad_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "hrm_text"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_808
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1536
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4096
        let decodedHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 16
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 12
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? hiddenSize / attentionHeads
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 2048
        initializerRange = try c.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        var decodedRopeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        if let ropeParameters = try c.decodeIfPresent(
            [String: StringOrNumber].self,
            forKey: .ropeParameters,
        ) {
            decodedRopeTheta =
                ropeParameters["rope_theta"]?.asFloat()
                ?? ropeParameters["theta"]?.asFloat()
                ?? decodedRopeTheta
        }
        ropeTheta = decodedRopeTheta

        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        highCycles = try c.decodeIfPresent(Int.self, forKey: .highCycles) ?? 2
        lowCycles = try c.decodeIfPresent(Int.self, forKey: .lowCycles) ?? 3
        lowBackpropCycles =
            try c.decodeIfPresent([Int].self, forKey: .lowBackpropCycles) ?? [2]
        embeddingScale =
            try c.decodeIfPresent(Float.self, forKey: .embeddingScale) ?? 1.0 / initializerRange
        prefixLM = try c.decodeIfPresent(Bool.self, forKey: .prefixLM) ?? true

        if let decodedLayersPerStack = try c.decodeIfPresent(Int.self, forKey: .layersPerStack) {
            layersPerStack = decodedLayersPerStack
            hiddenLayers = decodedHiddenLayers
        } else {
            layersPerStack = decodedHiddenLayers
            hiddenLayers = layersPerStack * highCycles * (lowCycles + 1)
        }

        bosTokenID = try c.decodeIfPresent(Int.self, forKey: .bosTokenID)
        eosTokenID = try c.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenID)
        padTokenID = try c.decodeIfPresent(Int.self, forKey: .padTokenID)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(vocabularySize, forKey: .vocabularySize)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(layersPerStack, forKey: .layersPerStack)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(headDim, forKey: .headDim)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try c.encode(initializerRange, forKey: .initializerRange)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encode(ropeTheta, forKey: .ropeTheta)
        try c.encode(attentionBias, forKey: .attentionBias)
        try c.encode(mlpBias, forKey: .mlpBias)
        try c.encode(highCycles, forKey: .highCycles)
        try c.encode(lowCycles, forKey: .lowCycles)
        try c.encode(lowBackpropCycles, forKey: .lowBackpropCycles)
        try c.encode(embeddingScale, forKey: .embeddingScale)
        try c.encode(prefixLM, forKey: .prefixLM)
        try c.encodeIfPresent(bosTokenID, forKey: .bosTokenID)
        try c.encodeIfPresent(eosTokenID, forKey: .eosTokenID)
        try c.encodeIfPresent(padTokenID, forKey: .padTokenID)
    }
}

private final class HRMTextRMSNormNoScale: Module, UnaryLayer {
    let eps: Float

    init(eps: Float) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

private final class HRMTextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: HRMTextConfiguration) {
        _gateProj.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.mlpBias,
        )
        _upProj.wrappedValue = Linear(
            config.hiddenSize,
            config.intermediateSize,
            bias: config.mlpBias,
        )
        _downProj.wrappedValue = Linear(
            config.intermediateSize,
            config.hiddenSize,
            bias: config.mlpBias,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

private final class HRMTextAttention: Module {
    let config: HRMTextConfiguration
    let scale: Float
    let rope: RoPE

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear

    init(_ config: HRMTextConfiguration) {
        self.config = config
        scale = pow(Float(config.headDim), -0.5)
        _qProj.wrappedValue = Linear(
            config.hiddenSize,
            config.attentionHeads * config.headDim,
            bias: config.attentionBias,
        )
        _kProj.wrappedValue = Linear(
            config.hiddenSize,
            config.kvHeads * config.headDim,
            bias: config.attentionBias,
        )
        _vProj.wrappedValue = Linear(
            config.hiddenSize,
            config.kvHeads * config.headDim,
            bias: config.attentionBias,
        )
        _oProj.wrappedValue = Linear(
            config.attentionHeads * config.headDim,
            config.hiddenSize,
            bias: config.attentionBias,
        )
        _gateProj.wrappedValue = Linear(
            config.hiddenSize,
            config.attentionHeads * config.headDim,
            bias: config.attentionBias,
        )
        rope = RoPE(dimensions: config.headDim, traditional: false, base: config.ropeTheta)
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
        var queries = qProj(x).reshaped(
            batchSize,
            sequenceLength,
            config.attentionHeads,
            config.headDim,
        )
        var keys = kProj(x).reshaped(
            batchSize,
            sequenceLength,
            config.kvHeads,
            config.headDim,
        )
        var values = vProj(x).reshaped(
            batchSize,
            sequenceLength,
            config.kvHeads,
            config.headDim,
        )
        let gate = gateProj(x).reshaped(
            batchSize,
            sequenceLength,
            config.attentionHeads,
            config.headDim,
        )

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        queries = applyRotaryPosition(rope, to: queries, offset: cache?.ropeOffset)
        keys = applyRotaryPosition(rope, to: keys, offset: cache?.ropeOffset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask,
        )
        .transposed(0, 2, 1, 3)

        return oProj((output * sigmoid(gate)).reshaped(batchSize, sequenceLength, -1))
    }
}

private final class HRMTextDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: HRMTextAttention
    @ModuleInfo var mlp: HRMTextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: HRMTextRMSNormNoScale
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: HRMTextRMSNormNoScale

    init(_ config: HRMTextConfiguration) {
        _attention.wrappedValue = HRMTextAttention(config)
        _mlp.wrappedValue = HRMTextMLP(config)
        _inputLayerNorm.wrappedValue = HRMTextRMSNormNoScale(eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = HRMTextRMSNormNoScale(eps: config.rmsNormEps)
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

private final class HRMTextStack: Module {
    @ModuleInfo var layers: [HRMTextDecoderLayer]
    @ModuleInfo(key: "final_norm") var finalNorm: HRMTextRMSNormNoScale

    init(_ config: HRMTextConfiguration) {
        _layers.wrappedValue = (0 ..< config.layersPerStack).map { _ in
            HRMTextDecoderLayer(config)
        }
        _finalNorm.wrappedValue = HRMTextRMSNormNoScale(eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: [KVCache]?,
    )
        -> MLXArray
    {
        var h = x
        for (index, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[index])
        }
        return finalNorm(h)
    }
}

private final class HRMTextModelInner: Module {
    let config: HRMTextConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "L_module") var lModule: HRMTextStack
    @ModuleInfo(key: "H_module") var hModule: HRMTextStack
    @ModuleInfo(key: "z_L_init") var zLInit: MLXArray

    var layers: [HRMTextDecoderLayer] {
        var result: [HRMTextDecoderLayer] = []
        for _ in 0 ..< config.highCycles {
            for _ in 0 ..< config.lowCycles {
                result.append(contentsOf: lModule.layers)
            }
            result.append(contentsOf: hModule.layers)
        }
        return result
    }

    init(_ config: HRMTextConfiguration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize,
        )
        _lModule.wrappedValue = HRMTextStack(config)
        _hModule.wrappedValue = HRMTextStack(config)
        _zLInit.wrappedValue = zeros([config.hiddenSize])
        super.init()
    }

    func callAsFunction(
        _ inputs: MLXArray,
        cache: [KVCache]? = nil,
        inputEmbeddings: MLXArray? = nil,
    )
        -> MLXArray
    {
        var hiddenHigh = (inputEmbeddings ?? embedTokens(inputs)) * config.embeddingScale
        var hiddenLow = broadcast(
            zLInit.asType(hiddenHigh.dtype)[.newAxis, .newAxis, 0...],
            to: hiddenHigh.shape,
        )

        let firstCache = cache?.first
        let mask = createAttentionMask(h: hiddenHigh, cache: firstCache)
        let numLayers = config.layersPerStack

        for highCycle in 0 ..< config.highCycles {
            for lowCycle in 0 ..< config.lowCycles {
                let offset = (highCycle * (config.lowCycles + 1) + lowCycle) * numLayers
                hiddenLow = lModule(
                    hiddenLow + hiddenHigh,
                    mask: mask,
                    cache: stackCache(cache, offset: offset),
                )
            }

            let offset = (highCycle * (config.lowCycles + 1) + config.lowCycles) * numLayers
            hiddenHigh = hModule(
                hiddenHigh + hiddenLow,
                mask: mask,
                cache: stackCache(cache, offset: offset),
            )
        }

        return hiddenHigh
    }

    private func stackCache(_ cache: [KVCache]?, offset: Int) -> [KVCache]? {
        guard let cache else {
            return nil
        }
        precondition(
            offset + config.layersPerStack <= cache.count,
            "HRMText cache count must match expanded recurrent layer count.",
        )
        return Array(cache[offset ..< offset + config.layersPerStack])
    }
}

private final class HRMTextLanguageModel: Module {
    let config: HRMTextConfiguration

    @ModuleInfo var model: HRMTextModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    var layers: [HRMTextDecoderLayer] {
        model.layers
    }

    init(_ config: HRMTextConfiguration) {
        self.config = config
        _model.wrappedValue = HRMTextModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
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
        if config.tieWordEmbeddings {
            return model.embedTokens.asLinear(out)
        }
        return lmHead!(out)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        let qSize = config.attentionHeads * config.headDim
        let kvSize = config.kvHeads * config.headDim

        for item in weights {
            let key = transformKey(item.key)
            let value = item.value

            if key.hasSuffix(".self_attn.gqkv_proj.weight") {
                let prefix = String(key.dropLast("gqkv_proj.weight".count))
                let parts = split(value, indices: [qSize, 2 * qSize, 2 * qSize + kvSize], axis: 0)
                sanitized["\(prefix)gate_proj.weight"] = parts[0]
                sanitized["\(prefix)q_proj.weight"] = parts[1]
                sanitized["\(prefix)k_proj.weight"] = parts[2]
                sanitized["\(prefix)v_proj.weight"] = parts[3]
            } else if key.hasSuffix(".self_attn.gqkv_proj.scales") {
                let prefix = String(key.dropLast("gqkv_proj.scales".count))
                let parts = split(value, indices: [qSize, 2 * qSize, 2 * qSize + kvSize], axis: 0)
                sanitized["\(prefix)gate_proj.scales"] = parts[0]
                sanitized["\(prefix)q_proj.scales"] = parts[1]
                sanitized["\(prefix)k_proj.scales"] = parts[2]
                sanitized["\(prefix)v_proj.scales"] = parts[3]
            } else if key.hasSuffix(".self_attn.gqkv_proj.biases") {
                let prefix = String(key.dropLast("gqkv_proj.biases".count))
                let parts = split(value, indices: [qSize, 2 * qSize, 2 * qSize + kvSize], axis: 0)
                sanitized["\(prefix)gate_proj.biases"] = parts[0]
                sanitized["\(prefix)q_proj.biases"] = parts[1]
                sanitized["\(prefix)k_proj.biases"] = parts[2]
                sanitized["\(prefix)v_proj.biases"] = parts[3]
            } else if key.hasSuffix(".self_attn.gqkv_proj.bias") {
                let prefix = String(key.dropLast("gqkv_proj.bias".count))
                let parts = split(value, indices: [qSize, 2 * qSize, 2 * qSize + kvSize], axis: 0)
                sanitized["\(prefix)gate_proj.bias"] = parts[0]
                sanitized["\(prefix)q_proj.bias"] = parts[1]
                sanitized["\(prefix)k_proj.bias"] = parts[2]
                sanitized["\(prefix)v_proj.bias"] = parts[3]
            } else if key.hasSuffix(".mlp.gate_up_proj.weight") {
                let prefix = String(key.dropLast("gate_up_proj.weight".count))
                let parts = split(value, parts: 2, axis: 0)
                sanitized["\(prefix)gate_proj.weight"] = parts[0]
                sanitized["\(prefix)up_proj.weight"] = parts[1]
            } else if key.hasSuffix(".mlp.gate_up_proj.scales") {
                let prefix = String(key.dropLast("gate_up_proj.scales".count))
                let parts = split(value, parts: 2, axis: 0)
                sanitized["\(prefix)gate_proj.scales"] = parts[0]
                sanitized["\(prefix)up_proj.scales"] = parts[1]
            } else if key.hasSuffix(".mlp.gate_up_proj.biases") {
                let prefix = String(key.dropLast("gate_up_proj.biases".count))
                let parts = split(value, parts: 2, axis: 0)
                sanitized["\(prefix)gate_proj.biases"] = parts[0]
                sanitized["\(prefix)up_proj.biases"] = parts[1]
            } else if key.hasSuffix(".mlp.gate_up_proj.bias") {
                let prefix = String(key.dropLast("gate_up_proj.bias".count))
                let parts = split(value, parts: 2, axis: 0)
                sanitized["\(prefix)gate_proj.bias"] = parts[0]
                sanitized["\(prefix)up_proj.bias"] = parts[1]
            } else if config.tieWordEmbeddings, key == "language_model.lm_head.weight" {
                continue
            } else {
                sanitized[key] = value
            }
        }

        return sanitized
    }

    private func transformKey(_ key: String) -> String {
        var key = key
        if key.hasPrefix("model.") {
            key = key.replacingOccurrences(
                of: "model.",
                with: "language_model.model.",
                options: [],
                range: key.startIndex ..< key.index(key.startIndex, offsetBy: 6),
            )
        } else if key.hasPrefix("lm_head.") {
            key = key.replacingOccurrences(
                of: "lm_head.",
                with: "language_model.lm_head.",
                options: [],
                range: key.startIndex ..< key.index(key.startIndex, offsetBy: 8),
            )
        }
        return key.replacingOccurrences(of: ".attn.", with: ".self_attn.")
    }
}

public final class HRMText: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let config: HRMTextConfiguration

    @ModuleInfo(key: "language_model") fileprivate var languageModel: HRMTextLanguageModel

    public init(_ config: HRMTextConfiguration) {
        self.config = config
        vocabularySize = config.vocabularySize
        kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        _languageModel.wrappedValue = HRMTextLanguageModel(config)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try (0 ..< config.hiddenLayers).map { _ in
            return try makeAttentionKVCache(configuration: configuration)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        languageModel.sanitize(weights: weights)
    }

    public var loraLayers: [Module] {
        languageModel.layers
    }
}
