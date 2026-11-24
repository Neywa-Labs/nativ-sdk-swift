import Foundation
import MLX
import MLXNN

public struct Mistral4Configuration: Codable, Sendable {
    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let maxPositionEmbeddings: Int
    public let tieWordEmbeddings: Bool
    public let attentionBias: Bool
    public let qLoraRank: Int?
    public let kvLoraRank: Int
    public let qkRopeHeadDim: Int
    public let qkNopeHeadDim: Int
    public let vHeadDim: Int
    public let ropeTraditional: Bool
    public let ropeParameters: [String: StringOrNumber]
    public let nRoutedExperts: Int?
    public let nSharedExperts: Int?
    public let numExpertsPerToken: Int
    public let moeIntermediateSize: Int
    public let normTopkProb: Bool
    public let routedScalingFactor: Float
    public let firstKDenseReplace: Int
    public let eosTokenID: IntOrIntArray?
    public let bosTokenID: Int?
    public let padTokenID: Int?

    public var qHeadDim: Int {
        qkNopeHeadDim + qkRopeHeadDim
    }

    public var ropeTheta: Float {
        ropeParameters["rope_theta"]?.asFloat() ?? 10000
    }

    public var llama4ScalingBeta: Float {
        ropeParameters["llama_4_scaling_beta"]?.asFloat() ?? 0
    }

    public var originalMaxPositionEmbeddings: Int {
        ropeParameters["original_max_position_embeddings"]?.asInt() ?? maxPositionEmbeddings
    }

    enum CodingKeys: String, CodingKey {
        case textConfig = "text_config"
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case qLoraRank = "q_lora_rank"
        case kvLoraRank = "kv_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case vHeadDim = "v_head_dim"
        case ropeTraditional = "rope_traditional"
        case ropeInterleave = "rope_interleave"
        case ropeParameters = "rope_parameters"
        case ropeScaling = "rope_scaling"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case routedScalingFactor = "routed_scaling_factor"
        case firstKDenseReplace = "first_k_dense_replace"
        case eosTokenID = "eos_token_id"
        case bosTokenID = "bos_token_id"
        case padTokenID = "pad_token_id"
    }

    public init(
        modelType: String = "mistral4",
        hiddenSize: Int = 4096,
        hiddenLayers: Int = 36,
        intermediateSize: Int = 12288,
        attentionHeads: Int = 32,
        kvHeads: Int? = nil,
        rmsNormEps: Float = 1e-6,
        vocabularySize: Int = 131_072,
        maxPositionEmbeddings: Int = 1_048_576,
        tieWordEmbeddings: Bool = false,
        attentionBias: Bool = false,
        qLoraRank: Int? = 1024,
        kvLoraRank: Int = 256,
        qkRopeHeadDim: Int = 64,
        qkNopeHeadDim: Int = 64,
        vHeadDim: Int = 128,
        ropeTraditional: Bool = true,
        ropeParameters: [String: StringOrNumber] = [
            "rope_theta": .float(10000),
            "llama_4_scaling_beta": .float(0.1),
            "original_max_position_embeddings": .int(8192),
        ],
        nRoutedExperts: Int? = 128,
        nSharedExperts: Int? = 1,
        numExpertsPerToken: Int = 4,
        moeIntermediateSize: Int = 2048,
        normTopkProb: Bool = true,
        routedScalingFactor: Float = 1,
        firstKDenseReplace: Int = 0,
        eosTokenID: IntOrIntArray? = nil,
        bosTokenID: Int? = nil,
        padTokenID: Int? = nil,
    ) {
        self.modelType = modelType
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads ?? attentionHeads
        self.rmsNormEps = rmsNormEps
        self.vocabularySize = vocabularySize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.tieWordEmbeddings = tieWordEmbeddings
        self.attentionBias = attentionBias
        self.qLoraRank = qLoraRank
        self.kvLoraRank = kvLoraRank
        self.qkRopeHeadDim = qkRopeHeadDim
        self.qkNopeHeadDim = qkNopeHeadDim
        self.vHeadDim = vHeadDim
        self.ropeTraditional = ropeTraditional
        self.ropeParameters = ropeParameters
        self.nRoutedExperts = nRoutedExperts
        self.nSharedExperts = nSharedExperts
        self.numExpertsPerToken = numExpertsPerToken
        self.moeIntermediateSize = moeIntermediateSize
        self.normTopkProb = normTopkProb
        self.routedScalingFactor = routedScalingFactor
        self.firstKDenseReplace = firstKDenseReplace
        self.eosTokenID = eosTokenID
        self.bosTokenID = bosTokenID
        self.padTokenID = padTokenID
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let nested = try c.decodeIfPresent(Mistral4Configuration.self, forKey: .textConfig) {
            self = nested
            return
        }

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "mistral4"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 36
        intermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 12288
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        vocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 131_072
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 1_048_576
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
        kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? 256
        qkRopeHeadDim =
            try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
        qkNopeHeadDim =
            try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 64
        vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 128
        let ropeInterleave = try c.decodeIfPresent(Bool.self, forKey: .ropeInterleave) ?? false
        let decodedTraditional =
            try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        ropeTraditional = ropeInterleave ? true : decodedTraditional
        ropeParameters =
            try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
            ?? c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
            ?? ["rope_theta": .float(10000)]
        nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
        nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts)
        numExpertsPerToken =
            try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 1
        moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? intermediateSize
        normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1
        firstKDenseReplace =
            try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        eosTokenID = try c.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenID)
        bosTokenID = try c.decodeIfPresent(Int.self, forKey: .bosTokenID)
        padTokenID = try c.decodeIfPresent(Int.self, forKey: .padTokenID)
    }

    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(modelType, forKey: .modelType)
        try c.encode(hiddenSize, forKey: .hiddenSize)
        try c.encode(hiddenLayers, forKey: .hiddenLayers)
        try c.encode(intermediateSize, forKey: .intermediateSize)
        try c.encode(attentionHeads, forKey: .attentionHeads)
        try c.encode(kvHeads, forKey: .kvHeads)
        try c.encode(rmsNormEps, forKey: .rmsNormEps)
        try c.encode(vocabularySize, forKey: .vocabularySize)
        try c.encode(maxPositionEmbeddings, forKey: .maxPositionEmbeddings)
        try c.encode(tieWordEmbeddings, forKey: .tieWordEmbeddings)
        try c.encode(attentionBias, forKey: .attentionBias)
        try c.encodeIfPresent(qLoraRank, forKey: .qLoraRank)
        try c.encode(kvLoraRank, forKey: .kvLoraRank)
        try c.encode(qkRopeHeadDim, forKey: .qkRopeHeadDim)
        try c.encode(qkNopeHeadDim, forKey: .qkNopeHeadDim)
        try c.encode(vHeadDim, forKey: .vHeadDim)
        try c.encode(ropeTraditional, forKey: .ropeTraditional)
        try c.encode(ropeParameters, forKey: .ropeParameters)
        try c.encodeIfPresent(nRoutedExperts, forKey: .nRoutedExperts)
        try c.encodeIfPresent(nSharedExperts, forKey: .nSharedExperts)
        try c.encode(numExpertsPerToken, forKey: .numExpertsPerToken)
        try c.encode(moeIntermediateSize, forKey: .moeIntermediateSize)
        try c.encode(normTopkProb, forKey: .normTopkProb)
        try c.encode(routedScalingFactor, forKey: .routedScalingFactor)
        try c.encode(firstKDenseReplace, forKey: .firstKDenseReplace)
        try c.encodeIfPresent(eosTokenID, forKey: .eosTokenID)
        try c.encodeIfPresent(bosTokenID, forKey: .bosTokenID)
        try c.encodeIfPresent(padTokenID, forKey: .padTokenID)
    }
}

private enum Mistral4Language {
    static func attentionScale(start: Int, stop: Int, beta: Float, maxPositionEmbeddings: Int)
        -> MLXArray
    {
        let positions = MLXArray(start ..< stop).asType(.float32)
        let scaling = 1 + beta * log(1 + floor(positions / Float(maxPositionEmbeddings)))
        return expandedDimensions(scaling, axis: -1)
    }

    final class Attention: Module {
        let config: Mistral4Configuration
        let numHeads: Int
        let qLoraRank: Int?
        let qkRopeHeadDim: Int
        let kvLoraRank: Int
        let vHeadDim: Int
        let qkNopeHeadDim: Int
        let qHeadDim: Int
        let scale: Float
        let rope: RoPELayer

        @ModuleInfo(key: "q_proj") var qProj: Linear?
        @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
        @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
        @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
        @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMQA: Linear
        @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
        @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        init(_ config: Mistral4Configuration) {
            self.config = config
            numHeads = config.attentionHeads
            qLoraRank = config.qLoraRank
            qkRopeHeadDim = config.qkRopeHeadDim
            kvLoraRank = config.kvLoraRank
            vHeadDim = config.vHeadDim
            qkNopeHeadDim = config.qkNopeHeadDim
            qHeadDim = config.qHeadDim
            scale = pow(Float(config.qHeadDim), -0.5)

            if let qLoraRank = config.qLoraRank {
                _qAProj.wrappedValue = Linear(
                    config.hiddenSize,
                    qLoraRank,
                    bias: config.attentionBias,
                )
                _qALayerNorm.wrappedValue = RMSNorm(
                    dimensions: qLoraRank,
                    eps: config.rmsNormEps,
                )
                _qBProj.wrappedValue = Linear(
                    qLoraRank,
                    config.attentionHeads * config.qHeadDim,
                    bias: false,
                )
            } else {
                _qProj.wrappedValue = Linear(
                    config.hiddenSize,
                    config.attentionHeads * config.qHeadDim,
                    bias: false,
                )
            }

            _kvAProjWithMQA.wrappedValue = Linear(
                config.hiddenSize,
                config.kvLoraRank + config.qkRopeHeadDim,
                bias: config.attentionBias,
            )
            _kvALayerNorm.wrappedValue = RMSNorm(
                dimensions: config.kvLoraRank,
                eps: config.rmsNormEps,
            )
            _kvBProj.wrappedValue = Linear(
                config.kvLoraRank,
                config.attentionHeads * (config.qkNopeHeadDim + config.vHeadDim),
                bias: false,
            )
            _oProj.wrappedValue = Linear(
                config.attentionHeads * config.vHeadDim,
                config.hiddenSize,
                bias: config.attentionBias,
            )

            rope = initializeRope(
                dims: config.qkRopeHeadDim,
                base: config.ropeTheta,
                traditional: config.ropeTraditional,
                scalingConfig: config.ropeParameters,
                maxPositionEmbeddings: config.maxPositionEmbeddings,
            )
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionScale: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let (batchSize, sequenceLength) = (x.dim(0), x.dim(1))

            var queries: MLXArray =
                if qLoraRank == nil {
                    qProj!(x)
                } else {
                    qBProj!(qALayerNorm!(qAProj!(x)))
                }

            queries = queries.reshaped(batchSize, sequenceLength, numHeads, qHeadDim)
                .transposed(0, 2, 1, 3)
            let qSplit = split(queries, indices: [qkNopeHeadDim], axis: -1)
            let qNope = qSplit[0]
            var qPe = qSplit[1]

            var compressedKV = kvAProjWithMQA(x)
            let compressedSplit = split(compressedKV, indices: [kvLoraRank], axis: -1)
            compressedKV = compressedSplit[0]
            var kPe = compressedSplit[1]
            kPe = kPe.reshaped(batchSize, sequenceLength, 1, qkRopeHeadDim)
                .transposed(0, 2, 1, 3)

            var kv = kvBProj(kvALayerNorm(compressedKV))
            kv = kv.reshaped(batchSize, sequenceLength, numHeads, -1)
                .transposed(0, 2, 1, 3)
            let kvSplit = split(kv, indices: [qkNopeHeadDim], axis: -1)
            let kNope = kvSplit[0]
            let values = kvSplit[1]

            qPe = applyRotaryPosition(rope, to: qPe, offset: cache?.ropeOffset)
            kPe = applyRotaryPosition(rope, to: kPe, offset: cache?.ropeOffset)
            kPe = repeated(kPe, count: numHeads, axis: 1)

            queries = concatenated([qNope, qPe], axis: -1) * attentionScale
            let keys = concatenated([kNope, kPe], axis: -1)
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
            return oProj(output)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        init(_ config: Mistral4Configuration, intermediateSize: Int? = nil) {
            let hidden = intermediateSize ?? config.intermediateSize
            _gateProj.wrappedValue = Linear(config.hiddenSize, hidden, bias: false)
            _upProj.wrappedValue = Linear(config.hiddenSize, hidden, bias: false)
            _downProj.wrappedValue = Linear(hidden, config.hiddenSize, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    final class MoE: Module, UnaryLayer {
        let config: Mistral4Configuration
        @ModuleInfo var gate: Linear
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
        @ModuleInfo(key: "shared_experts") var sharedExperts: MLP?

        init(_ config: Mistral4Configuration) {
            self.config = config
            _gate.wrappedValue = Linear(
                config.hiddenSize,
                config.nRoutedExperts ?? 1,
                bias: false,
            )
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.nRoutedExperts ?? 1,
                activation: silu,
                bias: false,
            )
            if let shared = config.nSharedExperts, shared > 0 {
                _sharedExperts.wrappedValue = MLP(
                    config,
                    intermediateSize: config.moeIntermediateSize * shared,
                )
            }
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let gates = softmax(gate(x).asType(.float32), axis: -1, precise: true)
            let topK = config.numExpertsPerToken
            let indices = argPartition(-gates, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
            var scores = takeAlong(gates, indices, axis: -1)
            if config.normTopkProb {
                scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-20)
            }
            scores = scores * config.routedScalingFactor

            var output = switchMLP(x, indices)
            output = (output * scores.asType(x.dtype)[.ellipsis, .newAxis]).sum(axis: -2)
            if let sharedExperts {
                output = output + sharedExperts(x)
            }
            return output
        }
    }

    final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo var mlp: Module & UnaryLayer
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: Mistral4Configuration, layerIndex: Int) {
            _attention.wrappedValue = Attention(config)
            if layerIndex >= config.firstKDenseReplace, config.nRoutedExperts != nil {
                _mlp.wrappedValue = MoE(config)
            } else {
                _mlp.wrappedValue = MLP(config)
            }
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionScale: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let attentionOutput = attention(
                inputLayerNorm(x),
                attentionScale: attentionScale,
                mask: mask,
                cache: cache,
            )
            let hidden = x + attentionOutput
            return hidden + mlp(postAttentionLayerNorm(hidden))
        }
    }

    final class Model: Module {
        let config: Mistral4Configuration
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo var layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm

        init(_ config: Mistral4Configuration) {
            self.config = config
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
                DecoderLayer(config, layerIndex: $0)
            }
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray? = nil,
        )
            -> MLXArray
        {
            var hidden: MLXArray
            if let inputEmbeddings {
                hidden = inputEmbeddings
            } else if let inputs {
                hidden = embedTokens(inputs)
            } else {
                preconditionFailure("inputs or inputEmbeddings must be provided")
            }

            let offset = cache?.first?.offset ?? 0
            let mask = createAttentionMask(h: hidden, cache: cache?.first)
            let scale = attentionScale(
                start: offset,
                stop: offset + hidden.dim(1),
                beta: config.llama4ScalingBeta,
                maxPositionEmbeddings: config.originalMaxPositionEmbeddings,
            )
            .asType(hidden.dtype)

            for (index, layer) in layers.enumerated() {
                hidden = layer(hidden, attentionScale: scale, mask: mask, cache: cache?[index])
            }

            return norm(hidden)
        }
    }

    final class LanguageModel: Module {
        let config: Mistral4Configuration
        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        init(_ config: Mistral4Configuration) {
            self.config = config
            _model.wrappedValue = Model(config)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.hiddenSize,
                    config.vocabularySize,
                    bias: false,
                )
            }
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray? = nil,
        )
            -> MLXArray
        {
            let hidden = model(inputs, cache: cache, inputEmbeddings: inputEmbeddings)
            if config.tieWordEmbeddings {
                return model.embedTokens.asLinear(hidden)
            }
            return lmHead!(hidden)
        }
    }
}

public final class Mistral4: Module, LLMModel, KVCacheDimensionProvider {
    public var toolCallFormat: ToolCallFormat? { .mistral }

    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let config: Mistral4Configuration

    @ModuleInfo(key: "language_model") private var languageModel: Mistral4Language.LanguageModel

    public init(_ config: Mistral4Configuration) {
        self.config = config
        vocabularySize = config.vocabularySize
        kvHeads = Array(repeating: config.attentionHeads, count: config.hiddenLayers)
        _languageModel.wrappedValue = Mistral4Language.LanguageModel(config)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var normalized: [String: MLXArray] = [:]

        for (key, value) in weights {
            if key.contains("rotary_emb.inv_freq") || key.contains("position_ids") {
                continue
            }

            var newKey = key
            if newKey.hasPrefix("model.") {
                newKey = "language_model." + newKey
            } else if newKey.hasPrefix("lm_head.") {
                newKey = "language_model." + newKey
            }

            guard newKey.hasPrefix("language_model.") else {
                continue
            }
            if config.tieWordEmbeddings, newKey == "language_model.lm_head.weight" {
                continue
            }

            normalized[newKey] = value
        }

        var sanitized: [String: MLXArray] = [:]
        for (key, value) in normalized {
            if key.contains("activation_scale") || key.contains("_scale_inv") {
                continue
            }
            var value = value
            if let scale = normalized["\(key)_scale_inv"] ?? normalized[scaleKey(for: key)] {
                value = value * scale
            }
            sanitized[key] = value
        }

        splitFusedExperts(weights: &sanitized)
        return sanitized
    }

    private func scaleKey(for key: String) -> String {
        if key.hasSuffix(".weight") {
            String(key.dropLast(".weight".count)) + ".weight_scale_inv"
        } else {
            "\(key)_scale_inv"
        }
    }

    private func splitFusedExperts(weights: inout [String: MLXArray]) {
        guard let routedExperts = config.nRoutedExperts, routedExperts > 0 else {
            return
        }

        for layer in 0 ..< config.hiddenLayers {
            let prefix = "language_model.model.layers.\(layer).mlp"
            for (sourceSuffix, destinationSuffix) in [
                ("", ".weight"),
                (".weight", ".weight"),
                (".scales", ".scales"),
                (".biases", ".biases"),
            ] {
                let fusedKey = "\(prefix).experts.gate_up_proj\(sourceSuffix)"
                if let fused = weights.removeValue(forKey: fusedKey) {
                    let parts = split(fused, parts: 2, axis: 1)
                    weights["\(prefix).switch_mlp.gate_proj\(destinationSuffix)"] = parts[0]
                    weights["\(prefix).switch_mlp.up_proj\(destinationSuffix)"] = parts[1]
                }

                let downKey = "\(prefix).experts.down_proj\(sourceSuffix)"
                if let down = weights.removeValue(forKey: downKey) {
                    weights["\(prefix).switch_mlp.down_proj\(destinationSuffix)"] = down
                }
            }
        }
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

struct Mistral4RoutingProbe: Decodable {
    struct TextConfig: Decodable {
        let modelType: String?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
        }
    }

    let modelType: String?
    let textConfig: TextConfig?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
    }
}
