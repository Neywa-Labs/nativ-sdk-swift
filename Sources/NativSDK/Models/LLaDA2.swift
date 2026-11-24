import Foundation
import MLX
import MLXNN

public struct LLaDA2Configuration: Codable, Sendable {
    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let hiddenActivation: String
    public let useQKVBias: Bool
    public let useQKNorm: Bool
    public let useBias: Bool
    public let rmsNormEps: Float
    public let tieWordEmbeddings: Bool
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Float
    public let ropeScaling: [String: StringOrNumber]?
    public let partialRotaryFactor: Float
    public let rotaryDim: Int
    public let numExperts: Int
    public let numSharedExperts: Int
    public let numExpertsPerToken: Int
    public let nGroup: Int
    public let topKGroup: Int
    public let routedScalingFactor: Float
    public let moeIntermediateSize: Int
    public let firstKDenseReplace: Int
    public let normTopkProb: Bool
    public let scoreFunction: String
    public let moeRouterEnableExpertBias: Bool
    public let padTokenID: Int
    public let eosTokenID: IntOrIntArray
    public let maskTokenID: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case hiddenActivation = "hidden_act"
        case useQKVBias = "use_qkv_bias"
        case useQKNorm = "use_qk_norm"
        case useBias = "use_bias"
        case rmsNormEps = "rms_norm_eps"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case partialRotaryFactor = "partial_rotary_factor"
        case rotaryDim = "rotary_dim"
        case numExperts = "num_experts"
        case numSharedExperts = "num_shared_experts"
        case numExpertsPerToken = "num_experts_per_tok"
        case nGroup = "n_group"
        case topKGroup = "topk_group"
        case routedScalingFactor = "routed_scaling_factor"
        case moeIntermediateSize = "moe_intermediate_size"
        case firstKDenseReplace = "first_k_dense_replace"
        case normTopkProb = "norm_topk_prob"
        case scoreFunction = "score_function"
        case moeRouterEnableExpertBias = "moe_router_enable_expert_bias"
        case padTokenID = "pad_token_id"
        case eosTokenID = "eos_token_id"
        case maskTokenID = "mask_token_id"
    }

    public init(
        modelType: String = "llada2_moe",
        vocabularySize: Int = 157_184,
        hiddenSize: Int = 2048,
        intermediateSize: Int = 5120,
        hiddenLayers: Int = 20,
        attentionHeads: Int = 16,
        kvHeads: Int = 4,
        headDim: Int? = nil,
        hiddenActivation: String = "silu",
        useQKVBias: Bool = false,
        useQKNorm: Bool = true,
        useBias: Bool = false,
        rmsNormEps: Float = 1e-6,
        tieWordEmbeddings: Bool = false,
        maxPositionEmbeddings: Int = 32768,
        ropeTheta: Float = 600_000,
        ropeScaling: [String: StringOrNumber]? = nil,
        partialRotaryFactor: Float = 0.5,
        rotaryDim: Int? = nil,
        numExperts: Int = 256,
        numSharedExperts: Int = 1,
        numExpertsPerToken: Int = 8,
        nGroup: Int = 8,
        topKGroup: Int = 4,
        routedScalingFactor: Float = 2.5,
        moeIntermediateSize: Int = 512,
        firstKDenseReplace: Int = 1,
        normTopkProb: Bool = true,
        scoreFunction: String = "sigmoid",
        moeRouterEnableExpertBias: Bool = true,
        padTokenID: Int = 156_892,
        eosTokenID: IntOrIntArray? = nil,
        maskTokenID: Int = 156_895,
    ) {
        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        let resolvedHeadDim = headDim ?? hiddenSize / attentionHeads
        self.headDim = resolvedHeadDim
        self.hiddenActivation = hiddenActivation
        self.useQKVBias = useQKVBias
        self.useQKNorm = useQKNorm
        self.useBias = useBias
        self.rmsNormEps = rmsNormEps
        self.tieWordEmbeddings = tieWordEmbeddings
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.ropeScaling = ropeScaling
        self.partialRotaryFactor = partialRotaryFactor
        self.rotaryDim =
            rotaryDim ?? Int(Float(resolvedHeadDim) * partialRotaryFactor)
        self.numExperts = numExperts
        self.numSharedExperts = numSharedExperts
        self.numExpertsPerToken = numExpertsPerToken
        self.nGroup = nGroup
        self.topKGroup = topKGroup
        self.routedScalingFactor = routedScalingFactor
        self.moeIntermediateSize = moeIntermediateSize
        self.firstKDenseReplace = firstKDenseReplace
        self.normTopkProb = normTopkProb
        self.scoreFunction = scoreFunction
        self.moeRouterEnableExpertBias = moeRouterEnableExpertBias
        self.padTokenID = padTokenID
        self.eosTokenID = eosTokenID ?? IntOrIntArray(padTokenID)
        self.maskTokenID = maskTokenID
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "llada2_moe"
        let vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 157_184
        let hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        let intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 5120
        let hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 20
        let attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        let kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        let headDim =
            try c.decodeIfPresent(Int.self, forKey: .headDim)
            ?? hiddenSize / attentionHeads
        let partialRotaryFactor =
            try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.5
        try self.init(
            modelType: modelType,
            vocabularySize: vocabularySize,
            hiddenSize: hiddenSize,
            intermediateSize: intermediateSize,
            hiddenLayers: hiddenLayers,
            attentionHeads: attentionHeads,
            kvHeads: kvHeads,
            headDim: headDim,
            hiddenActivation:
                c.decodeIfPresent(String.self, forKey: .hiddenActivation) ?? "silu",
            useQKVBias: c.decodeIfPresent(Bool.self, forKey: .useQKVBias) ?? false,
            useQKNorm: c.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? true,
            useBias: c.decodeIfPresent(Bool.self, forKey: .useBias) ?? false,
            rmsNormEps: c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6,
            tieWordEmbeddings:
                c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false,
            maxPositionEmbeddings:
                c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768,
            ropeTheta: c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 600_000,
            ropeScaling:
                c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling),
            partialRotaryFactor: partialRotaryFactor,
            rotaryDim: c.decodeIfPresent(Int.self, forKey: .rotaryDim),
            numExperts: c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 256,
            numSharedExperts:
                c.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 1,
            numExpertsPerToken:
                c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 8,
            nGroup: c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 8,
            topKGroup: c.decodeIfPresent(Int.self, forKey: .topKGroup) ?? 4,
            routedScalingFactor:
                c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 2.5,
            moeIntermediateSize:
                c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 512,
            firstKDenseReplace:
                c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 1,
            normTopkProb: c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true,
            scoreFunction:
                c.decodeIfPresent(String.self, forKey: .scoreFunction) ?? "sigmoid",
            moeRouterEnableExpertBias:
                c.decodeIfPresent(Bool.self, forKey: .moeRouterEnableExpertBias) ?? true,
            padTokenID: c.decodeIfPresent(Int.self, forKey: .padTokenID) ?? 156_892,
            eosTokenID: c.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenID),
            maskTokenID: c.decodeIfPresent(Int.self, forKey: .maskTokenID) ?? 156_895,
        )
    }

    func usesSparseMLP(_ layerIndex: Int) -> Bool {
        numExperts > 0 && layerIndex >= firstKDenseReplace
    }
}

enum LLaDA2Language {
    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        init(_ config: LLaDA2Configuration, intermediateSize: Int) {
            _gateProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
            _upProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
            _downProj.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    final class MoEGate: Module {
        let config: LLaDA2Configuration

        @ModuleInfo var weight: MLXArray
        @ModuleInfo(key: "expert_bias") var expertBias: MLXArray?

        init(_ config: LLaDA2Configuration) {
            self.config = config
            _weight.wrappedValue = zeros([config.numExperts, config.hiddenSize])
            if config.moeRouterEnableExpertBias {
                _expertBias.wrappedValue = zeros([config.numExperts])
            }
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
            let logits = x.asType(.float32).matmul(weight.T.asType(.float32))
            let scores: MLXArray
            switch config.scoreFunction {
            case "sigmoid":
                scores = sigmoid(logits)
            case "softmax":
                scores = softmax(logits, axis: -1, precise: true)
            default:
                preconditionFailure(
                    "Unsupported LLaDA2 MoE score function: \(config.scoreFunction)")
            }

            var choiceScores = scores
            if let expertBias {
                choiceScores = choiceScores + expertBias.asType(choiceScores.dtype)
            }
            choiceScores = groupLimitedScores(choiceScores)

            let topK = min(config.numExpertsPerToken, config.numExperts)
            let indices = argPartition(-choiceScores, kth: topK - 1, axis: -1)[
                .ellipsis, ..<topK,
            ]
            var topWeights = takeAlong(scores, indices, axis: -1)
            if topK > 1, config.normTopkProb {
                topWeights = topWeights / (topWeights.sum(axis: -1, keepDims: true) + 1e-20)
            }
            topWeights = topWeights * config.routedScalingFactor
            return (indices.asType(.int32), topWeights.asType(x.dtype))
        }

        private func groupLimitedScores(_ scores: MLXArray) -> MLXArray {
            guard config.nGroup > 1, config.topKGroup < config.nGroup else {
                return scores
            }
            let groupSize = config.numExperts / config.nGroup
            precondition(groupSize > 0, "num_experts must be divisible by n_group")

            let groupedShape = scores.shape.dropLast() + [config.nGroup, groupSize]
            let groupedScores = scores.reshaped(groupedShape)
            let groupScores =
                top(groupedScores, k: min(2, groupSize), axis: -1)
                .sum(axis: -1)
            let groupIndices = argPartition(
                -groupScores,
                kth: config.topKGroup - 1,
                axis: -1,
            )[.ellipsis, ..<config.topKGroup]
            var groupMask = putAlong(
                zeros(like: groupScores),
                groupIndices,
                values: MLXArray(1.0),
                axis: -1,
            )
            groupMask = broadcast(
                groupMask[.ellipsis, .newAxis],
                to: groupedShape,
            )
            let expertMask = flattened(groupMask, start: -2, end: -1).asType(.bool)
            return MLX.where(expertMask, scores, MLXArray(-1.0e30))
        }
    }

    class SparseMoEBlock: Module, UnaryLayer {
        let config: LLaDA2Configuration

        @ModuleInfo var gate: MoEGate
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
        @ModuleInfo(key: "shared_experts") var sharedExperts: MLP?

        init(_ config: LLaDA2Configuration) {
            self.config = config
            _gate.wrappedValue = MoEGate(config)
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.numExperts,
            )
            if config.numSharedExperts > 0 {
                _sharedExperts.wrappedValue = MLP(
                    config,
                    intermediateSize: config.moeIntermediateSize * config.numSharedExperts,
                )
            }
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (indices, scores) = gate(x)
            var y = switchMLP(x, indices)
            y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)
            if let sharedExperts {
                y = y + sharedExperts(x)
            }
            return y
        }
    }

    class Attention: Module {
        let config: LLaDA2Configuration
        let scale: Float
        let rope: RoPELayer

        @ModuleInfo(key: "query_key_value") var queryKeyValue: Linear
        @ModuleInfo(key: "query_layernorm") var queryLayerNorm: RMSNorm?
        @ModuleInfo(key: "key_layernorm") var keyLayerNorm: RMSNorm?
        @ModuleInfo var dense: Linear

        init(_ config: LLaDA2Configuration) {
            self.config = config
            scale = pow(Float(config.headDim), -0.5)
            _queryKeyValue.wrappedValue = Linear(
                config.hiddenSize,
                (config.attentionHeads + 2 * config.kvHeads) * config.headDim,
                bias: config.useQKVBias,
            )
            if config.useQKNorm {
                _queryLayerNorm.wrappedValue = RMSNorm(
                    dimensions: config.headDim,
                    eps: config.rmsNormEps,
                )
                _keyLayerNorm.wrappedValue = RMSNorm(
                    dimensions: config.headDim,
                    eps: config.rmsNormEps,
                )
            }
            _dense.wrappedValue = Linear(
                config.attentionHeads * config.headDim,
                config.hiddenSize,
                bias: config.useBias,
            )
            rope = initializeRope(
                dims: max(config.rotaryDim, 1),
                base: config.ropeTheta,
                traditional: false,
                scalingConfig: config.ropeScaling,
                maxPositionEmbeddings: config.maxPositionEmbeddings,
            )
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
            positionIDs: MLXArray? = nil,
        )
            -> MLXArray
        {
            let (batchSize, sequenceLength) = (x.dim(0), x.dim(1))
            let qkv = queryKeyValue(x).reshaped(
                batchSize,
                sequenceLength,
                config.attentionHeads + 2 * config.kvHeads,
                config.headDim,
            )
            let parts = split(
                qkv,
                indices: [config.attentionHeads, config.attentionHeads + config.kvHeads],
                axis: -2,
            )
            var queries = parts[0]
            var keys = parts[1]
            var values = parts[2]

            if let queryLayerNorm, let keyLayerNorm {
                queries = queryLayerNorm(queries)
                keys = keyLayerNorm(keys)
            }

            queries = queries.transposed(0, 2, 1, 3)
            keys = keys.transposed(0, 2, 1, 3)
            values = values.transposed(0, 2, 1, 3)

            (queries, keys) = applyRotary(queries: queries, keys: keys, cache: cache)

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
            return dense(output)
        }

        private func applyRotary(
            queries: MLXArray,
            keys: MLXArray,
            cache: KVCache?,
        )
            -> (MLXArray, MLXArray)
        {
            if config.rotaryDim <= 0 {
                return (queries, keys)
            }
            if config.rotaryDim >= config.headDim {
                return (
                    applyRotaryPosition(rope, to: queries, offset: cache?.ropeOffset),
                    applyRotaryPosition(rope, to: keys, offset: cache?.ropeOffset),
                )
            }

            let qParts = split(queries, indices: [config.rotaryDim], axis: -1)
            let kParts = split(keys, indices: [config.rotaryDim], axis: -1)
            return (
                concatenated(
                    [
                        applyRotaryPosition(rope, to: qParts[0], offset: cache?.ropeOffset),
                        qParts[1],
                    ], axis: -1,
                ),
                concatenated(
                    [
                        applyRotaryPosition(rope, to: kParts[0], offset: cache?.ropeOffset),
                        kParts[1],
                    ], axis: -1,
                ),
            )
        }
    }

    final class DecoderLayer: Module {
        @ModuleInfo var attention: Attention
        @ModuleInfo var mlp: Module & UnaryLayer
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: LLaDA2Configuration, layerIndex: Int) {
            _attention.wrappedValue = Attention(config)
            if config.usesSparseMLP(layerIndex) {
                _mlp.wrappedValue = SparseMoEBlock(config)
            } else {
                _mlp.wrappedValue = MLP(config, intermediateSize: config.intermediateSize)
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

    class Model: Module {
        let config: LLaDA2Configuration

        @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
        @ModuleInfo var layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm

        init(_ config: LLaDA2Configuration) {
            self.config = config
            _wordEmbeddings.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
                DecoderLayer(config, layerIndex: $0)
            }
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            cache: [KVCache]? = nil,
            inputEmbeddings: MLXArray? = nil,
        )
            -> MLXArray
        {
            var hidden = inputEmbeddings ?? wordEmbeddings(inputs)
            let mask = MLXFast.ScaledDotProductAttentionMaskMode.none
            for (index, layer) in layers.enumerated() {
                hidden = layer(hidden, mask: mask, cache: cache?[index])
            }
            return norm(hidden)
        }
    }

    final class LanguageModel: Module {
        let config: LLaDA2Configuration

        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        init(_ config: LLaDA2Configuration) {
            self.config = config
            _model.wrappedValue = Model(config)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
            }
            super.init()
        }

        func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
            let hidden = model(inputs, cache: cache)
            if config.tieWordEmbeddings {
                return model.wordEmbeddings.asLinear(hidden)
            }
            return lmHead!(hidden)
        }
    }
}

public final class LLaDA2: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let config: LLaDA2Configuration

    @ModuleInfo(key: "language_model") private var languageModel: LLaDA2Language.LanguageModel

    public init(_ config: LLaDA2Configuration) {
        self.config = config
        vocabularySize = config.vocabularySize
        kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
        _languageModel.wrappedValue = LLaDA2Language.LanguageModel(config)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("rotary_emb.inv_freq") || key.contains("position_ids") {
                continue
            }

            var newKey = key
            if newKey.hasPrefix("model.") || newKey.hasPrefix("lm_head.") {
                newKey = "language_model.\(newKey)"
            }

            if config.tieWordEmbeddings, newKey == "language_model.lm_head.weight" {
                continue
            }
            sanitized[newKey] = value
        }

        stackLegacyExpertWeights(weights: &sanitized)
        return sanitized
    }

    private func stackLegacyExpertWeights(weights: inout [String: MLXArray]) {
        guard config.numExperts > 0 else {
            return
        }

        for layerIndex in config.firstKDenseReplace ..< config.hiddenLayers {
            let prefix = "language_model.model.layers.\(layerIndex).mlp"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).experts.0.\(projection).\(suffix)"
                    guard weights[firstKey] != nil else {
                        continue
                    }
                    let tensors = (0 ..< config.numExperts).map { expert in
                        weights.removeValue(
                            forKey: "\(prefix).experts.\(expert).\(projection).\(suffix)",
                        )!
                    }
                    weights["\(prefix).switch_mlp.\(projection).\(suffix)"] =
                        MLX.stacked(tensors)
                }
            }
        }
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

extension LLaDA2: MaskedDiffusionLanguageModel {
    public var diffusionMaskTokenID: Int {
        config.maskTokenID
    }

    public var diffusionEOSTokenIDs: [Int] {
        config.eosTokenID.values
    }

    public var diffusionPrefersDiffusionGeneration: Bool {
        true
    }

    public func diffusionLogits(_ tokens: MLXArray) -> MLXArray {
        languageModel(tokens, cache: nil)
    }
}
