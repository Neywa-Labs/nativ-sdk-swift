import Foundation
import MLX
import MLXNN

public struct GLM4VMoEConfiguration: Decodable, Sendable {
    public struct RopeParameters: Codable, Sendable {
        public let mropeSection: [Int]
        public let partialRotaryFactor: Float
        public let ropeTheta: Float

        enum CodingKeys: String, CodingKey {
            case mropeSection = "mrope_section"
            case partialRotaryFactor = "partial_rotary_factor"
            case ropeTheta = "rope_theta"
        }

        init(
            mropeSection: [Int] = [8, 12, 12],
            partialRotaryFactor: Float = 0.5,
            ropeTheta: Float = 10000,
        ) {
            self.mropeSection = mropeSection
            self.partialRotaryFactor = partialRotaryFactor
            self.ropeTheta = ropeTheta
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            mropeSection =
                try c.decodeIfPresent([Int].self, forKey: .mropeSection) ?? [8, 12, 12]
            partialRotaryFactor =
                try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.5
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        }
    }

    public struct TextConfiguration: Decodable, Sendable {
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let moeIntermediateSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let vocabularySize: Int
        public let ropeParameters: RopeParameters
        public let rmsNormEps: Float
        public let tieWordEmbeddings: Bool
        public let attentionBias: Bool
        public let firstKDenseReplace: Int
        public let nRoutedExperts: Int?
        public let nSharedExperts: Int?
        public let numExpertsPerToken: Int
        public let routedScalingFactor: Float
        public let normTopkProb: Bool
        public let nGroup: Int
        public let topKGroup: Int
        public let topKMethod: String

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case moeIntermediateSize = "moe_intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case vocabularySize = "vocab_size"
            case ropeParameters = "rope_parameters"
            case ropeScaling = "rope_scaling"
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
            case rmsNormEps = "rms_norm_eps"
            case tieWordEmbeddings = "tie_word_embeddings"
            case attentionBias = "attention_bias"
            case firstKDenseReplace = "first_k_dense_replace"
            case nRoutedExperts = "n_routed_experts"
            case nSharedExperts = "n_shared_experts"
            case numExpertsPerToken = "num_experts_per_tok"
            case routedScalingFactor = "routed_scaling_factor"
            case normTopkProb = "norm_topk_prob"
            case nGroup = "n_group"
            case topKGroup = "topk_group"
            case topKMethod = "topk_method"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 46
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 10944
            moeIntermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 1408
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 96
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
            headDim =
                try c.decodeIfPresent(Int.self, forKey: .headDim)
                ?? (hiddenSize / attentionHeads)
            vocabularySize =
                try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_552
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
            tieWordEmbeddings =
                try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
            attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? true
            firstKDenseReplace =
                try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 1
            nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
            nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts)
            numExpertsPerToken =
                try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 8
            routedScalingFactor =
                try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
            normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
            nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
            topKGroup = try c.decodeIfPresent(Int.self, forKey: .topKGroup) ?? 1
            topKMethod = try c.decodeIfPresent(String.self, forKey: .topKMethod) ?? "noaux_tc"

            let decodedRope =
                try c.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
                ?? c.decodeIfPresent(RopeParameters.self, forKey: .ropeScaling)
                ?? RopeParameters()
            let partial =
                try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
                ?? decodedRope.partialRotaryFactor
            let theta =
                try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? decodedRope.ropeTheta
            ropeParameters = RopeParameters(
                mropeSection: decodedRope.mropeSection,
                partialRotaryFactor: partial,
                ropeTheta: theta,
            )
        }
    }

    public typealias VisionConfiguration = GLM4VConfiguration.VisionConfiguration

    public struct BaseConfiguration: Codable, Sendable {
        public let modelType: String
        private let _vocabularySize: Int?
        private let _imageTokenID: Int?
        private let _videoTokenID: Int?
        private let _visionStartTokenID: Int?
        private let _hiddenSize: Int?

        public var vocabularySize: Int {
            _vocabularySize ?? 151_552
        }

        public var imageTokenID: Int {
            _imageTokenID ?? 151_363
        }

        public var videoTokenID: Int {
            _videoTokenID ?? 151_364
        }

        public var visionStartTokenID: Int {
            _visionStartTokenID ?? 151_339
        }

        public var hiddenSize: Int {
            _hiddenSize ?? 4096
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case _vocabularySize = "vocab_size"
            case _imageTokenID = "image_token_id"
            case _videoTokenID = "video_token_id"
            case _visionStartTokenID = "vision_start_token_id"
            case _hiddenSize = "hidden_size"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let baseConfiguration: BaseConfiguration

    public var imageTokenIndex: Int {
        baseConfiguration.imageTokenID
    }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration = try container.decode(TextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try container.decode(
            VisionConfiguration.self,
            forKey: .visionConfiguration,
        )
        baseConfiguration = try BaseConfiguration(from: decoder)
    }
}

private enum GLM4VMoELanguage {
    static func applyRotaryPosEmb(
        q: MLXArray,
        k: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
    )
        -> (MLXArray, MLXArray)
    {
        var cos = cos[0..., .newAxis, 0..., 0...]
        var sin = sin[0..., .newAxis, 0..., 0...]

        let halfDim = cos.dim(-1) / 2
        cos = repeated(cos[.ellipsis, ..<halfDim], count: 2, axis: -1)
        sin = repeated(sin[.ellipsis, ..<halfDim], count: 2, axis: -1)

        let rotaryDim = cos.dim(-1)
        let qRot = q[.ellipsis, ..<rotaryDim]
        let qPass = q[.ellipsis, rotaryDim...]
        let kRot = k[.ellipsis, ..<rotaryDim]
        let kPass = k[.ellipsis, rotaryDim...]

        let qEmbed = (qRot * cos) + (RotaryEmbeddingOps.rotateInterleaved(qRot) * sin)
        let kEmbed = (kRot * cos) + (RotaryEmbeddingOps.rotateInterleaved(kRot) * sin)
        return (
            concatenated([qEmbed, qPass], axis: -1),
            concatenated([kEmbed, kPass], axis: -1),
        )
    }

    final class RotaryEmbedding {
        let mropeSplitIndices: [Int]
        let invFreq: MLXArray
        let attentionScaling: Float

        init(_ config: GLM4VMoEConfiguration.TextConfiguration) {
            var indices = [Int]()
            var cumsum = 0
            for section in config.ropeParameters.mropeSection.dropLast() {
                cumsum += section
                indices.append(cumsum)
            }
            mropeSplitIndices = indices

            let dim = Int(Float(config.headDim) * config.ropeParameters.partialRotaryFactor)
            let base = config.ropeParameters.ropeTheta
            attentionScaling = 1.0
            let p =
                MLXArray(stride(from: 0, to: dim, by: 2)).asType(.int64).asType(.float32)
                / Float(dim)
            invFreq = 1.0 / pow(base, p)
        }

        func applyMrope(_ freqs: MLXArray) -> MLXArray {
            let chunks = split(freqs, indices: mropeSplitIndices, axis: -1)
            let selected = chunks.enumerated().map { index, chunk in
                chunk[index % 3]
            }
            return concatenated(selected, axis: -1)
        }

        func callAsFunction(_ x: MLXArray, positionIDs: MLXArray) -> (MLXArray, MLXArray) {
            let batchSize = positionIDs.dim(1)
            var invFreqExpanded = invFreq[.newAxis, .newAxis, 0..., .newAxis].asType(.float32)
            invFreqExpanded = broadcast(
                invFreqExpanded,
                to: [3, batchSize, invFreq.dim(0), 1],
            )

            let positionIDsExpanded = positionIDs[0..., 0..., .newAxis, 0...].asType(.float32)
            let freqs = matmul(invFreqExpanded, positionIDsExpanded).transposed(0, 1, 3, 2)
            let mropeFreqs = applyMrope(freqs)
            let emb = concatenated([mropeFreqs, mropeFreqs], axis: -1)
            return (
                (MLX.cos(emb) * attentionScaling).asType(x.dtype),
                (MLX.sin(emb) * attentionScaling).asType(x.dtype),
            )
        }
    }

    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        init(_ config: GLM4VMoEConfiguration.TextConfiguration) {
            let dim = config.hiddenSize
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)

            _wq.wrappedValue = Linear(dim, heads * headDim, bias: config.attentionBias)
            _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.attentionBias)
            _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.attentionBias)
            _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
            positionEmbeddings: (MLXArray, MLXArray),
        )
            -> MLXArray
        {
            let (batchSize, sequenceLength) = (x.dim(0), x.dim(1))
            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(batchSize, sequenceLength, heads, headDim)
                .transposed(0, 2, 1, 3)
            keys = keys.reshaped(batchSize, sequenceLength, kvHeads, headDim)
                .transposed(0, 2, 1, 3)
            values = values.reshaped(batchSize, sequenceLength, kvHeads, headDim)
                .transposed(0, 2, 1, 3)

            let (cos, sin) = positionEmbeddings
            (queries, keys) = applyRotaryPosEmb(q: queries, k: keys, cos: cos, sin: sin)

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

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        init(_ config: GLM4VMoEConfiguration.TextConfiguration, intermediateSize: Int? = nil) {
            let inner = intermediateSize ?? config.intermediateSize
            _gateProj.wrappedValue = Linear(config.hiddenSize, inner, bias: false)
            _upProj.wrappedValue = Linear(config.hiddenSize, inner, bias: false)
            _downProj.wrappedValue = Linear(inner, config.hiddenSize, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    final class MoEGate: Module {
        let config: GLM4VMoEConfiguration.TextConfiguration
        @ModuleInfo var weight: MLXArray
        @ModuleInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

        init(_ config: GLM4VMoEConfiguration.TextConfiguration) {
            self.config = config
            _weight.wrappedValue = zeros([config.nRoutedExperts ?? 1, config.hiddenSize])
            _eScoreCorrectionBias.wrappedValue = zeros([config.nRoutedExperts ?? 1])
        }

        func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
            let topK = config.numExpertsPerToken
            var scores = sigmoid(x.matmul(weight.T).asType(.float32))
            let originalScores = scores
            var choiceScores = scores + eScoreCorrectionBias

            if config.nGroup > 1 {
                let shape = x.shape.dropLast() + [config.nGroup, -1]
                var grouped = choiceScores.reshaped(shape)
                let groupScores = top(grouped, k: 2, axis: -1).sum(axis: -1, keepDims: true)
                let keepGroups = config.nGroup - config.topKGroup
                var groupIndex = argPartition(groupScores, kth: keepGroups - 1, axis: -2)[
                    .ellipsis, ..<keepGroups, 0...,
                ]
                groupIndex = broadcast(
                    groupIndex,
                    to: x.shape.dropLast() + [
                        keepGroups,
                        (config.nRoutedExperts ?? 1) / config.nGroup,
                    ],
                )
                grouped = putAlong(
                    grouped, stopGradient(groupIndex), values: MLXArray(0.0), axis: -2)
                choiceScores = flattened(grouped, start: -2, end: -1)
            }

            let indices = argPartition(-choiceScores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
            scores = takeAlong(originalScores, indices, axis: -1)
            if topK > 1, config.normTopkProb {
                scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-20)
            }
            scores = scores * config.routedScalingFactor
            return (indices, scores)
        }
    }

    final class MoE: Module, UnaryLayer {
        let config: GLM4VMoEConfiguration.TextConfiguration
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
        @ModuleInfo var gate: MoEGate
        @ModuleInfo(key: "shared_experts") var sharedExperts: MLP?

        init(_ config: GLM4VMoEConfiguration.TextConfiguration) {
            self.config = config
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.nRoutedExperts ?? 1,
            )
            _gate.wrappedValue = MoEGate(config)
            if let shared = config.nSharedExperts {
                _sharedExperts.wrappedValue = MLP(
                    config,
                    intermediateSize: config.moeIntermediateSize * shared,
                )
            }
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

    final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        var mlp: UnaryLayer
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: GLM4VMoEConfiguration.TextConfiguration, layerIndex: Int) {
            _attention.wrappedValue = Attention(config)
            if config.nRoutedExperts != nil, layerIndex >= config.firstKDenseReplace {
                mlp = MoE(config)
            } else {
                mlp = MLP(config)
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
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
            positionEmbeddings: (MLXArray, MLXArray),
        )
            -> MLXArray
        {
            let attentionResidual = attention(
                inputLayerNorm(x),
                mask: mask,
                cache: cache,
                positionEmbeddings: positionEmbeddings,
            )
            let h = x + attentionResidual
            return h + mlp(postAttentionLayerNorm(h))
        }
    }

    final class TextModel: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        fileprivate let layers: [DecoderLayer]
        fileprivate let norm: RMSNorm
        let rotaryEmb: RotaryEmbedding

        init(_ config: GLM4VMoEConfiguration.TextConfiguration) {
            precondition(config.vocabularySize > 0)
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            layers = (0 ..< config.hiddenLayers).map { DecoderLayer(config, layerIndex: $0) }
            norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            rotaryEmb = RotaryEmbedding(config)
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
            positionIDs: MLXArray? = nil,
        )
            -> MLXArray
        {
            let h0: MLXArray
            if let inputEmbedding {
                h0 = inputEmbedding
            } else if let inputs {
                h0 = embedTokens(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            var posIDs: MLXArray
            if let positionIDs {
                posIDs = positionIDs
            } else {
                let offset = cache?.first?.offset ?? 0
                let sequenceLength = h0.dim(h0.ndim - 2)
                let positions = MLXArray(Int32(offset) ..< Int32(offset + sequenceLength))
                    .expandedDimensions(axis: 0)
                posIDs = tiled(positions, repetitions: [3, 1, 1])
            }

            let positionEmbeddings = rotaryEmb(h0, positionIDs: posIDs)
            let mask = createAttentionMask(h: h0, cache: cache?.first)
            var h = h0
            for (index, layer) in layers.enumerated() {
                h = layer(
                    h,
                    mask: mask,
                    cache: cache?[index],
                    positionEmbeddings: positionEmbeddings,
                )
            }
            return norm(h)
        }
    }

    final class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: TextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        var kvHeads: [Int]
        var positionIDs: MLXArray?
        var ropeDeltas: MLXArray?

        init(_ config: GLM4VMoEConfiguration.TextConfiguration) {
            model = TextModel(config)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.hiddenSize,
                    config.vocabularySize,
                    bias: false,
                )
            }
            kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
        )
            -> LMOutput
        {
            var currentPositionIDs: MLXArray?
            let cacheOffset = cache?.first?.offset ?? 0

            if let storedPositionIDs = positionIDs {
                let sequenceLength: Int =
                    if let inputEmbedding {
                        inputEmbedding.dim(inputEmbedding.ndim - 2)
                    } else if let inputs {
                        inputs.dim(inputs.ndim - 1)
                    } else {
                        0
                    }

                let storedLength = storedPositionIDs.dim(2)
                if cacheOffset + sequenceLength <= storedLength {
                    currentPositionIDs =
                        storedPositionIDs[
                            0..., 0..., cacheOffset ..< (cacheOffset + sequenceLength),
                        ]
                } else {
                    let delta = ropeDeltas ?? MLXArray(Int32(0))
                    let batchSize = inputEmbedding?.dim(0) ?? inputs?.dim(0) ?? 1
                    var posArrays = [MLXArray]()
                    for _ in 0 ..< 3 {
                        let pos = MLXArray(
                            Int32(cacheOffset) ..< Int32(cacheOffset + sequenceLength),
                        )
                        .expandedDimensions(axis: 0)
                        let tiledPos = tiled(pos, repetitions: [batchSize, 1])
                        posArrays.append((tiledPos + delta).expandedDimensions(axis: 0))
                    }
                    currentPositionIDs = concatenated(posArrays, axis: 0)
                }
            }

            var out = model(
                inputs,
                cache: cache,
                inputEmbedding: inputEmbedding,
                positionIDs: currentPositionIDs,
            )
            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }
            return LMOutput(logits: out)
        }
    }
}

public final class GLM4VMoE: Module, VLMModel, KVCacheDimensionProvider {
    public var toolCallFormat: ToolCallFormat? { .glm4 }

    @ModuleInfo(key: "vision_tower") private var visionModel: GLM4Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: GLM4VMoELanguage.LanguageModel

    public let config: GLM4VMoEConfiguration

    public var vocabularySize: Int {
        config.textConfiguration.vocabularySize
    }

    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: GLM4VMoEConfiguration) {
        self.config = config
        _visionModel.wrappedValue = GLM4Vision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = GLM4VMoELanguage.LanguageModel(config.textConfiguration)
    }

    private func getRopeIndex(inputIDs: MLXArray, imageGridThw: [THW]?) -> (MLXArray, MLXArray) {
        let batchSize = inputIDs.dim(0)
        let sequenceLength = inputIDs.dim(1)
        let spatialMergeSize = config.visionConfiguration.spatialMergeSize
        let imageTokenID = config.baseConfiguration.imageTokenID
        let videoTokenID = config.baseConfiguration.videoTokenID

        guard let imageGridThw, !imageGridThw.isEmpty else {
            let positions = MLXArray(0 ..< Int32(sequenceLength)).expandedDimensions(axis: 0)
            let positionIDs = tiled(
                broadcast(positions, to: [batchSize, sequenceLength]).expandedDimensions(axis: 0),
                repetitions: [3, 1, 1],
            )
            return (positionIDs, MLXArray(Int32(0)))
        }

        precondition(batchSize == 1, "GLM4VMoE getRopeIndex only supports batchSize == 1")
        let positionIDs = zeros([3, batchSize, sequenceLength], type: Int32.self)
        var imageIndex = 0
        var mropePositionDelta = 0

        for batchIndex in 0 ..< batchSize {
            let inputTokens = inputIDs[batchIndex].asArray(Int32.self)
            var dimT = [Int32]()
            var dimH = [Int32]()
            var dimW = [Int32]()
            dimT.reserveCapacity(sequenceLength)
            dimH.reserveCapacity(sequenceLength)
            dimW.reserveCapacity(sequenceLength)

            var start = 0
            var lastMax: Int32 = -1

            let appendTextPositions = { (count: Int) in
                guard count > 0 else {
                    return
                }
                let base = lastMax + 1
                for index in 0 ..< count {
                    let position = base + Int32(index)
                    dimT.append(position)
                    dimH.append(position)
                    dimW.append(position)
                }
                lastMax = base + Int32(count) - 1
            }

            while imageIndex < imageGridThw.count {
                let tokenSlice = inputTokens[start...]
                let imagePosition = tokenSlice.firstIndex(of: Int32(imageTokenID))
                let videoPosition = tokenSlice.firstIndex(of: Int32(videoTokenID))
                let end: Int
                if let imagePosition, let videoPosition {
                    end = min(imagePosition, videoPosition)
                } else if let imagePosition {
                    end = imagePosition
                } else if let videoPosition {
                    end = videoPosition
                } else {
                    break
                }

                let frame = imageGridThw[imageIndex]
                let llmGridT = frame.t
                let llmGridH = frame.h / spatialMergeSize
                let llmGridW = frame.w / spatialMergeSize
                imageIndex += 1

                appendTextPositions(end - start)

                let imageOffset = lastMax + 1
                for t in 0 ..< llmGridT {
                    for h in 0 ..< llmGridH {
                        for w in 0 ..< llmGridW {
                            dimT.append(Int32(t) + imageOffset)
                            dimH.append(Int32(h) + imageOffset)
                            dimW.append(Int32(w) + imageOffset)
                        }
                    }
                }
                lastMax = max(
                    Int32(llmGridT - 1) + imageOffset,
                    max(Int32(llmGridH - 1) + imageOffset, Int32(llmGridW - 1) + imageOffset),
                )
                start = end + llmGridT * llmGridH * llmGridW
            }

            appendTextPositions(inputTokens.count - start)

            positionIDs[0, batchIndex] = MLXArray(dimT)
            positionIDs[1, batchIndex] = MLXArray(dimH)
            positionIDs[2, batchIndex] = MLXArray(dimW)
            mropePositionDelta = Int(lastMax) + 1 - inputTokens.count
        }

        return (positionIDs, MLXArray(Int32(mropePositionDelta)))
    }

    private func inputEmbeddings(
        inputIDs: MLXArray,
        pixelValues: MLXArray?,
        frames: [THW]?,
    )
        -> MLXArray
    {
        guard let pixelValues, let frames else {
            languageModel.positionIDs = nil
            languageModel.ropeDeltas = nil
            return languageModel.model.embedTokens(inputIDs[.newAxis, .ellipsis])
        }

        let inputEmbeds = languageModel.model.embedTokens(inputIDs)
        var hiddenStates = visionModel(pixelValues, frames: frames)
        if hiddenStates.ndim == 2 {
            hiddenStates = hiddenStates[.newAxis, 0..., 0...]
        }

        let merged = MultimodalEmbedding.mergeInputIDsWithImageFeatures(
            inputIDs: inputIDs,
            inputEmbeds: inputEmbeds,
            imageFeatures: hiddenStates,
            imageTokenID: config.baseConfiguration.imageTokenID,
            videoTokenID: config.baseConfiguration.videoTokenID,
        )

        let (positionIDs, ropeDeltas) = getRopeIndex(inputIDs: inputIDs, imageGridThw: frames)
        languageModel.positionIDs = positionIDs
        languageModel.ropeDeltas = ropeDeltas

        return merged
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        var allPixels: MLXArray?
        var allFrames = [THW]()

        if let imagePixels = input.image?.pixels, let imageFrames = input.image?.frames {
            allPixels = visionModel.castPixelValues(imagePixels)
            allFrames.append(contentsOf: imageFrames)
        }

        let embeddings = inputEmbeddings(
            inputIDs: input.text.tokens,
            pixelValues: allPixels,
            frames: allFrames.isEmpty ? nil : allFrames,
        )
        let result = languageModel(nil, cache: cache, inputEmbedding: embeddings)
        prefill.progress?(embeddings.dim(1), embeddings.dim(1))
        return .logits(result)
    }

    public func callAsFunction(
        _ input: LMInput.Text,
        cache: [any KVCache]?,
        state _: LMOutput.State?,
    )
        -> LMOutput
    {
        languageModel(input.tokens, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var transformed = [String: MLXArray]()
        for (key, value) in weights {
            var key = key
            if key.contains("visual") && !key.contains("vision_tower") {
                key = key.replacingOccurrences(of: "model.", with: "")
                key = key.replacingOccurrences(of: "visual", with: "vision_tower")
            }
            if key.contains("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model",
                    with: "language_model.model",
                )
            }
            if key.contains("lm_head") && !key.hasPrefix("language_model") {
                key = key.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            }
            if key.contains("position_ids") || key.contains("rotary_emb.inv_freq") {
                continue
            }
            if key.contains("layers.\(config.textConfiguration.hiddenLayers).") {
                continue
            }
            transformed[key] = value
        }
        GLM4VMoE.stackExperts(weights: &transformed, config: config.textConfiguration)
        return visionModel.sanitize(weights: transformed)
    }

    private static func stackExperts(
        weights: inout [String: MLXArray],
        config: GLM4VMoEConfiguration.TextConfiguration,
    ) {
        guard let routedExperts = config.nRoutedExperts else {
            return
        }
        for layer in 0 ..< config.hiddenLayers {
            let prefix = "language_model.model.layers.\(layer).mlp"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).experts.0.\(projection).\(suffix)"
                    guard weights[firstKey] != nil else {
                        continue
                    }
                    var toStack = [MLXArray]()
                    for expert in 0 ..< routedExperts {
                        if let value = weights.removeValue(
                            forKey: "\(prefix).experts.\(expert).\(projection).\(suffix)",
                        ) {
                            toStack.append(value)
                        }
                    }
                    if toStack.count == routedExperts {
                        weights["\(prefix).switch_mlp.\(projection).\(suffix)"] =
                            stacked(toStack)
                    }
                }
            }
        }
    }
}
