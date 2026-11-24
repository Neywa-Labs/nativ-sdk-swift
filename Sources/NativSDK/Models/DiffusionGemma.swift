import CoreImage
import Foundation
import MLX
import MLXNN

private func diffusionGemmaDefaultVisionRopeParameters() -> [String: StringOrNumber] {
    [
        "rope_theta": .float(100.0),
        "rope_type": .string("default"),
    ]
}

private func diffusionGemmaMaskedScatter(
    inputTensor: MLXArray,
    mask: MLXArray,
    source: MLXArray,
)
    -> MLXArray
{
    let flattenedInput = inputTensor.flattened()
    let flattenedMask = mask.flattened().asArray(Bool.self)
    let flattenedSource = source.flattened()

    let targetIndices = flattenedMask.enumerated().compactMap { index, value in
        value ? Int32(index) : nil
    }
    guard !targetIndices.isEmpty else {
        return inputTensor
    }

    precondition(
        flattenedSource.dim(0) == targetIndices.count,
        "Masked scatter shape mismatch. source=\(flattenedSource.dim(0)) mask=\(targetIndices.count)",
    )

    let result = flattenedInput
    result[MLXArray(targetIndices, [targetIndices.count])] = flattenedSource
    return result.reshaped(inputTensor.shape)
}

private func diffusionGemmaOneHot(_ indices: MLXArray, numClasses: Int) -> MLXArray {
    expandedDimensions(indices, axis: -1) .== MLXArray(0 ..< numClasses)
}

private func diffusionGemmaRotateHalf(_ x: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    return concatenated([-x2, x1], axis: -1)
}

private func diffusionGemmaApplyMultiDimensionalRoPE(
    _ inputs: MLXArray,
    positions: MLXArray,
    baseFrequency: Float,
)
    -> MLXArray
{
    let headDim = inputs.dim(-1)
    if positions.ndim == 2 {
        let half = headDim / 2
        let freqExponents =
            (2.0 / Float(headDim)) * MLXArray(0 ..< half).asType(.float32)
        let timescale = MLX.pow(MLXArray(baseFrequency), freqExponents)
        let sinusoid = positions.asType(.float32).expandedDimensions(axis: -1) / timescale
        var cosValue = cos(sinusoid)
        var sinValue = sin(sinusoid)
        cosValue = concatenated([cosValue, cosValue], axis: -1).asType(inputs.dtype)
        sinValue = concatenated([sinValue, sinValue], axis: -1).asType(inputs.dtype)
        cosValue = expandedDimensions(cosValue, axis: 2)
        sinValue = expandedDimensions(sinValue, axis: 2)
        return inputs * cosValue + diffusionGemmaRotateHalf(inputs) * sinValue
    }

    let numDimensions = positions.dim(-1)
    let channelsPerDimension = 2 * (headDim / (2 * numDimensions))
    let halfPerDimension = channelsPerDimension / 2

    var parts: [MLXArray] = []
    parts.reserveCapacity(numDimensions)

    for dimension in 0 ..< numDimensions {
        let start = dimension * channelsPerDimension
        let end = start + channelsPerDimension
        let part = inputs[.ellipsis, start ..< end]

        let freqExponents =
            (2.0 / Float(channelsPerDimension))
            * MLXArray(0 ..< halfPerDimension).asType(.float32)
        let timescale = MLX.pow(MLXArray(baseFrequency), freqExponents)
        let dimPositions = positions[.ellipsis, dimension ..< dimension + 1].asType(.float32)
        let sinusoid = dimPositions / timescale

        var cosValue = cos(sinusoid)
        var sinValue = sin(sinusoid)
        cosValue = concatenated([cosValue, cosValue], axis: -1).asType(inputs.dtype)
        sinValue = concatenated([sinValue, sinValue], axis: -1).asType(inputs.dtype)
        cosValue = expandedDimensions(cosValue, axis: 2)
        sinValue = expandedDimensions(sinValue, axis: 2)

        parts.append(part * cosValue + diffusionGemmaRotateHalf(part) * sinValue)
    }

    return concatenated(parts, axis: -1)
}

private func diffusionGemmaEnsureFusedSDPA(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode,
)
    -> MLXArray
{
    let fusedDims = [64, 80, 128]
    let d = queries.dim(-1)
    let target = fusedDims.first(where: { d <= $0 }) ?? d

    guard target != d else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask,
        )
    }

    let paddedQueries = MLX.padded(
        queries,
        widths: [0, 0, 0, .init((0, target - d))],
    )
    let paddedKeys = MLX.padded(
        keys,
        widths: [0, 0, 0, .init((0, target - d))],
    )
    let paddedValues = MLX.padded(
        values,
        widths: [0, 0, 0, .init((0, target - d))],
    )

    return MLXFast.scaledDotProductAttention(
        queries: paddedQueries,
        keys: paddedKeys,
        values: paddedValues,
        scale: scale,
        mask: mask,
    )[.ellipsis, ..<d]
}

public struct DiffusionGemmaTextConfiguration: Decodable, Sendable {
    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let moeIntermediateSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let globalKVHeads: Int?
    public let headDim: Int
    public let globalHeadDim: Int?
    public let hiddenActivation: String
    public let rmsNormEps: Float
    public let maxPositionEmbeddings: Int
    public let padTokenID: Int
    public let eosTokenID: IntOrIntArray?
    public let bosTokenID: Int?
    public let tieWordEmbeddings: Bool
    public let ropeParameters: [String: [String: StringOrNumber]]
    public let attentionBias: Bool
    public let attentionDropout: Float
    public let slidingWindow: Int
    public let layerTypes: [String]
    public let finalLogitSoftcapping: Float
    public let useBidirectionalAttention: String?
    public let numExperts: Int
    public let topKExperts: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case globalKVHeads = "num_global_key_value_heads"
        case headDim = "head_dim"
        case globalHeadDim = "global_head_dim"
        case hiddenActivation = "hidden_activation"
        case rmsNormEps = "rms_norm_eps"
        case maxPositionEmbeddings = "max_position_embeddings"
        case padTokenID = "pad_token_id"
        case eosTokenID = "eos_token_id"
        case bosTokenID = "bos_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeParameters = "rope_parameters"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case finalLogitSoftcapping = "final_logit_softcapping"
        case useBidirectionalAttention = "use_bidirectional_attention"
        case numExperts = "num_experts"
        case topKExperts = "top_k_experts"
    }

    public init(
        modelType: String = "diffusion_gemma_text",
        vocabularySize: Int = 262_144,
        hiddenSize: Int = 2816,
        intermediateSize: Int = 2112,
        moeIntermediateSize: Int = 704,
        hiddenLayers: Int = 30,
        attentionHeads: Int = 16,
        kvHeads: Int = 8,
        globalKVHeads: Int? = 2,
        headDim: Int = 256,
        globalHeadDim: Int? = 512,
        hiddenActivation: String = "gelu_pytorch_tanh",
        rmsNormEps: Float = 1e-6,
        maxPositionEmbeddings: Int = 262_144,
        padTokenID: Int = 0,
        eosTokenID: IntOrIntArray? = IntOrIntArray(1),
        bosTokenID: Int? = 2,
        tieWordEmbeddings: Bool = true,
        ropeParameters: [String: [String: StringOrNumber]]? = nil,
        attentionBias: Bool = false,
        attentionDropout: Float = 0,
        slidingWindow: Int = 1024,
        layerTypes: [String]? = nil,
        finalLogitSoftcapping: Float = 30,
        useBidirectionalAttention: String? = "vision",
        numExperts: Int = 128,
        topKExperts: Int = 8,
    ) {
        self.modelType = modelType
        self.vocabularySize = vocabularySize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.moeIntermediateSize = moeIntermediateSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.kvHeads = kvHeads
        self.globalKVHeads = globalKVHeads
        self.headDim = headDim
        self.globalHeadDim = globalHeadDim
        self.hiddenActivation = hiddenActivation
        self.rmsNormEps = rmsNormEps
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.padTokenID = padTokenID
        self.eosTokenID = eosTokenID
        self.bosTokenID = bosTokenID
        self.tieWordEmbeddings = tieWordEmbeddings
        self.ropeParameters = ropeParameters ?? Self.defaultRopeParameters()
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
        self.slidingWindow = slidingWindow
        self.layerTypes = layerTypes ?? Self.defaultLayerTypes(hiddenLayers: hiddenLayers)
        self.finalLogitSoftcapping = finalLogitSoftcapping
        self.useBidirectionalAttention = useBidirectionalAttention
        self.numExperts = numExperts
        self.topKExperts = topKExperts
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 30
        try self.init(
            modelType:
                c.decodeIfPresent(String.self, forKey: .modelType)
                ?? "diffusion_gemma_text",
            vocabularySize: c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262_144,
            hiddenSize: c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2816,
            intermediateSize:
                c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2112,
            moeIntermediateSize:
                c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 704,
            hiddenLayers: hiddenLayers,
            attentionHeads: c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16,
            kvHeads: c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8,
            globalKVHeads: c.decodeIfPresent(Int.self, forKey: .globalKVHeads) ?? 2,
            headDim: c.decodeIfPresent(Int.self, forKey: .headDim) ?? 256,
            globalHeadDim: c.decodeIfPresent(Int.self, forKey: .globalHeadDim) ?? 512,
            hiddenActivation:
                c.decodeIfPresent(String.self, forKey: .hiddenActivation)
                ?? "gelu_pytorch_tanh",
            rmsNormEps: c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6,
            maxPositionEmbeddings:
                c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262_144,
            padTokenID: c.decodeIfPresent(Int.self, forKey: .padTokenID) ?? 0,
            eosTokenID: c.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenID),
            bosTokenID: c.decodeIfPresent(Int.self, forKey: .bosTokenID) ?? 2,
            tieWordEmbeddings:
                c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true,
            ropeParameters:
                c.decodeIfPresent(
                    [String: [String: StringOrNumber]].self,
                    forKey: .ropeParameters,
                ),
            attentionBias: c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false,
            attentionDropout:
                c.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0,
            slidingWindow: c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 1024,
            layerTypes: c.decodeIfPresent([String].self, forKey: .layerTypes),
            finalLogitSoftcapping:
                c.decodeIfPresent(Float.self, forKey: .finalLogitSoftcapping) ?? 30,
            useBidirectionalAttention:
                c.decodeIfPresent(String.self, forKey: .useBidirectionalAttention),
            numExperts: c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 128,
            topKExperts: c.decodeIfPresent(Int.self, forKey: .topKExperts) ?? 8,
        )
    }

    private static func defaultLayerTypes(hiddenLayers: Int) -> [String] {
        let pattern = Array(repeating: "sliding_attention", count: 5) + ["full_attention"]
        var types: [String] = []
        while types.count < hiddenLayers {
            types.append(contentsOf: pattern)
        }
        types = Array(types.prefix(hiddenLayers))
        if !types.isEmpty, types[types.count - 1] != "full_attention" {
            types[types.count - 1] = "full_attention"
        }
        return types
    }

    private static func defaultRopeParameters() -> [String: [String: StringOrNumber]] {
        [
            "sliding_attention": [
                "rope_type": .string("default"),
                "rope_theta": .float(10000),
            ],
            "full_attention": [
                "rope_type": .string("proportional"),
                "partial_rotary_factor": .float(0.25),
                "rope_theta": .float(1_000_000),
            ],
        ]
    }
}

public struct DiffusionGemmaVisionConfiguration: Decodable, Sendable {
    public let modelType: String
    public let hiddenLayers: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let headDim: Int
    public let patchSize: Int
    public let rmsNormEps: Float
    public let defaultOutputLength: Int
    public let positionEmbeddingSize: Int
    public let poolingKernelSize: Int
    public let useClippedLinears: Bool
    public let standardize: Bool
    public let ropeParameters: [String: StringOrNumber]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenLayers = "num_hidden_layers"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case keyValueHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case patchSize = "patch_size"
        case rmsNormEps = "rms_norm_eps"
        case defaultOutputLength = "default_output_length"
        case positionEmbeddingSize = "position_embedding_size"
        case poolingKernelSize = "pooling_kernel_size"
        case useClippedLinears = "use_clipped_linears"
        case standardize
        case ropeParameters = "rope_parameters"
    }

    public init(
        modelType: String = "gemma4_vision",
        hiddenLayers: Int = 16,
        hiddenSize: Int = 768,
        intermediateSize: Int = 3072,
        attentionHeads: Int = 12,
        keyValueHeads: Int? = nil,
        headDim: Int = 64,
        patchSize: Int = 16,
        rmsNormEps: Float = 1e-6,
        defaultOutputLength: Int = 280,
        positionEmbeddingSize: Int = 10240,
        poolingKernelSize: Int = 3,
        useClippedLinears: Bool = false,
        standardize: Bool = false,
        ropeParameters: [String: StringOrNumber]? = nil,
    ) {
        self.modelType = modelType
        self.hiddenLayers = hiddenLayers
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.attentionHeads = attentionHeads
        self.keyValueHeads = keyValueHeads ?? attentionHeads
        self.headDim = headDim
        self.patchSize = patchSize
        self.rmsNormEps = rmsNormEps
        self.defaultOutputLength = defaultOutputLength
        self.positionEmbeddingSize = positionEmbeddingSize
        self.poolingKernelSize = poolingKernelSize
        self.useClippedLinears = useClippedLinears
        self.standardize = standardize
        self.ropeParameters = ropeParameters ?? diffusionGemmaDefaultVisionRopeParameters()
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 12
        try self.init(
            modelType: c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_vision",
            hiddenLayers: c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 16,
            hiddenSize: c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 768,
            intermediateSize:
                c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072,
            attentionHeads: attentionHeads,
            keyValueHeads: c.decodeIfPresent(Int.self, forKey: .keyValueHeads)
                ?? attentionHeads,
            headDim: c.decodeIfPresent(Int.self, forKey: .headDim) ?? 64,
            patchSize: c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16,
            rmsNormEps: c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6,
            defaultOutputLength:
                c.decodeIfPresent(Int.self, forKey: .defaultOutputLength) ?? 280,
            positionEmbeddingSize:
                c.decodeIfPresent(Int.self, forKey: .positionEmbeddingSize) ?? 10240,
            poolingKernelSize:
                c.decodeIfPresent(Int.self, forKey: .poolingKernelSize) ?? 3,
            useClippedLinears:
                c.decodeIfPresent(Bool.self, forKey: .useClippedLinears) ?? false,
            standardize: c.decodeIfPresent(Bool.self, forKey: .standardize) ?? false,
            ropeParameters:
                c.decodeIfPresent(
                    [String: StringOrNumber].self,
                    forKey: .ropeParameters,
                ),
        )
    }
}

public struct DiffusionGemmaConfiguration: Decodable, Sendable {
    public let textConfiguration: DiffusionGemmaTextConfiguration
    public let visionConfiguration: DiffusionGemmaVisionConfiguration?
    public let modelType: String
    public let boiTokenID: Int?
    public let eoiTokenID: Int?
    public let imageTokenID: Int?
    public let videoTokenID: Int?
    public let initializerRange: Float
    public let canvasLength: Int
    public let eosTokenID: IntOrIntArray?
    public let dtype: String?

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case boiTokenID = "boi_token_id"
        case eoiTokenID = "eoi_token_id"
        case imageTokenID = "image_token_id"
        case videoTokenID = "video_token_id"
        case initializerRange = "initializer_range"
        case canvasLength = "canvas_length"
        case eosTokenID = "eos_token_id"
        case dtype
    }

    public init(
        textConfiguration: DiffusionGemmaTextConfiguration = DiffusionGemmaTextConfiguration(),
        visionConfiguration: DiffusionGemmaVisionConfiguration? = nil,
        modelType: String = "diffusion_gemma",
        boiTokenID: Int? = 255_999,
        eoiTokenID: Int? = 258_882,
        imageTokenID: Int? = 258_880,
        videoTokenID: Int? = 258_881,
        initializerRange: Float = 0.02,
        canvasLength: Int = 256,
        eosTokenID: IntOrIntArray? = nil,
        dtype: String? = nil,
    ) {
        self.textConfiguration = textConfiguration
        self.visionConfiguration = visionConfiguration
        self.modelType = modelType
        self.boiTokenID = boiTokenID
        self.eoiTokenID = eoiTokenID
        self.imageTokenID = imageTokenID
        self.videoTokenID = videoTokenID
        self.initializerRange = initializerRange
        self.canvasLength = canvasLength
        self.eosTokenID = eosTokenID
        self.dtype = dtype
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try self.init(
            textConfiguration:
                c.decodeIfPresent(
                    DiffusionGemmaTextConfiguration.self,
                    forKey: .textConfiguration,
                ) ?? DiffusionGemmaTextConfiguration(),
            visionConfiguration:
                c.decodeIfPresent(
                    DiffusionGemmaVisionConfiguration.self,
                    forKey: .visionConfiguration,
                ),
            modelType:
                c.decodeIfPresent(String.self, forKey: .modelType)
                ?? "diffusion_gemma",
            boiTokenID: c.decodeIfPresent(Int.self, forKey: .boiTokenID) ?? 255_999,
            eoiTokenID: c.decodeIfPresent(Int.self, forKey: .eoiTokenID) ?? 258_882,
            imageTokenID: c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? 258_880,
            videoTokenID: c.decodeIfPresent(Int.self, forKey: .videoTokenID) ?? 258_881,
            initializerRange:
                c.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02,
            canvasLength: c.decodeIfPresent(Int.self, forKey: .canvasLength) ?? 256,
            eosTokenID: c.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenID),
            dtype: c.decodeIfPresent(String.self, forKey: .dtype),
        )
    }
}

public enum DiffusionGemmaError: LocalizedError {
    case imageTokenCountMismatch(expectedVisionTokens: Int, actualPromptTokens: Int)
    case imageInputsUnsupported

    public var errorDescription: String? {
        switch self {
        case .imageTokenCountMismatch(let expectedVisionTokens, let actualPromptTokens):
            "Diffusion Gemma vision token count mismatch: vision encoder produced \(expectedVisionTokens) soft tokens, but the prompt contains \(actualPromptTokens) vision tokens."
        case .imageInputsUnsupported:
            "Diffusion Gemma image inputs require a vision configuration and loaded vision tower weights."
        }
    }
}

private enum DiffusionGemmaLanguage {
    final class RMSNormNoScale: Module {
        let eps: Float

        init(eps: Float) {
            self.eps = eps
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
        }
    }

    final class MLP: Module {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        init(_ config: DiffusionGemmaTextConfiguration) {
            _gateProj.wrappedValue = Linear(
                config.hiddenSize,
                config.intermediateSize,
                bias: false,
            )
            _upProj.wrappedValue = Linear(
                config.hiddenSize,
                config.intermediateSize,
                bias: false,
            )
            _downProj.wrappedValue = Linear(
                config.intermediateSize,
                config.hiddenSize,
                bias: false,
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(geluApproximate(gateProj(x)) * upProj(x))
        }
    }

    final class Router: Module {
        let config: DiffusionGemmaTextConfiguration
        let eps: Float
        let rootSize: Float

        @ModuleInfo var proj: Linear
        @ModuleInfo var scale: MLXArray
        @ModuleInfo(key: "per_expert_scale") var perExpertScale: MLXArray

        init(_ config: DiffusionGemmaTextConfiguration) {
            self.config = config
            eps = config.rmsNormEps
            rootSize = pow(Float(config.hiddenSize), -0.5)
            _proj.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
            _scale.wrappedValue = MLXArray.ones([config.hiddenSize])
            _perExpertScale.wrappedValue = MLXArray.ones([config.numExperts])
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, weights: MLXArray) {
            let normalized = MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
            let scores = proj(normalized * scale.asType(x.dtype) * rootSize)
            let topK = min(config.topKExperts, config.numExperts)
            let indices = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
            var weights = takeAlong(scores, indices, axis: -1)
            weights = softmax(weights, axis: -1, precise: true)
            weights = weights * perExpertScale[indices].asType(weights.dtype)
            return (indices.asType(.int32), weights.asType(x.dtype))
        }
    }

    final class Experts: Module {
        let hiddenDims: Int

        @ModuleInfo(key: "gate_up_proj") var gateUpProj: SwitchLinear
        @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

        init(_ config: DiffusionGemmaTextConfiguration) {
            hiddenDims = config.moeIntermediateSize
            _gateUpProj.wrappedValue = SwitchLinear(
                inputDims: config.hiddenSize,
                outputDims: 2 * config.moeIntermediateSize,
                numExperts: config.numExperts,
                bias: false,
            )
            _downProj.wrappedValue = SwitchLinear(
                inputDims: config.moeIntermediateSize,
                outputDims: config.hiddenSize,
                numExperts: config.numExperts,
                bias: false,
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray, indices: MLXArray, weights: MLXArray) -> MLXArray {
            var x = expandedDimensions(x, axes: [-2, -3])
            let doSort = indices.size >= 64
            var activeIndices = indices
            var inverseOrder = MLXArray()

            if doSort {
                (x, activeIndices, inverseOrder) = gatherSort(x: x, indices: indices)
            }

            let gateUp = gateUpProj(x, activeIndices, sortedIndices: doSort)
            let gate = gateUp[.ellipsis, ..<hiddenDims]
            let up = gateUp[.ellipsis, hiddenDims...]
            var y = downProj(geluApproximate(gate) * up, activeIndices, sortedIndices: doSort)

            if doSort {
                y = scatterUnsort(x: y, invOrder: inverseOrder, shape: indices.shape)
            }

            y = squeezed(y, axis: -2)
            return (y * weights[.ellipsis, .newAxis]).sum(axis: -2)
        }
    }

    final class Attention: Module {
        let config: DiffusionGemmaTextConfiguration
        let layerType: String
        let isSliding: Bool
        let headDim: Int
        let attentionHeads: Int
        let kvHeads: Int
        let scale: Float = 1

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear?
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
        @ModuleInfo(key: "v_norm") var vNorm: RMSNormNoScale
        @ModuleInfo var rope: RoPELayer

        init(_ config: DiffusionGemmaTextConfiguration, layerIndex: Int) {
            self.config = config
            layerType = config.layerTypes[layerIndex]
            isSliding = layerType == "sliding_attention"
            headDim =
                !isSliding
                ? (config.globalHeadDim ?? config.headDim)
                : config.headDim
            attentionHeads = config.attentionHeads
            kvHeads =
                !isSliding
                ? (config.globalKVHeads ?? config.kvHeads)
                : config.kvHeads

            _qProj.wrappedValue = Linear(
                config.hiddenSize,
                config.attentionHeads * headDim,
                bias: config.attentionBias,
            )
            _kProj.wrappedValue = Linear(
                config.hiddenSize,
                kvHeads * headDim,
                bias: config.attentionBias,
            )
            if isSliding {
                _vProj.wrappedValue = Linear(
                    config.hiddenSize,
                    kvHeads * headDim,
                    bias: config.attentionBias,
                )
            }
            _oProj.wrappedValue = Linear(
                config.attentionHeads * headDim,
                config.hiddenSize,
                bias: config.attentionBias,
            )
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            _vNorm.wrappedValue = RMSNormNoScale(eps: config.rmsNormEps)

            let ropeParameters = config.ropeParameters[layerType] ?? [:]
            rope = initializeRope(
                dims: headDim,
                base: ropeParameters["rope_theta"]?.asFloat() ?? 10000,
                traditional: false,
                scalingConfig: ropeParameters,
                maxPositionEmbeddings: config.maxPositionEmbeddings,
            )
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
            decoder: Bool = false,
            offset explicitOffset: Int? = nil,
        )
            -> MLXArray
        {
            let batchSize = x.dim(0)
            let sequenceLength = x.dim(1)
            let offset = explicitOffset ?? cacheOffset(cache)

            var queries = qProj(x).reshaped(batchSize, sequenceLength, attentionHeads, headDim)
            queries = qNorm(queries).transposed(0, 2, 1, 3)
            queries = rope(queries, offset: offset)

            var keys = kProj(x).reshaped(batchSize, sequenceLength, kvHeads, headDim)
            let rawValues = vProj?(x).reshaped(batchSize, sequenceLength, kvHeads, headDim) ?? keys
            keys = kNorm(keys).transposed(0, 2, 1, 3)
            keys = rope(keys, offset: offset)
            var values = vNorm(rawValues).transposed(0, 2, 1, 3)

            var activeMask = mask
            if decoder {
                if let (encoderKeys, encoderValues) = cacheState(cache) {
                    var prefixKeys = encoderKeys
                    var prefixValues = encoderValues
                    if isSliding {
                        let window = max(config.slidingWindow - 1, 0)
                        let encoderLength = prefixKeys.dim(2)
                        if window > 0, encoderLength > window, offset >= encoderLength {
                            prefixKeys = prefixKeys[.ellipsis, (encoderLength - window)..., 0...]
                            prefixValues =
                                prefixValues[.ellipsis, (encoderLength - window)..., 0...]
                        }
                    }
                    keys = concatenated([prefixKeys, keys], axis: 2)
                    values = concatenated([prefixValues, values], axis: 2)
                }
                let output = MLXFast.scaledDotProductAttention(
                    queries: queries,
                    keys: keys,
                    values: values,
                    scale: scale,
                    mask: activeMask,
                )
                .transposed(0, 2, 1, 3)
                .reshaped(batchSize, sequenceLength, -1)
                return oProj(output)
            }

            if case .array(let maskArray) = activeMask {
                let keyLength = (cache?.offset ?? 0) + sequenceLength
                if maskArray.dim(-1) != keyLength {
                    activeMask = .array(maskArray[.ellipsis, ..<keyLength])
                }
            }
            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: activeMask,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, sequenceLength, -1)
            return oProj(output)
        }
    }

    final class DecoderLayer: Module {
        let layerType: String

        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
        @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
        @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm
        @ModuleInfo var router: Router
        @ModuleInfo var experts: Experts
        @ModuleInfo(key: "post_feedforward_layernorm_1") var postFeedforwardLayerNorm1: RMSNorm
        @ModuleInfo(key: "post_feedforward_layernorm_2") var postFeedforwardLayerNorm2: RMSNorm
        @ModuleInfo(key: "pre_feedforward_layernorm_2") var preFeedforwardLayerNorm2: RMSNorm
        @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

        init(_ config: DiffusionGemmaTextConfiguration, layerIndex: Int) {
            layerType = config.layerTypes[layerIndex]
            _selfAttention.wrappedValue = Attention(config, layerIndex: layerIndex)
            _mlp.wrappedValue = MLP(config)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _preFeedforwardLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _postFeedforwardLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _router.wrappedValue = Router(config)
            _experts.wrappedValue = Experts(config)
            _postFeedforwardLayerNorm1.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _postFeedforwardLayerNorm2.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _preFeedforwardLayerNorm2.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _layerScalar.wrappedValue = MLXArray.ones([1])
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
            decoder: Bool = false,
            offset: Int? = nil,
            layerScalar overrideLayerScalar: MLXArray? = nil,
        )
            -> MLXArray
        {
            var residual = x
            var h = inputLayerNorm(x)
            h = selfAttention(h, mask: mask, cache: cache, decoder: decoder, offset: offset)
            h = postAttentionLayerNorm(h)
            h = residual + h

            residual = h
            let h1 = postFeedforwardLayerNorm1(mlp(preFeedforwardLayerNorm(h)))

            let flat = residual.reshaped(-1, residual.dim(-1))
            let (indices, weights) = router(flat)
            var h2 = preFeedforwardLayerNorm2(flat)
            h2 = experts(h2, indices: indices, weights: weights)
            h2 = h2.reshaped(residual.shape)
            h2 = postFeedforwardLayerNorm2(h2)

            h = postFeedforwardLayerNorm(h1 + h2)
            h = residual + h
            return h * (overrideLayerScalar ?? layerScalar).asType(h.dtype)
        }
    }

    final class EncoderLayerScalar: Module {
        @ModuleInfo(key: "layer_scalar") var layerScalar: MLXArray

        override init() {
            _layerScalar.wrappedValue = MLXArray.ones([1])
            super.init()
        }
    }

    final class EncoderLanguageModel: Module {
        @ModuleInfo var layers: [EncoderLayerScalar]

        init(layerCount: Int) {
            _layers.wrappedValue = (0 ..< layerCount).map { _ in EncoderLayerScalar() }
            super.init()
        }
    }

    final class SelfConditioning: Module {
        let embedScale: Float

        @ModuleInfo(key: "pre_norm") var preNorm: RMSNorm
        @ModuleInfo(key: "post_norm") var postNorm: RMSNormNoScale
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        init(_ config: DiffusionGemmaTextConfiguration) {
            embedScale = Float(config.hiddenSize).squareRoot()
            _preNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _postNorm.wrappedValue = RMSNormNoScale(eps: config.rmsNormEps)
            _gateProj.wrappedValue = Linear(
                config.hiddenSize,
                config.intermediateSize,
                bias: false,
            )
            _upProj.wrappedValue = Linear(
                config.hiddenSize,
                config.intermediateSize,
                bias: false,
            )
            _downProj.wrappedValue = Linear(
                config.intermediateSize,
                config.hiddenSize,
                bias: false,
            )
            super.init()
        }

        func callAsFunction(_ inputEmbeddings: MLXArray, signal: MLXArray) -> MLXArray {
            let normed = preNorm(signal)
            let update = downProj(geluApproximate(gateProj(normed)) * upProj(normed))
            return postNorm(inputEmbeddings + update)
        }
    }

    final class DecoderModel: Module {
        let config: DiffusionGemmaTextConfiguration
        let embedScale: Float

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo var layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm
        @ModuleInfo(key: "self_conditioning") var selfConditioning: SelfConditioning

        init(_ config: DiffusionGemmaTextConfiguration) {
            self.config = config
            embedScale = Float(config.hiddenSize).squareRoot()
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
                DecoderLayer(config, layerIndex: $0)
            }
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _selfConditioning.wrappedValue = SelfConditioning(config)
            super.init()
        }

        var prefersLogitsSelfConditioning: Bool {
            embedTokens is QuantizedEmbedding
        }

        func prepareSelfConditioning() -> MLXArray? {
            prefersLogitsSelfConditioning ? nil : embedTokens.weight
        }

        func diffusionSelfConditioning(
            processedLogits: MLXArray,
            embeddingWeight: MLXArray?,
        )
            -> MLXArray
        {
            if prefersLogitsSelfConditioning {
                return processedLogits
            }
            let weight = embeddingWeight ?? embedTokens.weight
            let probabilities = softmax(
                processedLogits.asType(weight.dtype),
                axis: -1,
                precise: true,
            )
            return matmul(probabilities.asType(weight.dtype), weight).asType(weight.dtype)
                * embedScale
        }

        func embedCanvas(_ canvasIDs: MLXArray, selfConditioningSignal: MLXArray?) -> MLXArray {
            let inputEmbeddings = embedTokens(canvasIDs) * embedScale
            let signal: MLXArray
            if let selfConditioningSignal {
                if prefersLogitsSelfConditioning,
                    let quantizedEmbedding = embedTokens as? QuantizedEmbedding
                {
                    let probabilities = softmax(selfConditioningSignal, axis: -1, precise: true)
                    signal =
                        quantizedMM(
                            probabilities.asType(inputEmbeddings.dtype),
                            quantizedEmbedding.weight,
                            scales: quantizedEmbedding.scales,
                            biases: quantizedEmbedding.biases,
                            transpose: false,
                            groupSize: quantizedEmbedding.groupSize,
                            bits: quantizedEmbedding.bits,
                            mode: quantizedEmbedding.mode,
                        ) * embedScale
                } else {
                    signal = selfConditioningSignal.asType(inputEmbeddings.dtype)
                }
            } else {
                signal = MLXArray.zeros(like: inputEmbeddings)
            }
            return selfConditioning(inputEmbeddings, signal: signal)
        }

        func decoderMasks(
            canvasIDs: MLXArray,
            cache: [KVCache],
            decoderAttentionMask _: MLXArray? = nil,
        )
            -> [String: MLXFast.ScaledDotProductAttentionMaskMode]
        {
            let batchSize = canvasIDs.dim(0)
            let canvasLength = canvasIDs.dim(1)
            var masks: [String: MLXFast.ScaledDotProductAttentionMaskMode] = [:]

            for layerType in Set(config.layerTypes) {
                let cacheForType = zip(layers, cache).first { layer, _ in
                    layer.layerType == layerType
                }?.1
                guard let state = cacheState(cacheForType) else {
                    masks[layerType] = MLXFast.ScaledDotProductAttentionMaskMode.none
                    continue
                }
                let encoderLength = state.keys.dim(2)
                let validEncoderLength = min(cacheForType?.offset ?? encoderLength, encoderLength)
                let keyLength = encoderLength + canvasLength

                if layerType == "full_attention" {
                    if encoderLength == validEncoderLength {
                        masks[layerType] = MLXFast.ScaledDotProductAttentionMaskMode.none
                    } else {
                        let encoderPositions = MLXArray(Int32(0) ..< Int32(encoderLength))
                        let encoderMask = encoderPositions .< MLXArray(Int32(validEncoderLength))
                        let row = concatenated(
                            [
                                encoderMask,
                                MLXArray.ones([canvasLength], dtype: .bool),
                            ],
                            axis: 0,
                        )
                        masks[layerType] = .array(
                            broadcast(
                                row[.newAxis, .newAxis, .newAxis, 0...],
                                to: [batchSize, 1, canvasLength, keyLength],
                            ),
                        )
                    }
                    continue
                }

                let windowPrefix = max(config.slidingWindow - 1, 0)
                if encoderLength == validEncoderLength, encoderLength <= windowPrefix {
                    masks[layerType] = MLXFast.ScaledDotProductAttentionMaskMode.none
                    continue
                }
                let start = max(0, validEncoderLength - windowPrefix)
                let retainedStart =
                    if windowPrefix > 0,
                        encoderLength > windowPrefix,
                        (cacheForType?.offset ?? encoderLength) >= encoderLength
                    {
                        encoderLength - windowPrefix
                    } else {
                        0
                    }
                let encoderPositions = MLXArray(Int32(retainedStart) ..< Int32(encoderLength))
                let encoderMask =
                    (encoderPositions .>= MLXArray(Int32(start)))
                    & (encoderPositions .< MLXArray(Int32(validEncoderLength)))
                let retainedKeyLength = encoderLength - retainedStart + canvasLength
                let row = concatenated(
                    [
                        encoderMask,
                        MLXArray.ones([canvasLength], dtype: .bool),
                    ],
                    axis: 0,
                )
                masks[layerType] = .array(
                    broadcast(
                        row[.newAxis, .newAxis, .newAxis, 0...],
                        to: [batchSize, 1, canvasLength, retainedKeyLength],
                    ),
                )
            }

            return masks
        }

        func callAsFunction(
            _ canvasIDs: MLXArray,
            cache: [KVCache],
            selfConditioningSignal: MLXArray? = nil,
            decoderAttentionMask: MLXArray? = nil,
        )
            -> MLXArray
        {
            var h = embedCanvas(canvasIDs, selfConditioningSignal: selfConditioningSignal)
            let masks = decoderMasks(
                canvasIDs: canvasIDs,
                cache: cache,
                decoderAttentionMask: decoderAttentionMask,
            )
            let offset = cache.first?.offset ?? 0

            for (layer, layerCache) in zip(layers, cache) {
                h = layer(
                    h,
                    mask: masks[layer.layerType] ?? .none,
                    cache: layerCache,
                    decoder: true,
                    offset: offset,
                )
            }
            return norm(h)
        }
    }

    final class VisionClippableLinear: Module {
        @ModuleInfo(key: "linear") var linear: Linear
        @ModuleInfo(key: "input_min") var inputMin: MLXArray?
        @ModuleInfo(key: "input_max") var inputMax: MLXArray?
        @ModuleInfo(key: "output_min") var outputMin: MLXArray?
        @ModuleInfo(key: "output_max") var outputMax: MLXArray?

        init(inFeatures: Int, outFeatures: Int, bias: Bool = false, useClipping: Bool) {
            _linear.wrappedValue = Linear(inFeatures, outFeatures, bias: bias)
            if useClipping {
                _inputMin.wrappedValue = MLXArray(-Float.infinity)
                _inputMax.wrappedValue = MLXArray(Float.infinity)
                _outputMin.wrappedValue = MLXArray(-Float.infinity)
                _outputMax.wrappedValue = MLXArray(Float.infinity)
            }
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let clippedInput =
                if let inputMin, let inputMax {
                    clip(x, min: inputMin, max: inputMax)
                } else {
                    x
                }
            let projected = linear(clippedInput)
            if let outputMin, let outputMax {
                return clip(projected, min: outputMin, max: outputMax)
            }
            return projected
        }
    }

    final class VisionRMSNorm: Module {
        let eps: Float

        @ModuleInfo var weight: MLXArray

        init(dimensions: Int, eps: Float) {
            self.eps = eps
            _weight.wrappedValue = MLXArray.ones([dimensions])
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let xFloat = x.asType(.float32)
            let variance = mean(xFloat.square(), axis: -1, keepDims: true)
            let normalized = xFloat * rsqrt(variance + eps)
            return (normalized * weight.asType(.float32)).asType(x.dtype)
        }
    }

    final class VisionRMSNormNoScale: Module {
        let eps: Float

        init(eps: Float) {
            self.eps = eps
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let xFloat = x.asType(.float32)
            let variance = mean(xFloat.square(), axis: -1, keepDims: true)
            return (xFloat * rsqrt(variance + eps)).asType(x.dtype)
        }
    }

    final class VisionAttention: Module {
        let numHeads: Int
        let numKVHeads: Int
        let headDim: Int
        let ropeBaseFrequency: Float

        @ModuleInfo(key: "q_proj") var qProj: VisionClippableLinear
        @ModuleInfo(key: "k_proj") var kProj: VisionClippableLinear
        @ModuleInfo(key: "v_proj") var vProj: VisionClippableLinear
        @ModuleInfo(key: "o_proj") var oProj: VisionClippableLinear
        @ModuleInfo(key: "q_norm") var qNorm: VisionRMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: VisionRMSNorm
        @ModuleInfo(key: "_v_norm") var vNorm: VisionRMSNormNoScale

        init(config: DiffusionGemmaVisionConfiguration) {
            numHeads = config.attentionHeads
            numKVHeads = config.keyValueHeads
            headDim = config.headDim
            ropeBaseFrequency = config.ropeParameters["rope_theta"]?.asFloat() ?? 100.0
            _qProj.wrappedValue = VisionClippableLinear(
                inFeatures: config.hiddenSize,
                outFeatures: numHeads * headDim,
                useClipping: config.useClippedLinears,
            )
            _kProj.wrappedValue = VisionClippableLinear(
                inFeatures: config.hiddenSize,
                outFeatures: numKVHeads * headDim,
                useClipping: config.useClippedLinears,
            )
            _vProj.wrappedValue = VisionClippableLinear(
                inFeatures: config.hiddenSize,
                outFeatures: numKVHeads * headDim,
                useClipping: config.useClippedLinears,
            )
            _oProj.wrappedValue = VisionClippableLinear(
                inFeatures: numHeads * headDim,
                outFeatures: config.hiddenSize,
                useClipping: config.useClippedLinears,
            )
            _qNorm.wrappedValue = VisionRMSNorm(
                dimensions: headDim,
                eps: config.rmsNormEps,
            )
            _kNorm.wrappedValue = VisionRMSNorm(
                dimensions: headDim,
                eps: config.rmsNormEps,
            )
            _vNorm.wrappedValue = VisionRMSNormNoScale(eps: config.rmsNormEps)
            super.init()
        }

        func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
            let batch = x.dim(0)
            let length = x.dim(1)

            var queries = qProj(x).reshaped(batch, length, numHeads, headDim)
            var keys = kProj(x).reshaped(batch, length, numKVHeads, headDim)
            var values = vProj(x).reshaped(batch, length, numKVHeads, headDim)

            queries = qNorm(queries)
            keys = kNorm(keys)
            values = vNorm(values)

            queries = diffusionGemmaApplyMultiDimensionalRoPE(
                queries,
                positions: positions,
                baseFrequency: ropeBaseFrequency,
            )
            keys = diffusionGemmaApplyMultiDimensionalRoPE(
                keys,
                positions: positions,
                baseFrequency: ropeBaseFrequency,
            )

            queries = queries.transposed(0, 2, 1, 3)
            keys = keys.transposed(0, 2, 1, 3)
            values = values.transposed(0, 2, 1, 3)

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode =
                if let mask {
                    .array(mask)
                } else {
                    .none
                }
            let output = diffusionGemmaEnsureFusedSDPA(
                queries: queries,
                keys: keys,
                values: values,
                scale: 1.0,
                mask: attentionMask,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)

            return oProj(output)
        }
    }

    final class VisionMLP: Module {
        @ModuleInfo(key: "gate_proj") var gateProj: VisionClippableLinear
        @ModuleInfo(key: "up_proj") var upProj: VisionClippableLinear
        @ModuleInfo(key: "down_proj") var downProj: VisionClippableLinear

        init(config: DiffusionGemmaVisionConfiguration) {
            _gateProj.wrappedValue = VisionClippableLinear(
                inFeatures: config.hiddenSize,
                outFeatures: config.intermediateSize,
                useClipping: config.useClippedLinears,
            )
            _upProj.wrappedValue = VisionClippableLinear(
                inFeatures: config.hiddenSize,
                outFeatures: config.intermediateSize,
                useClipping: config.useClippedLinears,
            )
            _downProj.wrappedValue = VisionClippableLinear(
                inFeatures: config.intermediateSize,
                outFeatures: config.hiddenSize,
                useClipping: config.useClippedLinears,
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(geluApproximate(gateProj(x)) * upProj(x))
        }
    }

    final class VisionTransformerBlock: Module {
        @ModuleInfo(key: "self_attn") var selfAttention: VisionAttention
        @ModuleInfo var mlp: VisionMLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
        @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforwardLayerNorm: RMSNorm
        @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforwardLayerNorm: RMSNorm

        init(config: DiffusionGemmaVisionConfiguration) {
            _selfAttention.wrappedValue = VisionAttention(config: config)
            _mlp.wrappedValue = VisionMLP(config: config)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _preFeedforwardLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _postFeedforwardLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray, positions: MLXArray, mask: MLXArray?) -> MLXArray {
            let normed = inputLayerNorm(x)
            let attentionOutput = selfAttention(normed, positions: positions, mask: mask)
            let h = x + postAttentionLayerNorm(attentionOutput)
            let ff = mlp(preFeedforwardLayerNorm(h))
            return h + postFeedforwardLayerNorm(ff)
        }
    }

    final class VisionPatchEmbedder: Module {
        let patchSize: Int
        let hiddenSize: Int
        let positionEmbeddingSize: Int

        @ModuleInfo(key: "input_proj") var inputProjection: Linear
        @ModuleInfo(key: "position_embedding_table") var positionEmbeddingTable: MLXArray

        init(config: DiffusionGemmaVisionConfiguration) {
            patchSize = config.patchSize
            hiddenSize = config.hiddenSize
            positionEmbeddingSize = config.positionEmbeddingSize
            _inputProjection.wrappedValue = Linear(
                3 * patchSize * patchSize,
                hiddenSize,
                bias: false,
            )
            _positionEmbeddingTable.wrappedValue = MLXArray.ones([
                2, positionEmbeddingSize, hiddenSize,
            ])
            super.init()
        }

        private func patchify(_ pixelValues: MLXArray) -> MLXArray {
            let batch = pixelValues.dim(0)
            let channels = pixelValues.dim(1)
            let height = pixelValues.dim(2)
            let width = pixelValues.dim(3)
            let patchesH = height / patchSize
            let patchesW = width / patchSize

            var patches = pixelValues.reshaped(
                batch,
                channels,
                patchesH,
                patchSize,
                patchesW,
                patchSize,
            )
            patches = patches.transposed(0, 2, 4, 3, 5, 1)
            patches = patches.reshaped(
                batch,
                patchesH * patchesW,
                channels * patchSize * patchSize,
            )
            patches = 2 * (patches - 0.5)
            return inputProjection(patches.asType(inputProjection.weight.dtype))
        }

        func callAsFunction(_ pixelValues: MLXArray, patchPositions: MLXArray) -> MLXArray {
            let hiddenStates = patchify(pixelValues)
            let batch = patchPositions.dim(0)
            let sequenceLength = patchPositions.dim(1)

            let xIndices = patchPositions[0..., 0..., 0].flattened().asType(.int32)
            let yIndices = patchPositions[0..., 0..., 1].flattened().asType(.int32)
            let xEmbeddings = take(positionEmbeddingTable[0], xIndices, axis: 0)
                .reshaped(batch, sequenceLength, hiddenSize)
            let yEmbeddings = take(positionEmbeddingTable[1], yIndices, axis: 0)
                .reshaped(batch, sequenceLength, hiddenSize)
            return hiddenStates + xEmbeddings + yEmbeddings
        }
    }

    final class VisionPooler: Module {
        let hiddenSize: Int
        let defaultOutputLength: Int
        let rootHiddenSize: Float

        init(config: DiffusionGemmaVisionConfiguration) {
            hiddenSize = config.hiddenSize
            defaultOutputLength = config.defaultOutputLength
            rootHiddenSize = pow(Float(config.hiddenSize), 0.5)
            super.init()
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            patchPositions: MLXArray,
            validCount: Int,
            outputLength: Int? = nil,
        )
            -> MLXArray
        {
            let paddingPositions = patchPositions[0..., 0..., 0] .< 0
            let pooledHiddenStates = MLX.where(
                expandedDimensions(paddingPositions, axis: -1),
                MLXArray(0.0, dtype: hiddenStates.dtype),
                hiddenStates,
            )
            let length = outputLength ?? defaultOutputLength
            if pooledHiddenStates.dim(1) <= length {
                return pooledHiddenStates
                    * MLXArray(rootHiddenSize, dtype: pooledHiddenStates.dtype)
            }

            let actualPositions = patchPositions[0, ..<validCount]
            let maxX = Int(actualPositions[0..., 0].max().item(Int32.self)) + 1
            let kernel = Int(sqrt(Double(max(1, validCount / max(length, 1)))))
            let divisor = max(kernel * kernel, 1)
            let pooledLength = max(length, 1)

            var kernelIndices = actualPositions.asType(.int32)
            kernelIndices = floor(kernelIndices.asType(.float32) / Float(kernel)).asType(.int32)
            let flatKernel =
                kernelIndices[0..., 0]
                + MLXArray(Int32(max(maxX / max(kernel, 1), 1))) * kernelIndices[0..., 1]
            let weights =
                diffusionGemmaOneHot(flatKernel, numClasses: pooledLength).asType(.float32)
                / Float(divisor)
            let output = einsum(
                "lL,bld->bLd",
                weights,
                pooledHiddenStates[0..., ..<validCount, 0...],
            )
            .asType(pooledHiddenStates.dtype)
            return output * MLXArray(rootHiddenSize, dtype: pooledHiddenStates.dtype)
        }
    }

    final class VisionTransformerModel: Module {
        @ModuleInfo(key: "layers") var layers: [VisionTransformerBlock]

        init(config: DiffusionGemmaVisionConfiguration) {
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
                VisionTransformerBlock(config: config)
            }
            super.init()
        }

        func callAsFunction(_ hiddenStates: MLXArray, positions: MLXArray, mask: MLXArray?)
            -> MLXArray
        {
            var h = hiddenStates
            for layer in layers {
                h = layer(h, positions: positions, mask: mask)
            }
            return h
        }
    }

    final class VisionModel: Module {
        let config: DiffusionGemmaVisionConfiguration
        let patchSize: Int
        let defaultOutputLength: Int
        let maxPatches: Int

        @ModuleInfo(key: "patch_embedder") var patchEmbedder: VisionPatchEmbedder
        @ModuleInfo(key: "encoder") var encoder: VisionTransformerModel
        @ModuleInfo(key: "pooler") var pooler: VisionPooler
        @ModuleInfo(key: "std_bias") var standardizationBias: MLXArray?
        @ModuleInfo(key: "std_scale") var standardizationScale: MLXArray?

        init(config: DiffusionGemmaVisionConfiguration) {
            self.config = config
            patchSize = config.patchSize
            defaultOutputLength = config.defaultOutputLength
            maxPatches =
                config.defaultOutputLength * config.poolingKernelSize * config.poolingKernelSize
            _patchEmbedder.wrappedValue = VisionPatchEmbedder(config: config)
            _encoder.wrappedValue = VisionTransformerModel(config: config)
            _pooler.wrappedValue = VisionPooler(config: config)
            if config.standardize {
                _standardizationBias.wrappedValue = MLXArray.zeros([config.hiddenSize])
                _standardizationScale.wrappedValue = MLXArray.ones([config.hiddenSize])
            }
            super.init()
        }

        private func patchPositions(batch: Int, height: Int, width: Int) -> (MLXArray, Int) {
            let patchesH = height / patchSize
            let patchesW = width / patchSize
            let realCount = patchesH * patchesW
            let paddedCount = max(maxPatches - realCount, 0)

            var values: [Int32] = []
            values.reserveCapacity(batch * (realCount + paddedCount) * 2)
            for _ in 0 ..< batch {
                for y in 0 ..< patchesH {
                    for x in 0 ..< patchesW {
                        values.append(Int32(x))
                        values.append(Int32(y))
                    }
                }
                for _ in 0 ..< paddedCount {
                    values.append(-1)
                    values.append(-1)
                }
            }

            let count = realCount + paddedCount
            return (MLXArray(values, [batch, count, 2]), realCount)
        }

        func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
            let pixels =
                if pixelValues.ndim == 3 {
                    expandedDimensions(pixelValues, axis: 0)
                } else {
                    pixelValues
                }
            let batch = pixels.dim(0)
            let height = pixels.dim(2)
            let width = pixels.dim(3)
            let (patchPositions, realCount) = patchPositions(
                batch: batch,
                height: height,
                width: width,
            )

            let realPositions = patchPositions[0..., ..<realCount, 0...]
            var hiddenStates = patchEmbedder(pixels, patchPositions: realPositions)

            let paddingCount = maxPatches - realCount
            if paddingCount > 0 {
                let pad = MLXArray.zeros(
                    [batch, paddingCount, hiddenStates.dim(2)],
                    dtype: hiddenStates.dtype,
                )
                hiddenStates = concatenated([hiddenStates, pad], axis: 1)
            }

            let validMask = patchPositions[0..., 0..., 0] .>= 0
            var attentionMask =
                expandedDimensions(validMask, axis: 1) * expandedDimensions(validMask, axis: 2)
            attentionMask = MLX.where(
                attentionMask,
                MLXArray(0.0, dtype: hiddenStates.dtype),
                MLXArray(-Float.infinity, dtype: hiddenStates.dtype),
            )
            attentionMask = expandedDimensions(attentionMask, axis: 1)

            hiddenStates = encoder(hiddenStates, positions: patchPositions, mask: attentionMask)
            hiddenStates = pooler(
                hiddenStates,
                patchPositions: patchPositions,
                validCount: realCount,
            )

            if let standardizationBias, let standardizationScale {
                hiddenStates = (hiddenStates - standardizationBias) * standardizationScale
            }
            return hiddenStates
        }
    }

    final class MultimodalEmbedder: Module {
        @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear
        @ModuleInfo(key: "embedding_pre_projection_norm") var embeddingPreProjectionNorm:
            RMSNormNoScale

        init(embeddingDim: Int, textHiddenSize: Int, eps: Float) {
            _embeddingProjection.wrappedValue = Linear(embeddingDim, textHiddenSize, bias: false)
            _embeddingPreProjectionNorm.wrappedValue = RMSNormNoScale(eps: eps)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            embeddingProjection(embeddingPreProjectionNorm(x))
        }
    }

    final class EncoderModel: Module {
        let config: DiffusionGemmaConfiguration
        let textConfig: DiffusionGemmaTextConfiguration
        let decoder: DecoderModel

        @ModuleInfo(key: "language_model") var languageModel: EncoderLanguageModel
        @ModuleInfo(key: "vision_tower") var visionTower: VisionModel?
        @ModuleInfo(key: "embed_vision") var embedVision: MultimodalEmbedder?

        init(_ config: DiffusionGemmaConfiguration, decoder: DecoderModel) {
            self.config = config
            textConfig = config.textConfiguration
            self.decoder = decoder
            _languageModel.wrappedValue = EncoderLanguageModel(layerCount: textConfig.hiddenLayers)
            if let visionConfiguration = config.visionConfiguration {
                _visionTower.wrappedValue = VisionModel(config: visionConfiguration)
                _embedVision.wrappedValue = MultimodalEmbedder(
                    embeddingDim: visionConfiguration.hiddenSize,
                    textHiddenSize: textConfig.hiddenSize,
                    eps: visionConfiguration.rmsNormEps,
                )
            }
            super.init()
        }

        func makeCache() -> [KVCache] {
            textConfig.layerTypes.map { layerType in
                if layerType == "full_attention" {
                    KVCacheSimple()
                } else {
                    RotatingKVCache(maxSize: textConfig.slidingWindow)
                }
            }
        }

        func getImageFeatures(_ pixelValues: MLXArray) -> MLXArray? {
            guard let visionTower, let embedVision else {
                return nil
            }
            return embedVision(visionTower(pixelValues))
        }

        func embedInputs(_ inputIDs: MLXArray, pixelValues: MLXArray? = nil) throws -> MLXArray {
            let imageMask: MLXArray =
                if let imageTokenID = config.imageTokenID {
                    inputIDs .== imageTokenID
                } else {
                    MLXArray.zeros(like: inputIDs).asType(.bool)
                }
            let videoMask: MLXArray =
                if let videoTokenID = config.videoTokenID {
                    inputIDs .== videoTokenID
                } else {
                    MLXArray.zeros(like: imageMask)
                }
            let visionMask = imageMask | videoMask
            let textIDs = MLX.where(visionMask, MLXArray(textConfig.padTokenID), inputIDs)
            var inputsEmbeds = decoder.embedTokens(textIDs) * decoder.embedScale

            guard let pixelValues else {
                return inputsEmbeds
            }
            guard let imageFeatures = getImageFeatures(pixelValues) else {
                throw DiffusionGemmaError.imageInputsUnsupported
            }

            let expectedVisionTokens = imageFeatures.dim(0) * imageFeatures.dim(1)
            let actualPromptTokens = visionMask.asType(.int32).sum().item(Int.self)
            guard expectedVisionTokens == actualPromptTokens else {
                throw DiffusionGemmaError.imageTokenCountMismatch(
                    expectedVisionTokens: expectedVisionTokens,
                    actualPromptTokens: actualPromptTokens,
                )
            }

            var imageMaskExpanded = expandedDimensions(visionMask, axis: -1)
            imageMaskExpanded = broadcast(imageMaskExpanded, to: inputsEmbeds.shape)
            inputsEmbeds = diffusionGemmaMaskedScatter(
                inputTensor: inputsEmbeds,
                mask: imageMaskExpanded,
                source: imageFeatures.asType(inputsEmbeds.dtype),
            )
            return inputsEmbeds
        }

        func encoderMasks(
            h: MLXArray,
            cache: [KVCache],
            attentionMask: MLXArray? = nil,
            visionOverlay: MLXArray? = nil,
        )
            -> [MLXFast.ScaledDotProductAttentionMaskMode]
        {
            var overlay = visionOverlay
            let sequenceLength = h.dim(1)
            let keyLength = sequenceLength + (cache.first?.offset ?? 0)
            if keyLength != sequenceLength {
                overlay = nil
            }

            guard let attentionMask else {
                if overlay == nil {
                    return zip(decoder.layers, cache).map { layer, layerCache in
                        createAttentionMask(
                            h: h,
                            cache: layerCache,
                            windowSize: layer.layerType == "sliding_attention"
                                ? textConfig.slidingWindow : nil,
                        )
                    }
                }

                let batchSize = h.dim(0)
                let keyMask = MLXArray.ones([batchSize, keyLength], dtype: .bool)
                let positions = MLXArray(Int32(0) ..< Int32(keyLength))
                let qPositions = MLXArray(Int32(keyLength - sequenceLength) ..< Int32(keyLength))[
                    0..., .newAxis,
                ]
                let base = qPositions .>= positions[.newAxis, 0...]

                return zip(decoder.layers, cache).map { layer, _ in
                    var mask = base
                    if layer.layerType == "sliding_attention" {
                        mask =
                            mask
                            & (qPositions .< positions[.newAxis, 0...] + textConfig.slidingWindow)
                    }
                    var fullMask = mask[.newAxis, .newAxis, 0..., 0...]
                    if let overlay {
                        fullMask = fullMask | overlay[0..., .newAxis, 0..., 0...]
                    }
                    fullMask = fullMask & keyMask[0..., .newAxis, .newAxis, 0...]
                    return .array(
                        broadcast(fullMask, to: [batchSize, 1, sequenceLength, keyLength]),
                    )
                }
            }

            let batchSize = h.dim(0)
            var keyMask = attentionMask.asType(.bool)
            if keyMask.dim(-1) != keyLength, keyMask.dim(-1) > keyLength {
                keyMask = keyMask[.ellipsis, (keyMask.dim(-1) - keyLength)...]
            }
            let positions = MLXArray(Int32(0) ..< Int32(keyLength))
            let qPositions = MLXArray(Int32(keyLength - sequenceLength) ..< Int32(keyLength))[
                0..., .newAxis,
            ]
            let base = qPositions .>= positions[.newAxis, 0...]

            return decoder.layers.map { layer in
                var mask = base
                if layer.layerType == "sliding_attention" {
                    mask =
                        mask & (qPositions .< positions[.newAxis, 0...] + textConfig.slidingWindow)
                }
                var fullMask = mask[.newAxis, .newAxis, 0..., 0...]
                if let overlay {
                    fullMask = fullMask | overlay[0..., .newAxis, 0..., 0...]
                }
                fullMask = fullMask & keyMask[0..., .newAxis, .newAxis, 0...]
                return .array(broadcast(fullMask, to: [batchSize, 1, sequenceLength, keyLength]))
            }
        }

        func visionBlockOverlay(inputIDs: MLXArray) -> MLXArray? {
            guard textConfig.useBidirectionalAttention == "vision", inputIDs.dim(1) > 1 else {
                return nil
            }

            let imageMask: MLXArray =
                if let imageTokenID = config.imageTokenID {
                    inputIDs .== imageTokenID
                } else {
                    MLXArray.zeros(like: inputIDs).asType(.bool)
                }
            let videoMask: MLXArray =
                if let videoTokenID = config.videoTokenID {
                    inputIDs .== videoTokenID
                } else {
                    MLXArray.zeros(like: imageMask)
                }
            let visionMask = imageMask | videoMask
            guard visionMask.asType(.int32).sum().item(Int.self) > 0 else {
                return nil
            }

            let previous = concatenated(
                [
                    MLXArray.zeros([inputIDs.dim(0), 1], dtype: .bool),
                    visionMask[0..., ..<(inputIDs.dim(1) - 1)],
                ],
                axis: 1,
            )
            let starts = visionMask & logicalNot(previous)
            let groupIDs = cumsum(starts.asType(.int32), axis: 1) - 1
            let blockIDs = MLX.where(
                visionMask,
                groupIDs,
                MLXArray.zeros(like: groupIDs) - 1,
            )
            let queryBlocks = expandedDimensions(blockIDs, axis: -1)
            let keyBlocks = expandedDimensions(blockIDs, axis: -2)
            return (queryBlocks .!= -1) & (queryBlocks .== keyBlocks)
        }

        func callAsFunction(
            _ inputIDs: MLXArray,
            attentionMask: MLXArray? = nil,
            cache: [KVCache]? = nil,
            pixelValues: MLXArray? = nil,
        )
            throws -> (hidden: MLXArray, cache: [KVCache])
        {
            var h = try embedInputs(inputIDs, pixelValues: pixelValues)
            let activeCache = cache ?? makeCache()
            let masks = encoderMasks(
                h: h,
                cache: activeCache,
                attentionMask: attentionMask,
                visionOverlay: visionBlockOverlay(inputIDs: inputIDs),
            )

            for index in 0 ..< decoder.layers.count {
                h = decoder.layers[index](
                    h,
                    mask: masks[index],
                    cache: activeCache[index],
                    decoder: false,
                    layerScalar: languageModel.layers[index].layerScalar,
                )
            }
            return (decoder.norm(h), activeCache)
        }
    }

    final class Backbone: Module {
        let config: DiffusionGemmaConfiguration

        @ModuleInfo var decoder: DecoderModel
        @ModuleInfo var encoder: EncoderModel

        init(_ config: DiffusionGemmaConfiguration) {
            self.config = config
            let decoder = DecoderModel(config.textConfiguration)
            _decoder.wrappedValue = decoder
            _encoder.wrappedValue = EncoderModel(config, decoder: decoder)
            super.init()
        }

        func diffusionPrefillCache(
            inputIDs: MLXArray,
            attentionMask: MLXArray?,
            cache: [KVCache],
            pixelValues: MLXArray?,
        )
            throws
        {
            _ = try encoder(
                inputIDs,
                attentionMask: attentionMask,
                cache: cache,
                pixelValues: pixelValues,
            )
        }

        func diffusionUpdateCache(inputIDs: MLXArray, cache: [KVCache]) {
            _ = try! encoder(inputIDs, attentionMask: nil, cache: cache, pixelValues: nil)
        }

        func diffusionDecoderLogits(
            canvasIDs: MLXArray,
            cache: [KVCache],
            selfConditioning: MLXArray?,
        )
            -> MLXArray
        {
            decoder(
                canvasIDs,
                cache: cache,
                selfConditioningSignal: selfConditioning,
            )
        }
    }

    static func cacheOffset(_ cache: KVCache?) -> Int {
        cache?.offset ?? 0
    }

    static func cacheState(_ cache: KVCache?) -> (keys: MLXArray, values: MLXArray)? {
        guard let cache else {
            return nil
        }
        let state = cache.state
        guard state.count == 2 else {
            return nil
        }
        return (state[0], state[1])
    }
}

private func cacheOffset(_ cache: KVCache?) -> Int {
    DiffusionGemmaLanguage.cacheOffset(cache)
}

private func cacheState(_ cache: KVCache?) -> (keys: MLXArray, values: MLXArray)? {
    DiffusionGemmaLanguage.cacheState(cache)
}

public final class DiffusionGemma: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let config: DiffusionGemmaConfiguration

    @ModuleInfo private var model: DiffusionGemmaLanguage.Backbone

    public init(_ config: DiffusionGemmaConfiguration) {
        self.config = config
        vocabularySize = config.textConfiguration.vocabularySize
        kvHeads = config.textConfiguration.layerTypes.map { layerType in
            if layerType == "sliding_attention" {
                config.textConfiguration.kvHeads
            } else {
                config.textConfiguration.globalKVHeads ?? config.textConfiguration.kvHeads
            }
        }
        _model.wrappedValue = DiffusionGemmaLanguage.Backbone(config)
        super.init()
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        try diffusionPrefill(input: input, cache: cache)
        let canvas = MLXArray([config.textConfiguration.padTokenID], [1, 1])
        let logits = diffusionDecoderLogits(canvas: canvas, cache: cache, selfConditioning: nil)
        prefill.progress?(input.text.tokens.size, input.text.tokens.size)
        return .logits(LMOutput(logits: logits))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let activeCache = cache ?? diffusionMakeCache(parameters: nil)
        return diffusionDecoderLogits(canvas: inputs, cache: activeCache, selfConditioning: nil)
    }

    public func newCache(configuration _: KVCacheConfiguration?) -> [KVCache] {
        model.encoder.makeCache()
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("rotary_emb") || key == "lm_head.weight" {
                continue
            }
            if key.hasPrefix("model.encoder.embed_vision.")
                || key.hasPrefix("model.encoder.vision_tower.")
            {
                if config.visionConfiguration == nil {
                    continue
                }
                if config.visionConfiguration?.useClippedLinears != true,
                    key.contains("input_min")
                        || key.contains("input_max")
                        || key.contains("output_min")
                        || key.contains("output_max")
                {
                    continue
                }
            }
            if key.hasPrefix("model.encoder.language_model.") {
                if key.hasSuffix(".layer_scalar") {
                    sanitized[key] = value
                }
                continue
            }
            if key.hasSuffix(".experts.down_proj") {
                sanitized["\(key).weight"] = value
                continue
            }
            if key.hasSuffix(".experts.gate_up_proj") {
                sanitized["\(key).weight"] = value
                continue
            }
            sanitized[key] = value
        }
        return sanitized
    }

    public var loraLayers: [Module] {
        model.decoder.layers
    }
}

extension DiffusionGemma: BlockDiffusionLanguageModel {
    public var diffusionCanvasLength: Int {
        config.canvasLength
    }

    public var diffusionEOSTokenIDs: [Int] {
        config.eosTokenID?.values
            ?? config.textConfiguration.eosTokenID?.values
            ?? [1]
    }

    public var diffusionPrefersDiffusionGeneration: Bool {
        true
    }

    public var diffusionVocabularySize: Int {
        config.textConfiguration.vocabularySize
    }

    public func diffusionMakeCache(parameters _: DiffusionGenerateParameters?) -> [KVCache] {
        model.encoder.makeCache()
    }

    public func diffusionPrefill(input: LMInput, cache: [KVCache]) throws {
        try model.diffusionPrefillCache(
            inputIDs: input.text.tokens,
            attentionMask: input.text.mask,
            cache: cache,
            pixelValues: diffusionGemmaVisionPixels(input),
        )
    }

    public func diffusionUpdateCache(tokens: MLXArray, cache: [KVCache]) {
        model.diffusionUpdateCache(inputIDs: tokens, cache: cache)
    }

    public func diffusionDecoderLogits(
        canvas: MLXArray,
        cache: [KVCache],
        selfConditioning: MLXArray?,
    )
        -> MLXArray
    {
        let hidden = model.diffusionDecoderLogits(
            canvasIDs: canvas,
            cache: cache,
            selfConditioning: selfConditioning,
        )
        let logits = model.decoder.embedTokens.asLinear(hidden)
        return tanh(logits.asType(.float32) / config.textConfiguration.finalLogitSoftcapping)
            * config.textConfiguration.finalLogitSoftcapping
    }

    public func diffusionSelfConditioning(logits: MLXArray) -> MLXArray? {
        model.decoder.diffusionSelfConditioning(
            processedLogits: logits,
            embeddingWeight: model.decoder.prepareSelfConditioning(),
        )
    }
}

private func diffusionGemmaVisionPixels(_ input: LMInput) -> MLXArray? {
    switch (input.image?.pixels, input.video?.pixels) {
    case (let imagePixels?, let videoPixels?):
        concatenated([imagePixels, videoPixels])
    case (let imagePixels?, nil):
        imagePixels
    case (nil, let videoPixels?):
        videoPixels
    case (nil, nil):
        nil
    }
}

public struct DiffusionGemmaProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String
    public let doNormalize: Bool
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let imageSeqLength: Int
    public let size: Gemma3ProcessorConfiguration.ImageSize?
    public let imageTokenID: Int
    public let videoTokenID: Int
    public let boiTokenID: Int
    public let eoiTokenID: Int?
    public let videoFPS: Double
    public let maxVideoFrames: Int

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case doNormalize = "do_normalize"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageSeqLength = "image_seq_length"
        case size
        case imageTokenID = "image_token_id"
        case videoTokenID = "video_token_id"
        case boiTokenID = "boi_token_id"
        case eoiTokenID = "eoi_token_id"
        case videoFPS = "video_fps"
        case fps
        case maxVideoFrames = "max_num_frames"
        case maxFrames = "max_frames"
        case numFrames = "num_frames"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass =
            try c.decodeIfPresent(String.self, forKey: .processorClass)
            ?? "DiffusionGemma4Processor"
        doNormalize = try c.decodeIfPresent(Bool.self, forKey: .doNormalize) ?? false
        imageMean = try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd = try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        imageSeqLength = try c.decodeIfPresent(Int.self, forKey: .imageSeqLength) ?? 280
        size = try c.decodeIfPresent(
            Gemma3ProcessorConfiguration.ImageSize.self,
            forKey: .size,
        )
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? 258_880
        videoTokenID =
            try c.decodeIfPresent(Int.self, forKey: .videoTokenID) ?? 258_881
        boiTokenID = try c.decodeIfPresent(Int.self, forKey: .boiTokenID) ?? 255_999
        eoiTokenID = try c.decodeIfPresent(Int.self, forKey: .eoiTokenID) ?? 258_882
        let decodedVideoFPS = try c.decodeIfPresent(Double.self, forKey: .videoFPS)
        let decodedFPS = try c.decodeIfPresent(Double.self, forKey: .fps)
        videoFPS = decodedVideoFPS ?? decodedFPS ?? 2
        let decodedMaxVideoFrames = try c.decodeIfPresent(Int.self, forKey: .maxVideoFrames)
        let decodedMaxFrames = try c.decodeIfPresent(Int.self, forKey: .maxFrames)
        let decodedNumFrames = try c.decodeIfPresent(Int.self, forKey: .numFrames)
        maxVideoFrames =
            decodedMaxVideoFrames
            ?? decodedMaxFrames
            ?? decodedNumFrames
            ?? Int.max
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    public var fixedSize: CGSize {
        if let size {
            return CGSize(width: size.width, height: size.height)
        }
        return CGSize(width: 800, height: 800)
    }
}

public struct DiffusionGemmaProcessor: UserInputProcessor {
    private let config: DiffusionGemmaProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: DiffusionGemmaProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?) -> CIImage {
        var userProcessing = processing ?? UserInput.Processing()
        let targetSize = config.fixedSize
        userProcessing.resize = targetSize

        let processedImage = MediaProcessing.apply(image, processing: userProcessing)
        let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
        let resizedImage = MediaProcessing.resampleBicubic(srgbImage, to: targetSize)
        if config.doNormalize {
            return MediaProcessing.normalize(
                resizedImage,
                mean: config.imageMeanTuple,
                std: config.imageStdTuple,
            )
        }
        return resizedImage
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray,
        THW,
    ) {
        let processedImages = images.map {
            MediaProcessing.asMLXArray(preprocess(image: $0, processing: processing))
        }
        return (
            concatenated(processedImages),
            THW(images.count, Int(config.fixedSize.height), Int(config.fixedSize.width)),
        )
    }

    private func preprocess(video: UserInput.Video, processing: UserInput.Processing?) async throws
        -> (MLXArray, THW)
    {
        let maxFrames = max(config.maxVideoFrames, 1)
        let sequence = try await MediaProcessing.asProcessedSequence(
            video,
            targetFPS: { _ in config.videoFPS > 0 ? config.videoFPS : 2 },
            maxFrames: maxFrames,
        ) { frame in
            try UserInput.VideoFrame(
                image: .ciImage(preprocess(image: frame.image.asCIImage(), processing: processing)),
                timeStamp: frame.timeStamp,
            )
        }
        let frames = Array(sequence.frames.prefix(maxFrames))

        guard !frames.isEmpty else {
            throw VLMError.processing("Diffusion Gemma video preprocessing produced no frames.")
        }

        return (
            concatenated(frames),
            THW(frames.count, Int(config.fixedSize.height), Int(config.fixedSize.width)),
        )
    }

    private func appendVisionTokens(tokenID: Int, count: Int, to tokens: inout [Int]) {
        tokens.append(config.boiTokenID)
        tokens.append(contentsOf: Array(repeating: tokenID, count: count))
        if let eoiTokenID = config.eoiTokenID {
            tokens.append(eoiTokenID)
        }
    }

    private func expandPromptTokens(
        _ promptTokens: [Int],
        imageFrames: [THW],
        videoFrames: [THW],
    )
        -> [Int]
    {
        var expandedTokens: [Int] = []
        expandedTokens.reserveCapacity(
            promptTokens.count
                + (imageFrames.count + videoFrames.count) * (config.imageSeqLength + 2),
        )
        var imageIndex = 0
        var videoIndex = 0

        for token in promptTokens {
            if token == config.imageTokenID, imageIndex < imageFrames.count {
                appendVisionTokens(
                    tokenID: config.imageTokenID,
                    count: config.imageSeqLength * max(imageFrames[imageIndex].t, 1),
                    to: &expandedTokens,
                )
                imageIndex += 1
            } else if token == config.videoTokenID, videoIndex < videoFrames.count {
                appendVisionTokens(
                    tokenID: config.videoTokenID,
                    count: config.imageSeqLength * max(videoFrames[videoIndex].t, 1),
                    to: &expandedTokens,
                )
                videoIndex += 1
            } else {
                expandedTokens.append(token)
            }
        }

        return expandedTokens
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Gemma4MessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext,
        )

        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            processedImage = LMInput.ProcessedImage(
                pixels: concatenated(imagePixelsAndFrames.map(\.0)),
                frames: imagePixelsAndFrames.map(\.1),
            )
        }

        var processedVideo: LMInput.ProcessedVideo?
        if !input.videos.isEmpty {
            var pixels: [MLXArray] = []
            var frames: [THW] = []
            for video in input.videos {
                let videoPixelsAndFrames = try await preprocess(
                    video: video,
                    processing: input.processing,
                )
                pixels.append(videoPixelsAndFrames.0)
                frames.append(videoPixelsAndFrames.1)
            }
            processedVideo = LMInput.ProcessedVideo(
                pixels: concatenated(pixels),
                frames: frames,
            )
        }

        promptTokens = expandPromptTokens(
            promptTokens,
            imageFrames: processedImage?.frames ?? [],
            videoFrames: processedVideo?.frames ?? [],
        )

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage,
            video: processedVideo,
        )
    }
}
