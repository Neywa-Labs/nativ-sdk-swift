import Foundation
import MLX
import MLXNN

// MARK: - Audio Encoder Configuration

public struct Gemma4AudioEncoderConfiguration: Codable, Sendable {
    public let modelType: String
    public let audioSamplesPerToken: Int
    public let audioEmbedDim: Int
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let attentionHeads: Int
    public let hiddenActivation: String
    public let subsamplingConvChannels: [Int]
    public let convKernelSize: Int
    public let residualWeight: Float
    public let attentionChunkSize: Int
    public let attentionContextLeft: Int
    public let attentionContextRight: Int
    public let attentionLogitCap: Float
    public let attentionInvalidLogitsValue: Float
    public let useClippedLinears: Bool
    public let rmsNormEps: Float
    public let gradientClipping: Float
    public let outputProjectionDimensions: Int?
    public let inputFeatureSize: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case audioSamplesPerToken = "audio_samples_per_token"
        case audioEmbedDim = "audio_embed_dim"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case hiddenActivation = "hidden_act"
        case subsamplingConvChannels = "subsampling_conv_channels"
        case convKernelSize = "conv_kernel_size"
        case residualWeight = "residual_weight"
        case attentionChunkSize = "attention_chunk_size"
        case attentionContextLeft = "attention_context_left"
        case attentionContextRight = "attention_context_right"
        case attentionLogitCap = "attention_logit_cap"
        case attentionInvalidLogitsValue = "attention_invalid_logits_value"
        case useClippedLinears = "use_clipped_linears"
        case rmsNormEps = "rms_norm_eps"
        case gradientClipping = "gradient_clipping"
        case outputProjectionDimensions = "output_proj_dims"
        case inputFeatureSize = "input_feature_size"
    }

    public init(
        modelType: String = "gemma4_unified_audio",
        audioSamplesPerToken: Int = 640,
        audioEmbedDim: Int = 640,
        hiddenSize: Int = 640,
        hiddenLayers: Int = 12,
        attentionHeads: Int = 8,
        hiddenActivation: String = "silu",
        subsamplingConvChannels: [Int] = [128, 32],
        convKernelSize: Int = 5,
        residualWeight: Float = 0.5,
        attentionChunkSize: Int = 12,
        attentionContextLeft: Int = 13,
        attentionContextRight: Int = 0,
        attentionLogitCap: Float = 50,
        attentionInvalidLogitsValue: Float = -1e9,
        useClippedLinears: Bool = true,
        rmsNormEps: Float = 1e-6,
        gradientClipping: Float = 1e10,
        outputProjectionDimensions: Int? = 640,
        inputFeatureSize: Int = 128,
    ) {
        self.modelType = modelType
        self.audioSamplesPerToken = audioSamplesPerToken
        self.audioEmbedDim = audioEmbedDim
        self.hiddenSize = hiddenSize
        self.hiddenLayers = hiddenLayers
        self.attentionHeads = attentionHeads
        self.hiddenActivation = hiddenActivation
        self.subsamplingConvChannels = subsamplingConvChannels
        self.convKernelSize = convKernelSize
        self.residualWeight = residualWeight
        self.attentionChunkSize = attentionChunkSize
        self.attentionContextLeft = attentionContextLeft
        self.attentionContextRight = attentionContextRight
        self.attentionLogitCap = attentionLogitCap
        self.attentionInvalidLogitsValue = attentionInvalidLogitsValue
        self.useClippedLinears = useClippedLinears
        self.rmsNormEps = rmsNormEps
        self.gradientClipping = gradientClipping
        self.outputProjectionDimensions = outputProjectionDimensions
        self.inputFeatureSize = inputFeatureSize
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma4_unified_audio"
        audioSamplesPerToken =
            try c.decodeIfPresent(Int.self, forKey: .audioSamplesPerToken) ?? 640
        audioEmbedDim = try c.decodeIfPresent(Int.self, forKey: .audioEmbedDim) ?? 640
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? audioEmbedDim
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 12
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
        hiddenActivation =
            try c.decodeIfPresent(String.self, forKey: .hiddenActivation) ?? "silu"
        subsamplingConvChannels =
            try c.decodeIfPresent([Int].self, forKey: .subsamplingConvChannels) ?? [128, 32]
        convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 5
        residualWeight = try c.decodeIfPresent(Float.self, forKey: .residualWeight) ?? 0.5
        attentionChunkSize =
            try c.decodeIfPresent(Int.self, forKey: .attentionChunkSize) ?? 12
        attentionContextLeft =
            try c.decodeIfPresent(Int.self, forKey: .attentionContextLeft) ?? 13
        attentionContextRight =
            try c.decodeIfPresent(Int.self, forKey: .attentionContextRight) ?? 0
        attentionLogitCap =
            try c.decodeIfPresent(Float.self, forKey: .attentionLogitCap) ?? 50
        attentionInvalidLogitsValue =
            try c.decodeIfPresent(Float.self, forKey: .attentionInvalidLogitsValue) ?? -1e9
        useClippedLinears =
            try c.decodeIfPresent(Bool.self, forKey: .useClippedLinears) ?? true
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        gradientClipping =
            try c.decodeIfPresent(Float.self, forKey: .gradientClipping) ?? 1e10
        outputProjectionDimensions =
            try c.decodeIfPresent(Int.self, forKey: .outputProjectionDimensions)
        inputFeatureSize = try c.decodeIfPresent(Int.self, forKey: .inputFeatureSize) ?? 128
    }

    public var effectiveOutputDimensions: Int {
        outputProjectionDimensions ?? hiddenSize
    }
}

public struct Gemma4UnifiedAudioModelConfiguration: Decodable, Sendable {
    public let base: Gemma4UnifiedConfiguration
    public let audioEncoderConfiguration: Gemma4AudioEncoderConfiguration?

    enum CodingKeys: String, CodingKey {
        case audioConfiguration = "audio_config"
    }

    public init(from decoder: any Decoder) throws {
        base = try Gemma4UnifiedConfiguration(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        audioEncoderConfiguration = try c.decodeIfPresent(
            Gemma4AudioEncoderConfiguration.self,
            forKey: .audioConfiguration,
        )
    }
}

// MARK: - Audio Encoder Layers

private final class Gemma4AudioRMSNorm: Module, UnaryLayer {
    let eps: Float
    @ModuleInfo var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

private final class Gemma4AudioClippableLinear: Module, UnaryLayer {
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

private final class Gemma4AudioSSCPConvBlock: Module {
    let timeStride = 2

    @ModuleInfo var conv: Conv2d
    @ModuleInfo var norm: LayerNorm

    init(config: Gemma4AudioEncoderConfiguration, index: Int) {
        let channels = config.subsamplingConvChannels
        let inputChannels = index == 0 ? 1 : channels[index - 1]
        let outputChannels = channels[index]
        _conv.wrappedValue = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: .init((3, 3)),
            stride: .init((2, 2)),
            padding: .init((0, 0)),
            bias: false,
        )
        _norm.wrappedValue = LayerNorm(
            dimensions: outputChannels,
            eps: config.rmsNormEps,
            bias: false,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        var hidden = MLX.where(
            expandedDimensions(expandedDimensions(mask, axis: -1), axis: -1),
            MLXArray(0.0),
            x,
        )
        hidden = padded(hidden, widths: [0, .init((1, 1)), .init((1, 1)), 0])
        hidden = relu(norm(conv(hidden)))

        let sourceLength = mask.dim(1)
        let indices = MLXArray(
            Array(stride(from: 0, to: sourceLength, by: timeStride)).map(Int32.init),
        )
        var outputMask = take(mask, indices, axis: 1)
        if outputMask.dim(1) > hidden.dim(1) {
            outputMask = outputMask[0..., ..<hidden.dim(1)]
        }
        return (hidden, outputMask)
    }
}

private final class Gemma4AudioSubSampleConvProjection: Module {
    let inputFeatureSize: Int

    @ModuleInfo(key: "layer0") var layer0: Gemma4AudioSSCPConvBlock
    @ModuleInfo(key: "layer1") var layer1: Gemma4AudioSSCPConvBlock
    @ModuleInfo(key: "input_proj_linear") var inputProjectionLinear: Linear

    init(config: Gemma4AudioEncoderConfiguration) {
        inputFeatureSize = config.inputFeatureSize
        _layer0.wrappedValue = Gemma4AudioSSCPConvBlock(config: config, index: 0)
        _layer1.wrappedValue = Gemma4AudioSSCPConvBlock(config: config, index: 1)

        var frequency = config.inputFeatureSize
        for _ in 0 ..< 2 {
            frequency = (frequency + 2 - 3) / 2 + 1
        }
        let projectionInputDimensions = frequency * (config.subsamplingConvChannels.last ?? 32)
        _inputProjectionLinear.wrappedValue = Linear(
            projectionInputDimensions,
            config.hiddenSize,
            bias: false,
        )
        super.init()
    }

    func callAsFunction(_ audioMel: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        var x = expandedDimensions(audioMel, axis: -1)
        var currentMask = mask
        (x, currentMask) = layer0(x, mask: currentMask)
        (x, currentMask) = layer1(x, mask: currentMask)

        let batch = x.dim(0)
        let time = x.dim(1)
        let frequency = x.dim(2)
        let channels = x.dim(3)
        x = x.reshaped(batch, time, frequency * channels)
        return (inputProjectionLinear(x), currentMask)
    }
}

private final class Gemma4AudioConformerFeedForward: Module, UnaryLayer {
    let gradientClipping: Float
    let residualWeight: Float

    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: Gemma4AudioRMSNorm
    @ModuleInfo(key: "ffw_layer_1") var feedForwardLayer1: Gemma4AudioClippableLinear
    @ModuleInfo(key: "ffw_layer_2") var feedForwardLayer2: Gemma4AudioClippableLinear
    @ModuleInfo(key: "post_layer_norm") var postLayerNorm: Gemma4AudioRMSNorm

    init(config: Gemma4AudioEncoderConfiguration) {
        gradientClipping = config.gradientClipping
        residualWeight = config.residualWeight
        _preLayerNorm.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        _feedForwardLayer1.wrappedValue = Gemma4AudioClippableLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.hiddenSize * 4,
            bias: false,
            useClipping: config.useClippedLinears,
        )
        _feedForwardLayer2.wrappedValue = Gemma4AudioClippableLinear(
            inFeatures: config.hiddenSize * 4,
            outFeatures: config.hiddenSize,
            bias: false,
            useClipping: config.useClippedLinears,
        )
        _postLayerNorm.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var hidden = clip(x, min: -gradientClipping, max: gradientClipping)
        hidden = preLayerNorm(hidden)
        hidden = feedForwardLayer1(hidden)
        hidden = silu(hidden)
        hidden = feedForwardLayer2(hidden)
        hidden = clip(hidden, min: -gradientClipping, max: gradientClipping)
        hidden = postLayerNorm(hidden)
        return residual + hidden * residualWeight
    }
}

private struct Gemma4AudioRelativePositionEmbedding {
    let heads: Int
    let channels: Int
    let headDim: Int
    let maxBackward: Int
    let maxForward: Int
    let inverseTimescales: [Float]

    init(config: Gemma4AudioEncoderConfiguration) {
        heads = config.attentionHeads
        channels = config.hiddenSize
        headDim = config.hiddenSize / config.attentionHeads
        maxBackward = max(0, config.attentionContextLeft - 1)
        maxForward = config.attentionContextRight
        let timescales = max(1, config.hiddenSize / 2)
        let increment = log(10000.0) / Double(max(timescales - 1, 1))
        inverseTimescales = (0 ..< timescales).map { index in
            Float(exp(Double(index) * -increment))
        }
    }

    private func timingSignal(_ positions: MLXArray, dtype: DType) -> MLXArray {
        let scaledTime =
            expandedDimensions(positions.asType(.float32), axis: -1)
            * MLXArray(inverseTimescales, [1, 1, inverseTimescales.count])
        return concatenated([sin(scaledTime), cos(scaledTime)], axis: -1).asType(dtype)
    }

    private func relativeShift(
        _ term: MLXArray,
        batch: Int,
        heads: Int,
        blocks: Int,
        blockSize: Int,
        contextSize: Int,
        maxSpanPlusOne: Int,
    )
        -> MLXArray
    {
        let padAmount = (contextSize + 1) - maxSpanPlusOne
        var shifted = padded(term, widths: [0, 0, 0, 0, .init((0, padAmount))])
        shifted = shifted.reshaped(batch, heads, blocks, blockSize * (contextSize + 1))
        shifted = shifted[0..., 0..., 0..., ..<(blockSize * contextSize)]
        return shifted.reshaped(batch, heads, blocks, blockSize, contextSize)
    }

    func callAsFunction(
        queries: MLXArray,
        keys: MLXArray,
        positionProjection: Linear,
    )
        -> MLXArray
    {
        let batch = queries.dim(0)
        let blocks = queries.dim(1)
        let blockSize = queries.dim(2)
        let keyContext = keys.dim(2)

        let positions = MLXArray(
            Array(stride(from: maxBackward, through: -maxForward, by: -1)).map(Int32.init),
            [1, maxBackward + maxForward + 1],
        )
        let maxSpanPlusOne = positions.dim(1)
        var sinusoidal = timingSignal(positions, dtype: queries.dtype)
        sinusoidal = positionProjection(sinusoidal.asType(positionProjection.weight.dtype))
        sinusoidal = sinusoidal.reshaped(maxSpanPlusOne, heads, headDim).asType(queries.dtype)

        let queriesProjected = queries.transposed(0, 3, 1, 2, 4)
        let keysProjected = keys.transposed(0, 3, 1, 4, 2)
        let termAC = matmul(queriesProjected, keysProjected)

        let sinusoidalTransposed = sinusoidal.transposed(1, 2, 0)
        var termBD = matmul(
            queriesProjected.reshaped(batch, heads, blocks * blockSize, headDim),
            sinusoidalTransposed,
        )
        termBD = termBD.reshaped(batch, heads, blocks, blockSize, maxSpanPlusOne)
        termBD = relativeShift(
            termBD,
            batch: batch,
            heads: heads,
            blocks: blocks,
            blockSize: blockSize,
            contextSize: keyContext,
            maxSpanPlusOne: maxSpanPlusOne,
        )
        return termAC + termBD
    }
}

private final class Gemma4AudioAttention: Module {
    let heads: Int
    let hiddenSize: Int
    let headDim: Int
    let chunkSize: Int
    let maxFutureHorizon: Int
    let maxPastHorizon: Int
    let contextSize: Int
    let invalidLogitsValue: Float
    let softcap: Float
    let queryScale: Float
    let keyScale: Float

    @ModuleInfo(key: "relative_k_proj") var relativeKeyProjection: Linear
    @ModuleInfo(key: "per_dim_scale") var perDimensionScale: MLXArray
    @ModuleInfo(key: "q_proj") var queryProjection: Gemma4AudioClippableLinear
    @ModuleInfo(key: "k_proj") var keyProjection: Gemma4AudioClippableLinear
    @ModuleInfo(key: "v_proj") var valueProjection: Gemma4AudioClippableLinear
    @ModuleInfo var post: Gemma4AudioClippableLinear

    private let relativePositionEmbedding: Gemma4AudioRelativePositionEmbedding

    init(config: Gemma4AudioEncoderConfiguration) {
        heads = config.attentionHeads
        hiddenSize = config.hiddenSize
        headDim = config.hiddenSize / config.attentionHeads
        chunkSize = config.attentionChunkSize
        maxFutureHorizon = config.attentionContextRight
        maxPastHorizon = max(0, config.attentionContextLeft - 1)
        contextSize = chunkSize + maxPastHorizon + maxFutureHorizon
        invalidLogitsValue = config.attentionInvalidLogitsValue
        softcap = config.attentionLogitCap
        queryScale = pow(Float(headDim), -0.5) / log(2.0)
        keyScale = log(1.0 + Float(M_E)) / log(2.0)

        _relativeKeyProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.attentionHeads * headDim,
            bias: false,
        )
        _perDimensionScale.wrappedValue = MLXArray.zeros([headDim])
        _queryProjection.wrappedValue = Gemma4AudioClippableLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.attentionHeads * headDim,
            bias: false,
            useClipping: config.useClippedLinears,
        )
        _keyProjection.wrappedValue = Gemma4AudioClippableLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.attentionHeads * headDim,
            bias: false,
            useClipping: config.useClippedLinears,
        )
        _valueProjection.wrappedValue = Gemma4AudioClippableLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.attentionHeads * headDim,
            bias: false,
            useClipping: config.useClippedLinears,
        )
        _post.wrappedValue = Gemma4AudioClippableLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.hiddenSize,
            bias: false,
            useClipping: config.useClippedLinears,
        )
        relativePositionEmbedding = Gemma4AudioRelativePositionEmbedding(config: config)
        super.init()
    }

    private func padTime(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
        var widths = Array(repeating: IntOrPair(0), count: x.ndim)
        widths[1] = .init((left, right))
        return padded(x, widths: widths)
    }

    private func convertToBlock(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let time = x.dim(1)
        let rest = Array(x.shape.dropFirst(2))
        let blocks = (time + chunkSize - 1) / chunkSize
        let padLength = blocks * chunkSize - time
        let paddedX = padLength > 0 ? padTime(x, left: 0, right: padLength) : x
        return paddedX.reshaped([batch, blocks, chunkSize] + rest)
    }

    private func extractBlockContext(_ x: MLXArray) -> MLXArray {
        let padLeft = maxPastHorizon
        let padRight = maxFutureHorizon + chunkSize - 1
        let paddedX = padTime(x, left: padLeft, right: padRight)
        let paddedTime = paddedX.dim(1)
        let blocks = (paddedTime - contextSize) / chunkSize + 1
        let starts = arange(0, blocks, dtype: .int32) * Int32(chunkSize)
        let offsets = arange(0, contextSize, dtype: .int32)
        let indices = expandedDimensions(starts, axis: -1) + expandedDimensions(offsets, axis: 0)
        return take(paddedX, indices, axis: 1)
    }

    func callAsFunction(
        hiddenStates: MLXArray,
        mask: MLXArray,
        causalValidMask: MLXArray,
    )
        -> MLXArray
    {
        let batch = hiddenStates.dim(0)
        let time = hiddenStates.dim(1)
        let qkvShape = [batch, time, heads, headDim]

        var query = queryProjection(hiddenStates).asType(.float32).reshaped(qkvShape)
        var key = keyProjection(hiddenStates).asType(.float32).reshaped(qkvShape)
        let value = valueProjection(hiddenStates).asType(.float32).reshaped(qkvShape)

        let perDimension = softplus(perDimensionScale)
        query = query * (queryScale * perDimension)
        key = key * keyScale

        let queryBlocks = convertToBlock(query)
        let keyBlocks = extractBlockContext(key)
        let valueBlocks = extractBlockContext(value)
        let blocks = queryBlocks.dim(1)

        let extractedValid = extractBlockContext(.!mask)
        let condition =
            expandedDimensions(expandedDimensions(extractedValid, axis: 1), axis: 3)
            .&& causalValidMask[.newAxis, .newAxis, .newAxis, 0..., 0...]

        var logits = relativePositionEmbedding(
            queries: queryBlocks,
            keys: keyBlocks,
            positionProjection: relativeKeyProjection,
        )
        logits = tanh(logits / softcap) * softcap
        logits = MLX.where(condition, logits, MLXArray(invalidLogitsValue))

        let probabilities = softmax(logits, axis: -1)
        var context = einsum("bnuwc,bucnh->buwnh", probabilities, valueBlocks)
        context = context.reshaped(batch, blocks * chunkSize, heads, headDim)
        context = context[0..., ..<time, 0..., 0...]
        context = context.reshaped(batch, time, heads * headDim)
        return post(context)
    }
}

private final class Gemma4AudioConformerLightConv1d: Module, UnaryLayer {
    let gradientClipping: Float
    let causalPadding: Int

    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: Gemma4AudioRMSNorm
    @ModuleInfo(key: "linear_start") var linearStart: Gemma4AudioClippableLinear
    @ModuleInfo(key: "depthwise_conv1d") var depthwiseConv1d: Conv1d
    @ModuleInfo(key: "conv_norm") var convNorm: Gemma4AudioRMSNorm
    @ModuleInfo(key: "linear_end") var linearEnd: Gemma4AudioClippableLinear

    init(config: Gemma4AudioEncoderConfiguration) {
        gradientClipping = config.gradientClipping
        causalPadding = config.convKernelSize - 1
        _preLayerNorm.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        _linearStart.wrappedValue = Gemma4AudioClippableLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.hiddenSize * 2,
            bias: false,
            useClipping: config.useClippedLinears,
        )
        _depthwiseConv1d.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: config.convKernelSize,
            padding: 0,
            groups: config.hiddenSize,
            bias: false,
        )
        _convNorm.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        _linearEnd.wrappedValue = Gemma4AudioClippableLinear(
            inFeatures: config.hiddenSize,
            outFeatures: config.hiddenSize,
            bias: false,
            useClipping: config.useClippedLinears,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var hidden = preLayerNorm(x)
        hidden = linearStart(hidden)
        let parts = split(hidden, parts: 2, axis: -1)
        hidden = parts[0] * sigmoid(parts[1])
        hidden = padded(hidden, widths: [0, .init((causalPadding, 0)), 0])
        hidden = depthwiseConv1d(hidden)
        hidden = clip(hidden, min: -gradientClipping, max: gradientClipping)
        hidden = convNorm(hidden)
        hidden = silu(hidden)
        hidden = linearEnd(hidden)
        return hidden + residual
    }
}

private final class Gemma4AudioConformerBlock: Module {
    let gradientClipping: Float

    @ModuleInfo(key: "feed_forward1") var feedForward1: Gemma4AudioConformerFeedForward
    @ModuleInfo(key: "self_attn") var selfAttention: Gemma4AudioAttention
    @ModuleInfo(key: "lconv1d") var lightConv1d: Gemma4AudioConformerLightConv1d
    @ModuleInfo(key: "feed_forward2") var feedForward2: Gemma4AudioConformerFeedForward
    @ModuleInfo(key: "norm_pre_attn") var normPreAttention: Gemma4AudioRMSNorm
    @ModuleInfo(key: "norm_post_attn") var normPostAttention: Gemma4AudioRMSNorm
    @ModuleInfo(key: "norm_out") var normOut: Gemma4AudioRMSNorm

    init(config: Gemma4AudioEncoderConfiguration) {
        gradientClipping = config.gradientClipping
        _feedForward1.wrappedValue = Gemma4AudioConformerFeedForward(config: config)
        _selfAttention.wrappedValue = Gemma4AudioAttention(config: config)
        _lightConv1d.wrappedValue = Gemma4AudioConformerLightConv1d(config: config)
        _feedForward2.wrappedValue = Gemma4AudioConformerFeedForward(config: config)
        _normPreAttention.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        _normPostAttention.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        _normOut.wrappedValue = Gemma4AudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, causalValidMask: MLXArray) -> MLXArray {
        var hidden = feedForward1(x)

        let residual = hidden
        hidden = clip(hidden, min: -gradientClipping, max: gradientClipping)
        hidden = normPreAttention(hidden)
        hidden = selfAttention(
            hiddenStates: hidden,
            mask: mask,
            causalValidMask: causalValidMask,
        )
        hidden = clip(hidden, min: -gradientClipping, max: gradientClipping)
        hidden = residual + normPostAttention(hidden)

        hidden = hidden * expandedDimensions((.!mask).asType(hidden.dtype), axis: -1)
        hidden = lightConv1d(hidden)
        hidden = feedForward2(hidden)
        hidden = clip(hidden, min: -gradientClipping, max: gradientClipping)
        return normOut(hidden)
    }
}

public final class Gemma4AudioEncoder: Module {
    public let config: Gemma4AudioEncoderConfiguration
    public var inputFeatureSize: Int {
        config.inputFeatureSize
    }

    @ModuleInfo(key: "subsample_conv_projection") private var subsampleConvProjection:
        Gemma4AudioSubSampleConvProjection
    @ModuleInfo private var layers: [Gemma4AudioConformerBlock]
    @ModuleInfo(key: "output_proj") private var outputProjection: Linear?

    public init(config: Gemma4AudioEncoderConfiguration) {
        self.config = config
        _subsampleConvProjection.wrappedValue = Gemma4AudioSubSampleConvProjection(
            config: config,
        )
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            Gemma4AudioConformerBlock(config: config)
        }
        if let outputProjectionDimensions = config.outputProjectionDimensions {
            _outputProjection.wrappedValue = Linear(
                config.hiddenSize,
                outputProjectionDimensions,
                bias: true,
            )
        }
        super.init()
    }

    static func sanitizeAudioTowerWeights(
        _ weights: [String: MLXArray],
        config: Gemma4AudioEncoderConfiguration,
    )
        -> [String: MLXArray]
    {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (key, value) in weights {
            if !config.useClippedLinears,
                key.contains("input_min")
                    || key.contains("input_max")
                    || key.contains("output_min")
                    || key.contains("output_max")
            {
                continue
            }

            var tensor = value
            if key.contains("subsample_conv_projection"),
                key.contains("conv.weight"),
                tensor.ndim == 4
            {
                let inputChannels = key.contains("layer0.") ? 1 : config.subsamplingConvChannels[0]
                if tensor.dim(-1) != inputChannels {
                    tensor = tensor.transposed(0, 2, 3, 1)
                }
            }
            if key.contains("depthwise_conv1d.weight"), tensor.ndim == 3, tensor.dim(-1) != 1 {
                tensor = tensor.transposed(0, 2, 1)
            }
            sanitized[key] = tensor
        }

        return sanitized
    }

    private func buildCausalValidMask() -> MLXArray {
        let chunkSize = config.attentionChunkSize
        let maxFutureHorizon = config.attentionContextRight
        let maxPastHorizon = max(0, config.attentionContextLeft - 1)
        let upperDiagonal = maxPastHorizon + maxFutureHorizon
        let contextSize = chunkSize + maxPastHorizon + maxFutureHorizon

        let lowerCausal = tril(ones([contextSize, chunkSize], dtype: .float32)).T
        let upperCausal = tril(ones([chunkSize, contextSize], dtype: .float32), k: upperDiagonal)
        return (lowerCausal * upperCausal).asType(.bool)
    }

    public func callAsFunction(_ audioMel: MLXArray, mask: MLXArray) -> (
        MLXArray, MLXArray,
    ) {
        callAsFunction(audioMel, mask: mask, causalValidMask: buildCausalValidMask())
    }

    /// Encoder variants can supply their checkpoint's exact attention context bounds.
    func callAsFunction(_ audioMel: MLXArray, mask: MLXArray, causalValidMask: MLXArray) -> (
        MLXArray, MLXArray,
    ) {
        var (audioEncodings, currentMask) = subsampleConvProjection(audioMel, mask: mask)

        var previousOutput: MLXArray?
        for layer in layers {
            audioEncodings = layer(
                audioEncodings,
                mask: currentMask,
                causalValidMask: causalValidMask,
            )
            // Keep at most two inference blocks in flight. Submit the next
            // block before waiting so graph construction overlaps GPU work,
            // while completed blocks release their temporary projections.
            if !training {
                asyncEval(audioEncodings)
                if let previousOutput { eval(previousOutput) }
                previousOutput = audioEncodings
            }
        }

        if let outputProjection {
            audioEncodings = outputProjection(audioEncodings)
        }

        if currentMask.dim(1) > audioEncodings.dim(1) {
            currentMask = currentMask[0..., ..<audioEncodings.dim(1)]
        }
        audioEncodings = MLX.where(
            expandedDimensions(currentMask, axis: -1),
            MLXArray(0.0),
            audioEncodings,
        )
        return (audioEncodings, currentMask)
    }
}

// MARK: - Gemma 4 Unified Wrapper

public final class Gemma4UnifiedAudioModel: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo var base: Gemma4Unified
    @ModuleInfo(key: "audio_tower") private var audioTower: Gemma4AudioEncoder?

    public let config: Gemma4UnifiedAudioModelConfiguration

    public var vocabularySize: Int {
        base.vocabularySize
    }

    public var kvHeads: [Int] {
        base.kvHeads
    }

    public var loraLayers: [Module] {
        base.loraLayers
    }

    public init(_ config: Gemma4UnifiedAudioModelConfiguration) {
        self.config = config
        _base.wrappedValue = Gemma4Unified(config.base)
        super.init()
    }

    public var toolCallFormat: ToolCallFormat? { base.toolCallFormat }
    public var reasoningConfig: ReasoningConfig? { base.reasoningConfig }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [any KVCache] {
        try base.newCache(configuration: configuration)
    }

    private func createAudioTowerIfNeeded() -> Gemma4AudioEncoder? {
        if audioTower == nil, let audioConfig = config.audioEncoderConfiguration {
            _audioTower.wrappedValue = Gemma4AudioEncoder(config: audioConfig)
        }
        return audioTower
    }

    private func encodeAudioIfNeeded(_ audio: LMInput.ProcessedAudio) throws
        -> LMInput.ProcessedAudio
    {
        guard let tower = audioTower, audio.features.dim(-1) == tower.inputFeatureSize else {
            if let expected = config.base.audioConfiguration?.outputProjectionDimensions,
                audio.features.dim(-1) == expected
            {
                return audio
            }
            throw VLMError.processing(
                "Gemma4 audio features require loaded audio_tower weights before inference.",
            )
        }

        let validMask =
            if let mask = audio.mask {
                mask.asType(.bool)
            } else {
                ones([audio.features.dim(0), audio.features.dim(1)], dtype: .bool)
            }
        let invalidMask = .!validMask
        let (encoded, invalidOutputMask) = tower(audio.features, mask: invalidMask)
        return LMInput.ProcessedAudio(features: encoded, mask: .!invalidOutputMask)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        guard let audio = input.audio else {
            return try base.prepare(input, cache: cache, state: state, prefill: prefill)
        }

        let encodedAudio = try encodeAudioIfNeeded(audio)
        let encodedInput = LMInput(
            text: input.text,
            image: input.image,
            video: input.video,
            audio: encodedAudio,
        )
        return try base.prepare(encodedInput, cache: cache, state: state, prefill: prefill)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        base(inputs, cache: cache)
    }

    public func callAsFunction(
        _ input: LMInput.Text,
        cache: [any KVCache]?,
        state: LMOutput.State?,
    )
        -> LMOutput
    {
        base(input, cache: cache, state: state)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var baseWeights: [String: MLXArray] = [:]
        var audioWeights: [String: MLXArray] = [:]
        var sanitized: [String: MLXArray] = [:]
        baseWeights.reserveCapacity(weights.count)
        audioWeights.reserveCapacity(weights.count)

        for (key, value) in weights {
            if key.hasPrefix("base.") {
                sanitized[key] = value
                continue
            }
            if key.hasPrefix("audio_tower.") {
                _ = createAudioTowerIfNeeded()
                sanitized[key] = value
                continue
            }

            let rootKey =
                key.hasPrefix("model.")
                ? String(key.dropFirst("model.".count))
                : key
            if rootKey.hasPrefix("audio_tower.") {
                audioWeights[String(rootKey.dropFirst("audio_tower.".count))] = value
            } else {
                baseWeights[key] = value
            }
        }

        for (key, value) in base.sanitize(weights: baseWeights) {
            sanitized["base.\(key)"] = value
        }

        if !audioWeights.isEmpty {
            let tower = createAudioTowerIfNeeded()
            let audioConfig = tower?.config ?? config.audioEncoderConfiguration ?? .init()
            for (key, value) in Gemma4AudioEncoder.sanitizeAudioTowerWeights(
                audioWeights, config: audioConfig)
            {
                sanitized["audio_tower.\(key)"] = value
            }
        }

        return sanitized
    }

}
