import Foundation
import MLX
import MLXNN

public enum Flux2Error: LocalizedError {
    case invalidDimensions(width: Int, height: Int)
    case invalidStepCount(Int)
    case emptyPrompt
    case missingWeights(URL)
    case invalidPromptTokenShape

    public var errorDescription: String? {
        switch self {
        case .invalidDimensions(let width, let height):
            "Flux2 image dimensions must be in [256, 2048] and multiples of 16; got \(width)x\(height)."
        case .invalidStepCount(let steps):
            "Flux2 image generation requires at least one denoising step; got \(steps)."
        case .emptyPrompt:
            "Flux2 prompt must not be empty."
        case .missingWeights(let url):
            "Missing Flux2 safetensors under \(url.path)."
        case .invalidPromptTokenShape:
            "Flux2 prompt tokens and attention mask must have the same 2D shape."
        }
    }
}

public enum Flux2Precision {
    public static let dtype: DType = .bfloat16
}

public struct Flux2TransformerConfiguration: Sendable, Equatable {
    public let patchSize: Int
    public let inChannels: Int
    public let outChannels: Int?
    public let numLayers: Int
    public let numSingleLayers: Int
    public let attentionHeadDim: Int
    public let numAttentionHeads: Int
    public let jointAttentionDim: Int
    public let timestepGuidanceChannels: Int
    public let mlpRatio: Float
    public let axesDimsRoPE: [Int]
    public let ropeTheta: Float
    public let guidanceEmbeds: Bool

    public init(
        patchSize: Int = 1,
        inChannels: Int = 128,
        outChannels: Int? = nil,
        numLayers: Int = 5,
        numSingleLayers: Int = 20,
        attentionHeadDim: Int = 128,
        numAttentionHeads: Int = 24,
        jointAttentionDim: Int = 7680,
        timestepGuidanceChannels: Int = 256,
        mlpRatio: Float = 3.0,
        axesDimsRoPE: [Int] = [32, 32, 32, 32],
        ropeTheta: Float = 2000,
        guidanceEmbeds: Bool = false,
    ) {
        self.patchSize = patchSize
        self.inChannels = inChannels
        self.outChannels = outChannels
        self.numLayers = numLayers
        self.numSingleLayers = numSingleLayers
        self.attentionHeadDim = attentionHeadDim
        self.numAttentionHeads = numAttentionHeads
        self.jointAttentionDim = jointAttentionDim
        self.timestepGuidanceChannels = timestepGuidanceChannels
        self.mlpRatio = mlpRatio
        self.axesDimsRoPE = axesDimsRoPE
        self.ropeTheta = ropeTheta
        self.guidanceEmbeds = guidanceEmbeds
    }
}

public struct Flux2TextEncoderConfiguration: Sendable, Equatable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let intermediateSize: Int
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Float
    public let rmsNormEps: Float
    public let headDim: Int
    public let attentionBias: Bool
    public let attentionScaling: Float

    public init(
        vocabSize: Int = 151_936,
        hiddenSize: Int = 2560,
        numHiddenLayers: Int = 36,
        numAttentionHeads: Int = 32,
        numKeyValueHeads: Int = 8,
        intermediateSize: Int = 9728,
        maxPositionEmbeddings: Int = 40960,
        ropeTheta: Float = 1_000_000,
        rmsNormEps: Float = 1e-6,
        headDim: Int = 128,
        attentionBias: Bool = false,
        attentionScaling: Float = 1.0,
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.intermediateSize = intermediateSize
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.rmsNormEps = rmsNormEps
        self.headDim = headDim
        self.attentionBias = attentionBias
        self.attentionScaling = attentionScaling
    }
}

public enum Flux2Dimensions {
    public static func validate(width: Int, height: Int) throws {
        guard (256 ... 2048).contains(width),
            (256 ... 2048).contains(height),
            width.isMultiple(of: 16),
            height.isMultiple(of: 16)
        else {
            throw Flux2Error.invalidDimensions(width: width, height: height)
        }
    }
}

public enum Flux2Latents {
    public static func patchify(_ latents: MLXArray) -> MLXArray {
        var latents = latents
        if latents.ndim == 5, latents.dim(2) == 1 {
            latents = latents[0..., 0..., 0, 0..., 0...]
        }
        let batch = latents.dim(0)
        let channels = latents.dim(1)
        let height = latents.dim(2)
        let width = latents.dim(3)
        latents = latents.reshaped(batch, channels, height / 2, 2, width / 2, 2)
        latents = latents.transposed(0, 1, 3, 5, 2, 4)
        return latents.reshaped(batch, channels * 4, height / 2, width / 2)
    }

    public static func pack(_ latents: MLXArray) -> MLXArray {
        let batch = latents.dim(0)
        let channels = latents.dim(1)
        let height = latents.dim(2)
        let width = latents.dim(3)
        return latents.reshaped(batch, channels, height * width).transposed(0, 2, 1)
    }

    public static func unpack(_ latents: MLXArray, latentHeight: Int, latentWidth: Int) -> MLXArray
    {
        if latents.ndim == 4 {
            return latents
        }
        return latents.reshaped(latents.dim(0), latentHeight, latentWidth, latents.dim(-1))
            .transposed(0, 3, 1, 2)
    }

    public static func preparePackedLatents(
        seed: UInt64,
        height: Int,
        width: Int,
        batchSize: Int = 1,
        numLatentChannels: Int = 32,
        vaeScaleFactor: Int = 8,
    )
        -> (latents: MLXArray, ids: MLXArray, latentHeight: Int, latentWidth: Int)
    {
        let scaledHeight = 2 * (height / (vaeScaleFactor * 2))
        let scaledWidth = 2 * (width / (vaeScaleFactor * 2))
        let latentHeight = scaledHeight / 2
        let latentWidth = scaledWidth / 2
        let latents = MLXRandom.normal(
            [batchSize, numLatentChannels * 4, latentHeight, latentWidth],
            key: MLXRandom.key(seed),
        ).asType(Flux2Precision.dtype)
        let ids = prepareGridIDs(latents, tCoordinate: 0)
        return (pack(latents), ids, latentHeight, latentWidth)
    }

    public static func prepareGridIDs(_ latents: MLXArray, tCoordinate: Int) -> MLXArray {
        let batch = latents.dim(0)
        let height = latents.dim(2)
        let width = latents.dim(3)
        let hIDs = broadcast(
            expandedDimensions(arange(height, dtype: .int32), axis: 1), to: [height, width])
        let wIDs = broadcast(
            expandedDimensions(arange(width, dtype: .int32), axis: 0), to: [height, width])
        let flatH = hIDs.reshaped(-1)
        let flatW = wIDs.reshaped(-1)
        let t = MLXArray.zeros(flatH.shape, dtype: .int32) + Int32(tCoordinate)
        let layerIDs = MLXArray.zeros(flatH.shape, dtype: .int32)
        let coords = stacked([t, flatH, flatW, layerIDs], axis: 1).expandedDimensions(axis: 0)
        return broadcast(coords, to: [batch, coords.dim(1), coords.dim(2)])
    }
}

public struct Flux2FlowMatchEulerDiscreteScheduler {
    public let timesteps: MLXArray
    public let sigmas: MLXArray

    public init(imageSequenceLength: Int, inferenceSteps: Int) {
        let values = Self.timestepsAndSigmas(
            imageSequenceLength: imageSequenceLength,
            inferenceSteps: inferenceSteps,
        )
        timesteps = values.timesteps
        sigmas = values.sigmas
    }

    public static func empiricalMu(imageSequenceLength: Int, steps: Int) -> Float {
        let a1: Float = 8.73809524e-05
        let b1: Float = 1.89833333
        let a2: Float = 0.00016927
        let b2: Float = 0.45666666
        if imageSequenceLength > 4300 {
            return a2 * Float(imageSequenceLength) + b2
        }
        let m200 = a2 * Float(imageSequenceLength) + b2
        let m10 = a1 * Float(imageSequenceLength) + b1
        let a = (m200 - m10) / 190.0
        let b = m200 - 200.0 * a
        return a * Float(steps) + b
    }

    public static func timestepsAndSigmas(
        imageSequenceLength: Int,
        inferenceSteps: Int,
        trainTimesteps: Int = 1000,
    )
        -> (timesteps: MLXArray, sigmas: MLXArray)
    {
        var sigmas = linspace(Float(1.0), Float(1.0 / Float(inferenceSteps)), count: inferenceSteps)
            .asType(.float32)
        let mu = empiricalMu(imageSequenceLength: imageSequenceLength, steps: inferenceSteps)
        sigmas = exp(MLXArray(mu)) / (exp(MLXArray(mu)) + (1.0 / sigmas - 1.0))
        let timesteps = sigmas * Float(trainTimesteps)
        sigmas = concatenated([sigmas, MLXArray.zeros([1], dtype: sigmas.dtype)], axis: 0)
        return (timesteps, sigmas)
    }

    public func step(noise: MLXArray, stepIndex: Int, latents: MLXArray) -> MLXArray {
        let dt = (sigmas[stepIndex + 1] - sigmas[stepIndex]).asType(latents.dtype)
        return latents + dt * noise.asType(latents.dtype)
    }
}

public struct Flux2PromptTokens {
    public let inputIDs: MLXArray
    public let attentionMask: MLXArray

    public init(inputIDs: MLXArray, attentionMask: MLXArray) {
        self.inputIDs = inputIDs
        self.attentionMask = attentionMask
    }
}

public enum Flux2PromptTokenizer {
    public static let defaultBuckets = [32, 64, 128, 256, 512]

    public static func tokenize(
        prompt: String,
        tokenizer: any Tokenizer,
        maxLength: Int = 512,
        bucketed: Bool = true,
    )
        throws -> Flux2PromptTokens
    {
        guard !prompt.isEmpty else {
            throw Flux2Error.emptyPrompt
        }
        let tokenIDs = try formattedPromptTokens(prompt, tokenizer: tokenizer)
        let effectiveMax =
            bucketed ? pickBucket(trueLength: tokenIDs.count, cap: maxLength) : maxLength
        let clipped = Array(tokenIDs.prefix(effectiveMax))
        let padCount = max(effectiveMax - clipped.count, 0)
        let paddedIDs = clipped + Array(repeating: 0, count: padCount)
        let mask =
            Array(repeating: Int32(1), count: clipped.count)
            + Array(repeating: Int32(0), count: padCount)
        return Flux2PromptTokens(
            inputIDs: MLXArray(paddedIDs.map(Int32.init), [1, effectiveMax]),
            attentionMask: MLXArray(mask, [1, effectiveMax]),
        )
    }

    public static func countTokens(prompt: String, tokenizer: any Tokenizer) throws
        -> Int
    {
        if prompt.isEmpty {
            return 0
        }
        return try formattedPromptTokens(prompt, tokenizer: tokenizer).count
    }

    private static func formattedPromptTokens(
        _ prompt: String, tokenizer: any Tokenizer
    ) throws -> [Int] {
        do {
            let messages: [[String: any Sendable]] = [
                ["role": "user", "content": prompt]
            ]
            return try tokenizer.applyChatTemplate(
                messages: messages,
                tools: [[String: any Sendable]]?.none,
                additionalContext: ["enable_thinking": false, "add_generation_prompt": true],
            )
        } catch {
            return tokenizer.encode(text: prompt, addSpecialTokens: true)
        }
    }

    private static func pickBucket(trueLength: Int, cap: Int) -> Int {
        for bucket in defaultBuckets where bucket <= cap {
            if bucket >= trueLength {
                return bucket
            }
        }
        return cap
    }
}

public enum Flux2PromptIDs {
    public static func prepareTextIDs(for promptEmbeddings: MLXArray) -> MLXArray {
        let batch = promptEmbeddings.dim(0)
        let seqLen = promptEmbeddings.dim(1)
        let t = MLXArray.zeros([seqLen], dtype: .int32)
        let h = MLXArray.zeros([seqLen], dtype: .int32)
        let w = MLXArray.zeros([seqLen], dtype: .int32)
        let tokenIDs = arange(seqLen, dtype: .int32)
        let one = stacked([t, h, w, tokenIDs], axis: 1).expandedDimensions(axis: 0)
        return broadcast(one, to: [batch, seqLen, 4])
    }
}

private final class Flux2RMSNorm: Module, UnaryLayer {
    @ModuleInfo var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        _weight.wrappedValue = MLXArray.ones([dimensions])
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let hidden = x.asType(.float32)
        let variance = mean(hidden * hidden, axis: -1, keepDims: true)
        return (weight.asType(.float32) * hidden * rsqrt(variance + eps)).asType(dtype)
    }
}

private final class Flux2QwenRotaryEmbedding: Module {
    let dim: Int
    let base: Float
    let scalingFactor: Float
    private let invFreqValues: [Float]

    init(dim: Int, base: Float = 1_000_000, scalingFactor: Float = 1.0) {
        self.dim = dim
        self.base = base
        self.scalingFactor = scalingFactor
        invFreqValues = stride(from: 0, to: dim, by: 2).map { index in
            1.0 / Foundation.pow(base, Float(index) / Float(dim))
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, positionIDs: MLXArray) -> (MLXArray, MLXArray) {
        var positions = positionIDs
        if positions.ndim == 1 {
            positions = positions.expandedDimensions(axis: 0)
        }
        let invFreq = MLXArray(invFreqValues)
        let inv = invFreq.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        let pos = positions.asType(.float32).expandedDimensions(axis: -1)
        let freqs = pos * inv
        let embeddings = concatenated([freqs, freqs], axis: -1)
        let cosines = cos(embeddings) * scalingFactor
        let sines = sin(embeddings) * scalingFactor
        return (cosines.asType(x.dtype), sines.asType(x.dtype))
    }
}

private enum Flux2AttentionMath {
    static func repeatKV(_ hidden: MLXArray, repeats: Int) -> MLXArray {
        let batch = hidden.dim(0)
        let heads = hidden.dim(1)
        let seqLen = hidden.dim(2)
        let headDim = hidden.dim(3)
        let expanded = hidden.expandedDimensions(axis: 2)
        return broadcast(expanded, to: [batch, heads, repeats, seqLen, headDim])
            .reshaped(batch, heads * repeats, seqLen, headDim)
    }

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, ..<half], x[.ellipsis, half...]], axis: -1)
    }

    static func applyQwenRoPE(
        queries: MLXArray,
        keys: MLXArray,
        cosines: MLXArray,
        sines: MLXArray,
    )
        -> (MLXArray, MLXArray)
    {
        let cosValue = cosines.expandedDimensions(axis: 1)
        let sinValue = sines.expandedDimensions(axis: 1)
        return (
            queries * cosValue + rotateHalf(queries) * sinValue,
            keys * cosValue + rotateHalf(keys) * sinValue,
        )
    }

    static func applyFluxRoPE(
        queries: MLXArray,
        keys: MLXArray,
        cosines: MLXArray,
        sines: MLXArray,
    )
        -> (MLXArray, MLXArray)
    {
        let dtype = queries.dtype
        let cosValue = cosines.reshaped(1, 1, cosines.dim(0), cosines.dim(1))
        let sinValue = sines.reshaped(1, 1, sines.dim(0), sines.dim(1))

        func mix(_ x: MLXArray) -> MLXArray {
            let x = x.asType(.float32)
            let x2 = x.reshaped(x.shape.dropLast() + [-1, 2])
            let real = x2[.ellipsis, 0]
            let imaginary = x2[.ellipsis, 1]
            let out0 = real * cosValue - imaginary * sinValue
            let out1 = imaginary * cosValue + real * sinValue
            return stacked([out0, out1], axis: -1).reshaped(x.shape)
        }

        return (mix(queries).asType(dtype), mix(keys).asType(dtype))
    }

    static func scaledAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        batch: Int,
        heads: Int,
        headDim: Int,
        mask: MLXArray? = nil,
    )
        -> MLXArray
    {
        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: 1.0 / sqrt(Float(headDim)),
            mask: mask,
        )
        return output.transposed(0, 2, 1, 3).reshaped(batch, -1, heads * headDim)
    }
}

private final class Flux2QwenMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(hiddenSize: Int, intermediateSize: Int) {
        _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        downProj(silu(gateProj(hidden)) * upProj(hidden))
    }
}

private final class Flux2QwenAttention: Module {
    let numAttentionHeads: Int
    let numKeyValueHeads: Int
    let numKeyValueGroups: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Flux2RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Flux2RMSNorm

    init(_ config: Flux2TextEncoderConfiguration) {
        numAttentionHeads = config.numAttentionHeads
        numKeyValueHeads = config.numKeyValueHeads
        numKeyValueGroups = config.numAttentionHeads / config.numKeyValueHeads
        headDim = config.headDim
        scale = 1.0 / sqrt(Float(config.headDim))
        _qProj.wrappedValue = Linear(
            config.hiddenSize,
            config.numAttentionHeads * config.headDim,
            bias: config.attentionBias,
        )
        _kProj.wrappedValue = Linear(
            config.hiddenSize,
            config.numKeyValueHeads * config.headDim,
            bias: config.attentionBias,
        )
        _vProj.wrappedValue = Linear(
            config.hiddenSize,
            config.numKeyValueHeads * config.headDim,
            bias: config.attentionBias,
        )
        _oProj.wrappedValue = Linear(
            config.numAttentionHeads * config.headDim,
            config.hiddenSize,
            bias: config.attentionBias,
        )
        _qNorm.wrappedValue = Flux2RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = Flux2RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        hidden: MLXArray,
        attentionMask: MLXArray?,
        positionEmbeddings: (MLXArray, MLXArray),
    )
        -> MLXArray
    {
        let batch = hidden.dim(0)
        let seqLen = hidden.dim(1)

        var queries = qProj(hidden).reshaped(batch, seqLen, numAttentionHeads, headDim)
        var keys = kProj(hidden).reshaped(batch, seqLen, numKeyValueHeads, headDim)
        var values = vProj(hidden).reshaped(batch, seqLen, numKeyValueHeads, headDim)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys).transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let rope = Flux2AttentionMath.applyQwenRoPE(
            queries: queries,
            keys: keys,
            cosines: positionEmbeddings.0,
            sines: positionEmbeddings.1,
        )
        queries = rope.0
        keys = rope.1

        if numKeyValueHeads != numAttentionHeads {
            keys = Flux2AttentionMath.repeatKV(keys, repeats: numKeyValueGroups)
            values = Flux2AttentionMath.repeatKV(values, repeats: numKeyValueGroups)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries.asType(.float32),
            keys: keys.asType(.float32),
            values: values.asType(.float32),
            scale: scale,
            mask: attentionMask,
        )
        return oProj(
            output.asType(queries.dtype)
                .transposed(0, 2, 1, 3)
                .reshaped(batch, seqLen, numAttentionHeads * headDim),
        )
    }
}

private final class Flux2QwenDecoderLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Flux2RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttention: Flux2QwenAttention
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Flux2RMSNorm
    @ModuleInfo var mlp: Flux2QwenMLP

    init(_ config: Flux2TextEncoderConfiguration) {
        _inputLayerNorm.wrappedValue = Flux2RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        _selfAttention.wrappedValue = Flux2QwenAttention(config)
        _postAttentionLayerNorm.wrappedValue = Flux2RMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        _mlp.wrappedValue = Flux2QwenMLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
        )
        super.init()
    }

    func callAsFunction(
        _ hidden: MLXArray,
        attentionMask: MLXArray?,
        positionEmbeddings: (MLXArray, MLXArray),
    )
        -> MLXArray
    {
        var hidden =
            hidden
            + selfAttention(
                hidden: inputLayerNorm(hidden),
                attentionMask: attentionMask,
                positionEmbeddings: positionEmbeddings,
            )
        hidden = hidden + mlp(postAttentionLayerNorm(hidden))
        return hidden
    }
}

public final class Flux2TextEncoder: Module {
    public let config: Flux2TextEncoderConfiguration

    @ModuleInfo(key: "embed_tokens") fileprivate var embedTokens: Embedding
    @ModuleInfo fileprivate var layers: [Flux2QwenDecoderLayer]
    @ModuleInfo fileprivate var norm: Flux2RMSNorm
    @ModuleInfo(key: "rotary_emb") fileprivate var rotaryEmbedding: Flux2QwenRotaryEmbedding

    public init(_ config: Flux2TextEncoderConfiguration = Flux2TextEncoderConfiguration()) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabSize,
            dimensions: config.hiddenSize,
        )
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            Flux2QwenDecoderLayer(config)
        }
        _norm.wrappedValue = Flux2RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _rotaryEmbedding.wrappedValue = Flux2QwenRotaryEmbedding(
            dim: config.headDim,
            base: config.ropeTheta,
            scalingFactor: config.attentionScaling,
        )
        super.init()
    }

    public func callAsFunction(
        inputIDs: MLXArray,
        attentionMask: MLXArray? = nil,
        outputHiddenStates: Bool = false,
    )
        throws -> (MLXArray, [MLXArray]?)
    {
        let batch = inputIDs.dim(0)
        let seqLen = inputIDs.dim(1)
        var hidden = embedTokens(inputIDs)

        let maskInput = attentionMask ?? MLXArray.ones([batch, seqLen], dtype: .int32)
        let maskDType = hidden.dtype
        let paddingMask = MLX.where(
            maskInput .== 1,
            MLXArray.zeros(maskInput.shape, dtype: maskDType),
            MLXArray.zeros(maskInput.shape, dtype: maskDType) - Float.infinity,
        ).expandedDimensions(axis: 1).expandedDimensions(axis: 1)

        var causalMask: MLXArray
        if seqLen == 1 {
            causalMask = MLXArray.zeros([batch, 1, 1, 1], dtype: maskDType)
        } else {
            let idx = arange(seqLen, dtype: .int32)
            let j = idx.expandedDimensions(axis: 0)
            let i = idx.expandedDimensions(axis: 1)
            let tri = j .> i
            causalMask = MLX.where(
                tri,
                MLXArray.zeros([seqLen, seqLen], dtype: maskDType) - Float.infinity,
                MLXArray.zeros([seqLen, seqLen], dtype: maskDType),
            )
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)
            causalMask = broadcast(causalMask, to: [batch, 1, seqLen, seqLen])
        }
        let attentionMask4D = causalMask + paddingMask

        let positionIDs = broadcast(
            arange(seqLen, dtype: .int32).expandedDimensions(axis: 0),
            to: [batch, seqLen],
        )
        let positionEmbeddings = rotaryEmbedding(hidden, positionIDs: positionIDs)

        var hiddenStates: [MLXArray]? = outputHiddenStates ? [hidden] : nil
        for layer in layers {
            hidden = layer(
                hidden, attentionMask: attentionMask4D, positionEmbeddings: positionEmbeddings)
            if outputHiddenStates {
                hiddenStates?.append(hidden)
            }
        }
        hidden = norm(hidden)
        return (hidden, hiddenStates)
    }

    public func promptEmbeddings(
        inputIDs: MLXArray,
        attentionMask: MLXArray? = nil,
        hiddenStateLayers: [Int] = [9, 18, 27],
    )
        throws -> MLXArray
    {
        let (_, statesOptional) = try callAsFunction(
            inputIDs: inputIDs,
            attentionMask: attentionMask,
            outputHiddenStates: true,
        )
        guard let states = statesOptional else {
            fatalError("Flux2 prompt hidden states were not collected.")
        }
        let selected = hiddenStateLayers.map { states[$0] }
        let stackedStates = stacked(selected, axis: 1)
        let batch = stackedStates.dim(0)
        let layerCount = stackedStates.dim(1)
        let seqLen = stackedStates.dim(2)
        let hiddenSize = stackedStates.dim(3)
        return stackedStates.transposed(0, 2, 1, 3)
            .reshaped(batch, seqLen, layerCount * hiddenSize)
    }
}

private struct Flux2ModulationSet {
    let shift: MLXArray
    let scale: MLXArray
    let gate: MLXArray
}

private final class Flux2SwiGLU: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let pieces = split(x, parts: 2, axis: -1)
        return silu(pieces[0]) * pieces[1]
    }
}

private final class Flux2FeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "linear_in") var linearIn: Linear
    @ModuleInfo var act: Flux2SwiGLU
    @ModuleInfo(key: "linear_out") var linearOut: Linear

    init(dim: Int, multiplier: Float = 3.0) {
        let inner = Int(Float(dim) * multiplier)
        _linearIn.wrappedValue = Linear(dim, inner * 2, bias: false)
        _act.wrappedValue = Flux2SwiGLU()
        _linearOut.wrappedValue = Linear(inner, dim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linearOut(act(linearIn(x)))
    }
}

private final class Flux2Modulation: Module {
    let dim: Int
    let modParamSets: Int
    @ModuleInfo var linear: Linear

    init(dim: Int, modParamSets: Int = 2) {
        self.dim = dim
        self.modParamSets = modParamSets
        _linear.wrappedValue = Linear(dim, dim * 3 * modParamSets, bias: false)
        super.init()
    }

    func callAsFunction(_ temb: MLXArray) -> [Flux2ModulationSet] {
        var mod = linear(silu(temb))
        if mod.ndim == 2 {
            mod = mod.expandedDimensions(axis: 1)
        }
        let pieces = split(mod, parts: 3 * modParamSets, axis: -1)
        return (0 ..< modParamSets).map { index in
            Flux2ModulationSet(
                shift: pieces[3 * index],
                scale: pieces[3 * index + 1],
                gate: pieces[3 * index + 2],
            )
        }
    }
}

final class Flux2PosEmbed: Module {
    let theta: Float
    let axesDims: [Int]

    init(theta: Float = 2000, axesDims: [Int] = [32, 32, 32, 32]) {
        self.theta = theta
        self.axesDims = axesDims
        super.init()
    }

    func callAsFunction(_ ids: MLXArray) -> (MLXArray, MLXArray) {
        let pos = ids.asType(.float32)
        var cosines: [MLXArray] = []
        var sines: [MLXArray] = []
        for (index, dim) in axesDims.enumerated() {
            let axisPosition = pos[.ellipsis, index]
            let rope = oneDimensionalRoPE(dim: dim, positions: axisPosition)
            cosines.append(rope.0)
            sines.append(rope.1)
        }
        return (concatenated(cosines, axis: -1), concatenated(sines, axis: -1))
    }

    private func oneDimensionalRoPE(dim: Int, positions: MLXArray) -> (MLXArray, MLXArray) {
        let scale = arange(0, dim, step: 2, dtype: .float32) / Float(dim)
        let omega = 1.0 / pow(MLXArray(theta), scale)
        let out = positions.expandedDimensions(axis: -1) * omega.expandedDimensions(axis: 0)
        return (cos(out), sin(out))
    }
}

private final class Flux2TimestepGuidanceEmbeddings: Module {
    let channels: Int
    let embeddingDim: Int
    let guidanceEmbeds: Bool

    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    @ModuleInfo(key: "guidance_linear_1") var guidanceLinear1: Linear?
    @ModuleInfo(key: "guidance_linear_2") var guidanceLinear2: Linear?

    init(channels: Int = 256, embeddingDim: Int, guidanceEmbeds: Bool = false) {
        self.channels = channels
        self.embeddingDim = embeddingDim
        self.guidanceEmbeds = guidanceEmbeds
        _linear1.wrappedValue = Linear(channels, embeddingDim, bias: false)
        _linear2.wrappedValue = Linear(embeddingDim, embeddingDim, bias: false)
        if guidanceEmbeds {
            _guidanceLinear1.wrappedValue = Linear(channels, embeddingDim, bias: false)
            _guidanceLinear2.wrappedValue = Linear(embeddingDim, embeddingDim, bias: false)
        } else {
            _guidanceLinear1.wrappedValue = nil
            _guidanceLinear2.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(timestep: MLXArray, guidance: MLXArray?) -> MLXArray {
        let timestepEmbedding = Self.timestepEmbedding(
            timesteps: timestep.asType(.float32),
            dimensions: channels,
        )
        var output = linear2(silu(linear1(timestepEmbedding)))
        if let guidance,
            let guidanceLinear1,
            let guidanceLinear2
        {
            let guidanceEmbedding = Self.timestepEmbedding(
                timesteps: guidance.asType(.float32),
                dimensions: channels,
            )
            output = output + guidanceLinear2(silu(guidanceLinear1(guidanceEmbedding)))
        }
        return output
    }

    static func timestepEmbedding(
        timesteps: MLXArray,
        dimensions: Int,
        flipSinToCos: Bool = true,
    )
        -> MLXArray
    {
        let half = dimensions / 2
        let frequencies = exp(
            -log(Float(10000.0)) * arange(0, half, dtype: .float32) / Float(half),
        )
        let args = timesteps[0..., .newAxis] * frequencies[.newAxis, 0...]
        var embedding = concatenated([sin(args), cos(args)], axis: -1)
        if flipSinToCos {
            embedding = concatenated(
                [embedding[0..., half...], embedding[0..., ..<half]], axis: -1)
        }
        if dimensions % 2 == 1 {
            embedding = concatenated(
                [embedding, MLXArray.zeros([embedding.dim(0), 1], dtype: embedding.dtype)],
                axis: -1,
            )
        }
        return embedding
    }
}

private final class Flux2AdaLayerNormContinuous: Module {
    let embeddingDim: Int
    @ModuleInfo var linear: Linear
    @ModuleInfo var norm: LayerNorm

    init(embeddingDim: Int, conditioningEmbeddingDim: Int) {
        self.embeddingDim = embeddingDim
        _linear.wrappedValue = Linear(conditioningEmbeddingDim, embeddingDim * 2, bias: false)
        _norm.wrappedValue = LayerNorm(dimensions: embeddingDim, eps: 1e-6, affine: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, textEmbeddings: MLXArray) -> MLXArray {
        let params = linear(silu(textEmbeddings).asType(Flux2Precision.dtype))
        let scale = params[0..., ..<embeddingDim]
        let shift = params[0..., embeddingDim ..< 2 * embeddingDim]
        return norm(x) * (1 + scale)[0..., .newAxis, 0...] + shift[0..., .newAxis, 0...]
    }
}

private final class Flux2KVLayerCache {
    var key: MLXArray?
    var value: MLXArray?

    func store(key: MLXArray, value: MLXArray) {
        self.key = key
        self.value = value
    }

    func get() -> (MLXArray, MLXArray) {
        guard let key, let value else {
            fatalError("Flux2 KV cache has not been populated.")
        }
        return (key, value)
    }

    func clear() {
        key = nil
        value = nil
    }
}

public final class Flux2KVCache {
    private var doubleBlockCaches: [Flux2KVLayerCache]
    private var singleBlockCaches: [Flux2KVLayerCache]
    public var referenceTokenCount = 0

    init(doubleLayers: Int, singleLayers: Int) {
        doubleBlockCaches = (0 ..< doubleLayers).map { _ in Flux2KVLayerCache() }
        singleBlockCaches = (0 ..< singleLayers).map { _ in Flux2KVLayerCache() }
    }

    fileprivate func double(_ index: Int) -> Flux2KVLayerCache {
        doubleBlockCaches[index]
    }

    fileprivate func single(_ index: Int) -> Flux2KVLayerCache {
        singleBlockCaches[index]
    }

    public func arrays() -> [MLXArray] {
        (doubleBlockCaches + singleBlockCaches).flatMap { cache in
            [cache.key, cache.value].compactMap(\.self)
        }
    }

    public func clear() {
        for cache in doubleBlockCaches + singleBlockCaches {
            cache.clear()
        }
        referenceTokenCount = 0
    }
}

private final class Flux2Attention: Module {
    let heads: Int
    let headDim: Int
    let innerDim: Int

    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "norm_added_q") var normAddedQ: RMSNorm
    @ModuleInfo(key: "norm_added_k") var normAddedK: RMSNorm
    @ModuleInfo(key: "add_q_proj") var addQProj: Linear
    @ModuleInfo(key: "add_k_proj") var addKProj: Linear
    @ModuleInfo(key: "add_v_proj") var addVProj: Linear
    @ModuleInfo(key: "to_add_out") var toAddOut: Linear

    init(dim: Int, heads: Int, headDim: Int, addedKVProjectionDim: Int) {
        self.heads = heads
        self.headDim = headDim
        innerDim = heads * headDim
        _toQ.wrappedValue = Linear(dim, innerDim, bias: false)
        _toK.wrappedValue = Linear(dim, innerDim, bias: false)
        _toV.wrappedValue = Linear(dim, innerDim, bias: false)
        _normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-5)
        _normK.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-5)
        _toOut.wrappedValue = Linear(innerDim, dim, bias: false)
        _normAddedQ.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-5)
        _normAddedK.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-5)
        _addQProj.wrappedValue = Linear(addedKVProjectionDim, innerDim, bias: false)
        _addKProj.wrappedValue = Linear(addedKVProjectionDim, innerDim, bias: false)
        _addVProj.wrappedValue = Linear(addedKVProjectionDim, innerDim, bias: false)
        _toAddOut.wrappedValue = Linear(innerDim, dim, bias: false)
        super.init()
    }

    private func processQKV(
        hidden: MLXArray,
        qProjection: Linear,
        kProjection: Linear,
        vProjection: Linear,
        qNorm: RMSNorm,
        kNorm: RMSNorm,
    )
        -> (MLXArray, MLXArray, MLXArray)
    {
        let batch = hidden.dim(0)
        let seqLen = hidden.dim(1)
        var queries = qProjection(hidden).reshaped(batch, seqLen, heads, headDim)
            .transposed(0, 2, 1, 3)
        var keys = kProjection(hidden).reshaped(batch, seqLen, heads, headDim)
            .transposed(0, 2, 1, 3)
        let values = vProjection(hidden).reshaped(batch, seqLen, heads, headDim)
            .transposed(0, 2, 1, 3)
        let queryDType = queries.dtype
        let keyDType = keys.dtype
        queries = qNorm(queries.asType(.float32)).asType(queryDType)
        keys = kNorm(keys.asType(.float32)).asType(keyDType)
        return (queries, keys, values)
    }

    func callAsFunction(
        hidden: MLXArray,
        encoderHidden: MLXArray,
        rotaryEmbeddings: (MLXArray, MLXArray),
        kvCache: Flux2KVLayerCache? = nil,
        kvCacheMode: String? = nil,
        referenceTokenCount: Int = 0,
    )
        -> (hidden: MLXArray, encoderHidden: MLXArray)
    {
        var qkv = processQKV(
            hidden: hidden,
            qProjection: toQ,
            kProjection: toK,
            vProjection: toV,
            qNorm: normQ,
            kNorm: normK,
        )
        let encoderQKV = processQKV(
            hidden: encoderHidden,
            qProjection: addQProj,
            kProjection: addKProj,
            vProjection: addVProj,
            qNorm: normAddedQ,
            kNorm: normAddedK,
        )
        qkv.0 = concatenated([encoderQKV.0, qkv.0], axis: 2)
        qkv.1 = concatenated([encoderQKV.1, qkv.1], axis: 2)
        qkv.2 = concatenated([encoderQKV.2, qkv.2], axis: 2)

        let rope = Flux2AttentionMath.applyFluxRoPE(
            queries: qkv.0,
            keys: qkv.1,
            cosines: rotaryEmbeddings.0,
            sines: rotaryEmbeddings.1,
        )
        qkv.0 = rope.0
        qkv.1 = rope.1

        let numTextTokens = encoderHidden.dim(1)
        if kvCacheMode == "extract", let kvCache, referenceTokenCount > 0 {
            let start = numTextTokens
            let end = numTextTokens + referenceTokenCount
            kvCache.store(
                key: qkv.1[0..., 0..., start ..< end, 0...],
                value: qkv.2[0..., 0..., start ..< end, 0...],
            )
        }

        let attended: MLXArray =
            if kvCacheMode == "extract", referenceTokenCount > 0 {
                computeKVCacheAttention(
                    queries: qkv.0,
                    keys: qkv.1,
                    values: qkv.2,
                    batch: hidden.dim(0),
                    textTokenCount: numTextTokens,
                    referenceTokenCount: referenceTokenCount,
                    kvCache: nil,
                )
            } else if kvCacheMode == "cached", let kvCache {
                computeKVCacheAttention(
                    queries: qkv.0,
                    keys: qkv.1,
                    values: qkv.2,
                    batch: hidden.dim(0),
                    textTokenCount: numTextTokens,
                    referenceTokenCount: 0,
                    kvCache: kvCache,
                )
            } else {
                Flux2AttentionMath.scaledAttention(
                    queries: qkv.0,
                    keys: qkv.1,
                    values: qkv.2,
                    batch: hidden.dim(0),
                    heads: heads,
                    headDim: headDim,
                )
            }

        let encoderLength = encoderHidden.dim(1)
        let encoderOut = toAddOut(attended[0..., ..<encoderLength, 0...])
        let hiddenOut = toOut(attended[0..., encoderLength..., 0...])
        return (hiddenOut, encoderOut)
    }

    private func computeKVCacheAttention(
        queries: MLXArray,
        keys: MLXArray,
        values: MLXArray,
        batch: Int,
        textTokenCount: Int,
        referenceTokenCount: Int,
        kvCache: Flux2KVLayerCache?,
    )
        -> MLXArray
    {
        if let kvCache {
            let cached = kvCache.get()
            let keys = concatenated(
                [
                    keys[0..., 0..., ..<textTokenCount, 0...],
                    cached.0,
                    keys[0..., 0..., textTokenCount..., 0...],
                ],
                axis: 2,
            )
            let values = concatenated(
                [
                    values[0..., 0..., ..<textTokenCount, 0...],
                    cached.1,
                    values[0..., 0..., textTokenCount..., 0...],
                ],
                axis: 2,
            )
            return Flux2AttentionMath.scaledAttention(
                queries: queries,
                keys: keys,
                values: values,
                batch: batch,
                heads: heads,
                headDim: headDim,
            )
        }

        guard referenceTokenCount > 0 else {
            return Flux2AttentionMath.scaledAttention(
                queries: queries,
                keys: keys,
                values: values,
                batch: batch,
                heads: heads,
                headDim: headDim,
            )
        }

        let refStart = textTokenCount
        let refEnd = textTokenCount + referenceTokenCount
        let queryText = queries[0..., 0..., ..<refStart, 0...]
        let queryRef = queries[0..., 0..., refStart ..< refEnd, 0...]
        let queryImage = queries[0..., 0..., refEnd..., 0...]
        let queryTextImage = concatenated([queryText, queryImage], axis: 2)
        let hiddenTextImage = Flux2AttentionMath.scaledAttention(
            queries: queryTextImage,
            keys: keys,
            values: values,
            batch: batch,
            heads: heads,
            headDim: headDim,
        )
        let hiddenText = hiddenTextImage[0..., ..<refStart, 0...]
        let hiddenImage = hiddenTextImage[0..., refStart..., 0...]
        let hiddenRef = Flux2AttentionMath.scaledAttention(
            queries: queryRef,
            keys: keys[0..., 0..., refStart ..< refEnd, 0...],
            values: values[0..., 0..., refStart ..< refEnd, 0...],
            batch: batch,
            heads: heads,
            headDim: headDim,
        )
        return concatenated([hiddenText, hiddenRef, hiddenImage], axis: 1)
    }
}

private final class Flux2ParallelSelfAttention: Module {
    let heads: Int
    let headDim: Int
    let innerDim: Int
    let mlpHiddenDim: Int

    @ModuleInfo(key: "to_qkv_mlp_proj") var toQKVMLPProj: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    @ModuleInfo(key: "mlp_act") var mlpAct: Flux2SwiGLU
    @ModuleInfo(key: "to_out") var toOut: Linear

    init(dim: Int, heads: Int, headDim: Int, mlpRatio: Float = 3.0) {
        self.heads = heads
        self.headDim = headDim
        innerDim = heads * headDim
        mlpHiddenDim = Int(Float(dim) * mlpRatio)
        _toQKVMLPProj.wrappedValue = Linear(
            dim,
            innerDim * 3 + mlpHiddenDim * 2,
            bias: false,
        )
        _normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-5)
        _normK.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-5)
        _mlpAct.wrappedValue = Flux2SwiGLU()
        _toOut.wrappedValue = Linear(innerDim + mlpHiddenDim, dim, bias: false)
        super.init()
    }

    func callAsFunction(
        _ hidden: MLXArray,
        rotaryEmbeddings: (MLXArray, MLXArray),
        kvCache: Flux2KVLayerCache? = nil,
        kvCacheMode: String? = nil,
        textTokenCount: Int = 0,
        referenceTokenCount: Int = 0,
    )
        -> MLXArray
    {
        let projection = toQKVMLPProj(hidden)
        let qkvAndMLP = split(projection, indices: [innerDim * 3], axis: -1)
        let qkv = split(qkvAndMLP[0], parts: 3, axis: -1)
        let batch = hidden.dim(0)
        let seqLen = hidden.dim(1)
        var queries = qkv[0].reshaped(batch, seqLen, heads, headDim).transposed(0, 2, 1, 3)
        var keys = qkv[1].reshaped(batch, seqLen, heads, headDim).transposed(0, 2, 1, 3)
        var values = qkv[2].reshaped(batch, seqLen, heads, headDim).transposed(0, 2, 1, 3)

        queries = normQ(queries.asType(.float32)).asType(Flux2Precision.dtype)
        keys = normK(keys.asType(.float32)).asType(Flux2Precision.dtype)

        let rope = Flux2AttentionMath.applyFluxRoPE(
            queries: queries,
            keys: keys,
            cosines: rotaryEmbeddings.0,
            sines: rotaryEmbeddings.1,
        )
        queries = rope.0
        keys = rope.1

        if kvCacheMode == "extract", let kvCache, referenceTokenCount > 0 {
            let start = textTokenCount
            let end = textTokenCount + referenceTokenCount
            kvCache.store(
                key: keys[0..., 0..., start ..< end, 0...],
                value: values[0..., 0..., start ..< end, 0...],
            )
        }

        let attended: MLXArray
        if kvCacheMode == "cached", let kvCache {
            let cached = kvCache.get()
            keys = concatenated(
                [
                    keys[0..., 0..., ..<textTokenCount, 0...],
                    cached.0,
                    keys[0..., 0..., textTokenCount..., 0...],
                ],
                axis: 2,
            )
            values = concatenated(
                [
                    values[0..., 0..., ..<textTokenCount, 0...],
                    cached.1,
                    values[0..., 0..., textTokenCount..., 0...],
                ],
                axis: 2,
            )
            attended = Flux2AttentionMath.scaledAttention(
                queries: queries,
                keys: keys,
                values: values,
                batch: batch,
                heads: heads,
                headDim: headDim,
            )
        } else {
            attended = Flux2AttentionMath.scaledAttention(
                queries: queries,
                keys: keys,
                values: values,
                batch: batch,
                heads: heads,
                headDim: headDim,
            )
        }

        let mlpHidden = mlpAct(qkvAndMLP[1])
        return toOut(concatenated([attended, mlpHidden], axis: -1))
    }
}

private final class Flux2TransformerBlock: Module {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo(key: "norm1_context") var norm1Context: LayerNorm
    @ModuleInfo var attn: Flux2Attention
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var ff: Flux2FeedForward
    @ModuleInfo(key: "norm2_context") var norm2Context: LayerNorm
    @ModuleInfo(key: "ff_context") var ffContext: Flux2FeedForward

    init(dim: Int, heads: Int, headDim: Int, mlpRatio: Float) {
        _norm1.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _norm1Context.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _attn.wrappedValue = Flux2Attention(
            dim: dim,
            heads: heads,
            headDim: headDim,
            addedKVProjectionDim: dim,
        )
        _norm2.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _ff.wrappedValue = Flux2FeedForward(dim: dim, multiplier: mlpRatio)
        _norm2Context.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _ffContext.wrappedValue = Flux2FeedForward(dim: dim, multiplier: mlpRatio)
        super.init()
    }

    func callAsFunction(
        hidden: MLXArray,
        encoderHidden: MLXArray,
        imageModulation: [Flux2ModulationSet],
        textModulation: [Flux2ModulationSet],
        rotaryEmbeddings: (MLXArray, MLXArray),
        kvCache: Flux2KVLayerCache? = nil,
        kvCacheMode: String? = nil,
        referenceTokenCount: Int = 0,
    )
        -> (encoderHidden: MLXArray, hidden: MLXArray)
    {
        let imgAttention = imageModulation[0]
        let imgMLP = imageModulation[1]
        let textAttention = textModulation[0]
        let textMLP = textModulation[1]

        let normHidden = (1 + imgAttention.scale) * norm1(hidden) + imgAttention.shift
        let normEncoder =
            (1 + textAttention.scale) * norm1Context(encoderHidden)
            + textAttention.shift

        let attended = attn(
            hidden: normHidden,
            encoderHidden: normEncoder,
            rotaryEmbeddings: rotaryEmbeddings,
            kvCache: kvCache,
            kvCacheMode: kvCacheMode,
            referenceTokenCount: referenceTokenCount,
        )

        var hidden = hidden + imgAttention.gate * attended.hidden
        var encoderHidden = encoderHidden + textAttention.gate * attended.encoderHidden

        let normHiddenMLP = (1 + imgMLP.scale) * norm2(hidden) + imgMLP.shift
        hidden = hidden + imgMLP.gate * ff(normHiddenMLP)

        let normEncoderMLP = (1 + textMLP.scale) * norm2Context(encoderHidden) + textMLP.shift
        encoderHidden = encoderHidden + textMLP.gate * ffContext(normEncoderMLP)
        return (encoderHidden, hidden)
    }
}

private final class Flux2SingleTransformerBlock: Module {
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var attn: Flux2ParallelSelfAttention

    init(dim: Int, heads: Int, headDim: Int, mlpRatio: Float) {
        _norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
        _attn.wrappedValue = Flux2ParallelSelfAttention(
            dim: dim,
            heads: heads,
            headDim: headDim,
            mlpRatio: mlpRatio,
        )
        super.init()
    }

    func callAsFunction(
        _ hidden: MLXArray,
        modulation: Flux2ModulationSet,
        rotaryEmbeddings: (MLXArray, MLXArray),
        kvCache: Flux2KVLayerCache? = nil,
        kvCacheMode: String? = nil,
        textTokenCount: Int = 0,
        referenceTokenCount: Int = 0,
    )
        -> MLXArray
    {
        let normHidden = (1 + modulation.scale) * norm(hidden) + modulation.shift
        let attended = attn(
            normHidden,
            rotaryEmbeddings: rotaryEmbeddings,
            kvCache: kvCache,
            kvCacheMode: kvCacheMode,
            textTokenCount: textTokenCount,
            referenceTokenCount: referenceTokenCount,
        )
        return hidden + modulation.gate * attended
    }
}

public final class Flux2Transformer: Module {
    public let config: Flux2TransformerConfiguration
    public let innerDim: Int
    public let outputChannels: Int

    @ModuleInfo(key: "pos_embed") fileprivate var posEmbed: Flux2PosEmbed
    @ModuleInfo(key: "time_guidance_embed") fileprivate var timeGuidanceEmbed:
        Flux2TimestepGuidanceEmbeddings
    @ModuleInfo(key: "double_stream_modulation_img") fileprivate var doubleStreamModulationImage:
        Flux2Modulation
    @ModuleInfo(key: "double_stream_modulation_txt") fileprivate var doubleStreamModulationText:
        Flux2Modulation
    @ModuleInfo(key: "single_stream_modulation") fileprivate var singleStreamModulation:
        Flux2Modulation
    @ModuleInfo(key: "x_embedder") fileprivate var xEmbedder: Linear
    @ModuleInfo(key: "context_embedder") fileprivate var contextEmbedder: Linear
    @ModuleInfo(key: "transformer_blocks") fileprivate var transformerBlocks:
        [Flux2TransformerBlock]
    @ModuleInfo(key: "single_transformer_blocks") fileprivate var singleTransformerBlocks:
        [Flux2SingleTransformerBlock]
    @ModuleInfo(key: "norm_out") fileprivate var normOut: Flux2AdaLayerNormContinuous
    @ModuleInfo(key: "proj_out") fileprivate var projOut: Linear

    public init(_ config: Flux2TransformerConfiguration = Flux2TransformerConfiguration()) {
        self.config = config
        let innerDimension = config.numAttentionHeads * config.attentionHeadDim
        let outputChannelCount = config.outChannels ?? config.inChannels
        innerDim = innerDimension
        outputChannels = outputChannelCount
        _posEmbed.wrappedValue = Flux2PosEmbed(
            theta: config.ropeTheta, axesDims: config.axesDimsRoPE)
        _timeGuidanceEmbed.wrappedValue = Flux2TimestepGuidanceEmbeddings(
            channels: config.timestepGuidanceChannels,
            embeddingDim: innerDimension,
            guidanceEmbeds: config.guidanceEmbeds,
        )
        _doubleStreamModulationImage.wrappedValue = Flux2Modulation(
            dim: innerDimension, modParamSets: 2)
        _doubleStreamModulationText.wrappedValue = Flux2Modulation(
            dim: innerDimension, modParamSets: 2)
        _singleStreamModulation.wrappedValue = Flux2Modulation(dim: innerDimension, modParamSets: 1)
        _xEmbedder.wrappedValue = Linear(config.inChannels, innerDimension, bias: false)
        _contextEmbedder.wrappedValue = Linear(
            config.jointAttentionDim, innerDimension, bias: false)
        _transformerBlocks.wrappedValue = (0 ..< config.numLayers).map { _ in
            Flux2TransformerBlock(
                dim: innerDimension,
                heads: config.numAttentionHeads,
                headDim: config.attentionHeadDim,
                mlpRatio: config.mlpRatio,
            )
        }
        _singleTransformerBlocks.wrappedValue = (0 ..< config.numSingleLayers).map { _ in
            Flux2SingleTransformerBlock(
                dim: innerDimension,
                heads: config.numAttentionHeads,
                headDim: config.attentionHeadDim,
                mlpRatio: config.mlpRatio,
            )
        }
        _normOut.wrappedValue = Flux2AdaLayerNormContinuous(
            embeddingDim: innerDimension,
            conditioningEmbeddingDim: innerDimension,
        )
        _projOut.wrappedValue = Linear(
            innerDimension,
            config.patchSize * config.patchSize * outputChannelCount,
            bias: false,
        )
        super.init()
    }

    public func callAsFunction(
        hiddenStates: MLXArray,
        encoderHiddenStates: MLXArray,
        timestep: MLXArray,
        imageIDs: MLXArray,
        textIDs: MLXArray,
        guidance: MLXArray? = nil,
        kvCache: Flux2KVCache? = nil,
        kvCacheMode: String? = nil,
        referenceTokenCount: Int = 0,
        referenceFixedTimestep: Float = 0.0,
    )
        -> MLXArray
    {
        let numTextTokens = encoderHiddenStates.dim(1)
        let numImageTokens = hiddenStates.dim(1)
        var timestep = timestep
        if timestep.ndim == 0 {
            timestep =
                MLXArray.zeros([hiddenStates.dim(0)], dtype: hiddenStates.dtype)
                + timestep.item(Float.self)
        }
        timestep = timestep.asType(hiddenStates.dtype)
        let timestepScale = MLX.where(max(timestep) .<= 1.0, MLXArray(1000.0), MLXArray(1.0))
            .asType(hiddenStates.dtype)
        timestep = timestep * timestepScale

        var guidance = guidance
        if var guidanceValue = guidance {
            if guidanceValue.ndim == 0 {
                guidanceValue =
                    MLXArray.zeros([hiddenStates.dim(0)], dtype: hiddenStates.dtype)
                    + guidanceValue.item(Float.self)
            }
            let guidanceScale = MLX.where(
                max(guidanceValue) .<= 1.0, MLXArray(1000.0), MLXArray(1.0)
            )
            .asType(hiddenStates.dtype)
            guidance = guidanceValue.asType(hiddenStates.dtype) * guidanceScale
        }

        let temb = timeGuidanceEmbed(timestep: timestep, guidance: guidance)
            .asType(Flux2Precision.dtype)

        var referenceTemb: MLXArray?
        if kvCacheMode == "extract", referenceTokenCount > 0 {
            let cache =
                kvCache
                ?? Flux2KVCache(
                    doubleLayers: transformerBlocks.count,
                    singleLayers: singleTransformerBlocks.count,
                )
            cache.referenceTokenCount = referenceTokenCount
            _ = cache
            let refTimestep =
                MLXArray.zeros(timestep.shape, dtype: timestep.dtype)
                + referenceFixedTimestep * Float(1000.0)
            referenceTemb = timeGuidanceEmbed(timestep: refTimestep, guidance: guidance)
                .asType(Flux2Precision.dtype)
        }

        var hidden = xEmbedder(hiddenStates)
        var encoderHidden = contextEmbedder(encoderHiddenStates)
        var imageIDs = imageIDs
        var textIDs = textIDs
        if imageIDs.ndim == 3 {
            imageIDs = imageIDs[0]
        }
        if textIDs.ndim == 3 {
            textIDs = textIDs[0]
        }

        let imageRotary = posEmbed(imageIDs)
        let textRotary = posEmbed(textIDs)
        let concatRotary = (
            concatenated([textRotary.0, imageRotary.0], axis: 0),
            concatenated([textRotary.1, imageRotary.1], axis: 0),
        )

        var imageModulation = doubleStreamModulationImage(temb)
        let textModulation = doubleStreamModulationText(temb)
        if let referenceTemb {
            imageModulation = Self.blendDoubleModulation(
                imageModulation,
                reference: doubleStreamModulationImage(referenceTemb),
                referenceTokenCount: referenceTokenCount,
                sequenceLength: numImageTokens,
            )
        }

        for (index, block) in transformerBlocks.enumerated() {
            let layerCache =
                kvCacheMode != nil && kvCache != nil
                ? kvCache?.double(index)
                : nil
            let out = block(
                hidden: hidden,
                encoderHidden: encoderHidden,
                imageModulation: imageModulation,
                textModulation: textModulation,
                rotaryEmbeddings: concatRotary,
                kvCache: layerCache,
                kvCacheMode: kvCacheMode,
                referenceTokenCount: referenceTokenCount,
            )
            encoderHidden = out.encoderHidden
            hidden = out.hidden
        }

        hidden = concatenated([encoderHidden, hidden], axis: 1)
        var singleModulation = singleStreamModulation(temb)[0]
        if let referenceTemb {
            singleModulation = Self.blendSingleModulation(
                singleModulation,
                reference: singleStreamModulation(referenceTemb)[0],
                textTokenCount: numTextTokens,
                referenceTokenCount: referenceTokenCount,
                sequenceLength: hidden.dim(1),
            )
        }
        for (index, block) in singleTransformerBlocks.enumerated() {
            let layerCache =
                kvCacheMode != nil && kvCache != nil
                ? kvCache?.single(index)
                : nil
            hidden = block(
                hidden,
                modulation: singleModulation,
                rotaryEmbeddings: concatRotary,
                kvCache: layerCache,
                kvCacheMode: kvCacheMode,
                textTokenCount: numTextTokens,
                referenceTokenCount: referenceTokenCount,
            )
        }

        var start = numTextTokens
        if kvCacheMode == "extract", referenceTokenCount > 0 {
            start += referenceTokenCount
        }
        hidden = hidden[0..., start..., 0...]
        hidden = normOut(hidden, textEmbeddings: temb)
        return projOut(hidden)
    }

    private static func expandModulation(_ value: MLXArray, sequenceLength: Int) -> MLXArray {
        var value = value
        if value.ndim == 2 {
            value = value.expandedDimensions(axis: 1)
        }
        if value.dim(1) == sequenceLength {
            return value
        }
        return broadcast(value, to: [value.dim(0), sequenceLength, value.dim(2)])
    }

    private static func blendSet(
        _ image: Flux2ModulationSet,
        reference: Flux2ModulationSet,
        referenceTokenCount: Int,
        sequenceLength: Int,
    )
        -> Flux2ModulationSet
    {
        func blend(_ image: MLXArray, _ reference: MLXArray) -> MLXArray {
            let imageExpanded = expandModulation(image, sequenceLength: sequenceLength)
            let referenceExpanded = expandModulation(reference, sequenceLength: referenceTokenCount)
            return concatenated(
                [referenceExpanded, imageExpanded[0..., referenceTokenCount..., 0...]],
                axis: 1,
            )
        }
        return Flux2ModulationSet(
            shift: blend(image.shift, reference.shift),
            scale: blend(image.scale, reference.scale),
            gate: blend(image.gate, reference.gate),
        )
    }

    private static func blendDoubleModulation(
        _ image: [Flux2ModulationSet],
        reference: [Flux2ModulationSet],
        referenceTokenCount: Int,
        sequenceLength: Int,
    )
        -> [Flux2ModulationSet]
    {
        zip(image, reference).map {
            blendSet(
                $0,
                reference: $1,
                referenceTokenCount: referenceTokenCount,
                sequenceLength: sequenceLength,
            )
        }
    }

    private static func blendSingleModulation(
        _ image: Flux2ModulationSet,
        reference: Flux2ModulationSet,
        textTokenCount: Int,
        referenceTokenCount: Int,
        sequenceLength: Int,
    )
        -> Flux2ModulationSet
    {
        func blend(_ image: MLXArray, _ reference: MLXArray) -> MLXArray {
            let imageExpanded = expandModulation(image, sequenceLength: sequenceLength)
            let referenceExpanded = expandModulation(reference, sequenceLength: referenceTokenCount)
            return concatenated(
                [
                    imageExpanded[0..., ..<textTokenCount, 0...],
                    referenceExpanded,
                    imageExpanded[0..., (textTokenCount + referenceTokenCount)..., 0...],
                ],
                axis: 1,
            )
        }
        return Flux2ModulationSet(
            shift: blend(image.shift, reference.shift),
            scale: blend(image.scale, reference.scale),
            gate: blend(image.gate, reference.gate),
        )
    }
}

public struct Flux2VAEConfiguration: Sendable, Equatable {
    public let decoderBlockOutChannels: [Int]
    public let encoderBlockOutChannels: [Int]
    public let includeEncoder: Bool
    public let latentChannels: Int
    public let spatialScale: Int
    public let layersPerBlock: Int
    public let normGroups: Int
    public let eps: Float
    public let batchNormEps: Float
    public let midBlockAddAttention: Bool

    public init(
        decoderBlockOutChannels: [Int] = [128, 256, 512, 512],
        encoderBlockOutChannels: [Int] = [128, 256, 512, 512],
        includeEncoder: Bool = false,
        latentChannels: Int = 32,
        spatialScale: Int = 8,
        layersPerBlock: Int = 2,
        normGroups: Int = 32,
        eps: Float = 1e-6,
        batchNormEps: Float = 1e-4,
        midBlockAddAttention: Bool = true,
    ) {
        self.decoderBlockOutChannels = decoderBlockOutChannels
        self.encoderBlockOutChannels = encoderBlockOutChannels
        self.includeEncoder = includeEncoder
        self.latentChannels = latentChannels
        self.spatialScale = spatialScale
        self.layersPerBlock = layersPerBlock
        self.normGroups = normGroups
        self.eps = eps
        self.batchNormEps = batchNormEps
        self.midBlockAddAttention = midBlockAddAttention
    }
}

private final class Flux2ConvNCHW: Module, UnaryLayer {
    @ModuleInfo var conv: Conv2d

    init(
        inputChannels: Int, outputChannels: Int, kernelSize: Int = 3, stride: Int = 1,
        padding: Int = 1
    ) {
        _conv.wrappedValue = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(stride),
            padding: IntOrPair(padding),
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(x.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
    }
}

private final class Flux2GroupNormNCHW: Module, UnaryLayer {
    @ModuleInfo var norm: GroupNorm
    private let activationDType: DType

    init(channels: Int, groups: Int, eps: Float, activationDType: DType) {
        self.activationDType = activationDType
        _norm.wrappedValue = GroupNorm(
            groupCount: groups,
            dimensions: channels,
            eps: eps,
            pytorchCompatible: true,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        norm(x.transposed(0, 2, 3, 1).asType(.float32))
            .asType(activationDType)
            .transposed(0, 3, 1, 2)
    }
}

private final class Flux2ResnetBlock2D: Module, UnaryLayer {
    @ModuleInfo var norm1: GroupNorm
    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var norm2: GroupNorm
    @ModuleInfo var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?
    private let activationDType: DType

    init(inChannels: Int, outChannels: Int, eps: Float, groups: Int, activationDType: DType) {
        self.activationDType = activationDType
        _norm1.wrappedValue = GroupNorm(
            groupCount: groups,
            dimensions: inChannels,
            eps: eps,
            pytorchCompatible: true,
        )
        _conv1.wrappedValue = Conv2d(
            inputChannels: inChannels,
            outputChannels: outChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
        )
        _norm2.wrappedValue = GroupNorm(
            groupCount: groups,
            dimensions: outChannels,
            eps: eps,
            pytorchCompatible: true,
        )
        _conv2.wrappedValue = Conv2d(
            inputChannels: outChannels,
            outputChannels: outChannels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
        )
        if inChannels != outChannels {
            _convShortcut.wrappedValue = Conv2d(
                inputChannels: inChannels,
                outputChannels: outChannels,
                kernelSize: 1,
                stride: 1,
                padding: 0,
            )
        } else {
            _convShortcut.wrappedValue = nil
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var residual = x.transposed(0, 2, 3, 1)
        var hidden = residual
        hidden = norm1(hidden.asType(.float32)).asType(activationDType)
        hidden = conv1(silu(hidden))
        hidden = norm2(hidden.asType(.float32)).asType(activationDType)
        hidden = conv2(silu(hidden))
        if let convShortcut {
            residual = convShortcut(residual)
        }
        return (hidden + residual).transposed(0, 3, 1, 2)
    }
}

private final class Flux2VAEAttentionBlock: Module, UnaryLayer {
    let channels: Int
    private let activationDType: DType
    @ModuleInfo(key: "group_norm") var groupNorm: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear

    init(channels: Int, groups: Int, eps: Float, activationDType: DType) {
        self.activationDType = activationDType
        self.channels = channels
        _groupNorm.wrappedValue = GroupNorm(
            groupCount: groups,
            dimensions: channels,
            eps: eps,
            pytorchCompatible: true,
        )
        _toQ.wrappedValue = Linear(channels, channels)
        _toK.wrappedValue = Linear(channels, channels)
        _toV.wrappedValue = Linear(channels, channels)
        _toOut.wrappedValue = Linear(channels, channels)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = x.transposed(0, 2, 3, 1)
        let batch = hidden.dim(0)
        let height = hidden.dim(1)
        let width = hidden.dim(2)
        let normed = groupNorm(hidden.asType(.float32)).asType(activationDType)
        let q = toQ(normed).reshaped(batch, height * width, 1, channels).transposed(0, 2, 1, 3)
        let k = toK(normed).reshaped(batch, height * width, 1, channels).transposed(0, 2, 1, 3)
        let v = toV(normed).reshaped(batch, height * width, 1, channels).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: 1.0 / sqrt(Float(channels)),
            mask: nil,
        )
        let out = toOut(attended.transposed(0, 2, 1, 3).reshaped(batch, height, width, channels))
        return (hidden + out).transposed(0, 3, 1, 2)
    }
}

private final class Flux2UNetMidBlock2D: Module, UnaryLayer {
    @ModuleInfo var resnets: [Flux2ResnetBlock2D]
    @ModuleInfo var attentions: [Flux2VAEAttentionBlock]

    init(channels: Int, eps: Float, groups: Int, addAttention: Bool, activationDType: DType) {
        _resnets.wrappedValue = [
            Flux2ResnetBlock2D(
                inChannels: channels, outChannels: channels, eps: eps, groups: groups,
                activationDType: activationDType),
            Flux2ResnetBlock2D(
                inChannels: channels, outChannels: channels, eps: eps, groups: groups,
                activationDType: activationDType),
        ]
        _attentions.wrappedValue =
            addAttention
            ? [
                Flux2VAEAttentionBlock(
                    channels: channels, groups: groups, eps: eps, activationDType: activationDType)
            ] : []
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = resnets[0](x)
        if let attention = attentions.first {
            hidden = attention(hidden)
        }
        return resnets[1](hidden)
    }
}

private final class Flux2Downsample2D: Module, UnaryLayer {
    @ModuleInfo var conv: Conv2d

    init(channels: Int, padding: Int = 0) {
        _conv.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: channels,
            kernelSize: 3,
            stride: 2,
            padding: IntOrPair(padding),
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let paddedInput = padded(x, widths: [0, 0, .init((0, 1)), .init((0, 1))])
        return conv(paddedInput.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
    }
}

private final class Flux2Upsample2D: Module, UnaryLayer {
    @ModuleInfo var conv: Conv2d

    init(channels: Int, outChannels: Int? = nil) {
        _conv.wrappedValue = Conv2d(
            inputChannels: channels,
            outputChannels: outChannels ?? channels,
            kernelSize: 3,
            stride: 1,
            padding: 1,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = repeated(x, count: 2, axis: 2)
        hidden = repeated(hidden, count: 2, axis: 3)
        return conv(hidden.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
    }
}

private final class Flux2DownEncoderBlock2D: Module, UnaryLayer {
    @ModuleInfo var resnets: [Flux2ResnetBlock2D]
    @ModuleInfo var downsamplers: [Flux2Downsample2D]

    init(
        inChannels: Int,
        outChannels: Int,
        layers: Int = 2,
        eps: Float = 1e-6,
        groups: Int = 32,
        addDownsample: Bool = true,
        downsamplePadding: Int = 0,
        activationDType: DType,
    ) {
        _resnets.wrappedValue = (0 ..< layers).map { index in
            Flux2ResnetBlock2D(
                inChannels: index == 0 ? inChannels : outChannels,
                outChannels: outChannels,
                eps: eps,
                groups: groups,
                activationDType: activationDType,
            )
        }
        _downsamplers.wrappedValue =
            addDownsample
            ? [Flux2Downsample2D(channels: outChannels, padding: downsamplePadding)] : []
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        for resnet in resnets {
            hidden = resnet(hidden)
        }
        for downsampler in downsamplers {
            hidden = downsampler(hidden)
        }
        return hidden
    }
}

private final class Flux2UpDecoderBlock2D: Module, UnaryLayer {
    @ModuleInfo var resnets: [Flux2ResnetBlock2D]
    @ModuleInfo var upsamplers: [Flux2Upsample2D]

    init(
        inChannels: Int,
        outChannels: Int,
        layers: Int = 3,
        eps: Float = 1e-6,
        groups: Int = 32,
        addUpsample: Bool = true,
        activationDType: DType,
    ) {
        _resnets.wrappedValue = (0 ..< layers).map { index in
            Flux2ResnetBlock2D(
                inChannels: index == 0 ? inChannels : outChannels,
                outChannels: outChannels,
                eps: eps,
                groups: groups,
                activationDType: activationDType,
            )
        }
        _upsamplers.wrappedValue =
            addUpsample ? [Flux2Upsample2D(channels: outChannels, outChannels: outChannels)] : []
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x
        for resnet in resnets {
            hidden = resnet(hidden)
        }
        for upsampler in upsamplers {
            hidden = upsampler(hidden)
        }
        return hidden
    }
}

final class Flux2Encoder: Module, UnaryLayer {
    @ModuleInfo(key: "conv_in") private var convIn: Flux2ConvNCHW
    @ModuleInfo(key: "down_blocks") private var downBlocks: [Flux2DownEncoderBlock2D]
    @ModuleInfo(key: "mid_block") private var midBlock: Flux2UNetMidBlock2D
    @ModuleInfo(key: "conv_norm_out") private var convNormOut: Flux2GroupNormNCHW
    @ModuleInfo(key: "conv_out") private var convOut: Flux2ConvNCHW
    private let activationDType: DType

    init(
        inChannels: Int = 3,
        outChannels: Int = 32,
        blockOutChannels: [Int],
        layersPerBlock: Int,
        normGroups: Int,
        eps: Float,
        midBlockAddAttention: Bool,
        activationDType: DType = Flux2Precision.dtype,
        hiddenDType: DType? = nil,
    ) {
        self.activationDType = activationDType
        _convIn.wrappedValue = Flux2ConvNCHW(
            inputChannels: inChannels,
            outputChannels: blockOutChannels[0],
        )
        _downBlocks.wrappedValue = blockOutChannels.enumerated().map { index, outChannel in
            let input = index > 0 ? blockOutChannels[index - 1] : blockOutChannels[0]
            return Flux2DownEncoderBlock2D(
                inChannels: input,
                outChannels: outChannel,
                layers: layersPerBlock,
                eps: eps,
                groups: normGroups,
                addDownsample: index != blockOutChannels.count - 1,
                downsamplePadding: 0,
                activationDType: hiddenDType ?? activationDType,
            )
        }
        _midBlock.wrappedValue = Flux2UNetMidBlock2D(
            channels: blockOutChannels.last!,
            eps: eps,
            groups: normGroups,
            addAttention: midBlockAddAttention,
            activationDType: hiddenDType ?? activationDType,
        )
        _convNormOut.wrappedValue = Flux2GroupNormNCHW(
            channels: blockOutChannels.last!,
            groups: normGroups,
            eps: eps,
            activationDType: activationDType,
        )
        _convOut.wrappedValue = Flux2ConvNCHW(
            inputChannels: blockOutChannels.last!,
            outputChannels: 2 * outChannels,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = convIn(x)
        for block in downBlocks {
            hidden = block(hidden)
        }
        hidden = midBlock(hidden)
        hidden = convNormOut(hidden)
        hidden = silu(hidden).asType(activationDType)
        return convOut(hidden)
    }
}

final class Flux2Decoder: Module, UnaryLayer {
    @ModuleInfo(key: "conv_in") private var convIn: Flux2ConvNCHW
    @ModuleInfo(key: "mid_block") private var midBlock: Flux2UNetMidBlock2D
    @ModuleInfo(key: "up_blocks") private var upBlocks: [Flux2UpDecoderBlock2D]
    @ModuleInfo(key: "conv_norm_out") private var convNormOut: Flux2GroupNormNCHW
    @ModuleInfo(key: "conv_out") private var convOut: Flux2ConvNCHW
    private let activationDType: DType
    private let evaluateBlocks: Bool

    init(
        inChannels: Int = 32,
        outChannels: Int = 3,
        blockOutChannels: [Int],
        layersPerBlock: Int,
        normGroups: Int,
        eps: Float,
        midBlockAddAttention: Bool,
        activationDType: DType = Flux2Precision.dtype,
        evaluateBlocks: Bool = false,
    ) {
        self.activationDType = activationDType
        self.evaluateBlocks = evaluateBlocks
        _convIn.wrappedValue = Flux2ConvNCHW(
            inputChannels: inChannels,
            outputChannels: blockOutChannels.last!,
        )
        _midBlock.wrappedValue = Flux2UNetMidBlock2D(
            channels: blockOutChannels.last!,
            eps: eps,
            groups: normGroups,
            addAttention: midBlockAddAttention,
            activationDType: activationDType,
        )
        let reversed = Array(blockOutChannels.reversed())
        _upBlocks.wrappedValue = reversed.enumerated().map { index, output in
            let previous = index == 0 ? output : reversed[index - 1]
            return Flux2UpDecoderBlock2D(
                inChannels: previous,
                outChannels: output,
                layers: layersPerBlock + 1,
                eps: eps,
                groups: normGroups,
                addUpsample: index != reversed.count - 1,
                activationDType: activationDType,
            )
        }
        _convNormOut.wrappedValue = Flux2GroupNormNCHW(
            channels: blockOutChannels[0],
            groups: normGroups,
            eps: eps,
            activationDType: activationDType,
        )
        _convOut.wrappedValue = Flux2ConvNCHW(
            inputChannels: blockOutChannels[0],
            outputChannels: outChannels,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = convIn(x)
        hidden = midBlock(hidden)
        for block in upBlocks {
            hidden = block(hidden)
            if evaluateBlocks {
                eval(hidden)
                MLX.Memory.clearCache()
            }
        }
        hidden = convNormOut(hidden)
        hidden = silu(hidden).asType(activationDType)
        return convOut(hidden)
    }
}

private final class Flux2BatchNormStats: Module {
    @ModuleInfo(key: "running_mean") var runningMean: MLXArray
    @ModuleInfo(key: "running_var") var runningVar: MLXArray
    let eps: Float

    init(features: Int, eps: Float = 1e-4) {
        _runningMean.wrappedValue = MLXArray.zeros([features], dtype: .float32)
        _runningVar.wrappedValue = MLXArray.ones([features], dtype: .float32)
        self.eps = eps
        super.init()
    }
}

public final class Flux2VAE: Module {
    public let config: Flux2VAEConfiguration
    public let scalingFactor: Float = 1.0
    public let shiftFactor: Float = 0.0

    public var supportsEncoding: Bool { encoder != nil && quantConv != nil }

    var encoderDirectory: URL?
    private let activationDType: DType
    private let encoderHiddenDType: DType?

    @ModuleInfo fileprivate var encoder: Flux2Encoder?
    @ModuleInfo fileprivate var decoder: Flux2Decoder
    @ModuleInfo(key: "quant_conv") fileprivate var quantConv: Conv2d?
    @ModuleInfo(key: "post_quant_conv") fileprivate var postQuantConv: Conv2d
    @ModuleInfo fileprivate var bn: Flux2BatchNormStats

    public convenience init(_ config: Flux2VAEConfiguration = Flux2VAEConfiguration()) {
        self.init(config, activationDType: Flux2Precision.dtype)
    }

    init(
        _ config: Flux2VAEConfiguration, activationDType: DType,
        encoderHiddenDType: DType? = nil, evaluateDecoderBlocks: Bool = false
    ) {
        self.activationDType = activationDType
        self.encoderHiddenDType = encoderHiddenDType
        self.config = config
        if config.includeEncoder {
            _encoder.wrappedValue = Self.makeEncoder(
                config, activationDType: activationDType, hiddenDType: encoderHiddenDType)
            _quantConv.wrappedValue = Self.makeQuantConv(config)
        } else {
            _encoder.wrappedValue = nil
            _quantConv.wrappedValue = nil
        }
        _decoder.wrappedValue = Flux2Decoder(
            inChannels: config.latentChannels,
            blockOutChannels: config.decoderBlockOutChannels,
            layersPerBlock: config.layersPerBlock,
            normGroups: config.normGroups,
            eps: config.eps,
            midBlockAddAttention: config.midBlockAddAttention,
            activationDType: activationDType,
            evaluateBlocks: evaluateDecoderBlocks,
        )
        _postQuantConv.wrappedValue = Conv2d(
            inputChannels: config.latentChannels,
            outputChannels: config.latentChannels,
            kernelSize: 1,
            padding: 0,
        )
        _bn.wrappedValue = Flux2BatchNormStats(
            features: 4 * config.latentChannels, eps: config.batchNormEps)
        super.init()
    }

    private static func makeEncoder(
        _ config: Flux2VAEConfiguration, activationDType: DType, hiddenDType: DType?
    ) -> Flux2Encoder {
        Flux2Encoder(
            outChannels: config.latentChannels,
            blockOutChannels: config.encoderBlockOutChannels,
            layersPerBlock: config.layersPerBlock,
            normGroups: config.normGroups,
            eps: config.eps,
            midBlockAddAttention: config.midBlockAddAttention, activationDType: activationDType,
            hiddenDType: hiddenDType)
    }

    private static func makeQuantConv(_ config: Flux2VAEConfiguration) -> Conv2d {
        Conv2d(
            inputChannels: 2 * config.latentChannels,
            outputChannels: 2 * config.latentChannels, kernelSize: 1, padding: 0)
    }

    public func encode(_ image: MLXArray) throws -> MLXArray {
        if !supportsEncoding, let encoderDirectory {
            try Task.checkCancellation()
            let encoder = Self.makeEncoder(
                config, activationDType: activationDType, hiddenDType: encoderHiddenDType)
            let quantConv = Self.makeQuantConv(config)
            try Flux2WeightLoader.loadVAEEncoderWeights(
                from: encoderDirectory, encoder: encoder, quantConv: quantConv)
            try Task.checkCancellation()
            update(
                modules: ModuleChildren.unflattened([
                    ("encoder", encoder as Module), ("quant_conv", quantConv as Module),
                ]))
        }
        guard let encoder, let quantConv else {
            throw Flux2Error.missingWeights(URL(fileURLWithPath: "Flux2VAE.encoder"))
        }
        var image = image
        if image.ndim == 5 {
            image = image[0..., 0..., 0, 0..., 0...]
        }
        var encoded = encoder(image)
        encoded = quantConv(encoded.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
        let mean = split(encoded, parts: 2, axis: 1)[0]
        return (mean - shiftFactor) * scalingFactor
    }

    public func decode(_ latents: MLXArray) -> MLXArray {
        var latents = latents
        if latents.ndim == 5 {
            latents = latents[0..., 0..., 0, 0..., 0...]
        }
        latents = latents / scalingFactor + shiftFactor
        latents = postQuantConv(latents.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
        return decoder(latents)
    }

    func encodePackedLatents(_ pixels: MLXArray) throws -> MLXArray {
        let raw = try encode(pixels)
        let (b, c, h, w) = (raw.dim(0), raw.dim(1), raw.dim(2), raw.dim(3))
        let packed = raw.reshaped(b, c, h / 2, 2, w / 2, 2)
            .transposed(0, 1, 3, 5, 2, 4).reshaped(b, c * 4, h / 2, w / 2)
        let mean = bn.runningMean.reshaped(1, -1, 1, 1).asType(packed.dtype)
        let std = sqrt(bn.runningVar.reshaped(1, -1, 1, 1) + bn.eps).asType(packed.dtype)
        return (packed - mean) / std
    }

    public func decodePackedLatents(_ packedLatents: MLXArray) -> MLXArray {
        var packedLatents = packedLatents
        if packedLatents.ndim == 5 {
            packedLatents = packedLatents[0..., 0..., 0, 0..., 0...]
        }
        let mean = bn.runningMean.reshaped(1, -1, 1, 1)
        let std = sqrt(bn.runningVar.reshaped(1, -1, 1, 1) + bn.eps)
        var latents = packedLatents * std + mean
        latents = Self.unpatchify(latents)
        return decode(latents)
    }

    public static func unpatchify(_ latents: MLXArray) -> MLXArray {
        let batch = latents.dim(0)
        let channels = latents.dim(1)
        let height = latents.dim(2)
        let width = latents.dim(3)
        var hidden = latents.reshaped(batch, channels / 4, 2, 2, height, width)
        hidden = hidden.transposed(0, 1, 4, 2, 5, 3)
        return hidden.reshaped(batch, channels / 4, height * 2, width * 2)
    }
}

public final class Flux2ImagePipeline: ImageGenerationModel {

    public let textEncoder: Flux2TextEncoder
    public let transformer: Flux2Transformer
    public let vae: Flux2VAE

    let defaultSteps: Int
    let defaultGuidance: Float
    let usesReferenceKVCache: Bool

    public init(
        textEncoder: Flux2TextEncoder,
        transformer: Flux2Transformer,
        vae: Flux2VAE,
        defaultSteps: Int = 4, defaultGuidance: Float = 1, usesReferenceKVCache: Bool = false,
    ) {
        self.textEncoder = textEncoder
        self.transformer = transformer
        self.vae = vae
        self.defaultSteps = defaultSteps
        self.defaultGuidance = defaultGuidance
        self.usesReferenceKVCache = usesReferenceKVCache
    }

    public func encodePrompt(
        _ prompt: String,
        tokenizer: any Tokenizer,
        maxLength: Int = 512,
        bucketed: Bool = true,
        hiddenStateLayers: [Int] = [9, 18, 27],
    )
        throws -> (embeddings: MLXArray, textIDs: MLXArray)
    {
        let tokens = try Flux2PromptTokenizer.tokenize(
            prompt: prompt,
            tokenizer: tokenizer,
            maxLength: maxLength,
            bucketed: bucketed,
        )
        let embeddings = try textEncoder.promptEmbeddings(
            inputIDs: tokens.inputIDs,
            attentionMask: tokens.attentionMask,
            hiddenStateLayers: hiddenStateLayers,
        )
        return (embeddings, Flux2PromptIDs.prepareTextIDs(for: embeddings))
    }

    public func generate(
        prompt: String,
        tokenizer: any Tokenizer,
        parameters: ImageGenerationParameters = ImageGenerationParameters(),
        maxLength: Int = 512,
        bucketed: Bool = true,
        hiddenStateLayers: [Int] = [9, 18, 27],
    )
        throws -> ImageGenerationResult
    {
        let promptInputs = try encodePrompt(
            prompt,
            tokenizer: tokenizer,
            maxLength: maxLength,
            bucketed: bucketed,
            hiddenStateLayers: hiddenStateLayers,
        )
        var negativeInputs: (embeddings: MLXArray, textIDs: MLXArray)?
        if (parameters.guidance ?? defaultGuidance) > 1.0 {
            negativeInputs = try encodePrompt(
                parameters.negativePrompt ?? " ",
                tokenizer: tokenizer,
                maxLength: maxLength,
                bucketed: bucketed,
                hiddenStateLayers: hiddenStateLayers,
            )
        }
        return try generate(
            promptEmbeddings: promptInputs.embeddings,
            textIDs: promptInputs.textIDs,
            negativePromptEmbeddings: negativeInputs?.embeddings,
            negativeTextIDs: negativeInputs?.textIDs,
            parameters: parameters,
        )
    }

    public func generate(
        promptEmbeddings: MLXArray,
        textIDs: MLXArray,
        parameters: ImageGenerationParameters,
    )
        throws -> ImageGenerationResult
    {
        try generate(
            promptEmbeddings: promptEmbeddings,
            textIDs: textIDs,
            negativePromptEmbeddings: nil,
            negativeTextIDs: nil,
            parameters: parameters,
        )
    }

    public func generate(
        promptEmbeddings: MLXArray,
        textIDs: MLXArray,
        negativePromptEmbeddings: MLXArray? = nil,
        negativeTextIDs: MLXArray? = nil,
        parameters: ImageGenerationParameters = ImageGenerationParameters(),
    )
        throws -> ImageGenerationResult
    {
        try Flux2Dimensions.validate(width: parameters.width, height: parameters.height)
        guard (parameters.steps ?? defaultSteps) >= 1 else {
            throw Flux2Error.invalidStepCount(
                (parameters.steps ?? defaultSteps))
        }

        let seed = parameters.seed ?? 0
        let prepared = Flux2Latents.preparePackedLatents(
            seed: seed,
            height: parameters.height,
            width: parameters.width,
            numLatentChannels: vae.config.latentChannels,
        )
        var latents = prepared.latents
        let scheduler = Flux2FlowMatchEulerDiscreteScheduler(
            imageSequenceLength: (parameters.height / 16) * (parameters.width / 16),
            inferenceSteps: (parameters.steps ?? defaultSteps),
        )

        for index in 0 ..< (parameters.steps ?? defaultSteps) {
            try Task.checkCancellation()
            var noise = predict(
                latents: latents,
                latentIDs: prepared.ids,
                promptEmbeddings: promptEmbeddings,
                textIDs: textIDs,
                timestep: scheduler.timesteps[index],
            )
            if (parameters.guidance ?? defaultGuidance) > 1.0,
                let negativePromptEmbeddings,
                let negativeTextIDs
            {
                let negativeNoise = predict(
                    latents: latents,
                    latentIDs: prepared.ids,
                    promptEmbeddings: negativePromptEmbeddings,
                    textIDs: negativeTextIDs,
                    timestep: scheduler.timesteps[index],
                )
                noise =
                    negativeNoise
                    + (parameters.guidance ?? defaultGuidance)
                    * (noise - negativeNoise)
            }
            latents = scheduler.step(noise: noise, stepIndex: index, latents: latents)
            eval(latents)
        }

        var packed = latents.reshaped(
            latents.dim(0),
            prepared.latentHeight,
            prepared.latentWidth,
            latents.dim(-1),
        )
        packed = packed.transposed(0, 3, 1, 2)
        let decoded = vae.decodePackedLatents(packed)
        let image = Self.imageArray(fromDecodedLatents: decoded)
        return ImageGenerationResult(
            image: image,
            seed: seed,
            width: parameters.width,
            height: parameters.height,
            steps: (parameters.steps ?? defaultSteps),
            guidance: (parameters.guidance ?? defaultGuidance),
            metadata: [
                "architecture": "dense",
                "vae_variant": "full",
            ],
        )
    }

    public func edit(
        prompt: String,
        referenceImages: [MLXArray],
        tokenizer: any Tokenizer,
        parameters: ImageGenerationParameters = ImageGenerationParameters(),
        maxLength: Int = 512,
        bucketed: Bool = true,
        hiddenStateLayers: [Int] = [9, 18, 27],
    )
        throws -> ImageGenerationResult
    {
        let promptInputs = try encodePrompt(
            prompt,
            tokenizer: tokenizer,
            maxLength: maxLength,
            bucketed: bucketed,
            hiddenStateLayers: hiddenStateLayers,
        )
        var negativeInputs: (embeddings: MLXArray, textIDs: MLXArray)?
        if (parameters.guidance ?? defaultGuidance) > 1.0 {
            negativeInputs = try encodePrompt(
                parameters.negativePrompt ?? " ",
                tokenizer: tokenizer,
                maxLength: maxLength,
                bucketed: bucketed,
                hiddenStateLayers: hiddenStateLayers,
            )
        }
        return try edit(
            promptEmbeddings: promptInputs.embeddings,
            textIDs: promptInputs.textIDs,
            referenceImages: referenceImages,
            negativePromptEmbeddings: negativeInputs?.embeddings,
            negativeTextIDs: negativeInputs?.textIDs,
            parameters: parameters,
        )
    }

    public func edit(
        promptEmbeddings: MLXArray,
        textIDs: MLXArray,
        referenceImages: [MLXArray],
        negativePromptEmbeddings: MLXArray? = nil,
        negativeTextIDs: MLXArray? = nil,
        parameters: ImageGenerationParameters = ImageGenerationParameters(),
    )
        throws -> ImageGenerationResult
    {
        try Flux2Dimensions.validate(width: parameters.width, height: parameters.height)
        guard (parameters.steps ?? defaultSteps) >= 1 else {
            throw Flux2Error.invalidStepCount(
                (parameters.steps ?? defaultSteps))
        }
        guard !referenceImages.isEmpty else {
            throw Flux2Error.missingWeights(URL(fileURLWithPath: "Flux2 reference image"))
        }

        let reference = try prepareReferenceLatents(referenceImages)
        let seed = parameters.seed ?? 0
        let prepared = Flux2Latents.preparePackedLatents(
            seed: seed,
            height: parameters.height,
            width: parameters.width,
            numLatentChannels: vae.config.latentChannels,
        )
        var latents = prepared.latents
        let scheduler = Flux2FlowMatchEulerDiscreteScheduler(
            imageSequenceLength: (parameters.height / 16) * (parameters.width / 16),
            inferenceSteps: (parameters.steps ?? defaultSteps),
        )

        let cache =
            usesReferenceKVCache && parameters.useReferenceCache
            ? Flux2KVCache(
                doubleLayers: transformer.transformerBlocks.count,
                singleLayers: transformer.singleTransformerBlocks.count,
            )
            : nil
        let negativeCache =
            usesReferenceKVCache && parameters.useReferenceCache
                && negativePromptEmbeddings != nil
            ? Flux2KVCache(
                doubleLayers: transformer.transformerBlocks.count,
                singleLayers: transformer.singleTransformerBlocks.count,
            )
            : nil

        for index in 0 ..< (parameters.steps ?? defaultSteps) {
            try Task.checkCancellation()
            var noise: MLXArray =
                if let cache {
                    predictEditWithKVCache(
                        latents: latents,
                        latentIDs: prepared.ids,
                        referenceLatents: reference.latents,
                        referenceIDs: reference.ids,
                        promptEmbeddings: promptEmbeddings,
                        textIDs: textIDs,
                        timestep: scheduler.timesteps[index],
                        cache: cache,
                    )
                } else {
                    predictEdit(
                        latents: latents,
                        latentIDs: prepared.ids,
                        referenceLatents: reference.latents,
                        referenceIDs: reference.ids,
                        promptEmbeddings: promptEmbeddings,
                        textIDs: textIDs,
                        timestep: scheduler.timesteps[index],
                    )
                }

            if (parameters.guidance ?? defaultGuidance) > 1.0,
                let negativePromptEmbeddings,
                let negativeTextIDs
            {
                let negativeNoise: MLXArray =
                    if let negativeCache {
                        predictEditWithKVCache(
                            latents: latents,
                            latentIDs: prepared.ids,
                            referenceLatents: reference.latents,
                            referenceIDs: reference.ids,
                            promptEmbeddings: negativePromptEmbeddings,
                            textIDs: negativeTextIDs,
                            timestep: scheduler.timesteps[index],
                            cache: negativeCache,
                        )
                    } else {
                        predictEdit(
                            latents: latents,
                            latentIDs: prepared.ids,
                            referenceLatents: reference.latents,
                            referenceIDs: reference.ids,
                            promptEmbeddings: negativePromptEmbeddings,
                            textIDs: negativeTextIDs,
                            timestep: scheduler.timesteps[index],
                        )
                    }
                noise =
                    negativeNoise
                    + (parameters.guidance ?? defaultGuidance)
                    * (noise - negativeNoise)
            }
            latents = scheduler.step(noise: noise, stepIndex: index, latents: latents)
            eval(latents)
        }

        cache?.clear()
        negativeCache?.clear()

        var packed = latents.reshaped(
            latents.dim(0),
            prepared.latentHeight,
            prepared.latentWidth,
            latents.dim(-1),
        )
        packed = packed.transposed(0, 3, 1, 2)
        let image = Self.imageArray(fromDecodedLatents: vae.decodePackedLatents(packed))
        return ImageGenerationResult(
            image: image,
            seed: seed,
            width: parameters.width,
            height: parameters.height,
            steps: (parameters.steps ?? defaultSteps),
            guidance: (parameters.guidance ?? defaultGuidance),
            metadata: [
                "architecture": "dense",
                "vae_variant": "full",
                "reference_count": "\(referenceImages.count)",
                "uses_reference_kv_cache": "\(usesReferenceKVCache)",
            ],
        )
    }

    public func prepareReferenceLatents(_ images: [MLXArray])
        throws -> (latents: MLXArray, ids: MLXArray)
    {
        var packedRefs: [MLXArray] = []
        var refIDs: [MLXArray] = []
        for (index, image) in images.enumerated() {
            var latents = try vae.encode(image)
            latents = Self.cropToEvenSpatial(latents)
            latents = Flux2Latents.patchify(latents)
            let mean = vae.bn.runningMean.reshaped(1, -1, 1, 1).asType(latents.dtype)
            let std = sqrt(vae.bn.runningVar.reshaped(1, -1, 1, 1) + vae.bn.eps)
                .asType(latents.dtype)
            latents = (latents - mean) / std
            packedRefs.append(Flux2Latents.pack(latents))
            refIDs.append(Flux2Latents.prepareGridIDs(latents, tCoordinate: 10 + 10 * index))
        }
        let referenceLatents = concatenated(packedRefs, axis: 1)
        let referenceIDs = concatenated(refIDs, axis: 1)
        eval(referenceLatents, referenceIDs)
        return (referenceLatents, referenceIDs)
    }

    public func predict(
        latents: MLXArray,
        latentIDs: MLXArray,
        promptEmbeddings: MLXArray,
        textIDs: MLXArray,
        timestep: MLXArray,
    )
        -> MLXArray
    {
        transformer(
            hiddenStates: latents,
            encoderHiddenStates: promptEmbeddings,
            timestep: timestep,
            imageIDs: latentIDs,
            textIDs: textIDs,
            guidance: nil,
        )
    }

    public func predictEdit(
        latents: MLXArray,
        latentIDs: MLXArray,
        referenceLatents: MLXArray,
        referenceIDs: MLXArray,
        promptEmbeddings: MLXArray,
        textIDs: MLXArray,
        timestep: MLXArray,
    )
        -> MLXArray
    {
        let hidden = concatenated([latents, referenceLatents], axis: 1)
        let imageIDs = concatenated([latentIDs, referenceIDs], axis: 1)
        let noise = transformer(
            hiddenStates: hidden,
            encoderHiddenStates: promptEmbeddings,
            timestep: timestep,
            imageIDs: imageIDs,
            textIDs: textIDs,
            guidance: nil,
        )
        return noise[0..., ..<latents.dim(1), 0...]
    }

    public func predictEditWithKVCache(
        latents: MLXArray,
        latentIDs: MLXArray,
        referenceLatents: MLXArray,
        referenceIDs: MLXArray,
        promptEmbeddings: MLXArray,
        textIDs: MLXArray,
        timestep: MLXArray,
        cache: Flux2KVCache,
    )
        -> MLXArray
    {
        if cache.referenceTokenCount == 0 {
            let hidden = concatenated([referenceLatents, latents], axis: 1)
            let imageIDs = concatenated([referenceIDs, latentIDs], axis: 1)
            return transformer(
                hiddenStates: hidden,
                encoderHiddenStates: promptEmbeddings,
                timestep: timestep,
                imageIDs: imageIDs,
                textIDs: textIDs,
                guidance: nil,
                kvCache: cache,
                kvCacheMode: "extract",
                referenceTokenCount: referenceLatents.dim(1),
            )
        }
        return transformer(
            hiddenStates: latents,
            encoderHiddenStates: promptEmbeddings,
            timestep: timestep,
            imageIDs: latentIDs,
            textIDs: textIDs,
            guidance: nil,
            kvCache: cache,
            kvCacheMode: "cached",
        )
    }

    private static func cropToEvenSpatial(_ latents: MLXArray) -> MLXArray {
        var latents = latents
        if latents.dim(2) % 2 != 0 {
            latents = latents[0..., 0..., ..<(latents.dim(2) - 1), 0...]
        }
        if latents.dim(3) % 2 != 0 {
            latents = latents[0..., 0..., 0..., ..<(latents.dim(3) - 1)]
        }
        return latents
    }

    public static func imageArray(fromDecodedLatents decoded: MLXArray) -> MLXArray {
        var images = clip(decoded / 2 + 0.5, min: Float32(0), max: Float32(1))
        images = images.transposed(0, 2, 3, 1).asType(.float32)
        let image = round(images * 255).asType(.uint8)[0]
        eval(image)
        return image
    }
}

public enum Flux2WeightLoader {
    public static func loadPipeline(
        from modelDirectory: URL,
    )
        throws -> Flux2ImagePipeline
    {
        let behavior = try ImageCheckpointMetadata.fluxBehavior(in: modelDirectory)
        let textEncoder = Flux2TextEncoder(
            try .read(from: modelDirectory, path: "text_encoder/config.json"))
        let transformer = Flux2Transformer(
            try .read(from: modelDirectory, path: "transformer/config.json"))
        let vae = Flux2VAE(try .read(from: modelDirectory))
        try loadTextEncoderWeights(
            from: modelDirectory.appending(path: "text_encoder"),
            into: textEncoder,
        )
        try loadTransformerWeights(
            from: modelDirectory.appending(path: "transformer"),
            into: transformer,
        )
        try loadVAEWeights(
            from: modelDirectory.appending(path: "vae"),
            into: vae,
        )
        vae.encoderDirectory = modelDirectory.appending(path: "vae")
        return Flux2ImagePipeline(
            textEncoder: textEncoder,
            transformer: transformer,
            vae: vae, defaultSteps: behavior.steps, defaultGuidance: behavior.guidance,
            usesReferenceKVCache: behavior.referenceCache,
        )
    }

    public static func loadTextEncoderWeights(from directory: URL, into model: Flux2TextEncoder)
        throws
    {
        let raw = try loadSafetensors(from: directory)
        let weights = sanitizeTextEncoderWeights(raw)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
    }

    public static func loadTransformerWeights(from directory: URL, into model: Flux2Transformer)
        throws
    {
        let raw = try loadSafetensors(from: directory)
        let weights = sanitizeTransformerWeights(raw)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
    }

    public static func loadVAEWeights(
        from directory: URL,
        into model: Flux2VAE,
        includeEncoder: Bool = false,
    )
        throws
    {
        let raw = try loadSafetensors(from: directory)
        let weights = sanitizeVAEWeights(raw, includeEncoder: includeEncoder)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
    }

    static func loadVAEEncoderWeights(
        from directory: URL, encoder: Flux2Encoder, quantConv: Conv2d
    ) throws {
        let raw = try loadSafetensors(from: directory).filter {
            $0.key.hasPrefix("encoder.") || $0.key.hasPrefix("quant_conv.")
        }
        let weights = sanitizeVAEWeights(raw, includeEncoder: true)
        for (prefix, module) in [
            ("encoder.", encoder as Module), ("quant_conv.", quantConv as Module),
        ] {
            let parameters = weights.filter { $0.key.hasPrefix(prefix) }.map {
                (String($0.key.dropFirst(prefix.count)), $0.value)
            }
            try module.update(parameters: ModuleParameters.unflattened(parameters), verify: [.all])
        }
        eval(encoder, quantConv)
    }

    public static func sanitizeTextEncoderWeights(_ weights: [String: MLXArray])
        -> [String: MLXArray]
    {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            guard key.hasPrefix("model.") else {
                continue
            }
            let stripped = String(key.dropFirst("model.".count))
            guard
                stripped.hasPrefix("embed_tokens.")
                    || stripped.hasPrefix("layers.")
                    || stripped.hasPrefix("norm.")
            else {
                continue
            }
            sanitized[stripped] = value.asType(Flux2Precision.dtype)
        }
        return sanitized
    }

    public static func sanitizeTransformerWeights(_ weights: [String: MLXArray])
        -> [String: MLXArray]
    {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            var mapped = key.replacingOccurrences(
                of: "time_guidance_embed.timestep_embedder.",
                with: "time_guidance_embed.",
            )
            mapped = mapped.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
            sanitized[mapped] = value.asType(Flux2Precision.dtype)
        }
        return sanitized
    }

    public static func sanitizeVAEWeights(
        _ weights: [String: MLXArray],
        includeEncoder: Bool = false,
    )
        -> [String: MLXArray]
    {
        sanitizeVAEWeights(
            weights, includeEncoder: includeEncoder, channelsLast: false,
            dtype: Flux2Precision.dtype)
    }

    static func sanitizeVAEWeights(
        _ weights: [String: MLXArray], includeEncoder: Bool,
        channelsLast: Bool, dtype: DType
    ) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasSuffix(".num_batches_tracked") {
                continue
            }
            let keep =
                key.hasPrefix("decoder.")
                || key.hasPrefix("post_quant_conv.")
                || key.hasPrefix("bn.")
                || (includeEncoder && key.hasPrefix("encoder."))
                || (includeEncoder && key.hasPrefix("quant_conv."))
            guard keep else {
                continue
            }
            var mapped = key.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
            mapped = mapVAEWrapperKey(mapped, prefix: "decoder.conv_in.", child: "conv")
            mapped = mapVAEWrapperKey(mapped, prefix: "decoder.conv_out.", child: "conv")
            mapped = mapVAEWrapperKey(mapped, prefix: "decoder.conv_norm_out.", child: "norm")
            mapped = mapVAEWrapperKey(mapped, prefix: "encoder.conv_in.", child: "conv")
            mapped = mapVAEWrapperKey(mapped, prefix: "encoder.conv_out.", child: "conv")
            mapped = mapVAEWrapperKey(mapped, prefix: "encoder.conv_norm_out.", child: "norm")
            var tensor = value.dtype.isFloatingPoint ? value.asType(dtype) : value
            if tensor.ndim == 4, !channelsLast {
                tensor = tensor.transposed(0, 2, 3, 1)
            }
            sanitized[mapped] = tensor
        }
        return sanitized
    }

    private static func mapVAEWrapperKey(_ key: String, prefix: String, child: String) -> String {
        guard key.hasPrefix(prefix) else {
            return key
        }
        return "\(prefix)\(child).\(key.dropFirst(prefix.count))"
    }

    private static func loadSafetensors(from directory: URL) throws -> [String: MLXArray] {
        guard
            let enumerator = FileManager.default.enumerator(
                at: directory,
                includingPropertiesForKeys: nil,
            )
        else {
            throw Flux2Error.missingWeights(directory)
        }
        var weights = [String: MLXArray]()
        for case let url as URL in enumerator {
            guard url.pathExtension == "safetensors",
                !url.lastPathComponent.hasPrefix("._")
            else {
                continue
            }
            let (arrays, _) = try loadArraysAndMetadata(url: url)
            for (key, value) in arrays {
                weights[key] = value
            }
        }
        if weights.isEmpty {
            throw Flux2Error.missingWeights(directory)
        }
        return weights
    }
}
