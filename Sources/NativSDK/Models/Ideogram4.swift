import Foundation
import MLX
import MLXNN

public enum Ideogram4Error: LocalizedError {
    case invalidDimensions(width: Int, height: Int)
    case invalidStepCount(Int)
    case emptyPrompt
    case missingWeights(URL)
    case invalidModelLayout(URL)
    case invalidPromptInputs(String)
    case missingComponent(String)

    public var errorDescription: String? {
        switch self {
        case .invalidDimensions(let width, let height):
            "Ideogram 4 image dimensions must be in [256, 2048], multiples of 16, and at most 6:1; got \(width)x\(height)."
        case .invalidStepCount(let steps):
            "Ideogram 4 image generation requires at least one denoising step; got \(steps)."
        case .emptyPrompt:
            "Ideogram 4 prompt must not be empty."
        case .missingWeights(let url):
            "Missing Ideogram 4 weights under \(url.path)."
        case .invalidModelLayout(let url):
            "\(url.path) is not an Ideogram 4 snapshot."
        case .invalidPromptInputs(let reason):
            "Invalid Ideogram 4 prompt inputs: \(reason)"
        case .missingComponent(let name):
            "Ideogram 4 pipeline is missing \(name)."
        }
    }
}

public struct Ideogram4TransformerConfiguration: Sendable, Equatable {
    public let embDim: Int
    public let numLayers: Int
    public let numHeads: Int
    public let intermediateSize: Int
    public let adaLNDim: Int
    public let inChannels: Int
    public let llmFeaturesDim: Int
    public let ropeTheta: Float
    public let mropeSection: [Int]
    public let normEps: Float

    public init(
        embDim: Int = 4608,
        numLayers: Int = 34,
        numHeads: Int = 18,
        intermediateSize: Int = 12288,
        adaLNDim: Int = 512,
        inChannels: Int = 128,
        llmFeaturesDim: Int = 4096 * 13,
        ropeTheta: Float = 5_000_000,
        mropeSection: [Int] = [24, 20, 20],
        normEps: Float = 1e-5,
    ) {
        self.embDim = embDim
        self.numLayers = numLayers
        self.numHeads = numHeads
        self.intermediateSize = intermediateSize
        self.adaLNDim = adaLNDim
        self.inChannels = inChannels
        self.llmFeaturesDim = llmFeaturesDim
        self.ropeTheta = ropeTheta
        self.mropeSection = mropeSection
        self.normEps = normEps
    }

    public var headDim: Int {
        embDim / numHeads
    }
}

public enum Ideogram4Dimensions {
    public static func validate(width: Int, height: Int) throws {
        let ratio = max(Float(width) / Float(height), Float(height) / Float(width))
        guard (256 ... 2048).contains(width),
            (256 ... 2048).contains(height),
            width.isMultiple(of: 16),
            height.isMultiple(of: 16),
            ratio <= 6.0
        else {
            throw Ideogram4Error.invalidDimensions(width: width, height: height)
        }
    }
}

public struct Ideogram4PreparedPrompt: Sendable {
    public let text: String
    public let isJSONCaption: Bool
    public let isStructuredCaption: Bool
    public let wasWrapped: Bool
    public let warnings: [String]

    public init(
        text: String,
        isJSONCaption: Bool,
        isStructuredCaption: Bool,
        wasWrapped: Bool,
        warnings: [String] = [],
    ) {
        self.text = text
        self.isJSONCaption = isJSONCaption
        self.isStructuredCaption = isStructuredCaption
        self.wasWrapped = wasWrapped
        self.warnings = warnings
    }
}

public enum Ideogram4Prompting {
    public static func preparePrompt(_ prompt: String, autoJSONCaption: Bool = true)
        throws -> Ideogram4PreparedPrompt
    {
        let stripped = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !stripped.isEmpty else {
            throw Ideogram4Error.emptyPrompt
        }
        if stripped.hasPrefix("{") {
            let issues = captionWarnings(stripped)
            return Ideogram4PreparedPrompt(
                text: prompt,
                isJSONCaption: true,
                isStructuredCaption: issues.isEmpty,
                wasWrapped: false,
                warnings: issues,
            )
        }
        if !autoJSONCaption {
            return Ideogram4PreparedPrompt(
                text: prompt,
                isJSONCaption: false,
                isStructuredCaption: false,
                wasWrapped: false,
            )
        }
        let caption: [String: Any] = [
            "high_level_description": stripped,
            "compositional_deconstruction": [
                "background":
                    "The setting, environment, and surrounding context implied by the prompt.",
                "elements": [
                    ["type": "obj", "desc": stripped]
                ],
            ],
        ]
        let data = try JSONSerialization.data(withJSONObject: caption, options: [])
        let formatted = String(data: data, encoding: .utf8) ?? stripped
        return Ideogram4PreparedPrompt(
            text: formatted,
            isJSONCaption: true,
            isStructuredCaption: true,
            wasWrapped: true,
        )
    }

    private static func captionWarnings(_ text: String) -> [String] {
        guard let data = text.data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else {
            return ["Invalid Ideogram 4 JSON caption."]
        }
        guard let compositional = object["compositional_deconstruction"] as? [String: Any] else {
            return ["Ideogram 4 JSON caption should include a compositional_deconstruction object."]
        }
        var warnings: [String] = []
        if (compositional["background"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines)
            .isEmpty != false
        {
            warnings.append(
                "Ideogram 4 JSON caption should include compositional_deconstruction.background.")
        }
        if !(compositional["elements"] is [[String: Any]]) {
            warnings.append(
                "Ideogram 4 JSON caption should include compositional_deconstruction.elements.")
        }
        return warnings
    }
}

public struct Ideogram4PromptInputs {
    public let llmFeatures: MLXArray
    public let positionIDs: MLXArray
    public let segmentIDs: MLXArray
    public let indicator: MLXArray
    public let tokenIDs: MLXArray
    public let numTextTokens: Int
    public let numImageTokens: Int
    public let gridHeight: Int
    public let gridWidth: Int
    public let preparedPrompt: Ideogram4PreparedPrompt
}

public struct Ideogram4SamplerPreset: Sendable, Equatable {
    public let name: String
    public let numSteps: Int
    public let guidanceSchedule: [Float]
    public let mu: Double
    public let std: Double

    public init(
        name: String, numSteps: Int, guidanceSchedule: [Float], mu: Double, std: Double = 1.0
    ) {
        self.name = name
        self.numSteps = numSteps
        self.guidanceSchedule = guidanceSchedule
        self.mu = mu
        self.std = std
    }

    public static let v4Default20 = Ideogram4SamplerPreset(
        name: "V4_DEFAULT_20",
        numSteps: 20,
        guidanceSchedule: Array(repeating: 3.0, count: 2) + Array(repeating: 7.0, count: 18),
        mu: 0.0,
        std: 1.75,
    )
}

public struct Ideogram4LogitNormalSchedule {
    public let mean: Double
    public let std: Double
    public let logSNRMin: Double
    public let logSNRMax: Double

    public init(
        mean: Double, std: Double = 1.0, logSNRMin: Double = -15.0, logSNRMax: Double = 18.0
    ) {
        self.mean = mean
        self.std = std
        self.logSNRMin = logSNRMin
        self.logSNRMax = logSNRMax
    }

    public static func forResolution(
        width: Int,
        height: Int,
        knownResolution: (width: Int, height: Int) = (512, 512),
        knownMean: Double = 1.0,
        std: Double = 1.0,
    )
        -> Ideogram4LogitNormalSchedule
    {
        let pixels = Double(width * height)
        let knownPixels = Double(knownResolution.width * knownResolution.height)
        return Ideogram4LogitNormalSchedule(
            mean: knownMean + 0.5 * Foundation.log(pixels / knownPixels), std: std)
    }

    public func value(at t: Double) -> Float {
        let tMin = 1.0 / (1.0 + Foundation.exp(0.5 * logSNRMax))
        let tMax = 1.0 / (1.0 + Foundation.exp(0.5 * logSNRMin))
        if t <= 0.0 {
            return Float(tMax)
        }
        if t >= 1.0 {
            return Float(tMin)
        }
        let z = Self.inverseNormalCDF(t)
        let shifted = mean + std * z
        let value = 1.0 - (1.0 / (1.0 + Foundation.exp(-shifted)))
        return Float(min(max(value, tMin), tMax))
    }

    private static func inverseNormalCDF(_ p: Double) -> Double {
        let a = [
            -3.969683028665376e+01, 2.209460984245205e+02,
            -2.759285104469687e+02, 1.383577518672690e+02,
            -3.066479806614716e+01, 2.506628277459239e+00,
        ]
        let b = [
            -5.447609879822406e+01, 1.615858368580409e+02,
            -1.556989798598866e+02, 6.680131188771972e+01,
            -1.328068155288572e+01,
        ]
        let c = [
            -7.784894002430293e-03, -3.223964580411365e-01,
            -2.400758277161838e+00, -2.549732539343734e+00,
            4.374664141464968e+00, 2.938163982698783e+00,
        ]
        let d = [
            7.784695709041462e-03, 3.224671290700398e-01,
            2.445134137142996e+00, 3.754408661907416e+00,
        ]
        let pLow = 0.02425
        let pHigh = 1.0 - pLow
        let clipped = min(max(p, Double.leastNonzeroMagnitude), 1.0 - Double.leastNonzeroMagnitude)

        if clipped < pLow {
            let q = Foundation.sqrt(-2.0 * Foundation.log(clipped))
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        }
        if clipped > pHigh {
            let q = Foundation.sqrt(-2.0 * Foundation.log(1.0 - clipped))
            return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        }
        let q = clipped - 0.5
        let r = q * q
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    }
}

private enum Ideogram4Constants {
    static let llmTokenIndicator: Int32 = 3
    static let outputImageIndicator: Int32 = 2
    static let imagePositionOffset: Int32 = 65536
    static let patchSize = 2
    static let aeScaleFactor = 8
    static let latentDim = 128
    static let maxTextTokens = 2048
    static let activationLayers = [0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 35]
}

private final class Ideogram4RMSNorm: Module, UnaryLayer {
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

private enum Ideogram4AttentionMath {
    static func applyRoPE(queries: MLXArray, keys: MLXArray, cosines: MLXArray, sines: MLXArray)
        -> (MLXArray, MLXArray)
    {
        let cosValue = cosines.expandedDimensions(axis: 1)
        let sinValue = sines.expandedDimensions(axis: 1)
        return (
            queries * cosValue + RotaryEmbeddingOps.rotateHalf(queries) * sinValue,
            keys * cosValue + RotaryEmbeddingOps.rotateHalf(keys) * sinValue,
        )
    }
}

private final class Ideogram4MRoPE: Module {
    let headDim: Int
    let base: Float
    let mropeSection: [Int]
    let invFreqValues: [Float]
    let axisSelectorValues: [Int32]

    init(headDim: Int, base: Float, mropeSection: [Int]) {
        self.headDim = headDim
        self.base = base
        self.mropeSection = mropeSection
        invFreqValues = stride(from: 0, to: headDim, by: 2).map { index in
            1.0 / Foundation.pow(base, Float(index) / Float(headDim))
        }
        var selector = Array(repeating: Int32(0), count: headDim / 2)
        for (axis, offset) in [(1, 1), (2, 2)] {
            let upper = min(
                (mropeSection.indices.contains(axis) ? mropeSection[axis] : 0) * 3, selector.count)
            if offset < upper {
                for index in stride(from: offset, to: upper, by: 3) {
                    selector[index] = Int32(axis)
                }
            }
        }
        axisSelectorValues = selector
        super.init()
    }

    func callAsFunction(_ positionIDs: MLXArray) -> (MLXArray, MLXArray) {
        let selector = MLXArray(axisSelectorValues)
        let selected = positionIDs.take(selector, axis: 2).asType(.float32)
        let inv = MLXArray(invFreqValues).reshaped(1, 1, -1)
        let frequencies = selected * inv
        let embeddings = concatenated([frequencies, frequencies], axis: -1)
        return (cos(embeddings), sin(embeddings))
    }
}

private final class Ideogram4Attention: Module {
    let hiddenSize: Int
    let numHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo var qkv: Linear
    @ModuleInfo(key: "norm_q") var normQ: Ideogram4RMSNorm
    @ModuleInfo(key: "norm_k") var normK: Ideogram4RMSNorm
    @ModuleInfo var o: Linear

    init(hiddenSize: Int, numHeads: Int, eps: Float = 1e-5) {
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        headDim = hiddenSize / numHeads
        scale = 1.0 / sqrt(Float(headDim))
        _qkv.wrappedValue = Linear(hiddenSize, hiddenSize * 3, bias: false)
        _normQ.wrappedValue = Ideogram4RMSNorm(dimensions: headDim, eps: eps)
        _normK.wrappedValue = Ideogram4RMSNorm(dimensions: headDim, eps: eps)
        _o.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        segmentIDs: MLXArray,
        cosines: MLXArray,
        sines: MLXArray,
    )
        -> MLXArray
    {
        let batch = x.dim(0)
        let seqLen = x.dim(1)
        let projected = qkv(x).reshaped(batch, seqLen, 3, numHeads, headDim)
        var queries = projected[0..., 0..., 0, 0..., 0...]
        var keys = projected[0..., 0..., 1, 0..., 0...]
        var values = projected[0..., 0..., 2, 0..., 0...]

        queries = normQ(queries).transposed(0, 2, 1, 3)
        keys = normK(keys).transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)
        (queries, keys) = Ideogram4AttentionMath.applyRoPE(
            queries: queries,
            keys: keys,
            cosines: cosines,
            sines: sines,
        )

        let sameSegment = segmentIDs[0..., 0..., .newAxis] .== segmentIDs[0..., .newAxis, 0...]
        let zeros = MLXArray.zeros(sameSegment.shape, dtype: x.dtype)
        let negInf = MLXArray.zeros(sameSegment.shape, dtype: x.dtype) - Float.infinity
        let mask = MLX.where(sameSegment, zeros, negInf).expandedDimensions(axis: 1)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries.asType(.float32),
            keys: keys.asType(.float32),
            values: values.asType(.float32),
            scale: scale,
            mask: mask,
        )
        return o(output.asType(x.dtype).transposed(0, 2, 1, 3).reshaped(batch, seqLen, hiddenSize))
    }
}

private final class Ideogram4MLP: Module, UnaryLayer {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear

    init(dim: Int, hiddenDim: Int) {
        _w1.wrappedValue = Linear(dim, hiddenDim, bias: false)
        _w2.wrappedValue = Linear(hiddenDim, dim, bias: false)
        _w3.wrappedValue = Linear(dim, hiddenDim, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

private final class Ideogram4TransformerBlock: Module {
    @ModuleInfo var attention: Ideogram4Attention
    @ModuleInfo(key: "feed_forward") var feedForward: Ideogram4MLP
    @ModuleInfo(key: "attention_norm1") var attentionNorm1: Ideogram4RMSNorm
    @ModuleInfo(key: "ffn_norm1") var ffnNorm1: Ideogram4RMSNorm
    @ModuleInfo(key: "attention_norm2") var attentionNorm2: Ideogram4RMSNorm
    @ModuleInfo(key: "ffn_norm2") var ffnNorm2: Ideogram4RMSNorm
    @ModuleInfo(key: "adaln_modulation") var adaLNModulation: Linear

    init(hiddenSize: Int, intermediateSize: Int, numHeads: Int, normEps: Float, adaLNDim: Int) {
        _attention.wrappedValue = Ideogram4Attention(
            hiddenSize: hiddenSize, numHeads: numHeads, eps: 1e-5)
        _feedForward.wrappedValue = Ideogram4MLP(dim: hiddenSize, hiddenDim: intermediateSize)
        _attentionNorm1.wrappedValue = Ideogram4RMSNorm(dimensions: hiddenSize, eps: normEps)
        _ffnNorm1.wrappedValue = Ideogram4RMSNorm(dimensions: hiddenSize, eps: normEps)
        _attentionNorm2.wrappedValue = Ideogram4RMSNorm(dimensions: hiddenSize, eps: normEps)
        _ffnNorm2.wrappedValue = Ideogram4RMSNorm(dimensions: hiddenSize, eps: normEps)
        _adaLNModulation.wrappedValue = Linear(adaLNDim, 4 * hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        segmentIDs: MLXArray,
        cosines: MLXArray,
        sines: MLXArray,
        adaLNInput: MLXArray,
    )
        -> MLXArray
    {
        let pieces = split(adaLNModulation(adaLNInput), parts: 4, axis: -1)
        let scaleMSA = 1.0 + pieces[0]
        let gateMSA = tanh(pieces[1])
        let scaleMLP = 1.0 + pieces[2]
        let gateMLP = tanh(pieces[3])
        var hidden = x
        let attended = attention(
            attentionNorm1(hidden) * scaleMSA,
            segmentIDs: segmentIDs,
            cosines: cosines,
            sines: sines,
        )
        hidden = hidden + gateMSA * attentionNorm2(attended)
        let mlpOut = feedForward(ffnNorm1(hidden) * scaleMLP)
        return hidden + gateMLP * ffnNorm2(mlpOut)
    }
}

private enum Ideogram4EmbeddingMath {
    static func sinusoidalEmbedding(_ t: MLXArray, dimensions: Int, scale: Float = 1e4) -> MLXArray
    {
        let half = dimensions / 2
        let frequency = Foundation.log(scale) / Float(half - 1)
        let freq = exp(arange(half, dtype: .float32) * -frequency)
        let emb = t.asType(.float32).expandedDimensions(axis: -1) * freq
        var output = concatenated([sin(emb), cos(emb)], axis: -1)
        if dimensions % 2 == 1 {
            output = padded(
                output,
                widths: Array(repeating: .init((0, 0)), count: output.ndim - 1) + [.init((0, 1))])
        }
        return output
    }

    static func layerNormNoAffine(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
        let dtype = x.dtype
        let meanValue = mean(x.asType(.float32), axis: -1, keepDims: true)
        let centered = x - meanValue.asType(dtype)
        let variance = mean(
            centered.asType(.float32) * centered.asType(.float32), axis: -1, keepDims: true)
        return centered * rsqrt(variance.asType(dtype) + eps)
    }
}

private final class Ideogram4EmbedScalar: Module {
    let dim: Int
    let rangeMin: Float
    let rangeMax: Float

    @ModuleInfo(key: "mlp_in") var mlpIn: Linear
    @ModuleInfo(key: "mlp_out") var mlpOut: Linear

    init(dim: Int, inputRange: (Float, Float)) {
        self.dim = dim
        rangeMin = inputRange.0
        rangeMax = inputRange.1
        _mlpIn.wrappedValue = Linear(dim, dim, bias: true)
        _mlpOut.wrappedValue = Linear(dim, dim, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let scaled = 1e4 * (x.asType(.float32) - rangeMin) / (rangeMax - rangeMin)
        let embedding = Ideogram4EmbeddingMath.sinusoidalEmbedding(scaled, dimensions: dim)
            .asType(mlpIn.weight.dtype)
        return mlpOut(silu(mlpIn(embedding)))
    }
}

private final class Ideogram4FinalLayer: Module {
    @ModuleInfo var linear: Linear
    @ModuleInfo(key: "adaln_modulation") var adaLNModulation: Linear

    init(hiddenSize: Int, outChannels: Int, adaLNDim: Int) {
        _linear.wrappedValue = Linear(hiddenSize, outChannels, bias: true)
        _adaLNModulation.wrappedValue = Linear(adaLNDim, hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, c: MLXArray) -> MLXArray {
        let scale = 1.0 + adaLNModulation(silu(c))
        return linear(Ideogram4EmbeddingMath.layerNormNoAffine(x) * scale)
    }
}

public final class Ideogram4Transformer: Module {
    public let config: Ideogram4TransformerConfiguration

    @ModuleInfo(key: "input_proj") fileprivate var inputProj: Linear
    @ModuleInfo(key: "llm_cond_norm") fileprivate var llmCondNorm: Ideogram4RMSNorm
    @ModuleInfo(key: "llm_cond_proj") fileprivate var llmCondProj: Linear
    @ModuleInfo(key: "t_embedding") fileprivate var tEmbedding: Ideogram4EmbedScalar
    @ModuleInfo(key: "adaln_proj") fileprivate var adaLNProj: Linear
    @ModuleInfo(key: "embed_image_indicator") fileprivate var embedImageIndicator: Embedding
    @ModuleInfo(key: "rotary_emb") fileprivate var rotaryEmbedding: Ideogram4MRoPE
    @ModuleInfo fileprivate var layers: [Ideogram4TransformerBlock]
    @ModuleInfo(key: "final_layer") fileprivate var finalLayer: Ideogram4FinalLayer

    public init(_ config: Ideogram4TransformerConfiguration = Ideogram4TransformerConfiguration()) {
        self.config = config
        _inputProj.wrappedValue = Linear(config.inChannels, config.embDim, bias: true)
        _llmCondNorm.wrappedValue = Ideogram4RMSNorm(dimensions: config.llmFeaturesDim, eps: 1e-6)
        _llmCondProj.wrappedValue = Linear(config.llmFeaturesDim, config.embDim, bias: true)
        _tEmbedding.wrappedValue = Ideogram4EmbedScalar(dim: config.embDim, inputRange: (0.0, 1.0))
        _adaLNProj.wrappedValue = Linear(config.embDim, config.adaLNDim, bias: true)
        _embedImageIndicator.wrappedValue = Embedding(embeddingCount: 2, dimensions: config.embDim)
        _rotaryEmbedding.wrappedValue = Ideogram4MRoPE(
            headDim: config.headDim,
            base: config.ropeTheta,
            mropeSection: config.mropeSection,
        )
        _layers.wrappedValue = (0 ..< config.numLayers).map { _ in
            Ideogram4TransformerBlock(
                hiddenSize: config.embDim,
                intermediateSize: config.intermediateSize,
                numHeads: config.numHeads,
                normEps: config.normEps,
                adaLNDim: config.adaLNDim,
            )
        }
        _finalLayer.wrappedValue = Ideogram4FinalLayer(
            hiddenSize: config.embDim,
            outChannels: config.inChannels,
            adaLNDim: config.adaLNDim,
        )
        super.init()
    }

    public func callAsFunction(
        llmFeatures: MLXArray?,
        x input: MLXArray,
        t: MLXArray,
        positionIDs: MLXArray,
        segmentIDs: MLXArray,
        indicator: MLXArray,
    )
        -> MLXArray
    {
        let parameterDType = inputProj.weight.dtype
        var x = input.asType(parameterDType)
        var t = t.asType(parameterDType)
        if t.ndim == 0 {
            t = MLXArray.zeros([x.dim(0)], dtype: parameterDType) + t.item(Float.self)
        }
        var llmFeatures = llmFeatures?.asType(parameterDType)

        let llmTokenMask = (indicator .== MLXArray(Ideogram4Constants.llmTokenIndicator))
            .asType(parameterDType)
            .expandedDimensions(axis: -1)
        let outputImageMask = (indicator .== MLXArray(Ideogram4Constants.outputImageIndicator))
            .asType(parameterDType)
            .expandedDimensions(axis: -1)

        x = x * outputImageMask
        x = inputProj(x) * outputImageMask

        var tConditioning = tEmbedding(t)
        if t.ndim == 1 {
            tConditioning = tConditioning.expandedDimensions(axis: 1)
        }
        let adaLNInput = silu(adaLNProj(tConditioning))

        var hidden: MLXArray
        if let features = llmFeatures {
            llmFeatures = llmCondProj(llmCondNorm(features * llmTokenMask)) * llmTokenMask
            hidden = x + llmFeatures!
        } else {
            hidden = x
        }
        hidden =
            hidden
            + embedImageIndicator(
                (indicator .== MLXArray(Ideogram4Constants.outputImageIndicator)).asType(.int32))

        var rotary = rotaryEmbedding(positionIDs)
        rotary.0 = rotary.0.asType(hidden.dtype)
        rotary.1 = rotary.1.asType(hidden.dtype)
        for layer in layers {
            hidden = layer(
                hidden,
                segmentIDs: segmentIDs,
                cosines: rotary.0,
                sines: rotary.1,
                adaLNInput: adaLNInput,
            )
        }
        return finalLayer(hidden, c: adaLNInput).asType(.float32)
    }
}

public final class Ideogram4ImagePipeline: ImageGenerationModel {
    // Factory-loaded pipelines alternate text and image components to bound residency.
    var componentDirectory: URL?

    public var textEncoder: Flux2TextEncoder?
    public var conditionalTransformer: Ideogram4Transformer?
    public var unconditionalTransformer: Ideogram4Transformer?
    public var vae: Flux2VAE?

    public init(
        textEncoder: Flux2TextEncoder? = nil,
        conditionalTransformer: Ideogram4Transformer? = nil,
        unconditionalTransformer: Ideogram4Transformer? = nil,
        vae: Flux2VAE? = nil,
    ) {
        self.textEncoder = textEncoder
        self.conditionalTransformer = conditionalTransformer
        self.unconditionalTransformer = unconditionalTransformer
        self.vae = vae
    }

    public func encodePrompt(
        _ prompt: String,
        tokenizer: any Tokenizer,
        width: Int,
        height: Int,
        autoJSONCaption: Bool = true,
    )
        throws -> Ideogram4PromptInputs
    {
        if textEncoder == nil, let directory = componentDirectory {
            conditionalTransformer = nil
            unconditionalTransformer = nil
            vae = nil
            Memory.clearCache()
            textEncoder = try Ideogram4WeightLoader.loadTextEncoder(from: directory)
        }
        guard let textEncoder else {
            throw Ideogram4Error.missingComponent("text_encoder")
        }
        try Ideogram4Dimensions.validate(width: width, height: height)
        let preparedPrompt = try Ideogram4Prompting.preparePrompt(
            prompt, autoJSONCaption: autoJSONCaption)
        let builtInputs = try buildInputs(
            prompt: preparedPrompt.text,
            tokenizer: tokenizer,
            height: height,
            width: width,
        )
        let attentionMask = MLXArray.ones(builtInputs.tokenIDs.shape, dtype: .int32)
        let (_, hiddenStatesOptional) = try textEncoder(
            inputIDs: builtInputs.tokenIDs,
            attentionMask: attentionMask,
            outputHiddenStates: true,
        )
        guard let hiddenStates = hiddenStatesOptional else {
            throw Ideogram4Error.invalidPromptInputs("hidden states were not collected")
        }
        let selected = try Ideogram4Constants.activationLayers.map { layer -> MLXArray in
            let index = layer + 1
            guard hiddenStates.indices.contains(index) else {
                throw Ideogram4Error.invalidPromptInputs(
                    "hidden layer \(layer) is outside the text encoder range")
            }
            return hiddenStates[index]
        }
        var stackedStates = stacked(selected, axis: 0)
        stackedStates = stackedStates.transposed(1, 2, 3, 0)
        let batch = stackedStates.dim(0)
        let seqLen = stackedStates.dim(1)
        let hiddenDim = stackedStates.dim(2)
        let layerCount = stackedStates.dim(3)
        let promptEmbeddings = stackedStates.reshaped(batch, seqLen, hiddenDim * layerCount)
        let imagePadding = MLXArray.zeros(
            [batch, builtInputs.numImageTokens, promptEmbeddings.dim(-1)],
            dtype: promptEmbeddings.dtype,
        )
        let llmFeatures = concatenated([promptEmbeddings, imagePadding], axis: 1).asType(.float32)
        return Ideogram4PromptInputs(
            llmFeatures: llmFeatures,
            positionIDs: builtInputs.positionIDs,
            segmentIDs: builtInputs.segmentIDs,
            indicator: builtInputs.indicator,
            tokenIDs: builtInputs.tokenIDs,
            numTextTokens: builtInputs.numTextTokens,
            numImageTokens: builtInputs.numImageTokens,
            gridHeight: builtInputs.gridHeight,
            gridWidth: builtInputs.gridWidth,
            preparedPrompt: preparedPrompt,
        )
    }

    public func generate(
        prompt: String,
        tokenizer: any Tokenizer,
        parameters: ImageGenerationParameters = ImageGenerationParameters(),
        autoJSONCaption: Bool = true,
    )
        throws -> ImageGenerationResult
    {
        let inputs = try encodePrompt(
            prompt,
            tokenizer: tokenizer,
            width: parameters.width,
            height: parameters.height,
            autoJSONCaption: autoJSONCaption,
        )
        return try generate(inputs: inputs, parameters: parameters)
    }

    public func generate(
        promptEmbeddings: MLXArray,
        textIDs: MLXArray,
        parameters: ImageGenerationParameters,
    )
        throws -> ImageGenerationResult
    {
        try Ideogram4Dimensions.validate(width: parameters.width, height: parameters.height)
        let gridHeight =
            parameters.height / (Ideogram4Constants.patchSize * Ideogram4Constants.aeScaleFactor)
        let gridWidth =
            parameters.width / (Ideogram4Constants.patchSize * Ideogram4Constants.aeScaleFactor)
        let numImageTokens = gridHeight * gridWidth
        guard textIDs.ndim == 3, textIDs.dim(2) == 3 else {
            throw Ideogram4Error.invalidPromptInputs(
                "textIDs must carry position ids with shape [batch, tokens, 3]")
        }
        let numTextTokens = textIDs.dim(1) - numImageTokens
        guard numTextTokens > 0,
            promptEmbeddings.ndim == 3,
            promptEmbeddings.dim(1) == textIDs.dim(1)
        else {
            throw
                Ideogram4Error
                .invalidPromptInputs(
                    "prompt embeddings and position ids do not match the requested size")
        }
        let indicator = Self.indicator(numTextTokens: numTextTokens, numImageTokens: numImageTokens)
        let segmentIDs = MLXArray.ones([1, numTextTokens + numImageTokens], dtype: .int32)
        let inputs = Ideogram4PromptInputs(
            llmFeatures: promptEmbeddings,
            positionIDs: textIDs,
            segmentIDs: segmentIDs,
            indicator: indicator,
            tokenIDs: MLXArray.zeros([1, numTextTokens], dtype: .int32),
            numTextTokens: numTextTokens,
            numImageTokens: numImageTokens,
            gridHeight: gridHeight,
            gridWidth: gridWidth,
            preparedPrompt: Ideogram4PreparedPrompt(
                text: "",
                isJSONCaption: false,
                isStructuredCaption: false,
                wasWrapped: false,
            ),
        )
        return try generate(inputs: inputs, parameters: parameters)
    }

    public func generate(
        inputs: Ideogram4PromptInputs,
        parameters: ImageGenerationParameters = ImageGenerationParameters(),
    )
        throws -> ImageGenerationResult
    {
        if let directory = componentDirectory {
            eval(inputs.llmFeatures, inputs.positionIDs)
            textEncoder = nil
            Memory.clearCache()
            if conditionalTransformer == nil {
                conditionalTransformer = try Ideogram4WeightLoader.loadTransformer(
                    from: directory, subfolder: "transformer")
            }
            if unconditionalTransformer == nil {
                unconditionalTransformer = try Ideogram4WeightLoader.loadTransformer(
                    from: directory, subfolder: "unconditional_transformer")
            }
            if vae == nil { vae = try Ideogram4WeightLoader.loadVAE(from: directory) }
        }
        guard let conditionalTransformer else {
            throw Ideogram4Error.missingComponent("transformer")
        }
        guard let unconditionalTransformer else {
            throw Ideogram4Error.missingComponent("unconditional_transformer")
        }
        guard let vae else {
            throw Ideogram4Error.missingComponent("vae")
        }
        try Ideogram4Dimensions.validate(width: parameters.width, height: parameters.height)
        guard (parameters.steps ?? 1) >= 1 else {
            throw Ideogram4Error.invalidStepCount(parameters.steps ?? 0)
        }

        let preset = parameters.ideogram4.sampler
        let actualSteps = parameters.steps ?? preset.numSteps
        let guidanceScale = parameters.guidance ?? preset.guidanceSchedule.last ?? 7.0
        let guidanceSchedule =
            parameters.ideogram4.guidanceSchedule
            ?? (parameters.guidance == nil && actualSteps == preset.numSteps
                ? preset.guidanceSchedule : nil)
        guard actualSteps > 0, preset.mu.isFinite, preset.std.isFinite, preset.std > 0,
            guidanceSchedule.map({
                $0.count == actualSteps && $0.allSatisfy { $0.isFinite && $0 >= 0 }
            }) ?? true
        else {
            throw ModelExecutionError.invalidOptions(
                "Invalid Ideogram sampler or guidance schedule")
        }
        let schedule = Ideogram4LogitNormalSchedule.forResolution(
            width: parameters.width,
            height: parameters.height,
            knownMean: preset.mu,
            std: preset.std,
        )
        let intervals = (0 ... actualSteps).map { Double($0) / Double(actualSteps) }

        let seed = parameters.seed ?? 0
        let batchSize = 1
        var z = MLXRandom.normal(
            [batchSize, inputs.numImageTokens, Ideogram4Constants.latentDim],
            key: MLXRandom.key(seed),
        ).asType(.float32)
        let textZPadding = MLXArray.zeros(
            [batchSize, inputs.numTextTokens, Ideogram4Constants.latentDim],
            dtype: .float32,
        )

        let negativePositionIDs = inputs.positionIDs[0..., inputs.numTextTokens..., 0...]
        let negativeSegmentIDs = inputs.segmentIDs[0..., inputs.numTextTokens...]
        let negativeIndicator = inputs.indicator[0..., inputs.numTextTokens...]

        for index in stride(from: actualSteps - 1, through: 0, by: -1) {
            try Task.checkCancellation()
            let tValue = schedule.value(at: intervals[index + 1])
            let sValue = schedule.value(at: intervals[index])
            let t = MLXArray([tValue], [batchSize]).asType(.float32)

            let posZ = concatenated([textZPadding, z], axis: 1)
            let posOut = conditionalTransformer(
                llmFeatures: inputs.llmFeatures,
                x: posZ,
                t: t,
                positionIDs: inputs.positionIDs,
                segmentIDs: inputs.segmentIDs,
                indicator: inputs.indicator,
            )
            let posVelocity = posOut[0..., inputs.numTextTokens..., 0...]
            let negVelocity = unconditionalTransformer(
                llmFeatures: nil,
                x: z,
                t: t,
                positionIDs: negativePositionIDs,
                segmentIDs: negativeSegmentIDs,
                indicator: negativeIndicator,
            )
            let stepGuidance = guidanceSchedule?[index] ?? guidanceScale
            let velocity = stepGuidance * posVelocity + (1.0 - stepGuidance) * negVelocity
            z = z + velocity * (sValue - tValue)
            eval(z)
        }

        let image = decode(z, gridHeight: inputs.gridHeight, gridWidth: inputs.gridWidth, vae: vae)
        return ImageGenerationResult(
            image: image,
            seed: seed,
            width: parameters.width,
            height: parameters.height,
            steps: actualSteps,
            guidance: guidanceScale,
            metadata: [
                "architecture": "single_stream_dit",
                "weight_load": "fp8_dequantized_to_bf16",
                "sampler_preset": preset.name,
                "auto_json_caption": inputs.preparedPrompt.isJSONCaption ? "true" : "false",
                "prompt_was_wrapped": inputs.preparedPrompt.wasWrapped ? "true" : "false",
                "prompt_tokens": "\(inputs.numTextTokens)",
            ],
        )
    }

    private func buildInputs(
        prompt: String,
        tokenizer: any Tokenizer,
        height: Int,
        width: Int,
    )
        throws -> (
            tokenIDs: MLXArray,
            positionIDs: MLXArray,
            segmentIDs: MLXArray,
            indicator: MLXArray,
            numTextTokens: Int,
            numImageTokens: Int,
            gridHeight: Int,
            gridWidth: Int,
        )
    {
        let tokenIDs = try Self.tokenize(prompt: prompt, tokenizer: tokenizer)
        let numTextTokens = tokenIDs.count
        guard numTextTokens <= Ideogram4Constants.maxTextTokens else {
            throw Ideogram4Error.invalidPromptInputs(
                "prompt has \(numTextTokens) tokens, exceeds maxTextTokens=\(Ideogram4Constants.maxTextTokens)",
            )
        }
        let patch = Ideogram4Constants.patchSize * Ideogram4Constants.aeScaleFactor
        guard height.isMultiple(of: patch), width.isMultiple(of: patch) else {
            throw Ideogram4Error.invalidDimensions(width: width, height: height)
        }
        let gridHeight = height / patch
        let gridWidth = width / patch
        let numImageTokens = gridHeight * gridWidth

        let textPosition = arange(numTextTokens, dtype: .int32)
        let textPosition3D = stacked([textPosition, textPosition, textPosition], axis: 1)
        let hIDs = broadcast(
            arange(gridHeight, dtype: .int32).reshaped(-1, 1), to: [gridHeight, gridWidth]
        )
        .reshaped(-1)
        let wIDs = broadcast(
            arange(gridWidth, dtype: .int32).reshaped(1, -1), to: [gridHeight, gridWidth]
        )
        .reshaped(-1)
        let tIDs = MLXArray.zeros(hIDs.shape, dtype: .int32)
        let imagePosition =
            stacked([tIDs, hIDs, wIDs], axis: 1) + Ideogram4Constants.imagePositionOffset
        let positionIDs = concatenated([textPosition3D, imagePosition], axis: 0).expandedDimensions(
            axis: 0)
        let indicator = Self.indicator(numTextTokens: numTextTokens, numImageTokens: numImageTokens)
        let segmentIDs = MLXArray.ones([1, numTextTokens + numImageTokens], dtype: .int32)
        return (
            MLXArray(tokenIDs.map(Int32.init), [1, numTextTokens]),
            positionIDs,
            segmentIDs,
            indicator,
            numTextTokens,
            numImageTokens,
            gridHeight,
            gridWidth,
        )
    }

    private static func tokenize(prompt: String, tokenizer: any Tokenizer) throws -> [Int] {
        let content: [[String: String]] = [["type": "text", "text": prompt]]
        let messages: [[String: any Sendable]] = [
            ["role": "user", "content": content]
        ]
        do {
            return try tokenizer.applyChatTemplate(
                messages: messages,
                tools: [[String: any Sendable]]?.none,
                additionalContext: ["add_generation_prompt": true, "add_vision_id": false],
            )
        } catch {
            return tokenizer.encode(text: prompt, addSpecialTokens: false)
        }
    }

    private static func indicator(numTextTokens: Int, numImageTokens: Int) -> MLXArray {
        let values =
            Array(repeating: Ideogram4Constants.llmTokenIndicator, count: numTextTokens)
            + Array(repeating: Ideogram4Constants.outputImageIndicator, count: numImageTokens)
        return MLXArray(values, [1, values.count])
    }

    private func decode(
        _ z: MLXArray,
        gridHeight: Int,
        gridWidth: Int,
        vae: Flux2VAE,
    )
        -> MLXArray
    {
        let shift = MLXArray(Ideogram4LatentNorm.shift, [1, 1, Ideogram4Constants.latentDim])
            .asType(z.dtype)
        let scale = MLXArray(Ideogram4LatentNorm.scale, [1, 1, Ideogram4Constants.latentDim])
            .asType(z.dtype)
        var z = z * scale + shift
        let patch = Ideogram4Constants.patchSize
        let batch = z.dim(0)
        let aeChannels = z.dim(-1) / (patch * patch)
        z = z.reshaped(batch, gridHeight, gridWidth, patch, patch, aeChannels)
        z = z.transposed(0, 5, 1, 3, 2, 4)
        let latents = z.reshaped(batch, aeChannels, gridHeight * patch, gridWidth * patch)
        var decoded = vae.decode(latents.asType(Flux2Precision.dtype))
        decoded = clip(decoded.asType(.float32), min: Float32(-1), max: Float32(1))
        decoded = round((decoded + 1.0) * 127.5).asType(.uint8)
        let image = decoded.transposed(0, 2, 3, 1)[0]
        eval(image)
        return image
    }
}

public enum Ideogram4WeightLoader {
    public static func loadPipeline(
        from modelDirectory: URL,
        loadTextEncoder: Bool = true,
        loadTransformers: Bool = true,
        loadVAE: Bool = true,
    )
        throws -> Ideogram4ImagePipeline
    {
        try validateModelLayout(modelDirectory)
        return try Ideogram4ImagePipeline(
            textEncoder: loadTextEncoder ? self.loadTextEncoder(from: modelDirectory) : nil,
            conditionalTransformer: loadTransformers
                ? loadTransformer(from: modelDirectory, subfolder: "transformer") : nil,
            unconditionalTransformer: loadTransformers
                ? loadTransformer(
                    from: modelDirectory,
                    subfolder: "unconditional_transformer",
                ) : nil,
            vae: loadVAE ? self.loadVAE(from: modelDirectory) : nil,
        )
    }

    public static func loadTextEncoder(from modelDirectory: URL) throws -> Flux2TextEncoder {
        let config = try readTextEncoderConfiguration(
            from: modelDirectory.appending(path: "text_encoder/config.json"))
        let model = Flux2TextEncoder(config)
        let raw = try loadSafetensors(from: modelDirectory.appending(path: "text_encoder"))
        let weights = sanitizeTextEncoderWeights(raw)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    public static func loadTransformer(from modelDirectory: URL, subfolder: String) throws
        -> Ideogram4Transformer
    {
        let directory = modelDirectory.appending(path: subfolder)
        let config = try readTransformerConfiguration(
            from: directory.appending(path: "config.json"))
        let model = Ideogram4Transformer(config)
        let raw = try loadSafetensors(from: directory)
        let weights = dequantizeFP8WeightOnly(raw)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    public static func loadVAE(from modelDirectory: URL) throws -> Flux2VAE {
        let vae = Flux2VAE(try .read(from: modelDirectory))
        try Flux2WeightLoader.loadVAEWeights(
            from: modelDirectory.appending(path: "vae"),
            into: vae,
            includeEncoder: false,
        )
        return vae
    }

    public static func sanitizeTextEncoderWeights(_ weights: [String: MLXArray]) -> [String:
        MLXArray]
    {
        let dequantized = dequantizeFP8WeightOnly(weights)
        var sanitized = [String: MLXArray]()
        for (key, value) in dequantized {
            var mapped = key
            if mapped.hasPrefix("language_model.") {
                mapped = String(mapped.dropFirst("language_model.".count))
            }
            guard
                mapped.hasPrefix("embed_tokens.")
                    || mapped.hasPrefix("layers.")
                    || mapped.hasPrefix("norm.")
            else {
                continue
            }
            sanitized[mapped] = value
        }
        return sanitized
    }

    public static func dequantizeFP8WeightOnly(
        _ weights: [String: MLXArray],
        precision: DType = Flux2Precision.dtype,
    )
        -> [String: MLXArray]
    {
        var converted = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasSuffix(".weight_scale") {
                continue
            }
            let scaleKey = "\(key)_scale"
            if key.hasSuffix(".weight"), let scale = weights[scaleKey] {
                converted[key] = decodeFP8E4M3(
                    value,
                    scale: scale,
                    precision: precision,
                )
            } else if value.dtype == .float16 || value.dtype == .float32 || value.dtype == .bfloat16
            {
                converted[key] = value.asType(precision)
            } else {
                converted[key] = value
            }
        }
        return converted
    }

    public static func decodeFP8E4M3(
        _ weight: MLXArray,
        scale: MLXArray,
        precision: DType = Flux2Precision.dtype,
    ) -> MLXArray {
        fromFP8(weight.asType(.uint8), dtype: precision)
            * scale.asType(precision).expandedDimensions(axis: -1)
    }

    public static func validateModelLayout(_ modelDirectory: URL) throws {
        let indexURL = modelDirectory.appending(path: "model_index.json")
        guard let data = try? Data(contentsOf: indexURL),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            object["_class_name"] as? String == "Ideogram4Pipeline"
        else {
            throw Ideogram4Error.invalidModelLayout(modelDirectory)
        }
        let required = [
            "tokenizer/tokenizer.json",
            "text_encoder/config.json",
            "transformer/config.json",
            "unconditional_transformer/config.json",
            "vae/config.json",
        ]
        for relative in required {
            guard
                FileManager.default.fileExists(
                    atPath: modelDirectory.appending(path: relative).path)
            else {
                throw Ideogram4Error.missingWeights(modelDirectory.appending(path: relative))
            }
        }
    }

    private static func readTextEncoderConfiguration(from url: URL) throws
        -> Flux2TextEncoderConfiguration
    {
        let data = try Data(contentsOf: url)
        let raw = try JSONDecoder().decode(TextEncoderConfigFile.self, from: data).textConfig
        let ropeTheta = raw.ropeTheta ?? raw.ropeParameters?.ropeTheta ?? 5_000_000
        return Flux2TextEncoderConfiguration(
            vocabSize: raw.vocabSize,
            hiddenSize: raw.hiddenSize,
            numHiddenLayers: raw.numHiddenLayers,
            numAttentionHeads: raw.numAttentionHeads,
            numKeyValueHeads: raw.numKeyValueHeads,
            intermediateSize: raw.intermediateSize,
            maxPositionEmbeddings: raw.maxPositionEmbeddings,
            ropeTheta: Float(ropeTheta),
            rmsNormEps: raw.rmsNormEps,
            headDim: raw.headDim,
            attentionBias: raw.attentionBias ?? false,
        )
    }

    private static func readTransformerConfiguration(from url: URL) throws
        -> Ideogram4TransformerConfiguration
    {
        let data = try Data(contentsOf: url)
        let raw = try JSONDecoder().decode(TransformerConfigFile.self, from: data)
        let heads = raw.numHeads ?? raw.numAttentionHeads ?? 18
        let headDim = raw.attentionHeadDim
        let embDim = raw.embDim ?? (headDim.map { heads * $0 }) ?? 4608
        return Ideogram4TransformerConfiguration(
            embDim: embDim,
            numLayers: raw.numLayers ?? 34,
            numHeads: heads,
            intermediateSize: raw.intermediateSize ?? 12288,
            adaLNDim: raw.adaLNDim ?? raw.adanLNDim ?? 512,
            inChannels: raw.inChannels ?? 128,
            llmFeaturesDim: raw.llmFeaturesDim ?? 4096 * 13,
            ropeTheta: Float(raw.ropeTheta ?? 5_000_000),
            mropeSection: raw.mropeSection ?? [24, 20, 20],
            normEps: raw.normEps ?? 1e-5,
        )
    }

    private static func loadSafetensors(from directory: URL) throws -> [String: MLXArray] {
        guard
            let enumerator = FileManager.default.enumerator(
                at: directory,
                includingPropertiesForKeys: nil,
            )
        else {
            throw Ideogram4Error.missingWeights(directory)
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
            throw Ideogram4Error.missingWeights(directory)
        }
        return weights
    }

    private struct TextEncoderConfigFile: Decodable {
        let textConfig: TextConfig

        enum CodingKeys: String, CodingKey {
            case textConfig = "text_config"
        }
    }

    private struct TextConfig: Decodable {
        let vocabSize: Int
        let hiddenSize: Int
        let numHiddenLayers: Int
        let numAttentionHeads: Int
        let numKeyValueHeads: Int
        let intermediateSize: Int
        let maxPositionEmbeddings: Int
        let ropeTheta: Double?
        let ropeParameters: RopeParameters?
        let rmsNormEps: Float
        let headDim: Int
        let attentionBias: Bool?

        enum CodingKeys: String, CodingKey {
            case vocabSize = "vocab_size"
            case hiddenSize = "hidden_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case intermediateSize = "intermediate_size"
            case maxPositionEmbeddings = "max_position_embeddings"
            case ropeTheta = "rope_theta"
            case ropeParameters = "rope_parameters"
            case rmsNormEps = "rms_norm_eps"
            case headDim = "head_dim"
            case attentionBias = "attention_bias"
        }
    }

    private struct RopeParameters: Decodable {
        let ropeTheta: Double?

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
        }
    }

    private struct TransformerConfigFile: Decodable {
        let embDim: Int?
        let numLayers: Int?
        let numHeads: Int?
        let numAttentionHeads: Int?
        let attentionHeadDim: Int?
        let intermediateSize: Int?
        let adaLNDim: Int?
        let adanLNDim: Int?
        let inChannels: Int?
        let llmFeaturesDim: Int?
        let ropeTheta: Double?
        let mropeSection: [Int]?
        let normEps: Float?

        enum CodingKeys: String, CodingKey {
            case embDim = "emb_dim"
            case numLayers = "num_layers"
            case numHeads = "num_heads"
            case numAttentionHeads = "num_attention_heads"
            case attentionHeadDim = "attention_head_dim"
            case intermediateSize = "intermediate_size"
            case adaLNDim = "adaln_dim"
            case adanLNDim = "adanln_dim"
            case inChannels = "in_channels"
            case llmFeaturesDim = "llm_features_dim"
            case ropeTheta = "rope_theta"
            case mropeSection = "mrope_section"
            case normEps = "norm_eps"
        }
    }
}

private enum Ideogram4LatentNorm {
    static let shift: [Float] = [
        0.01984364, 0.10149707, 0.29689495, 0.27188619, -0.21445648, -0.15979549, 0.05021099,
        -0.15083604,
        -0.15360136, -0.20131799, 0.01922352, 0.0622626, 0.10140969, -0.06739428, 0.3758261,
        -0.233712,
        0.35164491, -0.02590912, -0.0271935, -0.10833897, -0.1476848, -0.01130957, -0.2298372,
        0.23526423,
        -0.10893522, 0.11957631, 0.04047799, 0.3134589, -0.17225064, -0.18646109, -0.34691978,
        -0.03571246,
        0.02583857, 0.10190072, 0.28402294, 0.26952152, -0.21634675, -0.17938656, 0.04358909,
        -0.15007621,
        -0.1548502, -0.18971131, 0.02710861, 0.05609494, 0.10697846, -0.06854968, 0.38167698,
        -0.24269937,
        0.35705471, -0.03063305, -0.02946109, -0.11244286, -0.14336038, -0.01362137, -0.21863696,
        0.23228983,
        -0.11739769, 0.11693044, 0.02563311, 0.31356594, -0.17420591, -0.19006285, -0.34905377,
        -0.04025005,
        0.01924137, 0.07652984, 0.2995608, 0.2628057, -0.22011674, -0.12715361, 0.04879879,
        -0.14075719,
        -0.15935895, -0.2123584, 0.01974813, 0.05523547, 0.10011992, -0.06428964, 0.37781868,
        -0.21491644,
        0.34254215, -0.03153528, -0.0310082, -0.10761415, -0.14730405, -0.02475182, -0.2285588,
        0.2515081,
        -0.10445128, 0.12446, 0.07062869, 0.30880162, -0.18016875, -0.18869164, -0.34533499,
        -0.0129177,
        0.02578168, 0.07993659, 0.28642181, 0.26038408, -0.22459419, -0.14820155, 0.04059549,
        -0.14043529,
        -0.16111187, -0.2020305, 0.02602069, 0.04852717, 0.10432153, -0.06309942, 0.38402443,
        -0.22397003,
        0.34814481, -0.03774432, -0.03381438, -0.11245691, -0.14128767, -0.02853208, -0.21752016,
        0.24872463,
        -0.11399775, 0.1222687, 0.05620835, 0.309178, -0.18065738, -0.19401479, -0.34495114,
        -0.01760592,
    ]

    static let scale: [Float] = [
        1.63933691, 1.70204478, 1.73642566, 1.90004803, 1.6675316, 1.69059584, 1.56853198,
        1.62314944,
        1.89106626, 1.58086668, 1.60822129, 1.60962993, 1.63322129, 1.56074359, 1.73419528,
        1.7919265,
        1.64040632, 1.66802808, 1.60390303, 1.75480492, 1.63187587, 1.64334594, 1.61722884,
        1.60146046,
        1.63459219, 1.55291476, 1.68771497, 1.68415657, 1.78966054, 1.66631641, 1.65626686,
        1.65976433,
        1.63487607, 1.69513249, 1.72933756, 1.91310663, 1.67035057, 1.72286863, 1.56719251,
        1.61934825,
        1.88628859, 1.56911539, 1.59455129, 1.60829869, 1.62470611, 1.56052853, 1.73677003,
        1.77563606,
        1.63732541, 1.66370527, 1.59508952, 1.75153949, 1.63029275, 1.64517667, 1.61659342,
        1.59722044,
        1.64103121, 1.5408531, 1.68610394, 1.67772755, 1.78998563, 1.66621713, 1.65458955,
        1.66041308,
        1.64710857, 1.68163503, 1.74000294, 1.92784786, 1.67411194, 1.67395548, 1.57406532,
        1.62199356,
        1.87618195, 1.5584375, 1.57438785, 1.61711053, 1.63094305, 1.55644029, 1.73124302,
        1.80666627,
        1.6463621, 1.65932006, 1.60816188, 1.75682671, 1.64695873, 1.63121722, 1.61380832,
        1.60478651,
        1.63396035, 1.53505068, 1.65534289, 1.67132281, 1.80317197, 1.6767314, 1.65700938,
        1.68426259,
        1.65339716, 1.67540638, 1.73298504, 1.94067348, 1.67893609, 1.70635117, 1.5730906,
        1.61928553,
        1.87148809, 1.56244866, 1.56697152, 1.61584394, 1.62759496, 1.55480378, 1.73484107,
        1.79055143,
        1.64688773, 1.66121492, 1.60135887, 1.75254572, 1.64798332, 1.62989921, 1.61381592,
        1.60792883,
        1.63939668, 1.53075757, 1.65371318, 1.66801185, 1.80029087, 1.67591476, 1.65655173,
        1.68533454,
    ]
}
