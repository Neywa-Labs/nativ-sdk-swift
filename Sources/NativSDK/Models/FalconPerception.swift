import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct FalconPerceptionConfiguration: Decodable, Sendable {
    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let headDimension: Int
        private let _kvHeads: Int?
        public var kvHeads: Int {
            _kvHeads ?? attentionHeads
        }

        public let vocabularySize: Int
        public let intermediateSize: Int
        private let _rmsNormEps: Float?
        public var rmsNormEps: Float {
            _rmsNormEps ?? 1e-5
        }

        private let _maxPositionEmbeddings: Int?
        public var maxPositionEmbeddings: Int {
            _maxPositionEmbeddings ?? 8192
        }

        private let _ropeTheta: Float?
        public var ropeTheta: Float {
            _ropeTheta ?? 10000
        }

        private let _tieWordEmbeddings: Bool?
        public var tieWordEmbeddings: Bool {
            _tieWordEmbeddings ?? false
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenSizeAlias = "dim"
            case hiddenLayers = "num_hidden_layers"
            case hiddenLayersAlias = "n_layers"
            case attentionHeads = "num_attention_heads"
            case attentionHeadsAlias = "n_heads"
            case headDimension = "head_dim"
            case _kvHeads = "num_key_value_heads"
            case kvHeadsAlias = "n_kv_heads"
            case vocabularySize = "vocab_size"
            case intermediateSize = "intermediate_size"
            case intermediateSizeAlias = "ffn_dim"
            case _rmsNormEps = "rms_norm_eps"
            case rmsNormEpsAlias = "norm_eps"
            case _maxPositionEmbeddings = "max_position_embeddings"
            case maxPositionEmbeddingsAlias = "max_seq_len"
            case _ropeTheta = "rope_theta"
            case _tieWordEmbeddings = "tie_word_embeddings"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType) ?? "falcon_perception"
            hiddenSize =
                try c.decodeIfPresent(Int.self, forKey: .hiddenSize)
                ?? c.decode(Int.self, forKey: .hiddenSizeAlias)
            hiddenLayers =
                try c.decodeIfPresent(Int.self, forKey: .hiddenLayers)
                ?? c.decode(Int.self, forKey: .hiddenLayersAlias)
            attentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .attentionHeads)
                ?? c.decode(Int.self, forKey: .attentionHeadsAlias)
            headDimension = try c.decodeIfPresent(Int.self, forKey: .headDimension) ?? 128
            _kvHeads =
                try c.decodeIfPresent(Int.self, forKey: ._kvHeads)
                ?? c.decodeIfPresent(Int.self, forKey: .kvHeadsAlias)
            vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 65536
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize)
                ?? c.decodeIfPresent(Int.self, forKey: .intermediateSizeAlias)
                ?? 3072
            _rmsNormEps =
                try c.decodeIfPresent(Float.self, forKey: ._rmsNormEps)
                ?? c.decodeIfPresent(Float.self, forKey: .rmsNormEpsAlias)
            _maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: ._maxPositionEmbeddings)
                ?? c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddingsAlias)
            _ropeTheta = try c.decodeIfPresent(Float.self, forKey: ._ropeTheta)
            _tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: ._tieWordEmbeddings)
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        private let _spatialPatchSize: Int?
        public var spatialPatchSize: Int {
            _spatialPatchSize ?? 16
        }

        private let _temporalPatchSize: Int?
        public var temporalPatchSize: Int {
            _temporalPatchSize ?? 1
        }

        private let _channelSize: Int?
        public var channelSize: Int {
            _channelSize ?? 3
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case _spatialPatchSize = "spatial_patch_size"
            case _temporalPatchSize = "temporal_patch_size"
            case _channelSize = "channel_size"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType) ?? "falcon_perception"
            _spatialPatchSize = try c.decodeIfPresent(Int.self, forKey: ._spatialPatchSize)
            _temporalPatchSize = try c.decodeIfPresent(Int.self, forKey: ._temporalPatchSize)
            _channelSize = try c.decodeIfPresent(Int.self, forKey: ._channelSize)
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    public let vocabularySize: Int
    public let imageTokenID: Int
    public let eosTokenID: Int
    public let imageClassTokenID: Int
    public let imageReg1TokenID: Int
    public let imageReg2TokenID: Int
    public let imageReg3TokenID: Int
    public let imageReg4TokenID: Int
    public let imageEndTokenID: Int
    public let coordTokenID: Int
    public let sizeTokenID: Int
    public let segTokenID: Int
    public let coordEncoderDimensions: Int
    public let coordDecoderDimensions: Int
    public let coordOutputDimensions: Int
    public let sizeEncoderDimensions: Int
    public let sizeDecoderDimensions: Int
    public let sizeOutputDimensions: Int
    public let doSegmentation: Bool
    public let segmentationOutputDimensions: Int
    public let segmentationLayers: Int

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case imageTokenID = "img_id"
        case eosTokenID = "eos_id"
        case imageClassTokenID = "image_cls_token_id"
        case imageReg1TokenID = "image_reg_1_token_id"
        case imageReg2TokenID = "image_reg_2_token_id"
        case imageReg3TokenID = "image_reg_3_token_id"
        case imageReg4TokenID = "image_reg_4_token_id"
        case imageEndTokenID = "img_end_id"
        case coordTokenID = "coord_token_id"
        case sizeTokenID = "size_token_id"
        case segTokenID = "seg_token_id"
        case coordEncoderDimensions = "coord_enc_dim"
        case coordDecoderDimensions = "coord_dec_dim"
        case coordOutputDimensions = "coord_out_dim"
        case sizeEncoderDimensions = "size_enc_dim"
        case sizeDecoderDimensions = "size_dec_dim"
        case sizeOutputDimensions = "size_out_dim"
        case doSegmentation = "do_segmentation"
        case segmentationOutputDimensions = "segm_out_dim"
        case segmentationLayers = "num_segm_layers"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "falcon_perception"
        textConfiguration =
            try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? TextConfiguration(from: decoder)
        visionConfiguration =
            try c.decodeIfPresent(VisionConfiguration.self, forKey: .visionConfiguration)
            ?? VisionConfiguration(from: decoder)
        vocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .vocabularySize)
            ?? textConfiguration.vocabularySize
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? 227
        eosTokenID = try c.decodeIfPresent(Int.self, forKey: .eosTokenID) ?? 11
        imageClassTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imageClassTokenID) ?? 244
        imageReg1TokenID = try c.decodeIfPresent(Int.self, forKey: .imageReg1TokenID) ?? 245
        imageReg2TokenID = try c.decodeIfPresent(Int.self, forKey: .imageReg2TokenID) ?? 246
        imageReg3TokenID = try c.decodeIfPresent(Int.self, forKey: .imageReg3TokenID) ?? 247
        imageReg4TokenID = try c.decodeIfPresent(Int.self, forKey: .imageReg4TokenID) ?? 248
        imageEndTokenID = try c.decodeIfPresent(Int.self, forKey: .imageEndTokenID) ?? 230
        coordTokenID = try c.decodeIfPresent(Int.self, forKey: .coordTokenID) ?? 240
        sizeTokenID = try c.decodeIfPresent(Int.self, forKey: .sizeTokenID) ?? 241
        segTokenID = try c.decodeIfPresent(Int.self, forKey: .segTokenID) ?? 262
        coordEncoderDimensions =
            try c.decodeIfPresent(Int.self, forKey: .coordEncoderDimensions) ?? 512
        coordDecoderDimensions =
            try c.decodeIfPresent(Int.self, forKey: .coordDecoderDimensions) ?? 8192
        coordOutputDimensions =
            try c.decodeIfPresent(Int.self, forKey: .coordOutputDimensions) ?? 2048
        sizeEncoderDimensions =
            try c.decodeIfPresent(Int.self, forKey: .sizeEncoderDimensions) ?? 512
        sizeDecoderDimensions =
            try c.decodeIfPresent(Int.self, forKey: .sizeDecoderDimensions) ?? 8192
        sizeOutputDimensions =
            try c.decodeIfPresent(Int.self, forKey: .sizeOutputDimensions) ?? 2048
        doSegmentation = try c.decodeIfPresent(Bool.self, forKey: .doSegmentation) ?? true
        segmentationOutputDimensions =
            try c.decodeIfPresent(Int.self, forKey: .segmentationOutputDimensions) ?? 256
        segmentationLayers = try c.decodeIfPresent(Int.self, forKey: .segmentationLayers) ?? 3
    }
}

public struct FalconPerceptionProcessorConfiguration: Decodable, Sendable {
    private let _spatialPatchSize: Int?
    public var spatialPatchSize: Int {
        _spatialPatchSize ?? 16
    }

    private let _minImageSize: Int?
    public var minImageSize: Int {
        _minImageSize ?? 256
    }

    private let _maxImageSize: Int?
    public var maxImageSize: Int {
        _maxImageSize ?? 1024
    }

    private let _minPixels: Int?
    public var minPixels: Int {
        _minPixels ?? 56 * 56
    }

    private let _maxPixels: Int?
    public var maxPixels: Int {
        _maxPixels ?? 28 * 28 * 1280
    }

    private let _imageTokenID: Int?
    public var imageTokenID: Int {
        _imageTokenID ?? 227
    }

    private let _imageEndTokenID: Int?
    public var imageEndTokenID: Int {
        _imageEndTokenID ?? 230
    }

    private let _imageClassTokenID: Int?
    public var imageClassTokenID: Int {
        _imageClassTokenID ?? 244
    }

    private let _imageReg1TokenID: Int?
    public var imageReg1TokenID: Int {
        _imageReg1TokenID ?? 245
    }

    private let _imageReg2TokenID: Int?
    public var imageReg2TokenID: Int {
        _imageReg2TokenID ?? 246
    }

    private let _imageReg3TokenID: Int?
    public var imageReg3TokenID: Int {
        _imageReg3TokenID ?? 247
    }

    private let _imageReg4TokenID: Int?
    public var imageReg4TokenID: Int {
        _imageReg4TokenID ?? 248
    }

    enum CodingKeys: String, CodingKey {
        case _spatialPatchSize = "spatial_patch_size"
        case _minImageSize = "min_image_size"
        case _maxImageSize = "max_image_size"
        case _minPixels = "min_pixels"
        case _maxPixels = "max_pixels"
        case _imageTokenID = "img_id"
        case _imageEndTokenID = "img_end_id"
        case _imageClassTokenID = "image_cls_token_id"
        case _imageReg1TokenID = "image_reg_1_token_id"
        case _imageReg2TokenID = "image_reg_2_token_id"
        case _imageReg3TokenID = "image_reg_3_token_id"
        case _imageReg4TokenID = "image_reg_4_token_id"
    }
}

// MARK: - Processor

public struct FalconPerceptionProcessor: UserInputProcessor {
    private let config: FalconPerceptionProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imagePrefixIDs: [Int]

    public init(_ config: FalconPerceptionProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        imagePrefixIDs = [
            config.imageClassTokenID,
            config.imageReg1TokenID,
            config.imageReg2TokenID,
            config.imageReg3TokenID,
            config.imageReg4TokenID,
        ]
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        let prompt = Self.makePerceptionPrompt(Self.prompt(from: input))
        var tokens = tokenizer.encode(text: prompt, addSpecialTokens: false)

        guard !images.isEmpty else {
            let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
            return LMInput(text: .init(tokens: tokenArray, mask: ones(like: tokenArray)))
        }
        guard images.count == 1 else {
            throw VLMError.singleImageAllowed
        }

        let (pixels, gridH, gridW) = try preprocess(image: images[0].asCIImage())
        tokens = expandImageTokens(tokens, gridH: gridH, gridW: gridW)

        let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)),
            image: .init(
                pixels: pixels.expandedDimensions(axis: 0),
                frames: [THW(1, gridH, gridW)],
            ),
        )
    }

    private func preprocess(image: CIImage) throws -> (MLXArray, Int, Int) {
        let original = image.extent.size
        let resized = Self.resizeIfNecessary(
            width: Int(original.width),
            height: Int(original.height),
            shortest: config.minImageSize,
            longest: config.maxImageSize,
        )
        let target = Self.smartResize(
            width: resized.width,
            height: resized.height,
            factor: config.spatialPatchSize,
            minPixels: config.minPixels,
            maxPixels: config.maxPixels,
        )

        var processed = image.toSRGB()
        let resizedSize = CGSize(width: resized.width, height: resized.height)
        if resized.width != Int(original.width) || resized.height != Int(original.height) {
            processed = processed.resampled(to: resizedSize, method: .bicubic)
        }
        let targetSize = CGSize(width: target.width, height: target.height)
        if target.width != resized.width || target.height != resized.height {
            processed = processed.resampled(to: targetSize, method: .bicubic)
        }

        let normalized = Self.normalizedRGBArray(processed)
        return (
            normalized,
            target.height / config.spatialPatchSize,
            target.width / config.spatialPatchSize,
        )
    }

    private static func normalizedRGBArray(_ image: CIImage) -> MLXArray {
        let width = Int(image.extent.width.rounded())
        let height = Int(image.extent.height.rounded())
        let componentsPerPixel = 4
        let bytesPerPixel = componentsPerPixel * MemoryLayout<Float32>.size
        let bytesPerRow = width * bytesPerPixel
        var data = Data(count: width * height * bytesPerPixel)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)

        data.withUnsafeMutableBytes { pointer in
            CIContext().render(
                image,
                toBitmap: pointer.baseAddress!,
                rowBytes: bytesPerRow,
                bounds: image.extent,
                format: .RGBAf,
                colorSpace: colorSpace,
            )
        }

        let rgba = MLXArray(data, [height, width, 4], type: Float32.self)
        let rgb = rgba[0..., 0..., ..<3]
        let alpha = maximum(rgba[0..., 0..., 3...], MLXArray(Float32(1e-6)))
        let unpremultiplied = clip(rgb / alpha, min: Float32(0), max: Float32(1))
        return (unpremultiplied - Float32(0.5)) / Float32(0.5)
    }

    private func expandImageTokens(_ tokenIDs: [Int], gridH: Int, gridW: Int) -> [Int] {
        var expanded = [Int]()
        let patchCount = gridH * gridW
        for token in tokenIDs {
            if token == config.imageTokenID {
                expanded.append(contentsOf: imagePrefixIDs)
                expanded.append(
                    contentsOf: Array(repeating: config.imageTokenID, count: patchCount))
                expanded.append(config.imageEndTokenID)
            } else {
                expanded.append(token)
            }
        }
        return expanded
    }

    private static func prompt(from input: UserInput) -> String {
        switch input.prompt {
        case .text(let text):
            text
        case .messages:
            input.prompt.description
        case .chat(let messages):
            messages.map(\.content).joined(separator: "\n")
        }
    }

    private static func makePerceptionPrompt(_ query: String) -> String {
        "<|image|>Segment these expressions in the image:<|start_of_query|>\(query)<|REF_SEG|>"
    }

    private static func resizeIfNecessary(width: Int, height: Int) -> (width: Int, height: Int) {
        resizeIfNecessary(width: width, height: height, shortest: 256, longest: 1024)
    }

    private static func resizeIfNecessary(
        width: Int,
        height: Int,
        shortest: Int,
        longest: Int,
    )
        -> (width: Int, height: Int)
    {
        let aspect = Double(width) / Double(height)
        if shortest <= width, width <= longest, shortest <= height, height <= longest {
            return (width, height)
        }

        let isVertical = width < height
        var newWidth: Int
        var newHeight: Int
        if width < shortest || height < shortest {
            if isVertical {
                newWidth = shortest
                newHeight = Int(Double(shortest) / aspect)
            } else {
                newHeight = shortest
                newWidth = Int(Double(shortest) * aspect)
            }
        } else {
            if isVertical {
                newWidth = longest
                newHeight = Int(Double(newWidth) / aspect)
            } else {
                newHeight = longest
                newWidth = Int(Double(newHeight) * aspect)
            }
        }

        if newWidth > longest {
            newWidth = longest
            newHeight = Int(Double(newWidth) / aspect)
        }
        if newHeight > longest {
            newHeight = longest
            newWidth = Int(Double(newHeight) * aspect)
        }
        return (newWidth, newHeight)
    }

    private static func smartResize(width: Int, height: Int, factor: Int) -> (
        width: Int, height: Int
    ) {
        smartResize(
            width: width,
            height: height,
            factor: factor,
            minPixels: 56 * 56,
            maxPixels: 28 * 28 * 1280,
        )
    }

    private static func smartResize(
        width: Int,
        height: Int,
        factor: Int,
        minPixels: Int,
        maxPixels: Int,
    )
        -> (width: Int, height: Int)
    {
        var targetHeight = Int(round(Double(height) / Double(factor))) * factor
        var targetWidth = Int(round(Double(width) / Double(factor))) * factor

        if targetHeight * targetWidth > maxPixels {
            let beta = sqrt(Double(height * width) / Double(maxPixels))
            targetHeight = max(factor, Int(floor(Double(height) / beta / Double(factor))) * factor)
            targetWidth = max(factor, Int(floor(Double(width) / beta / Double(factor))) * factor)
        } else if targetHeight * targetWidth < minPixels {
            let beta = sqrt(Double(minPixels) / Double(height * width))
            targetHeight = Int(ceil(Double(height) * beta / Double(factor))) * factor
            targetWidth = Int(ceil(Double(width) * beta / Double(factor))) * factor
        }

        return (max(factor, targetWidth), max(factor, targetHeight))
    }
}

// MARK: - Language

private enum FalconPerceptionLanguage {
    static func precomputeFrequencies(
        dimensions: Int,
        end: Int,
        theta: Float,
    )
        -> (MLXArray, MLXArray)
    {
        let p =
            MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
            / Float(dimensions)
        let inverseFrequencies = 1.0 / pow(theta, p)
        let positions = MLXArray(0 ..< end).asType(.float32)
        let frequencies = positions[0..., .newAxis] * inverseFrequencies[.newAxis, 0...]
        return (MLX.cos(frequencies), MLX.sin(frequencies))
    }

    static func applyRotaryEmbedding1D(
        q: MLXArray,
        k: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
    )
        -> (MLXArray, MLXArray)
    {
        let qShape = q.shape
        let kShape = k.shape
        let dimensions = q.dim(-1)
        let qRot = q.asType(.float32).reshaped(qShape.dropLast() + [dimensions / 2, 2])
        let kRot = k.asType(.float32).reshaped(kShape.dropLast() + [dimensions / 2, 2])

        let q0 = qRot[.ellipsis, 0]
        let q1 = qRot[.ellipsis, 1]
        let k0 = kRot[.ellipsis, 0]
        let k1 = kRot[.ellipsis, 1]

        let c: MLXArray
        let s: MLXArray
        if cos.ndim == 2 {
            c = cos.reshaped(1, 1, -1, cos.dim(-1))
            s = sin.reshaped(1, 1, -1, sin.dim(-1))
        } else {
            c = cos.reshaped(cos.dim(0), 1, -1, cos.dim(-1))
            s = sin.reshaped(sin.dim(0), 1, -1, sin.dim(-1))
        }

        let qOut = stacked([q0 * c - q1 * s, q0 * s + q1 * c], axis: -1)
            .reshaped(qShape)
            .asType(q.dtype)
        let kOut = stacked([k0 * c - k1 * s, k0 * s + k1 * c], axis: -1)
            .reshaped(kShape)
            .asType(k.dtype)
        return (qOut, kOut)
    }

    static func computeGoldenFrequencies(
        frequencies: MLXArray,
        posHW: MLXArray,
    )
        -> (MLXArray, MLXArray)
    {
        let theta = einsum(
            "bsp,hfp->bshf",
            posHW.asType(.float32),
            frequencies.asType(.float32),
        )
        return (MLX.cos(theta), MLX.sin(theta))
    }

    static func applyGoldenRotaryEmbedding(_ x: MLXArray, cos2D: MLXArray, sin2D: MLXArray)
        -> MLXArray
    {
        let cos = cos2D.transposed(0, 2, 1, 3)
        let sin = sin2D.transposed(0, 2, 1, 3)
        let xFloat = x.asType(.float32)
        let xEven = xFloat[.ellipsis, stride(from: 0, to: x.dim(-1), by: 2)]
        let xOdd = xFloat[.ellipsis, stride(from: 1, to: x.dim(-1), by: 2)]
        let outEven = xEven * cos - xOdd * sin
        let outOdd = xEven * sin + xOdd * cos
        return stacked([outEven, outOdd], axis: -1).reshaped(x.shape).asType(x.dtype)
    }

    static func apply3DRotaryEmbedding(
        q: MLXArray,
        k: MLXArray,
        cos1D: MLXArray,
        sin1D: MLXArray,
        cos2D: MLXArray?,
        sin2D: MLXArray?,
    )
        -> (MLXArray, MLXArray)
    {
        let half = q.dim(-1) / 2
        var qT = q[.ellipsis, ..<half]
        var qHW = q[.ellipsis, half...]
        var kT = k[.ellipsis, ..<half]
        var kHW = k[.ellipsis, half...]

        (qT, kT) = applyRotaryEmbedding1D(q: qT, k: kT, cos: cos1D, sin: sin1D)
        if let cos2D, let sin2D {
            qHW = applyGoldenRotaryEmbedding(qHW, cos2D: cos2D, sin2D: sin2D)
            kHW = applyGoldenRotaryEmbedding(kHW, cos2D: cos2D, sin2D: sin2D)
        }

        return (
            concatenated([qT, qHW], axis: -1).asType(q.dtype),
            concatenated([kT, kHW], axis: -1).asType(k.dtype),
        )
    }

    fileprivate final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let repetitions: Int
        let headDimension: Int
        let scale: Float
        let eps: Float
        let qSize: Int
        let kvSize: Int

        @ModuleInfo(key: "wqkv") var wqkv: Linear
        @ModuleInfo(key: "wo") var wo: Linear
        @ParameterInfo(key: "sinks") var sinks: MLXArray

        init(_ config: FalconPerceptionConfiguration.TextConfiguration) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            repetitions = config.attentionHeads / config.kvHeads
            headDimension = config.headDimension
            scale = pow(Float(config.headDimension), -0.5)
            eps = config.rmsNormEps
            qSize = config.attentionHeads * config.headDimension
            kvSize = config.kvHeads * config.headDimension
            _wqkv.wrappedValue = Linear(
                config.hiddenSize,
                qSize + 2 * kvSize,
                bias: false,
            )
            _wo.wrappedValue = Linear(qSize, config.hiddenSize, bias: false)
            _sinks.wrappedValue = MLXArray.zeros([config.attentionHeads])
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
            cos1D: MLXArray?,
            sin1D: MLXArray?,
            cos2D: MLXArray?,
            sin2D: MLXArray?,
        )
            -> MLXArray
        {
            let (batch, length) = (x.dim(0), x.dim(1))
            let xNorm = MLXFast.rmsNorm(x, weight: MLXArray.ones([x.dim(-1)]), eps: eps)
            let qkv = wqkv(xNorm)

            var q = qkv[.ellipsis, ..<qSize]
            var k = qkv[.ellipsis, qSize ..< (qSize + kvSize)]
            var v = qkv[.ellipsis, (qSize + kvSize)...]

            q = q.reshaped(batch, length, heads, headDimension).transposed(0, 2, 1, 3)
            k = k.reshaped(batch, length, kvHeads, headDimension).transposed(0, 2, 1, 3)
            v = v.reshaped(batch, length, kvHeads, headDimension).transposed(0, 2, 1, 3)

            let qkNormWeight = MLXArray.ones([headDimension])
            q = MLXFast.rmsNorm(q, weight: qkNormWeight, eps: eps)
            k = MLXFast.rmsNorm(k, weight: qkNormWeight, eps: eps)

            if repetitions > 1 {
                k = repeated(k, count: repetitions, axis: 1)
                v = repeated(v, count: repetitions, axis: 1)
            }

            if let cos1D, let sin1D {
                (q, k) = apply3DRotaryEmbedding(
                    q: q,
                    k: k,
                    cos1D: cos1D,
                    sin1D: sin1D,
                    cos2D: cos2D,
                    sin2D: sin2D,
                )
            }

            let output = attentionWithCacheUpdateAndSinks(
                queries: q,
                keys: k,
                values: v,
                cache: cache,
                scale: scale,
                mask: mask,
                sinks: sinks,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)
            return wo(output)
        }
    }

    fileprivate final class MLP: Module, UnaryLayer {
        let hiddenDimensions: Int
        let eps: Float

        @ModuleInfo(key: "w13") var w13: Linear
        @ModuleInfo(key: "w2") var w2: Linear

        init(_ config: FalconPerceptionConfiguration.TextConfiguration) {
            hiddenDimensions = config.intermediateSize
            eps = config.rmsNormEps
            _w13.wrappedValue = Linear(
                config.hiddenSize,
                2 * config.intermediateSize,
                bias: false,
            )
            _w2.wrappedValue = Linear(
                config.intermediateSize,
                config.hiddenSize,
                bias: false,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let xNorm = MLXFast.rmsNorm(x, weight: MLXArray.ones([x.dim(-1)]), eps: eps)
            let projected = w13(xNorm)
            let gate = projected[.ellipsis, ..<hiddenDimensions]
            let up = projected[.ellipsis, hiddenDimensions...]
            let activated = square(relu(gate)) * up
            return w2(activated)
        }
    }

    fileprivate final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo var mlp: MLP

        init(_ config: FalconPerceptionConfiguration.TextConfiguration) {
            _attention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(config)
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
            cos1D: MLXArray?,
            sin1D: MLXArray?,
            cos2D: MLXArray?,
            sin2D: MLXArray?,
        )
            -> MLXArray
        {
            var h =
                x
                + attention(
                    x,
                    mask: mask,
                    cache: cache,
                    cos1D: cos1D,
                    sin1D: sin1D,
                    cos2D: cos2D,
                    sin2D: sin2D,
                )
            h = h + mlp(h)
            return h
        }
    }

    fileprivate final class TransformerModel: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "img_projector") var imageProjector: Linear
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm
        @ParameterInfo(key: "cos_1d") var cos1D: MLXArray
        @ParameterInfo(key: "sin_1d") var sin1D: MLXArray
        @ParameterInfo(key: "freqs_cis_golden") var frequenciesGolden: MLXArray
        var lastHiddenState: MLXArray?

        init(
            _ config: FalconPerceptionConfiguration.TextConfiguration,
            base: FalconPerceptionConfiguration
        ) {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )

            let patchDimension =
                base.visionConfiguration.temporalPatchSize
                * base.visionConfiguration.spatialPatchSize
                * base.visionConfiguration.spatialPatchSize
                * base.visionConfiguration.channelSize
            _imageProjector.wrappedValue = Linear(
                patchDimension,
                config.hiddenSize,
                bias: false,
            )
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
                DecoderLayer(config)
            }
            _norm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )

            let ropeDimension = config.headDimension / 2
            let (cos1D, sin1D) = precomputeFrequencies(
                dimensions: ropeDimension,
                end: config.maxPositionEmbeddings,
                theta: config.ropeTheta,
            )
            _cos1D.wrappedValue = cos1D
            _sin1D.wrappedValue = sin1D
            _frequenciesGolden.wrappedValue = MLXArray.zeros([
                config.attentionHeads,
                ropeDimension / 2,
                2,
            ])
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
            positionIDs: MLXArray? = nil,
            posHW: MLXArray? = nil,
            mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        )
            -> MLXArray
        {
            var h: MLXArray
            if let inputEmbedding {
                h = inputEmbedding
            } else if let inputs {
                h = embedTokens(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            let length = h.dim(1)
            let effectivePositionIDs: MLXArray
            if let positionIDs {
                if positionIDs.ndim == 2 {
                    effectivePositionIDs = positionIDs
                } else {
                    effectivePositionIDs = positionIDs.reshaped(-1)[..<length]
                }
            } else if let cache, let first = cache.first {
                let offset = first.offset
                effectivePositionIDs = MLXArray(offset ..< (offset + length))
            } else {
                effectivePositionIDs = MLXArray(0 ..< length)
            }

            let cosValues = cos1D[effectivePositionIDs]
            let sinValues = sin1D[effectivePositionIDs]

            let cos2D: MLXArray?
            let sin2D: MLXArray?
            if let posHW {
                (cos2D, sin2D) = computeGoldenFrequencies(
                    frequencies: frequenciesGolden,
                    posHW: posHW,
                )
            } else {
                cos2D = nil
                sin2D = nil
            }

            let attentionMask =
                mask ?? createAttentionMask(h: h, cache: cache?.first, returnArray: true)
            for (i, layer) in layers.enumerated() {
                h = layer(
                    h,
                    mask: attentionMask,
                    cache: cache?[i],
                    cos1D: cosValues,
                    sin1D: sinValues,
                    cos2D: cos2D,
                    sin2D: sin2D,
                )
            }
            h = norm(h)
            lastHiddenState = h
            return h
        }
    }

    fileprivate final class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: TransformerModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        let config: FalconPerceptionConfiguration.TextConfiguration
        let kvHeads: [Int]
        var positionIDs: MLXArray?
        var posHW: MLXArray?
        var ropeDeltas: MLXArray?
        var attentionMask: MLXArray?

        init(
            _ config: FalconPerceptionConfiguration.TextConfiguration,
            base: FalconPerceptionConfiguration
        ) {
            self.config = config
            _model.wrappedValue = TransformerModel(config, base: base)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.hiddenSize,
                    config.vocabularySize,
                    bias: false,
                )
            }
            kvHeads = (0 ..< config.hiddenLayers).map { _ in config.attentionHeads }
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            state _: LMOutput.State? = nil,
            inputEmbedding: MLXArray? = nil,
        )
            -> LMOutput
        {
            let cacheOffset = cache?.first?.offset ?? 0
            let length: Int =
                if let inputEmbedding {
                    inputEmbedding.dim(1)
                } else if let inputs {
                    inputs.ndim > 1 ? inputs.dim(1) : 1
                } else {
                    1
                }

            var effectivePositionIDs = positionIDs
            var effectivePosHW = posHW
            var effectiveAttentionMask: MLXFast.ScaledDotProductAttentionMaskMode? = nil

            if let fullMask = attentionMask, length > 1 {
                let end = cacheOffset + length
                effectiveAttentionMask = .array(fullMask[0..., 0..., cacheOffset ..< end, ..<end])
            }

            if inputEmbedding != nil, cacheOffset == 0 {
                if let current = effectivePositionIDs {
                    effectivePositionIDs =
                        current.ndim == 2
                        ? current[0..., cacheOffset ..< (cacheOffset + length)]
                        : current[cacheOffset ..< (cacheOffset + length)]
                }
                if let current = effectivePosHW {
                    effectivePosHW = current[0..., cacheOffset ..< (cacheOffset + length), 0...]
                }
            } else if cacheOffset > 0, let ropeDeltas {
                let start = MLXArray(cacheOffset).asType(.int32) + ropeDeltas.asType(.int32)
                effectivePositionIDs =
                    start.reshaped(1, 1)
                    + MLXArray(0 ..< length).asType(.int32).reshaped(1, -1)
                effectivePosHW = nil
            } else if inputEmbedding == nil {
                effectivePositionIDs = nil
                effectivePosHW = nil
            }

            var out = model(
                inputs,
                cache: cache,
                inputEmbedding: inputEmbedding,
                positionIDs: effectivePositionIDs,
                posHW: effectivePosHW,
                mask: effectiveAttentionMask,
            )
            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }
            return LMOutput(logits: out.asType(model.embedTokens.weight.dtype))
        }
    }
}

private final class FalconPerceptionFourierEncoder: Module {
    @ModuleInfo var embed: Linear
    @ModuleInfo var transform: Linear

    init(inputDimensions: Int, featureDimensions: Int, outputDimensions: Int) {
        _embed.wrappedValue = Linear(
            inputDimensions,
            featureDimensions / 2,
            bias: false,
        )
        _transform.wrappedValue = Linear(
            featureDimensions,
            outputDimensions,
            bias: false,
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let frequencies = Float(2 * Double.pi) * embed(x)
        return transform(concatenated([cos(frequencies), sin(frequencies)], axis: -1))
    }
}

private final class FalconPerceptionBboxDecoder: Module {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear

    init(inputDimensions: Int, hiddenDimensions: Int, outputDimensions: Int) {
        _w1.wrappedValue = Linear(inputDimensions, hiddenDimensions, bias: false)
        _w2.wrappedValue = Linear(hiddenDimensions, outputDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(square(relu(w1(x))))
    }
}

public struct FalconPerceptionDetection: Sendable, Equatable {
    public let x: Float
    public let y: Float
    public let height: Float
    public let width: Float
}

// MARK: - Model

public final class FalconPerception: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "language_model") private var languageModel:
        FalconPerceptionLanguage.LanguageModel
    @ModuleInfo(key: "coord_encoder") private var coordEncoder: FalconPerceptionFourierEncoder
    @ModuleInfo(key: "coord_decoder") private var coordDecoder: FalconPerceptionBboxDecoder
    @ModuleInfo(key: "size_encoder") private var sizeEncoder: FalconPerceptionFourierEncoder
    @ModuleInfo(key: "size_decoder") private var sizeDecoder: FalconPerceptionBboxDecoder

    public let config: FalconPerceptionConfiguration
    public var vocabularySize: Int {
        config.vocabularySize
    }

    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public private(set) var detections: [FalconPerceptionDetection] = []
    private var pendingCoordXY: MLXArray?
    private var pendingSizeHW: MLXArray?
    private var currentCoord: (x: Float, y: Float)?
    private var currentSize: (height: Float, width: Float)?

    public init(_ config: FalconPerceptionConfiguration) {
        self.config = config
        _languageModel.wrappedValue = FalconPerceptionLanguage.LanguageModel(
            config.textConfiguration,
            base: config,
        )
        _coordEncoder.wrappedValue = FalconPerceptionFourierEncoder(
            inputDimensions: 2,
            featureDimensions: config.coordEncoderDimensions,
            outputDimensions: config.textConfiguration.hiddenSize,
        )
        _coordDecoder.wrappedValue = FalconPerceptionBboxDecoder(
            inputDimensions: config.textConfiguration.hiddenSize,
            hiddenDimensions: config.coordDecoderDimensions,
            outputDimensions: config.coordOutputDimensions,
        )
        _sizeEncoder.wrappedValue = FalconPerceptionFourierEncoder(
            inputDimensions: 2,
            featureDimensions: config.sizeEncoderDimensions,
            outputDimensions: config.textConfiguration.hiddenSize,
        )
        _sizeDecoder.wrappedValue = FalconPerceptionBboxDecoder(
            inputDimensions: config.textConfiguration.hiddenSize,
            hiddenDimensions: config.sizeDecoderDimensions,
            outputDimensions: config.sizeOutputDimensions,
        )
    }

    private func patchifyAndProject(_ pixelValues: MLXArray) -> MLXArray {
        let spatialPatchSize = config.visionConfiguration.spatialPatchSize
        let temporalPatchSize = config.visionConfiguration.temporalPatchSize
        var pixels = pixelValues
        if pixels.ndim == 3 {
            pixels = pixels[.newAxis, 0..., 0..., 0...]
        }

        let (imageCount, height, width, channels) = (
            pixels.dim(0),
            pixels.dim(1),
            pixels.dim(2),
            pixels.dim(3),
        )
        let heightPatches = height / spatialPatchSize
        let widthPatches = width / spatialPatchSize
        var patches = pixels.reshaped(
            imageCount,
            heightPatches,
            spatialPatchSize,
            widthPatches,
            spatialPatchSize,
            channels,
        )
        patches = patches.transposed(0, 1, 3, 2, 4, 5)
        patches = patches.reshaped(
            imageCount * heightPatches * widthPatches,
            spatialPatchSize * spatialPatchSize * channels * temporalPatchSize,
        )

        return languageModel.model.imageProjector(
            patches.asType(languageModel.model.imageProjector.weight.dtype),
        )
    }

    private func mergeImageFeatures(
        imageFeatures: MLXArray,
        inputEmbeds: MLXArray,
        inputIDs: MLXArray,
    )
        -> MLXArray
    {
        let imagePositions = inputIDs.asArray(Int.self).enumerated()
            .filter { $0.element == config.imageTokenID }
            .map(\.offset)

        var result = inputEmbeds
        if result.ndim == 2 {
            result = result[.newAxis, 0..., 0...]
        }
        result[0..., MLXArray(imagePositions), 0...] = imageFeatures[.newAxis, 0..., 0...]
        return result
    }

    private func inputEmbeddings(inputIDs: MLXArray, pixelValues: MLXArray?) -> MLXArray {
        guard let pixelValues else {
            return languageModel.model.embedTokens(inputIDs)
        }
        let inputEmbeds = languageModel.model.embedTokens(inputIDs)
        let imageFeatures = patchifyAndProject(pixelValues)
        return mergeImageFeatures(
            imageFeatures: imageFeatures,
            inputEmbeds: inputEmbeds,
            inputIDs: inputIDs,
        )
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let pixels = input.image?.pixels
        guard let pixels else {
            resetPerceptionState()
            languageModel.positionIDs = nil
            languageModel.posHW = nil
            languageModel.ropeDeltas = nil
            languageModel.attentionMask = nil
            let result = languageModel(input.text.tokens, cache: cache)
            updatePendingPredictions()
            prefill.progress?(input.text.tokens.size, input.text.tokens.size)
            return .logits(result)
        }

        resetPerceptionState()
        let embeddings = inputEmbeddings(inputIDs: input.text.tokens, pixelValues: pixels)
        let (positionIDs, posHW, ropeDeltas, attentionMask) = getRopeIndex(
            inputIDs: input.text.tokens,
            frames: input.image?.frames,
        )

        languageModel.positionIDs = positionIDs
        languageModel.posHW = posHW
        languageModel.ropeDeltas = ropeDeltas
        languageModel.attentionMask = attentionMask

        let result = languageModel(nil, cache: cache, inputEmbedding: embeddings)
        updatePendingPredictions()
        prefill.progress?(embeddings.dim(1), embeddings.dim(1))
        return .logits(result)
    }

    public func callAsFunction(
        _ input: LMInput.Text,
        cache: [any KVCache]?,
        state: LMOutput.State?,
    )
        -> LMOutput
    {
        let tokenIDs = input.tokens.asArray(Int.self)
        let cacheOffset = cache?.first?.offset ?? 0
        let usePerceptionEmbedding = cacheOffset > 0 && tokenIDs.count == 1
        let result: LMOutput

        if usePerceptionEmbedding {
            let tokenID = tokenIDs[0]
            accumulateDetection(for: tokenID)
            let inputIDs =
                input.tokens.ndim == 2
                ? input.tokens
                : input.tokens.reshaped(1, -1)
            let embedding = perceptionEmbedding(inputIDs: inputIDs, tokenID: tokenID)
            result = languageModel(nil, cache: cache, state: state, inputEmbedding: embedding)
        } else {
            result = languageModel(input.tokens, cache: cache, state: state)
        }
        updatePendingPredictions()
        return result
    }

    private func resetPerceptionState() {
        pendingCoordXY = nil
        pendingSizeHW = nil
        currentCoord = nil
        currentSize = nil
        detections.removeAll()
    }

    private func perceptionEmbedding(inputIDs: MLXArray, tokenID: Int) -> MLXArray {
        var embedding = languageModel.model.embedTokens(inputIDs)
        if tokenID == config.coordTokenID, let pendingCoordXY {
            embedding = coordEncoder(pendingCoordXY.reshaped(-1, 2)).reshaped(1, 1, -1)
        } else if tokenID == config.sizeTokenID, let pendingSizeHW {
            embedding = sizeEncoder(pendingSizeHW.reshaped(-1, 2)).reshaped(1, 1, -1)
        }
        pendingCoordXY = nil
        pendingSizeHW = nil
        return embedding
    }

    private func accumulateDetection(for tokenID: Int) {
        if tokenID == config.coordTokenID,
            let coord = pendingCoordXY?.asArray(Float.self),
            coord.count >= 2
        {
            currentCoord = (x: coord[0], y: coord[1])
        } else if tokenID == config.sizeTokenID,
            let size = pendingSizeHW?.asArray(Float.self),
            size.count >= 2
        {
            currentSize = (height: size[0], width: size[1])
            appendDetectionIfComplete()
        } else if tokenID == config.segTokenID || tokenID == config.eosTokenID {
            appendDetectionIfComplete()
        }
    }

    private func appendDetectionIfComplete() {
        guard let currentCoord, let currentSize else {
            return
        }
        let detection = FalconPerceptionDetection(
            x: currentCoord.x,
            y: currentCoord.y,
            height: currentSize.height,
            width: currentSize.width,
        )
        if detections.last != detection {
            detections.append(detection)
        }
        self.currentCoord = nil
        self.currentSize = nil
    }

    private func updatePendingPredictions() {
        guard let hiddenState = languageModel.model.lastHiddenState else {
            return
        }
        let lastHiddenState = hiddenState[0..., (hiddenState.dim(1) - 1)..., 0...]
        pendingCoordXY = decodeCoordinates(lastHiddenState)
        pendingSizeHW = decodeSizes(lastHiddenState)
    }

    private func decodeCoordinates(_ hiddenState: MLXArray) -> MLXArray {
        let logits = coordDecoder(hiddenState)
        let binsPerAxis = max(config.coordOutputDimensions / 2, 1)
        let bins = argMax(logits.reshaped(-1, 2, binsPerAxis)[0], axis: -1)
            .asArray(Int.self)
        let denominator = Float(max(binsPerAxis - 1, 1))
        let x = Float(bins.first ?? 0) / denominator
        let y = Float(bins.dropFirst().first ?? 0) / denominator
        return MLXArray([x, y]).reshaped(1, 2)
    }

    private func decodeSizes(_ hiddenState: MLXArray) -> MLXArray {
        let logits = sizeDecoder(hiddenState)
        let binsPerAxis = max(config.sizeOutputDimensions / 2, 1)
        let bins = argMax(logits.reshaped(-1, 2, binsPerAxis)[0], axis: -1)
            .asArray(Int.self)
        let denominator = Float(max(binsPerAxis - 1, 1))
        let minSize = log2(Float(1) / Float(max(binsPerAxis, 1)))
        let hPred = Float(bins.first ?? 0) / denominator
        let wPred = Float(bins.dropFirst().first ?? 0) / denominator
        let height = pow(Float(2), hPred * (0 - minSize) + minSize)
        let width = pow(Float(2), wPred * (0 - minSize) + minSize)
        return MLXArray([height, width]).reshaped(1, 2)
    }

    private func getRopeIndex(inputIDs: MLXArray, frames: [THW]?)
        -> (MLXArray, MLXArray, MLXArray, MLXArray)
    {
        let ids = inputIDs.asArray(Int.self)
        var positionIDs = [Int32]()
        positionIDs.reserveCapacity(ids.count)
        var inImage = false
        var nextPosition: Int32 = 0

        for token in ids {
            if token == config.imageClassTokenID, !inImage {
                inImage = true
            }
            positionIDs.append(nextPosition)
            if !inImage {
                nextPosition += 1
            }
            if token == config.imageEndTokenID, inImage {
                inImage = false
                nextPosition += 1
            }
        }

        let delta = (positionIDs.max() ?? -1) + 1 - Int32(ids.count)
        let posHW = computePositionHW(inputIDs: ids, frames: frames)
        let mask = createFalconPerceptionMask(inputIDs: ids)
        return (
            MLXArray(positionIDs),
            posHW,
            MLXArray([delta]).asType(.int32).reshaped(1, 1),
            mask,
        )
    }

    private func computePositionHW(inputIDs: [Int], frames: [THW]?) -> MLXArray {
        var coordinates = [(Float, Float)]()
        if let frames {
            for frame in frames {
                let gridH = frame.h
                let gridW = frame.w
                for h in 0 ..< gridH {
                    for w in 0 ..< gridW {
                        let hScale = sqrt(Float(gridH) / Float(gridW))
                        let wScale = sqrt(Float(gridW) / Float(gridH))
                        let hValue = -hScale + 2 * hScale * Float(h) / Float(max(gridH - 1, 1))
                        let wValue = -wScale + 2 * wScale * Float(w) / Float(max(gridW - 1, 1))
                        coordinates.append((hValue, wValue))
                    }
                }
            }
        }

        var values = Array(repeating: Float(0), count: inputIDs.count * 2)
        var coordinateIndex = 0
        for (tokenIndex, token) in inputIDs.enumerated() where token == config.imageTokenID {
            if coordinateIndex < coordinates.count {
                let coordinate = coordinates[coordinateIndex]
                values[tokenIndex * 2] = coordinate.0
                values[tokenIndex * 2 + 1] = coordinate.1
                coordinateIndex += 1
            }
        }

        return MLXArray(values).reshaped(1, inputIDs.count, 2)
    }

    private func createFalconPerceptionMask(inputIDs: [Int]) -> MLXArray {
        let sequenceLength = inputIDs.count
        var inImage = Array(repeating: false, count: sequenceLength)
        var blockIDs = Array(repeating: 0, count: sequenceLength)
        var openBlocks = 0
        var closedBlocks = 0

        for (index, token) in inputIDs.enumerated() {
            if token == config.imageClassTokenID {
                openBlocks += 1
            }
            if token == config.imageEndTokenID {
                closedBlocks += 1
            }
            let inside = openBlocks - closedBlocks > 0
            inImage[index] = inside
            blockIDs[index] = inside ? openBlocks : 0
        }

        var values = Array(repeating: false, count: sequenceLength * sequenceLength)
        for query in 0 ..< sequenceLength {
            for key in 0 ..< sequenceLength {
                let causal = query >= key
                let sameImage =
                    inImage[query] && inImage[key] && blockIDs[query] == blockIDs[key]
                if causal || sameImage {
                    values[query * sequenceLength + key] = true
                }
            }
        }
        return MLXArray(values).reshaped(1, 1, sequenceLength, sequenceLength)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var transformed = [String: MLXArray]()
        for (key, value) in weights {
            if key.hasPrefix("proj_segm.") || key.hasPrefix("conv_segm.")
                || key.hasPrefix("itok_upsampler.")
            {
                continue
            }

            var newKey = key
            var newValue = value

            if key.hasPrefix("tok_embeddings.") {
                newKey = key.replacingOccurrences(
                    of: "tok_embeddings.",
                    with: "language_model.model.embed_tokens.",
                    options: [.anchored],
                )
            } else if key.hasPrefix("img_projector.") {
                newKey = key.replacingOccurrences(
                    of: "img_projector.",
                    with: "language_model.model.img_projector.",
                    options: [.anchored],
                )
            } else if key.hasPrefix("norm.") {
                newKey = key.replacingOccurrences(
                    of: "norm.",
                    with: "language_model.model.norm.",
                    options: [.anchored],
                )
            } else if key.hasPrefix("output.") {
                newKey = key.replacingOccurrences(
                    of: "output.",
                    with: "language_model.lm_head.",
                    options: [.anchored],
                )
            } else if key == "freqs_cis_golden" {
                newKey = "language_model.model.freqs_cis_golden"
            } else if key.hasPrefix("layers.") {
                newKey = key.replacingOccurrences(
                    of: "layers.",
                    with: "language_model.model.layers.",
                    options: [.anchored],
                )
                newKey = newKey.replacingOccurrences(of: ".attention.", with: ".self_attn.")
                newKey = newKey.replacingOccurrences(of: ".feed_forward.", with: ".mlp.")
            }

            if key.hasPrefix("layers."), newKey.contains(".w13."), newValue.ndim > 0 {
                let even = newValue[stride(from: 0, to: newValue.dim(0), by: 2), .ellipsis]
                let odd = newValue[stride(from: 1, to: newValue.dim(0), by: 2), .ellipsis]
                newValue = concatenated([even, odd], axis: 0)
            }

            transformed[newKey] = newValue
        }
        transformed["language_model.model.cos_1d"] = languageModel.model.cos1D
        transformed["language_model.model.sin_1d"] = languageModel.model.sin1D
        return transformed
    }
}

private func attentionWithCacheUpdateAndSinks(
    queries: MLXArray,
    keys: MLXArray,
    values: MLXArray,
    cache: KVCache?,
    scale: Float,
    mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
    sinks: MLXArray? = nil,
)
    -> MLXArray
{
    guard let cache else {
        return MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: mask,
            sinks: sinks,
        )
    }
    if let quantizedKVCache = cache as? QuantizedKVCacheProtocol {
        precondition(sinks == nil, "Quantized SDPA does not support attention sinks.")
        let (quantizedKeys, quantizedValues) = quantizedKVCache.updateQuantized(
            keys: keys,
            values: values,
        )
        return quantizedScaledDotProductAttention(
            queries: queries,
            quantizedKeys: quantizedKeys,
            quantizedValues: quantizedValues,
            scale: scale,
            mask: mask,
            groupSize: quantizedKVCache.groupSize,
            bits: quantizedKVCache.bits,
            mode: quantizedKVCache.mode,
        )
    }

    let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
    return MLXFast.scaledDotProductAttention(
        queries: queries,
        keys: cachedKeys,
        values: cachedValues,
        scale: scale,
        mask: mask,
        sinks: sinks,
    )
}
