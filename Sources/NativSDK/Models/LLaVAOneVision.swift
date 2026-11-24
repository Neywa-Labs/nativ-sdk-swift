import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct LLaVAOneVisionConfiguration: Decodable, Sendable {
    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let rmsNormEps: Float
        public let vocabularySize: Int
        public let kvHeads: Int
        public let ropeTheta: Float
        public let ropeTraditional: Bool
        public let ropeScaling: [String: StringOrNumber]?
        public let tieWordEmbeddings: Bool
        public let attentionBias: Bool

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case kvHeads = "num_key_value_heads"
            case ropeTheta = "rope_theta"
            case ropeTraditional = "rope_traditional"
            case ropeScaling = "rope_scaling"
            case tieWordEmbeddings = "tie_word_embeddings"
            case attentionBias = "attention_bias"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen2"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 896
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 24
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4864
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 14
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
            ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
            ropeScaling = try c.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeScaling)
            tieWordEmbeddings =
                try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
            attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? true
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenLayers: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let imageSize: Int
        public let patchSize: Int
        public let channels: Int
        public let layerNormEps: Float
        public let visionUseHead: Bool

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenLayers = "num_hidden_layers"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case channels = "num_channels"
            case layerNormEps = "layer_norm_eps"
            case visionUseHead = "vision_use_head"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType) ?? "siglip_vision_model"
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 26
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 384
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
            visionUseHead = try c.decodeIfPresent(Bool.self, forKey: .visionUseHead) ?? false
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    public let imageTokenIndex: Int
    public let videoTokenIndex: Int
    public let imageGridPinpoints: [[Int]]
    public let visionAspectRatio: String
    public let visionFeatureSelectStrategy: String
    public let visionFeatureLayer: Int
    public let useImageNewlineParameter: Bool

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case imageTokenIndex = "image_token_index"
        case videoTokenIndex = "video_token_index"
        case imageGridPinpoints = "image_grid_pinpoints"
        case visionAspectRatio = "vision_aspect_ratio"
        case visionFeatureSelectStrategy = "vision_feature_select_strategy"
        case visionFeatureLayer = "vision_feature_layer"
        case useImageNewlineParameter = "use_image_newline_parameter"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration = try c.decode(TextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try c.decode(VisionConfiguration.self, forKey: .visionConfiguration)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "llava_onevision"
        imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? 151_646
        videoTokenIndex = try c.decodeIfPresent(Int.self, forKey: .videoTokenIndex) ?? 151_647
        imageGridPinpoints =
            try c.decodeIfPresent([[Int]].self, forKey: .imageGridPinpoints) ?? [[384, 384]]
        visionAspectRatio =
            try c.decodeIfPresent(String.self, forKey: .visionAspectRatio) ?? "anyres_max_9"
        visionFeatureSelectStrategy =
            try c.decodeIfPresent(String.self, forKey: .visionFeatureSelectStrategy) ?? "full"
        visionFeatureLayer = try c.decodeIfPresent(Int.self, forKey: .visionFeatureLayer) ?? -1
        useImageNewlineParameter =
            try c.decodeIfPresent(Bool.self, forKey: .useImageNewlineParameter) ?? true
    }
}

/// Parses the trailing tile budget out of a `anyres_max_N` aspect-ratio string.
func llavaOneVisionMaxTiles(_ visionAspectRatio: String) -> Int {
    let digits = visionAspectRatio.reversed().prefix { $0.isNumber }.reversed()
    return Int(String(digits)) ?? 9
}

// MARK: - Processor

public struct LLaVAOneVisionProcessorConfiguration: Decodable, Sendable {
    public struct Size: Decodable, Sendable {
        public let height: Int?
        public let width: Int?
        public let shortestEdge: Int?

        public var heightValue: Int { height ?? shortestEdge ?? 384 }
        public var widthValue: Int { width ?? shortestEdge ?? 384 }

        enum CodingKeys: String, CodingKey {
            case height
            case width
            case shortestEdge = "shortest_edge"
        }
    }

    public let processorClass: String?
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let size: Size

    private let _patchSize: Int?
    public var patchSize: Int { _patchSize ?? 14 }

    private let _imageToken: String?
    public var imageToken: String { _imageToken ?? "<image>" }

    private let _videoToken: String?
    public var videoToken: String { _videoToken ?? "<video>" }

    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int { _imageTokenIndex ?? 151_646 }

    private let _visionFeatureSelectStrategy: String?
    public var visionFeatureSelectStrategy: String { _visionFeatureSelectStrategy ?? "full" }

    /// Only `config.json` carries `vision_aspect_ratio`, and the factory hands processors a
    /// single JSON file, so this falls back to the value every published onevision uses.
    private let _visionAspectRatio: String?
    public var visionAspectRatio: String { _visionAspectRatio ?? "anyres_max_9" }

    private let _numImageTokens: Int?

    private let _imageGridPinpoints: [[Int]]?
    public var imageGridPinpoints: [[Int]] {
        _imageGridPinpoints ?? [[384, 384]]
    }

    /// Patch grid side of a single tile; equals `sqrt(num_image_tokens)` on every checkpoint.
    public var patchesPerSide: Int {
        if let numImageTokens = _numImageTokens, numImageTokens > 0 {
            let side = Int(Double(numImageTokens).squareRoot().rounded())
            if side * side == numImageTokens {
                return side
            }
        }
        return size.heightValue / patchSize
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case size
        case _patchSize = "patch_size"
        case _imageToken = "image_token"
        case _videoToken = "video_token"
        case _imageTokenIndex = "image_token_index"
        case _visionFeatureSelectStrategy = "vision_feature_select_strategy"
        case _visionAspectRatio = "vision_aspect_ratio"
        case _numImageTokens = "num_image_tokens"
        case _imageGridPinpoints = "image_grid_pinpoints"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass = try c.decodeIfPresent(String.self, forKey: .processorClass)
        imageMean = try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd = try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        size = try c.decodeIfPresent(Size.self, forKey: .size) ?? Size(384)
        _patchSize = try c.decodeIfPresent(Int.self, forKey: ._patchSize)
        _imageToken = try c.decodeIfPresent(String.self, forKey: ._imageToken)
        _videoToken = try c.decodeIfPresent(String.self, forKey: ._videoToken)
        _imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: ._imageTokenIndex)
        _visionFeatureSelectStrategy = try c.decodeIfPresent(
            String.self, forKey: ._visionFeatureSelectStrategy)
        _visionAspectRatio = try c.decodeIfPresent(String.self, forKey: ._visionAspectRatio)
        _numImageTokens = try c.decodeIfPresent(Int.self, forKey: ._numImageTokens)
        _imageGridPinpoints = try c.decodeIfPresent([[Int]].self, forKey: ._imageGridPinpoints)
    }
}

extension LLaVAOneVisionProcessorConfiguration.Size {
    fileprivate init(_ square: Int) {
        self.height = square
        self.width = square
        self.shortestEdge = nil
    }
}

public struct LLaVAOneVisionProcessor: UserInputProcessor {
    private let config: LLaVAOneVisionProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageTokenIndex: Int

    public init(_ config: LLaVAOneVisionProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        imageTokenIndex =
            tokenizer.convertTokenToId(config.imageToken) ?? config.imageTokenIndex
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = Self.images(from: input)
        let messages = LLaVAMessageGenerator().generate(from: input)
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext,
        )

        guard !images.isEmpty else {
            let tokensArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            return LMInput(text: .init(tokens: tokensArray, mask: ones(like: tokensArray)))
        }

        if input.videos.isEmpty == false {
            throw VLMError.processing(
                "LLaVA-OneVision video input is not implemented in Swift yet.")
        }

        let processed = try images.map {
            try preprocess(image: $0.asCIImage(), processing: input.processing)
        }

        let tileHeight = config.size.heightValue
        let tileWidth = config.size.widthValue
        let maxTiles = processed.map(\.tileCount).max() ?? 1
        var padded: [MLXArray] = []
        for image in processed {
            var tiles = image.pixels
            if image.tileCount < maxTiles {
                let padding = MLXArray.zeros(
                    [maxTiles - image.tileCount, 3, tileHeight, tileWidth], dtype: tiles.dtype)
                tiles = concatenated([tiles, padding], axis: 0)
            }
            padded.append(tiles.expandedDimensions(axis: 0))
        }
        let pixels = concatenated(padded, axis: 0)

        let expandedTokens = expandImageTokens(
            in: promptTokens, counts: processed.map(\.tokenCount))
        let tokensArray = MLXArray(expandedTokens).expandedDimensions(axis: 0)

        return LMInput(
            text: .init(tokens: tokensArray, mask: ones(like: tokensArray).asType(.int8)),
            image: .init(
                pixels: pixels,
                frames: processed.map { THW(1, $0.originalHeight, $0.originalWidth) },
            ),
        )
    }

    fileprivate struct ProcessedImage {
        let pixels: MLXArray
        let tileCount: Int
        let tokenCount: Int
        let originalHeight: Int
        let originalWidth: Int
    }

    /// Replaces the i-th image placeholder with `counts[i]` copies of the image token.
    private func expandImageTokens(in promptTokens: [Int], counts: [Int]) -> [Int] {
        guard !counts.isEmpty else { return promptTokens }

        var tokens: [Int] = []
        var consumed = 0
        for token in promptTokens {
            guard token == imageTokenIndex, consumed < counts.count else {
                tokens.append(token)
                continue
            }
            tokens.append(
                contentsOf: Array(repeating: imageTokenIndex, count: max(counts[consumed], 1)))
            consumed += 1
        }

        guard consumed > 0 else {
            let expanded = counts.flatMap {
                Array(repeating: imageTokenIndex, count: max($0, 1))
            }
            return expanded + promptTokens
        }
        return tokens
    }
}

extension LLaVAOneVisionProcessor {
    fileprivate func preprocess(image: CIImage, processing: UserInput.Processing?) throws
        -> ProcessedImage
    {
        let canonical = Self.zeroOrigin(MediaProcessing.apply(image, processing: processing))
        let originalHeight = max(Int(canonical.extent.height.rounded()), 1)
        let originalWidth = max(Int(canonical.extent.width.rounded()), 1)
        let tileHeight = config.size.heightValue
        let tileWidth = config.size.widthValue
        guard tileHeight == tileWidth else {
            throw VLMError.processing("LLaVA-OneVision expects square tiles.")
        }

        let base = Self.render(
            canonical.resampled(
                to: CGSize(width: tileWidth, height: tileHeight), method: .bicubic),
            width: tileWidth,
            height: tileHeight,
        )

        let best = Self.selectBestResolution(
            height: originalHeight,
            width: originalWidth,
            pinpoints: config.imageGridPinpoints,
        )
        let fitted = Self.patchOutputSize(
            originalHeight: originalHeight,
            originalWidth: originalWidth,
            targetHeight: best.height,
            targetWidth: best.width,
        )
        let resized = Self.render(
            canonical.resampled(
                to: CGSize(width: fitted.width, height: fitted.height), method: .bicubic),
            width: fitted.width,
            height: fitted.height,
        )
        // transformers pads the resized image with zeros *before* rescale/normalize, so the
        // padded band lands on -1 after normalization rather than on the dataset mean.
        let paddedPixels = Self.pad(
            resized,
            width: fitted.width,
            height: fitted.height,
            targetWidth: best.width,
            targetHeight: best.height,
        )

        var tiles = [
            Self.normalizeCHW(
                base, width: tileWidth, height: tileHeight,
                mean: config.imageMean, std: config.imageStd)
        ]
        let rows = max(best.height / tileHeight, 1)
        let columns = max(best.width / tileWidth, 1)
        for row in 0 ..< rows {
            for column in 0 ..< columns {
                let tile = Self.crop(
                    paddedPixels,
                    sourceWidth: best.width,
                    x: column * tileWidth,
                    y: row * tileHeight,
                    width: tileWidth,
                    height: tileHeight,
                )
                tiles.append(
                    Self.normalizeCHW(
                        tile, width: tileWidth, height: tileHeight,
                        mean: config.imageMean, std: config.imageStd))
            }
        }

        let pixels = MLXArray(tiles.flatMap(\.self), [tiles.count, 3, tileHeight, tileWidth])
        return ProcessedImage(
            pixels: pixels,
            tileCount: tiles.count,
            tokenCount: imageTokenCount(
                originalHeight: originalHeight, originalWidth: originalWidth),
            originalHeight: originalHeight,
            originalWidth: originalWidth,
        )
    }

    /// Number of language positions one image expands to, matching the reference processor.
    func imageTokenCount(originalHeight: Int, originalWidth: Int) -> Int {
        let tileHeight = config.size.heightValue
        let tileWidth = config.size.widthValue
        let best = Self.selectBestResolution(
            height: originalHeight, width: originalWidth, pinpoints: config.imageGridPinpoints)
        let patchesSide = config.patchesPerSide
        var currentHeight = patchesSide * (best.height / tileHeight)
        var currentWidth = patchesSide * (best.width / tileWidth)

        let originalAspect = Double(originalWidth) / Double(originalHeight)
        let currentAspect = Double(currentWidth) / Double(currentHeight)
        if originalAspect > currentAspect {
            let scaled =
                Double(originalHeight) * (Double(currentWidth) / Double(originalWidth))
            let newHeight = Int((scaled * 1e7).rounded() / 1e7)
            currentHeight -= ((currentHeight - newHeight) / 2) * 2
        } else {
            let scaled =
                Double(originalWidth) * (Double(currentHeight) / Double(originalHeight))
            let newWidth = Int((scaled * 1e7).rounded() / 1e7)
            currentWidth -= ((currentWidth - newWidth) / 2) * 2
        }

        var unpadded = currentHeight * currentWidth
        var newlines = currentHeight
        let maxTiles = llavaOneVisionMaxTiles(config.visionAspectRatio)
        let ratio =
            (Double(currentHeight * currentWidth) / Double(maxTiles * patchesSide * patchesSide))
            .squareRoot()
        if ratio > 1.1 {
            unpadded =
                Int((Double(currentHeight) / ratio).rounded(.down))
                * Int((Double(currentWidth) / ratio).rounded(.down))
            newlines = Int((Double(currentHeight) / ratio).rounded(.down))
        }

        var total = unpadded + newlines + patchesSide * patchesSide
        if config.visionFeatureSelectStrategy == "default" {
            total -= 1
        }
        return total
    }

    fileprivate static func patchOutputSize(
        originalHeight: Int, originalWidth: Int, targetHeight: Int, targetWidth: Int,
    )
        -> (height: Int, width: Int)
    {
        let scaleWidth = Double(targetWidth) / Double(originalWidth)
        let scaleHeight = Double(targetHeight) / Double(originalHeight)
        if scaleWidth < scaleHeight {
            let height = min(
                Int((Double(originalHeight) * scaleWidth).rounded(.up)), targetHeight)
            return (height, targetWidth)
        }
        let width = min(Int((Double(originalWidth) * scaleHeight).rounded(.up)), targetWidth)
        return (targetHeight, width)
    }

    fileprivate static func selectBestResolution(
        height: Int, width: Int, pinpoints: [[Int]],
    )
        -> (height: Int, width: Int)
    {
        var best = pinpoints.first ?? [384, 384]
        var bestEffective = -1
        var bestWasted = Int.max
        for point in pinpoints where point.count >= 2 {
            let candidateHeight = point[0]
            let candidateWidth = point[1]
            let scale = min(
                Double(candidateWidth) / Double(width),
                Double(candidateHeight) / Double(height),
            )
            let downscaledWidth = Int(Double(width) * scale)
            let downscaledHeight = Int(Double(height) * scale)
            let effective = min(downscaledWidth * downscaledHeight, width * height)
            let wasted = candidateWidth * candidateHeight - effective
            if effective > bestEffective || (effective == bestEffective && wasted < bestWasted) {
                best = point
                bestEffective = effective
                bestWasted = wasted
            }
        }
        return (best[0], best[1])
    }

    fileprivate static func pad(
        _ pixels: [Float], width: Int, height: Int, targetWidth: Int, targetHeight: Int,
    )
        -> [Float]
    {
        var output = Array(repeating: Float(0), count: targetWidth * targetHeight * 3)
        let xOffset = max((targetWidth - width) / 2, 0)
        let yOffset = max((targetHeight - height) / 2, 0)
        for row in 0 ..< min(height, targetHeight) {
            for column in 0 ..< min(width, targetWidth) {
                let source = (row * width + column) * 3
                let target = ((yOffset + row) * targetWidth + xOffset + column) * 3
                output[target] = pixels[source]
                output[target + 1] = pixels[source + 1]
                output[target + 2] = pixels[source + 2]
            }
        }
        return output
    }

    fileprivate static func crop(
        _ pixels: [Float], sourceWidth: Int, x: Int, y: Int, width: Int, height: Int,
    )
        -> [Float]
    {
        var output = Array(repeating: Float(0), count: width * height * 3)
        for row in 0 ..< height {
            let source = ((y + row) * sourceWidth + x) * 3
            let target = row * width * 3
            output[target ..< target + width * 3] = pixels[source ..< source + width * 3]
        }
        return output
    }

    fileprivate static func normalizeCHW(
        _ pixels: [Float], width: Int, height: Int, mean: [CGFloat], std: [CGFloat],
    )
        -> [Float]
    {
        var output = Array(repeating: Float(0), count: 3 * width * height)
        let plane = width * height
        for index in 0 ..< plane {
            let source = index * 3
            output[index] = (pixels[source] - Float(mean[0])) / Float(std[0])
            output[plane + index] = (pixels[source + 1] - Float(mean[1])) / Float(std[1])
            output[2 * plane + index] = (pixels[source + 2] - Float(mean[2])) / Float(std[2])
        }
        return output
    }

    fileprivate static func render(_ image: CIImage, width: Int, height: Int) -> [Float] {
        let bytesPerPixel = 4
        let bytesPerRow = width * bytesPerPixel
        var data = Data(count: width * height * bytesPerPixel)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        data.withUnsafeMutableBytes { pointer in
            CIContext().render(
                zeroOrigin(image),
                toBitmap: pointer.baseAddress!,
                rowBytes: bytesPerRow,
                bounds: CGRect(x: 0, y: 0, width: width, height: height),
                format: .RGBA8,
                colorSpace: colorSpace,
            )
        }
        let rgba = data.withUnsafeBytes { Array($0.bindMemory(to: UInt8.self)) }
        var rgb = Array(repeating: Float(0), count: width * height * 3)
        for index in 0 ..< (width * height) {
            let source = index * 4
            let target = index * 3
            let alpha = max(Float(rgba[source + 3]) / 255, Float(1e-6))
            rgb[target] = min(max((Float(rgba[source]) / 255) / alpha, 0), 1)
            rgb[target + 1] = min(max((Float(rgba[source + 1]) / 255) / alpha, 0), 1)
            rgb[target + 2] = min(max((Float(rgba[source + 2]) / 255) / alpha, 0), 1)
        }
        return rgb
    }

    fileprivate static func zeroOrigin(_ image: CIImage) -> CIImage {
        let rect = image.extent.integral
        return
            image
            .cropped(to: rect)
            .transformed(by: CGAffineTransform(translationX: -rect.minX, y: -rect.minY))
    }

    fileprivate static func images(from input: UserInput) -> [UserInput.Image] {
        if !input.images.isEmpty {
            return input.images
        }
        guard case .chat(let messages) = input.prompt else {
            return []
        }
        return messages.flatMap(\.images)
    }
}

// MARK: - Vision

private enum LLaVAOneVisionVision {
    fileprivate final class Attention: Module {
        let numHeads: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "out_proj") var wo: Linear

        init(dimensions: Int, numHeads: Int, bias: Bool = true) {
            precondition(dimensions % numHeads == 0)
            self.numHeads = numHeads
            scale = pow(Float(dimensions / numHeads), -0.5)
            _wq.wrappedValue = Linear(dimensions, dimensions, bias: bias)
            _wk.wrappedValue = Linear(dimensions, dimensions, bias: bias)
            _wv.wrappedValue = Linear(dimensions, dimensions, bias: bias)
            _wo.wrappedValue = Linear(dimensions, dimensions, bias: bias)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (batch, length, _) = (x.dim(0), x.dim(1), x.dim(2))
            let queries = wq(x).reshaped(batch, length, numHeads, -1).transposed(0, 2, 1, 3)
            let keys = wk(x).reshaped(batch, length, numHeads, -1).transposed(0, 2, 1, 3)
            let values = wv(x).reshaped(batch, length, numHeads, -1).transposed(0, 2, 1, 3)
            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale, mask: .none,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)
            return wo(output)
        }
    }

    fileprivate final class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
        @ModuleInfo var mlp: TanhGELUMLP

        init(_ config: LLaVAOneVisionConfiguration.VisionConfiguration) {
            _attention.wrappedValue = Attention(
                dimensions: config.hiddenSize, numHeads: config.attentionHeads, bias: true)
            _layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            _layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            _mlp.wrappedValue = TanhGELUMLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let attended = x + attention(layerNorm1(x))
            return attended + mlp(layerNorm2(attended))
        }
    }

    fileprivate final class Encoder: Module {
        let layers: [EncoderLayer]

        init(_ config: LLaVAOneVisionConfiguration.VisionConfiguration) {
            layers = (0 ..< config.hiddenLayers).map { _ in EncoderLayer(config) }
        }
    }

    fileprivate final class VisionEmbeddings: Module {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        private let numPositions: Int

        init(_ config: LLaVAOneVisionConfiguration.VisionConfiguration) {
            _patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.channels,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: true,
            )
            let patchesPerSide = config.imageSize / config.patchSize
            numPositions = patchesPerSide * patchesPerSide
            _positionEmbedding.wrappedValue = Embedding(
                embeddingCount: numPositions, dimensions: config.hiddenSize)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let patches = patchEmbedding(x).flattened(start: 1, end: 2)
            let positionIDs = MLXArray(0 ..< numPositions).expandedDimensions(axis: 0)
            return patches + positionEmbedding(positionIDs)
        }
    }

    fileprivate final class SigLipVisionModel: Module {
        @ModuleInfo var embeddings: VisionEmbeddings
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        init(_ config: LLaVAOneVisionConfiguration.VisionConfiguration) {
            _embeddings.wrappedValue = VisionEmbeddings(config)
            _encoder.wrappedValue = Encoder(config)
            _postLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, [MLXArray]?,
        ) {
            var hidden = embeddings(x)
            var encoderStates: [MLXArray]? = outputHiddenStates ? [hidden] : nil
            for layer in encoder.layers {
                hidden = layer(hidden)
                if outputHiddenStates {
                    encoderStates?.append(hidden)
                }
            }
            return (postLayerNorm(hidden), hidden, encoderStates)
        }
    }

    fileprivate final class VisionModel: Module {
        @ModuleInfo(key: "vision_model") var visionModel: SigLipVisionModel

        init(_ config: LLaVAOneVisionConfiguration.VisionConfiguration) {
            _visionModel.wrappedValue = SigLipVisionModel(config)
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, [MLXArray]?,
        ) {
            visionModel(x, outputHiddenStates: outputHiddenStates)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized: [String: MLXArray] = [:]
            for (key, value) in weights {
                if key.contains("position_ids") {
                    continue
                }
                if key.contains("patch_embedding.weight") {
                    sanitized[key] =
                        Self.isChannelsLast(value)
                        ? value
                        : value.transposed(
                            0, 2, 3, 1)
                } else if key.contains("head.attention.in_proj_weight") {
                    sanitized[
                        key.replacingOccurrences(of: "in_proj_weight", with: "in_proj.weight")] =
                        value
                } else if key.contains("head.attention.in_proj_bias") {
                    sanitized[key.replacingOccurrences(of: "in_proj_bias", with: "in_proj.bias")] =
                        value
                } else {
                    sanitized[key] = value
                }
            }
            return sanitized
        }

        private static func isChannelsLast(_ array: MLXArray) -> Bool {
            guard array.ndim == 4 else { return false }
            let (outputChannels, kernelHeight, kernelWidth) =
                (array.dim(0), array.dim(1), array.dim(2))
            return outputChannels >= kernelHeight && outputChannels >= kernelWidth
                && kernelHeight == kernelWidth
        }
    }
}

// MARK: - Language

private enum LLaVAOneVisionLanguage {
    fileprivate final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear
        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        init(_ config: LLaVAOneVisionConfiguration.TextConfiguration) {
            let dimensions = config.hiddenSize
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = dimensions / config.attentionHeads
            scale = pow(Float(headDim), -0.5)

            _wq.wrappedValue = Linear(dimensions, heads * headDim, bias: config.attentionBias)
            _wk.wrappedValue = Linear(dimensions, kvHeads * headDim, bias: config.attentionBias)
            _wv.wrappedValue = Linear(dimensions, kvHeads * headDim, bias: config.attentionBias)
            _wo.wrappedValue = Linear(heads * headDim, dimensions, bias: false)

            let ropeScale: Float =
                if let ropeScaling = config.ropeScaling,
                    ropeScaling["type"] == .string("linear"),
                    let factor = ropeScaling["factor"]?.asFloat()
                {
                    1 / factor
                } else {
                    1
                }
            _rotaryEmbedding.wrappedValue = RoPE(
                dimensions: headDim,
                traditional: config.ropeTraditional,
                base: config.ropeTheta,
                scale: ropeScale,
            )
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        )
            -> MLXArray
        {
            let (batch, length) = (x.dim(0), x.dim(1))
            var queries = wq(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            var keys = wk(x).reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
            let values = wv(x).reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

            let offset = cache?.offset ?? 0
            queries = rotaryEmbedding(queries, offset: offset)
            keys = rotaryEmbedding(keys, offset: offset)

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)
            return wo(output)
        }
    }

    fileprivate final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: LLaVAOneVisionConfiguration.TextConfiguration) {
            _attention.wrappedValue = Attention(config)
            mlp = MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        )
            -> MLXArray
        {
            let attended = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
            return attended + mlp(postAttentionLayerNorm(attended))
        }
    }

    fileprivate final class Qwen2Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        fileprivate let layers: [DecoderLayer]
        fileprivate let norm: RMSNorm

        init(_ config: LLaVAOneVisionConfiguration.TextConfiguration) {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            layers = (0 ..< config.hiddenLayers).map { _ in DecoderLayer(config) }
            norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil,
        )
            -> MLXArray
        {
            var hidden: MLXArray
            if let inputEmbedding {
                hidden = inputEmbedding
            } else if let inputs {
                hidden = embedTokens(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            let mask = createAttentionMask(h: hidden, cache: cache?.first)
            for (index, layer) in layers.enumerated() {
                hidden = layer(hidden, mask: mask, cache: cache?[index])
            }
            return norm(hidden)
        }
    }

    fileprivate final class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: Qwen2Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?
        let kvHeads: [Int]

        init(_ config: LLaVAOneVisionConfiguration.TextConfiguration) {
            model = Qwen2Model(config)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.hiddenSize, config.vocabularySize, bias: false)
            }
            kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            state: LMOutput.State? = nil,
            inputEmbedding: MLXArray? = nil,
        )
            -> LMOutput
        {
            let hidden = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
            let logits = lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
            return LMOutput(logits: logits, state: state)
        }
    }
}

// MARK: - Projector

private final class LLaVAOneVisionMultiModalProjector: Module, UnaryLayer {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    let gelu = GELU()

    init(_ config: LLaVAOneVisionConfiguration) {
        _linear1.wrappedValue = Linear(
            config.visionConfiguration.hiddenSize,
            config.textConfiguration.hiddenSize,
            bias: true,
        )
        _linear2.wrappedValue = Linear(
            config.textConfiguration.hiddenSize,
            config.textConfiguration.hiddenSize,
            bias: true,
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(gelu(linear1(x)))
    }
}

// MARK: - Model

public final class LLaVAOneVision: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: LLaVAOneVisionVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel:
        LLaVAOneVisionLanguage.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") private var projector:
        LLaVAOneVisionMultiModalProjector
    @ModuleInfo(key: "image_newline") private var imageNewline: MLXArray?

    public let config: LLaVAOneVisionConfiguration

    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.model.layers }

    private var patchesPerSide: Int {
        config.visionConfiguration.imageSize / config.visionConfiguration.patchSize
    }

    private var maxTiles: Int { llavaOneVisionMaxTiles(config.visionAspectRatio) }

    public init(_ config: LLaVAOneVisionConfiguration) {
        self.config = config
        _visionTower.wrappedValue = LLaVAOneVisionVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = LLaVAOneVisionLanguage.LanguageModel(
            config.textConfiguration)
        _projector.wrappedValue = LLaVAOneVisionMultiModalProjector(config)
        if config.useImageNewlineParameter {
            let std = 1 / sqrt(Float(config.textConfiguration.hiddenSize))
            _imageNewline.wrappedValue =
                MLXRandom.normal([config.textConfiguration.hiddenSize]) * std
        }
    }

    private func selectFeatures(_ hiddenStates: [MLXArray]) -> MLXArray {
        let layer =
            config.visionFeatureLayer >= 0
            ? config.visionFeatureLayer
            : hiddenStates.count + config.visionFeatureLayer
        var selected = hiddenStates[layer]
        if config.visionFeatureSelectStrategy == "default" {
            selected = selected[0..., 1..., 0...]
        }
        return selected
    }

    /// Tile count AnyRes produces for `size`, including the base tile.
    private func numPatches(forHeight height: Int, width: Int) -> Int {
        let grid = anyResGridShape(height: height, width: width)
        return grid.rows * grid.columns + 1
    }

    private func anyResGridShape(height: Int, width: Int) -> (rows: Int, columns: Int) {
        let best = LLaVAOneVisionProcessor.selectBestResolution(
            height: height, width: width, pinpoints: config.imageGridPinpoints)
        let tile = config.visionConfiguration.imageSize
        return (best.height / tile, best.width / tile)
    }

    private static func unpadImage(_ tensor: MLXArray, height: Int, width: Int) -> MLXArray {
        let currentHeight = tensor.dim(1)
        let currentWidth = tensor.dim(2)
        let originalAspect = Double(width) / Double(height)
        let currentAspect = Double(currentWidth) / Double(currentHeight)

        if originalAspect > currentAspect {
            let scale = Double(currentWidth) / Double(width)
            let newHeight = Int(Double(height) * scale)
            let padding = (currentHeight - newHeight) / 2
            guard padding > 0 else { return tensor }
            return tensor[0..., padding ..< (currentHeight - padding), 0...]
        }
        let scale = Double(currentHeight) / Double(height)
        let newWidth = Int(Double(width) * scale)
        let padding = (currentWidth - newWidth) / 2
        guard padding > 0 else { return tensor }
        return tensor[0..., 0..., padding ..< (currentWidth - padding)]
    }

    private func packImageFeatures(_ features: [MLXArray], sizes: [(height: Int, width: Int)])
        -> [MLXArray]
    {
        var packed: [MLXArray] = []
        for (index, feature) in features.enumerated() {
            guard feature.dim(0) > 1 else {
                var single = feature[0]
                if let imageNewline {
                    single = concatenated(
                        [
                            single,
                            imageNewline.expandedDimensions(axis: 0).asType(single.dtype),
                        ],
                        axis: 0,
                    )
                }
                packed.append(single)
                continue
            }

            let base = feature[0]
            let side = patchesPerSide
            let grid = anyResGridShape(height: sizes[index].height, width: sizes[index].width)
            var tiles = feature[1...]
                .reshaped(grid.rows, grid.columns, side, side, -1)
                .transposed(4, 0, 2, 1, 3)
            let channels = tiles.dim(0)
            tiles = tiles.reshaped(channels, grid.rows * side, grid.columns * side)
            tiles = Self.unpadImage(
                tiles, height: sizes[index].height, width: sizes[index].width)

            let currentHeight = tiles.dim(1)
            let currentWidth = tiles.dim(2)
            let ratio =
                (Double(currentHeight * currentWidth) / Double(maxTiles * side * side))
                .squareRoot()
            if ratio > 1.1 {
                tiles = resizeBilinear(
                    tiles.transposed(1, 2, 0),
                    size: (
                        Int((Double(currentHeight) / ratio).rounded(.down)),
                        Int((Double(currentWidth) / ratio).rounded(.down))
                    ),
                    antialias: false,
                )
                .transposed(2, 0, 1)
            }

            if let imageNewline {
                let newline = broadcast(
                    imageNewline[0..., .newAxis, .newAxis].asType(tiles.dtype),
                    to: [tiles.dim(0), tiles.dim(1), 1],
                )
                tiles = concatenated([tiles, newline], axis: 2)
            }

            let flattened = tiles.transposed(1, 2, 0).reshaped(-1, tiles.dim(0))
            packed.append(concatenated([base, flattened], axis: 0))
        }
        return packed
    }

    private func imageFeatures(pixelValues: MLXArray, sizes: [(height: Int, width: Int)])
        -> MLXArray
    {
        var perImagePatches: [Int]
        var flat: MLXArray

        if pixelValues.ndim == 5 {
            // The processor pads the tile axis to the batch maximum; drop the padding.
            perImagePatches = sizes.map { numPatches(forHeight: $0.height, width: $0.width) }
            flat = concatenated(
                perImagePatches.enumerated().map { index, count in
                    pixelValues[index, 0 ..< count, 0..., 0..., 0...]
                },
                axis: 0,
            )
        } else {
            flat = pixelValues
            perImagePatches = [pixelValues.dim(0)]
        }

        let (_, _, hiddenStates) = visionTower(
            flat.transposed(0, 2, 3, 1), outputHiddenStates: true)
        guard let hiddenStates else {
            return projector(flat)
        }
        let projected = projector(selectFeatures(hiddenStates))

        var split: [MLXArray] = []
        var start = 0
        for count in perImagePatches {
            split.append(projected[start ..< (start + count)])
            start += count
        }
        return concatenated(packImageFeatures(split, sizes: sizes), axis: 0)
    }

    private func scatter(
        _ embeddings: MLXArray, inputIDs: MLXArray, features: MLXArray, tokenIndex: Int,
    )
        -> MLXArray
    {
        let mask = inputIDs .== tokenIndex
        let placeholders = mask.sum().item(Int.self)
        guard placeholders == features.dim(0), placeholders > 0 else {
            return embeddings
        }
        let typed = features.asType(embeddings.dtype)
        let positions = (cumsum(mask.reshaped(-1).asType(.int32)) - 1).reshaped(mask.shape)
        let gathered = typed[clip(positions, min: 0, max: typed.dim(0) - 1)]
        return which(mask[.ellipsis, .newAxis], gathered, embeddings)
    }

    private func inputEmbeddings(_ input: LMInput) -> MLXArray {
        let inputIDs = input.text.tokens
        guard let pixelValues = input.image?.pixels else {
            return languageModel.model.embedTokens(inputIDs)
        }
        let embeddings = languageModel.model.embedTokens(inputIDs)
        let sizes =
            input.image?.frames?.map { (height: $0.h, width: $0.w) }
            ?? [
                (
                    height: config.visionConfiguration.imageSize,
                    width: config.visionConfiguration.imageSize
                )
            ]
        let features = imageFeatures(pixelValues: pixelValues, sizes: sizes)
        return scatter(
            embeddings,
            inputIDs: inputIDs,
            features: features,
            tokenIndex: config.imageTokenIndex,
        )
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache],
        state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        let embeddings = inputEmbeddings(input)
        let totalPositions = embeddings.dim(1)
        let processed = try prefill.forEachChunk(total: totalPositions) { range in
            _ = languageModel(nil, cache: cache, inputEmbedding: embeddings[0..., range, 0...])
            asyncEval(cache)
        }
        eval(cache)

        let result = languageModel(
            nil, cache: cache, inputEmbedding: embeddings[0..., processed..., 0...])
        prefill.progress?(totalPositions, totalPositions)
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let renames = [
            ("model.vision_tower.", "vision_tower."),
            ("model.multi_modal_projector.", "multi_modal_projector."),
            ("model.language_model.", "language_model.model."),
            ("model.image_newline", "image_newline"),
        ]

        var renamed: [String: MLXArray] = [:]
        for (key, value) in weights where !key.contains("self_attn.rotary_emb.inv_freq") {
            var name = key
            if let rename = renames.first(where: { key.hasPrefix($0.0) }) {
                name = rename.1 + key.dropFirst(rename.0.count)
            } else if key == "lm_head.weight" {
                name = "language_model.lm_head.weight"
            }
            if name.hasPrefix("vision_tower."), !name.hasPrefix("vision_tower.vision_model.") {
                name = "vision_tower.vision_model." + name.dropFirst("vision_tower.".count)
            }
            renamed[name] = value
        }

        if config.textConfiguration.tieWordEmbeddings {
            renamed["language_model.lm_head.weight"] = nil
        }
        return visionTower.sanitize(weights: renamed)
    }
}
