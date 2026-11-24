import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct Granite4VisionConfiguration: Decodable, Sendable {
    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        private let _sharedIntermediateSize: Int?
        public var sharedIntermediateSize: Int? {
            _sharedIntermediateSize
        }

        public var useSharedMLP: Bool {
            _sharedIntermediateSize != nil
        }

        public let attentionHeads: Int
        private let _kvHeads: Int?
        public var kvHeads: Int {
            _kvHeads ?? attentionHeads
        }

        public let rmsNormEps: Float
        public let vocabularySize: Int
        private let _ropeTheta: Float?
        public var ropeTheta: Float {
            _ropeTheta ?? 10_000_000
        }

        private let _ropeTraditional: Bool?
        public var ropeTraditional: Bool {
            _ropeTraditional ?? false
        }

        public let ropeScaling: [String: StringOrNumber]?
        private let _maxPositionEmbeddings: Int?
        public var maxPositionEmbeddings: Int {
            _maxPositionEmbeddings ?? 131_072
        }

        private let _tieWordEmbeddings: Bool?
        public var tieWordEmbeddings: Bool {
            _tieWordEmbeddings ?? true
        }

        private let _attentionBias: Bool?
        public var attentionBias: Bool {
            _attentionBias ?? false
        }

        private let _mlpBias: Bool?
        public var mlpBias: Bool {
            _mlpBias ?? false
        }

        private let _embeddingMultiplier: Float?
        public var embeddingMultiplier: Float {
            _embeddingMultiplier ?? 12.0
        }

        private let _attentionMultiplier: Float?
        public var attentionMultiplier: Float {
            _attentionMultiplier ?? 0.015625
        }

        private let _residualMultiplier: Float?
        public var residualMultiplier: Float {
            _residualMultiplier ?? 0.22
        }

        private let _logitsScaling: Float?
        public var logitsScaling: Float {
            _logitsScaling ?? 10.0
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case _sharedIntermediateSize = "shared_intermediate_size"
            case attentionHeads = "num_attention_heads"
            case _kvHeads = "num_key_value_heads"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case _ropeTheta = "rope_theta"
            case _ropeTraditional = "rope_traditional"
            case ropeScaling = "rope_scaling"
            case _maxPositionEmbeddings = "max_position_embeddings"
            case _tieWordEmbeddings = "tie_word_embeddings"
            case _attentionBias = "attention_bias"
            case _mlpBias = "mlp_bias"
            case _embeddingMultiplier = "embedding_multiplier"
            case _attentionMultiplier = "attention_multiplier"
            case _residualMultiplier = "residual_multiplier"
            case _logitsScaling = "logits_scaling"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "granitemoehybrid"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2560
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 40
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 8192
            _sharedIntermediateSize = try c.decodeIfPresent(
                Int.self, forKey: ._sharedIntermediateSize,
            )
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 40
            _kvHeads = try c.decodeIfPresent(Int.self, forKey: ._kvHeads)
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
            vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 100_353
            _ropeTheta = try c.decodeIfPresent(Float.self, forKey: ._ropeTheta)
            _ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: ._ropeTraditional)
            ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
            _maxPositionEmbeddings = try c.decodeIfPresent(
                Int.self, forKey: ._maxPositionEmbeddings,
            )
            _tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: ._tieWordEmbeddings)
            _attentionBias = try c.decodeIfPresent(Bool.self, forKey: ._attentionBias)
            _mlpBias = try c.decodeIfPresent(Bool.self, forKey: ._mlpBias)
            _embeddingMultiplier = try c.decodeIfPresent(Float.self, forKey: ._embeddingMultiplier)
            _attentionMultiplier = try c.decodeIfPresent(Float.self, forKey: ._attentionMultiplier)
            _residualMultiplier = try c.decodeIfPresent(Float.self, forKey: ._residualMultiplier)
            _logitsScaling = try c.decodeIfPresent(Float.self, forKey: ._logitsScaling)
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let imageSize: Int
        public let patchSize: Int
        private let _channels: Int?
        public var channels: Int {
            _channels ?? 3
        }

        private let _layerNormEps: Float?
        public var layerNormEps: Float {
            _layerNormEps ?? 1e-6
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case _channels = "num_channels"
            case _layerNormEps = "layer_norm_eps"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType) ?? "siglip_vision_model"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 27
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 384
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
            _channels = try c.decodeIfPresent(Int.self, forKey: ._channels)
            _layerNormEps = try c.decodeIfPresent(Float.self, forKey: ._layerNormEps)
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int {
        _imageTokenIndex ?? 100_352
    }

    private let _visionFeatureSelectStrategy: String?
    public var visionFeatureSelectStrategy: String {
        _visionFeatureSelectStrategy ?? "full"
    }

    private let _imageGridPinpoints: [[Int]]?
    public var imageGridPinpoints: [[Int]] {
        _imageGridPinpoints ?? GraniteVisionConfiguration.defaultImageGridPinpoints
    }

    private let _downsampleRate: String?
    public var downsampleRate: String {
        _downsampleRate ?? "4/8"
    }

    private let _deepstackLayerMap: [[Int]]?
    public var deepstackLayerMap: [[Int]] {
        _deepstackLayerMap ?? [[-19, 9], [-13, 6], [-7, 3], [-1, 0]]
    }

    private let _useSpatialSampling: Bool?
    public var useSpatialSampling: Bool {
        _useSpatialSampling ?? true
    }

    private let _spatialVisionLayer: Int?
    public var spatialVisionLayer: Int {
        _spatialVisionLayer ?? -1
    }

    private let _spatialTargetLayers: [Int]?
    public var spatialTargetLayers: [Int] {
        _spatialTargetLayers ?? [12, 15, 18, 21]
    }

    private let _useImageNewlineParameter: Bool?
    public var useImageNewlineParameter: Bool {
        _useImageNewlineParameter ?? true
    }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _imageTokenIndex = "image_token_index"
        case _visionFeatureSelectStrategy = "vision_feature_select_strategy"
        case _imageGridPinpoints = "image_grid_pinpoints"
        case _downsampleRate = "downsample_rate"
        case _deepstackLayerMap = "deepstack_layer_map"
        case _useSpatialSampling = "use_spatial_sampling"
        case _spatialVisionLayer = "spatial_vision_layer"
        case _spatialTargetLayers = "spatial_target_layers"
        case _useImageNewlineParameter = "use_image_newline_parameter"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration =
            try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? TextConfiguration.defaultValue()
        visionConfiguration =
            try c.decodeIfPresent(
                VisionConfiguration.self, forKey: .visionConfiguration,
            ) ?? VisionConfiguration.defaultValue()
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "granite4_vision"
        _imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: ._imageTokenIndex)
        _visionFeatureSelectStrategy = try c.decodeIfPresent(
            String.self, forKey: ._visionFeatureSelectStrategy,
        )
        _imageGridPinpoints = try c.decodeIfPresent([[Int]].self, forKey: ._imageGridPinpoints)
        _downsampleRate = try c.decodeIfPresent(String.self, forKey: ._downsampleRate)
        _deepstackLayerMap = try c.decodeIfPresent([[Int]].self, forKey: ._deepstackLayerMap)
        _useSpatialSampling = try c.decodeIfPresent(Bool.self, forKey: ._useSpatialSampling)
        _spatialVisionLayer = try c.decodeIfPresent(Int.self, forKey: ._spatialVisionLayer)
        _spatialTargetLayers = try c.decodeIfPresent([Int].self, forKey: ._spatialTargetLayers)
        _useImageNewlineParameter = try c.decodeIfPresent(
            Bool.self, forKey: ._useImageNewlineParameter,
        )
    }
}

extension Granite4VisionConfiguration.TextConfiguration {
    fileprivate static func defaultValue() -> Self {
        try! JSONDecoder().decode(Self.self, from: "{}".data(using: .utf8)!)
    }
}

extension Granite4VisionConfiguration.VisionConfiguration {
    fileprivate static func defaultValue() -> Self {
        try! JSONDecoder().decode(Self.self, from: "{}".data(using: .utf8)!)
    }
}

// MARK: - Processor

public struct Granite4VisionProcessorConfiguration: Decodable, Sendable {
    public struct Size: Decodable, Sendable {
        public let height: Int?
        public let width: Int?
        public let shortestEdge: Int?
        public var heightValue: Int {
            height ?? shortestEdge ?? 384
        }

        public var widthValue: Int {
            width ?? shortestEdge ?? 384
        }

        enum CodingKeys: String, CodingKey {
            case height
            case width
            case shortestEdge = "shortest_edge"
        }
    }

    private struct NestedImageProcessor: Decodable {
        let cropSize: Size?
        let size: Size?
        let imageMean: [CGFloat]?
        let imageStd: [CGFloat]?
        let imageGridPinpoints: [[Int]]?

        enum CodingKeys: String, CodingKey {
            case cropSize = "crop_size"
            case size
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case imageGridPinpoints = "image_grid_pinpoints"
        }
    }

    public let processorClass: String?
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let size: Size
    public let cropSize: Size
    private let _patchSize: Int?
    public var patchSize: Int {
        _patchSize ?? 16
    }

    private let _visionFeatureSelectStrategy: String?
    public var visionFeatureSelectStrategy: String {
        _visionFeatureSelectStrategy ?? "full"
    }

    private let _imageToken: String?
    public var imageToken: String {
        _imageToken ?? "<image>"
    }

    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int {
        _imageTokenIndex ?? 100_352
    }

    private let _numAdditionalImageTokens: Int?
    public var numAdditionalImageTokens: Int {
        _numAdditionalImageTokens ?? 0
    }

    private let _imageGridPinpoints: [[Int]]?
    public var imageGridPinpoints: [[Int]] {
        _imageGridPinpoints ?? GraniteVisionConfiguration.defaultImageGridPinpoints
    }

    private let _downsampleRate: String?
    public var downsampleRate: String {
        _downsampleRate ?? "2/4"
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageProcessor = "image_processor"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case size
        case cropSize = "crop_size"
        case _patchSize = "patch_size"
        case _visionFeatureSelectStrategy = "vision_feature_select_strategy"
        case _imageToken = "image_token"
        case _imageTokenIndex = "image_token_index"
        case _numAdditionalImageTokens = "num_additional_image_tokens"
        case _imageGridPinpoints = "image_grid_pinpoints"
        case _downsampleRate = "downsample_rate"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let nested = try c.decodeIfPresent(NestedImageProcessor.self, forKey: .imageProcessor)
        processorClass = try c.decodeIfPresent(String.self, forKey: .processorClass)
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageMean)
            ?? nested?.imageMean ?? [0.5, 0.5, 0.5]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageStd)
            ?? nested?.imageStd ?? [0.5, 0.5, 0.5]
        let defaultSize = Size(height: 384, width: 384, shortestEdge: nil)
        size = try c.decodeIfPresent(Size.self, forKey: .size) ?? nested?.size ?? defaultSize
        cropSize =
            try c.decodeIfPresent(Size.self, forKey: .cropSize)
            ?? nested?.cropSize ?? size
        _patchSize = try c.decodeIfPresent(Int.self, forKey: ._patchSize)
        _visionFeatureSelectStrategy = try c.decodeIfPresent(
            String.self, forKey: ._visionFeatureSelectStrategy,
        )
        _imageToken = try c.decodeIfPresent(String.self, forKey: ._imageToken)
        _imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: ._imageTokenIndex)
        _numAdditionalImageTokens = try c.decodeIfPresent(
            Int.self, forKey: ._numAdditionalImageTokens,
        )
        _imageGridPinpoints =
            try c.decodeIfPresent([[Int]].self, forKey: ._imageGridPinpoints)
            ?? nested?.imageGridPinpoints
        _downsampleRate = try c.decodeIfPresent(String.self, forKey: ._downsampleRate)
    }
}

extension Granite4VisionProcessorConfiguration.Size {
    fileprivate init(height: Int, width: Int, shortestEdge: Int?) {
        self.height = height
        self.width = width
        self.shortestEdge = shortestEdge
    }
}

public struct Granite4VisionProcessor: UserInputProcessor {
    private let config: Granite4VisionProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageTokenIndex: Int

    public init(_ config: Granite4VisionProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        imageTokenIndex = config.imageTokenIndex
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        let messages = LLaVAMessageGenerator().generate(from: input)

        guard !images.isEmpty else {
            let promptTokens = try chatTemplateTokens(
                messages: messages,
                tools: input.tools,
                additionalContext: input.additionalContext,
            )
            let tokensArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            return LMInput(text: .init(tokens: tokensArray, mask: ones(like: tokensArray)))
        }
        guard images.count == 1 else {
            throw VLMError.singleImageAllowed
        }

        let processed = try preprocess(image: images[0].asCIImage(), processing: input.processing)
        let promptTokens = try chatTemplateTokens(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext,
        )
        let expandedTokens = expandImageToken(in: promptTokens, count: processed.imageTokenCount)
        let tokensArray = MLXArray(expandedTokens).expandedDimensions(axis: 0)
        let imageSize = MLXArray(
            [Int32(processed.originalHeight), Int32(processed.originalWidth)],
            [1, 2],
        )
        return LMInput(
            text: .init(tokens: tokensArray, mask: ones(like: tokensArray).asType(.int8)),
            image: .init(pixels: processed.pixels),
            video: .init(pixels: imageSize),
        )
    }

    private func chatTemplateTokens(
        messages: [[String: any Sendable]],
        tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?,
    )
        throws -> [Int]
    {
        do {
            return try tokenizer.applyChatTemplate(
                messages: messages,
                tools: tools,
                additionalContext: additionalContext,
            )
        } catch {
            return fallbackChatTemplateTokens(messages: messages)
        }
    }

    private func fallbackChatTemplateTokens(messages: [[String: any Sendable]]) -> [Int] {
        let defaultSystemMessage =
            "You are a helpful assistant. Please ensure responses are professional, accurate, and safe."
        var rendered = ""
        var startIndex = 0
        if let first = messages.first, Self.role(from: first) == "system" {
            rendered += "<|start_of_role|>system<|end_of_role|>"
            rendered += Self.content(from: first)
            rendered += "<|end_of_text|>\n"
            startIndex = 1
        } else {
            rendered += "<|start_of_role|>system<|end_of_role|>"
            rendered += defaultSystemMessage
            rendered += "<|end_of_text|>\n"
        }

        for message in messages.dropFirst(startIndex) {
            let role = Self.role(from: message)
            rendered += "<|start_of_role|>\(role)<|end_of_role|>"
            rendered += Self.content(from: message)
            rendered += "<|end_of_text|>\n"
        }
        rendered += "<|start_of_role|>assistant<|end_of_role|>"

        return rendered.components(separatedBy: config.imageToken)
            .enumerated()
            .flatMap { index, piece -> [Int] in
                var tokens = tokenizer.encode(text: piece, addSpecialTokens: false)
                if index < rendered.components(separatedBy: config.imageToken).count - 1 {
                    tokens.append(imageTokenIndex)
                }
                return tokens
            }
    }

    private static func role(from message: [String: any Sendable]) -> String {
        message["role"] as? String ?? "user"
    }

    private static func content(from message: [String: any Sendable]) -> String {
        guard let content = message["content"] else {
            return ""
        }
        if let text = content as? String {
            return text
        }
        if let chunks = content as? [[String: String]] {
            return chunks.map { chunk in
                switch chunk["type"] {
                case "image":
                    "<image>\n"
                case "text":
                    chunk["text"] ?? ""
                default:
                    ""
                }
            }.joined()
        }
        if let chunks = content as? [[String: any Sendable]] {
            return chunks.map { chunk in
                switch chunk["type"] as? String {
                case "image":
                    "<image>\n"
                case "text":
                    chunk["text"] as? String ?? ""
                default:
                    ""
                }
            }.joined()
        }
        return "\(content)"
    }

    private struct ProcessedImage {
        let pixels: MLXArray
        let imageTokenCount: Int
        let originalHeight: Int
        let originalWidth: Int
    }

    private func expandImageToken(in promptTokens: [Int], count: Int) -> [Int] {
        let repeatedTokens = Array(repeating: imageTokenIndex, count: max(count, 1))
        if let index = promptTokens.firstIndex(of: imageTokenIndex) {
            var tokens = promptTokens
            tokens.replaceSubrange(index ... index, with: repeatedTokens)
            return tokens
        }
        let decoded = tokenizer.decode(tokenIds: promptTokens, skipSpecialTokens: false)
        let pieces = decoded.components(separatedBy: config.imageToken)
        if pieces.count > 1 {
            return pieces.enumerated().flatMap { index, piece in
                var tokens = tokenizer.encode(text: piece, addSpecialTokens: false)
                if index < pieces.count - 1 {
                    tokens.append(contentsOf: repeatedTokens)
                }
                return tokens
            }
        }
        return repeatedTokens + promptTokens
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?) throws
        -> ProcessedImage
    {
        let canonical = MediaProcessing.zeroOrigin(
            MediaProcessing.apply(image, processing: processing))
        let originalHeight = max(Int(canonical.extent.height.rounded()), 1)
        let originalWidth = max(Int(canonical.extent.width.rounded()), 1)
        let cropHeight = config.cropSize.heightValue
        let cropWidth = config.cropSize.widthValue
        guard cropHeight == cropWidth else {
            throw VLMError.processing("Granite 4 Vision currently expects square image crops.")
        }

        let base = Self.resizeImage(
            image: canonical,
            targetHeight: cropHeight,
            targetWidth: cropWidth,
            mean: config.imageMean,
            std: config.imageStd,
        )
        let bestResolution = Self.selectBestResolution(
            height: originalHeight,
            width: originalWidth,
            pinpoints: config.imageGridPinpoints,
        )
        let resized = Self.resizePreservingAspect(
            image: canonical,
            targetHeight: bestResolution.height,
            targetWidth: bestResolution.width,
        )
        let padded = Self.pad(
            resized.pixels,
            width: resized.width,
            height: resized.height,
            targetWidth: bestResolution.width,
            targetHeight: bestResolution.height,
            value: 0.5,
        )

        var crops = [base.pixels]
        let cropRows = max(bestResolution.height / cropHeight, 1)
        let cropCols = max(bestResolution.width / cropWidth, 1)
        for row in 0 ..< cropRows {
            for col in 0 ..< cropCols {
                let rawCrop = MediaProcessing.cropPixels(
                    padded,
                    sourceWidth: bestResolution.width,
                    x: col * cropWidth,
                    y: row * cropHeight,
                    width: cropWidth,
                    height: cropHeight,
                    channels: 3,
                )
                crops.append(
                    MediaProcessing.normalizeCHW(
                        rawCrop,
                        width: cropWidth,
                        height: cropHeight,
                        mean: config.imageMean,
                        std: config.imageStd,
                    ),
                )
            }
        }

        let pixels = MLXArray(crops.flatMap(\.self), [crops.count, 3, cropHeight, cropWidth])
        let imageTokenCount = Self.imageTokenCount(
            originalHeight: originalHeight,
            originalWidth: originalWidth,
            imageHeight: cropHeight,
            imageWidth: cropWidth,
            patchSize: config.patchSize,
            downsampleRate: config.downsampleRate,
            pinpoints: config.imageGridPinpoints,
            numAdditionalImageTokens: config.numAdditionalImageTokens,
            visionFeatureSelectStrategy: config.visionFeatureSelectStrategy,
        )
        return ProcessedImage(
            pixels: pixels,
            imageTokenCount: imageTokenCount,
            originalHeight: originalHeight,
            originalWidth: originalWidth,
        )
    }

    private struct ResizedImage {
        let pixels: [Float]
        let width: Int
        let height: Int
    }

    private static func resizeImage(
        image: CIImage,
        targetHeight: Int,
        targetWidth: Int,
        mean: [CGFloat],
        std: [CGFloat],
    )
        -> ResizedImage
    {
        let resized = MediaProcessing.zeroOrigin(image).resampled(
            to: CGSize(width: targetWidth, height: targetHeight),
            method: .bicubic,
        )
        let raw = MediaProcessing.renderRGB8(resized, width: targetWidth, height: targetHeight)
        return ResizedImage(
            pixels: MediaProcessing.normalizeCHW(
                raw, width: targetWidth, height: targetHeight, mean: mean, std: std),
            width: targetWidth,
            height: targetHeight,
        )
    }

    private static func resizePreservingAspect(
        image: CIImage,
        targetHeight: Int,
        targetWidth: Int,
    )
        -> ResizedImage
    {
        let originalHeight = max(Int(image.extent.height.rounded()), 1)
        let originalWidth = max(Int(image.extent.width.rounded()), 1)
        let scale = min(
            Float(targetWidth) / Float(originalWidth),
            Float(targetHeight) / Float(originalHeight),
        )
        let resizedWidth = max(Int((Float(originalWidth) * scale).rounded()), 1)
        let resizedHeight = max(Int((Float(originalHeight) * scale).rounded()), 1)
        let resized = MediaProcessing.zeroOrigin(image).resampled(
            to: CGSize(width: resizedWidth, height: resizedHeight),
            method: .bicubic,
        )
        return ResizedImage(
            pixels: MediaProcessing.renderRGB8(resized, width: resizedWidth, height: resizedHeight),
            width: resizedWidth,
            height: resizedHeight,
        )
    }

    fileprivate static func imageTokenCount(
        originalHeight: Int,
        originalWidth: Int,
        imageHeight: Int,
        imageWidth: Int,
        patchSize: Int,
        downsampleRate: String,
        pinpoints: [[Int]],
        numAdditionalImageTokens: Int,
        visionFeatureSelectStrategy: String,
    )
        -> Int
    {
        let best = selectBestResolution(
            height: originalHeight,
            width: originalWidth,
            pinpoints: pinpoints,
        )
        let scaleHeight = best.height / imageHeight
        let scaleWidth = best.width / imageWidth
        let rate = Fraction(downsampleRate)
        let patchesHeight = (imageHeight / patchSize) * rate.numerator / rate.denominator
        let patchesWidth = (imageWidth / patchSize) * rate.numerator / rate.denominator
        let (unpaddedFeatures, newlineFeatures) = unpaddedFeatureCounts(
            originalHeight: originalHeight,
            originalWidth: originalWidth,
            patchesHeight: patchesHeight,
            patchesWidth: patchesWidth,
            scaleHeight: scaleHeight,
            scaleWidth: scaleWidth,
        )
        var total = unpaddedFeatures + newlineFeatures + patchesHeight * patchesWidth
        total += numAdditionalImageTokens
        if visionFeatureSelectStrategy == "default" {
            total -= 1
        }
        return total
    }

    fileprivate static func selectBestResolution(
        height: Int,
        width: Int,
        pinpoints: [[Int]],
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
                Float(candidateWidth) / Float(width),
                Float(candidateHeight) / Float(height),
            )
            let downscaledWidth = Int(Float(width) * scale)
            let downscaledHeight = Int(Float(height) * scale)
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

    fileprivate static func unpaddedFeatureCounts(
        originalHeight: Int,
        originalWidth: Int,
        patchesHeight: Int,
        patchesWidth: Int,
        scaleHeight: Int,
        scaleWidth: Int,
    )
        -> (Int, Int)
    {
        var currentHeight = patchesHeight * scaleHeight
        var currentWidth = patchesWidth * scaleWidth
        let originalAspectRatio = Float(originalWidth) / Float(originalHeight)
        let currentAspectRatio = Float(currentWidth) / Float(currentHeight)
        if originalAspectRatio > currentAspectRatio {
            let newHeight = Int(
                (Float(originalHeight) * (Float(currentWidth) / Float(originalWidth))).rounded())
            let padding = (currentHeight - newHeight) / 2
            currentHeight -= padding * 2
        } else {
            let newWidth = Int(
                (Float(originalWidth) * (Float(currentHeight) / Float(originalHeight))).rounded())
            let padding = (currentWidth - newWidth) / 2
            currentWidth -= padding * 2
        }
        return (currentHeight * currentWidth, currentHeight)
    }

    private static func pad(
        _ pixels: [Float],
        width: Int,
        height: Int,
        targetWidth: Int,
        targetHeight: Int,
        value: Float,
    )
        -> [Float]
    {
        var output = Array(repeating: value, count: targetWidth * targetHeight * 3)
        let xOffset = max((targetWidth - width) / 2, 0)
        let yOffset = max((targetHeight - height) / 2, 0)
        for row in 0 ..< height {
            for col in 0 ..< width {
                let source = (row * width + col) * 3
                let target = ((yOffset + row) * targetWidth + xOffset + col) * 3
                output[target] = pixels[source]
                output[target + 1] = pixels[source + 1]
                output[target + 2] = pixels[source + 2]
            }
        }
        return output
    }

}

private struct Fraction {
    let numerator: Int
    let denominator: Int

    init(_ value: String) {
        let pieces = value.split(separator: "/").compactMap { Int($0) }
        if pieces.count == 2, pieces[1] != 0 {
            numerator = pieces[0]
            denominator = pieces[1]
        } else {
            numerator = 1
            denominator = 2
        }
    }
}

// MARK: - Language

private enum Granite4VisionLanguage {
    final class Attention: Module {
        let config: Granite4VisionConfiguration.TextConfiguration
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        init(_ config: Granite4VisionConfiguration.TextConfiguration) {
            self.config = config
            headDim = config.hiddenSize / config.attentionHeads
            scale = config.attentionMultiplier
            _qProj.wrappedValue = Linear(
                config.hiddenSize,
                config.attentionHeads * headDim,
                bias: config.attentionBias,
            )
            _kProj.wrappedValue = Linear(
                config.hiddenSize,
                config.kvHeads * headDim,
                bias: config.attentionBias,
            )
            _vProj.wrappedValue = Linear(
                config.hiddenSize,
                config.kvHeads * headDim,
                bias: config.attentionBias,
            )
            _oProj.wrappedValue = Linear(
                config.attentionHeads * headDim,
                config.hiddenSize,
                bias: config.attentionBias,
            )
            _rotaryEmbedding.wrappedValue = RoPE(
                dimensions: headDim,
                traditional: config.ropeTraditional,
                base: config.ropeTheta,
            )
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let (batchSize, sequenceLength) = (x.dim(0), x.dim(1))
            var queries = qProj(x)
            var keys = kProj(x)
            var values = vProj(x)

            queries =
                queries
                .reshaped(batchSize, sequenceLength, config.attentionHeads, headDim)
                .transposed(0, 2, 1, 3)
            keys =
                keys
                .reshaped(batchSize, sequenceLength, config.kvHeads, headDim)
                .transposed(0, 2, 1, 3)
            values =
                values
                .reshaped(batchSize, sequenceLength, config.kvHeads, headDim)
                .transposed(0, 2, 1, 3)

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
            .reshaped(batchSize, sequenceLength, -1)

            return oProj(output)
        }
    }

    final class SharedMLP: Module, UnaryLayer {
        @ModuleInfo(key: "input_linear") var inputLinear: Linear
        @ModuleInfo(key: "output_linear") var outputLinear: Linear
        let intermediateSize: Int

        init(_ config: Granite4VisionConfiguration.TextConfiguration) {
            let intermediate = config.sharedIntermediateSize ?? config.intermediateSize
            intermediateSize = intermediate
            _inputLinear.wrappedValue = Linear(
                config.hiddenSize,
                intermediate * 2,
                bias: config.mlpBias,
            )
            _outputLinear.wrappedValue = Linear(
                intermediate,
                config.hiddenSize,
                bias: config.mlpBias,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let projected = inputLinear(x)
            let gate = projected[0..., 0..., 0 ..< intermediateSize]
            let up = projected[0..., 0..., intermediateSize ..< intermediateSize * 2]
            return outputLinear(silu(gate) * up)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        init(_ config: Granite4VisionConfiguration.TextConfiguration) {
            _gate.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize, bias: config.mlpBias,
            )
            _down.wrappedValue = Linear(
                config.intermediateSize, config.hiddenSize, bias: config.mlpBias,
            )
            _up.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize, bias: config.mlpBias,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo var mlp: MLP?
        @ModuleInfo(key: "shared_mlp") var sharedMLP: SharedMLP?
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
        let residualMultiplier: Float

        init(_ config: Granite4VisionConfiguration.TextConfiguration) {
            _attention.wrappedValue = Attention(config)
            if config.useSharedMLP {
                _sharedMLP.wrappedValue = SharedMLP(config)
            } else {
                _mlp.wrappedValue = MLP(config)
            }
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps,
            )
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps,
            )
            residualMultiplier = config.residualMultiplier
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            var residual = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + residual * residualMultiplier
            if let sharedMLP {
                residual = sharedMLP(postAttentionLayerNorm(h))
            } else if let mlp {
                residual = mlp(postAttentionLayerNorm(h))
            } else {
                residual = MLXArray.zeros(h.shape, dtype: h.dtype)
            }
            return h + residual * residualMultiplier
        }
    }

    final class TextModel: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        let layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm
        let embeddingMultiplier: Float

        init(_ config: Granite4VisionConfiguration.TextConfiguration) {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            layers = (0 ..< config.hiddenLayers).map { _ in DecoderLayer(config) }
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            embeddingMultiplier = config.embeddingMultiplier
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
            deepstackVisualEmbeds: [MLXArray]? = nil,
            deepstackTargetLayers: [Int]? = nil,
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
            h = h * embeddingMultiplier

            let mask = createAttentionMask(h: h, cache: cache?.first)
            for (index, layer) in layers.enumerated() {
                if let deepstackVisualEmbeds, let deepstackTargetLayers {
                    for (featureIndex, targetLayer) in deepstackTargetLayers.enumerated()
                    where targetLayer == index && featureIndex < deepstackVisualEmbeds.count {
                        h = h + deepstackVisualEmbeds[featureIndex]
                    }
                }
                h = layer(h, mask: mask, cache: cache?[index])
            }
            return norm(h)
        }
    }

    final class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: TextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear
        let config: Granite4VisionConfiguration.TextConfiguration
        let kvHeads: [Int]

        init(_ config: Granite4VisionConfiguration.TextConfiguration) {
            self.config = config
            _model.wrappedValue = TextModel(config)
            _lmHead.wrappedValue = Linear(
                config.hiddenSize,
                config.vocabularySize,
                bias: false,
            )
            kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
            deepstackVisualEmbeds: [MLXArray]? = nil,
            deepstackTargetLayers: [Int]? = nil,
        )
            -> LMOutput
        {
            let out = model(
                inputs,
                cache: cache,
                inputEmbedding: inputEmbedding,
                deepstackVisualEmbeds: deepstackVisualEmbeds,
                deepstackTargetLayers: deepstackTargetLayers,
            )
            return LMOutput(logits: lmHead(out) / config.logitsScaling)
        }
    }
}

// MARK: - Vision

private enum Granite4VisionVision {
    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: Granite4VisionConfiguration.VisionConfiguration) {
            heads = config.attentionHeads
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _qProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _kProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _vProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (batchSize, sequenceLength) = (x.dim(0), x.dim(1))
            var queries = qProj(x)
            var keys = kProj(x)
            var values = vProj(x)

            queries = queries.reshaped(batchSize, sequenceLength, heads, headDim)
                .transposed(0, 2, 1, 3)
            keys = keys.reshaped(batchSize, sequenceLength, heads, headDim)
                .transposed(0, 2, 1, 3)
            values = values.reshaped(batchSize, sequenceLength, heads, headDim)
                .transposed(0, 2, 1, 3)

            let output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .none,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, sequenceLength, -1)
            return outProj(output)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let activation = GELU(approximation: .fast)

        init(_ config: Granite4VisionConfiguration.VisionConfiguration) {
            _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
            _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    final class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

        init(_ config: Granite4VisionConfiguration.VisionConfiguration) {
            _attention.wrappedValue = Attention(config)
            _layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps,
            )
            _mlp.wrappedValue = MLP(config)
            _layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var y = attention(layerNorm1(x))
            let out = x + y
            y = mlp(layerNorm2(out))
            return out + y
        }
    }

    final class Encoder: Module {
        let layers: [EncoderLayer]

        init(_ config: Granite4VisionConfiguration.VisionConfiguration) {
            layers = (0 ..< config.hiddenLayers).map { _ in EncoderLayer(config) }
        }
    }

    final class VisionEmbeddings: Module {
        let numPositions: Int

        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        init(_ config: Granite4VisionConfiguration.VisionConfiguration) {
            numPositions =
                (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize)
            _patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.channels,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: true,
            )
            _positionEmbedding.wrappedValue = Embedding(
                embeddingCount: numPositions,
                dimensions: config.hiddenSize,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let embeddings = patchEmbedding(x).flattened(start: 1, end: 2)
            let positionIDs = MLXArray(0 ..< numPositions).expandedDimensions(axis: 0)
            return embeddings + positionEmbedding(positionIDs)
        }
    }

    final class MHA: Module {
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "in_proj") var inProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: Granite4VisionConfiguration.VisionConfiguration) {
            heads = config.attentionHeads
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _inProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * 3, bias: true)
            _outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        }
    }

    final class PoolingHead: Module {
        @ModuleInfo var probe: MLXArray
        @ModuleInfo var attention: MHA
        @ModuleInfo var layernorm: LayerNorm
        @ModuleInfo var mlp: MLP

        init(_ config: Granite4VisionConfiguration.VisionConfiguration) {
            _probe.wrappedValue = MLXArray.ones([1, 1, config.hiddenSize])
            _attention.wrappedValue = MHA(config)
            _layernorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps,
            )
            _mlp.wrappedValue = MLP(config)
        }
    }

    final class VisionModelInner: Module {
        @ModuleInfo var embeddings: VisionEmbeddings
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm
        @ModuleInfo var head: PoolingHead

        init(_ config: Granite4VisionConfiguration.VisionConfiguration) {
            _embeddings.wrappedValue = VisionEmbeddings(config)
            _encoder.wrappedValue = Encoder(config)
            _postLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps,
            )
            _head.wrappedValue = PoolingHead(config)
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, [MLXArray]?,
        ) {
            var x = embeddings(x)
            var encoderStates: [MLXArray]? = outputHiddenStates ? [x] : nil
            for layer in encoder.layers {
                x = layer(x)
                encoderStates?.append(x)
            }
            let poolerOutput = postLayerNorm(x)
            return (poolerOutput, x, encoderStates)
        }
    }

    final class VisionModel: Module {
        @ModuleInfo(key: "vision_model") var visionModel: VisionModelInner

        init(_ config: Granite4VisionConfiguration.VisionConfiguration) {
            _visionModel.wrappedValue = VisionModelInner(config)
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
                if key.contains("patch_embedding.weight"), value.ndim == 4 {
                    let shape = value.shape
                    if shape[1] == 3 {
                        sanitized[key] = value.transposed(0, 2, 3, 1)
                    } else {
                        sanitized[key] = value
                    }
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
    }
}

// MARK: - Downsampling

private enum Granite4VisionDownsampling {
    final class QFormerSelfAttention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo var query: Linear
        @ModuleInfo var key: Linear
        @ModuleInfo var value: Linear

        init(hiddenSize: Int, heads: Int, encoderHiddenSize: Int? = nil) {
            self.heads = heads
            headDim = hiddenSize / heads
            scale = pow(Float(headDim), -0.5)
            _query.wrappedValue = Linear(hiddenSize, hiddenSize)
            _key.wrappedValue = Linear(encoderHiddenSize ?? hiddenSize, hiddenSize)
            _value.wrappedValue = Linear(encoderHiddenSize ?? hiddenSize, hiddenSize)
        }

        func callAsFunction(_ hiddenStates: MLXArray, encoderHiddenStates: MLXArray? = nil)
            -> MLXArray
        {
            let (batchSize, sequenceLength) = (hiddenStates.dim(0), hiddenStates.dim(1))
            let kvInput = encoderHiddenStates ?? hiddenStates
            let sourceLength = kvInput.dim(1)

            var queries = query(hiddenStates)
            var keys = key(kvInput)
            var values = value(kvInput)

            queries = queries.reshaped(batchSize, sequenceLength, heads, headDim)
                .transposed(0, 2, 1, 3)
            keys = keys.reshaped(batchSize, sourceLength, heads, headDim)
                .transposed(0, 2, 1, 3)
            values = values.reshaped(batchSize, sourceLength, heads, headDim)
                .transposed(0, 2, 1, 3)

            return MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .none,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, sequenceLength, -1)
        }
    }

    final class QFormerSelfOutput: Module {
        @ModuleInfo var dense: Linear
        @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

        init(hiddenSize: Int) {
            _dense.wrappedValue = Linear(hiddenSize, hiddenSize)
            _layerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
        }

        func callAsFunction(_ hiddenStates: MLXArray, residual: MLXArray) -> MLXArray {
            layerNorm(dense(hiddenStates) + residual)
        }
    }

    final class QFormerAttention: Module {
        @ModuleInfo var attention: QFormerSelfAttention
        @ModuleInfo var output: QFormerSelfOutput

        init(hiddenSize: Int, heads: Int, encoderHiddenSize: Int? = nil) {
            _attention.wrappedValue = QFormerSelfAttention(
                hiddenSize: hiddenSize,
                heads: heads,
                encoderHiddenSize: encoderHiddenSize,
            )
            _output.wrappedValue = QFormerSelfOutput(hiddenSize: hiddenSize)
        }

        func callAsFunction(_ hiddenStates: MLXArray, encoderHiddenStates: MLXArray? = nil)
            -> MLXArray
        {
            output(
                attention(hiddenStates, encoderHiddenStates: encoderHiddenStates),
                residual: hiddenStates)
        }
    }

    final class QFormerIntermediate: Module {
        @ModuleInfo var dense: Linear

        init(hiddenSize: Int, intermediateSize: Int) {
            _dense.wrappedValue = Linear(hiddenSize, intermediateSize)
        }

        func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            gelu(dense(hiddenStates))
        }
    }

    final class QFormerOutput: Module {
        @ModuleInfo var dense: Linear
        @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm

        init(intermediateSize: Int, hiddenSize: Int) {
            _dense.wrappedValue = Linear(intermediateSize, hiddenSize)
            _layerNorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
        }

        func callAsFunction(_ hiddenStates: MLXArray, residual: MLXArray) -> MLXArray {
            layerNorm(dense(hiddenStates) + residual)
        }
    }

    final class QFormerLayer: Module {
        @ModuleInfo var attention: QFormerAttention
        @ModuleInfo var crossattention: QFormerAttention
        @ModuleInfo(key: "intermediate_query") var intermediateQuery: QFormerIntermediate
        @ModuleInfo(key: "output_query") var outputQuery: QFormerOutput

        init(hiddenSize: Int, heads: Int, intermediateSize: Int, encoderHiddenSize: Int) {
            _attention.wrappedValue = QFormerAttention(hiddenSize: hiddenSize, heads: heads)
            _crossattention.wrappedValue = QFormerAttention(
                hiddenSize: hiddenSize,
                heads: heads,
                encoderHiddenSize: encoderHiddenSize,
            )
            _intermediateQuery.wrappedValue = QFormerIntermediate(
                hiddenSize: hiddenSize,
                intermediateSize: intermediateSize,
            )
            _outputQuery.wrappedValue = QFormerOutput(
                intermediateSize: intermediateSize,
                hiddenSize: hiddenSize,
            )
        }

        func callAsFunction(_ hiddenStates: MLXArray, encoderHiddenStates: MLXArray) -> MLXArray {
            let attentionOutput = attention(hiddenStates)
            let crossOutput = crossattention(
                attentionOutput,
                encoderHiddenStates: encoderHiddenStates,
            )
            return outputQuery(intermediateQuery(crossOutput), residual: crossOutput)
        }
    }

    final class QFormerEncoder: Module {
        let layer: [QFormerLayer]

        init(hiddenSize: Int, heads: Int, intermediateSize: Int, encoderHiddenSize: Int) {
            layer = [
                QFormerLayer(
                    hiddenSize: hiddenSize,
                    heads: heads,
                    intermediateSize: intermediateSize,
                    encoderHiddenSize: encoderHiddenSize,
                )
            ]
        }

        func callAsFunction(_ hiddenStates: MLXArray, encoderHiddenStates: MLXArray) -> MLXArray {
            var hiddenStates = hiddenStates
            for layer in layer {
                hiddenStates = layer(hiddenStates, encoderHiddenStates: encoderHiddenStates)
            }
            return hiddenStates
        }
    }

    final class QFormerModel: Module {
        @ModuleInfo var layernorm: LayerNorm
        @ModuleInfo var encoder: QFormerEncoder

        init(hiddenSize: Int, heads: Int, intermediateSize: Int, encoderHiddenSize: Int) {
            _layernorm.wrappedValue = LayerNorm(dimensions: hiddenSize)
            _encoder.wrappedValue = QFormerEncoder(
                hiddenSize: hiddenSize,
                heads: heads,
                intermediateSize: intermediateSize,
                encoderHiddenSize: encoderHiddenSize,
            )
        }

        func callAsFunction(queryEmbeds: MLXArray, encoderHiddenStates: MLXArray) -> MLXArray {
            encoder(layernorm(queryEmbeds), encoderHiddenStates: encoderHiddenStates)
        }
    }

    final class WindowQFormerDownsampler: Module, UnaryLayer {
        @ModuleInfo var norm: LayerNorm
        @ModuleInfo var query: MLXArray
        @ModuleInfo(key: "image_positions") var imagePositions: MLXArray
        @ModuleInfo var qformer: QFormerModel
        @ModuleInfo(key: "out_linear") var outLinear: Linear

        let imageSide: Int
        let querySide: Int
        let windowSide: Int
        let queryLength: Int
        let spatialOffset: Int?

        init(_ config: Granite4VisionConfiguration, spatialOffset: Int? = nil) {
            let visionHidden = config.visionConfiguration.hiddenSize
            imageSide = config.visionConfiguration.imageSize / config.visionConfiguration.patchSize
            let rate = Fraction(config.downsampleRate)
            querySide = rate.numerator
            windowSide = rate.denominator
            queryLength = querySide * querySide
            self.spatialOffset = spatialOffset

            let std = 1 / sqrt(Float(visionHidden))
            _norm.wrappedValue = LayerNorm(dimensions: visionHidden, eps: 1e-6)
            _query.wrappedValue = MLXRandom.normal([1, queryLength, visionHidden]) * std
            _imagePositions.wrappedValue =
                MLXRandom.normal([1, windowSide * windowSide, visionHidden]) * std
            _qformer.wrappedValue = QFormerModel(
                hiddenSize: visionHidden,
                heads: visionHidden / 64,
                intermediateSize: 3072,
                encoderHiddenSize: visionHidden,
            )
            _outLinear.wrappedValue = Linear(
                visionHidden,
                config.textConfiguration.hiddenSize,
                bias: true,
            )
        }

        func callAsFunction(_ imageFeatures: MLXArray) -> MLXArray {
            let normalized = norm(imageFeatures)
            let windows = window(normalized, side: imageSide, window: windowSide)
            let downsampled = downsample(normalized)
            let windowCount = imageSide / windowSide
            let newSide = windowCount * querySide
            let downsampledWindows = window(downsampled, side: newSide, window: querySide)
            let queryEmbeds = query.asType(downsampledWindows.dtype) + downsampledWindows
            let encoderEmbeds = windows + imagePositions.asType(windows.dtype)
            let outputWindows = qformer(
                queryEmbeds: queryEmbeds,
                encoderHiddenStates: encoderEmbeds,
            )
            return outLinear(unwindow(outputWindows, count: windowCount, window: querySide))
        }

        private func downsample(_ imageFeatures: MLXArray) -> MLXArray {
            if let spatialOffset {
                let offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
                let (offsetH, offsetW) = offsets[spatialOffset]
                let batch = imageFeatures.dim(0)
                let channels = imageFeatures.dim(2)
                let newSide = imageSide / 2
                return
                    imageFeatures
                    .reshaped(batch, imageSide, imageSide, channels)
                    .reshaped(batch, newSide, 2, newSide, 2, channels)[
                        0..., 0..., offsetH, 0..., offsetW, 0...,
                    ]
                    .reshaped(batch, newSide * newSide, channels)
            }

            let batch = imageFeatures.dim(0)
            let channels = imageFeatures.dim(2)
            let newSide = imageSide * querySide / windowSide
            let ratio = imageSide / newSide
            return
                imageFeatures
                .reshaped(batch, imageSide, imageSide, channels)
                .reshaped(batch, newSide, ratio, newSide, ratio, channels)
                .mean(axis: 4)
                .mean(axis: 2)
                .reshaped(batch, newSide * newSide, channels)
        }

        private func window(_ x: MLXArray, side: Int, window: Int) -> MLXArray {
            let batch = x.dim(0)
            let channels = x.dim(2)
            let count = side / window
            return
                x
                .reshaped(batch, side, side, channels)
                .reshaped(batch, count, window, count, window, channels)
                .transposed(0, 1, 3, 2, 4, 5)
                .reshaped(batch * count * count, window * window, channels)
        }

        private func unwindow(_ x: MLXArray, count: Int, window: Int) -> MLXArray {
            let batchWindow = x.dim(0)
            let channels = x.dim(2)
            let batch = batchWindow / (count * count)
            let side = count * window
            return
                x
                .reshaped(batch, count, count, window, window, channels)
                .transposed(0, 1, 3, 2, 4, 5)
                .reshaped(batch, side * side, channels)
        }
    }
}

// MARK: - Model

public final class Granite4Vision: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionModel: Granite4VisionVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel:
        Granite4VisionLanguage.LanguageModel
    private let layerwise_projectors: [Granite4VisionDownsampling.WindowQFormerDownsampler]
    private let spatial_projectors: [Granite4VisionDownsampling.WindowQFormerDownsampler]
    @ModuleInfo(key: "image_newline") private var imageNewline: MLXArray

    public let config: Granite4VisionConfiguration
    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public init(_ config: Granite4VisionConfiguration) {
        self.config = config
        _visionModel.wrappedValue = Granite4VisionVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = Granite4VisionLanguage.LanguageModel(
            config.textConfiguration,
        )
        layerwise_projectors = config.deepstackLayerMap.map { _ in
            Granite4VisionDownsampling.WindowQFormerDownsampler(config)
        }
        spatial_projectors =
            config.useSpatialSampling
            ? (0 ..< 4).map {
                Granite4VisionDownsampling.WindowQFormerDownsampler(config, spatialOffset: $0)
            }
            : []
        let std = 1 / sqrt(Float(config.textConfiguration.hiddenSize))
        _imageNewline.wrappedValue =
            MLXRandom.normal([config.textConfiguration.hiddenSize]) * std
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    private struct EmbeddingFeatures {
        let inputsEmbeds: MLXArray
        let visualEmbeds: [MLXArray]?
        let targetLayers: [Int]?
    }

    private func inputEmbeddings(
        inputIDs: MLXArray,
        pixelValues: MLXArray?,
        imageSizes: MLXArray?,
    )
        -> EmbeddingFeatures
    {
        var inputsEmbeds = languageModel.model.embedTokens(inputIDs)
        guard let pixelValues else {
            return EmbeddingFeatures(
                inputsEmbeds: inputsEmbeds, visualEmbeds: nil, targetLayers: nil)
        }

        let visionMask = inputIDs .== config.imageTokenIndex
        let expandedMask = broadcast(visionMask[0..., 0..., .newAxis], to: inputsEmbeds.shape)
        inputsEmbeds = which(
            expandedMask,
            MLXArray.zeros(inputsEmbeds.shape, dtype: inputsEmbeds.dtype),
            inputsEmbeds,
        )

        let (_, _, hiddenStates) = visionModel(
            pixelValues.transposed(0, 2, 3, 1),
            outputHiddenStates: true,
        )
        guard let hiddenStates else {
            return EmbeddingFeatures(
                inputsEmbeds: inputsEmbeds, visualEmbeds: nil, targetLayers: nil)
        }

        var featureSets: [MLXArray] = []
        var targetLayers: [Int] = []

        for (projectorIndex, map) in config.deepstackLayerMap.enumerated()
        where map.count >= 2 && projectorIndex < layerwise_projectors.count {
            let visionLayer = map[0]
            let llmLayer = map[1]
            let layerIndex = visionLayer >= 0 ? visionLayer : hiddenStates.count + visionLayer
            var selected = hiddenStates[layerIndex]
            if config.visionFeatureSelectStrategy == "default" {
                selected = selected[0..., 1..., 0...]
            }
            let projected = layerwise_projectors[projectorIndex](selected)
            let packed = packImageFeatures(projected, imageSizes: imageSizes).asType(
                inputsEmbeds.dtype)
            featureSets.append(
                fullSequenceFeature(packed, inputIDs: inputIDs, dtype: inputsEmbeds.dtype))
            targetLayers.append(llmLayer)
        }

        if config.useSpatialSampling {
            let layerIndex =
                config.spatialVisionLayer >= 0
                ? config.spatialVisionLayer
                : hiddenStates.count + config.spatialVisionLayer
            var selected = hiddenStates[layerIndex]
            if config.visionFeatureSelectStrategy == "default" {
                selected = selected[0..., 1..., 0...]
            }
            for (projectorIndex, llmLayer) in config.spatialTargetLayers.enumerated()
            where projectorIndex < spatial_projectors.count {
                let projected = spatial_projectors[projectorIndex](selected)
                let packed = packImageFeatures(projected, imageSizes: imageSizes).asType(
                    inputsEmbeds.dtype)
                featureSets.append(
                    fullSequenceFeature(packed, inputIDs: inputIDs, dtype: inputsEmbeds.dtype))
                targetLayers.append(llmLayer)
            }
        }

        return EmbeddingFeatures(
            inputsEmbeds: inputsEmbeds,
            visualEmbeds: featureSets.isEmpty ? nil : featureSets,
            targetLayers: targetLayers.isEmpty ? nil : targetLayers,
        )
    }

    private func fullSequenceFeature(
        _ features: MLXArray,
        inputIDs: MLXArray,
        dtype: DType,
    )
        -> MLXArray
    {
        let ids = inputIDs[0].asArray(Int.self)
        let positions = ids.indices.filter { ids[$0] == config.imageTokenIndex }
        precondition(
            positions.count == features.dim(0),
            "Granite4Vision image feature/token count mismatch",
        )

        var pieces: [MLXArray] = []
        var cursor = 0
        var featureIndex = 0
        let hidden = config.textConfiguration.hiddenSize
        for position in positions {
            if position > cursor {
                pieces.append(MLXArray.zeros([position - cursor, hidden], dtype: dtype))
            }
            pieces.append(features[featureIndex].expandedDimensions(axis: 0))
            cursor = position + 1
            featureIndex += 1
        }
        if cursor < ids.count {
            pieces.append(MLXArray.zeros([ids.count - cursor, hidden], dtype: dtype))
        }
        return concatenated(pieces, axis: 0).expandedDimensions(axis: 0)
    }

    private func packImageFeatures(_ imageFeatures: MLXArray, imageSizes: MLXArray?) -> MLXArray {
        guard imageFeatures.dim(0) > 1 else {
            var feature = imageFeatures[0]
            if config.useImageNewlineParameter {
                feature = concatenated([feature, imageNewline[.newAxis, 0...]], axis: 0)
            }
            return feature
        }

        let originalSize = imageSizes?.asArray(Int32.self).map(Int.init)
        let originalHeight = originalSize?.first ?? config.visionConfiguration.imageSize
        let originalWidth = originalSize?.dropFirst().first ?? config.visionConfiguration.imageSize
        let best = Granite4VisionProcessor.selectBestResolution(
            height: originalHeight,
            width: originalWidth,
            pinpoints: config.imageGridPinpoints,
        )
        let cropSize = config.visionConfiguration.imageSize
        let numPatchHeight = max(best.height / cropSize, 1)
        let numPatchWidth = max(best.width / cropSize, 1)
        let rate = Fraction(config.downsampleRate)
        let patchSide =
            (config.visionConfiguration.imageSize / config.visionConfiguration.patchSize)
            * rate.numerator / rate.denominator
        let hidden = config.textConfiguration.hiddenSize

        let base = imageFeatures[0]
        var patches = imageFeatures[1..., 0..., 0...]
        patches = patches.reshaped(numPatchHeight, numPatchWidth, patchSide, patchSide, hidden)
        patches =
            patches
            .transposed(4, 0, 2, 1, 3)
            .reshaped(hidden, numPatchHeight * patchSide, numPatchWidth * patchSide)
        patches = unpadImageFeatures(
            patches,
            originalHeight: originalHeight,
            originalWidth: originalWidth,
        )

        if config.useImageNewlineParameter {
            let newline = broadcast(
                imageNewline.asType(patches.dtype)[0..., .newAxis, .newAxis],
                to: [hidden, patches.dim(1), 1],
            )
            patches = concatenated([patches, newline], axis: 2)
        }

        let flattened = patches.transposed(1, 2, 0).reshaped(-1, hidden)
        return concatenated([base, flattened], axis: 0)
    }

    private func unpadImageFeatures(
        _ features: MLXArray,
        originalHeight: Int,
        originalWidth: Int,
    )
        -> MLXArray
    {
        let currentHeight = features.dim(1)
        let currentWidth = features.dim(2)
        let originalAspectRatio = Float(originalWidth) / Float(originalHeight)
        let currentAspectRatio = Float(currentWidth) / Float(currentHeight)
        if originalAspectRatio > currentAspectRatio {
            let newHeight = Int(
                Float(originalHeight) * (Float(currentWidth) / Float(originalWidth)))
            let padding = max((currentHeight - newHeight) / 2, 0)
            return features[0..., padding ..< (currentHeight - padding), 0...]
        } else {
            let newWidth = Int(
                Float(originalWidth) * (Float(currentHeight) / Float(originalHeight)))
            let padding = max((currentWidth - newWidth) / 2, 0)
            return features[0..., 0..., padding ..< (currentWidth - padding)]
        }
    }

    private func sliceVisualFeatures(
        _ features: [MLXArray]?,
        range: Range<Int>,
    )
        -> [MLXArray]?
    {
        features?.map { $0[0..., range, 0...] }
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let features = inputEmbeddings(
            inputIDs: input.text.tokens,
            pixelValues: input.image?.pixels,
            imageSizes: input.video?.pixels,
        )

        let totalPositions = features.inputsEmbeds.dim(1)
        let processed = try prefill.forEachChunk(total: totalPositions) { range in
            _ = languageModel(
                nil,
                cache: cache,
                inputEmbedding: features.inputsEmbeds[0..., range, 0...],
                deepstackVisualEmbeds: sliceVisualFeatures(features.visualEmbeds, range: range),
                deepstackTargetLayers: features.targetLayers,
            )
            asyncEval(cache)
        }
        eval(cache)

        let finalRange = processed ..< totalPositions
        let result = languageModel(
            nil,
            cache: cache,
            inputEmbedding: features.inputsEmbeds[0..., finalRange, 0...],
            deepstackVisualEmbeds: sliceVisualFeatures(features.visualEmbeds, range: finalRange),
            deepstackTargetLayers: features.targetLayers,
        )
        prefill.progress?(totalPositions, totalPositions)
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        var loraA: [String: MLXArray] = [:]
        var loraB: [String: MLXArray] = [:]

        for (key, value) in weights {
            if key.contains("lora_A.weight") {
                var baseKey = key.replacingOccurrences(of: "lora_A.weight", with: "weight")
                if baseKey.hasPrefix("base_model.model.") {
                    baseKey.removeFirst("base_model.model.".count)
                }
                loraA[baseKey] = value
                continue
            }
            if key.contains("lora_B.weight") {
                var baseKey = key.replacingOccurrences(of: "lora_B.weight", with: "weight")
                if baseKey.hasPrefix("base_model.model.") {
                    baseKey.removeFirst("base_model.model.".count)
                }
                loraB[baseKey] = value
                continue
            }
        }

        for (key, value) in weights {
            if key.contains("lora_A") || key.contains("lora_B") || key.hasPrefix("base_model.") {
                continue
            }
            if key.contains("self_attn.rotary_emb.inv_freq") {
                continue
            }

            var newKey = key
            if newKey.hasPrefix("model.") {
                let suffix = String(newKey.dropFirst("model.".count))
                if suffix.hasPrefix("language_model.") {
                    newKey = "language_model.model." + suffix.dropFirst("language_model.".count)
                } else {
                    newKey = suffix
                }
            }
            if newKey == "lm_head.weight" {
                newKey = "language_model.lm_head.weight"
            }

            if let a = loraA[newKey], let b = loraB[newKey] {
                sanitized[newKey] = value + matmul(b, a)
            } else {
                sanitized[newKey] = value
            }
        }

        if sanitized["language_model.lm_head.weight"] == nil,
            let embedding = sanitized["language_model.model.embed_tokens.weight"]
        {
            sanitized["language_model.lm_head.weight"] = embedding
        }

        return visionModel.sanitize(weights: sanitized)
    }
}
