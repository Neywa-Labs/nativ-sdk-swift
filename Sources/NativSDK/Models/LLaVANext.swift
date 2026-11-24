import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct LLaVANextConfiguration: Decodable, Sendable {
    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        private let _hiddenSize: Int?
        public var hiddenSize: Int {
            _hiddenSize ?? 4096
        }

        private let _hiddenLayers: Int?
        public var hiddenLayers: Int {
            _hiddenLayers ?? 32
        }

        public let intermediateSize: Int
        private let _attentionHeads: Int?
        public var attentionHeads: Int {
            _attentionHeads ?? 32
        }

        private let _kvHeads: Int?
        public var kvHeads: Int {
            _kvHeads ?? attentionHeads
        }

        public let rmsNormEps: Float
        public let vocabularySize: Int
        private let _ropeTheta: Float?
        public var ropeTheta: Float {
            _ropeTheta ?? 1_000_000
        }

        private let _ropeTraditional: Bool?
        public var ropeTraditional: Bool {
            _ropeTraditional ?? false
        }

        public let ropeScaling: [String: StringOrNumber]?
        private let _maxPositionEmbeddings: Int?
        public var maxPositionEmbeddings: Int {
            _maxPositionEmbeddings ?? 4096
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case _hiddenSize = "hidden_size"
            case _hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case _attentionHeads = "num_attention_heads"
            case _kvHeads = "num_key_value_heads"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case _ropeTheta = "rope_theta"
            case _ropeTraditional = "rope_traditional"
            case ropeScaling = "rope_scaling"
            case _maxPositionEmbeddings = "max_position_embeddings"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "mistral"
            _hiddenSize = try c.decodeIfPresent(Int.self, forKey: ._hiddenSize)
            _hiddenLayers = try c.decodeIfPresent(Int.self, forKey: ._hiddenLayers)
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
            _attentionHeads = try c.decodeIfPresent(Int.self, forKey: ._attentionHeads)
            _kvHeads = try c.decodeIfPresent(Int.self, forKey: ._kvHeads)
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
            vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 32064
            _ropeTheta = try c.decodeIfPresent(Float.self, forKey: ._ropeTheta)
            _ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: ._ropeTraditional)
            ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
            _maxPositionEmbeddings = try c.decodeIfPresent(
                Int.self, forKey: ._maxPositionEmbeddings,
            )
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
            _layerNormEps ?? 1e-5
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
                try c.decodeIfPresent(String.self, forKey: .modelType) ?? "clip_vision_model"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 24
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4096
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 336
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            _channels = try c.decodeIfPresent(Int.self, forKey: ._channels)
            _layerNormEps = try c.decodeIfPresent(Float.self, forKey: ._layerNormEps)
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int {
        _imageTokenIndex ?? 32000
    }

    private let _visionFeatureLayer: Int?
    public var visionFeatureLayer: Int {
        _visionFeatureLayer ?? -2
    }

    private let _visionFeatureSelectStrategy: String?
    public var visionFeatureSelectStrategy: String {
        _visionFeatureSelectStrategy ?? "default"
    }

    private let _imageGridPinpoints: [[Int]]?
    public var imageGridPinpoints: [[Int]] {
        _imageGridPinpoints ?? [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
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
        case _visionFeatureLayer = "vision_feature_layer"
        case _visionFeatureSelectStrategy = "vision_feature_select_strategy"
        case _imageGridPinpoints = "image_grid_pinpoints"
        case _useImageNewlineParameter = "use_image_newline_parameter"
    }
}

// MARK: - Processor

public struct LLaVANextProcessorConfiguration: Decodable, Sendable {
    public struct Size: Decodable, Sendable {
        public let height: Int?
        public let width: Int?
        public let shortestEdge: Int?
        public var heightValue: Int {
            height ?? shortestEdge ?? 336
        }

        public var widthValue: Int {
            width ?? shortestEdge ?? 336
        }

        enum CodingKeys: String, CodingKey {
            case height
            case width
            case shortestEdge = "shortest_edge"
        }
    }

    public let processorClass: String?
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let cropSize: Size
    public let size: Size
    private let _patchSize: Int?
    public var patchSize: Int {
        _patchSize ?? 14
    }

    private let _visionFeatureSelectStrategy: String?
    public var visionFeatureSelectStrategy: String {
        _visionFeatureSelectStrategy ?? "default"
    }

    private let _imageToken: String?
    public var imageToken: String {
        _imageToken ?? "<image>"
    }

    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int {
        _imageTokenIndex ?? 32000
    }

    private let _numAdditionalImageTokens: Int?
    public var numAdditionalImageTokens: Int {
        _numAdditionalImageTokens ?? 1
    }

    private let _imageGridPinpoints: [[Int]]?
    public var imageGridPinpoints: [[Int]] {
        _imageGridPinpoints ?? [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case cropSize = "crop_size"
        case size
        case _patchSize = "patch_size"
        case _visionFeatureSelectStrategy = "vision_feature_select_strategy"
        case _imageToken = "image_token"
        case _imageTokenIndex = "image_token_index"
        case _numAdditionalImageTokens = "num_additional_image_tokens"
        case _imageGridPinpoints = "image_grid_pinpoints"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass = try c.decodeIfPresent(String.self, forKey: .processorClass)
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageMean)
            ?? [0.48145466, 0.4578275, 0.40821073]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageStd)
            ?? [0.26862954, 0.26130258, 0.27577711]
        let defaultSize = Size(height: 336, width: 336, shortestEdge: nil)
        size = try c.decodeIfPresent(Size.self, forKey: .size) ?? defaultSize
        cropSize = try c.decodeIfPresent(Size.self, forKey: .cropSize) ?? size
        _patchSize = try c.decodeIfPresent(Int.self, forKey: ._patchSize)
        _visionFeatureSelectStrategy = try c.decodeIfPresent(
            String.self, forKey: ._visionFeatureSelectStrategy,
        )
        _imageToken = try c.decodeIfPresent(String.self, forKey: ._imageToken)
        _imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: ._imageTokenIndex)
        _numAdditionalImageTokens = try c.decodeIfPresent(
            Int.self, forKey: ._numAdditionalImageTokens,
        )
        _imageGridPinpoints = try c.decodeIfPresent([[Int]].self, forKey: ._imageGridPinpoints)
    }
}

extension LLaVANextProcessorConfiguration.Size {
    fileprivate init(height: Int, width: Int, shortestEdge: Int?) {
        self.height = height
        self.width = width
        self.shortestEdge = shortestEdge
    }
}

public struct LLaVANextProcessor: UserInputProcessor {
    private let config: LLaVANextProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageTokenIndex: Int

    public init(_ config: LLaVANextProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        imageTokenIndex =
            tokenizer.convertTokenToId(config.imageToken)
            ?? config.imageTokenIndex
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        let messages = LLaVAMessageGenerator().generate(from: input)

        guard !images.isEmpty else {
            let promptTokens = try tokenizer.applyChatTemplate(
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
        let promptTokens = try tokenizer.applyChatTemplate(
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
            throw VLMError.processing("LLaVA-NeXT currently expects square image crops.")
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
        let patchesHeight = imageHeight / patchSize
        let patchesWidth = imageWidth / patchSize
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
        var best = pinpoints.first ?? [336, 336]
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

// MARK: - Language

private enum LLaVANextLanguage {
    final class Attention: Module {
        let config: LLaVANextConfiguration.TextConfiguration
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        init(_ config: LLaVANextConfiguration.TextConfiguration) {
            self.config = config
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _qProj.wrappedValue = Linear(
                config.hiddenSize,
                config.attentionHeads * headDim,
                bias: false,
            )
            _kProj.wrappedValue = Linear(
                config.hiddenSize,
                config.kvHeads * headDim,
                bias: false,
            )
            _vProj.wrappedValue = Linear(
                config.hiddenSize,
                config.kvHeads * headDim,
                bias: false,
            )
            _oProj.wrappedValue = Linear(
                config.attentionHeads * headDim,
                config.hiddenSize,
                bias: false,
            )

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

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        init(_ config: LLaVANextConfiguration.TextConfiguration) {
            _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
            _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
            _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: LLaVANextConfiguration.TextConfiguration) {
            _attention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(config)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps,
            )
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps,
            )
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            var residual = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + residual
            residual = mlp(postAttentionLayerNorm(h))
            return h + residual
        }
    }

    final class TextModel: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        let layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm

        init(_ config: LLaVANextConfiguration.TextConfiguration) {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            layers = (0 ..< config.hiddenLayers).map { _ in DecoderLayer(config) }
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
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

            let mask = createAttentionMask(h: h, cache: cache?.first)
            for (index, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[index])
            }
            return norm(h)
        }
    }

    final class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: TextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear
        let config: LLaVANextConfiguration.TextConfiguration
        let kvHeads: [Int]

        init(_ config: LLaVANextConfiguration.TextConfiguration) {
            self.config = config
            _model.wrappedValue = TextModel(config)
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
            kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
        )
            -> LMOutput
        {
            LMOutput(logits: lmHead(model(inputs, cache: cache, inputEmbedding: inputEmbedding)))
        }
    }
}

// MARK: - Vision

private enum LLaVANextVision {
    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: LLaVANextConfiguration.VisionConfiguration) {
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

        init(_ config: LLaVANextConfiguration.VisionConfiguration) {
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

        init(_ config: LLaVANextConfiguration.VisionConfiguration) {
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

        init(_ config: LLaVANextConfiguration.VisionConfiguration) {
            layers = (0 ..< config.hiddenLayers).map { _ in EncoderLayer(config) }
        }
    }

    final class VisionEmbeddings: Module {
        let config: LLaVANextConfiguration.VisionConfiguration
        let numPositions: Int

        @ModuleInfo(key: "class_embedding") var classEmbedding: MLXArray
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        init(_ config: LLaVANextConfiguration.VisionConfiguration) {
            self.config = config
            let numPatches =
                (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize)
            numPositions = numPatches + 1
            _classEmbedding.wrappedValue = MLXArray.zeros([config.hiddenSize])
            _patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.channels,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: false,
            )
            _positionEmbedding.wrappedValue = Embedding(
                embeddingCount: numPositions,
                dimensions: config.hiddenSize,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var embeddings = patchEmbedding(x).flattened(start: 1, end: 2)
            let classEmbeddings = broadcast(
                classEmbedding.asType(embeddings.dtype)[.newAxis, .newAxis, 0...],
                to: [x.dim(0), 1, config.hiddenSize],
            )
            embeddings = concatenated([classEmbeddings, embeddings], axis: 1)
            let positionIDs = MLXArray(0 ..< numPositions).expandedDimensions(axis: 0)
            return embeddings + positionEmbedding(positionIDs)
        }
    }

    final class VisionModelInner: Module {
        @ModuleInfo var embeddings: VisionEmbeddings
        @ModuleInfo(key: "pre_layrnorm") var preLayerNorm: LayerNorm
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        init(_ config: LLaVANextConfiguration.VisionConfiguration) {
            _embeddings.wrappedValue = VisionEmbeddings(config)
            _preLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
            _encoder.wrappedValue = Encoder(config)
            _postLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, [MLXArray]?,
        ) {
            var x = preLayerNorm(embeddings(x))
            var encoderStates: [MLXArray]? = outputHiddenStates ? [x] : nil
            for layer in encoder.layers {
                x = layer(x)
                encoderStates?.append(x)
            }
            let poolerOutput = postLayerNorm(x[0..., 0, 0...])
            return (poolerOutput, x, encoderStates)
        }
    }

    final class VisionModel: Module {
        @ModuleInfo(key: "vision_model") var visionModel: VisionModelInner

        init(_ config: LLaVANextConfiguration.VisionConfiguration) {
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
                } else {
                    sanitized[key] = value
                }
            }
            return sanitized
        }
    }
}

// MARK: - Model

private final class LLaVANextMultiModalProjector: Module, UnaryLayer {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    let activation = GELU()

    init(_ config: LLaVANextConfiguration) {
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
        linear2(activation(linear1(x)))
    }
}

public final class LLaVANext: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionModel: LLaVANextVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: LLaVANextLanguage.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") private var projector: LLaVANextMultiModalProjector
    @ModuleInfo(key: "image_newline") private var imageNewline: MLXArray

    public let config: LLaVANextConfiguration
    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public init(_ config: LLaVANextConfiguration) {
        self.config = config
        _visionModel.wrappedValue = LLaVANextVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = LLaVANextLanguage.LanguageModel(config.textConfiguration)
        _projector.wrappedValue = LLaVANextMultiModalProjector(config)
        let std = 1 / sqrt(Float(config.textConfiguration.hiddenSize))
        _imageNewline.wrappedValue =
            MLXRandom.normal([config.textConfiguration.hiddenSize]) * std
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    private func inputEmbeddings(
        inputIDs: MLXArray,
        pixelValues: MLXArray?,
        imageSizes: MLXArray?,
    )
        -> MLXArray
    {
        guard let pixelValues else {
            return languageModel.model.embedTokens(inputIDs)
        }

        let safeInputIDs = which(
            inputIDs .== config.imageTokenIndex,
            MLXArray(0),
            inputIDs,
        )
        let inputsEmbeds = languageModel.model.embedTokens(safeInputIDs)

        let (_, _, hiddenStates) = visionModel(
            pixelValues.transposed(0, 2, 3, 1),
            outputHiddenStates: true,
        )
        guard let hiddenStates else {
            return inputsEmbeds
        }

        let layerIndex =
            config.visionFeatureLayer >= 0
            ? config.visionFeatureLayer
            : hiddenStates.count + config.visionFeatureLayer
        var imageFeatures = hiddenStates[layerIndex]
        if config.visionFeatureSelectStrategy == "default" {
            imageFeatures = imageFeatures[0..., 1..., 0...]
        }
        imageFeatures = projector(imageFeatures).asType(inputsEmbeds.dtype)
        let packedFeatures = packImageFeatures(imageFeatures, imageSizes: imageSizes)
            .asType(inputsEmbeds.dtype)

        let ids = inputIDs[0].asArray(Int.self)
        let positions = ids.indices.filter { ids[$0] == config.imageTokenIndex }
        guard let first = positions.first, let last = positions.last else {
            return inputsEmbeds
        }

        let before = inputsEmbeds[0, 0 ..< first, 0...]
        let after = inputsEmbeds[0, (last + 1) ..< ids.count, 0...]
        return concatenated([before, packedFeatures, after], axis: 0)
            .expandedDimensions(axis: 0)
    }

    private func packImageFeatures(_ imageFeatures: MLXArray, imageSizes: MLXArray?) -> MLXArray {
        guard imageFeatures.dim(0) > 1 else {
            return imageFeatures[0]
        }

        let originalSize = imageSizes?.asArray(Int32.self).map(Int.init)
        let originalHeight = originalSize?.first ?? config.visionConfiguration.imageSize
        let originalWidth = originalSize?.dropFirst().first ?? config.visionConfiguration.imageSize
        let best = LLaVANextProcessor.selectBestResolution(
            height: originalHeight,
            width: originalWidth,
            pinpoints: config.imageGridPinpoints,
        )
        let cropSize = config.visionConfiguration.imageSize
        let numPatchHeight = max(best.height / cropSize, 1)
        let numPatchWidth = max(best.width / cropSize, 1)
        let patchSide = config.visionConfiguration.imageSize / config.visionConfiguration.patchSize
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
                (Float(originalHeight) * (Float(currentWidth) / Float(originalWidth))).rounded())
            let padding = max((currentHeight - newHeight) / 2, 0)
            return features[0..., padding ..< (currentHeight - padding), 0...]
        } else {
            let newWidth = Int(
                (Float(originalWidth) * (Float(currentHeight) / Float(originalHeight))).rounded())
            let padding = max((currentWidth - newWidth) / 2, 0)
            return features[0..., 0..., padding ..< (currentWidth - padding)]
        }
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let embeddings = inputEmbeddings(
            inputIDs: input.text.tokens,
            pixelValues: input.image?.pixels,
            imageSizes: input.video?.pixels,
        )

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
        let filtered = weights.filter { key, _ in
            !key.contains("self_attn.rotary_emb.inv_freq")
        }
        return visionModel.sanitize(weights: filtered)
    }
}
