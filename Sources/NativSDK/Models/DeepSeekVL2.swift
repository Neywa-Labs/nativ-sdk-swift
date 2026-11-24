import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct DeepSeekVL2Configuration: Decodable, Sendable {
    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let vocabularySize: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let moeIntermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        private let _kvHeads: Int?
        public var kvHeads: Int {
            _kvHeads ?? attentionHeads
        }

        public let nSharedExperts: Int?
        public let nRoutedExperts: Int?
        public let routedScalingFactor: Float
        public let numExpertsPerToken: Int
        public let firstKDenseReplace: Int
        public let moeLayerFrequency: Int
        public let maxPositionEmbeddings: Int
        public let rmsNormEps: Float
        public let ropeTheta: Float
        public let ropeTraditional: Bool
        public let ropeScaling: [String: StringOrNumber]?
        public let attentionBias: Bool
        public let topKMethod: String
        public let scoringFunction: String
        public let nGroup: Int
        public let topKGroup: Int
        public let kvLoraRank: Int?
        public let qLoraRank: Int?
        public let qkRopeHeadDim: Int
        public let vHeadDim: Int
        public let qkNopeHeadDim: Int

        public var usesMLA: Bool {
            kvLoraRank != nil || qLoraRank != nil || qkNopeHeadDim > 0 || qkRopeHeadDim > 0
                || vHeadDim > 0
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case moeIntermediateSize = "moe_intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case _kvHeads = "num_key_value_heads"
            case nSharedExperts = "n_shared_experts"
            case nRoutedExperts = "n_routed_experts"
            case routedScalingFactor = "routed_scaling_factor"
            case numExpertsPerToken = "num_experts_per_tok"
            case firstKDenseReplace = "first_k_dense_replace"
            case moeLayerFrequency = "moe_layer_freq"
            case maxPositionEmbeddings = "max_position_embeddings"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case ropeTraditional = "rope_traditional"
            case ropeScaling = "rope_scaling"
            case attentionBias = "attention_bias"
            case topKMethod = "topk_method"
            case scoringFunction = "scoring_func"
            case nGroup = "n_group"
            case topKGroup = "topk_group"
            case kvLoraRank = "kv_lora_rank"
            case qLoraRank = "q_lora_rank"
            case qkRopeHeadDim = "qk_rope_head_dim"
            case vHeadDim = "v_head_dim"
            case qkNopeHeadDim = "qk_nope_head_dim"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "deepseek_v2"
            vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 129_280
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1280
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6848
            moeIntermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 896
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 12
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 10
            _kvHeads = try c.decodeIfPresent(Int.self, forKey: ._kvHeads)
            nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 2
            nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 64
            routedScalingFactor =
                try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1.0
            numExpertsPerToken =
                try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 6
            firstKDenseReplace =
                try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 1
            moeLayerFrequency =
                try c.decodeIfPresent(Int.self, forKey: .moeLayerFrequency) ?? 1
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 4096
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
            ropeTraditional =
                try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? true
            ropeScaling = try c.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling,
            )
            attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            topKMethod = try c.decodeIfPresent(String.self, forKey: .topKMethod) ?? "greedy"
            scoringFunction =
                try c.decodeIfPresent(String.self, forKey: .scoringFunction) ?? "softmax"
            nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
            topKGroup = try c.decodeIfPresent(Int.self, forKey: .topKGroup) ?? nGroup
            kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank)
            qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
            qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 0
            vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 0
            qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 0
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let modelName: String?
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let imageSize: Int
        public let patchSize: Int
        public let channels: Int
        public let layerNormEps: Float
        public let mlpRatio: Float

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case modelName = "model_name"
            case hiddenSize = "width"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "layers"
            case attentionHeads = "num_attention_heads"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case channels = "num_channels"
            case layerNormEps = "layer_norm_eps"
            case mlpRatio = "mlp_ratio"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "vision"
            modelName = try c.decodeIfPresent(String.self, forKey: .modelName)
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 27
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 384
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
            mlpRatio = try c.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? 3.7362
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize)
                ?? Int((Float(hiddenSize) * mlpRatio).rounded())
        }
    }

    public struct ProjectorConfiguration: Decodable, Sendable {
        public let modelType: String?
        public let projectorType: String
        public let inputDimension: Int
        public let embeddingDimension: Int
        public let depth: Int
        public let mlpRatio: Int
        public let downsampleRatio: Int
        public let tokenPooling: Bool

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case projectorType = "projector_type"
            case inputDimension = "input_dim"
            case embeddingDimension = "n_embed"
            case depth
            case mlpRatio = "mlp_ratio"
            case downsampleRatio = "downsample_ratio"
            case tokenPooling = "token_pooling"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType)
            projectorType =
                try c.decodeIfPresent(String.self, forKey: .projectorType)
                ?? "downsample_mlp_gelu"
            inputDimension = try c.decodeIfPresent(Int.self, forKey: .inputDimension) ?? 1152
            embeddingDimension =
                try c.decodeIfPresent(Int.self, forKey: .embeddingDimension) ?? 1280
            depth = try c.decodeIfPresent(Int.self, forKey: .depth) ?? 2
            mlpRatio = try c.decodeIfPresent(Int.self, forKey: .mlpRatio) ?? 1
            downsampleRatio = try c.decodeIfPresent(Int.self, forKey: .downsampleRatio) ?? 2
            tokenPooling = try c.decodeIfPresent(Bool.self, forKey: .tokenPooling) ?? false
        }
    }

    public let modelType: String
    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let projectorConfiguration: ProjectorConfiguration
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int {
        guard let raw = _imageTokenIndex else {
            return 128_815
        }
        // DeepSeek-VL2 configs commonly carry the Python default 100015, but the
        // shipped tokenizer/checkpoint uses the added <image> token at 128815.
        return raw == 100_015 ? 128_815 : raw
    }

    public let tileTag: String
    public let globalViewPosition: String
    public let candidateResolutions: [[Int]]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case languageConfiguration = "language_config"
        case visionConfiguration = "vision_config"
        case projectorConfiguration = "projector_config"
        case _imageTokenIndex = "image_token_index"
        case tileTag = "tile_tag"
        case globalViewPosition = "global_view_pos"
        case candidateResolutions = "candidate_resolutions"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "deepseek_vl_v2"
        textConfiguration =
            try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? c.decode(TextConfiguration.self, forKey: .languageConfiguration)
        visionConfiguration = try c.decode(VisionConfiguration.self, forKey: .visionConfiguration)
        projectorConfiguration =
            try c.decodeIfPresent(ProjectorConfiguration.self, forKey: .projectorConfiguration)
            ?? ProjectorConfiguration(from: decoder)
        _imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: ._imageTokenIndex)
        tileTag = try c.decodeIfPresent(String.self, forKey: .tileTag) ?? "2D"
        globalViewPosition =
            try c.decodeIfPresent(String.self, forKey: .globalViewPosition) ?? "head"
        candidateResolutions =
            try c.decodeIfPresent([[Int]].self, forKey: .candidateResolutions)
            ?? [[384, 384]]
    }
}

public struct DeepSeekVL2ProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String?
    public let candidateResolutions: [[Int]]
    public let patchSize: Int
    public let downsampleRatio: Int
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let normalize: Bool
    public let imageToken: String
    public let padToken: String
    public let addSpecialToken: Bool
    public let sftFormat: String
    public let maskPrompt: Bool
    public let ignoreID: Int

    public var imageSize: Int {
        candidateResolutions.first?.first ?? 384
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case candidateResolutions = "candidate_resolutions"
        case patchSize = "patch_size"
        case downsampleRatio = "downsample_ratio"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case normalize
        case imageToken = "image_token"
        case padToken = "pad_token"
        case addSpecialToken = "add_special_token"
        case sftFormat = "sft_format"
        case maskPrompt = "mask_prompt"
        case ignoreID = "ignore_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass = try c.decodeIfPresent(String.self, forKey: .processorClass)
        candidateResolutions =
            try c.decodeIfPresent([[Int]].self, forKey: .candidateResolutions)
            ?? [[384, 384]]
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        downsampleRatio = try c.decodeIfPresent(Int.self, forKey: .downsampleRatio) ?? 2
        imageMean = try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd = try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        normalize = try c.decodeIfPresent(Bool.self, forKey: .normalize) ?? true
        imageToken = try c.decodeIfPresent(String.self, forKey: .imageToken) ?? "<image>"
        padToken =
            try c.decodeIfPresent(String.self, forKey: .padToken)
            ?? "<\u{ff5c}\u{2581}pad\u{2581}\u{ff5c}>"
        addSpecialToken = try c.decodeIfPresent(Bool.self, forKey: .addSpecialToken) ?? false
        sftFormat = try c.decodeIfPresent(String.self, forKey: .sftFormat) ?? "deepseek"
        maskPrompt = try c.decodeIfPresent(Bool.self, forKey: .maskPrompt) ?? true
        ignoreID = try c.decodeIfPresent(Int.self, forKey: .ignoreID) ?? -100
    }
}

// MARK: - Processor

public struct DeepSeekVL2Processor: UserInputProcessor {
    private let config: DeepSeekVL2ProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageTokenID: Int
    private let bosTokenID: Int

    public init(_ config: DeepSeekVL2ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        imageTokenID =
            tokenizer.convertTokenToId(config.imageToken)
            ?? tokenizer.encode(text: config.imageToken, addSpecialTokens: false).first
            ?? 128_815
        if let bos = tokenizer.bosToken {
            bosTokenID =
                tokenizer.convertTokenToId(bos)
                ?? tokenizer.encode(text: bos, addSpecialTokens: false).first
                ?? 0
        } else {
            bosTokenID = 0
        }
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty else {
            throw VLMError.singleMediaTypeAllowed
        }

        let images = MediaProcessing.images(from: input)
        let prompt = Self.prompt(from: input)
        let promptWithImages = try Self.promptWithImagePlaceholders(
            prompt,
            imageCount: images.count,
            imageToken: config.imageToken,
        )
        let promptText = Self.renderedPrompt(promptWithImages)

        guard !images.isEmpty else {
            let tokens = [bosTokenID] + tokenizer.encode(text: promptText, addSpecialTokens: false)
            return LMInput(
                text: .init(
                    tokens: MLXArray(tokens).expandedDimensions(axis: 0),
                    mask: MLXArray(Array(repeating: Int8(0), count: tokens.count))
                        .expandedDimensions(axis: 0),
                ),
            )
        }

        var tokenIDs = [bosTokenID]
        var imageMask = [Int8(0)]
        var tileTensors: [MLXArray] = []
        var spatialCropValues: [Int] = []
        var processedImages: [ProcessedImage] = []

        for image in images {
            let ciImage = try MediaProcessing.apply(
                image.asCIImage(),
                processing: input.processing,
            )
            .toSRGB()
            let processed = try preprocess(ciImage, cropping: images.count <= 2)
            processedImages.append(processed)
            tileTensors.append(contentsOf: processed.tiles)
            spatialCropValues.append(contentsOf: [processed.widthTiles, processed.heightTiles])
        }

        var imageIndex = 0
        let promptIDs = tokenizer.encode(text: promptText, addSpecialTokens: false)
        for token in promptIDs {
            if token == imageTokenID {
                guard imageIndex < processedImages.count else {
                    throw VLMError.processing(
                        "DeepSeek-VL2 prompt contains more image tokens than image inputs.",
                    )
                }
                let tokenCount = processedImages[imageIndex].tokenCount
                tokenIDs.append(contentsOf: Array(repeating: imageTokenID, count: tokenCount))
                imageMask.append(contentsOf: Array(repeating: 1, count: tokenCount))
                imageIndex += 1
            } else {
                tokenIDs.append(token)
                imageMask.append(0)
            }
        }
        guard imageIndex == processedImages.count else {
            throw VLMError.processing(
                "DeepSeek-VL2 prompt contains \(imageIndex) image tokens but \(processedImages.count) images were provided.",
            )
        }

        let pixels = stacked(tileTensors, axis: 0).expandedDimensions(axis: 0)
        let spatialCrop = MLXArray(spatialCropValues, [1, images.count, 2])
        return LMInput(
            text: .init(
                tokens: MLXArray(tokenIDs).expandedDimensions(axis: 0),
                mask: MLXArray(imageMask).expandedDimensions(axis: 0),
            ),
            image: .init(pixels: pixels),
            video: .init(pixels: spatialCrop),
        )
    }

    private struct ProcessedImage {
        let tiles: [MLXArray]
        let widthTiles: Int
        let heightTiles: Int
        let tokenCount: Int
    }

    private func preprocess(_ image: CIImage, cropping: Bool) throws -> ProcessedImage {
        let imageSize = config.imageSize
        let bestResolution =
            cropping
            ? Self.selectBestResolution(
                imageSize: image.extent.integral.size,
                candidates: config.candidateResolutions,
            )
            : (width: imageSize, height: imageSize)

        let global = Self.pad(
            image,
            to: CGSize(width: imageSize, height: imageSize),
            color: config.imageMeanTuple,
        )
        let local = Self.pad(
            image,
            to: CGSize(width: bestResolution.width, height: bestResolution.height),
            color: config.imageMeanTuple,
        )

        var tiles = [
            Self.normalizedRGBArray(
                global,
                mean: config.imageMean,
                std: config.imageStd,
                normalize: config.normalize,
            )
            .asType(.bfloat16)
        ]

        let widthTiles = max(bestResolution.width / imageSize, 1)
        let heightTiles = max(bestResolution.height / imageSize, 1)
        for row in 0 ..< heightTiles {
            for col in 0 ..< widthTiles {
                let y = bestResolution.height - (row + 1) * imageSize
                let rect = CGRect(
                    x: col * imageSize,
                    y: y,
                    width: imageSize,
                    height: imageSize,
                )
                let crop =
                    local
                    .cropped(to: rect)
                    .transformed(by: CGAffineTransform(translationX: -rect.minX, y: -rect.minY))
                tiles.append(
                    Self.normalizedRGBArray(
                        crop,
                        mean: config.imageMean,
                        std: config.imageStd,
                        normalize: config.normalize,
                    )
                    .asType(.bfloat16),
                )
            }
        }

        let side = Int(ceil(Float(imageSize / config.patchSize) / Float(config.downsampleRatio)))
        let tokenCount =
            side * (side + 1)
            + 1
            + (heightTiles * side) * (widthTiles * side + 1)
        return ProcessedImage(
            tiles: tiles,
            widthTiles: widthTiles,
            heightTiles: heightTiles,
            tokenCount: tokenCount,
        )
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

    private static func promptWithImagePlaceholders(
        _ prompt: String,
        imageCount: Int,
        imageToken: String,
    )
        throws -> String
    {
        let existing = prompt.components(separatedBy: imageToken).count - 1
        guard existing <= imageCount else {
            throw VLMError.processing(
                "Prompt contains \(existing) image placeholders but only \(imageCount) image inputs.",
            )
        }
        guard existing < imageCount else {
            return prompt
        }
        return String(repeating: "\(imageToken)\n", count: imageCount - existing) + prompt
    }

    private static func renderedPrompt(_ prompt: String) -> String {
        if prompt.contains("<|User|>") || prompt.contains("<|Assistant|>") {
            return prompt
        }
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        return "<|User|>: \(trimmed)\n\n<|Assistant|>:"
    }

    private static func selectBestResolution(
        imageSize: CGSize,
        candidates: [[Int]],
    )
        -> (width: Int, height: Int)
    {
        let originalWidth = max(imageSize.width, 1)
        let originalHeight = max(imageSize.height, 1)
        var best = candidates.first ?? [384, 384]
        var maxEffectiveResolution: CGFloat = 0
        var minWastedResolution = CGFloat.infinity

        for candidate in candidates where candidate.count >= 2 {
            let width = CGFloat(candidate[0])
            let height = CGFloat(candidate[1])
            let scale = min(width / originalWidth, height / originalHeight)
            let downscaledWidth = floor(originalWidth * scale)
            let downscaledHeight = floor(originalHeight * scale)
            let effective = min(
                downscaledWidth * downscaledHeight,
                originalWidth * originalHeight,
            )
            let wasted = width * height - effective
            if effective > maxEffectiveResolution
                || (effective == maxEffectiveResolution && wasted < minWastedResolution)
            {
                maxEffectiveResolution = effective
                minWastedResolution = wasted
                best = candidate
            }
        }
        return (max(best[0], 1), max(best[1], 1))
    }

    private static func pad(
        _ image: CIImage,
        to targetSize: CGSize,
        color: (CGFloat, CGFloat, CGFloat),
    )
        -> CIImage
    {
        let canonical = MediaProcessing.zeroOrigin(image)
        let extent = canonical.extent.integral
        let scale = min(targetSize.width / extent.width, targetSize.height / extent.height)
        let resizedWidth = max(Int((extent.width * scale).rounded()), 1)
        let resizedHeight = max(Int((extent.height * scale).rounded()), 1)
        let resized = canonical.resampled(
            to: CGSize(width: resizedWidth, height: resizedHeight),
            method: .bicubic,
        )
        let x = (targetSize.width - CGFloat(resizedWidth)) * 0.5
        let y = (targetSize.height - CGFloat(resizedHeight)) * 0.5
        let centered = resized.transformed(by: CGAffineTransform(translationX: x, y: y))
        let background = CIImage(
            color: CIColor(red: color.0, green: color.1, blue: color.2, alpha: 1),
        )
        .cropped(to: CGRect(origin: .zero, size: targetSize))
        return centered.composited(over: background)
    }

    private static func normalizedRGBArray(
        _ image: CIImage,
        mean: [CGFloat],
        std: [CGFloat],
        normalize: Bool,
    )
        -> MLXArray
    {
        let canonical = MediaProcessing.zeroOrigin(image)
        let width = Int(canonical.extent.width.rounded())
        let height = Int(canonical.extent.height.rounded())
        let componentsPerPixel = 4
        let bytesPerPixel = componentsPerPixel * MemoryLayout<Float32>.size
        let bytesPerRow = width * bytesPerPixel
        var data = Data(count: width * height * bytesPerPixel)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)

        data.withUnsafeMutableBytes { pointer in
            CIContext().render(
                canonical,
                toBitmap: pointer.baseAddress!,
                rowBytes: bytesPerRow,
                bounds: canonical.extent,
                format: .RGBAf,
                colorSpace: colorSpace,
            )
        }

        let rgba = MLXArray(data, [height, width, 4], type: Float32.self)
        let rgb = rgba[0..., 0..., ..<3]
        let alpha = maximum(rgba[0..., 0..., 3...], MLXArray(Float32(1e-6)))
        var array = clip(rgb / alpha, min: Float32(0), max: Float32(1))
        if normalize {
            let meanArray = MLXArray(mean.map { Float32($0) }).reshaped(1, 1, 3)
            let stdArray = MLXArray(std.map { Float32($0) }).reshaped(1, 1, 3)
            array = (array - meanArray) / stdArray
        }
        return array.transposed(2, 0, 1)
    }
}

// MARK: - Language

private enum DeepSeekVL2Language {
    final class Attention: Module {
        let config: DeepSeekVL2Configuration.TextConfiguration
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        init(_ config: DeepSeekVL2Configuration.TextConfiguration) {
            precondition(
                !config.usesMLA,
                "DeepSeek-VL2 MLA attention is not implemented yet.",
            )
            self.config = config
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
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
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        init(
            _ config: DeepSeekVL2Configuration.TextConfiguration,
            intermediateSize: Int? = nil,
        ) {
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
        let config: DeepSeekVL2Configuration.TextConfiguration
        @ModuleInfo var weight: MLXArray

        init(_ config: DeepSeekVL2Configuration.TextConfiguration) {
            self.config = config
            _weight.wrappedValue = zeros([config.nRoutedExperts ?? 1, config.hiddenSize])
        }

        func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
            let logits = x.matmul(weight.T)
            let scores: MLXArray =
                if config.scoringFunction == "sigmoid" {
                    sigmoid(logits)
                } else {
                    softmax(logits, axis: -1, precise: true)
                }
            let k = config.numExpertsPerToken
            let indices = argPartition(-scores, kth: k - 1, axis: -1)[.ellipsis, ..<k]
            let selected = takeAlong(scores, indices, axis: -1) * config.routedScalingFactor
            return (indices, selected)
        }
    }

    final class MoE: Module, UnaryLayer {
        let config: DeepSeekVL2Configuration.TextConfiguration
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
        @ModuleInfo var gate: MoEGate
        @ModuleInfo(key: "shared_experts") var sharedExperts: MLP?

        init(_ config: DeepSeekVL2Configuration.TextConfiguration) {
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

        init(_ config: DeepSeekVL2Configuration.TextConfiguration, layerIndex: Int) {
            _attention.wrappedValue = Attention(config)
            if config.nRoutedExperts != nil,
                layerIndex >= config.firstKDenseReplace,
                layerIndex % config.moeLayerFrequency == 0
            {
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
        )
            -> MLXArray
        {
            let r = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            return h + mlp(postAttentionLayerNorm(h))
        }
    }

    final class TextModel: Module {
        let config: DeepSeekVL2Configuration.TextConfiguration
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm

        init(_ config: DeepSeekVL2Configuration.TextConfiguration) {
            self.config = config
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
                DecoderLayer(config, layerIndex: $0)
            }
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]?,
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
                preconditionFailure("inputs or inputEmbedding must be provided")
            }

            let mask = createAttentionMask(h: h, cache: cache?.first)
            for (index, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[index])
            }
            return norm(h)
        }
    }

    final class LanguageModel: Module {
        let config: DeepSeekVL2Configuration.TextConfiguration
        @ModuleInfo var model: TextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear

        init(_ config: DeepSeekVL2Configuration.TextConfiguration) {
            precondition(
                !config.usesMLA,
                "DeepSeek-VL2 MLA attention is not implemented yet.",
            )
            self.config = config
            _model.wrappedValue = TextModel(config)
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }

        var kvHeads: [Int] {
            Array(repeating: config.kvHeads, count: config.hiddenLayers)
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

private enum DeepSeekVL2Vision {
    final class Attention: Module {
        let config: DeepSeekVL2Configuration.VisionConfiguration
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear

        init(_ config: DeepSeekVL2Configuration.VisionConfiguration) {
            self.config = config
            heads = config.attentionHeads
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _qkv.wrappedValue = Linear(
                config.hiddenSize,
                3 * config.hiddenSize,
                bias: true,
            )
            _proj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (batchSize, sequenceLength, channels) = (x.dim(0), x.dim(1), x.dim(2))
            let qkvStates = qkv(x)
                .reshaped(batchSize, sequenceLength, 3, heads, channels / heads)
                .transposed(2, 0, 3, 1, 4)
            let queries = qkvStates[0]
            let keys = qkvStates[1]
            let values = qkvStates[2]

            let output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .none,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, sequenceLength, -1)

            return proj(output)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let activation = GELU(approximation: .precise)

        init(hiddenSize: Int, intermediateSize: Int, bias: Bool = true) {
            _fc1.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
            _fc2.wrappedValue = Linear(intermediateSize, hiddenSize, bias: bias)
        }

        convenience init(_ config: DeepSeekVL2Configuration.VisionConfiguration) {
            self.init(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    final class EncoderLayer: Module {
        @ModuleInfo var attn: Attention
        @ModuleInfo var norm1: LayerNorm
        @ModuleInfo var mlp: MLP
        @ModuleInfo var norm2: LayerNorm

        init(_ config: DeepSeekVL2Configuration.VisionConfiguration) {
            _attn.wrappedValue = Attention(config)
            _norm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
            _mlp.wrappedValue = MLP(config)
            _norm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = x + attn(norm1(x))
            return h + mlp(norm2(h))
        }
    }

    final class VisionEmbeddings: Module {
        let config: DeepSeekVL2Configuration.VisionConfiguration
        let numPatches: Int

        @ModuleInfo var proj: Conv2d

        init(_ config: DeepSeekVL2Configuration.VisionConfiguration) {
            self.config = config
            numPatches =
                (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize)
            _proj.wrappedValue = Conv2d(
                inputChannels: config.channels,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: true,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            proj(x).flattened(start: 1, end: 2)
        }
    }

    final class AttentionPoolLatent: Module {
        let config: DeepSeekVL2Configuration.VisionConfiguration
        let heads: Int
        let headDim: Int
        let latentLength = 1

        @ModuleInfo var latent: MLXArray
        @ModuleInfo var q: Linear
        @ModuleInfo var kv: Linear
        @ModuleInfo var proj: Linear
        @ModuleInfo var norm: LayerNorm
        @ModuleInfo var mlp: MLP

        init(_ config: DeepSeekVL2Configuration.VisionConfiguration) {
            self.config = config
            heads = config.attentionHeads
            headDim = config.hiddenSize / config.attentionHeads
            _latent.wrappedValue = zeros([1, latentLength, config.hiddenSize])
            _q.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _kv.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * 2, bias: true)
            _proj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _norm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-5)
            _mlp.wrappedValue = MLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let batchSize = x.dim(0)
            let sequenceLength = x.dim(1)
            let channels = x.dim(2)
            let latentQuery = broadcast(
                latent.asType(x.dtype), to: [batchSize, latentLength, channels])
            var queries = q(latentQuery)
                .reshaped(batchSize, latentLength, heads, headDim)
                .transposed(0, 2, 1, 3)
            let kvStates = kv(x)
                .reshaped(batchSize, sequenceLength, 2, heads, headDim)
                .transposed(2, 0, 3, 1, 4)
            let keys = kvStates[0]
            let values = kvStates[1]
            queries = queries.asType(keys.dtype)

            var output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: pow(Float(headDim), -0.5),
                mask: .none,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, latentLength, channels)
            output = proj(output)
            output = output + mlp(norm(output))
            return output[0..., 0, 0...]
        }
    }

    final class SigLipVisionModel: Module {
        let config: DeepSeekVL2Configuration.VisionConfiguration
        @ModuleInfo(key: "patch_embed") var patchEmbed: VisionEmbeddings
        @ModuleInfo(key: "pos_embed") var posEmbed: MLXArray
        @ModuleInfo(key: "blocks") var blocks: [EncoderLayer]
        @ModuleInfo var norm: LayerNorm
        @ModuleInfo(key: "attn_pool") var attentionPool: AttentionPoolLatent

        init(_ config: DeepSeekVL2Configuration.VisionConfiguration) {
            self.config = config
            let numPatches =
                (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize)
            _patchEmbed.wrappedValue = VisionEmbeddings(config)
            _posEmbed.wrappedValue = zeros([1, numPatches, config.hiddenSize])
            _blocks.wrappedValue = (0 ..< config.hiddenLayers).map { _ in EncoderLayer(config) }
            _norm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
            _attentionPool.wrappedValue = AttentionPoolLatent(config)
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, [MLXArray]?,
        ) {
            var x = patchEmbed(x)
            x = x + posEmbed.asType(x.dtype)

            var encoderStates: [MLXArray]? = outputHiddenStates ? [x] : nil
            for block in blocks {
                x = block(x)
                encoderStates?.append(x)
            }

            return (norm(x), x, encoderStates)
        }
    }

    final class VisionModel: Module {
        let config: DeepSeekVL2Configuration.VisionConfiguration

        @ModuleInfo(key: "vision_tower") var visionTower: SigLipVisionModel

        init(_ config: DeepSeekVL2Configuration.VisionConfiguration) {
            self.config = config
            _visionTower.wrappedValue = SigLipVisionModel(config)
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, [MLXArray]?,
        ) {
            visionTower(x, outputHiddenStates: outputHiddenStates)
        }
    }
}

// MARK: - Projector

private final class DeepSeekVL2Projector: Module, UnaryLayer {
    let config: DeepSeekVL2Configuration.ProjectorConfiguration
    @ModuleInfo(key: "layers") var layers: [Module]

    init(_ config: DeepSeekVL2Configuration.ProjectorConfiguration) {
        self.config = config
        precondition(
            !config.tokenPooling, "DeepSeek-VL2 token pooling projector is not implemented yet.")

        switch config.projectorType {
        case "identity":
            _layers.wrappedValue = []
        case "linear":
            _layers.wrappedValue = [
                Linear(config.inputDimension, config.embeddingDimension, bias: true)
            ]
        case "mlp_gelu":
            var modules: [Module] = [
                Linear(config.inputDimension, config.embeddingDimension, bias: true)
            ]
            for _ in 1 ..< config.depth {
                modules.append(GELU())
                modules.append(
                    Linear(config.embeddingDimension, config.embeddingDimension, bias: true))
            }
            _layers.wrappedValue = modules
        default:
            let inner = config.embeddingDimension * config.mlpRatio
            var modules: [Module] = [
                Linear(
                    config.inputDimension * config.downsampleRatio * config.downsampleRatio,
                    inner,
                    bias: true,
                )
            ]
            for _ in 1 ..< max(config.depth - 1, 1) {
                modules.append(GELU())
                modules.append(Linear(inner, inner, bias: true))
            }
            modules.append(GELU())
            modules.append(Linear(inner, config.embeddingDimension, bias: true))
            _layers.wrappedValue = modules
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        if config.projectorType == "downsample_mlp_gelu" {
            h = downsample(h)
        }
        for layer in layers {
            h = (layer as! UnaryLayer)(h)
        }
        return h
    }

    private func downsample(_ x: MLXArray) -> MLXArray {
        let batchSize = x.dim(0)
        let hw = x.dim(1)
        let inputDim = x.dim(2)
        let side = Int(sqrt(Double(hw)))
        let ds = config.downsampleRatio
        let pad = side % ds == 0 ? 0 : ds - side % ds
        var grid = x.reshaped(batchSize, side, side, inputDim)
        if pad > 0 {
            grid = padded(grid, widths: [0, .init((0, pad)), .init((0, pad)), 0])
        }
        grid = grid.transposed(0, 3, 1, 2)
        let paddedSide = side + pad
        var patches: [MLXArray] = []
        for row in stride(from: 0, through: paddedSide - ds, by: ds) {
            for col in stride(from: 0, through: paddedSide - ds, by: ds) {
                patches.append(
                    grid[0..., 0..., row ..< (row + ds), col ..< (col + ds)]
                        .reshaped(batchSize, -1),
                )
            }
        }
        return stacked(patches, axis: 1)
    }
}

// MARK: - Model

public final class DeepSeekVL2: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo private var vision: DeepSeekVL2Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: DeepSeekVL2Language.LanguageModel
    @ModuleInfo private var projector: DeepSeekVL2Projector
    @ModuleInfo(key: "image_newline") private var imageNewline: MLXArray
    @ModuleInfo(key: "view_separator") private var viewSeparator: MLXArray

    public let config: DeepSeekVL2Configuration
    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: DeepSeekVL2Configuration) {
        precondition(config.tileTag == "2D", "Only DeepSeek-VL2 2D tile tags are implemented.")
        self.config = config
        _vision.wrappedValue = DeepSeekVL2Vision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue =
            DeepSeekVL2Language.LanguageModel(config.textConfiguration)
        _projector.wrappedValue = DeepSeekVL2Projector(config.projectorConfiguration)
        let std = 1 / sqrt(Float(config.projectorConfiguration.embeddingDimension))
        _imageNewline.wrappedValue =
            MLXRandom.normal([
                config.projectorConfiguration.embeddingDimension
            ]) * std
        _viewSeparator.wrappedValue =
            MLXRandom.normal([
                config.projectorConfiguration.embeddingDimension
            ]) * std
    }

    private func visualFeatures(pixelValues: MLXArray?, spatialCrop: MLXArray?) -> MLXArray? {
        guard let pixelValues, let spatialCrop else {
            return nil
        }
        precondition(pixelValues.dim(0) == 1, "DeepSeek-VL2 currently expects batch size 1.")
        let spatialValues = spatialCrop.asArray(Int.self)
        let imageCount = spatialCrop.dim(1)
        var totalTiles = 0
        var imageCrops: [(width: Int, height: Int)] = []
        for index in 0 ..< imageCount {
            let width = spatialValues[index * 2]
            let height = spatialValues[index * 2 + 1]
            guard width > 0, height > 0 else {
                break
            }
            imageCrops.append((width, height))
            totalTiles += 1 + width * height
        }
        guard totalTiles > 0 else {
            return nil
        }

        let tiles = pixelValues[0, 0 ..< totalTiles, 0..., 0..., 0...]
        let (hiddenStates, _, _) = vision(
            tiles.transposed(0, 2, 3, 1),
            outputHiddenStates: true,
        )
        let imageFeatures = projector(hiddenStates)
        let side = Int(sqrt(Double(imageFeatures.dim(1))))
        return processImageFeatures(
            imageFeatures,
            imageCrops: imageCrops,
            side: side,
            hiddenSize: imageFeatures.dim(2),
        )
    }

    private func processImageFeatures(
        _ imageFeatures: MLXArray,
        imageCrops: [(width: Int, height: Int)],
        side: Int,
        hiddenSize: Int,
    )
        -> MLXArray
    {
        var tileIndex = 0
        var imageFeatureRuns: [MLXArray] = []

        for crop in imageCrops {
            let localTileCount = crop.width * crop.height
            let globalFeatures = imageFeatures[tileIndex]
            let localFeatures =
                imageFeatures[(tileIndex + 1) ..< (tileIndex + 1 + localTileCount), 0..., 0...]
            tileIndex += 1 + localTileCount

            var globalGrid = globalFeatures.reshaped(side, side, hiddenSize)
            globalGrid = concatenated(
                [
                    globalGrid,
                    broadcast(
                        imageNewline[.newAxis, .newAxis, 0...],
                        to: [side, 1, hiddenSize],
                    ),
                ],
                axis: 1,
            )
            let globalFlattened = globalGrid.reshaped(-1, hiddenSize)

            var localGrid =
                localFeatures
                .reshaped(crop.height, crop.width, side, side, hiddenSize)
                .transposed(0, 2, 1, 3, 4)
                .reshaped(crop.height * side, crop.width * side, hiddenSize)
            localGrid = concatenated(
                [
                    localGrid,
                    broadcast(
                        imageNewline[.newAxis, .newAxis, 0...],
                        to: [crop.height * side, 1, hiddenSize],
                    ),
                ],
                axis: 1,
            )
            let localFlattened = localGrid.reshaped(-1, hiddenSize)
            let separator = viewSeparator[.newAxis, 0...]
            if config.globalViewPosition == "head" {
                imageFeatureRuns.append(
                    concatenated([globalFlattened, separator, localFlattened], axis: 0),
                )
            } else {
                imageFeatureRuns.append(
                    concatenated([localFlattened, separator, globalFlattened], axis: 0),
                )
            }
        }

        return concatenated(imageFeatureRuns, axis: 0)
    }

    private func inputEmbeddings(_ input: LMInput) -> MLXArray {
        let inputIDs = input.text.tokens
        let safeIDs: MLXArray =
            if let mask = input.text.mask {
                which(mask .!= MLXArray(0), MLXArray(0), inputIDs)
            } else {
                inputIDs
            }
        let embeddings = languageModel.model.embedTokens(safeIDs)
        guard
            let features = visualFeatures(
                pixelValues: input.image?.pixels,
                spatialCrop: input.video?.pixels,
            )?.asType(embeddings.dtype)
        else {
            return embeddings
        }

        let maskValues = input.text.mask?.asArray(Int8.self) ?? []
        let positions = maskValues.enumerated().compactMap { $0.element != 0 ? $0.offset : nil }
        guard !positions.isEmpty else {
            return embeddings
        }
        precondition(
            positions.count == features.dim(0),
            "DeepSeek-VL2 image token count \(positions.count) does not match features \(features.dim(0)).",
        )
        let result = embeddings
        result[0, MLXArray(positions), 0...] = features
        return result
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let embeddings = inputEmbeddings(input)
        let totalPositions = embeddings.dim(1)
        let processed = try prefill.forEachChunk(total: totalPositions) { range in
            _ = languageModel(nil, cache: cache, inputEmbedding: embeddings[0..., range, 0...])
            asyncEval(cache)
        }
        eval(cache)
        defer { prefill.progress?(totalPositions, totalPositions) }
        return .logits(
            languageModel(nil, cache: cache, inputEmbedding: embeddings[0..., processed..., 0...]),
        )
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        DeepSeekVL2Sanitizer.sanitize(weights: weights, config: config)
    }
}

private enum DeepSeekVL2Sanitizer {
    static func sanitize(
        weights: [String: MLXArray],
        config: DeepSeekVL2Configuration,
    )
        -> [String: MLXArray]
    {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.contains("position_ids") || key.contains("rotary_emb.inv_freq") {
                continue
            }
            var newKey = key
            if newKey.contains("language"), !newKey.contains("language_model") {
                newKey = newKey.replacingOccurrences(
                    of: "language.model",
                    with: "language_model.model",
                )
                newKey = newKey.replacingOccurrences(
                    of: "language.lm_head",
                    with: "language_model.lm_head",
                )
            }
            if newKey.hasPrefix("vision."), !newKey.hasPrefix("vision.vision_tower.") {
                newKey = "vision.vision_tower." + newKey.dropFirst("vision.".count)
            }
            newKey = newKey.replacingOccurrences(of: "view_seperator", with: "view_separator")

            if newKey.hasSuffix("patch_embed.proj.weight"), value.ndim == 4, value.shape[1] == 3 {
                sanitized[newKey] = value.transposed(0, 2, 3, 1)
            } else {
                sanitized[newKey] = value
            }
        }

        stackExperts(weights: &sanitized, config: config.textConfiguration)
        return sanitized
    }

    private static func stackExperts(
        weights: inout [String: MLXArray],
        config: DeepSeekVL2Configuration.TextConfiguration,
    ) {
        guard let routedExperts = config.nRoutedExperts else {
            return
        }
        for layer in 0 ..< config.hiddenLayers {
            let prefix = "language_model.model.layers.\(layer).mlp"
            for (source, target) in [
                ("gate_proj", "gate_proj"),
                ("up_proj", "up_proj"),
                ("down_proj", "down_proj"),
            ] {
                for suffix in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).experts.0.\(source).\(suffix)"
                    guard weights[firstKey] != nil else {
                        continue
                    }
                    var toStack: [MLXArray] = []
                    for expert in 0 ..< routedExperts {
                        let key = "\(prefix).experts.\(expert).\(source).\(suffix)"
                        if let value = weights.removeValue(forKey: key) {
                            toStack.append(value)
                        }
                    }
                    if toStack.count == routedExperts {
                        weights["\(prefix).switch_mlp.\(target).\(suffix)"] = stacked(toStack)
                    }
                }
            }
        }
    }
}
