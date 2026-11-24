import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct DeepSeekOCRConfiguration: Decodable, Sendable {
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
        public let rmsNormEps: Float
        public let ropeTheta: Float
        public let ropeTraditional: Bool
        public let attentionBias: Bool
        public let topKMethod: String
        public let scoringFunction: String
        public let nGroup: Int
        public let topKGroup: Int
        let slidingWindow: Int
        let maximumPositions: Int
        let ropeScaling: [String: StringOrNumber]?
        let sparseAttention: UnlimitedOCRSparseConfiguration?

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
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case ropeTraditional = "rope_traditional"
            case attentionBias = "attention_bias"
            case topKMethod = "topk_method"
            case scoringFunction = "scoring_func"
            case nGroup = "n_group"
            case topKGroup = "topk_group"
            case slidingWindow = "sliding_window_size"
            case alternateSlidingWindow = "sliding_window"
            case maximumPositions = "max_position_embeddings"
            case ropeScaling = "rope_scaling"
            case sparseAttention = "sparse_attention_config"
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
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
            ropeTraditional =
                try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
            attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            topKMethod = try c.decodeIfPresent(String.self, forKey: .topKMethod) ?? "greedy"
            scoringFunction =
                try c.decodeIfPresent(String.self, forKey: .scoringFunction) ?? "softmax"
            nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
            topKGroup = try c.decodeIfPresent(Int.self, forKey: .topKGroup) ?? nGroup
            slidingWindow =
                try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
                ?? c.decodeIfPresent(Int.self, forKey: .alternateSlidingWindow) ?? 128
            maximumPositions = try c.decodeIfPresent(Int.self, forKey: .maximumPositions) ?? 32768
            ropeScaling = try c.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeScaling)
            sparseAttention = try c.decodeIfPresent(
                UnlimitedOCRSparseConfiguration.self, forKey: .sparseAttention)
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public struct WidthEntry: Decodable, Sendable {
            public let width: Int?
            public let dim: Int?
            public let layers: Int?
            public let heads: Int?
            public let patchSize: Int?
            public let imageSize: Int?
            public let downsampleChannels: [Int]?
            public let globalAttentionIndexes: [Int]?

            enum CodingKeys: String, CodingKey {
                case width
                case dim
                case layers
                case heads
                case patchSize = "patch_size"
                case imageSize = "image_size"
                case downsampleChannels = "downsample_channels"
                case globalAttentionIndexes = "global_attn_indexes"
            }
        }

        public let modelType: String
        public let modelName: String
        public let globalImageSize: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let patchSize: Int
        public let imageSize: Int
        public let channels: Int
        public let layerNormEps: Float
        public let qwenConfiguration: Qwen2Configuration
        public let samConfiguration: SAMConfiguration

        public var isVersion2: Bool {
            modelName == "deepencoderv2"
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case modelName = "model_name"
            case globalImageSize = "image_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "layers"
            case attentionHeads = "num_attention_heads"
            case patchSize = "patch_size"
            case channels = "num_channels"
            case layerNormEps = "layer_norm_eps"
            case width
        }

        public init(from decoder: Decoder) throws {
            try self.init(from: decoder, defaultLayerNormEps: 1e-6)
        }

        init(from decoder: Decoder, defaultLayerNormEps: Float) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "vision"
            modelName =
                try c.decodeIfPresent(String.self, forKey: .modelName) ?? "deeplip_b_l"
            globalImageSize =
                try c.decodeIfPresent(Int.self, forKey: .globalImageSize) ?? 1024
            let widths =
                try c.decodeIfPresent([String: WidthEntry].self, forKey: .width) ?? [:]
            let clip = widths["clip-l-14-224"]
            let qwen = widths["qwen2-0-5b"]
            let sam = widths["sam_vit_b"]
            let qwenDim = qwen?.dim ?? 896

            if modelName == "deepencoderv2" {
                hiddenSize = qwenDim
                intermediateSize =
                    try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4864
                hiddenLayers = qwen?.layers ?? 24
                attentionHeads = qwen?.heads ?? 14
                patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
                imageSize =
                    try c.decodeIfPresent(Int.self, forKey: .globalImageSize) ?? 1024
            } else {
                hiddenSize =
                    try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? clip?.width ?? 1024
                intermediateSize =
                    try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4096
                hiddenLayers =
                    try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? clip?.layers ?? 24
                attentionHeads =
                    try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? clip?.heads ?? 16
                patchSize =
                    try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? clip?.patchSize ?? 14
                imageSize = clip?.imageSize ?? 224
            }
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            layerNormEps =
                try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? defaultLayerNormEps

            qwenConfiguration = Qwen2Configuration(
                dim: qwenDim,
                layers: qwen?.layers ?? 24,
                heads: qwen?.heads ?? 14,
                kvHeads: 2,
                intermediateSize: 4864,
                rmsNormEps: 1e-6,
                ropeTheta: 1_000_000,
            )
            samConfiguration = SAMConfiguration(
                imageSize: 1024,
                width: sam?.width ?? 768,
                layers: sam?.layers ?? 12,
                heads: sam?.heads ?? 12,
                patchSize: sam?.patchSize ?? 16,
                windowSize: 14,
                globalAttentionIndexes: sam?.globalAttentionIndexes ?? [2, 5, 8, 11],
                finalOutputChannels: modelName == "deepencoderv2" ? 896 : 1024,
            )
        }
    }

    public struct Qwen2Configuration: Sendable {
        public let dim: Int
        public let layers: Int
        public let heads: Int
        public let kvHeads: Int
        public let intermediateSize: Int
        public let rmsNormEps: Float
        public let ropeTheta: Float
    }

    public struct SAMConfiguration: Sendable {
        public let imageSize: Int
        public let width: Int
        public let layers: Int
        public let heads: Int
        public let patchSize: Int
        public let windowSize: Int
        public let globalAttentionIndexes: [Int]
        public let finalOutputChannels: Int
    }

    public struct ProjectorConfiguration: Decodable, Sendable {
        public let projectorType: String
        public let inputDimension: Int
        public let embeddingDimension: Int

        enum CodingKeys: String, CodingKey {
            case projectorType = "projector_type"
            case inputDimension = "input_dim"
            case embeddingDimension = "n_embed"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            projectorType = try c.decodeIfPresent(String.self, forKey: .projectorType) ?? "linear"
            inputDimension = try c.decodeIfPresent(Int.self, forKey: .inputDimension) ?? 2048
            embeddingDimension =
                try c.decodeIfPresent(Int.self, forKey: .embeddingDimension) ?? 1280
        }
    }

    public let modelType: String
    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let projectorConfiguration: ProjectorConfiguration
    public let imageTokenIndex: Int
    public let tileTag: String
    public let globalViewPosition: String
    let dashAttention: UnlimitedOCRDASHParameters?
    var isUnlimitedOCR: Bool { Self.isUnlimitedOCR(modelType) }

    static func isUnlimitedOCR(_ type: String) -> Bool {
        type == "unlimited-ocr" || type == "unlimited_ocr"
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case languageConfiguration = "language_config"
        case visionConfiguration = "vision_config"
        case projectorConfiguration = "projector_config"
        case imageTokenIndex = "image_token_index"
        case tileTag = "tile_tag"
        case globalViewPosition = "global_view_pos"
        case sparseAttention = "sparse_attention_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "deepseekocr"
        textConfiguration =
            try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? (c.decodeIfPresent(TextConfiguration.self, forKey: .languageConfiguration))
            ?? TextConfiguration(from: decoder)
        let unlimited = Self.isUnlimitedOCR(modelType)
        visionConfiguration =
            if unlimited, c.contains(.visionConfiguration) {
                try VisionConfiguration(
                    from: c.superDecoder(forKey: .visionConfiguration), defaultLayerNormEps: 1e-5)
            } else {
                try c.decode(VisionConfiguration.self, forKey: .visionConfiguration)
            }
        projectorConfiguration =
            try c.decodeIfPresent(ProjectorConfiguration.self, forKey: .projectorConfiguration)
            ?? ProjectorConfiguration(from: decoder)
        imageTokenIndex =
            try c.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? 128_815
        tileTag = try c.decodeIfPresent(String.self, forKey: .tileTag) ?? "2D"
        globalViewPosition =
            try c.decodeIfPresent(String.self, forKey: .globalViewPosition) ?? "head"
        let text = textConfiguration
        if unlimited {
            guard text.slidingWindow > 0, text.hiddenLayers > 0, text.attentionHeads > 0,
                text.hiddenSize > 0, text.hiddenSize.isMultiple(of: 2 * text.attentionHeads),
                text.kvHeads > 0, text.attentionHeads.isMultiple(of: text.kvHeads),
                text.maximumPositions > 0
            else {
                throw ModelExecutionError.invalidOptions("Invalid Unlimited-OCR decoder geometry")
            }
        }
        if unlimited,
            let sparse = try c.decodeIfPresent(
                UnlimitedOCRSparseConfiguration.self, forKey: .sparseAttention)
                ?? text.sparseAttention,
            sparse.isDASH
        {
            try validateRoPEConfiguration(text.ropeScaling, context: "Unlimited-OCR-2 rope_scaling")
            dashAttention = try sparse.parameters(
                headDim: text.hiddenSize / text.attentionHeads, layerCount: text.hiddenLayers,
                imageToken: imageTokenIndex)
        } else {
            dashAttention = nil
        }
    }
}

public struct DeepSeekOCRProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String?
    public let candidateResolutions: [[Int]]
    public let patchSize: Int
    public let downsampleRatio: Int
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let normalize: Bool
    public let imageToken: String
    public let imageTokenID: Int
    public let bosTokenID: Int
    public let padTokenID: Int
    public let localImageSize: Int?
    public let baseImageSize: Int?
    public let minPatches: Int?
    public let maxPatches: Int?
    let unlimitedPageLayout: Bool
    let unlimitedMaxCrops: Int?
    let unlimitedCropping: Bool?

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
        case imageTokenID = "image_token_id"
        case bosTokenID = "bos_token_id"
        case padTokenID = "pad_token_id"
        case localImageSize = "image_size"
        case baseImageSize = "base_size"
        case minPatches = "min_patches"
        case maxPatches = "max_patches"
        case tileSize = "tile_size"
        case imageTokenIndex = "image_token_index"
        case pageLayout = "page_layout"
        case maxCrops = "max_crops"
        case cropToPatches = "crop_to_patches"
        case sparseAttention = "sparse_attention_config"
        case textConfiguration = "text_config"
        case languageConfiguration = "language_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass = try c.decodeIfPresent(String.self, forKey: .processorClass)
        candidateResolutions =
            try c.decodeIfPresent([[Int]].self, forKey: .candidateResolutions)
            ?? [[1024, 1024]]
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        downsampleRatio =
            try c.decodeIfPresent(Int.self, forKey: .downsampleRatio) ?? 4
        imageMean = try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd = try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        normalize = try c.decodeIfPresent(Bool.self, forKey: .normalize) ?? true
        imageToken = try c.decodeIfPresent(String.self, forKey: .imageToken) ?? "<image>"
        imageTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imageTokenID)
            ?? c.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? 128_815
        bosTokenID = try c.decodeIfPresent(Int.self, forKey: .bosTokenID) ?? 0
        padTokenID = try c.decodeIfPresent(Int.self, forKey: .padTokenID) ?? 2
        localImageSize =
            try c.decodeIfPresent(Int.self, forKey: .localImageSize)
            ?? c.decodeIfPresent(Int.self, forKey: .tileSize)
        baseImageSize = try c.decodeIfPresent(Int.self, forKey: .baseImageSize)
        minPatches = try c.decodeIfPresent(Int.self, forKey: .minPatches)
        maxPatches = try c.decodeIfPresent(Int.self, forKey: .maxPatches)
        let sparse = try c.decodeIfPresent(
            UnlimitedOCRSparseConfiguration.self, forKey: .sparseAttention)
        let text =
            try c.decodeIfPresent(
                DeepSeekOCRConfiguration.TextConfiguration.self, forKey: .textConfiguration)
            ?? c.decodeIfPresent(
                DeepSeekOCRConfiguration.TextConfiguration.self, forKey: .languageConfiguration)
        let pageLayout = try c.decodeIfPresent(Bool.self, forKey: .pageLayout) ?? false
        unlimitedPageLayout = (sparse ?? text?.sparseAttention)?.isDASH ?? pageLayout
        unlimitedMaxCrops = try c.decodeIfPresent(Int.self, forKey: .maxCrops)
        unlimitedCropping = try c.decodeIfPresent(Bool.self, forKey: .cropToPatches)
    }
}

// MARK: - Processor

private enum DeepSeekOCRProcessorVariant {
    case v1
    case v2
    case unlimited
}

public struct DeepSeekOCRProcessor: UserInputProcessor {
    private let processor: DeepSeekOCRProcessorCore

    public init(_ config: DeepSeekOCRProcessorConfiguration, tokenizer: any Tokenizer) {
        processor = DeepSeekOCRProcessorCore(config, tokenizer: tokenizer, variant: .v1)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        try processor.prepare(input: input)
    }
}

public struct DeepSeekOCR2Processor: UserInputProcessor {
    private let processor: DeepSeekOCRProcessorCore

    public init(_ config: DeepSeekOCRProcessorConfiguration, tokenizer: any Tokenizer) {
        processor = DeepSeekOCRProcessorCore(config, tokenizer: tokenizer, variant: .v2)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        try processor.prepare(input: input)
    }
}

public struct UnlimitedOCRProcessor: UserInputProcessor {
    private let processor: DeepSeekOCRProcessorCore

    public init(_ config: DeepSeekOCRProcessorConfiguration, tokenizer: any Tokenizer) {
        processor = DeepSeekOCRProcessorCore(config, tokenizer: tokenizer, variant: .unlimited)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        try processor.prepare(input: input)
    }
}

private struct DeepSeekOCRProcessorCore {
    private let config: DeepSeekOCRProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let variant: DeepSeekOCRProcessorVariant
    private let imageTokenID: Int

    init(
        _ config: DeepSeekOCRProcessorConfiguration,
        tokenizer: any Tokenizer,
        variant: DeepSeekOCRProcessorVariant,
    ) {
        self.config = config
        self.tokenizer = tokenizer
        self.variant = variant
        imageTokenID =
            tokenizer.convertTokenToId(config.imageToken)
            ?? tokenizer.encode(text: config.imageToken, addSpecialTokens: false).first
            ?? config.imageTokenID
    }

    func prepare(input: UserInput) throws -> LMInput {
        guard input.videos.isEmpty else {
            throw VLMError.singleMediaTypeAllowed
        }
        let images = MediaProcessing.images(from: input)
        if variant == .unlimited { return try prepareUnlimited(input, images: images) }
        guard images.count <= 1 else {
            throw VLMError.singleImageAllowed
        }

        let prompt = Self.prompt(from: input)
        guard let image = images.first else {
            var tokens = [config.bosTokenID]
            tokens.append(contentsOf: tokenizer.encode(text: prompt, addSpecialTokens: false))
            if !tokens.isEmpty {
                tokens.removeLast()
            }
            return LMInput(
                text: .init(
                    tokens: MLXArray(tokens).expandedDimensions(axis: 0),
                    mask: MLXArray(Array(repeating: Int8(0), count: tokens.count))
                        .expandedDimensions(axis: 0),
                ),
            )
        }

        let conversation = Self.promptWithImageToken(prompt, imageToken: config.imageToken)
        let pieces = conversation.components(separatedBy: config.imageToken)
        guard pieces.count == 2 else {
            throw VLMError.processing("DeepSeek OCR expects exactly one image placeholder.")
        }

        let ciImage = try MediaProcessing.apply(image.asCIImage(), processing: input.processing)
            .toSRGB()

        let processed: ProcessedImage =
            switch variant {
            case .v1, .unlimited:
                try processV1Image(ciImage)
            case .v2:
                try processV2Image(ciImage)
            }

        var tokenIDs = [Int]()
        var imageMask = [Int8]()
        let prefixIDs = tokenizer.encode(text: pieces[0], addSpecialTokens: false)
        tokenIDs.append(contentsOf: prefixIDs)
        imageMask.append(contentsOf: Array(repeating: 0, count: prefixIDs.count))
        tokenIDs.append(
            contentsOf: Array(repeating: imageTokenID, count: processed.imageTokenCount))
        imageMask.append(contentsOf: Array(repeating: 1, count: processed.imageTokenCount))
        let suffixIDs = tokenizer.encode(text: pieces[1], addSpecialTokens: false)
        tokenIDs.append(contentsOf: suffixIDs)
        imageMask.append(contentsOf: Array(repeating: 0, count: suffixIDs.count))
        tokenIDs.insert(config.bosTokenID, at: 0)
        imageMask.insert(0, at: 0)

        if !tokenIDs.isEmpty {
            tokenIDs.removeLast()
            imageMask.removeLast()
        }

        let tokenArray = MLXArray(tokenIDs).expandedDimensions(axis: 0)
        let maskArray = MLXArray(imageMask).expandedDimensions(axis: 0)
        let imageInput = LMInput.ProcessedImage(
            pixels: processed.globalImages,
            frames: [THW(1, processed.cropWidth, processed.cropHeight)],
        )
        let videoInput = LMInput.ProcessedVideo(
            pixels: processed.localPatches,
            frames: [THW(1, processed.cropWidth, processed.cropHeight)],
        )
        return LMInput(
            text: .init(tokens: tokenArray, mask: maskArray), image: imageInput, video: videoInput)
    }

    private struct ProcessedImage {
        let localPatches: MLXArray
        let globalImages: MLXArray
        let cropWidth: Int
        let cropHeight: Int
        let imageTokenCount: Int
    }

    private func prepareUnlimited(_ input: UserInput, images: [UserInput.Image]) throws -> LMInput {
        let raw = Self.prompt(from: input)
        let tag = config.imageToken
        guard !tag.isEmpty else {
            throw VLMError.processing("Unlimited-OCR requires an image placeholder token.")
        }
        let pageLayout = config.unlimitedPageLayout
        var prompt = raw
        if !images.isEmpty {
            if pageLayout, !raw.contains("<PAGE>") {
                let pageImage = "<PAGE>" + tag
                let count = raw.components(separatedBy: tag).count - 1
                let markers = String(repeating: pageImage, count: count <= 1 ? images.count : 1)
                prompt =
                    count == 0 ? markers + raw : raw.replacingOccurrences(of: tag, with: markers)
            } else if !raw.contains(tag) {
                prompt = tag + raw
            }
        }
        let parts = prompt.components(separatedBy: tag)
        let placeholders = parts.count - 1
        let grouped = !pageLayout && placeholders == 1
        guard placeholders == images.count || grouped && !images.isEmpty else {
            throw VLMError.processing(
                "Unlimited-OCR image placeholders must match the supplied pages.")
        }
        let crop = input.additionalContext?["cropping"] as? Bool ?? config.unlimitedCropping ?? true
        let maxCrops =
            config.unlimitedMaxCrops
            ?? (pageLayout ? 16 : (config.maxPatches ?? 32))
        guard config.patchSize > 0, config.downsampleRatio > 0,
            config.imageMean.count == 3, config.imageStd.count == 3,
            (config.baseImageSize ?? 1024) > 0, (config.localImageSize ?? 640) > 0,
            maxCrops >= 2
        else {
            throw VLMError.processing("Invalid Unlimited-OCR image preprocessing configuration.")
        }
        var ids = [config.bosTokenID]
        var mask: [Int8] = [0]
        var frames: [THW] = []
        var global: [MLXArray] = []
        var patches: [MLXArray] = []
        func appendText(_ text: String) {
            let tokens = tokenizer.encode(text: text, addSpecialTokens: false)
            ids.append(contentsOf: tokens)
            mask.append(contentsOf: repeatElement(0, count: tokens.count))
        }
        for (index, image) in images.enumerated() {
            if !grouped || index == 0 { appendText(parts[index]) }
            let ci = try MediaProcessing.apply(image.asCIImage(), processing: input.processing)
                .toSRGB()
            let result = try processV1Image(ci, cropping: crop, maxPatches: maxCrops)
            let tiled = result.cropWidth > 1 || result.cropHeight > 1
            frames.append(THW(1, result.cropWidth, result.cropHeight))
            global.append(result.globalImages)
            if tiled { patches.append(result.localPatches) }
            ids.append(contentsOf: repeatElement(imageTokenID, count: result.imageTokenCount))
            mask.append(contentsOf: repeatElement(1, count: result.imageTokenCount))
        }
        appendText(parts[parts.count - 1])
        return LMInput(
            text: .init(
                tokens: MLXArray(ids).expandedDimensions(axis: 0),
                mask: MLXArray(mask).expandedDimensions(axis: 0)),
            image: global.isEmpty
                ? nil : .init(pixels: concatenated(global, axis: 0), frames: frames),
            video: patches.isEmpty
                ? nil : .init(pixels: concatenated(patches, axis: 0), frames: frames))
    }

    private func processV1Image(
        _ image: CIImage, cropping: Bool = true, maxPatches: Int = 9
    ) throws -> ProcessedImage {
        let baseSize = config.baseImageSize ?? 1024
        let imageSize = config.localImageSize ?? 640
        let size = image.extent.integral.size
        let cropWidth: Int
        let cropHeight: Int
        let patches: [CIImage]
        if !cropping || Int(size.width) <= imageSize && Int(size.height) <= imageSize {
            cropWidth = 1
            cropHeight = 1
            patches = []
        } else {
            let result = Self.dynamicPreprocess(
                image: image,
                minPatches: 2,
                maxPatches: maxPatches,
                imageSize: imageSize,
            )
            cropWidth = result.width
            cropHeight = result.height
            patches = result.patches
        }

        let global = Self.pad(
            image,
            to: baseSize,
            color: config.imageMeanTuple,
        )
        let globalTensor = Self.normalizedRGBArray(
            global,
            mean: config.imageMean,
            std: config.imageStd,
            normalize: config.normalize,
        )
        .asType(.bfloat16)
        let patchTensors = patches.map {
            Self.normalizedRGBArray(
                $0,
                mean: config.imageMean,
                std: config.imageStd,
                normalize: config.normalize,
            )
            .asType(.bfloat16)
        }
        let localPixels =
            patchTensors.isEmpty
            ? zeros([1, 3, imageSize, imageSize]).asType(.bfloat16)
            : stacked(patchTensors, axis: 0)
        let numQueries = Int(
            ceil(Float(imageSize / config.patchSize) / Float(config.downsampleRatio)))
        let baseQueries = Int(
            ceil(Float(baseSize / config.patchSize) / Float(config.downsampleRatio)))
        var tokenCount = (baseQueries + 1) * baseQueries + 1
        if cropWidth > 1 || cropHeight > 1 {
            tokenCount += (numQueries * cropWidth + 1) * (numQueries * cropHeight)
        }
        return ProcessedImage(
            localPatches: localPixels,
            globalImages: globalTensor.expandedDimensions(axis: 0),
            cropWidth: cropWidth,
            cropHeight: cropHeight,
            imageTokenCount: tokenCount,
        )
    }

    private func processV2Image(_ image: CIImage) throws -> ProcessedImage {
        let baseSize = config.baseImageSize ?? 1024
        let imageSize = config.localImageSize ?? 768
        let minPatches = config.minPatches ?? 1
        let maxPatches = config.maxPatches ?? 6
        let result = Self.dynamicPreprocess(
            image: image,
            minPatches: minPatches,
            maxPatches: maxPatches,
            imageSize: imageSize,
        )
        let global = Self.pad(
            image,
            to: baseSize,
            color: config.imageMeanTuple,
        )
        let globalTensor = Self.normalizedRGBArray(
            global,
            mean: config.imageMean,
            std: config.imageStd,
            normalize: config.normalize,
        )
        .asType(.bfloat16)
        let patchTensors = result.patches.map {
            Self.normalizedRGBArray(
                $0,
                mean: config.imageMean,
                std: config.imageStd,
                normalize: config.normalize,
            )
            .asType(.bfloat16)
        }
        let localPixels =
            patchTensors.isEmpty
            ? zeros([1, 3, imageSize, imageSize]).asType(.bfloat16)
            : stacked(patchTensors, axis: 0)
        let tokenCount = result.patches.count * 144 + 256 + 1
        return ProcessedImage(
            localPatches: localPixels,
            globalImages: globalTensor.expandedDimensions(axis: 0),
            cropWidth: result.width,
            cropHeight: result.height,
            imageTokenCount: tokenCount,
        )
    }

    private static func promptWithImageToken(_ prompt: String, imageToken: String) -> String {
        if prompt.contains(imageToken) {
            return prompt
        }
        return "\(imageToken)\n\(prompt)"
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

    private static func dynamicPreprocess(
        image: CIImage,
        minPatches: Int,
        maxPatches: Int,
        imageSize: Int,
    )
        -> (patches: [CIImage], width: Int, height: Int)
    {
        let extent = image.extent.integral
        let originalWidth = max(Int(extent.width.rounded()), 1)
        let originalHeight = max(Int(extent.height.rounded()), 1)
        let aspectRatio = Double(originalWidth) / Double(originalHeight)
        var targetRatios = [(Int, Int)]()
        for n in minPatches ... maxPatches {
            for width in 1 ... n {
                for height in 1 ... n {
                    let product = width * height
                    if product >= minPatches, product <= maxPatches {
                        targetRatios.append((width, height))
                    }
                }
            }
        }
        targetRatios.sort { lhs, rhs in
            let lp = lhs.0 * lhs.1
            let rp = rhs.0 * rhs.1
            if lp == rp {
                return lhs.0 < rhs.0
            }
            return lp < rp
        }
        let area = originalWidth * originalHeight
        var best = (1, 1)
        var bestDiff = Double.infinity
        for ratio in targetRatios {
            let target = Double(ratio.0) / Double(ratio.1)
            let diff = abs(aspectRatio - target)
            if diff < bestDiff {
                bestDiff = diff
                best = ratio
            } else if diff == bestDiff,
                area > imageSize * imageSize * ratio.0 * ratio.1 / 2
            {
                best = ratio
            }
        }

        let targetWidth = imageSize * best.0
        let targetHeight = imageSize * best.1
        let resized = MediaProcessing.zeroOrigin(image).resampled(
            to: CGSize(width: targetWidth, height: targetHeight),
            method: .bicubic,
        )
        var patches = [CIImage]()
        for index in 0 ..< (best.0 * best.1) {
            let col = index % best.0
            let row = index / best.0
            let y = targetHeight - (row + 1) * imageSize
            let rect = CGRect(x: col * imageSize, y: y, width: imageSize, height: imageSize)
            patches.append(
                resized
                    .cropped(to: rect)
                    .transformed(by: CGAffineTransform(translationX: -rect.minX, y: -rect.minY)),
            )
        }
        return (patches, best.0, best.1)
    }

    private static func pad(
        _ image: CIImage,
        to side: Int,
        color: (CGFloat, CGFloat, CGFloat),
    )
        -> CIImage
    {
        let canonical = MediaProcessing.zeroOrigin(image)
        let extent = canonical.extent.integral
        let scale = min(CGFloat(side) / extent.width, CGFloat(side) / extent.height)
        let resizedWidth = max(Int((extent.width * scale).rounded()), 1)
        let resizedHeight = max(Int((extent.height * scale).rounded()), 1)
        let resized = canonical.resampled(
            to: CGSize(width: resizedWidth, height: resizedHeight),
            method: .bicubic,
        )
        let x = CGFloat(side - resizedWidth) * 0.5
        let y = CGFloat(side - resizedHeight) * 0.5
        let centered = resized.transformed(by: CGAffineTransform(translationX: x, y: y))
        let background = CIImage(
            color: CIColor(red: color.0, green: color.1, blue: color.2, alpha: 1),
        )
        .cropped(to: CGRect(x: 0, y: 0, width: side, height: side))
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

private enum DeepSeekOCRLanguage {
    final class Attention: Module {
        let config: DeepSeekOCRConfiguration.TextConfiguration
        let headDim: Int
        let scale: Float
        let layerIndex: Int

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE
        @ModuleInfo(key: "dash_router") var dashRouter: UnlimitedOCRDASHRouter?

        init(_ config: DeepSeekOCRConfiguration.TextConfiguration, layerIndex: Int, dash: Bool) {
            self.config = config
            self.layerIndex = layerIndex
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
            _rotaryEmbedding.wrappedValue = RoPE(
                dimensions: headDim,
                traditional: config.ropeTraditional,
                base: config.ropeTheta,
            )
            if dash {
                _dashRouter.wrappedValue = UnlimitedOCRDASHRouter(
                    hiddenSize: config.hiddenSize, dimension: headDim, epsilon: config.rmsNormEps)
            }
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
            dash: UnlimitedOCRDASH? = nil,
            dashContext: UnlimitedOCRDASHContext? = nil,
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

            let output: MLXArray
            if let dash, let dashContext, let dashRouter {
                output = dash.attend(
                    x: x, queries: queries, keys: keys, values: values, scale: scale,
                    router: dashRouter, layer: layerIndex, cache: cache as! UnlimitedOCRDASHCache,
                    context: dashContext)
            } else {
                let offset = cache?.offset ?? 0
                queries = rotaryEmbedding(queries, offset: offset)
                keys = rotaryEmbedding(keys, offset: offset)
                output = attentionWithCacheUpdate(
                    queries: queries, keys: keys, values: values, cache: cache,
                    scale: scale, mask: mask)
            }
            return oProj(output.transposed(0, 2, 1, 3).reshaped(batchSize, sequenceLength, -1))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        init(
            _ config: DeepSeekOCRConfiguration.TextConfiguration,
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
        let config: DeepSeekOCRConfiguration.TextConfiguration
        let preciseSoftmax: Bool
        @ModuleInfo var weight: MLXArray

        init(_ config: DeepSeekOCRConfiguration.TextConfiguration, preciseSoftmax: Bool) {
            self.config = config
            self.preciseSoftmax = preciseSoftmax
            _weight.wrappedValue = zeros([config.nRoutedExperts ?? 1, config.hiddenSize])
        }

        func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
            let scoresInput = x.matmul(weight.T)
            let scores: MLXArray =
                if config.scoringFunction == "sigmoid" {
                    sigmoid(scoresInput)
                } else {
                    softmax(scoresInput, axis: -1, precise: preciseSoftmax)
                }
            let k = config.numExpertsPerToken
            let indices = argPartition(scores, kth: -k, axis: -1)[.ellipsis, (-k)...]
            var selected = takeAlong(scores, indices, axis: -1)
            selected = selected * config.routedScalingFactor
            return (indices, selected)
        }
    }

    final class MoE: Module, UnaryLayer {
        let config: DeepSeekOCRConfiguration.TextConfiguration
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
        @ModuleInfo var gate: MoEGate
        @ModuleInfo(key: "shared_experts") var sharedExperts: MLP?

        init(_ config: DeepSeekOCRConfiguration.TextConfiguration, preciseSoftmax: Bool) {
            self.config = config
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.nRoutedExperts ?? 1,
            )
            _gate.wrappedValue = MoEGate(config, preciseSoftmax: preciseSoftmax)
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

        init(
            _ config: DeepSeekOCRConfiguration.TextConfiguration, layerIndex: Int,
            dash: Bool, preciseRouting: Bool
        ) {
            _attention.wrappedValue = Attention(config, layerIndex: layerIndex, dash: dash)
            if config.nRoutedExperts != nil,
                layerIndex >= config.firstKDenseReplace,
                layerIndex % config.moeLayerFrequency == 0
            {
                mlp = MoE(config, preciseSoftmax: preciseRouting)
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
            dash: UnlimitedOCRDASH? = nil,
            dashContext: UnlimitedOCRDASHContext? = nil,
        )
            -> MLXArray
        {
            let r = attention(
                inputLayerNorm(x), mask: mask, cache: cache, dash: dash, dashContext: dashContext)
            let h = x + r
            return h + mlp(postAttentionLayerNorm(h))
        }
    }

    final class TextModel: Module {
        let config: DeepSeekOCRConfiguration.TextConfiguration
        let layers: [DecoderLayer]
        let dash: UnlimitedOCRDASH?
        let unlimited: Bool
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo var norm: RMSNorm

        init(
            _ config: DeepSeekOCRConfiguration.TextConfiguration, unlimited: Bool,
            dash: UnlimitedOCRDASHParameters?
        ) {
            self.config = config
            self.unlimited = unlimited
            self.dash = dash.map {
                UnlimitedOCRDASH(
                    $0, headDim: config.hiddenSize / config.attentionHeads, base: config.ropeTheta,
                    maxPositions: config.maximumPositions, scaling: config.ropeScaling)
            }
            layers = (0 ..< config.hiddenLayers).map {
                DecoderLayer(config, layerIndex: $0, dash: dash != nil, preciseRouting: unlimited)
            }
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]?,
            inputEmbedding: MLXArray? = nil,
        )
            -> MLXArray
        {
            let h = inputEmbedding ?? embedTokens(inputs!)
            let cache = cache ?? dash.map { _ in layers.map { _ in UnlimitedOCRDASHCache() } }
            if unlimited, let cache, let first = cache.first {
                if let dash, let dashCache = first as? UnlimitedOCRDASHCache,
                    dashCache.prompt == nil
                {
                    dashCache.startPrompt(inputs!.asArray(Int32.self), parameters: dash.parameters)
                } else if let ring = first as? UnlimitedOCRReferenceCache, ring.promptLength == nil
                {
                    for case let layer as UnlimitedOCRReferenceCache in cache {
                        layer.promptLength = first.offset + h.dim(1)
                    }
                }
                let promptEnd =
                    (first as? UnlimitedOCRDASHCache)?.prompt?.length
                    ?? (first as? UnlimitedOCRReferenceCache)?.promptLength ?? 0
                // Every generated query needs its own ring contents and page state.
                if h.dim(1) > 1, first.offset >= promptEnd {
                    return concatenated(
                        (0 ..< h.dim(1)).map {
                            forward(
                                h[0..., $0 ..< ($0 + 1), 0...],
                                tokens: inputs?[0..., $0 ..< ($0 + 1)],
                                cache: cache)
                        }, axis: 1)
                }
            }
            return forward(h, tokens: inputs, cache: cache)
        }

        private func forward(_ input: MLXArray, tokens: MLXArray?, cache: [KVCache]?) -> MLXArray {
            var h = input
            if unlimited, let type = (cache?.first as? KVCacheSimple)?.keys?.dtype {
                let common = (MLXArray(0, dtype: h.dtype) + MLXArray(0, dtype: type)).dtype
                h = h.asType(common)
            }
            let context = dash.map {
                precondition(h.dim(0) == 1, "Unlimited-OCR-2 expects one document per request")
                return $0.makeContext(
                    cache: cache!.first as! UnlimitedOCRDASHCache, tokens: tokens,
                    length: h.dim(1), dtype: h.dtype)
            }
            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                context == nil ? createAttentionMask(h: h, cache: cache?.first) : .none
            for (index, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[index], dash: dash, dashContext: context)
            }
            return norm(h)
        }
    }

    final class LanguageModel: Module {
        let config: DeepSeekOCRConfiguration.TextConfiguration
        @ModuleInfo var model: TextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear

        init(
            _ config: DeepSeekOCRConfiguration.TextConfiguration,
            unlimited: Bool = false, dash: UnlimitedOCRDASHParameters? = nil
        ) {
            self.config = config
            _model.wrappedValue = TextModel(config, unlimited: unlimited, dash: dash)
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

// MARK: - SAM

private enum DeepSeekOCRSAM {
    static func checkConvWeight(_ array: MLXArray) -> Bool {
        guard array.ndim == 4 else {
            return false
        }
        let shape = array.shape
        return shape[0] >= shape[1] && shape[0] >= shape[2] && shape[1] == shape[2]
    }

    static func absPosition(_ absPos: MLXArray, targetSize: Int) -> MLXArray {
        guard absPos.dim(1) != targetSize else {
            return absPos
        }
        let dtype = absPos.dtype
        let old = absPos.transposed(0, 3, 1, 2).asType(.float32)
        return bicubicInterpolate(
            old,
            size: (targetSize, targetSize),
            antialias: true,
        )
        .asType(dtype)
        .transposed(0, 2, 3, 1)
    }

    static func windowPartition(_ x: MLXArray, windowSize: Int) -> (MLXArray, Int, Int) {
        let (batch, height, width, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let padHeight = (windowSize - height % windowSize) % windowSize
        let padWidth = (windowSize - width % windowSize) % windowSize
        var paddedX = x
        if padHeight > 0 || padWidth > 0 {
            paddedX = padded(
                paddedX,
                widths: [0, .init((0, padHeight)), .init((0, padWidth)), 0],
            )
        }
        let paddedHeight = height + padHeight
        let paddedWidth = width + padWidth
        let windows =
            paddedX
            .reshaped(
                batch,
                paddedHeight / windowSize,
                windowSize,
                paddedWidth / windowSize,
                windowSize,
                channels,
            )
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(-1, windowSize, windowSize, channels)
        return (windows, paddedHeight, paddedWidth)
    }

    static func windowUnpartition(
        _ windows: MLXArray,
        windowSize: Int,
        paddedHeight: Int,
        paddedWidth: Int,
        height: Int,
        width: Int,
    )
        -> MLXArray
    {
        let batch = windows.dim(0) / (paddedHeight * paddedWidth / windowSize / windowSize)
        var x =
            windows
            .reshaped(
                batch,
                paddedHeight / windowSize,
                paddedWidth / windowSize,
                windowSize,
                windowSize,
                -1,
            )
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(batch, paddedHeight, paddedWidth, -1)
        if paddedHeight > height || paddedWidth > width {
            x = x[0..., 0 ..< height, 0 ..< width, 0...]
        }
        return x
    }

    static func relativePosition(qSize: Int, kSize: Int, relPos: MLXArray) -> MLXArray {
        let maxRelDistance = 2 * max(qSize, kSize) - 1
        let resized: MLXArray
        if relPos.dim(0) != maxRelDistance {
            let dtype = relPos.dtype
            let sourceLength = relPos.dim(0)
            let indices =
                MLXArray(0 ..< maxRelDistance).asType(.float32)
                * (Float(sourceLength) / Float(maxRelDistance))
            let floorIndex = floor(indices).asType(.int32)
            let ceilIndex = minimum(floorIndex + 1, MLXArray(sourceLength - 1))
            let weight = indices - floorIndex.asType(.float32)
            let left = relPos.take(floorIndex, axis: 0)
            let right = relPos.take(ceilIndex, axis: 0)
            resized = (left * (1 - weight[0..., .newAxis]) + right * weight[0..., .newAxis])
                .asType(dtype)
        } else {
            resized = relPos
        }
        let qCoords =
            MLXArray(0 ..< qSize).asType(.float32)[0..., .newAxis]
            * max(Float(kSize) / Float(qSize), 1)
        let kCoords =
            MLXArray(0 ..< kSize).asType(.float32)[.newAxis, 0...]
            * max(Float(qSize) / Float(kSize), 1)
        let relative =
            (qCoords - kCoords)
            + Float(kSize - 1) * max(Float(qSize) / Float(kSize), 1)
        return resized.take(relative.asType(.int32), axis: 0)
    }

    static func decomposedRelativePosition(
        q: MLXArray,
        relPosH: MLXArray,
        relPosW: MLXArray,
        height: Int,
        width: Int,
    )
        -> (MLXArray, MLXArray)
    {
        let rh = relativePosition(qSize: height, kSize: height, relPos: relPosH)
        let rw = relativePosition(qSize: width, kSize: width, relPos: relPosW)
        let batch = q.dim(0)
        let dim = q.dim(2)
        let reshapedQ = q.reshaped(batch, height, width, dim)
        var relH = einsum("bhwc,hkc->bhwk", reshapedQ, rh)
        var relW = einsum("bhwc,wkc->bhwk", reshapedQ, rw)
        relH = relH[.ellipsis, .newAxis].reshaped(batch, height * width, height, 1)
        relW = relW[.ellipsis, .newAxis, 0...].reshaped(batch, height * width, 1, width)
        return (relH, relW)
    }

    final class MLPBlock: Module, UnaryLayer {
        @ModuleInfo var lin1: Linear
        @ModuleInfo var lin2: Linear
        let activation = GELU(approximation: .precise)

        init(embeddingDimension: Int, mlpDimension: Int) {
            _lin1.wrappedValue = Linear(embeddingDimension, mlpDimension, bias: true)
            _lin2.wrappedValue = Linear(mlpDimension, embeddingDimension, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            lin2(activation(lin1(x)))
        }
    }

    final class Attention: Module {
        let heads: Int
        let scale: Float
        let useRelativePosition: Bool
        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear
        @ModuleInfo(key: "rel_pos_h") var relPosH: MLXArray
        @ModuleInfo(key: "rel_pos_w") var relPosW: MLXArray

        init(
            dimension: Int,
            heads: Int,
            useRelativePosition: Bool,
            inputSize: Int,
        ) {
            self.heads = heads
            let headDimension = dimension / heads
            scale = pow(Float(headDimension), -0.5)
            self.useRelativePosition = useRelativePosition
            _qkv.wrappedValue = Linear(dimension, dimension * 3, bias: true)
            _proj.wrappedValue = Linear(dimension, dimension, bias: true)
            _relPosH.wrappedValue = zeros([2 * inputSize - 1, headDimension])
            _relPosW.wrappedValue = zeros([2 * inputSize - 1, headDimension])
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (batch, height, width, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
            let qkvStates = qkv(x)
                .reshaped(batch, height * width, 3, heads, channels / heads)
                .transposed(2, 0, 3, 1, 4)
            var q = qkvStates[0].reshaped(batch * heads, height * width, -1)
            var k = qkvStates[1].reshaped(batch * heads, height * width, -1)
            var v = qkvStates[2].reshaped(batch * heads, height * width, -1)
            var attentionBias: MLXFast.ScaledDotProductAttentionMaskMode = .none
            if useRelativePosition {
                let (relH, relW) = decomposedRelativePosition(
                    q: q,
                    relPosH: relPosH,
                    relPosW: relPosW,
                    height: height,
                    width: width,
                )
                let bias = (relH + relW)
                    .reshaped(batch, heads, height * width, height * width)
                attentionBias = .array(bias)
            }
            q = q.reshaped(batch, heads, height * width, -1)
            k = k.reshaped(batch, heads, height * width, -1)
            v = v.reshaped(batch, heads, height * width, -1)
            let output = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: attentionBias,
            )
            .reshaped(batch, heads, height, width, -1)
            .transposed(0, 2, 3, 1, 4)
            .reshaped(batch, height, width, -1)
            return proj(output)
        }
    }

    final class Block: Module {
        let windowSize: Int
        @ModuleInfo var norm1: LayerNorm
        @ModuleInfo var attn: Attention
        @ModuleInfo var norm2: LayerNorm
        @ModuleInfo var mlp: MLPBlock

        init(
            dimension: Int,
            heads: Int,
            mlpRatio: Float,
            useRelativePosition: Bool,
            windowSize: Int,
            inputSize: Int,
        ) {
            self.windowSize = windowSize
            _norm1.wrappedValue = LayerNorm(dimensions: dimension, eps: 1e-6)
            _attn.wrappedValue = Attention(
                dimension: dimension,
                heads: heads,
                useRelativePosition: useRelativePosition,
                inputSize: windowSize == 0 ? inputSize : windowSize,
            )
            _norm2.wrappedValue = LayerNorm(dimensions: dimension, eps: 1e-6)
            _mlp.wrappedValue = MLPBlock(
                embeddingDimension: dimension,
                mlpDimension: Int(Float(dimension) * mlpRatio),
            )
        }

        func callAsFunction(_ input: MLXArray) -> MLXArray {
            var x = norm1(input)
            let originalHeight = x.dim(1)
            let originalWidth = x.dim(2)
            var paddedHeight = originalHeight
            var paddedWidth = originalWidth
            if windowSize > 0 {
                let partitioned = windowPartition(x, windowSize: windowSize)
                x = partitioned.0
                paddedHeight = partitioned.1
                paddedWidth = partitioned.2
            }
            x = attn(x)
            if windowSize > 0 {
                x = windowUnpartition(
                    x,
                    windowSize: windowSize,
                    paddedHeight: paddedHeight,
                    paddedWidth: paddedWidth,
                    height: originalHeight,
                    width: originalWidth,
                )
            }
            let h = input + x
            return h + mlp(norm2(h))
        }
    }

    final class PatchEmbedding: Module, UnaryLayer {
        @ModuleInfo var proj: Conv2d

        init(patchSize: Int, inputChannels: Int, embedDimension: Int) {
            _proj.wrappedValue = Conv2d(
                inputChannels: inputChannels,
                outputChannels: embedDimension,
                kernelSize: IntOrPair(patchSize),
                stride: IntOrPair(patchSize),
                bias: true,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            proj(x)
        }
    }

    final class Encoder: Module, UnaryLayer {
        let config: DeepSeekOCRConfiguration.SAMConfiguration
        let blocks: [Block]
        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbedding
        @ModuleInfo(key: "pos_embed") var posEmbed: MLXArray
        @ModuleInfo var neck: [Module]
        @ModuleInfo(key: "net_2") var net2: Conv2d
        @ModuleInfo(key: "net_3") var net3: Conv2d

        init(_ config: DeepSeekOCRConfiguration.SAMConfiguration) {
            self.config = config
            _patchEmbed.wrappedValue = PatchEmbedding(
                patchSize: config.patchSize,
                inputChannels: 3,
                embedDimension: config.width,
            )
            _posEmbed.wrappedValue = zeros([
                1,
                config.imageSize / config.patchSize,
                config.imageSize / config.patchSize,
                config.width,
            ])
            let inputSize = config.imageSize / config.patchSize
            blocks = (0 ..< config.layers).map { index in
                Block(
                    dimension: config.width,
                    heads: config.heads,
                    mlpRatio: 4.0,
                    useRelativePosition: true,
                    windowSize: config.globalAttentionIndexes.contains(index)
                        ? 0 : config.windowSize,
                    inputSize: inputSize,
                )
            }
            _neck.wrappedValue = [
                Conv2d(
                    inputChannels: config.width,
                    outputChannels: 256,
                    kernelSize: 1,
                    bias: false,
                ),
                LayerNorm(dimensions: 256, eps: 1e-6),
                Conv2d(
                    inputChannels: 256,
                    outputChannels: 256,
                    kernelSize: 3,
                    padding: 1,
                    bias: false,
                ),
                LayerNorm(dimensions: 256, eps: 1e-6),
            ]
            _net2.wrappedValue = Conv2d(
                inputChannels: 256,
                outputChannels: 512,
                kernelSize: 3,
                stride: 2,
                padding: 1,
                bias: false,
            )
            _net3.wrappedValue = Conv2d(
                inputChannels: 512,
                outputChannels: config.finalOutputChannels,
                kernelSize: 3,
                stride: 2,
                padding: 1,
                bias: false,
            )
        }

        func callAsFunction(_ input: MLXArray) -> MLXArray {
            var x = patchEmbed(input)
            x = x + absPosition(posEmbed, targetSize: x.dim(1))
            for block in blocks {
                x = block(x)
            }
            for layer in neck {
                if let conv = layer as? Conv2d {
                    x = conv(x)
                } else if let norm = layer as? LayerNorm {
                    x = norm(x)
                }
            }
            x = net2(x)
            return net3(x)
        }
    }
}

// MARK: - Vision v1

private enum DeepSeekOCRVision {
    final class Attention: Module {
        let heads: Int
        let scale: Float
        @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: DeepSeekOCRConfiguration.VisionConfiguration) {
            heads = config.attentionHeads
            let headDimension = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDimension), -0.5)
            _qkvProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * 3, bias: true)
            _outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (batch, sequence, channels) = (x.dim(0), x.dim(1), x.dim(2))
            let qkv = qkvProj(x)
            let chunks = split(qkv, parts: 3, axis: -1)
            let q = chunks[0].reshaped(batch, sequence, heads, -1).transposed(0, 2, 1, 3)
            let k = chunks[1].reshaped(batch, sequence, heads, -1).transposed(0, 2, 1, 3)
            let v = chunks[2].reshaped(batch, sequence, heads, -1).transposed(0, 2, 1, 3)
            return outProj(
                MLXFast.scaledDotProductAttention(
                    queries: q,
                    keys: k,
                    values: v,
                    scale: scale,
                    mask: .none,
                )
                .transposed(0, 2, 1, 3)
                .reshaped(batch, sequence, channels),
            )
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let activation = GELU(approximation: .precise)

        init(_ config: DeepSeekOCRConfiguration.VisionConfiguration) {
            _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    final class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

        init(_ config: DeepSeekOCRConfiguration.VisionConfiguration) {
            _selfAttention.wrappedValue = Attention(config)
            _layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
            _mlp.wrappedValue = MLP(config)
            _layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = x + selfAttention(layerNorm1(x))
            return h + mlp(layerNorm2(h))
        }
    }

    final class Embeddings: Module {
        let config: DeepSeekOCRConfiguration.VisionConfiguration
        @ModuleInfo(key: "class_embedding") var classEmbedding: MLXArray
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        init(_ config: DeepSeekOCRConfiguration.VisionConfiguration) {
            self.config = config
            let patches =
                (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize)
            _classEmbedding.wrappedValue = MLXRandom.normal([config.hiddenSize])
            _patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.channels,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: false,
            )
            _positionEmbedding.wrappedValue = Embedding(
                embeddingCount: patches + 1,
                dimensions: config.hiddenSize,
            )
        }

        private func absolutePosition(_ absPos: MLXArray, targetSize: Int) -> MLXArray {
            let dimension = absPos.dim(-1)
            let squeezed = absPos.squeezed(axis: 0)
            let clsToken = squeezed[..<1, 0...]
            var old = squeezed[1..., 0...]
            let sourceSize = Int(sqrt(Double(old.dim(0))))
            let target2D = Int(sqrt(Double(targetSize - 1)))
            guard sourceSize != target2D else {
                return absPos
            }
            old = old.reshaped(1, sourceSize, sourceSize, dimension)
                .transposed(0, 3, 1, 2)
                .asType(.float32)
            let interpolated = bicubicInterpolate(old, size: (target2D, target2D))
                .asType(absPos.dtype)
                .transposed(0, 2, 3, 1)
                .reshaped(target2D * target2D, dimension)
            return concatenated([clsToken, interpolated], axis: 0)
                .reshaped(1, targetSize, dimension)
        }

        func callAsFunction(_ x: MLXArray, patchEmbeds: MLXArray?) -> MLXArray {
            let batch = x.dim(0)
            let targetDtype = positionEmbedding.weight.dtype
            let patches = patchEmbeds ?? patchEmbedding(x)
            let patchTokens = patches.flattened(start: 1, end: 2)
            let classTokens = broadcast(
                classEmbedding.asType(targetDtype),
                to: [batch, 1, config.hiddenSize],
            )
            let tokens = concatenated([classTokens, patchTokens], axis: 1)
            let positions = MLXArray(0 ..< positionEmbedding.weight.dim(0))
                .expandedDimensions(axis: 0)
            return tokens
                + absolutePosition(positionEmbedding(positions), targetSize: tokens.dim(1))
                .asType(targetDtype)
        }
    }

    final class Transformer: Module, UnaryLayer {
        let layers: [EncoderLayer]

        init(_ config: DeepSeekOCRConfiguration.VisionConfiguration) {
            layers = (0 ..< config.hiddenLayers).map { _ in EncoderLayer(config) }
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var x = x
            for layer in layers {
                x = layer(x)
            }
            return x
        }
    }

    final class VisionModel: Module {
        @ModuleInfo var embeddings: Embeddings
        @ModuleInfo(key: "pre_layrnorm") var preLayerNorm: LayerNorm
        @ModuleInfo var transformer: Transformer

        init(_ config: DeepSeekOCRConfiguration.VisionConfiguration) {
            _embeddings.wrappedValue = Embeddings(config)
            _preLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
            _transformer.wrappedValue = Transformer(config)
        }

        func callAsFunction(_ x: MLXArray, patchEmbeds: MLXArray?) -> MLXArray {
            transformer(preLayerNorm(embeddings(x, patchEmbeds: patchEmbeds)))
        }
    }
}

// MARK: - Vision v2

private enum DeepSeekOCR2Vision {
    final class Qwen2RMSNorm: Module, UnaryLayer {
        @ModuleInfo var weight: MLXArray
        let eps: Float

        init(dimensions: Int, eps: Float) {
            self.eps = eps
            _weight.wrappedValue = ones([dimensions])
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let dtype = x.dtype
            let floatX = x.asType(.float32)
            let variance = mean(floatX * floatX, axis: -1, keepDims: true)
            return weight * (floatX * rsqrt(variance + eps)).asType(dtype)
        }
    }

    final class Qwen2RotaryEmbedding: Module {
        let dimensions: Int
        let base: Float

        init(dimensions: Int, base: Float) {
            self.dimensions = dimensions
            self.base = base
        }

        func callAsFunction(_ x: MLXArray, positions: MLXArray) -> (MLXArray, MLXArray) {
            let invFrequency =
                1
                / pow(
                    MLXArray(base),
                    MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
                        / Float(dimensions),
                )
            let frequencies =
                positions[0..., 0..., .newAxis].asType(.float32)
                * invFrequency[.newAxis, .newAxis, 0...]
            let embedding = concatenated([frequencies, frequencies], axis: -1)
            return (cos(embedding).asType(x.dtype), sin(embedding).asType(x.dtype))
        }
    }

    final class Qwen2Attention: Module {
        let config: DeepSeekOCRConfiguration.Qwen2Configuration
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let kvGroups: Int
        let scale: Float
        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: Qwen2RotaryEmbedding

        init(_ config: DeepSeekOCRConfiguration.Qwen2Configuration) {
            self.config = config
            heads = config.heads
            kvHeads = config.kvHeads
            headDim = config.dim / config.heads
            kvGroups = config.heads / config.kvHeads
            scale = pow(Float(headDim), -0.5)
            _qProj.wrappedValue = Linear(config.dim, config.heads * headDim, bias: true)
            _kProj.wrappedValue = Linear(config.dim, config.kvHeads * headDim, bias: true)
            _vProj.wrappedValue = Linear(config.dim, config.kvHeads * headDim, bias: true)
            _oProj.wrappedValue = Linear(config.heads * headDim, config.dim, bias: false)
            _rotaryEmbedding.wrappedValue = Qwen2RotaryEmbedding(
                dimensions: headDim,
                base: config.ropeTheta,
            )
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
            positionIDs: MLXArray,
        )
            -> MLXArray
        {
            let (batch, sequence) = (x.dim(0), x.dim(1))
            var q = qProj(x).reshaped(batch, sequence, heads, headDim).transposed(0, 2, 1, 3)
            var k = kProj(x).reshaped(batch, sequence, kvHeads, headDim).transposed(0, 2, 1, 3)
            var v = vProj(x).reshaped(batch, sequence, kvHeads, headDim).transposed(0, 2, 1, 3)
            let (cosines, sines) = rotaryEmbedding(v, positions: positionIDs)
            let cosinesExpanded = cosines[0..., .newAxis, 0..., 0...]
            let sinesExpanded = sines[0..., .newAxis, 0..., 0...]
            q = (q * cosinesExpanded) + (RotaryEmbeddingOps.rotateHalf(q) * sinesExpanded)
            k = (k * cosinesExpanded) + (RotaryEmbeddingOps.rotateHalf(k) * sinesExpanded)
            if kvGroups > 1 {
                k = repeated(k, count: kvGroups, axis: 1)
                v = repeated(v, count: kvGroups, axis: 1)
            }
            return oProj(
                MLXFast.scaledDotProductAttention(
                    queries: q,
                    keys: k,
                    values: v,
                    scale: scale,
                    mask: attentionMask,
                )
                .transposed(0, 2, 1, 3)
                .reshaped(batch, sequence, -1),
            )
        }
    }

    final class Qwen2MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        init(_ config: DeepSeekOCRConfiguration.Qwen2Configuration) {
            _gateProj.wrappedValue = Linear(config.dim, config.intermediateSize, bias: false)
            _upProj.wrappedValue = Linear(config.dim, config.intermediateSize, bias: false)
            _downProj.wrappedValue = Linear(config.intermediateSize, config.dim, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    final class Qwen2Layer: Module {
        @ModuleInfo(key: "self_attn") var selfAttention: Qwen2Attention
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: Qwen2RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: Qwen2RMSNorm
        @ModuleInfo var mlp: Qwen2MLP

        init(_ config: DeepSeekOCRConfiguration.Qwen2Configuration) {
            _selfAttention.wrappedValue = Qwen2Attention(config)
            _inputLayerNorm.wrappedValue = Qwen2RMSNorm(
                dimensions: config.dim,
                eps: config.rmsNormEps,
            )
            _postAttentionLayerNorm.wrappedValue = Qwen2RMSNorm(
                dimensions: config.dim,
                eps: config.rmsNormEps,
            )
            _mlp.wrappedValue = Qwen2MLP(config)
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
            positionIDs: MLXArray,
        )
            -> MLXArray
        {
            var h =
                x
                + selfAttention(
                    inputLayerNorm(x),
                    attentionMask: attentionMask,
                    positionIDs: positionIDs,
                )
            h = h + mlp(postAttentionLayerNorm(h))
            return h
        }
    }

    final class Qwen2Encoder: Module, UnaryLayer {
        let config: DeepSeekOCRConfiguration.Qwen2Configuration
        let layers: [Qwen2Layer]
        @ModuleInfo(key: "query_1024") var query1024: MLXArray
        @ModuleInfo(key: "query_768") var query768: MLXArray
        @ModuleInfo var norm: Qwen2RMSNorm

        init(_ config: DeepSeekOCRConfiguration.Qwen2Configuration) {
            self.config = config
            _query1024.wrappedValue = zeros([256, config.dim])
            _query768.wrappedValue = zeros([144, config.dim])
            layers = (0 ..< config.layers).map { _ in Qwen2Layer(config) }
            _norm.wrappedValue = Qwen2RMSNorm(dimensions: config.dim, eps: config.rmsNormEps)
        }

        func callAsFunction(_ samFeatures: MLXArray) -> MLXArray {
            let batch = samFeatures.dim(0)
            let imageTokens = samFeatures.reshaped(batch, -1, config.dim)
            let tokenCount = imageTokens.dim(1)
            let query: MLXArray =
                if tokenCount == 144 {
                    query768
                } else {
                    query1024
                }
            let numQueries = query.dim(0)
            let queries = broadcast(
                query[.newAxis, 0..., 0...], to: [batch, numQueries, config.dim])
            var hidden = concatenated([imageTokens, queries], axis: 1)
            let sequence = hidden.dim(1)
            let attentionMask = Self.attentionMask(
                imageTokens: tokenCount,
                queries: numQueries,
                dtype: hidden.dtype,
            )
            let positionIDs = broadcast(
                MLXArray(0 ..< sequence)[.newAxis, 0...], to: [batch, sequence])
            for layer in layers {
                hidden = layer(
                    hidden, attentionMask: .array(attentionMask), positionIDs: positionIDs)
            }
            hidden = norm(hidden)
            return hidden[0..., (-numQueries)..., 0...]
        }

        private static func attentionMask(
            imageTokens: Int,
            queries: Int,
            dtype: DType,
        )
            -> MLXArray
        {
            let sequence = imageTokens + queries
            var values = Array(repeating: Float32(-1e9), count: sequence * sequence)
            for row in 0 ..< imageTokens {
                for col in 0 ..< imageTokens {
                    values[row * sequence + col] = 0
                }
            }
            for row in imageTokens ..< sequence {
                for col in 0 ..< imageTokens {
                    values[row * sequence + col] = 0
                }
                for col in imageTokens ... row {
                    values[row * sequence + col] = 0
                }
            }
            return MLXArray(values, [1, 1, sequence, sequence]).asType(dtype)
        }
    }

    final class VisionModel: Module {
        @ModuleInfo(key: "qwen2_encoder") var qwen2Encoder: Qwen2Encoder

        init(_ config: DeepSeekOCRConfiguration.VisionConfiguration) {
            _qwen2Encoder.wrappedValue = Qwen2Encoder(config.qwenConfiguration)
        }

        func callAsFunction(_ samFeatures: MLXArray) -> MLXArray {
            qwen2Encoder(samFeatures)
        }
    }
}

// MARK: - Projector

private final class DeepSeekOCRProjector: Module, UnaryLayer {
    @ModuleInfo var layers: Linear

    init(_ config: DeepSeekOCRConfiguration.ProjectorConfiguration) {
        _layers.wrappedValue = Linear(
            config.inputDimension,
            config.embeddingDimension,
            bias: true,
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layers(x)
    }
}

// MARK: - Model

public final class DeepSeekOCR: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_model") private var visionModel: DeepSeekOCRVision.VisionModel
    @ModuleInfo(key: "sam_model") private var samModel: DeepSeekOCRSAM.Encoder
    @ModuleInfo(key: "language_model") private var languageModel: DeepSeekOCRLanguage.LanguageModel
    @ModuleInfo private var projector: DeepSeekOCRProjector
    @ModuleInfo(key: "image_newline") private var imageNewline: MLXArray
    @ModuleInfo(key: "view_separator") private var viewSeparator: MLXArray

    public let config: DeepSeekOCRConfiguration
    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: DeepSeekOCRConfiguration) {
        self.config = config
        _visionModel.wrappedValue = DeepSeekOCRVision.VisionModel(config.visionConfiguration)
        _samModel.wrappedValue = DeepSeekOCRSAM.Encoder(
            config.visionConfiguration.samConfiguration,
        )
        _languageModel.wrappedValue = DeepSeekOCRLanguage.LanguageModel(
            config.textConfiguration, unlimited: config.isUnlimitedOCR, dash: config.dashAttention,
        )
        _projector.wrappedValue = DeepSeekOCRProjector(config.projectorConfiguration)
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

    private func visualFeatures(
        localPatches: MLXArray?,
        globalImages: MLXArray?,
        cropFrames: [THW]?,
    )
        -> MLXArray?
    {
        guard let globalImages else {
            return nil
        }
        let globalHWC = globalImages.transposed(0, 2, 3, 1)
        let globalSam = samModel(globalHWC)
        let globalVision = visionModel(globalHWC, patchEmbeds: globalSam)
        var globalFeatures = concatenated(
            [
                globalVision[0..., 1..., 0...],
                globalSam.flattened(start: 1, end: 2),
            ],
            axis: -1,
        )
        globalFeatures = projector(globalFeatures)[0]
        let hidden = globalFeatures.dim(-1)
        let globalSide = Int(sqrt(Double(globalFeatures.dim(0))))
        var globalGrid = globalFeatures.reshaped(globalSide, globalSide, hidden)
        globalGrid = concatenated(
            [
                globalGrid,
                broadcast(imageNewline[.newAxis, .newAxis, 0...], to: [globalSide, 1, hidden]),
            ],
            axis: 1,
        )
        let globalFlattened = globalGrid.reshaped(-1, hidden)

        guard let crop = cropFrames?.first,
            crop.h > 1 || crop.w > 1,
            let localPatches,
            localPatches.size > 0
        else {
            return concatenated([globalFlattened, viewSeparator[.newAxis, 0...]], axis: 0)
        }

        let widthCrops = crop.h
        let heightCrops = crop.w
        let patchCount = widthCrops * heightCrops
        let patchesHWC = localPatches[..<patchCount, 0..., 0..., 0...].transposed(0, 2, 3, 1)
        let localSam = samModel(patchesHWC)
        let localVision = visionModel(patchesHWC, patchEmbeds: localSam)
        var localFeatures = concatenated(
            [
                localVision[0..., 1..., 0...],
                localSam.flattened(start: 1, end: 2),
            ],
            axis: -1,
        )
        localFeatures = projector(localFeatures)
        let localHidden = localFeatures.dim(-1)
        let localSide = Int(sqrt(Double(localFeatures.dim(1))))
        var localGrid =
            localFeatures
            .reshaped(heightCrops, widthCrops, localSide, localSide, localHidden)
            .transposed(0, 2, 1, 3, 4)
            .reshaped(heightCrops * localSide, widthCrops * localSide, localHidden)
        localGrid = concatenated(
            [
                localGrid,
                broadcast(
                    imageNewline[.newAxis, .newAxis, 0...],
                    to: [heightCrops * localSide, 1, localHidden],
                ),
            ],
            axis: 1,
        )
        let localFlattened = localGrid.reshaped(-1, localHidden)
        return concatenated(
            [localFlattened, globalFlattened, viewSeparator[.newAxis, 0...]], axis: 0)
    }

    private func inputEmbeddings(_ input: LMInput) -> MLXArray {
        let inputIDs = input.text.tokens
        let safeIDs = which(inputIDs .== config.imageTokenIndex, MLXArray(0), inputIDs)
        let embeddings = languageModel.model.embedTokens(safeIDs)
        guard
            let features = visualFeatures(
                localPatches: input.video?.pixels,
                globalImages: input.image?.pixels,
                cropFrames: input.image?.frames,
            )?.asType(embeddings.dtype)
        else {
            return embeddings
        }
        let maskValues = input.text.mask?.asArray(Int8.self) ?? []
        let positions = maskValues.enumerated().compactMap { $0.element != 0 ? $0.offset : nil }
        guard !positions.isEmpty else {
            return embeddings
        }
        let count = min(positions.count, features.dim(0))
        let result = embeddings
        result[0, MLXArray(Array(positions.prefix(count))), 0...] = features[..<count, 0...]
        return result
    }

    private func unlimitedEmbeddings(_ input: LMInput) throws -> MLXArray {
        let ids = input.text.tokens
        let safe = which(ids .== config.imageTokenIndex, 0, ids)
        let embeddings = languageModel.model.embedTokens(safe)
        guard let image = input.image, let frames = image.frames else { return embeddings }
        let positions = (input.text.mask?.asArray(Int8.self) ?? []).enumerated().compactMap {
            $0.element == 1 ? $0.offset : nil
        }
        guard frames.count == image.pixels.dim(0), !frames.isEmpty else {
            throw VLMError.processing("Unlimited-OCR page metadata must match the global views.")
        }
        let patches = input.video?.pixels
        var patchStart = 0
        var features: [MLXArray] = []
        for (index, frame) in frames.enumerated() {
            let tiled = frame.h > 1 || frame.w > 1
            let patchEnd = patchStart + (tiled ? frame.h * frame.w : 0)
            guard patchEnd <= (patches?.dim(0) ?? 0) else {
                throw VLMError.processing("Unlimited-OCR page tiles do not match their grid.")
            }
            let feature = visualFeatures(
                localPatches: tiled ? patches?[patchStart ..< patchEnd, 0..., 0..., 0...] : nil,
                globalImages: image.pixels[index ..< (index + 1), 0..., 0..., 0...],
                cropFrames: [frame])!
            asyncEval(feature)
            features.append(feature)
            patchStart = patchEnd
        }
        let fused = concatenated(features, axis: 0)
        guard fused.dim(0) == positions.count, patchStart == (patches?.dim(0) ?? 0) else {
            throw VLMError.processing(
                "Unlimited-OCR image features must match the prompt positions.")
        }
        let common = (MLXArray(0, dtype: embeddings.dtype) + MLXArray(0, dtype: fused.dtype)).dtype
        let result = embeddings.asType(common)
        result[0, MLXArray(positions), 0...] = fused.asType(common)
        return result
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        guard config.isUnlimitedOCR else {
            return try kvHeads.map { _ in try makeAttentionKVCache(configuration: configuration) }
        }
        if let configuration,
            configuration.strategy.identifier != .fullPrecision
                && configuration.compatibility != .allowPartial
        {
            throw ModelExecutionError.invalidOptions(
                "Unlimited-OCR uses model-specific full-precision attention caches")
        }
        if config.dashAttention != nil {
            return kvHeads.map { _ in UnlimitedOCRDASHCache() }
        }
        return kvHeads.map { _ in
            UnlimitedOCRReferenceCache(window: config.textConfiguration.slidingWindow)
        }
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let embeddings: MLXArray
        if config.isUnlimitedOCR {
            guard input.text.tokens.dim(0) == 1, input.text.tokens.dim(1) > 0,
                cache.count == kvHeads.count,
                cache.allSatisfy({ $0.offset == 0 })
            else {
                throw VLMError.processing(
                    "Unlimited-OCR prefill requires one document and fresh caches.")
            }
            if let dash = config.dashAttention, let first = cache.first as? UnlimitedOCRDASHCache,
                cache.allSatisfy({ $0 is UnlimitedOCRDASHCache })
            {
                first.startPrompt(input.text.tokens.asArray(Int32.self), parameters: dash)
            } else if config.dashAttention == nil,
                cache.allSatisfy({ $0 is UnlimitedOCRReferenceCache })
            {
                for case let ring as UnlimitedOCRReferenceCache in cache {
                    ring.promptLength = input.text.tokens.dim(1)
                }
            } else {
                throw VLMError.processing(
                    "Unlimited-OCR prefill requires model-specific attention caches.")
            }
            embeddings = try unlimitedEmbeddings(input)
        } else {
            embeddings = inputEmbeddings(input)
        }
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
        sanitize(weights: weights, metadata: [:])
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        var weights = DeepSeekOCRSanitizer.sanitize(
            weights: weights, modelType: config.modelType,
            isMLXLayout: metadata["format"] == "mlx")
        guard config.isUnlimitedOCR else { return weights }
        if config.dashAttention != nil {
            weights = weights.filter { !$0.key.contains(".self_attn.indexer.") }
        }
        let text = config.textConfiguration
        let expertCount = text.nRoutedExperts ?? 0
        if expertCount > 0 {
            for layer in 0 ..< text.hiddenLayers {
                let mlp = "language_model.model.layers.\(layer).mlp."
                for projection in ["gate_proj", "up_proj", "down_proj"] {
                    for suffix in ["weight", "scales", "biases"] {
                        let part = ".\(projection).\(suffix)"
                        guard weights[mlp + "experts.0" + part] != nil else { continue }
                        let keys = (0 ..< expertCount).map { mlp + "experts.\($0)" + part }
                        let entries = keys.compactMap { weights[$0] }
                        guard entries.count == expertCount else { continue }
                        weights[mlp + "switch_mlp" + part] = stacked(entries)
                        for key in keys { weights.removeValue(forKey: key) }
                    }
                }
            }
        }
        return weights
    }

    public func quantizationPredicate(path: String, module: Module) -> QuantizationDecision {
        path.contains(".dash_router.") ? .skip : .useDefault
    }
}

public final class DeepSeekOCR2: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_model") private var visionModel: DeepSeekOCR2Vision.VisionModel
    @ModuleInfo(key: "sam_model") private var samModel: DeepSeekOCRSAM.Encoder
    @ModuleInfo(key: "language_model") private var languageModel: DeepSeekOCRLanguage.LanguageModel
    @ModuleInfo private var projector: DeepSeekOCRProjector
    @ModuleInfo(key: "view_separator") private var viewSeparator: MLXArray

    public let config: DeepSeekOCRConfiguration
    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: DeepSeekOCRConfiguration) {
        self.config = config
        _visionModel.wrappedValue = DeepSeekOCR2Vision.VisionModel(config.visionConfiguration)
        _samModel.wrappedValue = DeepSeekOCRSAM.Encoder(
            config.visionConfiguration.samConfiguration,
        )
        _languageModel.wrappedValue = DeepSeekOCRLanguage.LanguageModel(
            config.textConfiguration,
        )
        _projector.wrappedValue = DeepSeekOCRProjector(config.projectorConfiguration)
        _viewSeparator.wrappedValue = zeros([config.projectorConfiguration.embeddingDimension])
    }

    private func visualFeatures(
        localPatches: MLXArray?,
        globalImages: MLXArray?,
        cropFrames: [THW]?,
    )
        -> MLXArray?
    {
        guard let globalImages else {
            return nil
        }
        var features = [MLXArray]()
        if let crop = cropFrames?.first,
            let localPatches,
            localPatches.size > 0
        {
            let patchCount = max(crop.h * crop.w, 0)
            for index in 0 ..< min(patchCount, localPatches.dim(0)) {
                let patch = localPatches[index ..< (index + 1), 0..., 0..., 0...]
                    .transposed(0, 2, 3, 1)
                let sam = samModel(patch)
                let vision = visionModel(sam)
                features.append(projector(vision)[0])
            }
        }
        let globalHWC = globalImages.transposed(0, 2, 3, 1)
        let globalSam = samModel(globalHWC)
        let globalVision = visionModel(globalSam)
        features.append(projector(globalVision)[0])
        features.append(viewSeparator[.newAxis, 0...])
        return concatenated(features, axis: 0)
    }

    private func inputEmbeddings(_ input: LMInput) -> MLXArray {
        let inputIDs = input.text.tokens
        let safeIDs = which(inputIDs .== config.imageTokenIndex, MLXArray(0), inputIDs)
        let embeddings = languageModel.model.embedTokens(safeIDs)
        guard
            let features = visualFeatures(
                localPatches: input.video?.pixels,
                globalImages: input.image?.pixels,
                cropFrames: input.image?.frames,
            )?.asType(embeddings.dtype)
        else {
            return embeddings
        }
        let maskValues = input.text.mask?.asArray(Int8.self) ?? []
        let positions = maskValues.enumerated().compactMap { $0.element != 0 ? $0.offset : nil }
        guard !positions.isEmpty else {
            return embeddings
        }
        let count = min(positions.count, features.dim(0))
        let result = embeddings
        result[0, MLXArray(Array(positions.prefix(count))), 0...] = features[..<count, 0...]
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
        DeepSeekOCRSanitizer.sanitize(weights: weights, modelType: config.modelType)
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        DeepSeekOCRSanitizer.sanitize(
            weights: weights, modelType: config.modelType,
            isMLXLayout: metadata["format"] == "mlx")
    }
}

private enum DeepSeekOCRSanitizer {
    static func sanitize(
        weights: [String: MLXArray], modelType _: String, isMLXLayout: Bool = false
    ) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        let convSuffixes: Set = [
            "neck.0.weight",
            "neck.2.weight",
            "sam_model.net_2.weight",
            "sam_model.net_3.weight",
            "patch_embed.proj.weight",
            "embeddings.patch_embedding.weight",
        ]
        for (key, value) in weights {
            if key.contains("position_ids") || key.contains("rotary_emb.inv_freq") {
                continue
            }
            var newKey = key
            if newKey.contains("qwen2_model.model.model.layers") {
                newKey = newKey.replacingOccurrences(
                    of: "model.qwen2_model.model.model.layers",
                    with: "vision_model.qwen2_encoder.layers",
                )
            }
            if newKey.contains("qwen2_model.model.model.norm") {
                newKey = newKey.replacingOccurrences(
                    of: "model.qwen2_model.model.model.norm",
                    with: "vision_model.qwen2_encoder.norm",
                )
            }
            newKey = newKey.replacingOccurrences(
                of: "model.qwen2_model.query_1024.weight",
                with: "vision_model.qwen2_encoder.query_1024",
            )
            newKey = newKey.replacingOccurrences(
                of: "model.qwen2_model.query_768.weight",
                with: "vision_model.qwen2_encoder.query_768",
            )
            newKey = newKey.replacingOccurrences(
                of: "model.qwen2_model.query_1024",
                with: "vision_model.qwen2_encoder.query_1024",
            )
            newKey = newKey.replacingOccurrences(
                of: "model.qwen2_model.query_768",
                with: "vision_model.qwen2_encoder.query_768",
            )
            if newKey.contains("model.layers"), !newKey.contains("language_model"),
                !newKey.contains("qwen2")
            {
                newKey = newKey.replacingOccurrences(
                    of: "model.layers",
                    with: "language_model.model.layers",
                )
            }
            if newKey.contains("model.embed_tokens"), !newKey.contains("language_model") {
                newKey = newKey.replacingOccurrences(
                    of: "model.embed_tokens",
                    with: "language_model.model.embed_tokens",
                )
            }
            if newKey.contains("model.norm"), !newKey.contains("language_model"),
                !newKey.contains("qwen2")
            {
                newKey = newKey.replacingOccurrences(
                    of: "model.norm",
                    with: "language_model.model.norm",
                )
            }
            newKey = newKey.replacingOccurrences(of: "model.vision_model", with: "vision_model")
            newKey = newKey.replacingOccurrences(of: "model.sam_model", with: "sam_model")
            newKey = newKey.replacingOccurrences(of: "model.projector", with: "projector")
            newKey = newKey.replacingOccurrences(
                of: "model.view_seperator",
                with: "view_separator",
            )
            newKey = newKey.replacingOccurrences(
                of: "model.image_newline",
                with: "image_newline",
            )
            if newKey.contains("lm_head.weight"), !newKey.contains("language_model") {
                newKey = newKey.replacingOccurrences(
                    of: "lm_head.weight",
                    with: "language_model.lm_head.weight",
                )
            }
            let suffix = newKey.split(separator: ".").suffix(3).joined(separator: ".")
            if !isMLXLayout, convSuffixes.contains(suffix), value.ndim == 4,
                !DeepSeekOCRSAM.checkConvWeight(value)
            {
                sanitized[newKey] = value.transposed(0, 2, 3, 1)
            } else {
                sanitized[newKey] = value
            }
        }
        return sanitized
    }
}
