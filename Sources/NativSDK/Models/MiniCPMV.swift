import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct MiniCPMVConfiguration: Decodable, Sendable {
    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let linearNumValueHeads: Int
        public let linearNumKeyHeads: Int
        public let linearKeyHeadDim: Int
        public let linearValueHeadDim: Int
        public let linearConvKernelDim: Int
        public let rmsNormEps: Float
        public let vocabularySize: Int
        public let ropeTheta: Float
        public let partialRotaryFactor: Float
        public let maxPositionEmbeddings: Int
        public let tieWordEmbeddings: Bool
        public let attentionBias: Bool
        public let headDim: Int
        public let ropeParameters: [String: StringOrNumber]
        public let fullAttentionInterval: Int
        public let numExperts: Int
        public let numExpertsPerToken: Int
        public let decoderSparseStep: Int
        public let sharedExpertIntermediateSize: Int
        public let moeIntermediateSize: Int
        public let normTopKProbability: Bool

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case linearNumValueHeads = "linear_num_value_heads"
            case linearNumKeyHeads = "linear_num_key_heads"
            case linearKeyHeadDim = "linear_key_head_dim"
            case linearValueHeadDim = "linear_value_head_dim"
            case linearConvKernelDim = "linear_conv_kernel_dim"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
            case maxPositionEmbeddings = "max_position_embeddings"
            case tieWordEmbeddings = "tie_word_embeddings"
            case attentionBias = "attention_bias"
            case headDim = "head_dim"
            case ropeParameters = "rope_parameters"
            case ropeScaling = "rope_scaling"
            case fullAttentionInterval = "full_attention_interval"
            case numExperts = "num_experts"
            case numExpertsPerToken = "num_experts_per_tok"
            case decoderSparseStep = "decoder_sparse_step"
            case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
            case moeIntermediateSize = "moe_intermediate_size"
            case normTopKProbability = "norm_topk_prob"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen3_5_text"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 24
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3584
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 2
            linearNumValueHeads =
                try c.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 16
            linearNumKeyHeads = try c.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
            linearKeyHeadDim = try c.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 128
            linearValueHeadDim =
                try c.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
            linearConvKernelDim =
                try c.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 248_094
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262_144
            tieWordEmbeddings =
                try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
            attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            headDim =
                try c.decodeIfPresent(Int.self, forKey: .headDim)
                ?? (hiddenSize / attentionHeads)
            fullAttentionInterval =
                try c.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4
            numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
            numExpertsPerToken =
                try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 0
            decoderSparseStep =
                try c.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
            sharedExpertIntermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
            moeIntermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
            normTopKProbability =
                try c.decodeIfPresent(Bool.self, forKey: .normTopKProbability) ?? true

            var rope =
                try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeParameters)
                ?? c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
                ?? [:]
            if rope["type"] == nil, let ropeType = rope["rope_type"] {
                rope["type"] = ropeType
            }
            rope["type"] = rope["type"] ?? .string("default")
            rope["mrope_section"] = rope["mrope_section"] ?? .ints([11, 11, 10])
            rope["rope_theta"] = rope["rope_theta"] ?? .float(10_000_000)
            rope["partial_rotary_factor"] = rope["partial_rotary_factor"] ?? .float(0.25)
            ropeParameters = rope
            ropeTheta =
                try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? rope["rope_theta"]?.asFloat()
                ?? 10_000_000
            partialRotaryFactor =
                try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
                ?? rope["partial_rotary_factor"]?.asFloat()
                ?? 0.25
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let channels: Int
        public let imageSize: Int
        public let patchSize: Int
        public let hiddenActivation: String
        public let layerNormEps: Float
        public let windowKernelSize: (Int, Int)

        public var windowHiddenSize: Int {
            hiddenSize * windowKernelSize.0 * windowKernelSize.1
        }

        public var windowIntermediateSize: Int {
            intermediateSize * windowKernelSize.0 * windowKernelSize.1
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case channels = "num_channels"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case hiddenActivation = "hidden_act"
            case layerNormEps = "layer_norm_eps"
            case windowKernelSize = "window_kernel_size"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType)
                ?? "minicpmv4_6_vision"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 27
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 980
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            hiddenActivation =
                try c.decodeIfPresent(String.self, forKey: .hiddenActivation)
                ?? "gelu_pytorch_tanh"
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
            let kernel = try c.decodeIfPresent([Int].self, forKey: .windowKernelSize)
            windowKernelSize = (
                kernel?.first ?? 2,
                kernel?.dropFirst().first ?? 2,
            )
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    var isOCR: Bool { modelType == "minicpm_ocr" }
    public let imageSize: Int
    public let imageTokenID: Int
    public let videoTokenID: Int
    public let imageStartID: Int
    public let imageEndID: Int
    public let sliceStartID: Int
    public let sliceEndID: Int
    public let newlineID: Int
    public let insertLayerID: Int
    public let downsampleMode: String
    public let mergeKernelSize: (Int, Int)
    public let mergerTimes: Int
    private let _eosTokenID: IntOrIntArray?
    public var eosTokenIDs: [Int]? {
        _eosTokenID?.values
    }

    private let _vocabularySize: Int?
    public var vocabularySize: Int {
        _vocabularySize ?? textConfiguration.vocabularySize
    }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case imageSize = "image_size"
        case imageTokenID = "image_token_id"
        case videoTokenID = "video_token_id"
        case imageStartID = "image_start_id"
        case imageEndID = "image_end_id"
        case sliceStartID = "slice_start_id"
        case sliceEndID = "slice_end_id"
        case newlineID = "newline_id"
        case insertLayerID = "insert_layer_id"
        case downsampleMode = "downsample_mode"
        case mergeKernelSize = "merge_kernel_size"
        case mergerTimes = "merger_times"
        case _eosTokenID = "eos_token_id"
        case _vocabularySize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration = try c.decode(TextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try c.decode(VisionConfiguration.self, forKey: .visionConfiguration)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "minicpmv4_6"
        imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 1120
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? 248_056
        videoTokenID = try c.decodeIfPresent(Int.self, forKey: .videoTokenID) ?? 248_057
        imageStartID = try c.decodeIfPresent(Int.self, forKey: .imageStartID) ?? 248_078
        imageEndID = try c.decodeIfPresent(Int.self, forKey: .imageEndID) ?? 248_079
        sliceStartID = try c.decodeIfPresent(Int.self, forKey: .sliceStartID) ?? 248_088
        sliceEndID = try c.decodeIfPresent(Int.self, forKey: .sliceEndID) ?? 248_089
        newlineID = try c.decodeIfPresent(Int.self, forKey: .newlineID) ?? 198
        insertLayerID = try c.decodeIfPresent(Int.self, forKey: .insertLayerID) ?? 6
        downsampleMode =
            try c.decodeIfPresent(String.self, forKey: .downsampleMode)
            ?? (modelType == "minicpm_ocr" ? "4x" : "16x")
        if modelType == "minicpm_ocr", !["4x", "16x"].contains(downsampleMode) {
            throw DecodingError.dataCorruptedError(
                forKey: .downsampleMode, in: c,
                debugDescription: "MiniCPM-OCR requires '4x' or '16x'")
        }
        let kernel = try c.decodeIfPresent([Int].self, forKey: .mergeKernelSize)
        mergeKernelSize = (kernel?.first ?? 2, kernel?.dropFirst().first ?? 2)
        mergerTimes = try c.decodeIfPresent(Int.self, forKey: .mergerTimes) ?? 1
        _eosTokenID = try c.decodeIfPresent(IntOrIntArray.self, forKey: ._eosTokenID)
        _vocabularySize = try c.decodeIfPresent(Int.self, forKey: ._vocabularySize)
    }
}

// MARK: - Processor

public struct MiniCPMVProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String?
    public internal(set) var maxSliceNums: Int
    public let maxVideoFrames: Int
    public let stackFrames: Int
    public let scaleResolution: Int
    public let patchSize: Int
    public let imageFeatureSize: Int
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let imageStart: String
    public let imageEnd: String
    public let sliceStart: String
    public let sliceEnd: String
    public let unknown: String
    public let imageIDStart: String
    public let imageIDEnd: String
    public internal(set) var sliceMode: Bool
    public internal(set) var useImageID: Bool
    public internal(set) var downsampleMode: String
    public let packImageByPatch: Bool
    let isOCR: Bool
    var ocrImageOptions: MiniCPMOCRImageOptions

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case modelType = "model_type"
        case maxSliceNums = "max_slice_nums"
        case maxVideoFrames = "max_num_frames"
        case stackFrames = "stack_frames"
        case scaleResolution = "scale_resolution"
        case patchSize = "patch_size"
        case imageFeatureSize = "image_feature_size"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageStart = "im_start"
        case imageEnd = "im_end"
        case sliceStart = "slice_start"
        case sliceEnd = "slice_end"
        case unknown = "unk"
        case imageIDStart = "im_id_start"
        case imageIDEnd = "im_id_end"
        case sliceMode = "slice_mode"
        case useImageID = "use_image_id"
        case downsampleMode = "downsample_mode"
        case packImageByPatch = "pack_image_by_patch"
        case imageProcessor = "image_processor"
        case videoProcessor = "video_processor"
    }

    public init(from decoder: Decoder) throws {
        let outer = try decoder.container(keyedBy: CodingKeys.self)
        let c =
            (try? outer.nestedContainer(keyedBy: CodingKeys.self, forKey: .imageProcessor))
            ?? outer
        let video =
            try? outer.nestedContainer(keyedBy: CodingKeys.self, forKey: .videoProcessor)
        processorClass = try outer.decodeIfPresent(String.self, forKey: .processorClass)
        let modelType = try outer.decodeIfPresent(String.self, forKey: .modelType)
        isOCR = processorClass == "MiniCPMOcrProcessor" || modelType == "minicpm_ocr"
        if isOCR {
            ocrImageOptions =
                try outer.decodeIfPresent(MiniCPMOCRImageOptions.self, forKey: .imageProcessor)
                ?? MiniCPMOCRImageOptions(from: decoder)
        } else {
            ocrImageOptions = MiniCPMOCRImageOptions()
        }
        maxSliceNums = try c.decodeIfPresent(Int.self, forKey: .maxSliceNums) ?? 9
        maxVideoFrames =
            try video?.decodeIfPresent(Int.self, forKey: .maxVideoFrames)
            ?? outer.decodeIfPresent(Int.self, forKey: .maxVideoFrames)
            ?? 128
        stackFrames =
            try video?.decodeIfPresent(Int.self, forKey: .stackFrames)
            ?? outer.decodeIfPresent(Int.self, forKey: .stackFrames)
            ?? 1
        scaleResolution = try c.decodeIfPresent(Int.self, forKey: .scaleResolution) ?? 448
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        imageFeatureSize = try c.decodeIfPresent(Int.self, forKey: .imageFeatureSize) ?? 64
        imageMean = try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd = try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        imageStart = try c.decodeIfPresent(String.self, forKey: .imageStart) ?? "<image>"
        imageEnd = try c.decodeIfPresent(String.self, forKey: .imageEnd) ?? "</image>"
        sliceStart = try c.decodeIfPresent(String.self, forKey: .sliceStart) ?? "<slice>"
        sliceEnd = try c.decodeIfPresent(String.self, forKey: .sliceEnd) ?? "</slice>"
        unknown = try c.decodeIfPresent(String.self, forKey: .unknown) ?? "<unk>"
        imageIDStart = try c.decodeIfPresent(String.self, forKey: .imageIDStart) ?? "<image_id>"
        imageIDEnd = try c.decodeIfPresent(String.self, forKey: .imageIDEnd) ?? "</image_id>"
        sliceMode = try c.decodeIfPresent(Bool.self, forKey: .sliceMode) ?? true
        useImageID = try c.decodeIfPresent(Bool.self, forKey: .useImageID) ?? true
        downsampleMode =
            try c.decodeIfPresent(String.self, forKey: .downsampleMode) ?? (isOCR ? "4x" : "16x")
        packImageByPatch = try c.decodeIfPresent(Bool.self, forKey: .packImageByPatch) ?? true
    }
}

public struct MiniCPMVProcessor: UserInputProcessor {
    private var config: MiniCPMVProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageTokenID: Int
    private let videoTokenID: Int
    private let imageStartID: Int
    private let imageEndID: Int
    private let sliceStartID: Int
    private let sliceEndID: Int

    public init(_ config: MiniCPMVProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        imageTokenID = tokenizer.convertTokenToId("<|image_pad|>") ?? 248_056
        videoTokenID = tokenizer.convertTokenToId("<|video_pad|>") ?? 248_057
        imageStartID = tokenizer.convertTokenToId(config.imageStart) ?? 248_078
        imageEndID = tokenizer.convertTokenToId(config.imageEnd) ?? 248_079
        sliceStartID = tokenizer.convertTokenToId(config.sliceStart) ?? 248_088
        sliceEndID = tokenizer.convertTokenToId(config.sliceEnd) ?? 248_089
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        if config.isOCR {
            var processor = self
            processor.config = try config.applyingOCROptions(input.additionalContext ?? [:])
            return try await processor.prepareInput(input: input)
        }
        return try await prepareInput(input: input)
    }

    private func prepareInput(input: UserInput) async throws -> LMInput {
        let prompt = Self.prompt(from: input)
        let images = MediaProcessing.images(from: input)
        let videos = Self.videos(from: input)

        let promptText: String
        if config.processorClass == "MiniCPMV4_7Processor" || config.isOCR {
            var messages = DefaultMessageGenerator().generate(from: input)
            if case .chat(let chat) = input.prompt {
                for index in chat.indices {
                    let prefix =
                        String(repeating: "<|image_pad|>\n", count: chat[index].images.count)
                        + String(repeating: "<|video_pad|>\n", count: chat[index].videos.count)
                    messages[index]["content"] = prefix + chat[index].content
                }
            } else if let lastUser = messages.lastIndex(where: { ($0["role"] as? String) == "user" }
            ),
                let text = messages[lastUser]["content"] as? String,
                !text.contains("<|image_pad|>"), !text.contains("<|video_pad|>")
            {
                messages[lastUser]["content"] =
                    String(repeating: "<|image_pad|>\n", count: images.count)
                    + String(repeating: "<|video_pad|>\n", count: videos.count) + text
            }
            var context = input.additionalContext ?? [:]
            context["enable_thinking"] = context["enable_thinking"] ?? false
            let tokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools, additionalContext: context)
            promptText = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
        } else if images.isEmpty, videos.isEmpty {
            promptText = "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
        } else {
            let imagePrefix = images.isEmpty ? "" : "<|image_pad|>\n"
            let videoPrefix = videos.isEmpty ? "" : "<|video_pad|>\n"
            promptText =
                "<|im_start|>user\n\(imagePrefix)\(videoPrefix)\(prompt)<|im_end|>\n<|im_start|>assistant\n"
        }

        var processedImages: [ProcessedMiniCPMImage] = []
        processedImages.reserveCapacity(images.count)
        for image in images {
            try processedImages.append(
                preprocess(image: image.asCIImage(), processing: input.processing),
            )
        }

        var processedVideos: [ProcessedMiniCPMVideo] = []
        processedVideos.reserveCapacity(videos.count)
        for video in videos {
            try await processedVideos.append(
                preprocess(video: video, processing: input.processing),
            )
        }

        let (tokens, imageBounds, videoBounds) = tokenize(
            promptText,
            processedImages: processedImages,
            processedVideos: processedVideos,
        )

        let imageCrops = processedImages.flatMap { Array(zip($0.patches, $0.targetFrames)) }
        let videoCrops = processedVideos.flatMap(\.frames).flatMap {
            Array(zip($0.patches, $0.targetFrames))
        }
        guard imageCrops.count == imageBounds.count, videoCrops.count == videoBounds.count else {
            throw VLMError.processing("MiniCPM-V media crops do not match placeholder spans")
        }
        // Image and video messages can be interleaved; features and grids must follow token order.
        let crops = (Array(zip(imageBounds, imageCrops)) + Array(zip(videoBounds, videoCrops)))
            .sorted { $0.0.0 < $1.0.0 }
        var imageInput: LMInput.ProcessedImage?
        if !crops.isEmpty {
            imageInput = .init(
                pixels: concatenated(crops.map { $0.1.0.flattened() }, axis: 0),
                frames: crops.map { $0.1.1 })
        }

        var videoInput: LMInput.ProcessedVideo?
        if let bounds = boundsArray((imageBounds + videoBounds).sorted { $0.0 < $1.0 }) {
            videoInput = .init(pixels: bounds)
        }

        return LMInput(
            text: .init(tokens: MLXArray(tokens)[.newAxis]),
            image: imageInput,
            video: videoInput,
        )
    }

    private struct ProcessedMiniCPMImage {
        let patches: [MLXArray]
        let targetSizes: [(Int, Int)]
        let grid: (Int, Int)?

        var targetFrames: [THW] {
            targetSizes.map { THW(1, $0.0, $0.1) }
        }
    }

    private struct ProcessedMiniCPMVideo {
        let frames: [ProcessedMiniCPMImage]

        var targetFrames: [THW] {
            frames.flatMap(\.targetFrames)
        }
    }

    private enum MultimodalMarkerKind {
        case image
        case video
    }

    private typealias Bounds = [(Int, Int)]

    private func tokenize(
        _ text: String,
        processedImages: [ProcessedMiniCPMImage],
        processedVideos: [ProcessedMiniCPMVideo],
    )
        -> ([Int], Bounds, Bounds)
    {
        guard !processedImages.isEmpty || !processedVideos.isEmpty else {
            return (tokenizer.encode(text: text, addSpecialTokens: false), [], [])
        }

        var tokens: [Int] = []
        var imageBounds: Bounds = []
        var videoBounds: Bounds = []
        var cursor = text.startIndex
        var imageIndex = 0
        var videoIndex = 0

        while let marker = nextMultimodalMarker(in: text, from: cursor) {
            if marker.range.lowerBound > cursor {
                tokens += tokenizer.encode(
                    text: String(text[cursor ..< marker.range.lowerBound]),
                    addSpecialTokens: false,
                )
            }

            switch marker.kind {
            case .image:
                if imageIndex < processedImages.count {
                    appendPlaceholder(
                        for: processedImages[imageIndex],
                        imageIndex: imageIndex,
                        fillTokenID: imageTokenID,
                        includeImageID: config.useImageID,
                        tokens: &tokens,
                        bounds: &imageBounds,
                    )
                    imageIndex += 1
                } else {
                    tokens += tokenizer.encode(
                        text: String(text[marker.range]), addSpecialTokens: false)
                }
            case .video:
                if videoIndex < processedVideos.count {
                    appendVideoPlaceholder(
                        for: processedVideos[videoIndex],
                        videoIndex: videoIndex,
                        tokens: &tokens,
                        bounds: &videoBounds,
                    )
                    videoIndex += 1
                } else {
                    tokens += tokenizer.encode(
                        text: String(text[marker.range]), addSpecialTokens: false)
                }
            }

            cursor = marker.range.upperBound
        }

        if cursor < text.endIndex {
            tokens += tokenizer.encode(text: String(text[cursor...]), addSpecialTokens: false)
        }

        var leading: [Int] = []
        var leadingImageBounds: Bounds = []
        var leadingVideoBounds: Bounds = []
        while imageIndex < processedImages.count {
            appendPlaceholder(
                for: processedImages[imageIndex],
                imageIndex: imageIndex,
                fillTokenID: imageTokenID,
                includeImageID: config.useImageID,
                tokens: &leading,
                bounds: &leadingImageBounds,
            )
            imageIndex += 1
        }
        while videoIndex < processedVideos.count {
            appendVideoPlaceholder(
                for: processedVideos[videoIndex],
                videoIndex: videoIndex,
                tokens: &leading,
                bounds: &leadingVideoBounds,
            )
            videoIndex += 1
        }

        if !leading.isEmpty {
            let delta = leading.count
            tokens = leading + tokens
            imageBounds = leadingImageBounds + imageBounds.map { ($0.0 + delta, $0.1 + delta) }
            videoBounds = leadingVideoBounds + videoBounds.map { ($0.0 + delta, $0.1 + delta) }
        }

        return (tokens, imageBounds, videoBounds)
    }

    private func appendPlaceholder(
        for image: ProcessedMiniCPMImage,
        imageIndex: Int,
        fillTokenID: Int,
        includeImageID: Bool,
        tokens: inout [Int],
        bounds: inout Bounds,
    ) {
        let divisor = config.downsampleMode == "4x" ? 4 : 16
        let counts = image.targetSizes.map { max(1, $0.0 * $0.1 / divisor) }
        guard let first = counts.first else {
            return
        }

        if includeImageID {
            tokens += tokenizer.encode(
                text: "\(config.imageIDStart)\(imageIndex)\(config.imageIDEnd)",
                addSpecialTokens: false,
            )
        }
        appendBoundedTokens(
            startID: imageStartID,
            fillCount: first,
            fillTokenID: fillTokenID,
            endID: imageEndID,
            tokens: &tokens,
            bounds: &bounds,
        )

        guard let grid = image.grid, grid.0 > 0, grid.1 > 0, counts.count > 1 else {
            return
        }

        let newline = tokenizer.encode(text: "\n", addSpecialTokens: false)
        var idx = 1
        for row in 0 ..< grid.1 {
            for _ in 0 ..< grid.0 {
                let count = idx < counts.count ? counts[idx] : counts[1]
                appendBoundedTokens(
                    startID: sliceStartID,
                    fillCount: count,
                    fillTokenID: fillTokenID,
                    endID: sliceEndID,
                    tokens: &tokens, bounds: &bounds,
                )
                idx += 1
            }
            if row != grid.1 - 1 {
                tokens += newline
            }
        }
    }

    private func appendVideoPlaceholder(
        for video: ProcessedMiniCPMVideo,
        videoIndex: Int,
        tokens: inout [Int],
        bounds: inout Bounds,
    ) {
        guard !video.frames.isEmpty else {
            return
        }
        if config.useImageID {
            tokens += tokenizer.encode(
                text: "\(config.imageIDStart)\(videoIndex)\(config.imageIDEnd)",
                addSpecialTokens: false,
            )
        }
        for (frameIndex, frame) in video.frames.enumerated() {
            appendPlaceholder(
                for: frame,
                imageIndex: frameIndex,
                fillTokenID: videoTokenID,
                includeImageID: false,
                tokens: &tokens,
                bounds: &bounds,
            )
        }
    }

    private func appendBoundedTokens(
        startID: Int,
        fillCount: Int,
        fillTokenID: Int,
        endID: Int,
        tokens: inout [Int],
        bounds: inout Bounds,
    ) {
        tokens.append(startID)
        let start = tokens.count
        tokens += Array(repeating: fillTokenID, count: fillCount)
        let end = tokens.count
        tokens.append(endID)
        bounds.append((start, end))
    }

    private func boundsArray(_ bounds: Bounds) -> MLXArray? {
        guard !bounds.isEmpty else {
            return nil
        }
        return MLXArray(bounds.flatMap { [$0.0, $0.1] }, [bounds.count, 2])
    }

    private func nextMultimodalMarker(
        in text: String,
        from cursor: String.Index,
    )
        -> (range: Range<String.Index>, kind: MultimodalMarkerKind)?
    {
        let markers: [(String, MultimodalMarkerKind)] = [
            ("<image>./</image>", .image),
            ("<|image_pad|>", .image),
            ("<image>", .image),
            ("<|video_pad|>", .video),
            ("<video>", .video),
        ]

        var best: (range: Range<String.Index>, kind: MultimodalMarkerKind)?
        for (marker, kind) in markers {
            guard let range = text[cursor...].range(of: marker) else {
                continue
            }
            if let current = best {
                if range.lowerBound < current.range.lowerBound
                    || (range.lowerBound == current.range.lowerBound
                        && range.upperBound > current.range.upperBound)
                {
                    best = (range, kind)
                }
            } else {
                best = (range, kind)
            }
        }
        return best
    }

    private func preprocess(
        image source: CIImage,
        processing: UserInput.Processing?,
    )
        throws -> ProcessedMiniCPMImage
    {
        var source = MediaProcessing.inSRGBToneCurveSpace(
            MediaProcessing.apply(source, processing: processing),
        )
        if config.isOCR {
            source = source.transformed(
                by: CGAffineTransform(
                    translationX: -source.extent.minX, y: -source.extent.minY))
        }
        let size = source.extent.size
        let grid =
            config.sliceMode ? slicedGrid(width: Int(size.width), height: Int(size.height)) : nil

        var images: [CIImage] = []
        if let grid {
            let best = bestResize(
                width: Int(size.width),
                height: Int(size.height),
                scaleResolution: config.scaleResolution,
                patchSize: config.patchSize,
                allowUpscale: false,
            )
            images.append(
                MediaProcessing.resampleBicubic(source, to: CGSize(width: best.0, height: best.1)))

            let refined = refinedSize(
                width: Int(size.width),
                height: Int(size.height),
                grid: grid,
                scaleResolution: config.scaleResolution,
                patchSize: config.patchSize,
            )
            let refinedImage = MediaProcessing.resampleBicubic(
                source,
                to: CGSize(width: refined.0, height: refined.1),
            )
            let cropWidth = refined.0 / grid.0
            let cropHeight = refined.1 / grid.1
            for row in 0 ..< grid.1 {
                for column in 0 ..< grid.0 {
                    let x = column * cropWidth
                    let y = refined.1 - ((row + 1) * cropHeight)
                    let rect = CGRect(x: x, y: y, width: cropWidth, height: cropHeight)
                    var crop = refinedImage.cropped(to: rect)
                    if config.isOCR, config.ocrImageOptions.adaptiveTileUpsize,
                        config.ocrImageOptions.adaptiveSmallImageUpsize
                    {
                        let width = Double(size.width)
                        let height = Double(size.height)
                        let left = (width * Double(column) / Double(grid.0)).rounded(
                            .toNearestOrEven)
                        let right = (width * Double(column + 1) / Double(grid.0)).rounded(
                            .toNearestOrEven)
                        let top = (height * Double(row) / Double(grid.1)).rounded(.toNearestOrEven)
                        let bottom = (height * Double(row + 1) / Double(grid.1)).rounded(
                            .toNearestOrEven)
                        let cellRect = CGRect(
                            x: left, y: height - bottom, width: right - left, height: bottom - top)
                        if cellRect.width * cellRect.height
                            <= CGFloat(config.scaleResolution * config.scaleResolution),
                            cellRect.width >= 1, cellRect.height >= 1
                        {
                            let cell = source.cropped(to: cellRect).transformed(
                                by: CGAffineTransform(
                                    translationX: -cellRect.minX, y: -cellRect.minY))
                            let tier = config.ocrImageOptions.tier(cell)
                            if tier != .full {
                                let desired = bestResize(
                                    width: Int(cellRect.width), height: Int(cellRect.height),
                                    scaleResolution: tier == .tiny
                                        ? config.ocrImageOptions.tinyResolution
                                        : config.scaleResolution,
                                    patchSize: config.patchSize, allowUpscale: tier == .tiny)
                                if desired.0 * desired.1 < cropWidth * cropHeight {
                                    crop = MediaProcessing.resampleBicubic(
                                        cell, to: CGSize(width: desired.0, height: desired.1))
                                }
                            }
                        }
                    }
                    images.append(crop)
                }
            }
        } else {
            var resolution = config.scaleResolution
            var upscale = true
            if config.isOCR {
                upscale = !config.ocrImageOptions.disableSmallImageUpsize
                if config.ocrImageOptions.adaptiveSmallImageUpsize {
                    switch config.ocrImageOptions.tier(source) {
                    case .full: upscale = true
                    case .tiny:
                        resolution = config.ocrImageOptions.tinyResolution
                        upscale = true
                    case .native: break
                    }
                }
            }
            let best = bestResize(
                width: Int(size.width),
                height: Int(size.height),
                scaleResolution: resolution,
                patchSize: config.patchSize,
                allowUpscale: upscale,
            )
            images.append(
                MediaProcessing.resampleBicubic(source, to: CGSize(width: best.0, height: best.1)))
        }

        var patches: [MLXArray] = []
        var targetSizes: [(Int, Int)] = []
        for image in images {
            let (array, targetSize) = try preprocessResized(image)
            patches.append(array)
            targetSizes.append(targetSize)
        }
        return ProcessedMiniCPMImage(patches: patches, targetSizes: targetSizes, grid: grid)
    }

    private func preprocess(
        video: UserInput.Video,
        processing: UserInput.Processing?,
    )
        async throws -> ProcessedMiniCPMVideo
    {
        let maxFrames = max(config.maxVideoFrames, 1)
        var selectedFrames: [CIImage] = []
        _ = try await MediaProcessing.asProcessedSequence(
            video,
            targetFPS: { duration in
                let seconds = duration.seconds
                guard seconds.isFinite, seconds > 0 else {
                    return 1
                }
                return Double(maxFrames) / seconds
            },
            maxFrames: maxFrames,
            frameProcessing: { frame in
                selectedFrames.append(try frame.image.asCIImage())
                return frame
            })

        guard !selectedFrames.isEmpty else {
            throw VLMError.processing("MiniCPM-V video produced no frames.")
        }

        let units = stackFrameUnits(selectedFrames)
        return try ProcessedMiniCPMVideo(
            frames: units.map { try preprocess(image: $0, processing: processing) },
        )
    }

    private func stackFrameUnits(_ frames: [CIImage]) -> [CIImage] {
        let stackFrames = max(config.stackFrames, 1)
        guard stackFrames > 1 else {
            return frames
        }

        var units: [CIImage] = []
        units.reserveCapacity((frames.count + stackFrames - 1) / stackFrames)
        var index = 0
        while index < frames.count {
            let end = min(index + stackFrames, frames.count)
            units.append(concatFramesAsImage(Array(frames[index ..< end])))
            index = end
        }
        return units
    }

    private func concatFramesAsImage(_ frames: [CIImage]) -> CIImage {
        guard frames.count > 1 else {
            return frames[0]
        }

        let first = MediaProcessing.inSRGBToneCurveSpace(frames[0])
        let width = max(Int(first.extent.width.rounded()), 1)
        let height = max(Int(first.extent.height.rounded()), 1)
        let count = frames.count
        let lineWidth = 6
        let rows: Int
        let columns: Int
        if count == 4 {
            rows = 2
            columns = 2
        } else if count == 3 {
            rows = width >= height ? 1 : 3
            columns = width >= height ? 3 : 1
        } else if count == 2 {
            rows = width >= height ? 1 : 2
            columns = width >= height ? 2 : 1
        } else {
            rows = 1
            columns = count
        }

        let canvasWidth = columns * width + (columns - 1) * lineWidth
        let canvasHeight = rows * height + (rows - 1) * lineWidth
        var canvas = CIImage(color: .black)
            .cropped(to: CGRect(x: 0, y: 0, width: canvasWidth, height: canvasHeight))

        for (index, frame) in frames.enumerated() {
            let row = index / columns
            let column = index % columns
            let resized = MediaProcessing.resampleBicubic(
                MediaProcessing.inSRGBToneCurveSpace(frame),
                to: CGSize(width: width, height: height),
            )
            let zeroOrigin = resized.transformed(
                by: CGAffineTransform(
                    translationX: -resized.extent.origin.x,
                    y: -resized.extent.origin.y,
                ),
            )
            let x = CGFloat(column * (width + lineWidth))
            let y = CGFloat((rows - row - 1) * (height + lineWidth))
            let placed = zeroOrigin.transformed(by: CGAffineTransform(translationX: x, y: y))
            canvas = placed.composited(over: canvas)
        }

        return canvas.cropped(to: CGRect(x: 0, y: 0, width: canvasWidth, height: canvasHeight))
    }

    private func preprocessResized(_ image: CIImage) throws -> (MLXArray, (Int, Int)) {
        let width = Int(image.extent.width.rounded())
        let height = Int(image.extent.height.rounded())
        var array = MediaProcessing.renderRGBArray(image)
        let mean = MLXArray(config.imageMean.map { Float32($0) }).reshaped(1, 1, 3)
        let std = MLXArray(config.imageStd.map { Float32($0) }).reshaped(1, 1, 3)
        array = (array - mean) / std
        let chw = array.transposed(2, 0, 1)
        let patch = config.patchSize
        let validHeight = (height / patch) * patch
        let validWidth = (width / patch) * patch
        guard validHeight > 0, validWidth > 0 else {
            throw VLMError.processing("MiniCPM-V image is smaller than one patch.")
        }
        var cropped = chw
        if validHeight != height || validWidth != width {
            cropped = cropped[0..., 0 ..< validHeight, 0 ..< validWidth]
        }
        let patches =
            cropped
            .reshaped(3, validHeight / patch, patch, validWidth / patch, patch)
            .transposed(0, 2, 1, 3, 4)
            .reshaped(3, patch, -1)
        return (patches.asType(.float32), (validHeight / patch, validWidth / patch))
    }

    private func slicedGrid(width: Int, height: Int) -> (Int, Int)? {
        let ratio = Double(width * height) / Double(config.scaleResolution * config.scaleResolution)
        let multiple = min(Int(ceil(ratio)), config.maxSliceNums)
        guard multiple > 1 else {
            return nil
        }

        var candidates: [(Int, Int)] = []
        for gridNum in max(2, multiple - 1) ... min(config.maxSliceNums, multiple + 1) {
            for factor in 1 ... gridNum where gridNum % factor == 0 {
                candidates.append((factor, gridNum / factor))
            }
        }
        guard !candidates.isEmpty else {
            return nil
        }

        let logRatio = log(Double(width) / Double(max(height, 1)))
        return candidates.min { lhs, rhs in
            abs(logRatio - log(Double(lhs.0) / Double(lhs.1)))
                < abs(logRatio - log(Double(rhs.0) / Double(rhs.1)))
        }
    }

    private func bestResize(
        width: Int,
        height: Int,
        scaleResolution: Int,
        patchSize: Int,
        allowUpscale: Bool,
    )
        -> (Int, Int)
    {
        var width = width
        var height = height
        if width * height > scaleResolution * scaleResolution || allowUpscale {
            let ratio = Double(width) / Double(max(height, 1))
            height = Int(Double(scaleResolution) / sqrt(max(ratio, 1e-6)))
            width = Int(Double(height) * ratio)
        }
        let merge = patchSize * (config.isOCR && config.downsampleMode == "4x" ? 2 : 4)
        return (ensureDivide(width, by: merge), ensureDivide(height, by: merge))
    }

    private func refinedSize(
        width: Int,
        height: Int,
        grid: (Int, Int),
        scaleResolution: Int,
        patchSize: Int,
    )
        -> (Int, Int)
    {
        let refinedWidth = ensureDivide(width, by: grid.0)
        let refinedHeight = ensureDivide(height, by: grid.1)
        let gridWidth = refinedWidth / grid.0
        let gridHeight = refinedHeight / grid.1
        let best = bestResize(
            width: gridWidth,
            height: gridHeight,
            scaleResolution: scaleResolution,
            patchSize: patchSize,
            allowUpscale: true,
        )
        return (best.0 * grid.0, best.1 * grid.1)
    }

    private func ensureDivide(_ length: Int, by patchSize: Int) -> Int {
        max(
            Int(
                (Double(length) / Double(patchSize)).rounded(
                    config.isOCR ? .toNearestOrEven : .toNearestOrAwayFromZero)) * patchSize,
            patchSize)
    }

    private static func prompt(from input: UserInput) -> String {
        switch input.prompt {
        case .text(let text):
            text
        case .messages:
            input.prompt.description
        case .chat(let messages):
            messages.last(where: { $0.role == .user })?.content
                ?? messages.map(\.content).joined(separator: "\n")
        }
    }

    private static func videos(from input: UserInput) -> [UserInput.Video] {
        if !input.videos.isEmpty {
            return input.videos
        }
        guard case .chat(let messages) = input.prompt else {
            return []
        }
        return messages.flatMap(\.videos)
    }
}

// MARK: - Language

private func computeMiniCPMVGatedDeltaG(_ aLog: MLXArray, _ a: MLXArray, _ dtBias: MLXArray)
    -> MLXArray
{
    let decay = exp(-exp(aLog.asType(.float32)) * softplus(a + dtBias))
    return decay.asType(a.dtype)
}

func miniCPMVGatedDeltaStepOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray,
    mask: MLXArray? = nil,
)
    -> (MLXArray, MLXArray)
{
    let oldState = state
    let decay: MLXArray
    if g.ndim == 2 {
        decay = expandedDimensions(g, axes: [2, 3])
    } else if g.ndim == 3 {
        decay = expandedDimensions(g, axis: -2)
    } else {
        fatalError("Unsupported gating shape \(g.shape)")
    }

    var state = state * decay
    let kvMem = (state * expandedDimensions(k, axis: -2)).sum(axis: -1)
    let delta = (v - kvMem) * expandedDimensions(beta, axis: -1)
    state = state + expandedDimensions(k, axis: -2) * expandedDimensions(delta, axis: -1)
    let y = (state * expandedDimensions(q, axis: -2)).sum(axis: -1)

    if let mask {
        let expandedMask: MLXArray
        if mask.ndim == 1 {
            expandedMask = expandedDimensions(mask, axes: [1, 2, 3])
        } else if mask.ndim == 2 {
            expandedMask = expandedDimensions(mask, axes: [2, 3])
        } else if mask.ndim == 3 {
            expandedMask = expandedDimensions(mask, axis: -1)
        } else {
            fatalError("Unsupported mask shape \(mask.shape)")
        }
        state = MLX.where(expandedMask, state, oldState)
    }

    return (y, state)
}

// Preserve intermediate dtype casts while fusing elementwise state updates.
let compiledMiniCPMVGatedDeltaStep: @Sendable ([MLXArray]) -> [MLXArray] = compile(shapeless: true)
{ inputs in
    let (y, state) = miniCPMVGatedDeltaStepOps(
        q: inputs[0], k: inputs[1], v: inputs[2],
        g: inputs[3], beta: inputs[4], state: inputs[5])
    return [y, state]
}

private func miniCPMVGatedDeltaOps(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    g: MLXArray,
    beta: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil,
)
    -> (MLXArray, MLXArray)
{
    let batch = q.dim(0)
    let length = q.dim(1)
    let keyHeads = q.dim(2)
    let keyDim = q.dim(3)
    let valueHeads = v.dim(2)
    let valueDim = v.dim(3)

    var q = q
    var k = k
    let repeatFactor = valueHeads / keyHeads
    if repeatFactor > 1 {
        q = repeated(q, count: repeatFactor, axis: -2)
        k = repeated(k, count: repeatFactor, axis: -2)
    }

    var state = state ?? MLXArray.zeros([batch, valueHeads, valueDim, keyDim], dtype: q.dtype)
    var outputs: [MLXArray] = []
    outputs.reserveCapacity(length)

    for index in 0 ..< length {
        let maskStep = mask == nil ? nil : mask![0..., index]
        let output: MLXArray
        let nextState: MLXArray
        if maskStep == nil {
            let result = compiledMiniCPMVGatedDeltaStep([
                q[0..., index], k[0..., index], v[0..., index],
                g[0..., index], beta[0..., index], state,
            ])
            (output, nextState) = (result[0], result[1])
        } else {
            (output, nextState) = miniCPMVGatedDeltaStepOps(
                q: q[0..., index], k: k[0..., index], v: v[0..., index],
                g: g[0..., index], beta: beta[0..., index], state: state, mask: maskStep)
        }
        outputs.append(output)
        state = nextState
    }

    return (stacked(outputs, axis: 1), state)
}

private func miniCPMVGatedDeltaUpdate(
    q: MLXArray,
    k: MLXArray,
    v: MLXArray,
    a: MLXArray,
    b: MLXArray,
    aLog: MLXArray,
    dtBias: MLXArray,
    state: MLXArray? = nil,
    mask: MLXArray? = nil,
)
    -> (MLXArray, MLXArray)
{
    let beta = sigmoid(b)
    let g = computeMiniCPMVGatedDeltaG(aLog, a, dtBias)

    let batch = q.dim(0)
    let keyDim = q.dim(3)
    let valueHeads = v.dim(2)
    let valueDim = v.dim(3)
    let state = state ?? MLXArray.zeros([batch, valueHeads, valueDim, keyDim], dtype: q.dtype)

    return miniCPMVGatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
}

private enum MiniCPMVLanguage {
    final class RotaryEmbedding {
        private let invFreq: MLXArray
        private let mropeSection: [Int]

        init(dim: Int, base: Float, mropeSection: [Int]) {
            let safeDim = max(1, dim)
            var freq = MLXArray(stride(from: 0, to: safeDim, by: 2)).asType(.float32)
            freq = freq / Float(safeDim)
            invFreq = 1.0 / pow(MLXArray(base), freq)
            self.mropeSection = mropeSection.count >= 3 ? mropeSection : [11, 11, 10]
        }

        private func applyInterleavedMRope(_ freqs: MLXArray) -> MLXArray {
            let freqsT = freqs[0, 0..., 0..., 0...]
            let dims = freqsT.dim(-1)
            var slices: [MLXArray] = []
            slices.reserveCapacity(dims)

            for idx in 0 ..< dims {
                var slice = freqsT[0..., 0..., idx]
                for (dim, offset) in [(1, 1), (2, 2)] {
                    let length = min(mropeSection[dim] * 3, dims)
                    if idx >= offset, idx < length, (idx - offset) % 3 == 0 {
                        slice = freqs[dim, 0..., 0..., idx]
                        break
                    }
                }
                slices.append(slice)
            }
            return stacked(slices, axis: -1)
        }

        func callAsFunction(x: MLXArray, positionIDs: MLXArray) -> (MLXArray, MLXArray) {
            var positionIDs = positionIDs
            if positionIDs.ndim == 2 {
                positionIDs = broadcast(
                    positionIDs[.newAxis, 0..., 0...],
                    to: [3, positionIDs.dim(0), positionIDs.dim(1)],
                )
            }

            let pos = positionIDs.asType(.float32)
            let inv = invFreq.asType(.float32)[.newAxis, .newAxis, .newAxis, 0...]
            var freqs = pos[0..., 0..., 0..., .newAxis] * inv
            freqs = applyInterleavedMRope(freqs)
            let emb = concatenated([freqs, freqs], axis: -1)
            return (cos(emb).asType(x.dtype), sin(emb).asType(x.dtype))
        }
    }

    static func applyMultimodalRotaryPosEmb(
        q: MLXArray,
        k: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
    )
        -> (MLXArray, MLXArray)
    {
        let cos = expandedDimensions(cos, axis: 1)
        let sin = expandedDimensions(sin, axis: 1)
        let rotaryDim = cos.dim(-1)
        let qRot = q[.ellipsis, ..<rotaryDim]
        let kRot = k[.ellipsis, ..<rotaryDim]
        let qEmbedded = (qRot * cos) + (RotaryEmbeddingOps.rotateHalf(qRot) * sin)
        let kEmbedded = (kRot * cos) + (RotaryEmbeddingOps.rotateHalf(kRot) * sin)
        let qOut =
            rotaryDim < q.dim(-1)
            ? concatenated([qEmbedded, q[.ellipsis, rotaryDim...]], axis: -1)
            : qEmbedded
        let kOut =
            rotaryDim < k.dim(-1)
            ? concatenated([kEmbedded, k[.ellipsis, rotaryDim...]], axis: -1)
            : kEmbedded
        return (qOut, kOut)
    }

    final class RMSNormGated: Module {
        @ParameterInfo(key: "weight") var weight: MLXArray
        let eps: Float
        let preservePrecision: Bool

        init(dimensions: Int, eps: Float = 1e-6, preservePrecision: Bool = false) {
            self.eps = eps
            self.preservePrecision = preservePrecision
            _weight.wrappedValue = MLXArray.ones([dimensions])
            super.init()
        }

        func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray? = nil) -> MLXArray {
            var x = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: eps)
            if let gate {
                if preservePrecision {
                    return (silu(gate.asType(.float32)) * x.asType(.float32)).asType(
                        hiddenStates.dtype)
                }
                x = x * silu(gate)
            }
            return preservePrecision ? x.asType(hiddenStates.dtype) : x
        }
    }

    final class Attention: Module {
        let numKeyValueHeads: Int
        let numAttentionHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        let rotaryEmbedding: RotaryEmbedding

        init(_ config: MiniCPMVConfiguration.TextConfiguration, preserveNormPrecision: Bool = false)
        {
            numKeyValueHeads = config.kvHeads
            numAttentionHeads = config.attentionHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)
            _qProj.wrappedValue = Linear(
                config.hiddenSize, numAttentionHeads * headDim * 2, bias: config.attentionBias,
            )
            _kProj.wrappedValue = Linear(
                config.hiddenSize, numKeyValueHeads * headDim, bias: config.attentionBias,
            )
            _vProj.wrappedValue = Linear(
                config.hiddenSize, numKeyValueHeads * headDim, bias: config.attentionBias,
            )
            _oProj.wrappedValue = Linear(
                numAttentionHeads * headDim, config.hiddenSize, bias: config.attentionBias,
            )
            _qNorm.wrappedValue =
                preserveNormPrecision
                ? MiniCPMOCRRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
                : RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            _kNorm.wrappedValue =
                preserveNormPrecision
                ? MiniCPMOCRRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
                : RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

            rotaryEmbedding = RotaryEmbedding(
                dim: Int(Float(headDim) * config.partialRotaryFactor),
                base: config.ropeTheta,
                mropeSection: config.ropeParameters["mrope_section"]?.asInts() ?? [11, 11, 10],
            )
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIDs: MLXArray?,
        )
            -> MLXArray
        {
            let batch = x.dim(0)
            let length = x.dim(1)
            let qSplit = qProj(x).reshaped(batch, length, numAttentionHeads, -1)
                .split(parts: 2, axis: -1)
            var queries = qSplit[0]
            let gate = qSplit[1].reshaped(batch, length, -1)
            var keys = kProj(x)
            var values = vProj(x)

            queries = qNorm(queries).transposed(0, 2, 1, 3)
            keys = kNorm(keys.reshaped(batch, length, numKeyValueHeads, -1))
                .transposed(0, 2, 1, 3)
            values = values.reshaped(batch, length, numKeyValueHeads, -1)
                .transposed(0, 2, 1, 3)

            var kvSeqLen = keys.dim(-2)
            var positionIDs = positionIDs
            if positionIDs == nil {
                let offset = cache?.offset ?? 0
                kvSeqLen += offset + 1
                var base = MLXArray(stride(from: offset, to: offset + length, by: 1))
                    .asType(.int32)
                base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
                positionIDs = tiled(base[.newAxis, 0..., 0...], repetitions: [3, 1, 1])
            } else if let cache {
                kvSeqLen += cache.offset + 1
            }

            let (cosValues, sinValues) = rotaryEmbedding(x: values, positionIDs: positionIDs!)
            (queries, keys) = applyMultimodalRotaryPosEmb(
                q: queries, k: keys, cos: cosValues, sin: sinValues,
            )

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode =
                mask.map { .array($0[.ellipsis, 0 ..< kvSeqLen]) } ?? .none
            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: attentionMask,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)

            return oProj(output * sigmoid(gate))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    final class GatedDeltaNet: Module {
        let useOCRPrecision: Bool
        let hiddenSize: Int
        let numVHeads: Int
        let numKHeads: Int
        let headKDim: Int
        let headVDim: Int
        let keyDim: Int
        let valueDim: Int
        let convKernelSize: Int
        let convDim: Int

        @ModuleInfo(key: "conv1d") var conv1d: Conv1d
        @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
        @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
        @ModuleInfo(key: "in_proj_b") var inProjB: Linear
        @ModuleInfo(key: "in_proj_a") var inProjA: Linear
        @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
        @ParameterInfo(key: "A_log") var aLog: MLXArray
        @ModuleInfo(key: "norm") var norm: RMSNormGated
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: MiniCPMVConfiguration.TextConfiguration, useOCRPrecision: Bool = false) {
            self.useOCRPrecision = useOCRPrecision
            hiddenSize = config.hiddenSize
            numVHeads = config.linearNumValueHeads
            numKHeads = config.linearNumKeyHeads
            headKDim = config.linearKeyHeadDim
            headVDim = config.linearValueHeadDim
            keyDim = headKDim * numKHeads
            valueDim = headVDim * numVHeads
            convKernelSize = config.linearConvKernelDim
            convDim = keyDim * 2 + valueDim
            _conv1d.wrappedValue = Conv1d(
                inputChannels: convDim,
                outputChannels: convDim,
                kernelSize: convKernelSize,
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: convDim,
                bias: false,
            )
            _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
            _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
            _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
            _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
            _dtBias.wrappedValue = MLXArray.ones([numVHeads])
            _aLog.wrappedValue = log(MLXRandom.uniform(low: 0, high: 16, [numVHeads]))
            _norm.wrappedValue = RMSNormGated(
                dimensions: headVDim, eps: config.rmsNormEps, preservePrecision: useOCRPrecision)
            _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)
            super.init()
        }

        func callAsFunction(_ inputs: MLXArray, mask: MLXArray? = nil, cache: MambaCache? = nil)
            -> MLXArray
        {
            let batch = inputs.dim(0)
            let sequence = inputs.dim(1)
            var mixedQKV = inProjQKV(inputs)
            let z = inProjZ(inputs).reshaped(batch, sequence, numVHeads, headVDim)
            let b = inProjB(inputs)
            let a = inProjA(inputs)
            let convState =
                cache?[0]
                ?? MLXArray.zeros(
                    [batch, max(0, convKernelSize - 1), convDim], dtype: inputs.dtype)
            if let mask {
                mixedQKV = MLX.where(mask[.ellipsis, .newAxis], mixedQKV, 0)
            }
            let convInput = concatenated([convState, mixedQKV], axis: 1)
            if let cache, convKernelSize > 1 {
                cache[0] = contiguous(convInput[0..., (-(convKernelSize - 1))..., 0...])
            }

            let convPre: MLXArray
            if useOCRPrecision, sequence == 1,
                [.bfloat16, .float16].contains(conv1d.weight.dtype)
            {
                let weight = conv1d.weight[0..., 0..., 0].transposed(1, 0)
                convPre =
                    (convInput.asType(.float32) * weight[.newAxis, 0..., 0...].asType(.float32))
                    .sum(axis: 1, keepDims: true).asType(convInput.dtype)
            } else {
                convPre = conv1d(convInput)
            }
            let convOut = silu(convPre)
            let split = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
            let q = split[0].reshaped(batch, sequence, numKHeads, headKDim)
            let k = split[1].reshaped(batch, sequence, numKHeads, headKDim)
            let v = split[2].reshaped(batch, sequence, numVHeads, headVDim)
            var state = cache?[1]
            let dtype = q.dtype
            let invScale = pow(Float(headKDim), -0.5)
            let normEps: Float = useOCRPrecision ? 1e-6 / Float(headKDim) : 1e-6
            let qNormed =
                MLXArray(pow(invScale, 2)).asType(dtype)
                * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: normEps)
            let kNormed =
                MLXArray(invScale).asType(dtype)
                * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: normEps)
            var out: MLXArray
            if useOCRPrecision {
                (out, state) = gatedDeltaUpdate(
                    q: qNormed, k: kNormed, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
                    state: state, mask: mask, useKernel: !training)
            } else {
                (out, state) = miniCPMVGatedDeltaUpdate(
                    q: qNormed, k: kNormed, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
                    state: state, mask: mask)
            }
            if let cache {
                cache[1] = state
            }
            out = norm(out, gate: z)
            return outProj(out.reshaped(batch, sequence, -1))
        }
    }

    final class SparseMoEBlock: Module, UnaryLayer {
        let normTopKProbability: Bool
        let numExperts: Int
        let topK: Int
        @ModuleInfo(key: "gate") var gate: Linear
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU?
        var expertStore: ExpertWeightStore?
        var layerIndex = 0
        @ModuleInfo(key: "shared_expert") var sharedExpert: MLP
        @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

        init(_ config: MiniCPMVConfiguration.TextConfiguration) {
            normTopKProbability = config.normTopKProbability
            numExperts = config.numExperts
            topK = config.numExpertsPerToken
            _gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.numExperts,
            )
            _sharedExpert.wrappedValue = MLP(
                dimensions: config.hiddenSize,
                hiddenDimensions: config.sharedExpertIntermediateSize,
            )
            _sharedExpertGate.wrappedValue = Linear(config.hiddenSize, 1, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray { forward(x) }
        func forward(_ x: MLXArray, nextPrediction: (SparseMoEBlock, MLXArray)? = nil) -> MLXArray {
            let gates = softmax(gate(x).asType(.float32), axis: -1, precise: true)
            let kth = gates.dim(-1) - topK
            let indices = argPartition(gates, kth: kth, axis: -1)[.ellipsis, kth...]
            var scores = takeAlong(gates, indices, axis: -1)
            if normTopKProbability {
                scores = scores / scores.sum(axis: -1, keepDims: true)
            }
            scores = scores.asType(x.dtype)
            if let (next, predictionInput) = nextPrediction,
                let nextStore = next.expertStore, nextStore.options.prefetch
            {
                let predicted = next.predictions(predictionInput)
                eval(indices, predicted)
                nextStore.prefetch(layer: next.layerIndex, experts: predicted.asArray(Int.self))
            }
            let outputs: MLXArray
            if let expertStore {
                outputs = expertStore.selectedExperts(x, indices: indices, layer: layerIndex)
            } else {
                outputs = switchMLP!(x, indices)
            }
            let combined = (outputs * expandedDimensions(scores, axis: -1))
                .sum(axis: -2)
            return combined + sigmoid(sharedExpertGate(x)) * sharedExpert(x)
        }

        func predict(_ x: MLXArray) {
            guard let expertStore, expertStore.options.prefetch else { return }
            expertStore.prefetch(layer: layerIndex, experts: predictions(x).asArray(Int.self))
        }

        private func predictions(_ x: MLXArray) -> MLXArray {
            let logits = gate(x[0..., (x.dim(1) - 1)..., 0...])
            let kth = numExperts - topK
            return argPartition(logits, kth: kth, axis: -1)[.ellipsis, kth...]
        }

    }

    final class DecoderLayer: Module {
        let isLinear: Bool
        @ModuleInfo(key: "self_attn") var selfAttention: Attention?
        @ModuleInfo(key: "linear_attn") var linearAttention: GatedDeltaNet?
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
        @ModuleInfo(key: "mlp") var mlp: Module

        init(
            _ config: MiniCPMVConfiguration.TextConfiguration, layerIndex: Int,
            preserveNormPrecision: Bool = false
        ) {
            isLinear = (layerIndex + 1) % config.fullAttentionInterval != 0
            if isLinear {
                _linearAttention.wrappedValue = GatedDeltaNet(
                    config, useOCRPrecision: preserveNormPrecision)
            } else {
                _selfAttention.wrappedValue = Attention(
                    config, preserveNormPrecision: preserveNormPrecision)
            }
            if config.numExperts > 0 {
                _mlp.wrappedValue = SparseMoEBlock(config)
            } else {
                _mlp.wrappedValue = MLP(
                    dimensions: config.hiddenSize,
                    hiddenDimensions: config.intermediateSize,
                )
            }
            _inputLayerNorm.wrappedValue =
                preserveNormPrecision
                ? MiniCPMOCRRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
                : RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _postAttentionLayerNorm.wrappedValue =
                preserveNormPrecision
                ? MiniCPMOCRRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
                : RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionMask: MLXArray?,
            ssmMask: MLXArray?,
            cache: KVCache?,
            positionIDs: MLXArray?,
            nextLayer: DecoderLayer? = nil,
        )
            -> MLXArray
        {
            let residual: MLXArray =
                if isLinear {
                    linearAttention!(
                        inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache,
                    )
                } else {
                    selfAttention!(
                        inputLayerNorm(x),
                        mask: attentionMask,
                        cache: cache,
                        positionIDs: positionIDs,
                    )
                }
            let h = x + residual
            var nextPrediction: (SparseMoEBlock, MLXArray)?
            if let nextLayer, let nextMoE = nextLayer.mlp as? SparseMoEBlock {
                let predictionInput = nextLayer.postAttentionLayerNorm(h)
                if (mlp as? SparseMoEBlock)?.expertStore?.options.sharedExpertBuffers == true {
                    nextPrediction = (nextMoE, predictionInput)
                } else {
                    nextMoE.predict(predictionInput)
                }
            }
            let normed = postAttentionLayerNorm(h)
            if let sparse = mlp as? SparseMoEBlock {
                return h + sparse.forward(normed, nextPrediction: nextPrediction)
            }
            return h + (mlp as! UnaryLayer)(normed)
        }
    }

    final class Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm
        let ssmIndex = 0
        let fullAttentionIndex: Int

        init(_ config: MiniCPMVConfiguration.TextConfiguration, preserveNormPrecision: Bool = false)
        {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize,
            )
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
                DecoderLayer(config, layerIndex: $0, preserveNormPrecision: preserveNormPrecision)
            }
            _norm.wrappedValue =
                preserveNormPrecision
                ? MiniCPMOCRRMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
                : RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            fullAttentionIndex = config.fullAttentionInterval - 1
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            positionIDs: MLXArray? = nil,
        )
            -> MLXArray
        {
            var h = inputsEmbeds ?? embedTokens(inputs)
            let cacheArray = cache ?? Array(repeating: nil as KVCache?, count: layers.count)
            let attentionMaskMode = createAttentionMask(
                h: h, cache: cacheArray[fullAttentionIndex], returnArray: true,
            )
            let attentionMask: MLXArray? =
                if case .array(let maskArray) = attentionMaskMode {
                    maskArray
                } else {
                    nil
                }
            let ssmMask = createSSMMask(h: h, cache: cacheArray[ssmIndex] as? MambaCache)
            for (index, layer) in layers.enumerated() {
                h = layer(
                    h,
                    attentionMask: attentionMask,
                    ssmMask: layer.isLinear ? ssmMask : nil,
                    cache: cacheArray[index],
                    positionIDs: positionIDs,
                    nextLayer: index + 1 < layers.count ? layers[index + 1] : nil,
                )
            }
            return norm(h)
        }
    }

    final class LanguageModel: Module, QuantizationPolicyProviding {
        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?
        let config: MiniCPMVConfiguration
        let kvHeads: [Int]

        func quantizationPredicate(path: String, module: Module) -> QuantizationDecision {
            if config.textConfiguration.numExperts > 0,
                path.hasSuffix("mlp.gate") || path.hasSuffix("shared_expert_gate")
            {
                return .parameters(.init(groupSize: 64, bits: 8))
            }
            return .useDefault
        }

        init(_ config: MiniCPMVConfiguration) {
            self.config = config
            _model.wrappedValue = Model(
                config.textConfiguration, preserveNormPrecision: config.isOCR)
            kvHeads = Array(
                repeating: config.textConfiguration.kvHeads,
                count: config.textConfiguration.hiddenLayers,
            )
            if !config.textConfiguration.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.textConfiguration.hiddenSize,
                    config.textConfiguration.vocabularySize,
                    bias: false,
                )
            }
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache]? = nil,
            state: LMOutput.State? = nil,
            positionIDs: MLXArray? = nil,
        )
            -> LMOutput
        {
            let typedCache = cache?.map { Optional($0) }
            var out = model(
                inputs,
                inputsEmbeds: inputsEmbeds,
                cache: typedCache,
                positionIDs: positionIDs,
            )
            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }
            return LMOutput(logits: out, state: state)
        }

        func makeCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
            try model.layers.map { layer in
                if layer.isLinear {
                    return MambaCache()
                }
                return try makeAttentionKVCache(configuration: configuration)
            }
        }
    }
}

// MARK: - Vision

private enum MiniCPMVVision {
    static func checkConvWeight(_ array: MLXArray) -> Bool {
        guard array.ndim == 4 else {
            return false
        }
        let shape = array.shape
        return shape[0] >= shape[1] && shape[0] >= shape[2] && shape[1] == shape[2]
    }

    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: MiniCPMVConfiguration.VisionConfiguration) {
            heads = config.attentionHeads
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _kProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _vProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _qProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let batch = x.dim(0)
            let length = x.dim(1)
            let q = qProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let k = kProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let v = vProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let output = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: nil,
            )
            return outProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let activation: GELU

        init(_ config: MiniCPMVConfiguration.VisionConfiguration) {
            _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
            activation = GELU(approximation: .fast)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    final class EncoderLayer: Module {
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo var mlp: MLP

        init(_ config: MiniCPMVConfiguration.VisionConfiguration) {
            _layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            _layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            _selfAttention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(config)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = x + selfAttention(layerNorm1(x))
            h = h + mlp(layerNorm2(h))
            return h
        }
    }

    final class Encoder: Module {
        @ModuleInfo(key: "layers") var layers: [EncoderLayer]

        init(_ config: MiniCPMVConfiguration.VisionConfiguration) {
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in EncoderLayer(config) }
            super.init()
        }
    }

    final class Embeddings: Module {
        let config: MiniCPMVConfiguration.VisionConfiguration
        let patchSize: Int
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        init(_ config: MiniCPMVConfiguration.VisionConfiguration) {
            self.config = config
            patchSize = config.patchSize
            _patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.channels,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: true,
            )
            let side = config.imageSize / config.patchSize
            _positionEmbedding.wrappedValue = Embedding(
                embeddingCount: side * side,
                dimensions: config.hiddenSize,
            )
            super.init()
        }

        func callAsFunction(_ pixelValues: MLXArray, targetSizes: MLXArray) -> MLXArray {
            let batch = pixelValues.dim(0)
            let height = pixelValues.dim(1)
            let width = pixelValues.dim(2)
            let embeddings: MLXArray
            if height == patchSize, width % patchSize == 0 {
                let numPatches = width / patchSize
                let patches =
                    pixelValues
                    .reshaped(batch, patchSize, numPatches, patchSize, config.channels)
                    .transposed(0, 2, 1, 3, 4)
                    .reshaped(batch, numPatches, -1)
                let weight = patchEmbedding.weight.reshaped(config.hiddenSize, -1).transposed(1, 0)
                var projected = patches.matmul(weight)
                if let bias = patchEmbedding.bias {
                    projected = projected + bias
                }
                embeddings = projected
            } else {
                embeddings = patchEmbedding(pixelValues).flattened(start: 1, end: 2)
            }

            let positionIDs = buildPositionIDs(targetSizes: targetSizes, count: embeddings.dim(1))
            return embeddings + positionEmbedding(positionIDs)
        }

        private func buildPositionIDs(targetSizes: MLXArray, count: Int) -> MLXArray {
            let gridSide = Int(round(sqrt(Float(positionEmbedding.weight.dim(0)))))
            var all: [Int] = []
            all.reserveCapacity(targetSizes.dim(0) * count)
            let sizes = targetSizes.asArray(Int.self)
            for batchIndex in 0 ..< targetSizes.dim(0) {
                let targetH = max(sizes[batchIndex * 2], 1)
                let targetW = max(sizes[batchIndex * 2 + 1], 1)
                let heightBuckets = miniCPMVPositionBuckets(count: targetH, gridSide: gridSide)
                let widthBuckets = miniCPMVPositionBuckets(count: targetW, gridSide: gridSide)
                for index in 0 ..< count {
                    let h = index / targetW
                    let w = index % targetW
                    let bucketH = heightBuckets[min(h, targetH - 1)]
                    let bucketW = widthBuckets[w]
                    all.append(bucketH * gridSide + bucketW)
                }
            }
            return MLXArray(all, [targetSizes.dim(0), count])
        }
    }

    final class VisionModel: Module {
        @ModuleInfo var embeddings: Embeddings
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        init(_ config: MiniCPMVConfiguration.VisionConfiguration) {
            _embeddings.wrappedValue = Embeddings(config)
            _encoder.wrappedValue = Encoder(config)
            _postLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            super.init()
        }
    }
}

private final class MiniCPMVCrossAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(dimensions: Int, heads: Int) {
        self.heads = heads
        headDim = dimensions / heads
        scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(dimensions, dimensions, bias: true)
        _kProj.wrappedValue = Linear(dimensions, dimensions, bias: true)
        _vProj.wrappedValue = Linear(dimensions, dimensions, bias: true)
        _outProj.wrappedValue = Linear(dimensions, dimensions, bias: true)
        super.init()
    }

    func callAsFunction(queries: MLXArray, keys: MLXArray, values: MLXArray) -> MLXArray {
        let batch = queries.dim(0)
        let qLength = queries.dim(1)
        let kvLength = keys.dim(1)
        let q = qProj(queries).reshaped(batch, qLength, heads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(keys).reshaped(batch, kvLength, heads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(values).reshaped(batch, kvLength, heads, headDim).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil,
        )
        return outProj(out.transposed(0, 2, 1, 3).reshaped(batch, qLength, -1))
    }
}

private final class MiniCPMVVitMerger: Module {
    let visionHiddenSize: Int
    let group: (Int, Int)
    let groupTokens: Int
    let groupHiddenSize: Int
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "self_attn") var selfAttention: MiniCPMVCrossAttention
    @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    let activation: GELU

    init(_ config: MiniCPMVConfiguration.VisionConfiguration, approximateGELU: Bool = false) {
        activation = GELU(approximation: approximateGELU ? .tanh : .precise)
        visionHiddenSize = config.hiddenSize
        group = config.windowKernelSize
        groupTokens = group.0 * group.1
        groupHiddenSize = config.hiddenSize * groupTokens
        _preNorm.wrappedValue = LayerNorm(dimensions: groupHiddenSize, eps: 1e-6)
        _selfAttention.wrappedValue = MiniCPMVCrossAttention(
            dimensions: config.hiddenSize,
            heads: config.attentionHeads,
        )
        _layerNorm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
        _linear1.wrappedValue = Linear(groupHiddenSize, config.windowIntermediateSize, bias: true)
        _linear2.wrappedValue = Linear(config.windowIntermediateSize, config.hiddenSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gridH: Int, gridW: Int) -> (MLXArray, Int, Int) {
        let mergedH = gridH / group.0
        let mergedW = gridW / group.1
        var windows = x.reshaped(gridH, gridW, visionHiddenSize)
            .reshaped(mergedH, group.0, mergedW, group.1, visionHiddenSize)
            .transposed(0, 2, 1, 3, 4)
            .reshaped(mergedH * mergedW, groupTokens, visionHiddenSize)
        windows =
            windows
            + selfAttention(
                queries: layerNorm1(windows),
                keys: layerNorm1(windows),
                values: layerNorm1(windows),
            )
        let residual = windows.mean(axis: 1)
        let merged = windows.reshaped(mergedH * mergedW, groupHiddenSize)
        return (linear2(activation(linear1(preNorm(merged)))) + residual, mergedH, mergedW)
    }
}

private final class MiniCPMVMergerBlock: Module, UnaryLayer {
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    let activation = GELU(approximation: .precise)

    init(inputSize: Int, outputSize: Int) {
        _preNorm.wrappedValue = LayerNorm(dimensions: inputSize, eps: 1e-6)
        _linear1.wrappedValue = Linear(inputSize, inputSize, bias: true)
        _linear2.wrappedValue = Linear(inputSize, outputSize, bias: true)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(activation(linear1(preNorm(x))))
    }
}

private final class MiniCPMVMerger: Module {
    let mergeKernel: (Int, Int)
    @ModuleInfo(key: "mlp") var layers: [MiniCPMVMergerBlock]

    init(hiddenSize: Int, outputSize: Int, times: Int, mergeKernel: (Int, Int)) {
        self.mergeKernel = mergeKernel
        let mergeTokens = mergeKernel.0 * mergeKernel.1
        _layers.wrappedValue = (0 ..< times).map { index in
            MiniCPMVMergerBlock(
                inputSize: hiddenSize * mergeTokens,
                outputSize: index == times - 1 ? outputSize : hiddenSize,
            )
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gridH: Int, gridW: Int) -> (MLXArray, Int, Int) {
        var curH = gridH
        var curW = gridW
        var hidden = x
        for layer in layers {
            let mergedH = curH / mergeKernel.0
            let mergedW = curW / mergeKernel.1
            let inner = hidden.dim(-1)
            hidden = hidden.reshaped(curH, curW, inner)
                .reshaped(mergedH, mergeKernel.0, mergedW, mergeKernel.1, inner)
                .transposed(0, 2, 1, 3, 4)
                .reshaped(mergedH * mergedW, inner * mergeKernel.0 * mergeKernel.1)
            hidden = layer(hidden)
            curH = mergedH
            curW = mergedW
        }
        return (hidden, curH, curW)
    }
}

// MARK: - Model

public final class MiniCPMV: Module, VLMModel, KVCacheDimensionProvider, ExpertStreamingModel,
    MixedQuantizationPolicyProviding
{
    @ModuleInfo(key: "language_model") private var languageModel: MiniCPMVLanguage.LanguageModel
    @ModuleInfo(key: "vision_tower") private var visionTower: MiniCPMVVision.VisionModel
    @ModuleInfo(key: "vit_merger") private var vitMerger: MiniCPMVVitMerger
    @ModuleInfo(key: "merger") private var merger: MiniCPMVMerger

    public let config: MiniCPMVConfiguration
    private var expertStore: ExpertWeightStore?
    private static let positionDeltaKey = LMOutput.Key<MLXArray>("minicpmv.positionDelta")

    public func quantizationPredicate(path: String, module: Module) -> QuantizationDecision {
        if path.hasPrefix("vit_merger") || path.hasPrefix("merger")
            || ["linear_attn", "embed_tokens", "lm_head"].contains(where: path.contains)
        {
            return .skip
        }
        return languageModel.quantizationPredicate(path: path, module: module)
    }

    /// The 4.7 mixed recipe preserves routed experts and compresses resident weights.
    /// Unlike the upstream accuracy-oriented predicate, this includes embeddings,
    /// linear attention and mergers. Existing packed weights are always preserved.
    public func mixedQuantizationPredicate(path: String, module: Module) -> QuantizationDecision {
        if config.modelType == "minicpmv4_7" {
            if path.contains(".switch_mlp.") { return .skip }
            if !path.hasPrefix("language_model.") {
                return .parameters(.init(groupSize: 64, bits: 8))
            }
        } else if path.hasPrefix("vit_merger") || path.hasPrefix("merger")
            || ["linear_attn", "embed_tokens", "lm_head"].contains(where: path.contains)
        {
            return .skip
        }
        return languageModel.quantizationPredicate(path: path, module: module)
    }

    func configureExpertStreaming(
        urls: [URL], options: ExpertStreamingOptions,
        quantization: BaseConfiguration.Quantization?,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization?
    ) throws -> Set<String> {
        let store = try ExpertWeightStore(
            urls: urls, configuration: config, options: options,
            quantization: quantization, perLayerQuantization: perLayerQuantization)
        for (index, layer) in languageModel.model.layers.enumerated() {
            guard let moe = layer.mlp as? MiniCPMVLanguage.SparseMoEBlock else { continue }
            try moe.update(modules: ModuleChildren(values: ["switch_mlp": .none]), verify: .all)
            moe.expertStore = store
            moe.layerIndex = index
        }
        expertStore = store
        return store.excludedWeightNames
    }

    func checkInferenceError() throws { try expertStore?.checkError() }

    public init(_ config: MiniCPMVConfiguration) {
        self.config = config
        _languageModel.wrappedValue = MiniCPMVLanguage.LanguageModel(config)
        _visionTower.wrappedValue = MiniCPMVVision.VisionModel(config.visionConfiguration)
        _vitMerger.wrappedValue = MiniCPMVVitMerger(
            config.visionConfiguration, approximateGELU: config.isOCR)
        _merger.wrappedValue = MiniCPMVMerger(
            hiddenSize: config.visionConfiguration.hiddenSize,
            outputSize: config.textConfiguration.hiddenSize,
            times: config.mergerTimes,
            mergeKernel: config.mergeKernelSize,
        )
        super.init()
    }

    public var toolCallFormat: ToolCallFormat? { config.modelType == "minicpmv4_7" ? .qwen35 : nil }
    public var reasoningConfig: ReasoningConfig? {
        config.modelType == "minicpmv4_7" ? QwenReasoningProtocol.tagged : nil
    }

    public var vocabularySize: Int {
        config.vocabularySize
    }

    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try languageModel.makeCache(configuration: configuration)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        var inputIDs = input.text.tokens
        if config.isOCR, inputIDs.ndim != 2 || inputIDs.dim(0) != 1 {
            throw VLMError.processing("MiniCPM-OCR expects a single input sequence")
        }
        var inputEmbeddings: MLXArray?
        if let image = input.image {
            let textEmbeddings = languageModel.model.embedTokens(inputIDs)
            let mode =
                config.isOCR
                ? try miniCPMOCRDownsampleMode(
                    frames: image.frames ?? [], bounds: input.video?.pixels)
                : config.downsampleMode
            let features = try getVisionEmbeddings(
                flatPixels: image.pixels,
                targetFrames: image.frames ?? [],
                downsampleMode: mode,
            )
            .asType(textEmbeddings.dtype)

            inputEmbeddings = try mergeVisionFeatures(
                features: features,
                inputEmbeddings: textEmbeddings,
                bounds: input.video?.pixels,
            )
        }
        if config.isOCR, let mask = input.text.mask {
            guard mask.shape == inputIDs.shape else {
                throw VLMError.processing("MiniCPM-OCR mask must match tokens")
            }
            let valid = mask.asArray(Int.self).enumerated().compactMap {
                $0.element != 0 ? $0.offset : nil
            }
            guard let first = valid.first, let last = valid.last, last - first + 1 == valid.count
            else {
                throw VLMError.processing(
                    "MiniCPM-OCR requires nonempty input with contiguous left or right padding")
            }
            // Remove padding before recurrent prefill so it cannot alter convolution or delta state.
            inputIDs = inputIDs[0..., first ... last]
            inputEmbeddings = inputEmbeddings?[0..., first ... last, 0...]
        }
        var outputState = state ?? LMOutput.State()
        var positions: MLXArray?
        if config.isOCR {
            let offset = cache[languageModel.model.fullAttentionIndex].offset
            let sequential = MLXArray(Array(offset ..< offset + inputIDs.dim(1)))
            positions = tiled(sequential[.newAxis, .newAxis, 0...], repetitions: [3, 1, 1])
        } else if config.modelType == "minicpmv4_7", let image = input.image,
            let bounds = input.video?.pixels
        {
            guard inputIDs.dim(0) == 1 else {
                throw VLMError.processing("MiniCPM-V canvas positions require a single sequence")
            }
            let values = bounds.asArray(Int.self)
            guard values.count % 2 == 0 else {
                throw VLMError.processing("MiniCPM-V invalid crop bounds")
            }
            guard
                stride(from: 0, to: values.count, by: 2).allSatisfy({ values[$0] <= values[$0 + 1] }
                )
            else {
                throw VLMError.processing("MiniCPM-V reversed crop bounds")
            }
            let ranges = stride(from: 0, to: values.count, by: 2).map {
                values[$0] ..< values[$0 + 1]
            }
            let canvas = try MiniCPMVCanvasPositions.make(
                tokens: inputIDs.asArray(Int.self), bounds: ranges,
                grids: image.frames ?? [], configuration: config)
            let offset = cache[languageModel.model.fullAttentionIndex].offset
            let previousDelta = state?[Self.positionDeltaKey]?.item(Int.self) ?? 0
            positions = canvas.positions + offset + previousDelta
            outputState[Self.positionDeltaKey] = MLXArray(previousDelta + canvas.delta)
        } else {
            positions = continuationPositions(inputIDs, cache: cache, state: state)
        }
        // Bound prefill activations as well as expert residency. Each chunk is fenced before
        // constructing the next one, otherwise lazy graphs retain the loaded expert weights.
        let count = inputIDs.dim(1)
        let step = prefill.chunkLength(forChunking: count) ?? count
        var output: LMOutput?
        for start in stride(from: 0, to: count, by: max(1, step)) {
            try Task.checkCancellation()
            let range = start ..< min(start + step, count)
            output = autoreleasepool {
                let chunk = languageModel(
                    inputIDs[0..., range], inputsEmbeds: inputEmbeddings?[0..., range, 0...],
                    cache: cache, state: outputState, positionIDs: positions?[0..., 0..., range])
                eval(chunk.logits, cache)
                return chunk
            }
            try checkInferenceError()
            prefill.progress?(range.upperBound, count)
        }
        guard let output else { throw VLMError.processing("MiniCPM-V received an empty prompt") }
        return .logits(output)
    }

    private func continuationPositions(
        _ tokens: MLXArray, cache: [KVCache]?, state: LMOutput.State?
    ) -> MLXArray? {
        guard let delta = state?[Self.positionDeltaKey] else { return nil }
        let offset = cache?[languageModel.model.fullAttentionIndex].offset ?? 0
        let positions = MLXArray(Array(offset ..< offset + tokens.dim(1))) + delta
        return tiled(positions[.newAxis, .newAxis, 0...], repetitions: [3, tokens.dim(0), 1])
    }

    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        languageModel(
            input.tokens, cache: cache, state: state,
            positionIDs: continuationPositions(input.tokens, cache: cache, state: state))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    private func getVisionEmbeddings(
        flatPixels: MLXArray, targetFrames: [THW], downsampleMode: String
    ) throws -> MLXArray {
        var offset = 0
        var outputs: [MLXArray] = []
        let patch = config.visionConfiguration.patchSize
        let dtype = visionTower.embeddings.patchEmbedding.weight.dtype

        for frame in targetFrames {
            let gridH = frame.h
            let gridW = frame.w
            let count = 3 * patch * gridH * gridW * patch
            let chunk = flatPixels[offset ..< (offset + count)]
                .reshaped(3, patch, gridH * gridW * patch)
                .transposed(1, 2, 0)
                .expandedDimensions(axis: 0)
                .asType(dtype)
            offset += count
            let target = MLXArray([gridH, gridW], [1, 2])
            var hidden = visionTower.embeddings(chunk, targetSizes: target)
            hidden = hidden.asType(visionTower.embeddings.patchEmbedding.weight.dtype)
            var currentH = gridH
            var currentW = gridW
            for (index, layer) in visionTower.encoder.layers.enumerated() {
                hidden = layer(hidden)
                if downsampleMode != "4x", index == config.insertLayerID {
                    let (merged, h, w) = vitMerger(hidden[0], gridH: currentH, gridW: currentW)
                    hidden = merged.expandedDimensions(axis: 0)
                    currentH = h
                    currentW = w
                }
            }
            hidden = visionTower.postLayerNorm(hidden)[0]
            let (mergedTokens, _, _) = merger(hidden, gridH: currentH, gridW: currentW)
            outputs.append(mergedTokens)
        }
        guard !outputs.isEmpty else {
            return MLXArray.zeros([0, config.textConfiguration.hiddenSize])
        }
        return concatenated(outputs, axis: 0)
    }

    private func mergeVisionFeatures(
        features: MLXArray,
        inputEmbeddings: MLXArray,
        bounds: MLXArray?,
    )
        throws -> MLXArray
    {
        guard let bounds, bounds.size > 0 else {
            return inputEmbeddings
        }
        let boundValues = bounds.asArray(Int.self)
        var indices: [Int] = []
        for idx in stride(from: 0, to: boundValues.count, by: 2) {
            guard idx + 1 < boundValues.count else {
                continue
            }
            let start = boundValues[idx]
            let end = boundValues[idx + 1]
            if end > start {
                indices.append(contentsOf: start ..< end)
            }
        }
        guard indices.count == features.dim(0) else {
            throw VLMError.processing(
                "MiniCPM-V vision features (\(features.dim(0))) do not match placeholder count (\(indices.count)).",
            )
        }
        let result = inputEmbeddings
        result[0..., MLXArray(indices), 0...] = features.expandedDimensions(axis: 0)
        return result
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        if metadata["format"]?.lowercased() == "mlx" {
            return config.isOCR
                ? Dictionary(
                    uniqueKeysWithValues: weights.map { key, value in
                        (
                            key,
                            Self.normWeightSuffixes.contains(where: key.hasSuffix)
                                ? value.asType(.float32) : value
                        )
                    }) : weights
        }
        return sanitize(weights: weights)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)
        let shiftOCRNorms = weights.contains { key, value in
            key.contains("conv1d.weight") && value.ndim == 3 && value.dim(-1) != 1
        }
        for (originalKey, originalValue) in weights {
            guard let key = Self.canonicalWeightKey(originalKey) else { continue }
            var value = originalValue

            if key.contains("position_ids") {
                continue
            }
            if key.contains("conv1d.weight"), value.ndim == 3, value.dim(-1) != 1 {
                value = value.movedAxis(source: 2, destination: 1)
            }
            if key.hasSuffix("embeddings.patch_embedding.weight"),
                !MiniCPMVVision.checkConvWeight(value)
            {
                value = value.transposed(0, 2, 3, 1)
            }
            if Self.normWeightSuffixes.contains(where: { key.hasSuffix($0) }), value.ndim == 1 {
                if config.isOCR { value = value.asType(.float32) }
                if !config.isOCR || shiftOCRNorms || !originalKey.hasPrefix("language_model.") {
                    value = value + MLXArray(1, dtype: value.dtype)
                }
            }
            sanitized[key] = value
        }
        for layer in 0 ..< config.textConfiguration.hiddenLayers {
            let prefix = "language_model.model.layers.\(layer).mlp"
            if let gateUp = sanitized.removeValue(forKey: prefix + ".experts.gate_up_proj") {
                let middle = gateUp.dim(-2) / 2
                sanitized[prefix + ".switch_mlp.gate_proj.weight"] =
                    gateUp[.ellipsis, ..<middle, 0...]
                sanitized[prefix + ".switch_mlp.up_proj.weight"] =
                    gateUp[.ellipsis, middle..., 0...]
            }
            if let down = sanitized.removeValue(forKey: prefix + ".experts.down_proj") {
                sanitized[prefix + ".switch_mlp.down_proj.weight"] = down
            }
        }
        if config.textConfiguration.tieWordEmbeddings {
            sanitized["language_model.lm_head.weight"] = nil
        }
        return sanitized
    }

    static let normWeightSuffixes = [
        ".input_layernorm.weight", ".post_attention_layernorm.weight", "model.norm.weight",
        ".q_norm.weight", ".k_norm.weight",
    ]

    static func canonicalWeightKey(_ original: String) -> String? {
        let key = original.hasPrefix("model.") ? String(original.dropFirst(6)) : original
        if key.hasPrefix("language_model.model.") || key.hasPrefix("language_model.lm_head.") {
            return key
        }
        if key.hasPrefix("language_model.") { return "language_model.model." + key.dropFirst(15) }
        if key.hasPrefix("lm_head.") { return "language_model." + key }
        if key.hasPrefix("vision_tower.vit_merger.") { return String(key.dropFirst(13)) }
        if key.hasPrefix("vision_tower.") || key.hasPrefix("vit_merger.")
            || key.hasPrefix("merger.")
        {
            return key
        }
        if key.hasPrefix("vpm.") { return "vision_tower." + key.dropFirst(4) }
        if key.hasPrefix("llm.") { return "language_model.model." + key.dropFirst(4) }
        if key.hasPrefix("visual.") { return "vision_tower." + key.dropFirst(7) }
        return nil
    }
}
