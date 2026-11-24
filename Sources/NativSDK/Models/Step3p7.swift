import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct Step3p7Configuration: Decodable, Sendable {
    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let width: Int
        public let layers: Int
        public let heads: Int
        public let channels: Int
        public let imageSize: Int
        public let mlpRatio: Float
        public let patchSize: Int
        public let hiddenActivation: String
        public let layerNormEps: Float
        public let useCLSToken: Bool
        public let useLNPre: Bool
        public let useLNPost: Bool
        public let useAbsolutePositionEmbedding: Bool
        public let useRoPE2D: Bool
        public let layerScaleInitValue: Float?
        public let ropeTheta: Float

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case width
            case layers
            case heads
            case channels = "num_channels"
            case imageSize = "image_size"
            case mlpRatio = "mlp_ratio"
            case patchSize = "patch_size"
            case hiddenActivation = "hidden_act"
            case layerNormEps = "layer_norm_eps"
            case useCLSToken = "use_cls_token"
            case misspelledUseCLSToken = "ues_cls_token"
            case useLNPre = "use_ln_pre"
            case useLNPost = "use_ln_post"
            case useAbsolutePositionEmbedding = "use_abs_posemb"
            case useRoPE2D = "use_rope2d"
            case layerScaleInitValue = "ls_init_value"
            case ropeTheta = "rope_theta"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType) ?? "perception_encoder"
            width = try c.decodeIfPresent(Int.self, forKey: .width) ?? 1536
            layers = try c.decodeIfPresent(Int.self, forKey: .layers) ?? 47
            heads = try c.decodeIfPresent(Int.self, forKey: .heads) ?? 16
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 728
            mlpRatio =
                try c.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? Float(8960.0 / 1536.0)
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            hiddenActivation =
                try c.decodeIfPresent(String.self, forKey: .hiddenActivation) ?? "quick_gelu"
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
            useCLSToken =
                try c.decodeIfPresent(Bool.self, forKey: .misspelledUseCLSToken)
                ?? c.decodeIfPresent(Bool.self, forKey: .useCLSToken)
                ?? false
            useLNPre = try c.decodeIfPresent(Bool.self, forKey: .useLNPre) ?? true
            useLNPost = try c.decodeIfPresent(Bool.self, forKey: .useLNPost) ?? false
            useAbsolutePositionEmbedding =
                try c.decodeIfPresent(Bool.self, forKey: .useAbsolutePositionEmbedding) ?? true
            useRoPE2D = try c.decodeIfPresent(Bool.self, forKey: .useRoPE2D) ?? true
            layerScaleInitValue =
                try c.decodeIfPresent(Float.self, forKey: .layerScaleInitValue)
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        }
    }

    public struct TextConfiguration: Decodable, Sendable {
        public struct AttentionOtherSetting: Decodable, Sendable {
            public let attentionType: String?
            public let attentionHeads: Int?
            public let attentionGroups: Int?
            public let headDim: Int?

            enum CodingKeys: String, CodingKey {
                case attentionType = "attention_type"
                case attentionHeads = "num_attention_heads"
                case attentionGroups = "num_attention_groups"
                case headDim = "head_dim"
            }
        }

        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let attentionGroups: Int
        public let hiddenLayers: Int
        public let vocabularySize: Int
        public let rmsNormEps: Float
        public let moeIntermediateSize: Int
        public let moeExperts: Int
        public let moeTopK: Int
        public let ropeTheta: [Float]
        public let ropeScaling: [String: StringOrNumber]?
        public let maxPositionEmbeddings: Int
        public let sharedExpertDim: Int
        public let headDim: Int
        public let normExpertWeight: Bool
        public let layerTypes: [String]
        public let slidingWindow: Int?
        public let padTokenID: Int
        public let useHeadWiseAttentionGate: Bool
        public let moeRouterScalingFactor: Float
        public let swigluLimits: [Float?]
        public let swigluSharedLimits: [Float?]
        public let yarnOnlyTypes: [String]?
        public let partialRotaryFactors: [Float]
        public let moeLayers: Set<Int>
        public let eosTokenIDs: [Int]
        public let bosTokenID: Int?
        public let attentionOtherSetting: AttentionOtherSetting?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case attentionGroups = "num_attention_groups"
            case hiddenLayers = "num_hidden_layers"
            case vocabularySize = "vocab_size"
            case rmsNormEps = "rms_norm_eps"
            case moeIntermediateSize = "moe_intermediate_size"
            case moeExperts = "moe_num_experts"
            case moeTopK = "moe_top_k"
            case ropeTheta = "rope_theta"
            case ropeScaling = "rope_scaling"
            case maxPositionEmbeddings = "max_position_embeddings"
            case sharedExpertDim = "share_expert_dim"
            case sharedExpertDims = "share_expert_dims"
            case headDim = "head_dim"
            case normExpertWeight = "norm_expert_weight"
            case layerTypes = "layer_types"
            case slidingWindow = "sliding_window"
            case padTokenID = "pad_token_id"
            case useHeadWiseAttentionGate = "use_head_wise_attn_gate"
            case moeRouterScalingFactor = "moe_router_scaling_factor"
            case swigluLimits = "swiglu_limits"
            case swigluSharedLimits = "swiglu_limits_shared"
            case yarnOnlyTypes = "yarn_only_types"
            case partialRotaryFactors = "partial_rotary_factors"
            case moeLayers = "moe_layers_enum"
            case eosTokenIDs = "eos_token_id"
            case bosTokenID = "bos_token_id"
            case attentionOtherSetting = "attention_other_setting"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "step3p5"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 11264
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 64
            attentionGroups = try c.decodeIfPresent(Int.self, forKey: .attentionGroups) ?? 8
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 45
            vocabularySize =
                try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 128_896
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
            moeIntermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 1280
            moeExperts = try c.decodeIfPresent(Int.self, forKey: .moeExperts) ?? 288
            moeTopK = try c.decodeIfPresent(Int.self, forKey: .moeTopK) ?? 8
            ropeTheta = try Self.decodeFloatArray(
                from: c,
                forKey: .ropeTheta,
                defaultValue: 10000,
                count: hiddenLayers,
            )
            ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 262_144
            sharedExpertDim =
                try c.decodeIfPresent(Int.self, forKey: .sharedExpertDims)
                ?? c.decodeIfPresent(Int.self, forKey: .sharedExpertDim)
                ?? 1280
            headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
            normExpertWeight =
                try c.decodeIfPresent(Bool.self, forKey: .normExpertWeight) ?? true
            layerTypes = try Array(
                (c.decodeIfPresent([String].self, forKey: .layerTypes) ?? [])
                    .prefix(hiddenLayers),
            )
            slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
            padTokenID = try c.decodeIfPresent(Int.self, forKey: .padTokenID) ?? 1
            useHeadWiseAttentionGate =
                try c.decodeIfPresent(Bool.self, forKey: .useHeadWiseAttentionGate) ?? false
            moeRouterScalingFactor =
                try c.decodeIfPresent(Float.self, forKey: .moeRouterScalingFactor) ?? 1
            swigluLimits = try Self.padded(
                c.decodeIfPresent([Float?].self, forKey: .swigluLimits),
                count: hiddenLayers,
            )
            swigluSharedLimits = try Self.padded(
                c.decodeIfPresent([Float?].self, forKey: .swigluSharedLimits),
                count: hiddenLayers,
            )
            yarnOnlyTypes = try c.decodeIfPresent([String].self, forKey: .yarnOnlyTypes)
            partialRotaryFactors = try Self.padded(
                c.decodeIfPresent([Float].self, forKey: .partialRotaryFactors),
                count: hiddenLayers,
                defaultValue: 1,
            )
            moeLayers =
                try Self.decodeMoELayers(from: c, forKey: .moeLayers, hiddenLayers: hiddenLayers)
            if let ids = try? c.decode([Int].self, forKey: .eosTokenIDs) {
                eosTokenIDs = ids
            } else if let id = try? c.decode(Int.self, forKey: .eosTokenIDs) {
                eosTokenIDs = [id]
            } else {
                eosTokenIDs = []
            }
            bosTokenID = try c.decodeIfPresent(Int.self, forKey: .bosTokenID)
            attentionOtherSetting =
                try c.decodeIfPresent(AttentionOtherSetting.self, forKey: .attentionOtherSetting)
        }

        public func isSliding(layer: Int) -> Bool {
            if layer < layerTypes.count {
                return layerTypes[layer] == "sliding_attention"
            }
            return layer % 2 == 0
        }

        public func ropeScaling(layer: Int) -> [String: StringOrNumber]? {
            guard let yarnOnlyTypes, layer < layerTypes.count else {
                return ropeScaling
            }
            return yarnOnlyTypes.contains(layerTypes[layer]) ? ropeScaling : nil
        }

        private static func decodeFloatArray<K: CodingKey>(
            from container: KeyedDecodingContainer<K>,
            forKey key: K,
            defaultValue: Float,
            count: Int,
        )
            throws -> [Float]
        {
            if let values = try container.decodeIfPresent([Float].self, forKey: key) {
                return padded(values, count: count, defaultValue: defaultValue)
            }
            if let value = try container.decodeIfPresent(Float.self, forKey: key) {
                return Array(repeating: value, count: count)
            }
            return Array(repeating: defaultValue, count: count)
        }

        private static func padded(_ values: [Float?]?, count: Int) -> [Float?] {
            var values = Array((values ?? []).prefix(count))
            if values.count < count {
                values.append(contentsOf: Array(repeating: nil, count: count - values.count))
            }
            return values
        }

        private static func padded(
            _ values: [Float]?,
            count: Int,
            defaultValue: Float,
        )
            -> [Float]
        {
            var values = Array((values ?? []).prefix(count))
            if values.count < count {
                values.append(
                    contentsOf: Array(repeating: defaultValue, count: count - values.count))
            }
            return values
        }

        private static func decodeMoELayers<K: CodingKey>(
            from container: KeyedDecodingContainer<K>,
            forKey key: K,
            hiddenLayers: Int,
        )
            throws -> Set<Int>
        {
            if let string = try? container.decodeIfPresent(String.self, forKey: key) {
                return Set(
                    string.split(separator: ",")
                        .compactMap { Int($0.trimmingCharacters(in: .whitespacesAndNewlines)) }
                        .filter { $0 < hiddenLayers },
                )
            }
            if let values = try? container.decodeIfPresent([Int].self, forKey: key) {
                return Set(values.filter { $0 < hiddenLayers })
            }
            return Set(3 ..< hiddenLayers)
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    public let projectorBias: Bool
    public let imageTokenID: Int
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int {
        _imageTokenIndex ?? imageTokenID
    }

    public let vocabularySize: Int
    public let eosTokenIDs: [Int]

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case projectorBias = "projector_bias"
        case imageTokenID = "image_token_id"
        case _imageTokenIndex = "image_token_index"
        case vocabularySize = "vocab_size"
        case eosTokenIDs = "eos_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration = try c.decode(TextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try c.decode(VisionConfiguration.self, forKey: .visionConfiguration)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "step3p7"
        projectorBias = try c.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? false
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? 128_001
        _imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: ._imageTokenIndex)
        vocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .vocabularySize)
            ?? textConfiguration.vocabularySize
        if let ids = try? c.decode([Int].self, forKey: .eosTokenIDs) {
            eosTokenIDs = ids
        } else if let id = try? c.decode(Int.self, forKey: .eosTokenIDs) {
            eosTokenIDs = [id]
        } else {
            eosTokenIDs = textConfiguration.eosTokenIDs
        }
    }
}

public struct Step3VLProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String
    public let imageSize: Int
    public let patchImageSize: Int
    public let maxImageSize: Int
    public let imageFeatureSize: Int
    public let patchFeatureSize: Int
    public let imageToken: String
    public let imageStartToken: String
    public let imageEndToken: String
    public let patchStartToken: String
    public let patchEndToken: String
    public let patchNewlineToken: String
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageSize = "image_size"
        case patchImageSize = "patch_image_size"
        case maxImageSize = "max_image_size"
        case imageFeatureSize = "num_image_feature_size"
        case patchFeatureSize = "num_patch_feature_size"
        case imageToken = "image_token"
        case imageStartToken = "image_start_token"
        case imageEndToken = "image_end_token"
        case patchStartToken = "patch_start_token"
        case patchEndToken = "patch_end_token"
        case patchNewlineToken = "patch_newline_token"
        case imageMean = "image_mean"
        case imageStd = "image_std"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass =
            try c.decodeIfPresent(String.self, forKey: .processorClass) ?? "Step3VLProcessor"
        imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 728
        patchImageSize = try c.decodeIfPresent(Int.self, forKey: .patchImageSize) ?? 504
        maxImageSize = try c.decodeIfPresent(Int.self, forKey: .maxImageSize) ?? 3024
        imageFeatureSize = try c.decodeIfPresent(Int.self, forKey: .imageFeatureSize) ?? 169
        patchFeatureSize = try c.decodeIfPresent(Int.self, forKey: .patchFeatureSize) ?? 81
        imageToken = try c.decodeIfPresent(String.self, forKey: .imageToken) ?? "<im_patch>"
        imageStartToken =
            try c.decodeIfPresent(String.self, forKey: .imageStartToken) ?? "<im_start>"
        imageEndToken = try c.decodeIfPresent(String.self, forKey: .imageEndToken) ?? "<im_end>"
        patchStartToken =
            try c.decodeIfPresent(String.self, forKey: .patchStartToken) ?? "<patch_start>"
        patchEndToken =
            try c.decodeIfPresent(String.self, forKey: .patchEndToken) ?? "<patch_end>"
        patchNewlineToken =
            try c.decodeIfPresent(String.self, forKey: .patchNewlineToken) ?? "<patch_newline>"
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageMean)
            ?? [0.48145466, 0.4578275, 0.40821073]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageStd)
            ?? [0.26862954, 0.26130258, 0.27577711]
    }
}

// MARK: - Processor

public struct Step3VLProcessor: UserInputProcessor {
    private let config: Step3VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Step3VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty else {
            throw VLMError.singleMediaTypeAllowed
        }

        let images = MediaProcessing.images(from: input)
        if images.isEmpty {
            let prompt = renderPrompt(input: input, imageReplacements: [])
            let tokens = tokenizer.encode(text: prompt, addSpecialTokens: false)
            let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
            return LMInput(
                text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)))
        }

        let processed = try images.map {
            try preprocess(image: $0.asCIImage(), processing: input.processing)
        }
        let replacements = processed.map {
            replacement(numPatches: $0.numPatches, newlines: $0.newlines)
        }
        let prompt = renderPrompt(input: input, imageReplacements: replacements)
        let tokens = tokenizer.encode(text: prompt, addSpecialTokens: false)
        let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
        let basePixels = concatenated(processed.map(\.basePixels), axis: 0)
        let patchPixels = processed.flatMap(\.patchPixels)
        let patchCounts = processed.map { THW($0.numPatches, 0, 0) }

        return LMInput(
            text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)),
            image: .init(pixels: basePixels, frames: patchCounts),
            video: patchPixels.isEmpty
                ? nil
                : .init(pixels: concatenated(patchPixels, axis: 0), frames: patchCounts),
        )
    }

    private struct ProcessedImage {
        let basePixels: MLXArray
        let patchPixels: [MLXArray]
        let numPatches: Int
        let newlines: [Bool]?
    }

    private func renderPrompt(input: UserInput, imageReplacements: [String]) -> String {
        let messages: [Chat.Message] =
            switch input.prompt {
            case .text(let text):
                [.user(text)]
            case .chat(let chat):
                chat
            case .messages:
                [.user(input.prompt.description)]
            }

        var result = tokenizer.bosToken ?? "<｜begin▁of▁sentence｜>"
        var imageIndex = 0

        for message in messages {
            result += "<|im_start|>\(roleName(message.role))\n"
            var content = message.content
            if content.contains(config.imageToken), imageIndex < imageReplacements.count {
                while content.contains(config.imageToken), imageIndex < imageReplacements.count {
                    content = content.replacingOccurrences(
                        of: config.imageToken,
                        with: imageReplacements[imageIndex],
                        options: [],
                        range: content.range(of: config.imageToken),
                    )
                    imageIndex += 1
                }
            } else {
                var prefix = ""
                let messageImageCount =
                    message.images.isEmpty
                    ? ((message.role == .user && imageIndex == 0) ? imageReplacements.count : 0)
                    : message.images.count
                for _ in 0 ..< messageImageCount where imageIndex < imageReplacements.count {
                    prefix += imageReplacements[imageIndex]
                    imageIndex += 1
                }
                content = prefix + content
            }
            result += content
            result += "<|im_end|>\n"
        }

        while imageIndex < imageReplacements.count {
            result = result.replacingOccurrences(
                of: "<|im_start|>user\n",
                with: "<|im_start|>user\n" + imageReplacements[imageIndex],
                options: [],
                range: result.range(of: "<|im_start|>user\n"),
            )
            imageIndex += 1
        }

        result += "<|im_start|>assistant\n<think>\n"
        return result
    }

    private func roleName(_ role: Chat.Message.Role) -> String {
        switch role {
        case .system:
            "system"
        case .assistant:
            "assistant"
        case .user:
            "user"
        default:
            role.rawValue
        }
    }

    private func replacement(numPatches: Int, newlines: [Bool]?) -> String {
        var result = ""
        let patchBody = String(repeating: config.imageToken, count: config.patchFeatureSize)
        for index in 0 ..< numPatches {
            result += config.patchStartToken + patchBody + config.patchEndToken
            if newlines?[safe: index] == true {
                result += config.patchNewlineToken
            }
        }
        result += config.imageStartToken
        result += String(repeating: config.imageToken, count: config.imageFeatureSize)
        result += config.imageEndToken
        return result
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?) throws
        -> ProcessedImage
    {
        var image = MediaProcessing.zeroOrigin(
            MediaProcessing.apply(image, processing: processing).toSRGB())
        var width = max(Int(image.extent.width.rounded()), 1)
        var height = max(Int(image.extent.height.rounded()), 1)
        if width != height {
            let size = max(width, height)
            let background = CIImage(color: .black).cropped(
                to: CGRect(x: 0, y: 0, width: size, height: size),
            )
            image = MediaProcessing.zeroOrigin(image).composited(over: background)
            width = size
            height = size
        }
        if max(width, height) > config.maxImageSize {
            let scale = CGFloat(config.maxImageSize) / CGFloat(max(width, height))
            width = max(Int(CGFloat(width) * scale), 1)
            height = max(Int(CGFloat(height) * scale), 1)
            image = image.resampled(to: CGSize(width: width, height: height), method: .bicubic)
        }

        let basePixels = tensor(image: image, size: config.imageSize)
        let window = determineWindowSize(long: max(width, height), short: min(width, height))
        guard window > 0 else {
            return ProcessedImage(
                basePixels: basePixels,
                patchPixels: [],
                numPatches: 0,
                newlines: nil,
            )
        }

        let cropSize = cropSize(width: width, height: height, window: window)
        let cropImage = image.resampled(
            to: CGSize(width: cropSize.width, height: cropSize.height),
            method: .bicubic,
        )
        let windows = cropWindows(width: cropSize.width, height: cropSize.height, window: window)
        let patchPixels = windows.rects.map { rect in
            tensor(
                image: cropImage.cropped(to: rect).transformed(
                    by: CGAffineTransform(translationX: -rect.minX, y: -rect.minY),
                ),
                size: config.patchImageSize,
            )
        }
        var newlines = windows.rects.indices.map { ($0 + 1) % windows.columns == 0 }
        if newlines.last == true {
            newlines[newlines.count - 1] = false
        }
        return ProcessedImage(
            basePixels: basePixels,
            patchPixels: patchPixels,
            numPatches: patchPixels.count,
            newlines: newlines,
        )
    }

    private func determineWindowSize(long: Int, short: Int) -> Int {
        if long <= config.imageSize {
            return Double(long) / Double(max(short, 1)) > 1.5 ? short : 0
        }
        return Double(long) / Double(max(short, 1)) > 4
            ? min(short, config.patchImageSize)
            : config.patchImageSize
    }

    private func cropSize(width: Int, height: Int, window: Int) -> (width: Int, height: Int) {
        func adjusted(_ value: Int) -> Int {
            let ratio = Double(value) / Double(window)
            if ratio < 1 {
                return value
            }
            let whole = Int(ratio)
            return window * (whole + (ratio - Double(whole) > 0.2 ? 1 : 0))
        }
        return (adjusted(width), adjusted(height))
    }

    private func cropWindows(width: Int, height: Int, window: Int)
        -> (rects: [CGRect], columns: Int)
    {
        let columns = width <= window ? 1 : Int(ceil((Double(width - window) / Double(window)) + 1))
        let rows = height <= window ? 1 : Int(ceil((Double(height - window) / Double(window)) + 1))
        var xs = (0 ..< columns).map { window * $0 }
        var ys = (0 ..< rows).map { window * $0 }
        if xs.count > 1, xs[xs.count - 1] + window > width {
            xs[xs.count - 1] = width - window
        }
        if ys.count > 1, ys[ys.count - 1] + window > height {
            ys[ys.count - 1] = height - window
        }
        let rects = ys.flatMap { y in
            xs.map { x in CGRect(x: x, y: y, width: window, height: window) }
        }
        return (rects, columns)
    }

    private func tensor(image: CIImage, size: Int) -> MLXArray {
        let resized = MediaProcessing.zeroOrigin(image).resampled(
            to: CGSize(width: size, height: size),
            method: .bicubic,
        )
        let componentsPerPixel = 4
        let bytesPerPixel = componentsPerPixel * MemoryLayout<Float32>.size
        let bytesPerRow = size * bytesPerPixel
        var data = Data(count: size * size * bytesPerPixel)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)

        data.withUnsafeMutableBytes { pointer in
            CIContext().render(
                resized,
                toBitmap: pointer.baseAddress!,
                rowBytes: bytesPerRow,
                bounds: CGRect(x: 0, y: 0, width: size, height: size),
                format: .RGBAf,
                colorSpace: colorSpace,
            )
        }

        let rgba = MLXArray(data, [size, size, 4], type: Float32.self)
        let rgb = rgba[0..., 0..., ..<3]
        let alpha = maximum(rgba[0..., 0..., 3...], MLXArray(Float32(1e-6)))
        let unpremultiplied = clip(rgb / alpha, min: Float32(0), max: Float32(1))
        let mean = MLXArray(config.imageMean.map { Float32($0) })
        let std = MLXArray(config.imageStd.map { Float32($0) })
        return ((unpremultiplied - mean) / std)
            .transposed(2, 0, 1)
            .expandedDimensions(axis: 0)
    }

}

// MARK: - Language

private enum Step3p7Language {
    static func swiglu(_ up: MLXArray, gate: MLXArray, limit: Float?) -> MLXArray {
        guard let limit, limit > 0 else {
            return silu(gate) * up
        }
        let clampedGate = clip(silu(gate), max: limit)
        let clampedUp = clip(up, min: -limit, max: limit)
        return clampedGate * clampedUp
    }

    final class MLP: Module, UnaryLayer {
        let limit: Float?
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        init(_ config: Step3p7Configuration.TextConfiguration, intermediateSize: Int, limit: Float?)
        {
            self.limit = limit
            _gateProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
            _upProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
            _downProj.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(Step3p7Language.swiglu(upProj(x), gate: gateProj(x), limit: limit))
        }
    }

    final class MoEGate: Module {
        let topK: Int
        let routedScalingFactor: Float
        let normTopKProbability: Bool
        @ModuleInfo var gate: Linear
        @ModuleInfo(key: "router_bias") var routerBias: MLXArray

        init(_ config: Step3p7Configuration.TextConfiguration) {
            topK = config.moeTopK
            routedScalingFactor = config.moeRouterScalingFactor
            normTopKProbability = config.normExpertWeight
            _gate.wrappedValue = Linear(config.hiddenSize, config.moeExperts, bias: false)
            _routerBias.wrappedValue = MLXArray.zeros([config.moeExperts], dtype: .float32)
        }

        func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
            let scores = sigmoid(gate(x).asType(.float32))
            let corrected = scores + routerBias
            let indices = argPartition(-corrected, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
            var weights = takeAlong(scores, indices, axis: -1)
            if topK > 1, normTopKProbability {
                weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
            }
            return (indices, weights * routedScalingFactor)
        }
    }

    final class StepSwitchGLU: Module {
        let limit: Float?
        @ModuleInfo(key: "gate_proj") var gateProj: SwitchLinear
        @ModuleInfo(key: "up_proj") var upProj: SwitchLinear
        @ModuleInfo(key: "down_proj") var downProj: SwitchLinear

        init(_ config: Step3p7Configuration.TextConfiguration, limit: Float?) {
            self.limit = limit
            _gateProj.wrappedValue = SwitchLinear(
                inputDims: config.hiddenSize,
                outputDims: config.moeIntermediateSize,
                numExperts: config.moeExperts,
                bias: false,
            )
            _upProj.wrappedValue = SwitchLinear(
                inputDims: config.hiddenSize,
                outputDims: config.moeIntermediateSize,
                numExperts: config.moeExperts,
                bias: false,
            )
            _downProj.wrappedValue = SwitchLinear(
                inputDims: config.moeIntermediateSize,
                outputDims: config.hiddenSize,
                numExperts: config.moeExperts,
                bias: false,
            )
        }

        func callAsFunction(_ x: MLXArray, _ indices: MLXArray) -> MLXArray {
            let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
            let up = upProj(expanded, indices)
            let gate = gateProj(expanded, indices)
            let activated = Step3p7Language.swiglu(up, gate: gate, limit: limit)
            return squeezed(downProj(activated, indices), axis: -2)
        }
    }

    final class MoE: Module, UnaryLayer {
        @ModuleInfo var gate: MoEGate
        @ModuleInfo(key: "switch_mlp") var switchMLP: StepSwitchGLU
        @ModuleInfo(key: "share_expert") var sharedExpert: MLP

        init(_ config: Step3p7Configuration.TextConfiguration, layerIndex: Int) {
            let limit = config.swigluLimits[safe: layerIndex] ?? nil
            let sharedLimit = config.swigluSharedLimits[safe: layerIndex] ?? nil
            _gate.wrappedValue = MoEGate(config)
            _switchMLP.wrappedValue = StepSwitchGLU(config, limit: limit)
            _sharedExpert.wrappedValue = MLP(
                config,
                intermediateSize: config.sharedExpertDim,
                limit: sharedLimit,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (indices, weights) = gate(x)
            let routed = switchMLP(x, indices)
            return (routed * weights[.ellipsis, .newAxis]).sum(axis: -2).asType(routed.dtype)
                + sharedExpert(x)
        }
    }

    final class Attention: Module {
        let isSliding: Bool
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let useHeadWiseGate: Bool
        let rope: any RoPELayer

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
        @ModuleInfo(key: "g_proj") var gProj: Linear?

        init(_ config: Step3p7Configuration.TextConfiguration, layerIndex: Int) {
            isSliding = config.isSliding(layer: layerIndex)
            let override = isSliding ? config.attentionOtherSetting : nil
            heads = override?.attentionHeads ?? config.attentionHeads
            kvHeads = override?.attentionGroups ?? config.attentionGroups
            headDim = override?.headDim ?? config.headDim
            scale = pow(Float(headDim), -0.5)
            useHeadWiseGate = config.useHeadWiseAttentionGate
            _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
            _kProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
            _vProj.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
            _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            if useHeadWiseGate {
                _gProj.wrappedValue = Linear(config.hiddenSize, heads, bias: false)
            }
            let partial = config.partialRotaryFactors[safe: layerIndex] ?? 1
            rope = initializeRope(
                dims: Int(Float(headDim) * partial),
                base: config.ropeTheta[safe: layerIndex] ?? 10000,
                traditional: false,
                scalingConfig: config.ropeScaling(layer: layerIndex),
                maxPositionEmbeddings: config.maxPositionEmbeddings,
            )
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let batch = x.dim(0)
            let length = x.dim(1)
            var q = qNorm(qProj(x).reshaped(batch, length, heads, headDim))
                .transposed(0, 2, 1, 3)
            var k = kNorm(kProj(x).reshaped(batch, length, kvHeads, headDim))
                .transposed(0, 2, 1, 3)
            let v = vProj(x).reshaped(batch, length, kvHeads, headDim)
                .transposed(0, 2, 1, 3)
            q = applyRotaryPosition(rope, to: q, offset: cache?.ropeOffset)
            k = applyRotaryPosition(rope, to: k, offset: cache?.ropeOffset)
            var y = attentionWithCacheUpdate(
                queries: q,
                keys: k,
                values: v,
                cache: cache,
                scale: scale,
                mask: mask,
            )
            .transposed(0, 2, 1, 3)
            if let gProj {
                y = y * sigmoid(gProj(x))[.ellipsis, .newAxis]
            }
            return oProj(y.reshaped(batch, length, heads * headDim))
        }
    }

    final class DecoderLayer: Module {
        let isSliding: Bool
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        var mlp: UnaryLayer
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: Step3p7Configuration.TextConfiguration, layerIndex: Int) {
            let attention = Attention(config, layerIndex: layerIndex)
            isSliding = attention.isSliding
            _selfAttention.wrappedValue = attention
            if config.moeLayers.contains(layerIndex) {
                mlp = MoE(config, layerIndex: layerIndex)
            } else {
                mlp = MLP(
                    config,
                    intermediateSize: config.intermediateSize,
                    limit: config.swigluSharedLimits[safe: layerIndex] ?? nil,
                )
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
            let h = x + selfAttention(inputLayerNorm(x), mask: mask, cache: cache)
            return h + mlp(postAttentionLayerNorm(h))
        }
    }

    final class TextModel: Module {
        let config: Step3p7Configuration.TextConfiguration
        let firstSlidingLayer: Int?
        let firstFullLayer: Int?

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm

        init(_ config: Step3p7Configuration.TextConfiguration) {
            self.config = config
            let layers = (0 ..< config.hiddenLayers).map { DecoderLayer(config, layerIndex: $0) }
            firstSlidingLayer = layers.firstIndex { $0.isSliding }
            firstFullLayer = layers.firstIndex { !$0.isSliding }
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            _layers.wrappedValue = layers
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ inputIDs: MLXArray?,
            cache: [KVCache]?,
            inputEmbedding: MLXArray? = nil,
        )
            -> MLXArray
        {
            var h: MLXArray
            if let inputEmbedding {
                h = inputEmbedding
            } else if let inputIDs {
                h = embedTokens(inputIDs)
            } else {
                preconditionFailure("inputIds or inputEmbedding must be provided")
            }
            let fullMask =
                firstFullLayer.map {
                    createAttentionMask(h: h, cache: cache?[$0])
                } ?? .none
            let slidingMask =
                firstSlidingLayer.map {
                    createAttentionMask(
                        h: h,
                        cache: cache?[$0],
                        windowSize: config.slidingWindow,
                    )
                } ?? .none
            for (index, layer) in layers.enumerated() {
                h = layer(
                    h,
                    mask: layer.isSliding ? slidingMask : fullMask,
                    cache: cache?[index],
                )
            }
            return norm(h)
        }
    }

    final class LanguageModel: Module {
        let config: Step3p7Configuration.TextConfiguration
        @ModuleInfo var model: TextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear

        init(_ config: Step3p7Configuration.TextConfiguration) {
            self.config = config
            _model.wrappedValue = TextModel(config)
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }

        var kvHeads: [Int] {
            model.layers.map(\.selfAttention.kvHeads)
        }

        func callAsFunction(
            _ inputIDs: MLXArray?,
            cache: [KVCache]?,
            inputEmbedding: MLXArray? = nil,
        )
            -> LMOutput
        {
            LMOutput(logits: lmHead(model(inputIDs, cache: cache, inputEmbedding: inputEmbedding)))
        }
    }
}

// MARK: - Vision

private enum Step3p7Vision {
    static func quickGELU(_ x: MLXArray) -> MLXArray {
        x * sigmoid(1.702 * x)
    }

    final class EncoderRoPE2D {
        let dim: Int
        let theta: Float

        init(dim: Int, theta: Float) {
            self.dim = dim
            self.theta = theta
        }

        func frequencies(height: Int, width: Int) -> MLXArray {
            let invFreq =
                1.0
                / MLX.pow(
                    MLXArray(theta),
                    MLXArray(stride(from: 0, to: dim / 2, by: 2)).asType(.float32)
                        / Float(dim / 2),
                )
            let rows = MLXArray(0 ..< height).asType(.float32)
            let cols = MLXArray(0 ..< width).asType(.float32)
            let freqsH = (rows[0..., .newAxis] * invFreq[.newAxis, 0...])
                .expandedDimensions(axis: 1)
            let freqsW = (cols[0..., .newAxis] * invFreq[.newAxis, 0...])
                .expandedDimensions(axis: 0)
            let hBroadcast = broadcast(freqsH, to: [height, width, invFreq.dim(0)])
            let wBroadcast = broadcast(freqsW, to: [height, width, invFreq.dim(0)])
            return concatenated([wBroadcast, hBroadcast], axis: -1).reshaped(height * width, -1)
        }

        func callAsFunction(_ q: MLXArray, _ k: MLXArray, grid: (Int, Int))
            -> (MLXArray, MLXArray)
        {
            let freqs = frequencies(height: grid.0, width: grid.1)
                .expandedDimensions(axis: 0)
                .expandedDimensions(axis: 0)
            let cosValues = repeated(cos(freqs), count: 2, axis: -1)
            let sinValues = repeated(sin(freqs), count: 2, axis: -1)
            return (
                q * cosValues + RotaryEmbeddingOps.rotateInterleaved(q) * sinValues,
                k * cosValues + RotaryEmbeddingOps.rotateInterleaved(k) * sinValues,
            )
        }
    }

    final class MLP: Module, UnaryLayer {
        let hiddenActivation: String
        @ModuleInfo(key: "c_fc") var cFC: Linear
        @ModuleInfo(key: "c_proj") var cProj: Linear

        init(_ config: Step3p7Configuration.VisionConfiguration) {
            hiddenActivation = config.hiddenActivation
            let hidden = Int(Float(config.width) * config.mlpRatio)
            _cFC.wrappedValue = Linear(config.width, hidden, bias: true)
            _cProj.wrappedValue = Linear(hidden, config.width, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let hidden = cFC(x)
            return cProj(
                hiddenActivation == "quick_gelu" ? Step3p7Vision.quickGELU(hidden) : gelu(hidden))
        }
    }

    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float
        let rope: EncoderRoPE2D?
        @ModuleInfo(key: "in_proj") var inProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: Step3p7Configuration.VisionConfiguration) {
            heads = config.heads
            headDim = config.width / config.heads
            scale = pow(Float(headDim), -0.5)
            rope =
                config.useRoPE2D
                ? EncoderRoPE2D(dim: headDim, theta: config.ropeTheta)
                : nil
            _inProj.wrappedValue = Linear(config.width, config.width * 3, bias: true)
            _outProj.wrappedValue = Linear(config.width, config.width, bias: true)
        }

        func callAsFunction(_ x: MLXArray, grid: (Int, Int)) -> MLXArray {
            let batch = x.dim(0)
            let length = x.dim(1)
            let qkv = inProj(x).reshaped(batch, length, 3, heads, headDim)
            let parts = split(qkv, parts: 3, axis: 2)
            var q = parts[0].squeezed(axis: 2).transposed(0, 2, 1, 3)
            var k = parts[1].squeezed(axis: 2).transposed(0, 2, 1, 3)
            let v = parts[2].squeezed(axis: 2).transposed(0, 2, 1, 3)
            if let rope {
                (q, k) = rope(q, k, grid: grid)
            }
            let y = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: .none,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)
            return outProj(y)
        }
    }

    final class LayerScale: Module, UnaryLayer {
        @ModuleInfo var gamma: MLXArray

        init(dim: Int, value: Float?) {
            _gamma.wrappedValue = MLXArray.ones([dim]) * (value ?? 1)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            x * gamma
        }
    }

    final class Block: Module {
        @ModuleInfo(key: "ln_1") var ln1: LayerNorm
        @ModuleInfo(key: "ln_2") var ln2: LayerNorm
        @ModuleInfo var attn: Attention
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "ls_1") var ls1: LayerScale
        @ModuleInfo(key: "ls_2") var ls2: LayerScale

        init(_ config: Step3p7Configuration.VisionConfiguration) {
            _ln1.wrappedValue = LayerNorm(dimensions: config.width, eps: config.layerNormEps)
            _ln2.wrappedValue = LayerNorm(dimensions: config.width, eps: config.layerNormEps)
            _attn.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(config)
            _ls1.wrappedValue = LayerScale(dim: config.width, value: config.layerScaleInitValue)
            _ls2.wrappedValue = LayerScale(dim: config.width, value: config.layerScaleInitValue)
        }

        func callAsFunction(_ x: MLXArray, grid: (Int, Int)) -> MLXArray {
            let h = x + ls1(attn(ln1(x), grid: grid))
            return h + ls2(mlp(ln2(h)))
        }
    }

    final class VisionModel: Module {
        let config: Step3p7Configuration.VisionConfiguration
        let gridSize: Int
        @ModuleInfo var conv1: Conv2d
        @ModuleInfo(key: "ln_pre") var lnPre: LayerNorm?
        @ModuleInfo(key: "ln_post") var lnPost: LayerNorm?
        @ModuleInfo(key: "class_embedding") var classEmbedding: MLXArray?
        @ModuleInfo(key: "positional_embedding") var positionalEmbedding: MLXArray?
        @ModuleInfo var transformer: [Block]
        @ModuleInfo(key: "vit_downsampler1") var vitDownsampler1: Conv2d
        @ModuleInfo(key: "vit_downsampler2") var vitDownsampler2: Conv2d

        init(_ config: Step3p7Configuration.VisionConfiguration) {
            self.config = config
            gridSize = config.imageSize / config.patchSize
            _conv1.wrappedValue = Conv2d(
                inputChannels: config.channels,
                outputChannels: config.width,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: false,
            )
            if config.useLNPre {
                _lnPre.wrappedValue = LayerNorm(
                    dimensions: config.width,
                    eps: config.layerNormEps,
                )
            }
            if config.useLNPost {
                _lnPost.wrappedValue = LayerNorm(
                    dimensions: config.width,
                    eps: config.layerNormEps,
                )
            }
            if config.useCLSToken {
                _classEmbedding.wrappedValue =
                    MLXRandom.normal([config.width]) * pow(Float(config.width), -0.5)
            }
            if config.useAbsolutePositionEmbedding {
                _positionalEmbedding.wrappedValue =
                    MLXRandom.normal([
                        (config.useCLSToken ? 1 : 0) + gridSize * gridSize,
                        config.width,
                    ]) * pow(Float(config.width), -0.5)
            }
            _transformer.wrappedValue = (0 ..< config.layers).map { _ in Block(config) }
            _vitDownsampler1.wrappedValue = Conv2d(
                inputChannels: config.width,
                outputChannels: config.width * 2,
                kernelSize: IntOrPair(3),
                stride: IntOrPair(2),
                padding: IntOrPair(1),
            )
            _vitDownsampler2.wrappedValue = Conv2d(
                inputChannels: config.width * 2,
                outputChannels: config.width * 4,
                kernelSize: IntOrPair(3),
                stride: IntOrPair(2),
                padding: IntOrPair(1),
            )
        }

        func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
            var pixelValues = pixelValues
            if pixelValues.ndim == 3 {
                pixelValues = pixelValues.expandedDimensions(axis: 0)
            }
            if pixelValues.dim(1) == config.channels {
                pixelValues = pixelValues.transposed(0, 2, 3, 1)
            }
            let batch = pixelValues.dim(0)
            let height = pixelValues.dim(1)
            let width = pixelValues.dim(2)
            let gridH = height / config.patchSize
            let gridW = width / config.patchSize
            var x = conv1(pixelValues).reshaped(batch, gridH * gridW, config.width)
            if let classEmbedding {
                let cls = broadcast(
                    classEmbedding.reshaped(1, 1, -1),
                    to: [batch, 1, config.width],
                )
                x = concatenated([cls, x], axis: 1)
            }
            if let positionalEmbedding {
                x =
                    x
                    + positionEmbedding(
                        positionalEmbedding,
                        gridH: gridH,
                        gridW: gridW,
                    ).asType(x.dtype)
            }
            if let lnPre {
                x = lnPre(x)
            }
            for block in transformer {
                x = block(x, grid: (gridH, gridW))
            }
            if let lnPost {
                x = lnPost(x)
            }
            if classEmbedding != nil {
                x = x[0..., 1..., 0...]
            }
            return x
        }

        private func positionEmbedding(_ embedding: MLXArray, gridH: Int, gridW: Int) -> MLXArray {
            if gridSize == gridH, gridSize == gridW {
                return embedding.expandedDimensions(axis: 0)
            }
            var pos = embedding
            var cls: MLXArray?
            if config.useCLSToken {
                cls = pos[..<1, 0...]
                pos = pos[1..., 0...]
            }
            let source = pos.reshaped(1, gridSize, gridSize, config.width)
                .transposed(0, 3, 1, 2)
            let interpolated = bicubicInterpolate(source, size: (gridH, gridW))
                .transposed(0, 2, 3, 1)
                .reshaped(gridH * gridW, config.width)
            let result = cls.map { concatenated([$0, interpolated], axis: 0) } ?? interpolated
            return result.expandedDimensions(axis: 0)
        }

        func downsample(_ imageFeatures: MLXArray) -> MLXArray {
            let batch = imageFeatures.dim(0)
            let patches = imageFeatures.dim(1)
            let channels = imageFeatures.dim(2)
            let side = Int(Double(patches).squareRoot())
            var x = imageFeatures.reshaped(batch, side, side, channels)
            x = vitDownsampler1(x)
            x = vitDownsampler2(x)
            return x.reshaped(batch, x.dim(1) * x.dim(2), x.dim(3))
        }
    }
}

// MARK: - Model

public final class Step3p7: Module, VLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    let config: Step3p7Configuration
    @ModuleInfo(key: "vision_model") private var visionModel: Step3p7Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Step3p7Language.LanguageModel
    @ModuleInfo(key: "vit_large_projector") private var projector: Linear

    public init(_ config: Step3p7Configuration) {
        self.config = config
        vocabularySize = config.vocabularySize
        _visionModel.wrappedValue = Step3p7Vision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue =
            Step3p7Language.LanguageModel(config.textConfiguration)
        _projector.wrappedValue = Linear(
            config.visionConfiguration.width * 4,
            config.textConfiguration.hiddenSize,
            bias: config.projectorBias,
        )
    }

    private func processImageFeatures(_ pixels: MLXArray) -> MLXArray {
        projector(visionModel.downsample(visionModel(pixels)))
    }

    private func visionEmbeddings(image: LMInput.ProcessedImage?, patches: LMInput.ProcessedVideo?)
        -> MLXArray?
    {
        guard let image else {
            return nil
        }
        let baseFeatures = processImageFeatures(image.pixels)
        let patchFeatures = patches.map { processImageFeatures($0.pixels) }
        let patchCounts =
            image.frames?.map(\.t) ?? Array(repeating: 0, count: baseFeatures.dim(0))
        var merged: [MLXArray] = []
        var cursor = 0
        for index in 0 ..< baseFeatures.dim(0) {
            var parts: [MLXArray] = []
            let count = patchCounts[safe: index] ?? 0
            if count > 0, let patchFeatures {
                parts.append(
                    patchFeatures[cursor ..< cursor + count, 0..., 0...]
                        .reshaped(-1, patchFeatures.dim(-1)),
                )
                cursor += count
            }
            parts.append(baseFeatures[index, 0..., 0...])
            merged.append(parts.count == 1 ? parts[0] : concatenated(parts, axis: 0))
        }
        return concatenated(merged, axis: 0)
    }

    private func inputEmbeddings(_ input: LMInput) -> MLXArray {
        let embeddings = languageModel.model.embedTokens(input.text.tokens)
        guard let imageFeatures = visionEmbeddings(image: input.image, patches: input.video) else {
            return embeddings
        }
        let ids = input.text.tokens.flattened().asArray(Int.self)
        let positions = ids.indices.filter { ids[$0] == config.imageTokenIndex }
        precondition(
            positions.count == imageFeatures.dim(0),
            "Step3p7 image feature/token count mismatch: tokens \(positions.count), features \(imageFeatures.dim(0)).",
        )
        embeddings[0..., MLXArray(positions), 0...] =
            imageFeatures.asType(embeddings.dtype).expandedDimensions(axis: 0)
        return embeddings
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
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
            languageModel(nil, cache: cache, inputEmbedding: embeddings[0..., processed..., 0...]))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try (0 ..< config.textConfiguration.hiddenLayers).map { layer in
            if config.textConfiguration.isSliding(layer: layer),
                let window = config.textConfiguration.slidingWindow
            {
                return RotatingKVCache(maxSize: window, keep: 4)
            }
            return try makeAttentionKVCache(configuration: configuration)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Step3p7Sanitizer.sanitize(weights: weights, config: config)
    }
}

private enum Step3p7Sanitizer {
    static func sanitize(
        weights: [String: MLXArray],
        config: Step3p7Configuration,
    )
        -> [String: MLXArray]
    {
        var sanitized = [String: MLXArray]()
        let vanillaNorms = weights.contains { key, _ in
            languageRemappings.contains { src, dst in key.contains(src) && !key.contains(dst) }
        }
        for (key, value) in weights {
            if key.contains(".mtp") {
                continue
            }
            var key = key
            var value = value
            if key.hasPrefix("model.language_model.") {
                key = key.replacingOccurrences(
                    of: "model.language_model.",
                    with: "language_model.model.",
                    options: [],
                    range: key.range(of: "model.language_model."),
                )
            } else if key.hasPrefix("model.layers.") {
                if shouldDropLayer(key, hiddenLayers: config.textConfiguration.hiddenLayers) {
                    continue
                }
                key = key.replacingOccurrences(
                    of: "model.",
                    with: "language_model.model.",
                    options: [],
                    range: key.range(of: "model."),
                )
            } else if key.hasPrefix("model.embed_tokens.") || key.hasPrefix("model.norm.") {
                key = key.replacingOccurrences(
                    of: "model.",
                    with: "language_model.model.",
                    options: [],
                    range: key.range(of: "model."),
                )
            } else if key.hasPrefix("model.vision_model.") {
                key = key.replacingOccurrences(
                    of: "model.vision_model.",
                    with: "vision_model.",
                    options: [],
                    range: key.range(of: "model.vision_model."),
                )
            } else if key.hasPrefix("model.vit_large_projector.") {
                key = key.replacingOccurrences(
                    of: "model.vit_large_projector.",
                    with: "vit_large_projector.",
                    options: [],
                    range: key.range(of: "model.vit_large_projector."),
                )
            } else if key.hasPrefix("lm_head.") {
                key = key.replacingOccurrences(
                    of: "lm_head.",
                    with: "language_model.lm_head.",
                    options: [],
                    range: key.range(of: "lm_head."),
                )
            } else if isBareVisionKey(key) {
                key = "vision_model.\(key)"
            }

            if key.hasPrefix("vision_model.") {
                let subKey = key.replacingOccurrences(
                    of: "vision_model.",
                    with: "",
                    options: [],
                    range: key.range(of: "vision_model."),
                )
                let (newSubKey, newValue) = sanitizeVision(
                    key: subKey,
                    value: value,
                    config: config.visionConfiguration,
                )
                sanitized["vision_model.\(newSubKey)"] = newValue
                continue
            }

            if shouldDropLayer(key, hiddenLayers: config.textConfiguration.hiddenLayers) {
                continue
            }
            key = sanitizeLanguageKey(key)
            if vanillaNorms, key.hasSuffix(".weight"), key.contains("norm") {
                value = value + 1
            }
            sanitized[key] = value
        }
        return sanitized
    }

    private static let languageRemappings: [(String, String)] = [
        (".moe.gate_proj.", ".mlp.switch_mlp.gate_proj."),
        (".moe.up_proj.", ".mlp.switch_mlp.up_proj."),
        (".moe.down_proj.", ".mlp.switch_mlp.down_proj."),
        (".moe.gate.", ".mlp.gate.gate."),
        (".moe.router_bias", ".mlp.gate.router_bias"),
        (".share_expert.", ".mlp.share_expert."),
    ]

    private static func sanitizeLanguageKey(_ key: String) -> String {
        for (source, destination) in languageRemappings
        where key.contains(source) && !key.contains(destination) {
            return key.replacingOccurrences(of: source, with: destination)
        }
        return key
    }

    private static func sanitizeVision(
        key: String,
        value: MLXArray,
        config: Step3p7Configuration.VisionConfiguration,
    )
        -> (String, MLXArray)
    {
        var key = key
        var value = value
        if key.hasPrefix("transformer.resblocks.") {
            key = key.replacingOccurrences(
                of: "transformer.resblocks.",
                with: "transformer.",
                options: [],
                range: key.range(of: "transformer.resblocks."),
            )
        }
        if key.hasSuffix("attn.in_proj_weight") {
            key = key.replacingOccurrences(of: "attn.in_proj_weight", with: "attn.in_proj.weight")
        } else if key.hasSuffix("attn.in_proj_bias") {
            key = key.replacingOccurrences(of: "attn.in_proj_bias", with: "attn.in_proj.bias")
        }
        if key == "conv1.weight",
            value.ndim == 4,
            value.shape.last != config.channels
        {
            value = value.transposed(0, 2, 3, 1)
        } else if key.contains("vit_downsampler"),
            key.hasSuffix(".weight"),
            value.ndim == 4
        {
            let expectedInput = config.width * (key.contains("vit_downsampler2") ? 2 : 1)
            if value.shape.last != expectedInput {
                value = value.transposed(0, 2, 3, 1)
            }
        }
        return (key, value)
    }

    private static func shouldDropLayer(_ key: String, hiddenLayers: Int) -> Bool {
        let parts = key.split(separator: ".").map(String.init)
        guard let layerIndex = parts.firstIndex(of: "layers"),
            parts.indices.contains(layerIndex + 1),
            let layer = Int(parts[layerIndex + 1])
        else {
            return false
        }
        return layer >= hiddenLayers
    }

    private static func isBareVisionKey(_ key: String) -> Bool {
        key.hasPrefix("conv1.")
            || key.hasPrefix("ln_pre.")
            || key.hasPrefix("ln_post.")
            || key.hasPrefix("transformer.")
            || key.hasPrefix("vit_downsampler")
            || key == "positional_embedding"
            || key == "class_embedding"
    }
}

extension Array {
    fileprivate subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
