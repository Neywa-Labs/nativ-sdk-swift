import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct KimiVLConfiguration: Decodable, Sendable {
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
        public let kvLoraRank: Int
        public let qLoraRank: Int?
        public let qkRopeHeadDim: Int
        public let vHeadDim: Int
        public let qkNopeHeadDim: Int
        public let normTopkProb: Bool
        public let nGroup: Int
        public let topKGroup: Int
        public let numExpertsPerToken: Int
        public let moeLayerFrequency: Int
        public let firstKDenseReplace: Int
        public let maxPositionEmbeddings: Int
        public let rmsNormEps: Float
        public let ropeTheta: Float
        public let ropeScaling: [String: StringOrNumber]?
        public let attentionBias: Bool
        public let topKMethod: String
        public let scoringFunction: String

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
            case kvLoraRank = "kv_lora_rank"
            case qLoraRank = "q_lora_rank"
            case qkRopeHeadDim = "qk_rope_head_dim"
            case vHeadDim = "v_head_dim"
            case qkNopeHeadDim = "qk_nope_head_dim"
            case normTopkProb = "norm_topk_prob"
            case nGroup = "n_group"
            case topKGroup = "topk_group"
            case numExpertsPerToken = "num_experts_per_tok"
            case moeLayerFrequency = "moe_layer_freq"
            case firstKDenseReplace = "first_k_dense_replace"
            case maxPositionEmbeddings = "max_position_embeddings"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case ropeScaling = "rope_scaling"
            case attentionBias = "attention_bias"
            case topKMethod = "topk_method"
            case scoringFunction = "scoring_func"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "deepseek_v3"
            vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 163_840
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 11264
            moeIntermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 1408
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 27
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            _kvHeads = try c.decodeIfPresent(Int.self, forKey: ._kvHeads)
            nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts) ?? 2
            nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts) ?? 64
            routedScalingFactor =
                try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 2.446
            kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? 512
            qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
            qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
            vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 128
            qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 128
            normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
            nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
            topKGroup = try c.decodeIfPresent(Int.self, forKey: .topKGroup) ?? nGroup
            numExpertsPerToken =
                try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 6
            moeLayerFrequency =
                try c.decodeIfPresent(Int.self, forKey: .moeLayerFrequency) ?? 1
            firstKDenseReplace =
                try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 1
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 800_000
            ropeScaling = try c.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling,
            )
            attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            topKMethod = try c.decodeIfPresent(String.self, forKey: .topKMethod) ?? "noaux_tc"
            scoringFunction =
                try c.decodeIfPresent(String.self, forKey: .scoringFunction) ?? "sigmoid"
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let embedDim: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let imageSize: Int
        public let patchSize: Int
        public let channels: Int
        public let layerNormEps: Float
        public let initPosEmbeddingHeight: Int
        public let initPosEmbeddingWidth: Int
        public let spatialMergeSize: Int
        public let mergeKernelSize: [Int]

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case embedDim = "embed_dim"
            case intermediateSize = "intermediate_size"
            case depth
            case hiddenLayers = "num_hidden_layers"
            case numHeads = "num_heads"
            case attentionHeads = "num_attention_heads"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case channels = "num_channels"
            case layerNormEps = "layer_norm_eps"
            case initPosEmbeddingHeight = "init_pos_emb_height"
            case initPosEmbeddingWidth = "init_pos_emb_width"
            case spatialMergeSize = "spatial_merge_size"
            case mergeKernelSize = "merge_kernel_size"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "moonvit"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            embedDim = try c.decodeIfPresent(Int.self, forKey: .embedDim) ?? hiddenSize
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
            hiddenLayers =
                try c.decodeIfPresent(Int.self, forKey: .hiddenLayers)
                ?? c.decodeIfPresent(Int.self, forKey: .depth)
                ?? 27
            attentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .attentionHeads)
                ?? c.decodeIfPresent(Int.self, forKey: .numHeads)
                ?? 16
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 384
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
            initPosEmbeddingHeight =
                try c.decodeIfPresent(Int.self, forKey: .initPosEmbeddingHeight) ?? 64
            initPosEmbeddingWidth =
                try c.decodeIfPresent(Int.self, forKey: .initPosEmbeddingWidth)
                ?? initPosEmbeddingHeight
            spatialMergeSize =
                try c.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
            mergeKernelSize =
                try c.decodeIfPresent([Int].self, forKey: .mergeKernelSize)
                ?? [spatialMergeSize, spatialMergeSize]
        }
    }

    public let modelType: String
    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let mediaPlaceholderTokenID: Int
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int {
        _imageTokenIndex ?? mediaPlaceholderTokenID
    }

    public let eosTokenIDs: [Int]
    public let vocabularySize: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case mediaPlaceholderTokenID = "media_placeholder_token_id"
        case _imageTokenIndex = "image_token_index"
        case eosTokenIDs = "eos_token_id"
        case vocabularySize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "kimi_vl"
        textConfiguration = try c.decode(TextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try c.decode(VisionConfiguration.self, forKey: .visionConfiguration)
        mediaPlaceholderTokenID =
            try c.decodeIfPresent(Int.self, forKey: .mediaPlaceholderTokenID) ?? 163_592
        _imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: ._imageTokenIndex)
        vocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .vocabularySize)
            ?? textConfiguration.vocabularySize
        if let ids = try? c.decode([Int].self, forKey: .eosTokenIDs) {
            eosTokenIDs = ids
        } else if let id = try? c.decode(Int.self, forKey: .eosTokenIDs) {
            eosTokenIDs = [id]
        } else {
            eosTokenIDs = [163_584, 163_586]
        }
    }
}

public struct KimiVLProcessorConfiguration: Decodable, Sendable {
    private struct NestedImageProcessor: Decodable {
        let patchSize: Int?
        let padInput: Bool?
        let imageMean: [CGFloat]?
        let imageStd: [CGFloat]?
        let inTokenLimit: Int?
        let mergeKernelSize: [Int]?

        enum CodingKeys: String, CodingKey {
            case patchSize = "patch_size"
            case padInput = "pad_input"
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case inTokenLimit = "in_token_limit"
            case mergeKernelSize = "merge_kernel_size"
        }
    }

    public let processorClass: String
    public let patchSize: Int
    public let padInput: Bool
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let inTokenLimit: Int
    public let mergeKernelSize: [Int]
    public let imageToken: String

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case patchSize = "patch_size"
        case padInput = "pad_input"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case inTokenLimit = "in_token_limit"
        case mergeKernelSize = "merge_kernel_size"
        case imageProcessor = "image_processor"
        case imageToken = "image_token"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let nested = try c.decodeIfPresent(NestedImageProcessor.self, forKey: .imageProcessor)
        processorClass =
            try c.decodeIfPresent(String.self, forKey: .processorClass)
            ?? "KimiVLProcessor"
        patchSize =
            try c.decodeIfPresent(Int.self, forKey: .patchSize)
            ?? nested?.patchSize
            ?? 14
        padInput =
            try c.decodeIfPresent(Bool.self, forKey: .padInput)
            ?? nested?.padInput
            ?? true
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageMean)
            ?? nested?.imageMean
            ?? [0.5, 0.5, 0.5]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageStd)
            ?? nested?.imageStd
            ?? [0.5, 0.5, 0.5]
        inTokenLimit =
            try c.decodeIfPresent(Int.self, forKey: .inTokenLimit)
            ?? nested?.inTokenLimit
            ?? 4096
        mergeKernelSize =
            try c.decodeIfPresent([Int].self, forKey: .mergeKernelSize)
            ?? nested?.mergeKernelSize
            ?? [2, 2]
        imageToken = try c.decodeIfPresent(String.self, forKey: .imageToken) ?? "<|media_pad|>"
    }
}

// MARK: - Processor

public struct KimiVLProcessor: UserInputProcessor {
    private let config: KimiVLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: KimiVLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        guard input.videos.isEmpty else {
            throw VLMError.singleMediaTypeAllowed
        }

        if images.isEmpty {
            let prompt = renderPrompt(input: input, imageTokenCounts: [])
            let tokens = tokenizer.encode(text: prompt, addSpecialTokens: false)
            let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
            return LMInput(
                text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)))
        }

        let processedImages = try images.map {
            try preprocess(image: $0.asCIImage(), processing: input.processing)
        }
        let counts = processedImages.map(\.placeholderCount)
        let prompt = renderPrompt(input: input, imageTokenCounts: counts)
        let tokens = tokenizer.encode(text: prompt, addSpecialTokens: false)
        let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)),
            image: .init(
                pixels: concatenated(processedImages.map(\.pixels), axis: 0),
                frames: processedImages.map { THW(1, $0.gridHeight, $0.gridWidth) },
            ),
        )
    }

    private struct ProcessedImage {
        let pixels: MLXArray
        let gridHeight: Int
        let gridWidth: Int
        let placeholderCount: Int
    }

    private func renderPrompt(input: UserInput, imageTokenCounts: [Int]) -> String {
        let messages: [Chat.Message] =
            switch input.prompt {
            case .text(let text):
                [.user(text)]
            case .chat(let chat):
                chat
            case .messages:
                [.user(input.prompt.description)]
            }

        var result = ""
        if messages.first?.role != .system {
            result += "<|im_system|>system<|im_middle|>You are a helpful assistant<|im_end|>"
        }

        var imageIndex = 0
        for message in messages {
            switch message.role {
            case .system:
                result += "<|im_system|>"
            case .user:
                result += "<|im_user|>"
            case .assistant:
                result += "<|im_assistant|>"
            default:
                result += "<|im_user|>"
            }
            result += "\(message.role.rawValue)<|im_middle|>"
            for _ in message.images {
                result += imagePlaceholder(count: imageTokenCounts[safe: imageIndex] ?? 1)
                imageIndex += 1
            }
            if message.images.isEmpty, imageIndex < imageTokenCounts.count, message.role == .user {
                result += imagePlaceholder(count: imageTokenCounts[safe: imageIndex] ?? 1)
                imageIndex += 1
            }
            result += message.content
            result += "<|im_end|>"
        }

        while imageIndex < imageTokenCounts.count {
            result = result.replacingOccurrences(
                of: "<|im_user|>user<|im_middle|>",
                with: "<|im_user|>user<|im_middle|>"
                    + imagePlaceholder(count: imageTokenCounts[imageIndex]),
                options: [],
                range: result.range(of: "<|im_user|>user<|im_middle|>"),
            )
            imageIndex += 1
        }

        result += "<|im_assistant|>assistant<|im_middle|>"
        return result
    }

    private func imagePlaceholder(count: Int) -> String {
        "<|media_start|>image<|media_content|>"
            + String(repeating: config.imageToken, count: max(count, 1))
            + "<|media_end|>"
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?) throws
        -> ProcessedImage
    {
        let image = MediaProcessing.zeroOrigin(
            MediaProcessing.apply(image, processing: processing).toSRGB())
        let target = try targetImage(image)
        let normalized = normalizedCHWArray(
            target.image, width: target.width, height: target.height)
        let patches = patchify(normalized, width: target.width, height: target.height)
        let gridHeight = target.height / config.patchSize
        let gridWidth = target.width / config.patchSize
        let mergeLength = max(
            (config.mergeKernelSize.first ?? 2) * (config.mergeKernelSize.dropFirst().first ?? 2),
            1,
        )
        return ProcessedImage(
            pixels: patches,
            gridHeight: gridHeight,
            gridWidth: gridWidth,
            placeholderCount: gridHeight * gridWidth / mergeLength,
        )
    }

    private func targetImage(_ image: CIImage) throws -> (image: CIImage, width: Int, height: Int) {
        var width = max(Int(image.extent.width.rounded()), 1)
        var height = max(Int(image.extent.height.rounded()), 1)
        let patch = config.patchSize
        let currentTokens = max((width / patch) * (height / patch), 1)
        var working = image
        if currentTokens > config.inTokenLimit {
            let scale = sqrt(Float(config.inTokenLimit) / Float(currentTokens))
            width = max(Int(Float(width) * scale), patch)
            height = max(Int(Float(height) * scale), patch)
            working = working.resampled(to: CGSize(width: width, height: height), method: .bicubic)
        }

        let factorH = max((config.mergeKernelSize.first ?? 2) * patch, 1)
        let factorW = max((config.mergeKernelSize.dropFirst().first ?? 2) * patch, 1)
        if config.padInput {
            width = Int(ceil(Double(width) / Double(factorW))) * factorW
            height = Int(ceil(Double(height) / Double(factorH))) * factorH
            let background = CIImage(color: .black).cropped(
                to: CGRect(x: 0, y: 0, width: width, height: height),
            )
            working = MediaProcessing.zeroOrigin(working).composited(over: background)
        } else {
            width = max(width - width % factorW, factorW)
            height = max(height - height % factorH, factorH)
            let source = MediaProcessing.zeroOrigin(working)
            let x = max((Int(source.extent.width.rounded()) - width) / 2, 0)
            let y = max((Int(source.extent.height.rounded()) - height) / 2, 0)
            working =
                source
                .cropped(to: CGRect(x: x, y: y, width: width, height: height))
                .transformed(by: CGAffineTransform(translationX: -CGFloat(x), y: -CGFloat(y)))
        }

        let gridHeight = height / patch
        let gridWidth = width / patch
        guard gridHeight < 512, gridWidth < 512 else {
            throw VLMError.processing("Kimi-VL image grid exceeds positional embedding limit.")
        }
        return (MediaProcessing.zeroOrigin(working), width, height)
    }

    private func normalizedCHWArray(_ image: CIImage, width: Int, height: Int) -> MLXArray {
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
                bounds: CGRect(x: 0, y: 0, width: width, height: height),
                format: .RGBAf,
                colorSpace: colorSpace,
            )
        }

        let rgba = MLXArray(data, [height, width, 4], type: Float32.self)
        let rgb = rgba[0..., 0..., ..<3]
        let alpha = maximum(rgba[0..., 0..., 3...], MLXArray(Float32(1e-6)))
        let unpremultiplied = clip(rgb / alpha, min: Float32(0), max: Float32(1))
        let mean = MLXArray(config.imageMean.map { Float32($0) })
        let std = MLXArray(config.imageStd.map { Float32($0) })
        return ((unpremultiplied - mean) / std).transposed(2, 0, 1)
    }

    private func patchify(_ image: MLXArray, width: Int, height: Int) -> MLXArray {
        let channels = image.dim(0)
        let gridHeight = height / config.patchSize
        let gridWidth = width / config.patchSize
        var patches = image.reshaped(
            channels,
            gridHeight,
            config.patchSize,
            gridWidth,
            config.patchSize,
        )
        patches = patches.transposed(1, 3, 0, 2, 4)
        return patches.reshaped(
            gridHeight * gridWidth, channels, config.patchSize, config.patchSize)
    }

}

// MARK: - Language

private enum KimiVLLanguage {
    static func clippedSilu(_ x: MLXArray) -> MLXArray {
        clip(x * sigmoid(x), min: -100, max: 100)
    }

    final class Attention: Module {
        let config: KimiVLConfiguration.TextConfiguration
        let numHeads: Int
        let qLoraRank: Int?
        let qkRopeHeadDim: Int
        let kvLoraRank: Int
        let vHeadDim: Int
        let qkNopeHeadDim: Int
        let qHeadDim: Int
        var scale: Float
        let rope: RoPELayer

        @ModuleInfo(key: "q_proj") var qProj: Linear?
        @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
        @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
        @ModuleInfo(key: "q_b_proj") var qBProj: Linear?
        @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMqa: Linear
        @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
        @ModuleInfo(key: "kv_b_proj") var kvBProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        init(_ config: KimiVLConfiguration.TextConfiguration) {
            self.config = config
            numHeads = config.attentionHeads
            qLoraRank = config.qLoraRank
            qkRopeHeadDim = config.qkRopeHeadDim
            kvLoraRank = config.kvLoraRank
            vHeadDim = config.vHeadDim
            qkNopeHeadDim = config.qkNopeHeadDim
            qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim
            scale = pow(Float(qHeadDim), -0.5)

            if let qLoraRank = config.qLoraRank {
                _qAProj.wrappedValue = Linear(
                    config.hiddenSize,
                    qLoraRank,
                    bias: config.attentionBias,
                )
                _qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank, eps: 1e-6)
                _qBProj.wrappedValue = Linear(
                    qLoraRank,
                    config.attentionHeads * qHeadDim,
                    bias: false,
                )
            } else {
                _qProj.wrappedValue = Linear(
                    config.hiddenSize,
                    config.attentionHeads * qHeadDim,
                    bias: false,
                )
            }

            _kvAProjWithMqa.wrappedValue = Linear(
                config.hiddenSize,
                config.kvLoraRank + config.qkRopeHeadDim,
                bias: config.attentionBias,
            )
            _kvALayerNorm.wrappedValue = RMSNorm(dimensions: config.kvLoraRank, eps: 1e-6)
            _kvBProj.wrappedValue = Linear(
                config.kvLoraRank,
                config.attentionHeads * (qHeadDim - config.qkRopeHeadDim + config.vHeadDim),
                bias: false,
            )
            _oProj.wrappedValue = Linear(
                config.attentionHeads * config.vHeadDim,
                config.hiddenSize,
                bias: config.attentionBias,
            )

            if let ropeScaling = config.ropeScaling {
                let mScaleAllDim = ropeScaling["mscale_all_dim"]?.asFloat() ?? 0
                if mScaleAllDim != 0 {
                    let scalingFactor = ropeScaling["factor"]?.asFloat() ?? 1
                    if scalingFactor > 1 {
                        let s = 0.1 * mScaleAllDim * log(scalingFactor) + 1
                        scale = scale * s * s
                    }
                }
            }
            rope = initializeRope(
                dims: config.qkRopeHeadDim,
                base: config.ropeTheta,
                traditional: true,
                scalingConfig: config.ropeScaling,
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
            let (batchSize, sequenceLength) = (x.dim(0), x.dim(1))
            var q: MLXArray =
                if qLoraRank == nil {
                    qProj!(x)
                } else {
                    qBProj!(qALayerNorm!(qAProj!(x)))
                }

            q = q.reshaped(batchSize, sequenceLength, numHeads, qHeadDim)
                .transposed(0, 2, 1, 3)
            let qSplit = split(q, indices: [qkNopeHeadDim], axis: -1)
            let qNope = qSplit[0]
            var qPe = qSplit[1]

            var compressedKV = kvAProjWithMqa(x)
            let compressedSplit = split(compressedKV, indices: [kvLoraRank], axis: -1)
            compressedKV = compressedSplit[0]
            var kPe = compressedSplit[1]
            kPe = kPe.reshaped(batchSize, sequenceLength, 1, qkRopeHeadDim)
                .transposed(0, 2, 1, 3)

            var kv = kvBProj(kvALayerNorm(compressedKV))
            kv = kv.reshaped(batchSize, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)
            let kvSplit = split(kv, indices: [qkNopeHeadDim], axis: -1)
            let kNope = kvSplit[0]
            var values = kvSplit[1]

            qPe = applyRotaryPosition(rope, to: qPe, offset: cache?.ropeOffset)
            kPe = applyRotaryPosition(rope, to: kPe, offset: cache?.ropeOffset)
            kPe = repeated(kPe, count: numHeads, axis: 1)

            let keys: MLXArray
            if let cache {
                (keys, values) = cache.update(
                    keys: concatenated([kNope, kPe], axis: -1),
                    values: values,
                )
            } else {
                keys = concatenated([kNope, kPe], axis: -1)
            }

            let queries = concatenated([qNope, qPe], axis: -1)
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

        init(_ config: KimiVLConfiguration.TextConfiguration, intermediateSize: Int? = nil) {
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
        let config: KimiVLConfiguration.TextConfiguration
        @ModuleInfo var weight: MLXArray
        @ModuleInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

        init(_ config: KimiVLConfiguration.TextConfiguration) {
            self.config = config
            _weight.wrappedValue = zeros([config.nRoutedExperts ?? 1, config.hiddenSize])
            _eScoreCorrectionBias.wrappedValue = zeros([config.nRoutedExperts ?? 1])
        }

        func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
            let topK = config.numExpertsPerToken
            var scores = sigmoid(x.matmul(weight.T).asType(.float32))
            let originalScores = scores
            var choiceScores = scores + eScoreCorrectionBias

            if config.nGroup > 1 {
                let shape = x.shape.dropLast() + [config.nGroup, -1]
                var grouped = choiceScores.reshaped(shape)
                let groupScores = top(grouped, k: 2, axis: -1).sum(axis: -1, keepDims: true)
                let keepGroups = config.nGroup - config.topKGroup
                var groupIndex = argPartition(groupScores, kth: keepGroups - 1, axis: -2)[
                    .ellipsis, ..<keepGroups, 0...,
                ]
                groupIndex = broadcast(
                    groupIndex,
                    to: x.shape.dropLast() + [
                        keepGroups,
                        (config.nRoutedExperts ?? 1) / config.nGroup,
                    ],
                )
                grouped = putAlong(
                    grouped, stopGradient(groupIndex), values: MLXArray(0.0), axis: -2)
                choiceScores = flattened(grouped, start: -2, end: -1)
            }

            let indices = argPartition(-choiceScores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
            scores = takeAlong(originalScores, indices, axis: -1)
            if topK > 1, config.normTopkProb {
                scores = scores / (scores.sum(axis: -1, keepDims: true) + 1e-20)
            }
            scores = scores * config.routedScalingFactor
            return (indices, scores)
        }
    }

    final class MoE: Module, UnaryLayer {
        let config: KimiVLConfiguration.TextConfiguration
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
        @ModuleInfo var gate: MoEGate
        @ModuleInfo(key: "shared_experts") var sharedExperts: MLP?

        init(_ config: KimiVLConfiguration.TextConfiguration) {
            self.config = config
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.nRoutedExperts ?? 1,
                activation: KimiVLLanguage.clippedSilu,
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
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        var mlp: UnaryLayer
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: KimiVLConfiguration.TextConfiguration, layerIndex: Int) {
            _selfAttention.wrappedValue = Attention(config)
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
            let residual = selfAttention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + residual
            return h + mlp(postAttentionLayerNorm(h))
        }
    }

    final class TextModel: Module {
        let config: KimiVLConfiguration.TextConfiguration
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm

        init(_ config: KimiVLConfiguration.TextConfiguration) {
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
            var hidden: MLXArray
            if let inputEmbedding {
                hidden = inputEmbedding
            } else if let inputs {
                hidden = embedTokens(inputs)
            } else {
                preconditionFailure("inputs or inputEmbedding must be provided")
            }

            let mask = createAttentionMask(h: hidden, cache: cache?.first)
            for (index, layer) in layers.enumerated() {
                hidden = layer(hidden, mask: mask, cache: cache?[index])
            }
            return norm(hidden)
        }
    }

    final class LanguageModel: Module {
        let config: KimiVLConfiguration.TextConfiguration
        @ModuleInfo var model: TextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear

        init(_ config: KimiVLConfiguration.TextConfiguration) {
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

private enum KimiVLVision {
    final class Learnable2DInterpPosEmb: Module {
        @ModuleInfo var weight: MLXArray

        init(height: Int, width: Int, dim: Int) {
            _weight.wrappedValue = MLXArray.ones([height, width, dim])
        }

        func callAsFunction(frames: [THW]) -> MLXArray {
            let sourceHeight = weight.dim(0)
            let sourceWidth = weight.dim(1)
            let dim = weight.dim(2)
            let source = weight.transposed(2, 0, 1).reshaped(1, dim, sourceHeight, sourceWidth)
            let parts = frames.map { frame -> MLXArray in
                if frame.h == sourceHeight, frame.w == sourceWidth {
                    return weight.reshaped(sourceHeight * sourceWidth, dim)
                }
                let interpolated = bicubicInterpolate(source, size: (frame.h, frame.w))
                return interpolated.reshaped(dim, frame.h * frame.w).transposed(1, 0)
            }
            return concatenated(parts, axis: 0)
        }
    }

    final class PatchEmbed: Module {
        @ModuleInfo var proj: Conv2d
        @ModuleInfo(key: "pos_emb") var posEmbedding: Learnable2DInterpPosEmb

        init(_ config: KimiVLConfiguration.VisionConfiguration) {
            _proj.wrappedValue = Conv2d(
                inputChannels: config.channels,
                outputChannels: config.embedDim,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: true,
            )
            _posEmbedding.wrappedValue = Learnable2DInterpPosEmb(
                height: config.initPosEmbeddingHeight,
                width: config.initPosEmbeddingWidth,
                dim: config.embedDim,
            )
        }

        func callAsFunction(_ x: MLXArray, frames: [THW]) -> MLXArray {
            let hidden = proj(x).reshaped(x.dim(0), -1)
            return hidden + posEmbedding(frames: frames).asType(hidden.dtype)
        }
    }

    final class Rope2DPosEmb {
        let dim: Int
        let thetaBase: Float

        init(dim: Int, thetaBase: Float = 10000) {
            self.dim = dim
            self.thetaBase = thetaBase
        }

        func cosSin(frames: [THW]) -> (MLXArray, MLXArray) {
            let half = dim / 2
            let quarter = dim / 4
            var cosValues: [Float] = []
            var sinValues: [Float] = []
            let total = frames.reduce(0) { $0 + $1.h * $1.w }
            cosValues.reserveCapacity(total * half)
            sinValues.reserveCapacity(total * half)

            for frame in frames {
                for row in 0 ..< frame.h {
                    for col in 0 ..< frame.w {
                        for index in 0 ..< quarter {
                            let frequency = pow(thetaBase, -Float(index * 4) / Float(dim))
                            let xAngle = Float(col) * frequency
                            let yAngle = Float(row) * frequency
                            cosValues.append(cos(xAngle))
                            sinValues.append(sin(xAngle))
                            cosValues.append(cos(yAngle))
                            sinValues.append(sin(yAngle))
                        }
                    }
                }
            }

            return (MLXArray(cosValues, [total, half]), MLXArray(sinValues, [total, half]))
        }
    }

    static func apply2DRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let sequence = x.dim(0)
        let heads = x.dim(1)
        let headDim = x.dim(2)
        let half = headDim / 2
        let paired = x.asType(.float32).reshaped(sequence, heads, half, 2)
        let real = paired[0..., 0..., 0..., 0]
        let imag = paired[0..., 0..., 0..., 1]
        let cos = cos.expandedDimensions(axis: 1)
        let sin = sin.expandedDimensions(axis: 1)
        let rotated = stacked(
            [
                real * cos - imag * sin,
                real * sin + imag * cos,
            ], axis: -1,
        )
        return rotated.reshaped(sequence, heads, headDim).asType(x.dtype)
    }

    static func blockAttentionMask(frames: [THW], dtype: DType) -> MLXArray? {
        guard frames.count > 1 else {
            return nil
        }
        let lengths = frames.map { $0.h * $0.w }
        let total = lengths.reduce(0, +)
        var values = Array(repeating: Float(-1e9), count: total * total)
        var offset = 0
        for length in lengths {
            for row in offset ..< offset + length {
                for col in offset ..< offset + length {
                    values[row * total + col] = 0
                }
            }
            offset += length
        }
        return MLXArray(values, [1, 1, total, total]).asType(dtype)
    }

    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo var wqkv: Linear
        @ModuleInfo var wo: Linear

        init(_ config: KimiVLConfiguration.VisionConfiguration) {
            heads = config.attentionHeads
            headDim = config.embedDim / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _wqkv.wrappedValue = Linear(config.embedDim, config.embedDim * 3, bias: true)
            _wo.wrappedValue = Linear(config.embedDim, config.embedDim, bias: true)
        }

        func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?)
            -> MLXArray
        {
            let sequence = x.dim(0)
            let qkv = wqkv(x).reshaped(sequence, 3, heads, headDim)
            let parts = split(qkv, parts: 3, axis: 1)
            var query = parts[0].squeezed(axis: 1)
            var key = parts[1].squeezed(axis: 1)
            let value = parts[2].squeezed(axis: 1)

            query = KimiVLVision.apply2DRotary(query, cos: cos, sin: sin)
            key = KimiVLVision.apply2DRotary(key, cos: cos, sin: sin)

            query = query.transposed(1, 0, 2).expandedDimensions(axis: 0)
            key = key.transposed(1, 0, 2).expandedDimensions(axis: 0)
            let valueT = value.transposed(1, 0, 2).expandedDimensions(axis: 0)

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode =
                mask.map { .array($0) } ?? .none
            let output = MLXFast.scaledDotProductAttention(
                queries: query,
                keys: key,
                values: valueT,
                scale: scale,
                mask: attentionMask,
            )
            .squeezed(axis: 0)
            .transposed(1, 0, 2)
            .reshaped(sequence, -1)
            return wo(output)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var fc0: Linear
        @ModuleInfo var fc1: Linear
        let activation = GELU()

        init(_ config: KimiVLConfiguration.VisionConfiguration) {
            _fc0.wrappedValue = Linear(config.embedDim, config.intermediateSize, bias: true)
            _fc1.wrappedValue = Linear(config.intermediateSize, config.embedDim, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc1(activation(fc0(x)))
        }
    }

    final class VisionBlock: Module {
        @ModuleInfo var norm0: LayerNorm
        @ModuleInfo var norm1: LayerNorm
        @ModuleInfo var attn: Attention
        @ModuleInfo var mlp: MLP

        init(_ config: KimiVLConfiguration.VisionConfiguration) {
            _norm0.wrappedValue = LayerNorm(
                dimensions: config.embedDim,
                eps: config.layerNormEps,
            )
            _norm1.wrappedValue = LayerNorm(
                dimensions: config.embedDim,
                eps: config.layerNormEps,
            )
            _attn.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(config)
        }

        func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mask: MLXArray?)
            -> MLXArray
        {
            let hidden = x + attn(norm0(x), cos: cos, sin: sin, mask: mask)
            return hidden + mlp(norm1(hidden))
        }
    }

    final class VisionModel: Module {
        let config: KimiVLConfiguration.VisionConfiguration
        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        @ModuleInfo(key: "blocks") var blocks: [VisionBlock]
        @ModuleInfo(key: "final_layernorm") var finalLayerNorm: LayerNorm
        let ropePosEmbedding: Rope2DPosEmb

        init(_ config: KimiVLConfiguration.VisionConfiguration) {
            self.config = config
            _patchEmbed.wrappedValue = PatchEmbed(config)
            _blocks.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
                VisionBlock(config)
            }
            _finalLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
            ropePosEmbedding = Rope2DPosEmb(dim: config.embedDim / config.attentionHeads)
        }

        var patchDType: DType {
            patchEmbed.proj.weight.dtype
        }

        func callAsFunction(_ x: MLXArray, frames: [THW]) -> [MLXArray] {
            var hidden = patchEmbed(x, frames: frames)
            let (cos, sin) = ropePosEmbedding.cosSin(frames: frames)
            let mask = KimiVLVision.blockAttentionMask(frames: frames, dtype: hidden.dtype)
            for block in blocks {
                hidden = block(
                    hidden, cos: cos.asType(hidden.dtype), sin: sin.asType(hidden.dtype), mask: mask
                )
            }
            hidden = finalLayerNorm(hidden)
            return patchMerger(hidden, frames: frames)
        }

        private func patchMerger(_ x: MLXArray, frames: [THW]) -> [MLXArray] {
            let mergeHeight = config.mergeKernelSize.first ?? 2
            let mergeWidth = config.mergeKernelSize.dropFirst().first ?? 2
            var outputs: [MLXArray] = []
            var offset = 0
            for frame in frames {
                let length = frame.h * frame.w
                let sequence = x[offset ..< offset + length, 0...]
                let newHeight = frame.h / mergeHeight
                let newWidth = frame.w / mergeWidth
                let hidden = sequence.dim(-1)
                let merged =
                    sequence
                    .reshaped(newHeight, mergeHeight, newWidth, mergeWidth, hidden)
                    .transposed(0, 2, 1, 3, 4)
                    .reshaped(newHeight * newWidth, mergeHeight * mergeWidth, hidden)
                outputs.append(merged)
                offset += length
            }
            return outputs
        }
    }
}

// MARK: - Projector

private final class KimiVLMultiModalProjector: Module {
    let inputDim: Int
    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo var act: GELU
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ config: KimiVLConfiguration) {
        let visionHidden = config.visionConfiguration.hiddenSize
        let merge = config.visionConfiguration.mergeKernelSize
        inputDim = visionHidden * (merge.first ?? 2) * (merge.dropFirst().first ?? 2)
        _preNorm.wrappedValue = LayerNorm(dimensions: visionHidden, eps: 1e-5)
        _linear1.wrappedValue = Linear(inputDim, inputDim, bias: true)
        _act.wrappedValue = GELU()
        _linear2.wrappedValue = Linear(
            inputDim,
            config.textConfiguration.hiddenSize,
            bias: true,
        )
    }

    func callAsFunction(_ imageFeatures: [MLXArray]) -> MLXArray {
        let normalized = preNorm(concatenated(imageFeatures, axis: 0))
        let hidden = normalized.reshaped(-1, inputDim)
        return linear2(act(linear1(hidden)))
    }
}

// MARK: - Model

public final class KimiVL: Module, VLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    let config: KimiVLConfiguration
    @ModuleInfo(key: "vision_tower") private var visionTower: KimiVLVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: KimiVLLanguage.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector:
        KimiVLMultiModalProjector

    public init(_ config: KimiVLConfiguration) {
        self.config = config
        vocabularySize = config.vocabularySize
        _visionTower.wrappedValue = KimiVLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = KimiVLLanguage.LanguageModel(config.textConfiguration)
        _multiModalProjector.wrappedValue = KimiVLMultiModalProjector(config)
    }

    private func inputEmbeddings(inputIDs: MLXArray, image: LMInput.ProcessedImage?) -> MLXArray {
        let embeddings = languageModel.model.embedTokens(inputIDs)
        guard let image, let frames = image.frames, !frames.isEmpty else {
            return embeddings
        }
        let pixels = image.pixels.asType(visionTower.patchDType).transposed(0, 2, 3, 1)
        let imageFeatures = multiModalProjector(visionTower(pixels, frames: frames))
            .asType(embeddings.dtype)
        let ids = inputIDs.flattened().asArray(Int.self)
        let imageTokenIDs = Set([config.imageTokenIndex, config.mediaPlaceholderTokenID])
        let positions = ids.indices.filter { imageTokenIDs.contains(ids[$0]) }
        precondition(
            positions.count == imageFeatures.dim(0),
            "Kimi-VL image feature/token count mismatch: tokens \(positions.count), features \(imageFeatures.dim(0)).",
        )
        embeddings[0..., MLXArray(positions), 0...] = imageFeatures.expandedDimensions(axis: 0)
        return embeddings
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let embeddings = inputEmbeddings(inputIDs: input.text.tokens, image: input.image)
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

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        KimiVLSanitizer.sanitize(weights: weights, config: config)
    }
}

private enum KimiVLSanitizer {
    static func sanitize(
        weights: [String: MLXArray],
        config: KimiVLConfiguration,
    )
        -> [String: MLXArray]
    {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.contains("position_ids") || key.contains("rotary_emb.inv_freq") {
                continue
            }
            var newKey = key.replacingOccurrences(
                of: "vision_tower.encoder.",
                with: "vision_tower.",
            )
            if newKey.contains("vision_tower.blocks"), !newKey.contains(".attn.") {
                newKey =
                    newKey
                    .replacingOccurrences(of: ".wqkv.", with: ".attn.wqkv.")
                    .replacingOccurrences(of: ".wo.", with: ".attn.wo.")
            }
            if newKey.hasSuffix("patch_embed.proj.weight"),
                value.ndim == 4,
                value.shape[1] == config.visionConfiguration.channels
            {
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
        config: KimiVLConfiguration.TextConfiguration,
    ) {
        guard let routedExperts = config.nRoutedExperts else {
            return
        }
        for layer in 0 ..< config.hiddenLayers {
            let prefix = "language_model.model.layers.\(layer).mlp"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).experts.0.\(projection).\(suffix)"
                    guard weights[firstKey] != nil else {
                        continue
                    }
                    var toStack: [MLXArray] = []
                    for expert in 0 ..< routedExperts {
                        let key = "\(prefix).experts.\(expert).\(projection).\(suffix)"
                        if let value = weights.removeValue(forKey: key) {
                            toStack.append(value)
                        }
                    }
                    if toStack.count == routedExperts {
                        weights["\(prefix).switch_mlp.\(projection).\(suffix)"] =
                            stacked(toStack)
                    }
                }
            }
        }
    }
}

extension Array {
    fileprivate subscript(safe index: Int) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
