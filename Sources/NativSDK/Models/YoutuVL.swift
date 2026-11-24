import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct YoutuVLConfiguration: Decodable, Sendable {
    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let outHiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let channels: Int
        public let numPatches: Int
        public let patchSize: Int
        public let hiddenActivation: String
        public let layerNormEps: Float
        public let attentionDropout: Float
        public let spatialMergeSize: Int
        public let windowSize: Int
        public let fullAttentionBlockIndexes: [Int]

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case outHiddenSize = "out_hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case channels = "num_channels"
            case numPatches = "num_patches"
            case patchSize = "patch_size"
            case hiddenActivation = "hidden_act"
            case layerNormEps = "layer_norm_eps"
            case attentionDropout = "attention_dropout"
            case spatialMergeSize = "spatial_merge_size"
            case windowSize = "window_size"
            case fullAttentionBlockIndexes = "fullatt_block_indexes"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType)
                ?? "siglip2_vision_model"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            outHiddenSize = try c.decodeIfPresent(Int.self, forKey: .outHiddenSize) ?? 2560
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 27
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            numPatches = try c.decodeIfPresent(Int.self, forKey: .numPatches) ?? 4096
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
            hiddenActivation =
                try c.decodeIfPresent(String.self, forKey: .hiddenActivation)
                ?? "gelu_pytorch_tanh"
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
            attentionDropout =
                try c.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0
            spatialMergeSize =
                try c.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
            windowSize = try c.decodeIfPresent(Int.self, forKey: .windowSize) ?? 256
            fullAttentionBlockIndexes =
                try c.decodeIfPresent([Int].self, forKey: .fullAttentionBlockIndexes)
                ?? [7, 15, 23, 26]
        }
    }

    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let vocabularySize: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        private let _kvHeads: Int?
        public var kvHeads: Int {
            _kvHeads ?? attentionHeads
        }

        public let maxPositionEmbeddings: Int
        public let rmsNormEps: Float
        public let kvLoraRank: Int
        public let qLoraRank: Int?
        public let qkRopeHeadDim: Int
        public let vHeadDim: Int
        public let qkNopeHeadDim: Int
        public let ropeTheta: Float
        public let ropeInterleave: Bool
        public let ropeTraditional: Bool
        public let ropeScaling: [String: StringOrNumber]?
        public let nSharedExperts: Int?
        public let nRoutedExperts: Int?
        public let moeIntermediateSize: Int?
        public let numExpertsPerToken: Int
        public let nGroup: Int
        public let topKGroup: Int
        public let routedScalingFactor: Float
        public let normTopkProb: Bool
        public let moeLayerFrequency: Int
        public let firstKDenseReplace: Int
        public let topKMethod: String
        public let scoringFunction: String
        public let tieWordEmbeddings: Bool
        public let attentionBias: Bool
        public let mlpBias: Bool

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case _kvHeads = "num_key_value_heads"
            case maxPositionEmbeddings = "max_position_embeddings"
            case rmsNormEps = "rms_norm_eps"
            case kvLoraRank = "kv_lora_rank"
            case qLoraRank = "q_lora_rank"
            case qkRopeHeadDim = "qk_rope_head_dim"
            case vHeadDim = "v_head_dim"
            case qkNopeHeadDim = "qk_nope_head_dim"
            case ropeTheta = "rope_theta"
            case ropeInterleave = "rope_interleave"
            case ropeTraditional = "rope_traditional"
            case ropeScaling = "rope_scaling"
            case nSharedExperts = "n_shared_experts"
            case nRoutedExperts = "n_routed_experts"
            case moeIntermediateSize = "moe_intermediate_size"
            case numExpertsPerToken = "num_experts_per_tok"
            case nGroup = "n_group"
            case topKGroup = "topk_group"
            case routedScalingFactor = "routed_scaling_factor"
            case normTopkProb = "norm_topk_prob"
            case moeLayerFrequency = "moe_layer_freq"
            case firstKDenseReplace = "first_k_dense_replace"
            case topKMethod = "topk_method"
            case scoringFunction = "scoring_func"
            case tieWordEmbeddings = "tie_word_embeddings"
            case attentionBias = "attention_bias"
            case mlpBias = "mlp_bias"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "youtu_vl"
            vocabularySize =
                try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 283_386
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2560
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 9728
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 40
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
            _kvHeads = try c.decodeIfPresent(Int.self, forKey: ._kvHeads)
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            kvLoraRank = try c.decodeIfPresent(Int.self, forKey: .kvLoraRank) ?? 512
            qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
            qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim) ?? 64
            vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim) ?? 128
            qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim) ?? 128
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000
            ropeInterleave = try c.decodeIfPresent(Bool.self, forKey: .ropeInterleave) ?? true
            let decodedTraditional =
                try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? true
            ropeTraditional = ropeInterleave ? true : decodedTraditional
            ropeScaling = try c.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling,
            )
            nSharedExperts = try c.decodeIfPresent(Int.self, forKey: .nSharedExperts)
            nRoutedExperts = try c.decodeIfPresent(Int.self, forKey: .nRoutedExperts)
            moeIntermediateSize = try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
            numExpertsPerToken =
                try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 1
            nGroup = try c.decodeIfPresent(Int.self, forKey: .nGroup) ?? 1
            topKGroup = try c.decodeIfPresent(Int.self, forKey: .topKGroup) ?? 1
            routedScalingFactor =
                try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1
            normTopkProb = try c.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
            moeLayerFrequency =
                try c.decodeIfPresent(Int.self, forKey: .moeLayerFrequency) ?? 1
            firstKDenseReplace =
                try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
            topKMethod = try c.decodeIfPresent(String.self, forKey: .topKMethod) ?? "noaux_tc"
            scoringFunction =
                try c.decodeIfPresent(String.self, forKey: .scoringFunction) ?? "sigmoid"
            tieWordEmbeddings =
                try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
            attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    public let imageTokenID: Int
    public let videoTokenID: Int
    public let visionStartTokenID: Int
    public let visionEndTokenID: Int
    public let vocabularySize: Int
    public let eosTokenIDs: [Int]

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case imageTokenID = "image_token_id"
        case videoTokenID = "video_token_id"
        case visionStartTokenID = "vision_start_token_id"
        case visionEndTokenID = "vision_end_token_id"
        case vocabularySize = "vocab_size"
        case eosTokenIDs = "eos_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "youtu_vl"
        visionConfiguration = try c.decode(VisionConfiguration.self, forKey: .visionConfiguration)
        if let nested = try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration) {
            textConfiguration = nested
        } else {
            textConfiguration = try TextConfiguration(from: decoder)
        }
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? 128_264
        videoTokenID = try c.decodeIfPresent(Int.self, forKey: .videoTokenID) ?? 128_265
        visionStartTokenID =
            try c.decodeIfPresent(Int.self, forKey: .visionStartTokenID) ?? 128_262
        visionEndTokenID =
            try c.decodeIfPresent(Int.self, forKey: .visionEndTokenID) ?? 128_263
        vocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .vocabularySize)
            ?? textConfiguration.vocabularySize
        if let ids = try? c.decode([Int].self, forKey: .eosTokenIDs) {
            eosTokenIDs = ids
        } else if let id = try? c.decode(Int.self, forKey: .eosTokenIDs) {
            eosTokenIDs = [id]
        } else {
            eosTokenIDs = [128_001]
        }
    }
}

public struct YoutuVLProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String
    public let patchSize: Int
    public let mergeSize: Int
    public let maxImagePatches: Int
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let imageToken: String
    public let videoToken: String
    public let visionStartToken: String
    public let visionEndToken: String

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case patchSize = "patch_size"
        case mergeSize = "merge_size"
        case maxImagePatches = "max_image_patches"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageToken = "image_token"
        case videoToken = "video_token"
        case visionStartToken = "vision_start_token"
        case visionEndToken = "vision_end_token"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass =
            try c.decodeIfPresent(String.self, forKey: .processorClass)
            ?? "YoutuVLProcessor"
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
        maxImagePatches =
            try c.decodeIfPresent(Int.self, forKey: .maxImagePatches) ?? 36864
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        imageToken = try c.decodeIfPresent(String.self, forKey: .imageToken) ?? "<|image_pad|>"
        videoToken = try c.decodeIfPresent(String.self, forKey: .videoToken) ?? "<|video_pad|>"
        visionStartToken =
            try c.decodeIfPresent(String.self, forKey: .visionStartToken) ?? "<|vision_start|>"
        visionEndToken =
            try c.decodeIfPresent(String.self, forKey: .visionEndToken) ?? "<|vision_end|>"
    }
}

// MARK: - Processor

public struct YoutuVLProcessor: UserInputProcessor {
    private let config: YoutuVLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: YoutuVLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty else {
            throw VLMError.singleMediaTypeAllowed
        }
        let images = MediaProcessing.images(from: input)

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
        let prompt = renderPrompt(
            input: input,
            imageTokenCounts: processedImages.map(\.placeholderCount),
        )
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
            result += "<|begin_of_text|>system\nYou are a helpful assistant.<|end_of_text|>\n"
        }

        var imageIndex = 0
        for message in messages {
            result += "<|begin_of_text|>\(message.role.rawValue)\n"
            for _ in message.images {
                result += imagePlaceholder(count: imageTokenCounts[safe: imageIndex] ?? 1)
                imageIndex += 1
            }
            if message.images.isEmpty, imageIndex < imageTokenCounts.count, message.role == .user {
                result += imagePlaceholder(count: imageTokenCounts[safe: imageIndex] ?? 1)
                imageIndex += 1
            }
            result += message.content
            result += "<|end_of_text|>\n"
        }

        while imageIndex < imageTokenCounts.count {
            let marker = "<|begin_of_text|>user\n"
            result = result.replacingOccurrences(
                of: marker,
                with: marker + imagePlaceholder(count: imageTokenCounts[imageIndex]),
                options: [],
                range: result.range(of: marker),
            )
            imageIndex += 1
        }

        result += "<|begin_of_text|>assistant\n"
        return result
    }

    private func imagePlaceholder(count: Int) -> String {
        config.visionStartToken
            + String(repeating: config.imageToken, count: max(count, 1))
            + config.visionEndToken
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?) throws
        -> ProcessedImage
    {
        let source = MediaProcessing.zeroOrigin(
            MediaProcessing.apply(image, processing: processing).toSRGB())
        let target = targetSize(height: Int(source.extent.height), width: Int(source.extent.width))
        let resized = source.resampled(
            to: CGSize(width: target.width, height: target.height),
            method: .bicubic,
        )
        let normalized = normalizedCHWArray(
            MediaProcessing.zeroOrigin(resized),
            width: target.width,
            height: target.height,
        )
        let patches = patchify(normalized, width: target.width, height: target.height)
        let gridHeight = target.height / config.patchSize
        let gridWidth = target.width / config.patchSize
        return ProcessedImage(
            pixels: patches,
            gridHeight: gridHeight,
            gridWidth: gridWidth,
            placeholderCount: gridHeight * gridWidth / max(config.mergeSize * config.mergeSize, 1),
        )
    }

    private func targetSize(height: Int, width: Int) -> (height: Int, width: Int) {
        let patchMultiple = config.patchSize * config.mergeSize
        func scaledSize(scale: Double, size: Int) -> Int {
            let scaled = Double(size) * scale
            return max(patchMultiple, Int(ceil(scaled / Double(patchMultiple))) * patchMultiple)
        }
        var scale = 1.0
        while true {
            let targetHeight = scaledSize(scale: scale, size: height)
            let targetWidth = scaledSize(scale: scale, size: width)
            let patches = (targetHeight / config.patchSize) * (targetWidth / config.patchSize)
            if patches > config.maxImagePatches {
                scale -= 0.02
            } else {
                return (targetHeight, targetWidth)
            }
        }
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
            gridHeight / config.mergeSize,
            config.mergeSize,
            config.patchSize,
            gridWidth / config.mergeSize,
            config.mergeSize,
            config.patchSize,
        )
        patches = patches.transposed(1, 4, 2, 5, 3, 6, 0)
        return patches.reshaped(gridHeight * gridWidth, -1)
    }

}

// MARK: - Language

private enum YoutuVLLanguage {
    final class Attention: Module {
        let config: YoutuVLConfiguration.TextConfiguration
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

        init(_ config: YoutuVLConfiguration.TextConfiguration) {
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
                _qALayerNorm.wrappedValue = RMSNorm(
                    dimensions: qLoraRank,
                    eps: config.rmsNormEps,
                )
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
            _kvALayerNorm.wrappedValue = RMSNorm(
                dimensions: config.kvLoraRank,
                eps: config.rmsNormEps,
            )
            _kvBProj.wrappedValue = Linear(
                config.kvLoraRank,
                config.attentionHeads * (config.qkNopeHeadDim + config.vHeadDim),
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
                traditional: config.ropeTraditional,
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

        init(_ config: YoutuVLConfiguration.TextConfiguration, intermediateSize: Int? = nil) {
            let inner = intermediateSize ?? config.intermediateSize
            _gateProj.wrappedValue = Linear(config.hiddenSize, inner, bias: config.mlpBias)
            _upProj.wrappedValue = Linear(config.hiddenSize, inner, bias: config.mlpBias)
            _downProj.wrappedValue = Linear(inner, config.hiddenSize, bias: config.mlpBias)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    final class MoEGate: Module {
        let config: YoutuVLConfiguration.TextConfiguration
        @ModuleInfo var weight: MLXArray
        @ModuleInfo(key: "e_score_correction_bias") var eScoreCorrectionBias: MLXArray

        init(_ config: YoutuVLConfiguration.TextConfiguration) {
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
        let config: YoutuVLConfiguration.TextConfiguration
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
        @ModuleInfo var gate: MoEGate
        @ModuleInfo(key: "shared_experts") var sharedExperts: MLP?

        init(_ config: YoutuVLConfiguration.TextConfiguration) {
            self.config = config
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize ?? config.intermediateSize,
                numExperts: config.nRoutedExperts ?? 1,
                activation: silu,
            )
            _gate.wrappedValue = MoEGate(config)
            if let shared = config.nSharedExperts {
                _sharedExperts.wrappedValue = MLP(
                    config,
                    intermediateSize: (config.moeIntermediateSize ?? config.intermediateSize)
                        * shared,
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

        init(_ config: YoutuVLConfiguration.TextConfiguration, layerIndex: Int) {
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
            let hidden = x + residual
            return hidden + mlp(postAttentionLayerNorm(hidden))
        }
    }

    final class TextModel: Module {
        let config: YoutuVLConfiguration.TextConfiguration
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm

        init(_ config: YoutuVLConfiguration.TextConfiguration) {
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
        let config: YoutuVLConfiguration.TextConfiguration
        @ModuleInfo var model: TextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        init(_ config: YoutuVLConfiguration.TextConfiguration) {
            self.config = config
            _model.wrappedValue = TextModel(config)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
            }
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
            let hidden = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
            if config.tieWordEmbeddings {
                return LMOutput(logits: model.embedTokens.asLinear(hidden))
            }
            return LMOutput(logits: lmHead!(hidden))
        }
    }
}

// MARK: - Vision

private enum YoutuVLVision {
    static func applyRotary(_ tensor: MLXArray, freqs: MLXArray) -> MLXArray {
        var cos = cos(freqs)
        var sin = sin(freqs)
        cos = expandedDimensions(cos, axis: 1)
        cos = tiled(cos, repetitions: [1, 1, 2])
        sin = expandedDimensions(sin, axis: 1)
        sin = tiled(sin, repetitions: [1, 1, 2])
        return ((tensor * cos) + (RotaryEmbeddingOps.rotateHalf(tensor) * sin)).asType(tensor.dtype)
    }

    final class VisionRotaryEmbedding {
        let dimensions: Int
        let theta: Float
        let inverseFreq: MLXArray

        init(dimensions: Int, theta: Float = 10000) {
            self.dimensions = dimensions
            self.theta = theta
            let positions = MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
            inverseFreq = 1.0 / pow(theta, positions / Float(dimensions))
        }

        func callAsFunction(sequenceLength: Int) -> MLXArray {
            outer(MLXArray(0 ..< sequenceLength).asType(inverseFreq.dtype), inverseFreq)
        }
    }

    final class Embeddings: Module, UnaryLayer {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear

        init(_ config: YoutuVLConfiguration.VisionConfiguration) {
            _patchEmbedding.wrappedValue = Linear(
                config.channels * config.patchSize * config.patchSize,
                config.hiddenSize,
                bias: true,
            )
        }

        func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
            patchEmbedding(pixelValues.reshaped(-1, pixelValues.dim(-1)))
        }
    }

    final class Attention: Module {
        let numHeads: Int
        let headDim: Int
        let scale: Float
        @ModuleInfo(key: "q_proj") var qProjection: Linear
        @ModuleInfo(key: "k_proj") var kProjection: Linear
        @ModuleInfo(key: "v_proj") var vProjection: Linear
        @ModuleInfo(key: "out_proj") var outputProjection: Linear

        init(_ config: YoutuVLConfiguration.VisionConfiguration) {
            numHeads = config.attentionHeads
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _qProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _kProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _vProjection.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _outputProjection.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSize, bias: true)
        }

        func callAsFunction(
            _ x: MLXArray,
            cuSeqlens: MLXArray,
            rotaryPositionEmbedding: MLXArray,
        )
            -> MLXArray
        {
            let sequenceLength = x.dim(0)
            var q = qProjection(x).reshaped(sequenceLength, numHeads, headDim)
            var k = kProjection(x).reshaped(sequenceLength, numHeads, headDim)
            let v = vProjection(x).reshaped(sequenceLength, numHeads, headDim)

            q = applyRotary(q, freqs: rotaryPositionEmbedding)
            k = applyRotary(k, freqs: rotaryPositionEmbedding)

            q = q.transposed(1, 0, 2)
            k = k.transposed(1, 0, 2)
            let value = v.transposed(1, 0, 2)

            let seqlens = cuSeqlens.asArray(Int.self)
            var outputs = [MLXArray]()
            for index in 0 ..< max(seqlens.count - 1, 0) {
                let start = seqlens[index]
                let end = seqlens[index + 1]
                guard end > start else {
                    continue
                }
                let output = MLXFast.scaledDotProductAttention(
                    queries: q[0..., start ..< end, 0...].expandedDimensions(axis: 0),
                    keys: k[0..., start ..< end, 0...].expandedDimensions(axis: 0),
                    values: value[0..., start ..< end, 0...].expandedDimensions(axis: 0),
                    scale: scale,
                    mask: .none,
                )
                outputs.append(output.squeezed(axis: 0))
            }

            let output = concatenated(outputs, axis: 1)
                .transposed(1, 0, 2)
                .reshaped(sequenceLength, -1)
            return outputProjection(output)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let activation = GELU(approximation: .tanh)

        init(_ config: YoutuVLConfiguration.VisionConfiguration) {
            _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    final class EncoderLayer: Module {
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
        @ModuleInfo var mlp: MLP

        init(_ config: YoutuVLConfiguration.VisionConfiguration) {
            _layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
            _selfAttention.wrappedValue = Attention(config)
            _layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
            _mlp.wrappedValue = MLP(config)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            cuSeqlens: MLXArray,
            rotaryPositionEmbedding: MLXArray,
        )
            -> MLXArray
        {
            let hidden =
                hiddenStates
                + selfAttention(
                    layerNorm1(hiddenStates),
                    cuSeqlens: cuSeqlens,
                    rotaryPositionEmbedding: rotaryPositionEmbedding,
                )
            return hidden + mlp(layerNorm2(hidden))
        }
    }

    final class Encoder: Module {
        let config: YoutuVLConfiguration.VisionConfiguration
        let spatialMergeSize: Int
        let windowSize: Int
        let patchSize: Int
        let spatialMergeUnit: Int
        let fullAttentionBlockIndexes: [Int]
        let rotaryPositionEmbedding: VisionRotaryEmbedding
        @ModuleInfo(key: "layers") var layers: [EncoderLayer]

        init(_ config: YoutuVLConfiguration.VisionConfiguration) {
            self.config = config
            spatialMergeSize = config.spatialMergeSize
            windowSize = config.windowSize
            patchSize = config.patchSize
            spatialMergeUnit = config.spatialMergeSize * config.spatialMergeSize
            fullAttentionBlockIndexes = config.fullAttentionBlockIndexes
            let headDimensions = config.hiddenSize / config.attentionHeads
            rotaryPositionEmbedding = VisionRotaryEmbedding(dimensions: headDimensions / 2)
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in EncoderLayer(config) }
        }

        func callAsFunction(_ hiddenStates: MLXArray, frames: [THW]) -> MLXArray {
            var hiddenStates = hiddenStates
            let rotaryEmbedding = rotaryPositionEmbeddings(frames)
            let (windowIndex, cuWindowSeqlens) = getWindowIndex(frames)
            let sequenceLength = hiddenStates.dim(0)
            let fullSeqlens = fullCuSeqlens(frames)

            hiddenStates = hiddenStates.reshaped(
                sequenceLength / spatialMergeUnit,
                spatialMergeUnit,
                -1,
            )
            hiddenStates = hiddenStates[windowIndex, 0..., 0...].reshaped(sequenceLength, -1)
            var rotary = rotaryEmbedding.reshaped(
                sequenceLength / spatialMergeUnit,
                spatialMergeUnit,
                -1,
            )
            rotary = rotary[windowIndex, 0..., 0...].reshaped(sequenceLength, -1)

            for (index, layer) in layers.enumerated() {
                let cuSeqlens =
                    fullAttentionBlockIndexes.contains(index)
                    ? fullSeqlens
                    : cuWindowSeqlens
                hiddenStates = layer(
                    hiddenStates,
                    cuSeqlens: cuSeqlens,
                    rotaryPositionEmbedding: rotary,
                )
            }

            hiddenStates = hiddenStates.reshaped(
                sequenceLength / spatialMergeUnit,
                spatialMergeUnit,
                -1,
            )
            let reverseIndex = argSort(windowIndex, axis: 0)
            return hiddenStates[reverseIndex, 0..., 0...].reshaped(sequenceLength, -1)
        }

        private func rotaryPositionEmbeddings(_ frames: [THW]) -> MLXArray {
            var positionIDs = [MLXArray]()
            for frame in frames {
                let h = frame.h
                let w = frame.w
                var hIDs = expandedDimensions(MLXArray(0 ..< h), axis: 1)
                hIDs = repeated(hIDs, count: w, axis: 1)
                hIDs =
                    hIDs
                    .reshaped(
                        h / spatialMergeSize, spatialMergeSize, w / spatialMergeSize,
                        spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()
                var wIDs = expandedDimensions(MLXArray(0 ..< w), axis: 0)
                wIDs = repeated(wIDs, count: h, axis: 0)
                wIDs =
                    wIDs
                    .reshaped(
                        h / spatialMergeSize, spatialMergeSize, w / spatialMergeSize,
                        spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()
                positionIDs.append(stacked([hIDs, wIDs], axis: -1))
            }
            let indices = concatenated(positionIDs, axis: 0)
            let maxFrameSize = frames.map { max($0.h, $0.w) }.max() ?? 0
            return rotaryPositionEmbedding(sequenceLength: maxFrameSize)[indices].reshaped(
                indices.dim(0), -1)
        }

        private func getWindowIndex(_ frames: [THW]) -> (MLXArray, MLXArray) {
            var windowIndex = [MLXArray]()
            var cuWindowSeqlens = [0]
            var windowIndexID = 0
            let vitMergerWindowSize = windowSize / spatialMergeSize / patchSize

            for frame in frames {
                let gridH = frame.h
                let gridW = frame.w
                let llmGridH = gridH / spatialMergeSize
                let llmGridW = gridW / spatialMergeSize
                let index = MLXArray(0 ..< (llmGridH * llmGridW)).reshaped(llmGridH, llmGridW)
                let padH =
                    (vitMergerWindowSize - llmGridH % vitMergerWindowSize)
                    % vitMergerWindowSize
                let padW =
                    (vitMergerWindowSize - llmGridW % vitMergerWindowSize)
                    % vitMergerWindowSize
                let numWindowsH = (llmGridH + padH) / vitMergerWindowSize
                let numWindowsW = (llmGridW + padW) / vitMergerWindowSize
                let paddedIndex = padded(
                    index,
                    widths: [[0, padH], [0, padW]],
                    mode: .constant,
                    value: MLXArray(-100),
                )
                let transposedIndex =
                    paddedIndex
                    .reshaped(numWindowsH, vitMergerWindowSize, numWindowsW, vitMergerWindowSize)
                    .transposed(0, 2, 1, 3)
                    .reshaped(numWindowsH * numWindowsW, vitMergerWindowSize, vitMergerWindowSize)
                let sequenceLengths = sum(transposedIndex .!= -100, axes: [1, 2]).reshaped(-1)
                let flattened = transposedIndex.flattened()
                let validIndices =
                    flattened.asArray(Int.self).enumerated()
                    .filter { $0.element != -100 }
                    .map(\.offset)
                windowIndex.append(flattened[MLXArray(validIndices)] + windowIndexID)
                let temporary =
                    cumsum(sequenceLengths, axis: 0) * spatialMergeUnit + cuWindowSeqlens.last!
                cuWindowSeqlens.append(contentsOf: temporary.asArray(Int.self))
                windowIndexID += llmGridH * llmGridW
            }

            let combined = concatenated(windowIndex, axis: 0)
            let seqlensArray = MLXArray(cuWindowSeqlens)
            var seen = Set<Int>()
            var unique = [Int]()
            for (index, value) in cuWindowSeqlens.enumerated() where !seen.contains(value) {
                seen.insert(value)
                unique.append(index)
            }
            return (combined, seqlensArray[MLXArray(unique)])
        }

        private func fullCuSeqlens(_ frames: [THW]) -> MLXArray {
            var seqlens = [0]
            for frame in frames {
                seqlens.append(seqlens.last! + frame.h * frame.w)
            }
            return MLXArray(seqlens)
        }
    }

    final class PatchMerger: Module, UnaryLayer {
        let hiddenSize: Int
        @ModuleInfo(key: "ln_q") var norm: RMSNorm
        @ModuleInfo(key: "mlp") var mlp: (Linear, GELU, Linear)

        init(_ config: YoutuVLConfiguration.VisionConfiguration) {
            hiddenSize = config.hiddenSize * config.spatialMergeSize * config.spatialMergeSize
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _mlp.wrappedValue = (
                Linear(hiddenSize, hiddenSize),
                GELU(),
                Linear(hiddenSize, config.outHiddenSize),
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let hidden = norm(x).reshaped(-1, hiddenSize)
            return mlp.2(gelu(mlp.0(hidden)))
        }
    }

    final class VisionModel: Module {
        let config: YoutuVLConfiguration.VisionConfiguration
        @ModuleInfo var embeddings: Embeddings
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm
        @ModuleInfo var merger: PatchMerger

        init(_ config: YoutuVLConfiguration.VisionConfiguration) {
            self.config = config
            _embeddings.wrappedValue = Embeddings(config)
            _encoder.wrappedValue = Encoder(config)
            _postLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
            _merger.wrappedValue = PatchMerger(config)
        }

        var patchDType: DType {
            embeddings.patchEmbedding.weight.dtype
        }

        func callAsFunction(_ pixelValues: MLXArray, frames: [THW]) -> MLXArray {
            var hiddenStates = embeddings(pixelValues)
            hiddenStates = encoder(hiddenStates, frames: frames)
            hiddenStates = postLayerNorm(hiddenStates)
            return merger(hiddenStates)
        }
    }
}

// MARK: - Model

public final class YoutuVL: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: YoutuVLVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: YoutuVLLanguage.LanguageModel

    public let config: YoutuVLConfiguration
    public var vocabularySize: Int {
        config.vocabularySize
    }

    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: YoutuVLConfiguration) {
        self.config = config
        _visionTower.wrappedValue = YoutuVLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = YoutuVLLanguage.LanguageModel(config.textConfiguration)
    }

    private func inputEmbeddings(inputIDs: MLXArray, image: LMInput.ProcessedImage?) throws
        -> MLXArray
    {
        let embeddings = languageModel.model.embedTokens(inputIDs)
        guard let image, let frames = image.frames, !frames.isEmpty else {
            return embeddings
        }
        let features = visionTower(image.pixels.asType(visionTower.patchDType), frames: frames)
            .asType(embeddings.dtype)
        let ids = inputIDs.flattened().asArray(Int.self)
        let imageTokenIDs = Set([config.imageTokenID, config.videoTokenID])
        let positions = ids.indices.filter { imageTokenIDs.contains(ids[$0]) }
        precondition(
            positions.count == features.dim(0),
            "YouTu-VL image feature/token count mismatch: tokens \(positions.count), features \(features.dim(0)).",
        )
        embeddings[0..., MLXArray(positions), 0...] = features.expandedDimensions(axis: 0)
        return embeddings
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let embeddings = try inputEmbeddings(inputIDs: input.text.tokens, image: input.image)
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
        YoutuVLSanitizer.sanitize(weights: weights, config: config)
    }
}

private enum YoutuVLSanitizer {
    static func sanitize(
        weights: [String: MLXArray],
        config: YoutuVLConfiguration,
    )
        -> [String: MLXArray]
    {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.contains("position_ids")
                || key.contains("position_embedding")
                || key.contains("rotary_emb.inv_freq")
            {
                continue
            }
            if key == "lm_head.weight", config.textConfiguration.tieWordEmbeddings {
                continue
            }

            var newKey = key
            if newKey.hasPrefix("siglip2.vision_model.") {
                newKey = newKey.replacingOccurrences(
                    of: "siglip2.vision_model.",
                    with: "vision_tower.",
                    options: [],
                    range: newKey.startIndex ..< newKey.index(newKey.startIndex, offsetBy: 22),
                )
            } else if newKey.hasPrefix("siglip2.") {
                newKey = newKey.replacingOccurrences(
                    of: "siglip2.",
                    with: "vision_tower.",
                    options: [],
                    range: newKey.startIndex ..< newKey.index(newKey.startIndex, offsetBy: 8),
                )
            } else if newKey.hasPrefix("merger.") {
                newKey = "vision_tower." + newKey
            } else if newKey.hasPrefix("model.") {
                newKey = newKey.replacingOccurrences(
                    of: "model.",
                    with: "language_model.model.",
                    options: [],
                    range: newKey.startIndex ..< newKey.index(newKey.startIndex, offsetBy: 6),
                )
            } else if newKey.hasPrefix("lm_head.") {
                newKey = newKey.replacingOccurrences(
                    of: "lm_head.",
                    with: "language_model.lm_head.",
                    options: [],
                    range: newKey.startIndex ..< newKey.index(newKey.startIndex, offsetBy: 8),
                )
            }
            sanitized[newKey] = value
        }
        stackExperts(weights: &sanitized, config: config.textConfiguration)
        return sanitized
    }

    private static func stackExperts(
        weights: inout [String: MLXArray],
        config: YoutuVLConfiguration.TextConfiguration,
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
