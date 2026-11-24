import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct Llama4Configuration: Decodable, Sendable {
    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let vocabularySize: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let intermediateSizeMLP: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let rmsNormEps: Float
        public let ropeTheta: Float
        public let ropeScaling: [String: StringOrNumber]?
        public let maxPositionEmbeddings: Int
        public let tieWordEmbeddings: Bool
        public let attentionBias: Bool
        public let useQKNorm: Bool
        public let numExpertsPerToken: Int
        public let numLocalExperts: Int
        public let interleaveMoELayerStep: Int
        public let noRopeLayers: [Int]
        public let attentionChunkSize: Int
        public let attnTemperatureTuning: Int
        public let floorScale: Float
        public let attnScale: Float

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case intermediateSizeMLP = "intermediate_size_mlp"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case ropeScaling = "rope_scaling"
            case maxPositionEmbeddings = "max_position_embeddings"
            case tieWordEmbeddings = "tie_word_embeddings"
            case attentionBias = "attention_bias"
            case useQKNorm = "use_qk_norm"
            case numExpertsPerToken = "num_experts_per_tok"
            case numLocalExperts = "num_local_experts"
            case interleaveMoELayerStep = "interleave_moe_layer_step"
            case noRopeLayers = "no_rope_layers"
            case attentionChunkSize = "attention_chunk_size"
            case attnTemperatureTuning = "attn_temperature_tuning"
            case floorScale = "floor_scale"
            case attnScale = "attn_scale"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "llama4_text"
            vocabularySize =
                try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 202_048
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 5120
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 8192
            intermediateSizeMLP =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSizeMLP) ?? 16384
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 48
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 40
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
            headDim =
                try c.decodeIfPresent(Int.self, forKey: .headDim)
                ?? hiddenSize / attentionHeads
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000
            ropeScaling = try c.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling,
            )
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
                ?? 10_485_760
            tieWordEmbeddings =
                try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
            attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            useQKNorm = try c.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? true
            numExpertsPerToken =
                try c.decodeIfPresent(Int.self, forKey: .numExpertsPerToken) ?? 1
            numLocalExperts = try c.decodeIfPresent(Int.self, forKey: .numLocalExperts) ?? 16
            interleaveMoELayerStep =
                try c.decodeIfPresent(Int.self, forKey: .interleaveMoELayerStep) ?? 1
            noRopeLayers =
                try c.decodeIfPresent([Int].self, forKey: .noRopeLayers)
                ?? (0 ..< hiddenLayers).map { ($0 + 1) % 4 == 0 ? 0 : 1 }
            attentionChunkSize =
                try c.decodeIfPresent(Int.self, forKey: .attentionChunkSize) ?? 8192
            attnTemperatureTuning =
                try c.decodeIfPresent(Int.self, forKey: .attnTemperatureTuning) ?? 4
            floorScale = try c.decodeIfPresent(Float.self, forKey: .floorScale) ?? 8192
            attnScale = try c.decodeIfPresent(Float.self, forKey: .attnScale) ?? 0.1
        }

        public func usesRope(layerIndex: Int) -> Bool {
            if layerIndex < noRopeLayers.count {
                return noRopeLayers[layerIndex] != 0
            }
            return (layerIndex + 1) % 4 != 0
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
        public let normEps: Float
        public let pixelShuffleRatio: Float
        public let projectorInputDim: Int
        public let projectorOutputDim: Int
        public let projectorDropout: Float
        public let visionOutputDim: Int
        public let ropeTheta: Float
        public let visionFeatureLayer: Int
        public let visionFeatureSelectStrategy: String

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case channels = "num_channels"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case normEps = "norm_eps"
            case pixelShuffleRatio = "pixel_shuffle_ratio"
            case projectorInputDim = "projector_input_dim"
            case projectorOutputDim = "projector_output_dim"
            case projectorDropout = "projector_dropout"
            case visionOutputDim = "vision_output_dim"
            case ropeTheta = "rope_theta"
            case visionFeatureLayer = "vision_feature_layer"
            case visionFeatureSelectStrategy = "vision_feature_select_strategy"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType)
                ?? "llama4_vision_model"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1408
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 5632
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 34
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 336
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
            pixelShuffleRatio =
                try c.decodeIfPresent(Float.self, forKey: .pixelShuffleRatio) ?? 0.5
            projectorInputDim =
                try c.decodeIfPresent(Int.self, forKey: .projectorInputDim) ?? 4096
            projectorOutputDim =
                try c.decodeIfPresent(Int.self, forKey: .projectorOutputDim) ?? 4096
            projectorDropout =
                try c.decodeIfPresent(Float.self, forKey: .projectorDropout) ?? 0
            visionOutputDim =
                try c.decodeIfPresent(Int.self, forKey: .visionOutputDim)
                ?? projectorOutputDim
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
            visionFeatureLayer =
                try c.decodeIfPresent(Int.self, forKey: .visionFeatureLayer) ?? -1
            visionFeatureSelectStrategy =
                try c.decodeIfPresent(String.self, forKey: .visionFeatureSelectStrategy)
                ?? "default"
        }

        public var patchesPerChunk: Int {
            let side = imageSize / patchSize
            let downsample = Int(round(1.0 / (pixelShuffleRatio * pixelShuffleRatio)))
            return side * side / max(downsample, 1)
        }
    }

    public let modelType: String
    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let imageTokenID: Int
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int {
        _imageTokenIndex ?? imageTokenID
    }

    public let vocabularySize: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case imageTokenID = "image_token_id"
        case _imageTokenIndex = "image_token_index"
        case vocabularySize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "llama4"
        textConfiguration = try c.decode(TextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try c.decode(VisionConfiguration.self, forKey: .visionConfiguration)
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? 200_092
        _imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: ._imageTokenIndex)
        vocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .vocabularySize)
            ?? textConfiguration.vocabularySize
    }
}

// MARK: - Processor

public struct Llama4ProcessorConfiguration: Decodable, Sendable {
    public struct Size: Decodable, Sendable {
        public let height: Int
        public let width: Int
    }

    public let processorClass: String
    public let size: Size
    public let maxPatches: Int
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let patchSize: Int
    public let pixelShuffleRatio: Float
    public let fakeImageToken: String
    public let imageToken: String
    public let startOfImageToken: String
    public let endOfImageToken: String
    public let patchToken: String
    public let tileXSeparatorToken: String
    public let tileYSeparatorToken: String

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case size
        case maxPatches = "max_patches"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case patchSize = "patch_size"
        case pixelShuffleRatio = "pixel_shuffle_ratio"
        case fakeImageToken = "fake_image_token"
        case imageToken = "image_token"
        case startOfImageToken = "start_of_image_token"
        case endOfImageToken = "end_of_image_token"
        case patchToken = "patch_token"
        case tileXSeparatorToken = "tile_x_separator_token"
        case tileYSeparatorToken = "tile_y_separator_token"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass =
            try c.decodeIfPresent(String.self, forKey: .processorClass)
            ?? "Llama4Processor"
        size =
            try c.decodeIfPresent(Size.self, forKey: .size)
            ?? Size(height: 336, width: 336)
        maxPatches = try c.decodeIfPresent(Int.self, forKey: .maxPatches) ?? 16
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageMean)
            ?? [0.5, 0.5, 0.5]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageStd)
            ?? [0.5, 0.5, 0.5]
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        pixelShuffleRatio =
            try c.decodeIfPresent(Float.self, forKey: .pixelShuffleRatio) ?? 0.5
        fakeImageToken =
            try c.decodeIfPresent(String.self, forKey: .fakeImageToken) ?? "<|image|>"
        imageToken =
            try c.decodeIfPresent(String.self, forKey: .imageToken) ?? "<|image|>"
        startOfImageToken =
            try c.decodeIfPresent(String.self, forKey: .startOfImageToken)
            ?? "<|image_start|>"
        endOfImageToken =
            try c.decodeIfPresent(String.self, forKey: .endOfImageToken)
            ?? "<|image_end|>"
        patchToken =
            try c.decodeIfPresent(String.self, forKey: .patchToken) ?? "<|patch|>"
        tileXSeparatorToken =
            try c.decodeIfPresent(String.self, forKey: .tileXSeparatorToken)
            ?? "<|tile_x_separator|>"
        tileYSeparatorToken =
            try c.decodeIfPresent(String.self, forKey: .tileYSeparatorToken)
            ?? "<|tile_y_separator|>"
    }

    public var downsampleRatio: Int {
        max(Int(round(1.0 / (pixelShuffleRatio * pixelShuffleRatio))), 1)
    }
}

public struct Llama4Processor: UserInputProcessor {
    private let config: Llama4ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Llama4ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty else {
            throw VLMError.singleMediaTypeAllowed
        }

        let images = MediaProcessing.images(from: input)
        let processed = try images.map {
            try preprocess(image: $0.asCIImage(), processing: input.processing)
        }
        let prompt = renderPrompt(input: input, processedImages: processed)
        let tokens = tokenizer.encode(text: prompt, addSpecialTokens: false)
        let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)

        guard !processed.isEmpty else {
            return LMInput(
                text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)))
        }

        return LMInput(
            text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)),
            image: .init(pixels: concatenated(processed.map(\.pixels), axis: 0)),
        )
    }

    private struct ProcessedImage {
        let pixels: MLXArray
        let aspectRatio: (height: Int, width: Int)
        let patchesPerChunk: Int
        let placeholderCount: Int
    }

    private func renderPrompt(input: UserInput, processedImages: [ProcessedImage]) -> String {
        let messages: [Chat.Message] =
            switch input.prompt {
            case .text(let text):
                [.user(text)]
            case .chat(let chat):
                chat
            case .messages:
                [.user(input.prompt.description)]
            }

        var result = tokenizer.bosToken ?? "<|begin_of_text|>"
        var imageIndex = 0
        for message in messages {
            result += "<|header_start|>\(message.role.rawValue)<|header_end|>\n\n"
            for _ in message.images {
                result += imagePlaceholder(processedImage(at: imageIndex, in: processedImages))
                imageIndex += 1
            }
            if message.images.isEmpty, imageIndex < processedImages.count, message.role == .user {
                result += imagePlaceholder(processedImage(at: imageIndex, in: processedImages))
                imageIndex += 1
            }
            result += message.content
            result += "<|eot|>"
        }

        while imageIndex < processedImages.count {
            result += imagePlaceholder(processedImage(at: imageIndex, in: processedImages))
            imageIndex += 1
        }
        result += "<|header_start|>assistant<|header_end|>\n\n"
        return result
    }

    private func processedImage(
        at index: Int,
        in processedImages: [ProcessedImage],
    )
        -> ProcessedImage?
    {
        guard processedImages.indices.contains(index) else {
            return nil
        }
        return processedImages[index]
    }

    private func imagePlaceholder(_ processed: ProcessedImage?) -> String {
        let aspect = processed?.aspectRatio ?? (height: 1, width: 1)
        let patches = processed?.patchesPerChunk ?? patchesPerChunk
        var result = config.startOfImageToken
        if aspect.height * aspect.width > 1 {
            for _ in 0 ..< aspect.height {
                for x in 0 ..< aspect.width {
                    result += String(repeating: config.patchToken, count: patches)
                    if x < aspect.width - 1 {
                        result += config.tileXSeparatorToken
                    }
                }
                result += config.tileYSeparatorToken
            }
        }
        result += config.imageToken
        result += String(repeating: config.patchToken, count: patches)
        result += config.endOfImageToken
        return result
    }

    private var patchesPerChunk: Int {
        let h = config.size.height / config.patchSize
        let w = config.size.width / config.patchSize
        return h * w / config.downsampleRatio
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?) throws
        -> ProcessedImage
    {
        let image = MediaProcessing.zeroOrigin(
            MediaProcessing.apply(image, processing: processing).toSRGB())
        let width = max(Int(image.extent.width.rounded()), 1)
        let height = max(Int(image.extent.height.rounded()), 1)
        let aspect = bestResolution(width: width, height: height)
        let tiles = splitIntoTiles(image: image, aspect: aspect)
        let tilePixels = tiles.map { normalizedCHWArray($0) }
        return ProcessedImage(
            pixels: concatenated(tilePixels, axis: 0),
            aspectRatio: aspect,
            patchesPerChunk: patchesPerChunk,
            placeholderCount: tilePixels.count * patchesPerChunk,
        )
    }

    private func bestResolution(width: Int, height: Int) -> (height: Int, width: Int) {
        let resolutions = [
            (1, 1), (1, 2), (1, 3), (1, 4),
            (2, 1), (2, 2), (3, 1), (4, 1),
        ]
        var best = (height: 1, width: 1)
        var bestFit = Double.greatestFiniteMagnitude
        for resolution in resolutions where resolution.0 * resolution.1 <= config.maxPatches {
            let scale = min(
                Double(resolution.1 * config.size.width) / Double(width),
                Double(resolution.0 * config.size.height) / Double(height),
            )
            let fit =
                abs(Double(width) * scale - Double(resolution.1 * config.size.width))
                + abs(Double(height) * scale - Double(resolution.0 * config.size.height))
            if fit < bestFit {
                bestFit = fit
                best = (height: resolution.0, width: resolution.1)
            }
        }
        return best
    }

    private func splitIntoTiles(
        image: CIImage,
        aspect: (height: Int, width: Int),
    )
        -> [CIImage]
    {
        let targetWidth = aspect.width * config.size.width
        let targetHeight = aspect.height * config.size.height
        let resized = MediaProcessing.zeroOrigin(
            image.resampled(
                to: CGSize(width: targetWidth, height: targetHeight),
                method: .bicubic,
            ),
        )
        var tiles = [CIImage]()
        for y in 0 ..< aspect.height {
            for x in 0 ..< aspect.width {
                let crop = CGRect(
                    x: x * config.size.width,
                    y: y * config.size.height,
                    width: config.size.width,
                    height: config.size.height,
                )
                tiles.append(MediaProcessing.zeroOrigin(resized.cropped(to: crop)))
            }
        }
        if aspect.height * aspect.width > 1 {
            tiles.append(
                MediaProcessing.zeroOrigin(
                    image.resampled(
                        to: CGSize(width: config.size.width, height: config.size.height),
                        method: .bicubic,
                    ),
                ),
            )
        }
        return tiles
    }

    private func normalizedCHWArray(_ image: CIImage) -> MLXArray {
        let width = config.size.width
        let height = config.size.height
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
        return ((unpremultiplied - mean) / std)
            .transposed(2, 0, 1)
            .expandedDimensions(axis: 0)
    }

}

// MARK: - Language

private enum Llama4Language {
    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let useRope: Bool
        let useQKNorm: Bool
        let attnTemperatureTuning: Int
        let floorScale: Float
        let attnScale: Float
        let rope: RoPELayer?

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        init(_ config: Llama4Configuration.TextConfiguration, layerIndex: Int) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)
            useRope = config.usesRope(layerIndex: layerIndex)
            useQKNorm = config.useQKNorm && useRope
            attnTemperatureTuning = config.attnTemperatureTuning
            floorScale = config.floorScale
            attnScale = config.attnScale
            if useRope {
                rope = initializeRope(
                    dims: headDim,
                    base: config.ropeTheta,
                    traditional: true,
                    scalingConfig: config.ropeScaling,
                    maxPositionEmbeddings: config.maxPositionEmbeddings,
                )
            } else {
                rope = nil
            }

            _qProj.wrappedValue = Linear(
                config.hiddenSize,
                config.attentionHeads * config.headDim,
                bias: config.attentionBias,
            )
            _kProj.wrappedValue = Linear(
                config.hiddenSize,
                config.kvHeads * config.headDim,
                bias: config.attentionBias,
            )
            _vProj.wrappedValue = Linear(
                config.hiddenSize,
                config.kvHeads * config.headDim,
                bias: config.attentionBias,
            )
            _oProj.wrappedValue = Linear(
                config.attentionHeads * config.headDim,
                config.hiddenSize,
                bias: config.attentionBias,
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
            var queries = qProj(x).reshaped(batchSize, sequenceLength, heads, headDim)
                .transposed(0, 2, 1, 3)
            var keys = kProj(x).reshaped(batchSize, sequenceLength, kvHeads, headDim)
                .transposed(0, 2, 1, 3)
            let values = vProj(x).reshaped(batchSize, sequenceLength, kvHeads, headDim)
                .transposed(0, 2, 1, 3)

            if let rope {
                queries = applyRotaryPosition(rope, to: queries, offset: cache?.ropeOffset)
                keys = applyRotaryPosition(rope, to: keys, offset: cache?.ropeOffset)
            }

            if useQKNorm {
                let normWeight = ones([headDim], type: Float32.self).asType(queries.dtype)
                queries = MLXFast.rmsNorm(queries, weight: normWeight, eps: 1e-6)
                keys = MLXFast.rmsNorm(keys, weight: normWeight, eps: 1e-6)
            }

            if attnTemperatureTuning != 0, !useRope {
                let offset = cache?.offset ?? 0
                let positions = MLXArray(Int32(offset + 1) ..< Int32(offset + sequenceLength + 1))
                    .asType(.float32)
                let scales =
                    MLX.log(MLX.floor(positions / floorScale) + 1.0) * attnScale + 1.0
                queries = (queries * scales[.newAxis, .newAxis, 0..., .newAxis])
                    .asType(queries.dtype)
            }

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
        @ModuleInfo(key: "down_proj") var downProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear

        init(_ config: Llama4Configuration.TextConfiguration, intermediateSize: Int? = nil) {
            let inner = intermediateSize ?? config.intermediateSize
            _gateProj.wrappedValue = Linear(config.hiddenSize, inner, bias: false)
            _downProj.wrappedValue = Linear(inner, config.hiddenSize, bias: false)
            _upProj.wrappedValue = Linear(config.hiddenSize, inner, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    final class MoE: Module, UnaryLayer {
        let topK: Int
        @ModuleInfo var router: Linear
        @ModuleInfo var experts: SwitchGLU
        @ModuleInfo(key: "shared_expert") var sharedExpert: MLP

        init(_ config: Llama4Configuration.TextConfiguration) {
            topK = config.numExpertsPerToken
            _experts.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.intermediateSize,
                numExperts: config.numLocalExperts,
                bias: false,
            )
            _router.wrappedValue = Linear(
                config.hiddenSize,
                config.numLocalExperts,
                bias: false,
            )
            _sharedExpert.wrappedValue = MLP(config)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let logits = router(x)
            let indices = argPartition(-logits, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
            let scores = sigmoid(takeAlong(logits, indices, axis: -1).asType(.float32))
                .asType(x.dtype)
            let expertOutputs = experts(x, indices)
            let routed = (expertOutputs * scores[.ellipsis, .newAxis]).sum(axis: -2)
            return routed + sharedExpert(x)
        }
    }

    final class DecoderLayer: Module {
        let usesChunkedAttention: Bool
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "feed_forward") var feedForward: Module & UnaryLayer
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: Llama4Configuration.TextConfiguration, layerIndex: Int) {
            usesChunkedAttention = config.usesRope(layerIndex: layerIndex)
            _attention.wrappedValue = Attention(config, layerIndex: layerIndex)
            let isMoELayer =
                layerIndex % config.interleaveMoELayerStep
                == config.interleaveMoELayerStep - 1
            if isMoELayer {
                _feedForward.wrappedValue = MoE(config)
            } else {
                _feedForward.wrappedValue = MLP(
                    config, intermediateSize: config.intermediateSizeMLP)
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
            let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
            return h + feedForward(postAttentionLayerNorm(h))
        }
    }

    final class TextModel: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        fileprivate let layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm
        let config: Llama4Configuration.TextConfiguration

        init(_ config: Llama4Configuration.TextConfiguration) {
            self.config = config
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            layers = (0 ..< config.hiddenLayers).map {
                DecoderLayer(config, layerIndex: $0)
            }
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
        )
            -> MLXArray
        {
            let h0: MLXArray
            if let inputEmbedding {
                h0 = inputEmbedding
            } else if let inputs {
                h0 = embedTokens(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            let globalMask = createAttentionMask(h: h0, cache: cache?.first)
            let chunkMask = createChunkedAttentionMask(
                sequenceLength: h0.dim(1),
                attentionChunkSize: config.attentionChunkSize,
                offset: cache?.first?.offset ?? 0,
            )
            var h = h0
            for (index, layer) in layers.enumerated() {
                let mask = layer.usesChunkedAttention ? chunkMask : globalMask
                h = layer(h, mask: mask, cache: cache?[index])
            }
            return norm(h)
        }

        private func createChunkedAttentionMask(
            sequenceLength: Int,
            attentionChunkSize: Int,
            offset: Int,
        )
            -> MLXFast.ScaledDotProductAttentionMaskMode
        {
            let end = offset + sequenceLength
            let left = MLXArray(Int32(0) ..< Int32(end))
            let right = MLXArray(Int32(offset) ..< Int32(end))[0..., .newAxis]
            let blockPositions =
                abs(left.floorDivide(attentionChunkSize) - right.floorDivide(attentionChunkSize))
            let tokenPositions = left .<= right
            return .array((blockPositions .== 0) & tokenPositions)
        }
    }

    final class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: TextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear?
        let tieWordEmbeddings: Bool
        let vocabularySize: Int
        var kvHeads: [Int]

        init(_ config: Llama4Configuration.TextConfiguration) {
            _model.wrappedValue = TextModel(config)
            tieWordEmbeddings = config.tieWordEmbeddings
            vocabularySize = config.vocabularySize
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.hiddenSize,
                    config.vocabularySize,
                    bias: false,
                )
            }
            kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
        )
            -> LMOutput
        {
            var output = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
            if let lmHead {
                output = lmHead(output)
            } else {
                output = model.embedTokens.asLinear(output)
            }
            return LMOutput(logits: output)
        }
    }
}

// MARK: - Vision

private enum Llama4Vision {
    static func applyRotary(
        q: MLXArray,
        k: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
    )
        -> (MLXArray, MLXArray)
    {
        let cos = cos[.newAxis, 0..., .newAxis, 0...]
        let sin = sin[.newAxis, 0..., .newAxis, 0...]

        func apply(_ x: MLXArray) -> MLXArray {
            let originalShape = x.shape
            let paired = x.asType(.float32).reshaped(originalShape.dropLast() + [x.dim(-1) / 2, 2])
            let x0 = paired[.ellipsis, 0]
            let x1 = paired[.ellipsis, 1]
            return stacked([x0 * cos - x1 * sin, x0 * sin + x1 * cos], axis: -1)
                .reshaped(originalShape)
                .asType(x.dtype)
        }

        return (apply(q), apply(k))
    }

    final class MultiModalProjector: Module, UnaryLayer {
        @ModuleInfo(key: "linear_1") var linear1: Linear

        init(_ config: Llama4Configuration) {
            _linear1.wrappedValue = Linear(
                config.visionConfiguration.visionOutputDim,
                config.textConfiguration.hiddenSize,
                bias: false,
            )
        }

        func callAsFunction(_ imageFeatures: MLXArray) -> MLXArray {
            linear1(imageFeatures)
        }
    }

    final class VisionMLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let activation = GELU(approximation: .fast)
        let isProjector: Bool

        init(
            _ config: Llama4Configuration.VisionConfiguration, bias: Bool = true,
            isProjector: Bool = false
        ) {
            self.isProjector = isProjector
            let fc1Input = isProjector ? config.intermediateSize : config.hiddenSize
            let fc1Output = isProjector ? config.projectorInputDim : config.intermediateSize
            let fc2Input = isProjector ? config.projectorOutputDim : config.intermediateSize
            let fc2Output = isProjector ? config.projectorOutputDim : config.hiddenSize
            _fc1.wrappedValue = Linear(fc1Input, fc1Output, bias: bias)
            _fc2.wrappedValue = Linear(fc2Input, fc2Output, bias: bias)
        }

        func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            let hidden = activation(fc1(hiddenStates))
            if isProjector {
                return activation(fc2(hidden))
            }
            return fc2(hidden)
        }
    }

    final class PixelShuffleMLP: Module, UnaryLayer {
        let scaleFactor: Int
        @ModuleInfo var mlp: VisionMLP

        init(_ config: Llama4Configuration.VisionConfiguration) {
            scaleFactor = max(Int(round(1.0 / config.pixelShuffleRatio)), 1)
            _mlp.wrappedValue = VisionMLP(config, bias: false, isProjector: true)
        }

        func callAsFunction(_ encodedPatches: MLXArray) -> MLXArray {
            mlp(pixelShuffle(encodedPatches))
        }

        private func pixelShuffle(_ x: MLXArray) -> MLXArray {
            let batch = x.dim(0)
            let sequence = x.dim(1)
            let dim = x.dim(2)
            let side = Int(Double(sequence).squareRoot())
            var hidden = x.reshaped(batch, side, side, dim)
            hidden = hidden.reshaped(batch, side, side / scaleFactor, dim * scaleFactor)
            hidden = hidden.transposed(0, 2, 1, 3)
            hidden = hidden.reshaped(
                batch,
                side / scaleFactor,
                side / scaleFactor,
                dim * scaleFactor * scaleFactor,
            )
            hidden = hidden.transposed(0, 2, 1, 3)
            return hidden.reshaped(
                batch,
                sequence / (scaleFactor * scaleFactor),
                dim * scaleFactor * scaleFactor,
            )
        }
    }

    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float
        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        init(_ config: Llama4Configuration.VisionConfiguration) {
            heads = config.attentionHeads
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _qProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _kProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _vProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _oProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
        }

        func callAsFunction(_ hiddenStates: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
            let (batch, length) = (hiddenStates.dim(0), hiddenStates.dim(1))
            var queries = qProj(hiddenStates).reshaped(batch, length, heads, headDim)
            var keys = kProj(hiddenStates).reshaped(batch, length, heads, headDim)
            var values = vProj(hiddenStates).reshaped(batch, length, heads, headDim)
            (queries, keys) = applyRotary(q: queries, k: keys, cos: cos, sin: sin)
            queries = queries.transposed(0, 2, 1, 3)
            keys = keys.transposed(0, 2, 1, 3)
            values = values.transposed(0, 2, 1, 3)
            let output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .none,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)
            return oProj(output)
        }
    }

    final class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo var mlp: VisionMLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: LayerNorm

        init(_ config: Llama4Configuration.VisionConfiguration) {
            _selfAttention.wrappedValue = Attention(config)
            _mlp.wrappedValue = VisionMLP(config)
            _inputLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
            _postAttentionLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.normEps,
            )
        }

        func callAsFunction(_ hiddenState: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
            let h = hiddenState + selfAttention(inputLayerNorm(hiddenState), cos: cos, sin: sin)
            return h + mlp(postAttentionLayerNorm(h))
        }
    }

    final class Encoder: Module {
        @ModuleInfo var layers: [EncoderLayer]

        init(_ config: Llama4Configuration.VisionConfiguration) {
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
                EncoderLayer(config)
            }
        }

        func callAsFunction(_ hiddenStates: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
            var hidden = hiddenStates
            for layer in layers {
                hidden = layer(hidden, cos: cos, sin: sin)
            }
            return hidden
        }
    }

    final class UnfoldConvolution: Module {
        let channels: Int
        let patchSize: Int
        @ModuleInfo var linear: Linear

        init(_ config: Llama4Configuration.VisionConfiguration) {
            channels = config.channels
            patchSize = config.patchSize
            _linear.wrappedValue = Linear(
                config.channels * config.patchSize * config.patchSize,
                config.hiddenSize,
                bias: false,
            )
        }

        func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            let batch = hiddenStates.dim(0)
            let height = hiddenStates.dim(2)
            let width = hiddenStates.dim(3)
            let gridHeight = height / patchSize
            let gridWidth = width / patchSize
            let patches =
                hiddenStates
                .reshaped(batch, channels, gridHeight, patchSize, gridWidth, patchSize)
                .transposed(0, 2, 4, 1, 3, 5)
                .reshaped(batch, gridHeight * gridWidth, channels * patchSize * patchSize)
            return linear(patches)
        }
    }

    final class VisionModel: Module {
        let config: Llama4Configuration.VisionConfiguration
        let numPatches: Int
        @ModuleInfo(key: "class_embedding") var classEmbedding: MLXArray
        @ModuleInfo(key: "positional_embedding_vlm") var positionalEmbeddingVLM: MLXArray
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: UnfoldConvolution
        @ModuleInfo(key: "layernorm_pre") var layerNormPre: LayerNorm
        @ModuleInfo(key: "layernorm_post") var layerNormPost: LayerNorm
        @ModuleInfo var model: Encoder
        @ModuleInfo(key: "vision_adapter") var visionAdapter: PixelShuffleMLP

        init(_ config: Llama4Configuration.VisionConfiguration) {
            self.config = config
            let side = config.imageSize / config.patchSize
            numPatches = side * side + 1
            let scale = pow(Float(config.hiddenSize), -0.5)
            _classEmbedding.wrappedValue = MLXRandom.normal([config.hiddenSize]) * scale
            _positionalEmbeddingVLM.wrappedValue =
                MLXRandom.normal([
                    numPatches,
                    config.hiddenSize,
                ]) * scale
            _patchEmbedding.wrappedValue = UnfoldConvolution(config)
            _layerNormPre.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
            _layerNormPost.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.normEps)
            _model.wrappedValue = Encoder(config)
            _visionAdapter.wrappedValue = PixelShuffleMLP(config)
        }

        func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
            var hidden = patchEmbedding(pixelValues)
            let batch = hidden.dim(0)
            let cls = broadcast(
                classEmbedding[.newAxis, .newAxis, 0...],
                to: [
                    batch,
                    1,
                    config.hiddenSize,
                ],
            )
            hidden = concatenated([hidden, cls], axis: 1)
            hidden = hidden + positionalEmbeddingVLM[.newAxis, 0..., 0...]
            hidden = layerNormPre(hidden)
            let (cos, sin) = rotaryEmbedding(dtype: hidden.dtype)
            hidden = model(hidden, cos: cos, sin: sin)
            hidden = layerNormPost(hidden)
            return visionAdapter(hidden[0..., ..<(hidden.dim(1) - 1), 0...])
        }

        private func rotaryEmbedding(dtype: DType) -> (MLXArray, MLXArray) {
            let side = config.imageSize / config.patchSize
            var indices = MLXArray(Int32(0) ..< Int32(side * side)).reshaped(side * side, 1)
            indices = concatenated([indices, indices[..<1]], axis: 0)
            indices[indices.dim(0) - 1, 0] = MLXArray(Int32(-2))
            let freqX = indices % side
            let freqY = indices.floorDivide(side)
            let freqDim = config.hiddenSize / config.attentionHeads / 2
            let positions =
                MLXArray(stride(from: 0, to: freqDim, by: 2)).asType(.float32)
                / Float(freqDim)
            let ropeFreq = 1.0 / pow(config.ropeTheta, positions)

            let xFreqs = repeated((freqX + 1).asType(.float32) * ropeFreq, count: 2, axis: -1)
            let yFreqs = repeated((freqY + 1).asType(.float32) * ropeFreq, count: 2, axis: -1)
            var freqs = concatenated([xFreqs, yFreqs], axis: -1)[
                0..., stride(from: 0, to: freqDim * 2, by: 2),
            ]
            let mask = indices .< 0
            freqs = which(mask, zeros(like: freqs), freqs)
            return (cos(freqs).asType(dtype), sin(freqs).asType(dtype))
        }
    }
}

// MARK: - Model

public final class Llama4: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_model") private var visionModel: Llama4Vision.VisionModel
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector:
        Llama4Vision.MultiModalProjector
    @ModuleInfo(key: "language_model") private var languageModel: Llama4Language.LanguageModel

    public let config: Llama4Configuration
    public var vocabularySize: Int {
        config.textConfiguration.vocabularySize
    }

    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: Llama4Configuration) {
        self.config = config
        _visionModel.wrappedValue = Llama4Vision.VisionModel(config.visionConfiguration)
        _multiModalProjector.wrappedValue = Llama4Vision.MultiModalProjector(config)
        _languageModel.wrappedValue = Llama4Language.LanguageModel(config.textConfiguration)
    }

    private func inputEmbeddings(_ input: LMInput) -> MLXArray {
        let inputIDs = input.text.tokens
        let embeddings = languageModel.model.embedTokens(inputIDs)
        guard let pixelValues = input.image?.pixels else {
            return embeddings
        }

        let imageFeatures = visionModel(pixelValues)
        let projected = multiModalProjector(imageFeatures.reshaped(-1, imageFeatures.dim(-1)))
            .asType(embeddings.dtype)
        let tokenValues = inputIDs.asArray(Int.self)
        let positions = tokenValues.enumerated()
            .compactMap { $0.element == config.imageTokenIndex ? $0.offset : nil }
        precondition(
            positions.count == projected.dim(0),
            "Llama 4 image token count \(positions.count) does not match image features \(projected.dim(0)).",
        )
        let result = embeddings
        if !positions.isEmpty {
            result[0, MLXArray(positions), 0...] = projected
        }
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
        Llama4Sanitizer.sanitize(weights: weights)
    }
}

private enum Llama4Sanitizer {
    static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.contains("position_ids") || key.contains("rotary_emb.inv_freq") {
                continue
            }
            var newKey = key
            if newKey.hasPrefix("model.language_model.") {
                newKey = newKey.replacingOccurrences(
                    of: "model.language_model.",
                    with: "language_model.",
                )
            }
            if newKey.hasPrefix("model.vision_model.") {
                newKey = newKey.replacingOccurrences(
                    of: "model.vision_model.",
                    with: "vision_model.",
                )
            }
            if newKey.hasPrefix("model.multi_modal_projector.") {
                newKey = newKey.replacingOccurrences(
                    of: "model.multi_modal_projector.",
                    with: "multi_modal_projector.",
                )
            }
            if newKey.hasPrefix("lm_head.") {
                newKey = "language_model." + newKey
            }
            sanitized[newKey] = value
        }
        return sanitized
    }
}
