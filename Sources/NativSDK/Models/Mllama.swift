import CoreImage
import Foundation
import MLX
import MLXNN

private let mllamaCrossAttentionStatesKey = LMOutput.Key<MLXArray>(
    "nativsdk.mllama.crossAttentionStates",
)

// MARK: - Configuration

public struct MllamaConfiguration: Decodable, Sendable {
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

        public let rmsNormEps: Float
        public let ropeTheta: Float
        private let _ropeTraditional: Bool?
        public var ropeTraditional: Bool {
            _ropeTraditional ?? false
        }

        public let ropeScaling: [String: StringOrNumber]?
        public let maxPositionEmbeddings: Int
        public let crossAttentionLayers: [Int]
        private let _tieWordEmbeddings: Bool?
        public var tieWordEmbeddings: Bool {
            _tieWordEmbeddings ?? false
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case _kvHeads = "num_key_value_heads"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case _ropeTraditional = "rope_traditional"
            case ropeScaling = "rope_scaling"
            case maxPositionEmbeddings = "max_position_embeddings"
            case crossAttentionLayers = "cross_attention_layers"
            case _tieWordEmbeddings = "tie_word_embeddings"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType)
                ?? "mllama_text_model"
            vocabularySize =
                try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 128_256
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 40
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
            _kvHeads = try c.decodeIfPresent(Int.self, forKey: ._kvHeads)
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000
            _ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: ._ropeTraditional)
            ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
            crossAttentionLayers =
                try c.decodeIfPresent([Int].self, forKey: .crossAttentionLayers)
                ?? [3, 8, 13, 18, 23, 28, 33, 38]
            _tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: ._tieWordEmbeddings)
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let imageSize: Int
        public let patchSize: Int
        public let channels: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let maxTiles: Int
        public let maxAspectRatioID: Int
        public let globalLayers: Int
        public let normEps: Float
        public let outputDimension: Int
        public let intermediateLayerIndices: [Int]
        public let supportedAspectRatios: [[Int]]

        public var patchesPerTile: Int {
            let side = imageSize / patchSize
            return side * side + 1
        }

        enum CodingKeys: String, CodingKey {
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case channels = "num_channels"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case attentionHeadsAlt = "attention_heads"
            case maxTiles = "max_num_tiles"
            case maxAspectRatioID = "max_aspect_ratio_id"
            case globalLayers = "num_global_layers"
            case normEps = "norm_eps"
            case outputDimension = "vision_output_dim"
            case intermediateLayerIndices = "intermediate_layers_indices"
            case supportedAspectRatios = "supported_aspect_ratios"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 560
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1280
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 5120
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
            attentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .attentionHeads)
                ?? c.decodeIfPresent(Int.self, forKey: .attentionHeadsAlt)
                ?? 16
            maxTiles = try c.decodeIfPresent(Int.self, forKey: .maxTiles) ?? 4
            maxAspectRatioID =
                try c.decodeIfPresent(Int.self, forKey: .maxAspectRatioID) ?? 8
            globalLayers = try c.decodeIfPresent(Int.self, forKey: .globalLayers) ?? 8
            normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
            outputDimension =
                try c.decodeIfPresent(Int.self, forKey: .outputDimension) ?? 7680
            intermediateLayerIndices =
                try c.decodeIfPresent([Int].self, forKey: .intermediateLayerIndices)
                ?? [3, 7, 15, 23, 30]
            supportedAspectRatios =
                try c.decodeIfPresent([[Int]].self, forKey: .supportedAspectRatios)
                ?? [[1, 1], [1, 2], [1, 3], [1, 4], [2, 1], [2, 2], [3, 1], [4, 1]]
        }
    }

    public let modelType: String
    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let imageTokenIndex: Int
    public let vocabularySize: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case imageTokenIndex = "image_token_index"
        case vocabularySize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "mllama"
        textConfiguration = try c.decode(TextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try c.decode(VisionConfiguration.self, forKey: .visionConfiguration)
        imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? 128_256
        vocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .vocabularySize)
            ?? textConfiguration.vocabularySize
    }
}

// MARK: - Processor

public struct MllamaProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String?
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let size: Size
    public let maxImageTiles: Int
    public let imageTokenID: Int

    public struct Size: Decodable, Sendable {
        public let height: Int
        public let width: Int
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case size
        case maxImageTiles = "max_image_tiles"
        case imageTokenID = "image_token_index"
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
        size =
            try c.decodeIfPresent(Size.self, forKey: .size)
            ?? Size(height: 560, width: 560)
        maxImageTiles = try c.decodeIfPresent(Int.self, forKey: .maxImageTiles) ?? 4
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? 128_256
    }
}

public struct MllamaProcessor: UserInputProcessor {
    private let config: MllamaProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: MllamaProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        let prompt = renderPrompt(from: input, imageCount: images.count)
        let tokenIDs = tokenizer.encode(text: prompt, addSpecialTokens: false)
        let tokenArray = MLXArray(tokenIDs).expandedDimensions(axis: 0)

        guard !images.isEmpty else {
            return LMInput(text: .init(tokens: tokenArray, mask: ones(like: tokenArray)))
        }

        var processedImages = [ProcessedImage]()
        for image in images {
            try processedImages.append(
                preprocess(image: image.asCIImage(), processing: input.processing))
        }

        let maxTiles = config.maxImageTiles
        let tileSize = config.size.height
        var pixels = [Float]()
        var numTiles = [Int]()
        for image in processedImages {
            pixels.append(contentsOf: image.tiles)
            if image.tileCount < maxTiles {
                pixels.append(
                    contentsOf: Array(
                        repeating: 0,
                        count: (maxTiles - image.tileCount) * 3 * tileSize * tileSize,
                    ),
                )
            }
            numTiles.append(image.tileCount)
        }

        let crossMask = Self.crossAttentionMask(
            tokenIDs: tokenIDs,
            imageTokenID: config.imageTokenID,
            numTiles: numTiles,
            maxTiles: maxTiles,
        )

        return LMInput(
            text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)),
            image: .init(
                pixels: MLXArray(
                    pixels,
                    [1, processedImages.count, maxTiles, 3, tileSize, tileSize],
                ),
                frames: processedImages.map {
                    THW($0.aspectRatioID, $0.tileCount, maxTiles)
                },
            ),
            video: .init(
                pixels: MLXArray(
                    crossMask,
                    [1, tokenIDs.count, max(processedImages.count, 1), maxTiles],
                ),
            ),
        )
    }

    private struct ProcessedImage {
        let tiles: [Float]
        let tileCount: Int
        let aspectRatioID: Int
    }

    private func renderPrompt(from input: UserInput, imageCount: Int) -> String {
        switch input.prompt {
        case .chat(let messages):
            var rendered = "<|begin_of_text|>"
            for message in messages {
                rendered += "<|start_header_id|>\(message.role.rawValue)<|end_header_id|>\n\n"
                for _ in message.images {
                    rendered += "<|image|>"
                }
                rendered += message.content
                rendered += "<|eot_id|>"
            }
            if messages.last?.role != .assistant {
                rendered += "<|start_header_id|>assistant<|end_header_id|>\n\n"
            }
            return rendered

        case .text(let text):
            let imageTags = String(repeating: "<|image|>", count: imageCount)
            return
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n\(imageTags)\(text)<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

        case .messages:
            let text = input.prompt.description
            let imageTags = String(repeating: "<|image|>", count: imageCount)
            return
                "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n\(imageTags)\(text)<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        }
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?) throws
        -> ProcessedImage
    {
        let canonical = MediaProcessing.zeroOrigin(
            MediaProcessing.apply(image, processing: processing))
        let originalHeight = max(Int(canonical.extent.height.rounded()), 1)
        let originalWidth = max(Int(canonical.extent.width.rounded()), 1)
        let tileSize = config.size.height
        let canvas = Self.optimalCanvas(
            imageHeight: originalHeight,
            imageWidth: originalWidth,
            maxTiles: config.maxImageTiles,
            tileSize: tileSize,
        )
        let target = Self.imageSizeFitToCanvas(
            imageHeight: originalHeight,
            imageWidth: originalWidth,
            canvasHeight: canvas.height,
            canvasWidth: canvas.width,
            tileSize: tileSize,
        )

        let resized = canonical.resampled(
            to: CGSize(width: target.width, height: target.height),
            method: .bicubic,
        )
        let raw = MediaProcessing.renderRGB8(resized, width: target.width, height: target.height)
        var padded = Array(repeating: Float(0), count: canvas.height * canvas.width * 3)
        for row in 0 ..< target.height {
            for col in 0 ..< target.width {
                let source = (row * target.width + col) * 3
                let destination = (row * canvas.width + col) * 3
                padded[destination] = raw[source]
                padded[destination + 1] = raw[source + 1]
                padded[destination + 2] = raw[source + 2]
            }
        }

        var tiles = [Float]()
        let hTiles = canvas.height / tileSize
        let wTiles = canvas.width / tileSize
        for h in 0 ..< hTiles {
            for w in 0 ..< wTiles {
                let tile = MediaProcessing.cropPixels(
                    padded,
                    sourceWidth: canvas.width,
                    x: w * tileSize,
                    y: h * tileSize,
                    width: tileSize,
                    height: tileSize,
                    channels: 3,
                )
                tiles.append(
                    contentsOf: MediaProcessing.normalizeCHW(
                        tile,
                        width: tileSize,
                        height: tileSize,
                        mean: config.imageMean,
                        std: config.imageStd,
                    ),
                )
            }
        }

        let aspectID = Self.aspectRatioID(
            heightTiles: hTiles,
            widthTiles: wTiles,
            maxTiles: config.maxImageTiles,
        )
        return ProcessedImage(
            tiles: tiles,
            tileCount: hTiles * wTiles,
            aspectRatioID: aspectID,
        )
    }

    private static func crossAttentionMask(
        tokenIDs: [Int],
        imageTokenID: Int,
        numTiles: [Int],
        maxTiles: Int,
    )
        -> [Int32]
    {
        let imageLocations = tokenIDs.enumerated().compactMap { index, token in
            token == imageTokenID ? index : nil
        }
        let imageCount = max(imageLocations.count, 1)
        var result = Array(repeating: Int32(0), count: tokenIDs.count * imageCount * maxTiles)
        guard !imageLocations.isEmpty else {
            return result
        }

        var ranges = [(Int, Int)]()
        if imageLocations.count == 1 {
            ranges.append((imageLocations[0], tokenIDs.count))
        } else {
            for index in 0 ..< (imageLocations.count - 1) {
                ranges.append((imageLocations[index], imageLocations[index + 1]))
            }
            ranges.append((imageLocations.last!, tokenIDs.count))
            var lastEnd = ranges.last!.1
            for index in stride(from: ranges.count - 1, through: 0, by: -1) {
                if ranges[index].0 == ranges[index].1 - 1 {
                    ranges[index].1 = lastEnd
                }
                lastEnd = ranges[index].1
            }
        }

        for (imageIndex, range) in ranges.enumerated() {
            let validTiles = numTiles.indices.contains(imageIndex) ? numTiles[imageIndex] : maxTiles
            let start = max(range.0, 0)
            let end = min(range.1, tokenIDs.count)
            guard start < end else {
                continue
            }
            for token in start ..< end {
                for tile in 0 ..< min(validTiles, maxTiles) {
                    result[(token * imageCount + imageIndex) * maxTiles + tile] = 1
                }
            }
        }
        return result
    }

    private static func aspectRatioID(heightTiles: Int, widthTiles: Int, maxTiles: Int) -> Int {
        var supported = [(Int, Int)]()
        for height in 1 ... maxTiles {
            for width in 1 ... maxTiles where height * width <= maxTiles {
                supported.append((height, width))
            }
        }
        return (supported.firstIndex { $0 == (heightTiles, widthTiles) } ?? 0) + 1
    }

    private static func optimalCanvas(
        imageHeight: Int,
        imageWidth: Int,
        maxTiles: Int,
        tileSize: Int,
    )
        -> (height: Int, width: Int)
    {
        var candidates = [(height: Int, width: Int, scale: Float, area: Int)]()
        for heightTiles in 1 ... maxTiles {
            for widthTiles in 1 ... maxTiles where heightTiles * widthTiles <= maxTiles {
                let height = heightTiles * tileSize
                let width = widthTiles * tileSize
                let scale = min(
                    Float(height) / Float(imageHeight), Float(width) / Float(imageWidth))
                candidates.append((height, width, scale, height * width))
            }
        }
        let upscales = candidates.filter { $0.scale >= 1 }
        let selectedScale =
            if !upscales.isEmpty {
                upscales.map(\.scale).min()!
            } else {
                candidates.filter { $0.scale < 1 }.map(\.scale).max() ?? 1
            }
        let selected =
            candidates
            .filter { abs($0.scale - selectedScale) < 1e-6 }
            .min { lhs, rhs in lhs.area < rhs.area }!
        return (selected.height, selected.width)
    }

    private static func imageSizeFitToCanvas(
        imageHeight: Int,
        imageWidth: Int,
        canvasHeight: Int,
        canvasWidth: Int,
        tileSize: Int,
    )
        -> (height: Int, width: Int)
    {
        let targetWidth = min(max(imageWidth, tileSize), canvasWidth)
        let targetHeight = min(max(imageHeight, tileSize), canvasHeight)
        let scaleH = Float(targetHeight) / Float(imageHeight)
        let scaleW = Float(targetWidth) / Float(imageWidth)
        if scaleW < scaleH {
            let newWidth = targetWidth
            let newHeight = min(max(Int(floor(Float(imageHeight) * scaleW)), 1), targetHeight)
            return (newHeight, newWidth)
        } else {
            let newHeight = targetHeight
            let newWidth = min(max(Int(floor(Float(imageWidth) * scaleH)), 1), targetWidth)
            return (newHeight, newWidth)
        }
    }

}

// MARK: - Text

private enum MllamaText {
    final class SelfAttention: Module {
        let config: MllamaConfiguration.TextConfiguration
        let headDim: Int
        let scale: Float
        let rope: RoPELayer

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        init(_ config: MllamaConfiguration.TextConfiguration) {
            self.config = config
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            rope = initializeRope(
                dims: headDim,
                base: config.ropeTheta,
                traditional: config.ropeTraditional,
                scalingConfig: config.ropeScaling,
                maxPositionEmbeddings: config.maxPositionEmbeddings,
            )
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
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let (batch, length) = (x.dim(0), x.dim(1))
            var queries = qProj(x)
                .reshaped(batch, length, config.attentionHeads, headDim)
                .transposed(0, 2, 1, 3)
            var keys = kProj(x)
                .reshaped(batch, length, config.kvHeads, headDim)
                .transposed(0, 2, 1, 3)
            let values = vProj(x)
                .reshaped(batch, length, config.kvHeads, headDim)
                .transposed(0, 2, 1, 3)

            queries = applyRotaryPosition(rope, to: queries, offset: cache?.ropeOffset)
            keys = applyRotaryPosition(rope, to: keys, offset: cache?.ropeOffset)

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask,
            )
            return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
        }
    }

    final class CrossAttention: Module {
        let config: MllamaConfiguration.TextConfiguration
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        init(_ config: MllamaConfiguration.TextConfiguration) {
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
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            crossAttentionStates: MLXArray?,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
        )
            -> MLXArray
        {
            guard let crossAttentionStates else {
                return MLXArray.zeros(hiddenStates.shape)
            }
            let (batch, length) = (hiddenStates.dim(0), hiddenStates.dim(1))
            let query = qNorm(
                qProj(hiddenStates)
                    .reshaped(batch, length, config.attentionHeads, headDim)
                    .transposed(0, 2, 1, 3),
            )
            let keys = kNorm(
                kProj(crossAttentionStates)
                    .reshaped(batch, -1, config.kvHeads, headDim)
                    .transposed(0, 2, 1, 3),
            )
            let values = vProj(crossAttentionStates)
                .reshaped(batch, -1, config.kvHeads, headDim)
                .transposed(0, 2, 1, 3)
            let output = MLXFast.scaledDotProductAttention(
                queries: query,
                keys: keys,
                values: values,
                scale: scale,
                mask: mask,
            )
            return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        init(_ config: MllamaConfiguration.TextConfiguration) {
            _gateProj.wrappedValue = Linear(
                config.hiddenSize,
                config.intermediateSize,
                bias: false,
            )
            _upProj.wrappedValue = Linear(
                config.hiddenSize,
                config.intermediateSize,
                bias: false,
            )
            _downProj.wrappedValue = Linear(
                config.intermediateSize,
                config.hiddenSize,
                bias: false,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    final class SelfAttentionLayer: Module {
        @ModuleInfo(key: "self_attn") var selfAttention: SelfAttention
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: MllamaConfiguration.TextConfiguration) {
            _selfAttention.wrappedValue = SelfAttention(config)
            _mlp.wrappedValue = MLP(config)
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

    final class CrossAttentionLayer: Module {
        @ModuleInfo(key: "cross_attn") var crossAttention: CrossAttention
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
        @ModuleInfo(key: "cross_attn_attn_gate") var attentionGate: MLXArray
        @ModuleInfo(key: "cross_attn_mlp_gate") var mlpGate: MLXArray

        init(_ config: MllamaConfiguration.TextConfiguration) {
            _crossAttention.wrappedValue = CrossAttention(config)
            _mlp.wrappedValue = MLP(config)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _attentionGate.wrappedValue = MLXArray.zeros([1])
            _mlpGate.wrappedValue = MLXArray.zeros([1])
        }

        func callAsFunction(
            _ x: MLXArray,
            crossAttentionStates: MLXArray?,
            crossAttentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
            fullTextRowMaskedOutMask: MLXArray?,
        )
            -> MLXArray
        {
            let h =
                x
                + tanh(attentionGate)
                * crossAttention(
                    inputLayerNorm(x),
                    crossAttentionStates: crossAttentionStates,
                    mask: crossAttentionMask,
                )
            var residual = mlp(postAttentionLayerNorm(h))
            if let fullTextRowMaskedOutMask {
                residual = fullTextRowMaskedOutMask[0..., 0, 0..., 0...] * residual
            }
            return h + tanh(mlpGate) * residual
        }
    }

    enum DecoderLayer {
        case selfAttention(SelfAttentionLayer)
        case crossAttention(CrossAttentionLayer)

        var module: Module {
            switch self {
            case .selfAttention(let layer): layer
            case .crossAttention(let layer): layer
            }
        }
    }

    final class TextModel: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo var layers: [Module]
        @ModuleInfo var norm: RMSNorm

        let config: MllamaConfiguration.TextConfiguration
        let decoderLayers: [DecoderLayer]

        init(_ config: MllamaConfiguration.TextConfiguration) {
            self.config = config
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize + 8,
                dimensions: config.hiddenSize,
            )
            var modules = [Module]()
            var decoderLayers = [DecoderLayer]()
            for index in 0 ..< config.hiddenLayers {
                if config.crossAttentionLayers.contains(index) {
                    let layer = CrossAttentionLayer(config)
                    modules.append(layer)
                    decoderLayers.append(.crossAttention(layer))
                } else {
                    let layer = SelfAttentionLayer(config)
                    modules.append(layer)
                    decoderLayers.append(.selfAttention(layer))
                }
            }
            self.decoderLayers = decoderLayers
            _layers.wrappedValue = modules
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ inputIDs: MLXArray,
            cache: [KVCache]?,
            crossAttentionStates: MLXArray?,
            crossAttentionMask: MLXArray?,
            fullTextRowMaskedOutMask: MLXArray?,
        )
            -> MLXArray
        {
            var h = embedTokens(inputIDs)
            let selfMask = createAttentionMask(h: h, cache: cache?.first)
            let crossMask =
                crossAttentionMask.map {
                    MLXFast.ScaledDotProductAttentionMaskMode.array($0)
                } ?? .none
            for (index, layer) in decoderLayers.enumerated() {
                switch layer {
                case .selfAttention(let layer):
                    h = layer(h, mask: selfMask, cache: cache?[index])
                case .crossAttention(let layer):
                    h = layer(
                        h,
                        crossAttentionStates: crossAttentionStates,
                        crossAttentionMask: crossMask,
                        fullTextRowMaskedOutMask: fullTextRowMaskedOutMask,
                    )
                }
            }
            return norm(h)
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo var model: TextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear

        let config: MllamaConfiguration.TextConfiguration

        init(_ config: MllamaConfiguration.TextConfiguration) {
            self.config = config
            _model.wrappedValue = TextModel(config)
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }

        func callAsFunction(
            _ inputIDs: MLXArray,
            cache: [KVCache]?,
            crossAttentionStates: MLXArray?,
            crossAttentionMask: MLXArray?,
            fullTextRowMaskedOutMask: MLXArray?,
        )
            -> MLXArray
        {
            lmHead(
                model(
                    inputIDs,
                    cache: cache,
                    crossAttentionStates: crossAttentionStates,
                    crossAttentionMask: crossAttentionMask,
                    fullTextRowMaskedOutMask: fullTextRowMaskedOutMask,
                ),
            )
        }
    }
}

// MARK: - Vision

private enum MllamaVision {
    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        init(_ config: MllamaConfiguration.VisionConfiguration) {
            heads = config.attentionHeads
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _qProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
            _kProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
            _vProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
            _oProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        )
            -> MLXArray
        {
            let (batch, length) = (x.dim(0), x.dim(1))
            let query = qProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let keys = kProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let values = vProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let output = MLXFast.scaledDotProductAttention(
                queries: query,
                keys: keys,
                values: values,
                scale: scale,
                mask: attentionMask,
            )
            return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let gelu = GELU()

        init(_ config: MllamaConfiguration.VisionConfiguration) {
            _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(gelu(fc1(x)))
        }
    }

    final class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: LayerNorm
        @ModuleInfo(key: "gate_attn") var gateAttention: MLXArray?
        @ModuleInfo(key: "gate_ffn") var gateFFN: MLXArray?

        let gated: Bool

        init(_ config: MllamaConfiguration.VisionConfiguration, gated: Bool) {
            self.gated = gated
            _attention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(config)
            _inputLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.normEps,
            )
            _postAttentionLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.normEps,
            )
            if gated {
                _gateAttention.wrappedValue = MLXArray.zeros([1])
                _gateFFN.wrappedValue = MLXArray.zeros([1])
            }
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
        )
            -> MLXArray
        {
            var h = attention(inputLayerNorm(x), attentionMask: attentionMask)
            if let gateAttention {
                h = tanh(gateAttention) * h
            }
            let out = x + h
            h = mlp(postAttentionLayerNorm(out))
            if let gateFFN {
                h = tanh(gateFFN) * h
            }
            return out + h
        }
    }

    final class Encoder: Module {
        @ModuleInfo var layers: [EncoderLayer]

        init(_ config: MllamaConfiguration.VisionConfiguration, count: Int, gated: Bool) {
            _layers.wrappedValue = (0 ..< count).map { _ in
                EncoderLayer(config, gated: gated)
            }
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionMask: MLXFast.ScaledDotProductAttentionMaskMode,
            collectStates: Bool,
        )
            -> (MLXArray, [MLXArray])
        {
            var h = x
            var states = [MLXArray]()
            for layer in layers {
                h = layer(h, attentionMask: attentionMask)
                if collectStates {
                    states.append(h)
                }
            }
            return (h, states)
        }
    }

    final class AspectRatioEmbedding: Module {
        @ModuleInfo var embedding: Embedding
        @ModuleInfo var gate: MLXArray?

        let config: MllamaConfiguration.VisionConfiguration

        init(_ config: MllamaConfiguration.VisionConfiguration, gated: Bool) {
            self.config = config
            _embedding.wrappedValue = Embedding(
                embeddingCount: config.maxAspectRatioID + 1,
                dimensions: config.maxTiles * config.hiddenSize,
            )
            if gated {
                _gate.wrappedValue = MLXArray.zeros([1])
            }
        }

        func callAsFunction(_ hiddenState: MLXArray, aspectRatioIDs: MLXArray) -> MLXArray {
            var embeddings = embedding(aspectRatioIDs)
                .reshaped(-1, config.maxTiles, 1, config.hiddenSize)
            if let gate {
                embeddings = embeddings * tanh(gate)
            }
            return hiddenState + embeddings
        }
    }

    final class PositionEmbedding: Module {
        @ModuleInfo var embedding: MLXArray
        @ModuleInfo var gate: MLXArray
        @ModuleInfo(key: "tile_embedding") var tileEmbedding: Embedding

        let config: MllamaConfiguration.VisionConfiguration

        init(_ config: MllamaConfiguration.VisionConfiguration) {
            self.config = config
            _embedding.wrappedValue = MLXArray.zeros([
                config.patchesPerTile,
                config.hiddenSize,
            ])
            _gate.wrappedValue = MLXArray.zeros([1])
            _tileEmbedding.wrappedValue = Embedding(
                embeddingCount: config.maxAspectRatioID + 1,
                dimensions: config.maxTiles * config.patchesPerTile * config.hiddenSize,
            )
        }

        func callAsFunction(_ hiddenState: MLXArray, aspectRatioIDs: MLXArray) -> MLXArray {
            var h = hiddenState
            let positionEmbedding = (1 - tanh(gate)) * embedding
            h = h + positionEmbedding.reshaped(1, 1, config.patchesPerTile, config.hiddenSize)
            let batch = h.dim(0)
            let tilePositionEmbedding = tileEmbedding(aspectRatioIDs)
                .reshaped(batch, config.maxTiles, config.patchesPerTile, config.hiddenSize)
            h = h + tanh(gate) * tilePositionEmbedding
            return h
        }
    }

    final class VisionModel: Module {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "class_embedding") var classEmbedding: MLXArray
        @ModuleInfo(key: "gated_positional_embedding") var gatedPositionEmbedding: PositionEmbedding
        @ModuleInfo(key: "pre_tile_positional_embedding") var preTilePositionEmbedding:
            AspectRatioEmbedding
        @ModuleInfo(key: "post_tile_positional_embedding") var postTilePositionEmbedding:
            AspectRatioEmbedding
        @ModuleInfo(key: "layernorm_pre") var layerNormPre: LayerNorm
        @ModuleInfo(key: "layernorm_post") var layerNormPost: LayerNorm
        @ModuleInfo var transformer: Encoder
        @ModuleInfo(key: "global_transformer") var globalTransformer: Encoder

        let config: MllamaConfiguration.VisionConfiguration

        init(_ config: MllamaConfiguration.VisionConfiguration) {
            self.config = config
            _patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.channels,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: false,
            )
            _classEmbedding.wrappedValue = MLXArray.zeros([config.hiddenSize])
            _gatedPositionEmbedding.wrappedValue = PositionEmbedding(config)
            _preTilePositionEmbedding.wrappedValue = AspectRatioEmbedding(config, gated: true)
            _postTilePositionEmbedding.wrappedValue = AspectRatioEmbedding(config, gated: true)
            _layerNormPre.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.normEps,
            )
            _layerNormPost.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.normEps,
            )
            _transformer.wrappedValue = Encoder(
                config,
                count: config.hiddenLayers,
                gated: false,
            )
            _globalTransformer.wrappedValue = Encoder(
                config,
                count: config.globalLayers,
                gated: true,
            )
        }

        func callAsFunction(
            pixelValues: MLXArray,
            aspectRatioIDs: MLXArray,
            aspectRatioMask: MLXArray,
        )
            -> MLXArray
        {
            let batchSize = pixelValues.dim(0)
            let mediaCount = pixelValues.dim(1)
            let tileCount = pixelValues.dim(2)
            let height = pixelValues.dim(4)
            let width = pixelValues.dim(5)
            let mediaBatch = batchSize * mediaCount
            let patchesPerTile = config.patchesPerTile

            let flatPixels =
                pixelValues
                .reshaped(batchSize * mediaCount * tileCount, config.channels, height, width)
                .transposed(0, 2, 3, 1)
            var hiddenState = patchEmbedding(flatPixels).flattened(start: 1, end: 2)
            var patchCount = hiddenState.dim(1)
            hiddenState = hiddenState.reshaped(mediaBatch, tileCount, patchCount, config.hiddenSize)
            hiddenState = preTilePositionEmbedding(
                hiddenState,
                aspectRatioIDs: aspectRatioIDs.reshaped(mediaBatch, -1),
            )

            hiddenState = hiddenState.reshaped(
                mediaBatch * tileCount,
                patchCount,
                config.hiddenSize,
            )
            let classEmbedding = broadcast(
                classEmbedding.asType(hiddenState.dtype)[.newAxis, .newAxis, 0...],
                to: [mediaBatch * tileCount, 1, config.hiddenSize],
            )
            hiddenState = concatenated([classEmbedding, hiddenState], axis: 1)
            patchCount += 1

            hiddenState = hiddenState.reshaped(mediaBatch, tileCount, patchCount, config.hiddenSize)
            hiddenState = gatedPositionEmbedding(
                hiddenState,
                aspectRatioIDs: aspectRatioIDs.reshaped(mediaBatch, -1),
            )
            hiddenState = layerNormPre(hiddenState)

            let paddingPatches = (8 - (hiddenState.dim(2) % 8)) % 8
            if paddingPatches > 0 {
                hiddenState = padded(
                    hiddenState,
                    widths: [
                        .init((0, 0)),
                        .init((0, 0)),
                        .init((0, paddingPatches)),
                        .init((0, 0)),
                    ],
                )
            }

            let attentionMask = Self.aspectRatioAttentionMask(
                aspectRatioMask: aspectRatioMask.reshaped(mediaBatch, -1),
                patchesPerTile: patchesPerTile,
                targetLength: hiddenState.dim(2),
            )
            hiddenState = hiddenState.reshaped(mediaBatch, -1, config.hiddenSize)
            let encoderOutput = transformer(
                hiddenState,
                attentionMask: .array(attentionMask),
                collectStates: true,
            )
            hiddenState = layerNormPost(encoderOutput.0)

            hiddenState = hiddenState.reshaped(
                mediaBatch,
                tileCount,
                patchCount + paddingPatches,
                config.hiddenSize,
            )
            hiddenState = postTilePositionEmbedding(
                hiddenState,
                aspectRatioIDs: aspectRatioIDs.reshaped(mediaBatch, -1),
            )
            hiddenState = hiddenState.reshaped(
                mediaBatch,
                tileCount * (patchCount + paddingPatches),
                config.hiddenSize,
            )
            hiddenState =
                globalTransformer(
                    hiddenState,
                    attentionMask: .array(attentionMask),
                    collectStates: false,
                ).0
            hiddenState = hiddenState.reshaped(
                mediaBatch,
                tileCount,
                patchCount + paddingPatches,
                config.hiddenSize,
            )
            hiddenState = hiddenState[0..., 0..., 0 ..< patchCount, 0...]
                .reshaped(batchSize, mediaCount, tileCount, patchCount, config.hiddenSize)

            let allStates = stacked(encoderOutput.1, axis: -1)
            var intermediate = allStates[.ellipsis, MLXArray(config.intermediateLayerIndices)]
            intermediate = intermediate.reshaped(
                mediaBatch,
                tileCount,
                patchCount + paddingPatches,
                -1,
            )
            intermediate = intermediate[0..., 0..., 0 ..< patchCount, 0...]
                .reshaped(batchSize, mediaCount, tileCount, patchCount, -1)

            return concatenated([hiddenState, intermediate], axis: -1)
        }

        private static func aspectRatioAttentionMask(
            aspectRatioMask: MLXArray,
            patchesPerTile: Int,
            targetLength: Int,
        )
            -> MLXArray
        {
            var mask = aspectRatioMask.asType(.float32)
            let batch = mask.dim(0)
            let maxTiles = mask.dim(1)
            mask = mask.reshaped(batch, maxTiles, 1, 1)
            mask = tiled(mask, repetitions: [1, 1, targetLength, 1])
            let padPatches = targetLength - patchesPerTile
            if padPatches > 0 {
                mask[0..., 0..., (targetLength - padPatches)..., 0...] = MLXArray(0.0)
            }
            mask = 1 - mask
            mask = mask.reshaped(batch, maxTiles * targetLength, 1)
            let attention = matmul(mask, mask.transposed(0, 2, 1)) * MLXArray(-1e9)
            return attention[0..., .newAxis, 0..., 0...]
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized = [String: MLXArray]()
            for (key, value) in weights {
                if key.contains("position_ids") {
                    continue
                }
                if key.contains("patch_embedding.weight"), value.ndim == 4 {
                    let shape = value.shape
                    sanitized[key] =
                        shape[1] == config.patchSize ? value : value.transposed(0, 2, 3, 1)
                } else {
                    sanitized[key] = value
                }
            }
            return sanitized
        }
    }
}

// MARK: - Model

public final class Mllama: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") fileprivate var visionTower: MllamaVision.VisionModel
    @ModuleInfo(key: "language_model") fileprivate var languageModel: MllamaText.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") var multiModalProjector: Linear

    public let config: MllamaConfiguration
    public var kvHeads: [Int] {
        (0 ..< config.textConfiguration.hiddenLayers).map { _ in
            config.textConfiguration.kvHeads
        }
    }

    public init(_ config: MllamaConfiguration) {
        self.config = config
        _visionTower.wrappedValue = MllamaVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = MllamaText.LanguageModel(config.textConfiguration)
        _multiModalProjector.wrappedValue = Linear(
            config.visionConfiguration.outputDimension,
            config.textConfiguration.hiddenSize,
            bias: true,
        )
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    private func prepareCrossAttentionMask(
        _ crossAttentionMask: MLXArray?,
        numVisionTokens: Int,
        positions: Range<Int>? = nil,
    )
        -> (MLXArray?, MLXArray?)
    {
        guard var crossAttentionMask else {
            return (nil, nil)
        }
        if let positions {
            crossAttentionMask = crossAttentionMask[0..., positions, 0..., 0...]
        }
        let batch = crossAttentionMask.dim(0)
        let textLength = crossAttentionMask.dim(1)
        crossAttentionMask = repeated(crossAttentionMask, count: numVisionTokens, axis: 3)
        crossAttentionMask = crossAttentionMask.reshaped(batch, textLength, -1)
        crossAttentionMask = crossAttentionMask[0..., .newAxis, 0..., 0...]
        let inverted = 1.0 - crossAttentionMask
        let filled = MLXArray(-1e9)
        crossAttentionMask = MLX.where(inverted.asType(.bool), filled, crossAttentionMask)
        let fullRow = (crossAttentionMask .!= MLXArray(-1e9)).any(axis: -1, keepDims: true)
            .asType(crossAttentionMask.dtype)
        return (crossAttentionMask * fullRow, fullRow)
    }

    private func visionFeatures(
        pixelValues: MLXArray?,
        frames: [THW]?,
    )
        -> MLXArray?
    {
        guard let pixelValues else {
            return nil
        }
        let imageCount = pixelValues.dim(1)
        let maxTiles = pixelValues.dim(2)
        let frameValues = frames ?? Array(repeating: THW(1, 1, maxTiles), count: imageCount)
        let aspectRatioIDs = MLXArray(
            frameValues.map { Int32($0.t) },
            [1, imageCount],
        )
        var maskValues = [Int32]()
        for frame in frameValues {
            for index in 0 ..< maxTiles {
                maskValues.append(index < frame.h ? 1 : 0)
            }
        }
        let aspectRatioMask = MLXArray(maskValues, [1, imageCount, maxTiles])
        let features = visionTower(
            pixelValues: pixelValues,
            aspectRatioIDs: aspectRatioIDs,
            aspectRatioMask: aspectRatioMask,
        )
        return multiModalProjector(features).reshaped(
            pixelValues.dim(0),
            -1,
            config.textConfiguration.hiddenSize,
        )
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let crossStates = visionFeatures(
            pixelValues: input.image?.pixels,
            frames: input.image?.frames,
        )
        let (crossMask, fullRowMask) = prepareCrossAttentionMask(
            input.video?.pixels,
            numVisionTokens: config.visionConfiguration.patchesPerTile,
        )
        let logits = languageModel(
            input.text.tokens,
            cache: cache,
            crossAttentionStates: crossStates,
            crossAttentionMask: crossMask,
            fullTextRowMaskedOutMask: fullRowMask,
        )
        var state = LMOutput.State()
        if let crossStates {
            state[mllamaCrossAttentionStatesKey] = crossStates
        }
        defer { prefill.progress?(input.text.tokens.size, input.text.tokens.size) }
        return .logits(LMOutput(logits: logits, state: state))
    }

    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let crossStates = state?[mllamaCrossAttentionStatesKey]
        let logits = languageModel(
            input.tokens,
            cache: cache,
            crossAttentionStates: crossStates,
            crossAttentionMask: nil,
            fullTextRowMaskedOutMask: nil,
        )
        var outputState = LMOutput.State()
        if let crossStates {
            outputState[mllamaCrossAttentionStatesKey] = crossStates
        }
        return LMOutput(logits: logits, state: outputState)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(
            inputs,
            cache: cache,
            crossAttentionStates: nil,
            crossAttentionMask: nil,
            fullTextRowMaskedOutMask: nil,
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.contains("self_attn.rotary_emb.inv_freq") {
                continue
            }
            var newKey = key
            if !newKey.contains("vision_tower") {
                newKey = newKey.replacingOccurrences(of: "vision_model", with: "vision_tower")
            }
            sanitized[newKey] = value
        }
        return visionTower.sanitize(weights: sanitized)
    }
}
