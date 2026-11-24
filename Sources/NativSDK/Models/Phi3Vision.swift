import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct Phi3VisionConfiguration: Decodable, Sendable {
    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        private let _hiddenSize: Int?
        public var hiddenSize: Int {
            _hiddenSize ?? 1024
        }

        private let _hiddenLayers: Int?
        public var hiddenLayers: Int {
            _hiddenLayers ?? 24
        }

        public let intermediateSize: Int
        private let _attentionHeads: Int?
        public var attentionHeads: Int {
            _attentionHeads ?? 16
        }

        private let _imageSize: Int?
        public var imageSize: Int {
            _imageSize ?? 336
        }

        private let _patchSize: Int?
        public var patchSize: Int {
            _patchSize ?? 14
        }

        private let _channels: Int?
        public var channels: Int {
            _channels ?? 3
        }

        private let _layerNormEps: Float?
        public var layerNormEps: Float {
            _layerNormEps ?? 1e-5
        }

        private let _imageDimOut: Int?
        public var imageDimOut: Int {
            _imageDimOut ?? hiddenSize
        }

        private let _numImageTokens: Int?
        public var numImageTokens: Int {
            _numImageTokens ?? 144
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case name
            case _hiddenSize = "hidden_size"
            case _hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case _attentionHeads = "num_attention_heads"
            case _imageSize = "image_size"
            case _patchSize = "patch_size"
            case _channels = "num_channels"
            case _layerNormEps = "layer_norm_eps"
            case _imageDimOut = "image_dim_out"
            case _numImageTokens = "num_img_tokens"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType)
                ?? c.decodeIfPresent(String.self, forKey: .name)
                ?? "clip_vision_model"
            _hiddenSize = try c.decodeIfPresent(Int.self, forKey: ._hiddenSize)
            _hiddenLayers = try c.decodeIfPresent(Int.self, forKey: ._hiddenLayers)
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4096
            _attentionHeads = try c.decodeIfPresent(Int.self, forKey: ._attentionHeads)
            _imageSize = try c.decodeIfPresent(Int.self, forKey: ._imageSize)
            _patchSize = try c.decodeIfPresent(Int.self, forKey: ._patchSize)
            _channels = try c.decodeIfPresent(Int.self, forKey: ._channels)
            _layerNormEps = try c.decodeIfPresent(Float.self, forKey: ._layerNormEps)
            _imageDimOut = try c.decodeIfPresent(Int.self, forKey: ._imageDimOut)
            _numImageTokens = try c.decodeIfPresent(Int.self, forKey: ._numImageTokens)
        }
    }

    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    private let _kvHeads: Int?
    public var kvHeads: Int {
        _kvHeads ?? attentionHeads
    }

    public let rmsNormEps: Float
    public let vocabularySize: Int
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int {
        _imageTokenIndex ?? 257_152
    }

    public let ropeTheta: Float
    private let _ropeTraditional: Bool?
    public var ropeTraditional: Bool {
        _ropeTraditional ?? false
    }

    public let ropeScaling: [String: StringOrNumber]?
    private let _partialRotaryFactor: Float?
    public var partialRotaryFactor: Float {
        _partialRotaryFactor ?? 1.0
    }

    public let maxPositionEmbeddings: Int
    public let originalMaxPositionEmbeddings: Int
    private let _tieWordEmbeddings: Bool?
    public var tieWordEmbeddings: Bool {
        _tieWordEmbeddings ?? false
    }

    public let visionConfiguration: VisionConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case _kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case _imageTokenIndex = "image_token_index"
        case ropeTheta = "rope_theta"
        case _ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case _partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case originalMaxPositionEmbeddings = "original_max_position_embeddings"
        case _tieWordEmbeddings = "tie_word_embeddings"
        case visionConfiguration = "vision_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "phi3_v"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 3072
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 8192
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        _kvHeads = try c.decodeIfPresent(Int.self, forKey: ._kvHeads)
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 32064
        _imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: ._imageTokenIndex)
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        _ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: ._ropeTraditional)
        ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        _partialRotaryFactor = try c.decodeIfPresent(Float.self, forKey: ._partialRotaryFactor)
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        originalMaxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .originalMaxPositionEmbeddings) ?? 4096
        _tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: ._tieWordEmbeddings)
        visionConfiguration =
            try c.decodeIfPresent(VisionConfiguration.self, forKey: .visionConfiguration)
            ?? JSONDecoder.json5().decode(VisionConfiguration.self, from: Data("{}".utf8))
    }
}

// MARK: - Processor

public struct Phi3VisionProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String?
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    private let _numCrops: Int?
    public var numCrops: Int {
        _numCrops ?? 4
    }

    private let _numImageTokens: Int?
    public var numImageTokens: Int {
        _numImageTokens ?? 144
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case _numCrops = "num_crops"
        case _numImageTokens = "num_img_tokens"
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
        _numCrops = try c.decodeIfPresent(Int.self, forKey: ._numCrops)
        _numImageTokens = try c.decodeIfPresent(Int.self, forKey: ._numImageTokens)
    }
}

public struct Phi3VisionProcessor: UserInputProcessor {
    private let config: Phi3VisionProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Phi3VisionProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        guard !images.isEmpty else {
            let prompt = renderPrompt(from: input, imageCount: 0)
            let tokens = tokenizer.encode(text: prompt, addSpecialTokens: true)
            let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
            return LMInput(text: .init(tokens: tokenArray, mask: ones(like: tokenArray)))
        }

        var processedImages: [ProcessedImage] = []
        for image in images {
            try processedImages.append(
                preprocess(image: image.asCIImage(), processing: input.processing),
            )
        }

        let prompt = renderPrompt(from: input, imageCount: processedImages.count)
        let tokenCounts = processedImages.map(\.imageTokenCount)
        let inputIDs = tokenizePrompt(prompt, imageTokenCounts: tokenCounts)
        let tokenArray = MLXArray(inputIDs).expandedDimensions(axis: 0)

        let maxTiles = config.numCrops + 1
        let pixels = processedImages.flatMap { image -> [Float] in
            var result = image.pixels
            let missingTiles = max(maxTiles - image.tileCount, 0)
            if missingTiles > 0 {
                result.append(
                    contentsOf: Array(
                        repeating: Float(0),
                        count: missingTiles * 3 * 336 * 336,
                    ),
                )
            }
            return result
        }
        let imageSizes = processedImages.flatMap {
            [Int32($0.transformedHeight), Int32($0.transformedWidth)]
        }

        return LMInput(
            text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)),
            image: .init(
                pixels: MLXArray(
                    pixels,
                    [processedImages.count, maxTiles, 3, 336, 336],
                ),
            ),
            video: .init(pixels: MLXArray(imageSizes, [processedImages.count, 2])),
        )
    }

    private struct ProcessedImage {
        let pixels: [Float]
        let imageTokenCount: Int
        let transformedHeight: Int
        let transformedWidth: Int
        let tileCount: Int
    }

    private func renderPrompt(from input: UserInput, imageCount: Int) -> String {
        switch input.prompt {
        case .chat(let messages):
            var nextImage = 1
            var rendered = ""
            for message in messages {
                rendered += "<|\(message.role.rawValue)|>\n"
                for _ in message.images {
                    rendered += "<|image_\(nextImage)|>\n"
                    nextImage += 1
                }
                rendered += message.content
                rendered += "<|end|>\n"
            }
            if messages.last?.role != .assistant {
                rendered += "<|assistant|>\n"
            }
            return rendered

        case .text(let text):
            let imageTags = (1 ... max(imageCount, 1)).prefix(imageCount)
                .map { "<|image_\($0)|>" }
                .joined(separator: "\n")
            let prefix = imageTags.isEmpty ? "" : "\(imageTags)\n"
            return "<|user|>\n\(prefix)\(text)<|end|>\n<|assistant|>\n"

        case .messages:
            let text = input.prompt.description
            let imageTags = (1 ... max(imageCount, 1)).prefix(imageCount)
                .map { "<|image_\($0)|>" }
                .joined(separator: "\n")
            let prefix = imageTags.isEmpty ? "" : "\(imageTags)\n"
            return "<|user|>\n\(prefix)\(text)<|end|>\n<|assistant|>\n"
        }
    }

    private func tokenizePrompt(_ prompt: String, imageTokenCounts: [Int]) -> [Int] {
        let regex = try! NSRegularExpression(pattern: #"<\|image_(\d+)\|>"#)
        let nsRange = NSRange(prompt.startIndex ..< prompt.endIndex, in: prompt)
        let matches = regex.matches(in: prompt, range: nsRange)
        var result: [Int] = []
        var cursor = prompt.startIndex
        var isFirstTextChunk = true

        for match in matches {
            guard let tagRange = Range(match.range, in: prompt),
                let idRange = Range(match.range(at: 1), in: prompt),
                let imageID = Int(prompt[idRange])
            else {
                continue
            }
            let chunk = String(prompt[cursor ..< tagRange.lowerBound])
            result.append(
                contentsOf: tokenizer.encode(
                    text: chunk,
                    addSpecialTokens: isFirstTextChunk,
                ),
            )
            isFirstTextChunk = false
            let tokenCount =
                imageTokenCounts.indices.contains(imageID - 1)
                ? imageTokenCounts[imageID - 1]
                : 0
            result.append(contentsOf: Array(repeating: -imageID, count: tokenCount))
            cursor = tagRange.upperBound
        }

        let tail = String(prompt[cursor...])
        result.append(
            contentsOf: tokenizer.encode(
                text: tail,
                addSpecialTokens: isFirstTextChunk,
            ),
        )
        return result
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?) throws
        -> ProcessedImage
    {
        let canonical = MediaProcessing.zeroOrigin(
            MediaProcessing.apply(image, processing: processing))
        let originalHeight = max(Int(canonical.extent.height.rounded()), 1)
        let originalWidth = max(Int(canonical.extent.width.rounded()), 1)
        let hd = Self.hdTransform(
            image: canonical,
            originalWidth: originalWidth,
            originalHeight: originalHeight,
            numCrops: config.numCrops,
        )
        let heightTiles = hd.height / 336
        let widthTiles = hd.width / 336
        let imageTokenCount =
            (heightTiles * widthTiles + 1) * config.numImageTokens
            + 1 + (heightTiles + 1) * 12

        var images = [
            Self.resizeNormalize(
                image: canonical,
                targetWidth: 336,
                targetHeight: 336,
                mean: config.imageMean,
                std: config.imageStd,
            )
        ]
        for row in 0 ..< heightTiles {
            for col in 0 ..< widthTiles {
                let crop = MediaProcessing.cropPixels(
                    hd.pixels,
                    sourceWidth: hd.width,
                    x: col * 336,
                    y: row * 336,
                    width: 336,
                    height: 336,
                    channels: 3,
                )
                images.append(
                    MediaProcessing.normalizeCHW(
                        crop,
                        width: 336,
                        height: 336,
                        mean: config.imageMean,
                        std: config.imageStd,
                    ),
                )
            }
        }

        return ProcessedImage(
            pixels: images.flatMap(\.self),
            imageTokenCount: imageTokenCount,
            transformedHeight: hd.height,
            transformedWidth: hd.width,
            tileCount: images.count,
        )
    }

    private struct HDImage {
        let pixels: [Float]
        let width: Int
        let height: Int
    }

    private static func hdTransform(
        image: CIImage,
        originalWidth: Int,
        originalHeight: Int,
        numCrops: Int,
    )
        -> HDImage
    {
        let transposed = originalWidth < originalHeight
        let major = transposed ? originalHeight : originalWidth
        let minor = transposed ? originalWidth : originalHeight
        let ratio = Float(major) / Float(minor)
        var scale = 1
        while scale * Int(ceil(Float(scale) / ratio)) <= numCrops {
            scale += 1
        }
        scale = max(scale - 1, 1)

        let resizedMajor = scale * 336
        let resizedMinor = max(Int(Float(resizedMajor) / ratio), 1)
        let paddedMinor = Int(ceil(Float(resizedMinor) / 336)) * 336

        let contentWidth = transposed ? resizedMinor : resizedMajor
        let contentHeight = transposed ? resizedMajor : resizedMinor
        let targetWidth = transposed ? paddedMinor : resizedMajor
        let targetHeight = transposed ? resizedMajor : paddedMinor
        let xOffset = transposed ? (targetWidth - contentWidth) / 2 : 0
        let yOffset = transposed ? 0 : (targetHeight - contentHeight) / 2

        let resized = MediaProcessing.zeroOrigin(image).resampled(
            to: CGSize(width: contentWidth, height: contentHeight),
            method: .bicubic,
        )
        let raw = MediaProcessing.renderRGB8(resized, width: contentWidth, height: contentHeight)
        var padded = Array(repeating: Float(1), count: targetWidth * targetHeight * 3)
        for row in 0 ..< contentHeight {
            for col in 0 ..< contentWidth {
                let source = (row * contentWidth + col) * 3
                let target = ((yOffset + row) * targetWidth + xOffset + col) * 3
                padded[target] = raw[source]
                padded[target + 1] = raw[source + 1]
                padded[target + 2] = raw[source + 2]
            }
        }
        return HDImage(pixels: padded, width: targetWidth, height: targetHeight)
    }

    private static func resizeNormalize(
        image: CIImage,
        targetWidth: Int,
        targetHeight: Int,
        mean: [CGFloat],
        std: [CGFloat],
    )
        -> [Float]
    {
        let resized = MediaProcessing.zeroOrigin(image).resampled(
            to: CGSize(width: targetWidth, height: targetHeight),
            method: .bicubic,
        )
        let raw = MediaProcessing.renderRGB8(resized, width: targetWidth, height: targetHeight)
        return MediaProcessing.normalizeCHW(
            raw, width: targetWidth, height: targetHeight, mean: mean, std: std)
    }

}

// MARK: - Language

private enum Phi3VisionLanguage {
    final class Attention: Module {
        let config: Phi3VisionConfiguration
        let headDim: Int
        let scale: Float
        let rope: RoPELayer

        @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        init(_ config: Phi3VisionConfiguration) {
            self.config = config
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _qkvProj.wrappedValue = Linear(
                config.hiddenSize,
                (config.attentionHeads + 2 * config.kvHeads) * headDim,
                bias: false,
            )
            _oProj.wrappedValue = Linear(
                config.attentionHeads * headDim,
                config.hiddenSize,
                bias: false,
            )

            let ropeDim = Int(Float(headDim) * config.partialRotaryFactor)
            if let ropeScaling = config.ropeScaling,
                ropeScaling["type"] == .string("su")
                    || ropeScaling["type"] == .string("longrope"),
                let shortFactor = ropeScaling["short_factor"]?.asFloats(),
                let longFactor = ropeScaling["long_factor"]?.asFloats()
            {
                rope = SuScaledRoPE(
                    dimensions: ropeDim,
                    base: config.ropeTheta,
                    maxPositionEmbeddings: config.maxPositionEmbeddings,
                    originalMaxPositionEmbeddings: config.originalMaxPositionEmbeddings,
                    shortFactor: shortFactor,
                    longFactor: longFactor,
                )
            } else {
                let ropeScale: Float =
                    if let ropeScaling = config.ropeScaling,
                        ropeScaling["type"] == .string("linear"),
                        let factor = ropeScaling["factor"]?.asFloat()
                    {
                        1 / factor
                    } else {
                        1
                    }
                rope = RoPE(
                    dimensions: ropeDim,
                    traditional: config.ropeTraditional,
                    base: config.ropeTheta,
                    scale: ropeScale,
                )
            }
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let (batchSize, sequenceLength) = (x.dim(0), x.dim(1))
            let queryPosition = config.attentionHeads * headDim
            let qkv = split(
                qkvProj(x),
                indices: [queryPosition, queryPosition + config.kvHeads * headDim],
                axis: -1,
            )
            var queries = qkv[0]
            var keys = qkv[1]
            var values = qkv[2]

            queries = queries.reshaped(batchSize, sequenceLength, config.attentionHeads, -1)
                .transposed(0, 2, 1, 3)
            keys = keys.reshaped(batchSize, sequenceLength, config.kvHeads, -1)
                .transposed(0, 2, 1, 3)
            values = values.reshaped(batchSize, sequenceLength, config.kvHeads, -1)
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
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, sequenceLength, -1)
            return oProj(output)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_up_proj") var gateUp: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(_ config: Phi3VisionConfiguration) {
            _gateUp.wrappedValue = Linear(
                config.hiddenSize,
                2 * config.intermediateSize,
                bias: false,
            )
            _down.wrappedValue = Linear(
                config.intermediateSize,
                config.hiddenSize,
                bias: false,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let gateUp = split(gateUp(x), parts: 2, axis: -1)
            return down(silu(gateUp[0]) * gateUp[1])
        }
    }

    final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: Phi3VisionConfiguration) {
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
}

// MARK: - Vision

private enum Phi3VisionCLIP {
    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: Phi3VisionConfiguration.VisionConfiguration) {
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
            let queries = qProj(x).reshaped(batchSize, sequenceLength, heads, headDim)
                .transposed(0, 2, 1, 3)
            let keys = kProj(x).reshaped(batchSize, sequenceLength, heads, headDim)
                .transposed(0, 2, 1, 3)
            let values = vProj(x).reshaped(batchSize, sequenceLength, heads, headDim)
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

        init(_ config: Phi3VisionConfiguration.VisionConfiguration) {
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

        init(_ config: Phi3VisionConfiguration.VisionConfiguration) {
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

        init(_ config: Phi3VisionConfiguration.VisionConfiguration) {
            layers = (0 ..< config.hiddenLayers).map { _ in EncoderLayer(config) }
        }
    }

    final class Embeddings: Module {
        let config: Phi3VisionConfiguration.VisionConfiguration
        let numPositions: Int

        @ModuleInfo(key: "class_embedding") var classEmbedding: MLXArray
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        init(_ config: Phi3VisionConfiguration.VisionConfiguration) {
            self.config = config
            let patches = config.imageSize / config.patchSize
            numPositions = patches * patches + 1
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
            let cls = broadcast(
                classEmbedding.asType(embeddings.dtype)[.newAxis, .newAxis, 0...],
                to: [x.dim(0), 1, config.hiddenSize],
            )
            embeddings = concatenated([cls, embeddings], axis: 1)
            let positionIDs = MLXArray(0 ..< numPositions).expandedDimensions(axis: 0)
            return embeddings + positionEmbedding(positionIDs)
        }
    }

    final class VisionModel: Module {
        @ModuleInfo var embeddings: Embeddings
        @ModuleInfo(key: "pre_layrnorm") var preLayerNorm: LayerNorm
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        init(_ config: Phi3VisionConfiguration.VisionConfiguration) {
            _embeddings.wrappedValue = Embeddings(config)
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

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized: [String: MLXArray] = [:]
            for (key, value) in weights {
                if key.contains("position_ids") {
                    continue
                }
                if key.contains("patch_embedding.weight"), value.ndim == 4 {
                    let shape = value.shape
                    sanitized[key] = shape[1] == 3 ? value.transposed(0, 2, 3, 1) : value
                } else {
                    sanitized[key] = value
                }
            }
            return sanitized
        }
    }

    final class Processor: Module {
        @ModuleInfo(key: "vision_model") var visionModel: VisionModel

        init(_ config: Phi3VisionConfiguration.VisionConfiguration) {
            _visionModel.wrappedValue = VisionModel(config)
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, [MLXArray]?,
        ) {
            visionModel(x, outputHiddenStates: outputHiddenStates)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            visionModel.sanitize(weights: weights)
        }
    }
}

private final class Phi3VisionEmbedTokens: Module {
    @ModuleInfo(key: "img_processor") var imageProcessor: Phi3VisionCLIP.Processor
    @ModuleInfo(key: "glb_GN") var globalSeparator: MLXArray
    @ModuleInfo(key: "sub_GN") var subSeparator: MLXArray
    @ModuleInfo(key: "img_projection") var imageProjection: [Module]

    let config: Phi3VisionConfiguration

    init(_ config: Phi3VisionConfiguration) {
        self.config = config
        let imageDim = config.visionConfiguration.imageDimOut * 4
        _imageProcessor.wrappedValue =
            Phi3VisionCLIP.Processor(config.visionConfiguration)
        _globalSeparator.wrappedValue = MLXArray.zeros([1, 1, imageDim])
        _subSeparator.wrappedValue = MLXArray.zeros([1, 1, 1, imageDim])
        _imageProjection.wrappedValue = [
            Linear(imageDim, config.hiddenSize),
            GELU(),
            Linear(config.hiddenSize, config.hiddenSize),
        ]
    }

    func encode(_ pixelValues: MLXArray, imageSizes: MLXArray?) -> [MLXArray] {
        let batchSize = pixelValues.dim(0)
        let tileCount = pixelValues.dim(1)
        let (_, _, hiddenStates) = imageProcessor(
            pixelValues.reshaped(-1, 3, 336, 336).transposed(0, 2, 3, 1),
            outputHiddenStates: true,
        )
        guard let hiddenStates else {
            return []
        }

        let features = hiddenStates[hiddenStates.count - 2][0..., 1..., 0...]
            .reshaped(batchSize, tileCount, -1, config.visionConfiguration.imageDimOut)
        let imageSizeValues = imageSizes?.asArray(Int32.self).map(Int.init) ?? []
        var packed: [MLXArray] = []
        for batchIndex in 0 ..< batchSize {
            let height =
                imageSizeValues.indices.contains(batchIndex * 2)
                ? imageSizeValues[batchIndex * 2]
                : 336
            let width =
                imageSizeValues.indices.contains(batchIndex * 2 + 1)
                ? imageSizeValues[batchIndex * 2 + 1]
                : 336
            packed.append(pack(features[batchIndex], height: height, width: width))
        }
        return packed
    }

    private func pack(_ features: MLXArray, height: Int, width: Int) -> MLXArray {
        let hTiles = max(height / 336, 1)
        let wTiles = max(width / 336, 1)
        let localTiles = hTiles * wTiles
        let side = 12
        let imageDim = config.visionConfiguration.imageDimOut
        let expandedDim = imageDim * 4

        func reshapeAndAppendRows(
            _ image: MLXArray,
            inputTiles: Int,
            outputRows: Int,
            outputCols: Int,
        )
            -> MLXArray
        {
            let reshaped =
                image
                .reshaped(inputTiles, side, 2, side, 2, imageDim)
                .transposed(0, 1, 3, 2, 4, 5)
                .reshaped(1, outputRows, outputCols, expandedDim)
            let separator = broadcast(
                subSeparator.asType(reshaped.dtype),
                to: [1, reshaped.dim(1), 1, expandedDim],
            )
            return concatenated([reshaped, separator], axis: 2)
                .reshaped(1, -1, expandedDim)
        }

        let globalImage = reshapeAndAppendRows(
            features[0],
            inputTiles: 1,
            outputRows: side,
            outputCols: side,
        )
        let localImages = reshapeAndAppendRows(
            features[1 ..< (localTiles + 1), 0..., 0...],
            inputTiles: localTiles,
            outputRows: hTiles * side,
            outputCols: wTiles * side,
        )
        let joined = concatenated(
            [localImages, globalSeparator.asType(localImages.dtype), globalImage],
            axis: 1,
        )
        return project(joined)[0]
    }

    private func project(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in imageProjection {
            h = (layer as! UnaryLayer)(h)
        }
        return h
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        imageProcessor.sanitize(weights: weights)
    }
}

private final class Phi3VisionModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [Phi3VisionLanguage.DecoderLayer]
    @ModuleInfo(key: "vision_embed_tokens") var visionEmbedTokens: Phi3VisionEmbedTokens
    @ModuleInfo var norm: RMSNorm

    let config: Phi3VisionConfiguration

    init(_ config: Phi3VisionConfiguration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize,
        )
        _layers.wrappedValue =
            (0 ..< config.hiddenLayers).map { _ in Phi3VisionLanguage.DecoderLayer(config) }
        _visionEmbedTokens.wrappedValue = Phi3VisionEmbedTokens(config)
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

// MARK: - Model

public final class Phi3Vision: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo fileprivate var model: Phi3VisionModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let config: Phi3VisionConfiguration
    public var kvHeads: [Int] {
        (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
    }

    public init(_ config: Phi3VisionConfiguration) {
        self.config = config
        _model.wrappedValue = Phi3VisionModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
    }

    public var loraLayers: [Module] {
        model.layers
    }

    private func inputEmbeddings(
        inputIDs: MLXArray,
        pixelValues: MLXArray?,
        imageSizes: MLXArray?,
    )
        -> MLXArray
    {
        let ids = inputIDs[0].asArray(Int.self)
        let safeInputIDs = MLXArray(ids.map { max($0, 0) }).expandedDimensions(axis: 0)
        let textEmbeds = model.embedTokens(safeInputIDs)
        guard let pixelValues else {
            return textEmbeds
        }

        let imageFeatures = model.visionEmbedTokens.encode(pixelValues, imageSizes: imageSizes)
        guard !imageFeatures.isEmpty else {
            return textEmbeds
        }

        var parts: [MLXArray] = []
        var index = 0
        while index < ids.count {
            if ids[index] < 0 {
                let imageIndex = -ids[index] - 1
                let runStart = index
                while index < ids.count, ids[index] == ids[runStart] {
                    index += 1
                }
                if imageFeatures.indices.contains(imageIndex) {
                    parts.append(imageFeatures[imageIndex].asType(textEmbeds.dtype))
                }
            } else {
                let start = index
                while index < ids.count, ids[index] >= 0 {
                    index += 1
                }
                parts.append(textEmbeds[0, start ..< index, 0...])
            }
        }

        return concatenated(parts, axis: 0).expandedDimensions(axis: 0)
    }

    private func logits(_ hidden: MLXArray) -> MLXArray {
        if config.tieWordEmbeddings {
            return model.embedTokens.asLinear(hidden)
        }
        guard let lmHead else {
            fatalError("Phi3Vision requires lm_head when embeddings are not tied")
        }
        return lmHead(hidden)
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
            _ = model(nil, cache: cache, inputEmbedding: embeddings[0..., range, 0...])
            asyncEval(cache)
        }
        eval(cache)

        let hidden = model(nil, cache: cache, inputEmbedding: embeddings[0..., processed..., 0...])
        defer { prefill.progress?(totalPositions, totalPositions) }
        return .logits(LMOutput(logits: logits(hidden)))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        logits(model(inputs, cache: cache))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let filtered = weights.filter { key, _ in
            !key.contains("self_attn.rotary_emb.inv_freq")
        }
        return model.visionEmbedTokens.sanitize(weights: filtered)
    }
}
