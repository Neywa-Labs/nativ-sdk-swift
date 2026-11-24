import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct LLMjpVLConfiguration: Decodable, Sendable {
    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let rmsNormEps: Float
        public let vocabularySize: Int
        public let ropeTheta: Float
        public let ropeTraditional: Bool
        public let tieWordEmbeddings: Bool
        let ropeScaling: [String: StringOrNumber]?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case ropeTheta = "rope_theta"
            case ropeTraditional = "rope_traditional"
            case tieWordEmbeddings = "tie_word_embeddings"
            case ropeScaling = "rope_scaling"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "llama"
            ropeScaling = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14336
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
            headDim =
                try c.decodeIfPresent(Int.self, forKey: .headDim) ?? (hiddenSize / attentionHeads)
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 196_608
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000
            ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
            tieWordEmbeddings =
                try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let intermediateSize: Int
        public let patchSize: Int
        public let imageSize: Int
        public let channels: Int
        public let layerNormEps: Float

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case intermediateSize = "intermediate_size"
            case patchSize = "patch_size"
            case imageSize = "image_size"
            case channels = "num_channels"
            case layerNormEps = "layer_norm_eps"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType) ?? "siglip_vision_model"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 27
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 512
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    public let imageTokenIndex: Int
    public let downsampleRatio: Float
    public let selectLayer: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case languageConfiguration = "llm_config"
        case visionConfiguration = "vision_config"
        case imageContextTokenID = "img_context_token_id"
        case imageTokenIndex = "image_token_index"
        case downsampleRatio = "downsample_ratio"
        case selectLayer = "select_layer"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "llmjpvl"

        // The checkpoints nest the decoder under `llm_config`; mlx-vlm normalises that to
        // `text_config` globally in utils.py, which has no counterpart here.
        if let text = try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration) {
            textConfiguration = text
        } else {
            textConfiguration = try c.decode(
                TextConfiguration.self, forKey: .languageConfiguration)
        }
        visionConfiguration = try c.decode(
            VisionConfiguration.self, forKey: .visionConfiguration)

        imageTokenIndex =
            try c.decodeIfPresent(Int.self, forKey: .imageContextTokenID)
            ?? c.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? 14
        downsampleRatio = try c.decodeIfPresent(Float.self, forKey: .downsampleRatio) ?? 0.5
        selectLayer = try c.decodeIfPresent(Int.self, forKey: .selectLayer) ?? -1
        let factor = 1 / downsampleRatio
        guard factor.isFinite, factor >= 1, factor <= Float(visionConfiguration.imageSize),
            factor.rounded() == factor, visionConfiguration.patchSize > 0,
            visionConfiguration.imageSize > 0,
            visionConfiguration.imageSize.isMultiple(
                of: visionConfiguration.patchSize * Int(factor))
        else {
            throw DecodingError.dataCorruptedError(
                forKey: .downsampleRatio, in: c,
                debugDescription:
                    "Image patches must form a grid divisible by the pixel-shuffle factor.")
        }
    }
}

// MARK: - Processor

public struct LLMjpVLProcessorConfiguration: Decodable, Sendable {
    public struct Size: Decodable, Sendable {
        public let height: Int?
        public let width: Int?
        public let shortestEdge: Int?

        public var value: Int { height ?? shortestEdge ?? 512 }

        enum CodingKeys: String, CodingKey {
            case height
            case width
            case shortestEdge = "shortest_edge"
        }
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let size: Size

    private let _minDynamicPatch: Int?
    public var minDynamicPatch: Int { _minDynamicPatch ?? 1 }

    private let _maxDynamicPatch: Int?
    public var maxDynamicPatch: Int { _maxDynamicPatch ?? 12 }

    private let _useThumbnail: Bool?
    public var useThumbnail: Bool { _useThumbnail ?? true }

    private let _imageSeqLength: Int?
    public var imageSeqLength: Int { _imageSeqLength ?? 256 }

    /// Bounds the tile count so a long prompt cannot push the image block past the
    /// context window, as the reference processor does with `tokenizer.model_max_length`.
    private let _modelMaxLength: Int?
    public var modelMaxLength: Int { _modelMaxLength ?? 4096 }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case size
        case _minDynamicPatch = "min_dynamic_patch"
        case _maxDynamicPatch = "max_dynamic_patch"
        case _useThumbnail = "use_thumbnail"
        case _imageSeqLength = "image_seq_length"
        case _modelMaxLength = "model_max_length"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        imageMean = try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd = try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        size = try c.decodeIfPresent(Size.self, forKey: .size) ?? Size(512)
        _minDynamicPatch = try c.decodeIfPresent(Int.self, forKey: ._minDynamicPatch)
        _maxDynamicPatch = try c.decodeIfPresent(Int.self, forKey: ._maxDynamicPatch)
        _useThumbnail = try c.decodeIfPresent(Bool.self, forKey: ._useThumbnail)
        _imageSeqLength = try c.decodeIfPresent(Int.self, forKey: ._imageSeqLength)
        _modelMaxLength = try c.decodeIfPresent(Int.self, forKey: ._modelMaxLength)
    }
}

extension LLMjpVLProcessorConfiguration.Size {
    fileprivate init(_ square: Int) {
        self.height = square
        self.width = square
        self.shortestEdge = nil
    }
}

public struct LLMjpVLProcessor: UserInputProcessor {
    private let config: LLMjpVLProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imagePadTokenID: Int
    private let imageStartTokenID: Int
    private let imageEndTokenID: Int

    private static let defaultSystemPrompt =
        "You are LLM-jp-VL, a Multimodal LLM trained by LLM-jp."

    public init(_ config: LLMjpVLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        imagePadTokenID = tokenizer.convertTokenToId("<|image_pad|>") ?? 14
        imageStartTokenID = tokenizer.convertTokenToId("<|image_start|>") ?? 15
        imageEndTokenID = tokenizer.convertTokenToId("<|image_end|>") ?? 16
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard config.size.value > 0, config.imageSeqLength > 0,
            config.minDynamicPatch > 0, config.maxDynamicPatch >= config.minDynamicPatch,
            config.imageMean.count == 3, config.imageStd.count == 3,
            config.imageStd.allSatisfy({ $0 > 0 })
        else {
            throw VLMError.processing("Invalid LLMjpVL image geometry or normalization settings.")
        }
        let images = Self.images(from: input)
        guard input.videos.isEmpty, input.audios.isEmpty else {
            throw VLMError.processing("LLMjpVL accepts images and text only.")
        }
        var messages = DefaultMessageGenerator().generate(from: input)
        switch input.prompt {
        case .chat(let chat):
            for (index, message) in chat.enumerated() where !message.images.isEmpty {
                messages[index]["content"] =
                    String(repeating: "<|image_pad|>", count: message.images.count)
                    + message.content
            }
        case .text(let text):
            messages = [
                [
                    "role": "user",
                    "content": String(repeating: "<|image_pad|>", count: images.count) + text,
                ]
            ]
        case .messages:
            // Raw messages carry their own image markers.
            break
        }
        if !messages.contains(where: { $0["role"] as? String == "system" }) {
            messages.insert(["role": "system", "content": Self.defaultSystemPrompt], at: 0)
        }

        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext,
        )

        guard promptTokens.filter({ $0 == imagePadTokenID }).count == images.count else {
            throw VLMError.processing("The prompt must contain one image marker per image.")
        }
        guard !images.isEmpty else {
            let tokensArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            return LMInput(text: .init(tokens: tokensArray, mask: ones(like: tokensArray)))
        }

        let textTokens = max(promptTokens.count - images.count, 0)
        let maxTiles = Self.tileBudget(
            imageCount: images.count,
            textTokens: textTokens,
            imageSeqLength: config.imageSeqLength,
            modelMaxLength: config.modelMaxLength,
            maxDynamicPatch: config.maxDynamicPatch,
        )

        var tileArrays: [MLXArray] = []
        var tileCounts: [Int] = []
        for image in images {
            let tiles = try self.tiles(
                from: image.asCIImage(), maxTiles: maxTiles, processing: input.processing)
            tileCounts.append(tiles.count)
            tileArrays.append(contentsOf: tiles)
        }

        let expanded = expandImageMarkers(in: promptTokens, tileCounts: tileCounts)
        let tokensArray = MLXArray(expanded).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: tokensArray, mask: ones(like: tokensArray).asType(.int8)),
            image: .init(pixels: concatenated(tileArrays, axis: 0)),
        )
    }

    /// Mirrors the reference processor: each image costs `(tiles + 1) * seq + 2` tokens.
    static func tileBudget(
        imageCount: Int, textTokens: Int, imageSeqLength: Int, modelMaxLength: Int,
        maxDynamicPatch: Int,
    )
        -> Int
    {
        guard imageCount > 0 else { return maxDynamicPatch }
        let budget = modelMaxLength - textTokens
        let perImage = (budget / imageCount - 2) / imageSeqLength - 1
        return max(1, min(maxDynamicPatch, perImage))
    }

    /// The (columns, rows) tiling whose aspect ratio is closest to the image's.
    static func closestAspectRatio(
        width: Int, height: Int, minTiles: Int, maxTiles: Int, tileSize: Int,
    )
        -> (columns: Int, rows: Int)
    {
        var candidates: [(Int, Int)] = []
        for total in minTiles ... maxTiles {
            for columns in 1 ... total {
                for rows in 1 ... total
                where columns * rows <= maxTiles && columns * rows >= minTiles {
                    if !candidates.contains(where: { $0 == (columns, rows) }) {
                        candidates.append((columns, rows))
                    }
                }
            }
        }
        candidates.sort { $0.0 * $0.1 < $1.0 * $1.1 }

        let aspect = Double(width) / Double(height)
        let area = width * height
        var best = (1, 1)
        var bestDifference = Double.infinity
        for candidate in candidates {
            let candidateAspect = Double(candidate.0) / Double(candidate.1)
            let difference = abs(aspect - candidateAspect)
            if difference < bestDifference {
                bestDifference = difference
                best = candidate
            } else if difference == bestDifference {
                // Upstream prefers the larger tiling when the image has the area to fill it.
                if Double(area) > 0.5 * Double(tileSize * tileSize * candidate.0 * candidate.1) {
                    best = candidate
                }
            }
        }
        return best
    }

    private func tiles(from image: CIImage, maxTiles: Int, processing: UserInput.Processing?) throws
        -> [MLXArray]
    {
        let canonical = MediaProcessing.inSRGBToneCurveSpace(
            MediaProcessing.apply(image, processing: processing))
        let width = max(Int(canonical.extent.width.rounded()), 1)
        let height = max(Int(canonical.extent.height.rounded()), 1)
        let tileSize = config.size.value

        let ratio = Self.closestAspectRatio(
            width: width,
            height: height,
            minTiles: min(config.minDynamicPatch, maxTiles),
            maxTiles: maxTiles,
            tileSize: tileSize,
        )
        let targetWidth = tileSize * ratio.columns
        let targetHeight = tileSize * ratio.rows

        let resized = MediaProcessing.resampleBicubic(
            canonical, to: CGSize(width: targetWidth, height: targetHeight), clampEdges: true)

        var crops: [CIImage] = []
        for index in 0 ..< (ratio.columns * ratio.rows) {
            let column = index % (targetWidth / tileSize)
            let row = index / (targetWidth / tileSize)
            crops.append(
                Self.crop(
                    resized,
                    x: column * tileSize,
                    y: row * tileSize,
                    side: tileSize,
                    canvasHeight: targetHeight,
                ),
            )
        }
        if config.useThumbnail, crops.count != 1 {
            crops.append(
                MediaProcessing.resampleBicubic(
                    canonical, to: CGSize(width: tileSize, height: tileSize), clampEdges: true))
        }

        return crops.map { tile in
            MediaProcessing.asMLXArray(
                MediaProcessing.normalize(
                    tile,
                    mean: (config.imageMean[0], config.imageMean[1], config.imageMean[2]),
                    std: (config.imageStd[0], config.imageStd[1], config.imageStd[2]),
                ),
            )
        }
    }

    /// PIL crops from the top-left; CoreImage's origin is bottom-left, so the row index is
    /// flipped before cropping.
    private static func crop(
        _ image: CIImage, x: Int, y: Int, side: Int, canvasHeight: Int,
    )
        -> CIImage
    {
        let flippedY = canvasHeight - y - side
        return
            image
            .cropped(to: CGRect(x: x, y: flippedY, width: side, height: side))
            .transformed(by: CGAffineTransform(translationX: -CGFloat(x), y: -CGFloat(flippedY)))
    }

    private func expandImageMarkers(in promptTokens: [Int], tileCounts: [Int]) -> [Int] {
        guard !tileCounts.isEmpty else { return promptTokens }

        var tokens: [Int] = []
        var consumed = 0
        for token in promptTokens {
            guard token == imagePadTokenID, consumed < tileCounts.count else {
                tokens.append(token)
                continue
            }
            tokens.append(imageStartTokenID)
            tokens.append(
                contentsOf: Array(
                    repeating: imagePadTokenID,
                    count: config.imageSeqLength * tileCounts[consumed]))
            tokens.append(imageEndTokenID)
            consumed += 1
        }
        return tokens
    }

    private static func images(from input: UserInput) -> [UserInput.Image] {
        if !input.images.isEmpty {
            return input.images
        }
        guard case .chat(let messages) = input.prompt else {
            return []
        }
        return messages.flatMap(\.images)
    }
}

// MARK: - Vision

private enum LLMjpVLVision {
    final class GELUMLP: Module {
        let fc1: Linear
        let fc2: Linear
        init(hiddenSize: Int, intermediateSize: Int) {
            fc1 = Linear(hiddenSize, intermediateSize)
            fc2 = Linear(intermediateSize, hiddenSize)
        }
        func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
    }

    fileprivate final class Attention: Module {
        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "out_proj") var wo: Linear

        private let numHeads: Int
        private let scale: Float

        init(dimensions: Int, numHeads: Int) {
            precondition(dimensions % numHeads == 0)
            self.numHeads = numHeads
            scale = pow(Float(dimensions / numHeads), -0.5)
            _wq.wrappedValue = Linear(dimensions, dimensions, bias: true)
            _wk.wrappedValue = Linear(dimensions, dimensions, bias: true)
            _wv.wrappedValue = Linear(dimensions, dimensions, bias: true)
            _wo.wrappedValue = Linear(dimensions, dimensions, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (batch, length) = (x.dim(0), x.dim(1))
            let queries = wq(x).reshaped(batch, length, numHeads, -1).transposed(0, 2, 1, 3)
            let keys = wk(x).reshaped(batch, length, numHeads, -1).transposed(0, 2, 1, 3)
            let values = wv(x).reshaped(batch, length, numHeads, -1).transposed(0, 2, 1, 3)
            return wo(
                MLXFast.scaledDotProductAttention(
                    queries: queries, keys: keys, values: values, scale: scale, mask: .none,
                )
                .transposed(0, 2, 1, 3)
                .reshaped(batch, length, -1),
            )
        }
    }

    fileprivate final class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
        @ModuleInfo var mlp: GELUMLP

        init(_ config: LLMjpVLConfiguration.VisionConfiguration) {
            _attention.wrappedValue = Attention(
                dimensions: config.hiddenSize, numHeads: config.attentionHeads)
            _layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            _layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            _mlp.wrappedValue = GELUMLP(
                hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let attended = x + attention(layerNorm1(x))
            return attended + mlp(layerNorm2(attended))
        }
    }

    fileprivate final class Encoder: Module {
        let layers: [EncoderLayer]

        init(_ config: LLMjpVLConfiguration.VisionConfiguration) {
            layers = (0 ..< config.hiddenLayers).map { _ in EncoderLayer(config) }
        }
    }

    fileprivate final class Embeddings: Module {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        private let numPositions: Int

        init(_ config: LLMjpVLConfiguration.VisionConfiguration) {
            _patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.channels,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
            )
            let side = config.imageSize / config.patchSize
            numPositions = side * side
            _positionEmbedding.wrappedValue = Embedding(
                embeddingCount: numPositions, dimensions: config.hiddenSize)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let patches = patchEmbedding(x).flattened(start: 1, end: 2)
            let positionIDs = MLXArray(0 ..< numPositions).expandedDimensions(axis: 0)
            return patches + positionEmbedding(positionIDs)
        }
    }

    fileprivate final class SigLipVisionModel: Module {
        @ModuleInfo var embeddings: Embeddings
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        init(_ config: LLMjpVLConfiguration.VisionConfiguration) {
            _embeddings.wrappedValue = Embeddings(config)
            _encoder.wrappedValue = Encoder(config)
            // Upstream constructs this one without an eps, so it keeps MLX's 1e-5 default
            // rather than the 1e-6 the encoder norms use.
            _postLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var hidden = embeddings(x)
            for layer in encoder.layers {
                hidden = layer(hidden)
            }
            return postLayerNorm(hidden)
        }
    }

    fileprivate final class VisionModel: Module {
        @ModuleInfo(key: "vision_model") var visionModel: SigLipVisionModel

        init(_ config: LLMjpVLConfiguration.VisionConfiguration) {
            _visionModel.wrappedValue = SigLipVisionModel(config)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            visionModel(x)
        }
    }
}

// MARK: - Language

private enum LLMjpVLLanguage {
    fileprivate final class Attention: Module {
        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear
        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE
        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

        private let heads: Int
        private let kvHeads: Int
        private let headDim: Int
        private let scale: Float

        init(_ config: LLMjpVLConfiguration.TextConfiguration) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)

            _wq.wrappedValue = Linear(
                config.hiddenSize, heads * headDim, bias: config.modelType == "qwen2")
            _wk.wrappedValue = Linear(
                config.hiddenSize, kvHeads * headDim, bias: config.modelType == "qwen2")
            _wv.wrappedValue = Linear(
                config.hiddenSize, kvHeads * headDim, bias: config.modelType == "qwen2")
            _wo.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: false)
            if config.modelType == "qwen3" {
                _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
                _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            }
            let scale: Float =
                config.ropeScaling?["type"] == .string("linear")
                ? 1 / (config.ropeScaling?["factor"]?.asFloat() ?? 1) : 1
            _rotaryEmbedding.wrappedValue = RoPE(
                dimensions: headDim,
                traditional: config.ropeTraditional,
                base: config.ropeTheta, scale: scale
            )
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        )
            -> MLXArray
        {
            let (batch, length) = (x.dim(0), x.dim(1))
            var queries = wq(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            var keys = wk(x).reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
            let values = wv(x).reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

            queries = qNorm?(queries) ?? queries
            keys = kNorm?(keys) ?? keys
            let offset = cache?.offset ?? 0
            queries = rotaryEmbedding(queries, offset: offset)
            keys = rotaryEmbedding(keys, offset: offset)

            let output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values, cache: cache, scale: scale,
                mask: mask,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)
            return wo(output)
        }
    }

    fileprivate final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: SwiGLUMLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: LLMjpVLConfiguration.TextConfiguration) {
            _attention.wrappedValue = Attention(config)
            mlp = SwiGLUMLP(
                inputDimensions: config.hiddenSize,
                hiddenDimensions: config.intermediateSize)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        )
            -> MLXArray
        {
            let attended = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
            return attended + mlp(postAttentionLayerNorm(attended))
        }
    }

    fileprivate final class Llama: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        fileprivate let layers: [DecoderLayer]
        fileprivate let norm: RMSNorm

        init(_ config: LLMjpVLConfiguration.TextConfiguration) {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            layers = (0 ..< config.hiddenLayers).map { _ in DecoderLayer(config) }
            norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ inputs: MLXArray?, cache: [KVCache]? = nil, inputEmbedding: MLXArray? = nil,
        )
            -> MLXArray
        {
            var hidden: MLXArray
            if let inputEmbedding {
                hidden = inputEmbedding
            } else if let inputs {
                hidden = embedTokens(inputs)
            } else {
                fatalError("one of inputs or inputEmbedding must be non-nil")
            }

            let mask = createAttentionMask(h: hidden, cache: cache?.first)
            for (index, layer) in layers.enumerated() {
                hidden = layer(hidden, mask: mask, cache: cache?[index])
            }
            return norm(hidden)
        }
    }

    fileprivate final class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: Llama
        @ModuleInfo(key: "lm_head") var lmHead: Linear?
        let kvHeads: [Int]

        init(_ config: LLMjpVLConfiguration.TextConfiguration) {
            model = Llama(config)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.hiddenSize, config.vocabularySize, bias: false)
            }
            kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            state: LMOutput.State? = nil,
            inputEmbedding: MLXArray? = nil,
        )
            -> LMOutput
        {
            let hidden = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
            let logits = lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
            return LMOutput(logits: logits, state: state)
        }
    }
}

// MARK: - Model

public final class LLMjpVL: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_backbone") private var visionBackbone: LLMjpVLVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: LLMjpVLLanguage.LanguageModel
    // nn.Sequential upstream: LayerNorm, Linear, GELU, Linear, so the weights are named
    // mlp1.0, mlp1.1 and mlp1.3.
    @ModuleInfo var mlp1: [UnaryLayer]

    public let config: LLMjpVLConfiguration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.model.layers }

    public init(_ config: LLMjpVLConfiguration) {
        self.config = config
        _visionBackbone.wrappedValue = LLMjpVLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = LLMjpVLLanguage.LanguageModel(config.textConfiguration)

        let factor = Int((1 / config.downsampleRatio).rounded())
        let widened = config.visionConfiguration.hiddenSize * factor * factor
        _mlp1.wrappedValue = [
            LayerNorm(dimensions: widened),
            Linear(widened, config.textConfiguration.hiddenSize),
            SequentialActivation(GELU()),
            Linear(config.textConfiguration.hiddenSize, config.textConfiguration.hiddenSize),
        ]
    }

    private func pixelShuffle(_ input: MLXArray) -> MLXArray {
        let (batch, patches, channels) = input.shape3
        let side = Int(Double(patches).squareRoot())
        let factor = Int((1 / config.downsampleRatio).rounded())
        return input.reshaped(batch, side, side / factor, channels * factor)
            .transposed(0, 2, 1, 3)
            .reshaped(batch, side / factor, side / factor, channels * factor * factor)
            .transposed(0, 2, 1, 3).reshaped(batch, -1, channels * factor * factor)
    }

    func inputEmbeddings(_ input: LMInput) throws -> MLXArray {
        let inputIDs =
            input.text.tokens.ndim == 1
            ? input.text.tokens.expandedDimensions(axis: 0) : input.text.tokens
        guard inputIDs.ndim == 2, inputIDs.dim(0) == 1 else {
            throw VLMError.processing("Image input requires a single sequence.")
        }
        let embeddings = languageModel.model.embedTokens(inputIDs)
        guard var pixels = input.image?.pixels else {
            return embeddings
        }
        if pixels.ndim == 5 {
            pixels = pixels[0]
        }

        // SigLIP has no class token, so nothing is stripped from the front.
        var features = visionBackbone(
            pixels.transposed(0, 2, 3, 1).asType(embeddings.dtype))
        features = pixelShuffle(features)
        for layer in mlp1 {
            features = layer(features)
        }
        features = features.reshaped(-1, features.dim(features.ndim - 1)).asType(
            embeddings.dtype)

        let mask = inputIDs .== config.imageTokenIndex
        let placeholders = mask.sum().item(Int.self)
        guard placeholders == features.dim(0), placeholders > 0 else {
            throw VLMError.processing(
                "Image token count (\(placeholders)) does not match vision features (\(features.dim(0)))."
            )
        }

        let positions = (cumsum(mask.reshaped(-1).asType(.int32)) - 1).reshaped(mask.shape)
        let gathered = features[clip(positions, min: 0, max: features.dim(0) - 1)]
        return which(mask[.ellipsis, .newAxis], gathered, embeddings)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache],
        state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        guard config.selectLayer == -1 else {
            throw VLMError.processing(
                "llmjpvl supports select_layer -1 only, got \(config.selectLayer)")
        }
        let embeddings = try inputEmbeddings(input)
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
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.contains("rotary_emb.inv_freq") {
                continue
            }
            // The SigLIP attention-pooling head is unused when only the last hidden state
            // is consumed.
            if key.contains(".vision_model.head.") {
                continue
            }
            if key.contains("patch_embedding.weight"), value.ndim == 4,
                value.dim(1) == config.visionConfiguration.channels
            {
                sanitized[key] = value.transposed(0, 2, 3, 1)
            } else {
                sanitized[key] = value
            }
        }
        return sanitized
    }
}
