import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct InternVLConfiguration: Codable, Sendable {
    public struct TextConfiguration: Codable, Sendable {
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
        private let _headDim: Int?
        public var headDim: Int {
            _headDim ?? hiddenSize / attentionHeads
        }

        private let _ropeTheta: Float?
        public var ropeTheta: Float {
            _ropeTheta ?? 1_000_000
        }

        private let _ropeTraditional: Bool?
        public var ropeTraditional: Bool {
            _ropeTraditional ?? false
        }

        private let _tieWordEmbeddings: Bool?
        public var tieWordEmbeddings: Bool {
            _tieWordEmbeddings ?? false
        }

        private let _attentionBias: Bool?
        public var attentionBias: Bool {
            _attentionBias ?? true
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case _kvHeads = "num_key_value_heads"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case _headDim = "head_dim"
            case _ropeTheta = "rope_theta"
            case _ropeTraditional = "rope_traditional"
            case _tieWordEmbeddings = "tie_word_embeddings"
            case _attentionBias = "attention_bias"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let attentionHeads: Int
        public let patchSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let imageSize: Int
        private let _channels: Int?
        public var channels: Int {
            _channels ?? 3
        }

        private let _layerNormEps: Float?
        public var layerNormEps: Float {
            _layerNormEps ?? 1e-6
        }

        private let _qkvBias: Bool?
        public var qkvBias: Bool {
            _qkvBias ?? true
        }

        private let _qkNormalization: Bool?
        public var qkNormalization: Bool {
            _qkNormalization ?? false
        }

        private let _normType: String?
        public var normType: String {
            _normType ?? "layer_norm"
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case attentionHeads = "num_attention_heads"
            case patchSize = "patch_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case imageSize = "image_size"
            case _channels = "num_channels"
            case _layerNormEps = "layer_norm_eps"
            case _qkvBias = "qkv_bias"
            case _qkNormalization = "qk_normalization"
            case _normType = "norm_type"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int {
        _imageTokenIndex ?? 151_667
    }

    private let _videoTokenIndex: Int?
    public var videoTokenIndex: Int {
        _videoTokenIndex ?? 151_656
    }

    private let _downsampleRatio: Float?
    public var downsampleRatio: Float {
        _downsampleRatio ?? 0.5
    }

    private let _forceImageSize: Int?
    public var forceImageSize: Int {
        _forceImageSize ?? visionConfiguration.imageSize
    }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _imageTokenIndex = "image_token_index"
        case _videoTokenIndex = "video_token_index"
        case _downsampleRatio = "downsample_ratio"
        case _forceImageSize = "force_image_size"
    }
}

public struct InternVLProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String?
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    private let _size: Int?
    public var size: Int {
        _size ?? 448
    }

    private let _patchSize: Int?
    public var patchSize: Int {
        _patchSize ?? 14
    }

    private let _downsampleRatio: Float?
    public var downsampleRatio: Float {
        _downsampleRatio ?? 0.5
    }

    private let _numImageToken: Int?
    public var numImageToken: Int {
        _numImageToken ?? Int(pow(Float(size / patchSize), 2) * pow(downsampleRatio, 2))
    }

    private let _doDynamicPreprocess: Bool?
    public var doDynamicPreprocess: Bool {
        _doDynamicPreprocess ?? true
    }

    private let _dynamicMinNum: Int?
    public var dynamicMinNum: Int {
        _dynamicMinNum ?? 1
    }

    private let _dynamicMaxNum: Int?
    public var dynamicMaxNum: Int {
        _dynamicMaxNum ?? 12
    }

    private let _dynamicUseThumbnail: Bool?
    public var dynamicUseThumbnail: Bool {
        _dynamicUseThumbnail ?? true
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case _size = "size"
        case _imageSize = "image_size"
        case _patchSize = "patch_size"
        case _downsampleRatio = "downsample_ratio"
        case _numImageToken = "num_image_token"
        case _doDynamicPreprocess = "do_dynamic_preprocess"
        case _dynamicMinNum = "dynamic_min_num"
        case _dynamicMaxNum = "dynamic_max_num"
        case _dynamicUseThumbnail = "dynamic_use_thumbnail"
        case _minDynamicPatch = "min_dynamic_patch"
        case _maxDynamicPatch = "max_dynamic_patch"
        case _useThumbnail = "use_thumbnail"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass = try c.decodeIfPresent(String.self, forKey: .processorClass)
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageMean)
            ?? [0.485, 0.456, 0.406]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageStd)
            ?? [0.229, 0.224, 0.225]
        _size =
            try c.decodeIfPresent(Int.self, forKey: ._size)
            ?? c.decodeIfPresent(Int.self, forKey: ._imageSize)
        _patchSize = try c.decodeIfPresent(Int.self, forKey: ._patchSize)
        _downsampleRatio = try c.decodeIfPresent(Float.self, forKey: ._downsampleRatio)
        _numImageToken = try c.decodeIfPresent(Int.self, forKey: ._numImageToken)
        _doDynamicPreprocess = try c.decodeIfPresent(Bool.self, forKey: ._doDynamicPreprocess)
        _dynamicMinNum =
            try c.decodeIfPresent(Int.self, forKey: ._dynamicMinNum)
            ?? c.decodeIfPresent(Int.self, forKey: ._minDynamicPatch)
        _dynamicMaxNum =
            try c.decodeIfPresent(Int.self, forKey: ._dynamicMaxNum)
            ?? c.decodeIfPresent(Int.self, forKey: ._maxDynamicPatch)
        _dynamicUseThumbnail =
            try c.decodeIfPresent(Bool.self, forKey: ._dynamicUseThumbnail)
            ?? c.decodeIfPresent(Bool.self, forKey: ._useThumbnail)
    }
}

// MARK: - Processor

public struct InternVLProcessor: UserInputProcessor {
    private struct TileRatio: Hashable {
        let columns: Int
        let rows: Int

        var count: Int {
            columns * rows
        }

        var aspect: Double {
            Double(columns) / Double(rows)
        }
    }

    private let config: InternVLProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageStartTokenID: Int
    private let imageEndTokenID: Int
    private let imageContextTokenID: Int

    public init(_ config: InternVLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        imageStartTokenID = tokenizer.convertTokenToId("<img>") ?? 151_665
        imageEndTokenID = tokenizer.convertTokenToId("</img>") ?? 151_666
        imageContextTokenID = tokenizer.convertTokenToId("<IMG_CONTEXT>") ?? 151_667
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        let prompt = Self.prompt(from: input)

        if images.isEmpty {
            let tokens = tokenizer.encode(text: "User: \(prompt)\nAssistant:\n")
            let tokensArray = MLXArray(tokens).expandedDimensions(axis: 0)
            return LMInput(text: .init(tokens: tokensArray, mask: ones(like: tokensArray)))
        }

        var tileCounts: [Int] = []
        var tileArrays: [MLXArray] = []
        tileArrays.reserveCapacity(images.count)

        for image in images {
            let tiles = try imageTiles(from: image)
            tileCounts.append(tiles.count)
            tileArrays.append(contentsOf: tiles)
        }

        guard !tileArrays.isEmpty else {
            throw VLMError.imageProcessingFailure(
                "InternVL dynamic preprocessing produced no image tiles.")
        }

        let pixels = concatenated(tileArrays, axis: 0).expandedDimensions(axis: 0)
        let promptText = Self.ensureImagePlaceholders(in: prompt, imageCount: images.count)
        let tokens = tokenizedPrompt(
            "User: \(promptText)\nAssistant:\n",
            tileCounts: tileCounts,
        )
        let promptArray = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: promptArray, mask: ones(like: promptArray)),
            image: .init(pixels: pixels),
        )
    }

    func imageTiles(from inputImage: UserInput.Image) throws -> [MLXArray] {
        let image =
            try inputImage
            .asCIImage()
            .toSRGB()

        let tiles = try preprocessedTiles(from: image)
        return tiles.map { tile in
            tile
                .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
                .asMLXArray()
        }
    }

    private func preprocessedTiles(from image: CIImage) throws -> [CIImage] {
        if !config.doDynamicPreprocess {
            return [
                image
                    .resampled(
                        to: CGSize(width: config.size, height: config.size),
                        method: .bicubic,
                    )
            ]
        }

        let width = image.extent.width
        let height = image.extent.height
        guard width > 0, height > 0 else {
            throw VLMError.imageProcessingFailure(
                "Image has invalid dimensions \(width)x\(height).")
        }

        let target = closestAspectRatio(width: width, height: height)
        let targetWidth = CGFloat(config.size * target.columns)
        let targetHeight = CGFloat(config.size * target.rows)
        let resized =
            image
            .resampled(
                to: CGSize(width: targetWidth, height: targetHeight),
                method: .bicubic,
            )

        var tiles: [CIImage] = []
        tiles.reserveCapacity(target.count + 1)
        for index in 0 ..< target.count {
            let row = index / target.columns
            let column = index % target.columns
            let crop = CGRect(
                x: CGFloat(column * config.size),
                y: CGFloat((target.rows - row - 1) * config.size),
                width: CGFloat(config.size),
                height: CGFloat(config.size),
            )
            let tile =
                resized
                .cropped(to: crop)
                .transformed(by: CGAffineTransform(translationX: -crop.minX, y: -crop.minY))
            tiles.append(tile)
        }

        if config.dynamicUseThumbnail, target.count > 1 {
            tiles.append(
                image.resampled(
                    to: CGSize(width: config.size, height: config.size),
                    method: .bicubic,
                ),
            )
        }

        return tiles
    }

    private func closestAspectRatio(width: CGFloat, height: CGFloat) -> TileRatio {
        let aspectRatio = Double(width / height)
        let area = Double(width * height)
        var best = TileRatio(columns: 1, rows: 1)
        var bestDifference = Double.infinity

        for ratio in candidateRatios() {
            let difference = abs(aspectRatio - ratio.aspect)
            if difference < bestDifference {
                bestDifference = difference
                best = ratio
            } else if difference == bestDifference {
                let targetArea = Double(config.size * config.size * ratio.count)
                let bestArea = Double(config.size * config.size * best.count)
                if abs(area - targetArea) < abs(area - bestArea) {
                    best = ratio
                }
            }
        }

        return best
    }

    private func candidateRatios() -> [TileRatio] {
        let minBlocks = max(1, config.dynamicMinNum)
        let maxBlocks = max(minBlocks, config.dynamicMaxNum)
        var ratios: [TileRatio] = []
        var seen = Set<TileRatio>()

        for n in minBlocks ... maxBlocks {
            for columns in 1 ... n {
                for rows in 1 ... n {
                    let ratio = TileRatio(columns: columns, rows: rows)
                    if ratio.count >= minBlocks, ratio.count <= maxBlocks,
                        seen.insert(ratio).inserted
                    {
                        ratios.append(ratio)
                    }
                }
            }
        }

        return ratios.sorted {
            if $0.count != $1.count {
                return $0.count < $1.count
            }
            if $0.columns != $1.columns {
                return $0.columns < $1.columns
            }
            return $0.rows < $1.rows
        }
    }

    private func tokenizedPrompt(_ text: String, tileCounts: [Int]) -> [Int] {
        let placeholder = "<image>"
        var remaining = text[...]
        var tokens: [Int] = []

        for tileCount in tileCounts {
            guard let range = remaining.range(of: placeholder) else {
                tokens += imageTokenRun(tileCount: tileCount)
                continue
            }
            tokens += tokenizer.encode(text: String(remaining[..<range.lowerBound]))
            tokens += imageTokenRun(tileCount: tileCount)
            remaining = remaining[range.upperBound...]
        }

        tokens += tokenizer.encode(text: String(remaining))
        return tokens
    }

    private func imageTokenRun(tileCount: Int) -> [Int] {
        [imageStartTokenID]
            + Array(repeating: imageContextTokenID, count: config.numImageToken * tileCount)
            + [imageEndTokenID]
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

    private static func ensureImagePlaceholders(in prompt: String, imageCount: Int) -> String {
        let placeholder = "<image>"
        let existing = prompt.components(separatedBy: placeholder).count - 1
        guard existing < imageCount else {
            return prompt
        }
        return String(repeating: "\(placeholder)\n", count: imageCount - existing) + prompt
    }
}

// MARK: - Language

private enum InternVLLanguage {
    final class Attention: Module {
        let config: InternVLConfiguration.TextConfiguration
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        init(_ config: InternVLConfiguration.TextConfiguration) {
            self.config = config
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)
            _qProj.wrappedValue = Linear(
                config.hiddenSize, config.attentionHeads * headDim, bias: config.attentionBias,
            )
            _kProj.wrappedValue = Linear(
                config.hiddenSize, config.kvHeads * headDim, bias: config.attentionBias,
            )
            _vProj.wrappedValue = Linear(
                config.hiddenSize, config.kvHeads * headDim, bias: config.attentionBias,
            )
            _oProj.wrappedValue = Linear(
                config.attentionHeads * headDim, config.hiddenSize, bias: false,
            )
            _rotaryEmbedding.wrappedValue = RoPE(
                dimensions: headDim,
                traditional: config.ropeTraditional,
                base: config.ropeTheta,
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
        @ModuleInfo(key: "down_proj") var downProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear

        init(_ config: InternVLConfiguration.TextConfiguration) {
            _gateProj.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize, bias: false,
            )
            _downProj.wrappedValue = Linear(
                config.intermediateSize, config.hiddenSize, bias: false,
            )
            _upProj.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize, bias: false,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: InternVLConfiguration.TextConfiguration) {
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
            let r = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            return h + mlp(postAttentionLayerNorm(h))
        }
    }

    final class TextModel: Module {
        let config: InternVLConfiguration.TextConfiguration
        let layers: [DecoderLayer]

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo var norm: RMSNorm

        init(_ config: InternVLConfiguration.TextConfiguration) {
            self.config = config
            layers = (0 ..< config.hiddenLayers).map { _ in DecoderLayer(config) }
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
        @ModuleInfo var model: TextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear?
        let config: InternVLConfiguration.TextConfiguration

        init(_ config: InternVLConfiguration.TextConfiguration) {
            self.config = config
            _model.wrappedValue = TextModel(config)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.hiddenSize, config.vocabularySize, bias: false,
                )
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
            let out = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
            if config.tieWordEmbeddings {
                return LMOutput(logits: model.embedTokens.asLinear(out))
            }
            return LMOutput(logits: lmHead!(out))
        }
    }
}

// MARK: - Vision

private enum InternVLVision {
    final class Attention: Module {
        let config: InternVLConfiguration.VisionConfiguration
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear

        init(_ config: InternVLConfiguration.VisionConfiguration) {
            self.config = config
            heads = config.attentionHeads
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _qkv.wrappedValue = Linear(
                config.hiddenSize, 3 * config.hiddenSize, bias: config.qkvBias,
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

        init(_ config: InternVLConfiguration.VisionConfiguration) {
            _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
            _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    final class EncoderLayer: Module {
        @ModuleInfo var attn: Attention
        @ModuleInfo var mlp: MLP
        @ModuleInfo var norm1: LayerNorm
        @ModuleInfo var norm2: LayerNorm
        @ModuleInfo var ls1: MLXArray
        @ModuleInfo var ls2: MLXArray

        init(_ config: InternVLConfiguration.VisionConfiguration) {
            _attn.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(config)
            _norm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps,
            )
            _norm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps,
            )
            _ls1.wrappedValue = MLXArray.ones([config.hiddenSize])
            _ls2.wrappedValue = MLXArray.ones([config.hiddenSize])
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = x + attn(norm1(x)) * ls1
            return h + mlp(norm2(h)) * ls2
        }
    }

    final class Encoder: Module {
        let layers: [EncoderLayer]

        init(_ config: InternVLConfiguration.VisionConfiguration) {
            layers = (0 ..< config.hiddenLayers).map { _ in EncoderLayer(config) }
        }
    }

    final class VisionEmbeddings: Module {
        let config: InternVLConfiguration.VisionConfiguration
        let numPositions: Int

        @ModuleInfo(key: "class_embedding") var classEmbedding: MLXArray
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: MLXArray

        init(_ config: InternVLConfiguration.VisionConfiguration) {
            self.config = config
            let numPatches =
                (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize)
            numPositions = numPatches + 1
            _classEmbedding.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
            _patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.channels,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: true,
            )
            _positionEmbedding.wrappedValue = MLXArray.zeros([
                1, numPositions, config.hiddenSize,
            ])
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var patchEmbeddings = patchEmbedding(x).flattened(start: 1, end: 2)
            let classEmbeddings = broadcast(
                classEmbedding.asType(patchEmbeddings.dtype),
                to: [x.dim(0), 1, config.hiddenSize],
            )
            patchEmbeddings = concatenated([classEmbeddings, patchEmbeddings], axis: 1)
            return patchEmbeddings + positionEmbedding.asType(patchEmbeddings.dtype)
        }
    }

    final class VisionModel: Module {
        @ModuleInfo var embeddings: VisionEmbeddings
        @ModuleInfo var encoder: Encoder

        init(_ config: InternVLConfiguration.VisionConfiguration) {
            _embeddings.wrappedValue = VisionEmbeddings(config)
            _encoder.wrappedValue = Encoder(config)
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, [MLXArray]?,
        ) {
            var x = embeddings(x)
            var encoderStates: [MLXArray]? = outputHiddenStates ? [x] : nil
            for layer in encoder.layers {
                x = layer(x)
                encoderStates?.append(x)
            }
            let poolerOutput = x[0..., 0, 0...]
            return (x, poolerOutput, encoderStates.map { Array($0.dropFirst()) })
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized: [String: MLXArray] = [:]
            for (key, value) in weights {
                if key.contains("position_ids") {
                    continue
                }
                if key.contains("patch_embedding.weight"), value.ndim == 4 {
                    let shape = value.shape
                    if shape[1] == 3 {
                        sanitized[key] = value.transposed(0, 2, 3, 1)
                    } else {
                        sanitized[key] = value
                    }
                } else {
                    sanitized[key] = value
                }
            }
            return sanitized
        }
    }
}

// MARK: - Model

private final class InternVLProjector: Module, UnaryLayer {
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear
    let activation = GELU()

    init(_ config: InternVLConfiguration) {
        let factor = Int(1 / config.downsampleRatio)
        let inputSize = config.visionConfiguration.hiddenSize * factor * factor
        let hiddenSize = config.textConfiguration.hiddenSize
        _norm.wrappedValue = LayerNorm(dimensions: inputSize)
        _linear1.wrappedValue = Linear(inputSize, hiddenSize, bias: true)
        _linear2.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(activation(linear1(norm(x))))
    }
}

public final class InternVL: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_model") private var visionModel: InternVLVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: InternVLLanguage.LanguageModel
    @ModuleInfo(key: "mlp1") private var projector: InternVLProjector

    public let config: InternVLConfiguration
    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public init(_ config: InternVLConfiguration) {
        self.config = config
        _visionModel.wrappedValue = InternVLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = InternVLLanguage.LanguageModel(config.textConfiguration)
        _projector.wrappedValue = InternVLProjector(config)
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    private func inputEmbeddings(inputIDs: MLXArray, pixelValues: MLXArray?) -> MLXArray {
        guard let pixelValues else {
            return languageModel.model.embedTokens(inputIDs)
        }

        var pixels = pixelValues
        if pixels.ndim == 5 {
            pixels = pixels[0]
        }

        let safeInputIDs = which(
            (inputIDs .== config.imageTokenIndex) | (inputIDs .== config.videoTokenIndex),
            MLXArray(0),
            inputIDs,
        )
        let inputsEmbeds = languageModel.model.embedTokens(safeInputIDs)

        let (hiddenStates, _, _) = visionModel(pixels.transposed(0, 2, 3, 1))
        var imageFeatures = hiddenStates[0..., 1..., 0...]
        imageFeatures = pixelShuffle(imageFeatures, downsampleRatio: config.downsampleRatio)
        imageFeatures = projector(imageFeatures).asType(inputsEmbeds.dtype)
            .reshaped(-1, config.textConfiguration.hiddenSize)

        let ids = inputIDs[0].asArray(Int.self)
        var positions = ids.indices.filter { ids[$0] == config.imageTokenIndex }
        if positions.isEmpty {
            positions = ids.indices.filter { ids[$0] == config.videoTokenIndex }
        }
        guard let first = positions.first, let last = positions.last else {
            return inputsEmbeds
        }

        let before = inputsEmbeds[0, 0 ..< first, 0...]
        let after = inputsEmbeds[0, (last + 1) ..< ids.count, 0...]
        let merged = concatenated([before, imageFeatures, after], axis: 0)
        return merged.expandedDimensions(axis: 0)
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
        )

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
        let filtered = weights.filter { key, _ in
            !key.contains("self_attn.rotary_emb.inv_freq")
        }
        let remapped = Dictionary(
            uniqueKeysWithValues: filtered.map { key, value in
                if key.hasPrefix("mlp1.0.") {
                    return (key.replacingOccurrences(of: "mlp1.0.", with: "mlp1.norm."), value)
                }
                if key.hasPrefix("mlp1.1.") {
                    return (key.replacingOccurrences(of: "mlp1.1.", with: "mlp1.linear1."), value)
                }
                if key.hasPrefix("mlp1.3.") {
                    return (key.replacingOccurrences(of: "mlp1.3.", with: "mlp1.linear2."), value)
                }
                return (key, value)
            },
        )
        return visionModel.sanitize(weights: remapped)
    }

    private func pixelShuffle(_ input: MLXArray, downsampleRatio: Float) -> MLXArray {
        let batchSize = input.dim(0)
        let numPatches = input.dim(1)
        let channels = input.dim(2)
        let side = Int(Double(numPatches).squareRoot())
        let factor = Int((1 / downsampleRatio).rounded())

        var x = input.reshaped(batchSize, side, side, channels)
        x = x.reshaped(batchSize, side, side / factor, channels * factor)
        x = x.transposed(0, 2, 1, 3)
        x = x.reshaped(batchSize, side / factor, side / factor, channels * factor * factor)
        x = x.transposed(0, 2, 1, 3)
        return x.reshaped(batchSize, -1, channels * factor * factor)
    }
}
