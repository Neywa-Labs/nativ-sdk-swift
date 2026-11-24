import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct AyaVisionConfiguration: Decodable, Sendable {
    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        private let _headDim: Int?
        public var headDim: Int {
            _headDim ?? hiddenSize / attentionHeads
        }

        private let _kvHeads: Int?
        public var kvHeads: Int {
            _kvHeads ?? attentionHeads
        }

        private let _ropeTheta: Float?
        public var ropeTheta: Float {
            _ropeTheta ?? 50000
        }

        private let _vocabularySize: Int?
        public var vocabularySize: Int {
            _vocabularySize ?? 256_000
        }

        private let _layerNormEps: Float?
        public var layerNormEps: Float {
            _layerNormEps ?? 1e-5
        }

        private let _logitScale: Float?
        public var logitScale: Float {
            _logitScale ?? 0.25
        }

        private let _attentionBias: Bool?
        public var attentionBias: Bool {
            _attentionBias ?? false
        }

        private let _layerNormBias: Bool?
        public var layerNormBias: Bool {
            _layerNormBias ?? false
        }

        private let _slidingWindow: Int?
        public var slidingWindow: Int {
            _slidingWindow ?? 4096
        }

        private let _slidingWindowPattern: Int?
        public var slidingWindowPattern: Int {
            _slidingWindowPattern ?? 4
        }

        private let _maxPositionEmbeddings: Int?
        public var maxPositionEmbeddings: Int {
            _maxPositionEmbeddings ?? 4096
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case _headDim = "head_dim"
            case _kvHeads = "num_key_value_heads"
            case _ropeTheta = "rope_theta"
            case _vocabularySize = "vocab_size"
            case _layerNormEps = "layer_norm_eps"
            case _logitScale = "logit_scale"
            case _attentionBias = "attention_bias"
            case _layerNormBias = "layer_norm_bias"
            case _slidingWindow = "sliding_window"
            case _slidingWindowPattern = "sliding_window_pattern"
            case _maxPositionEmbeddings = "max_position_embeddings"
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let attentionHeads: Int
        public let patchSize: Int
        private let _hiddenLayers: Int?
        public var hiddenLayers: Int {
            _hiddenLayers ?? 12
        }

        private let _intermediateSize: Int?
        public var intermediateSize: Int {
            _intermediateSize ?? 3072
        }

        private let _imageSize: Int?
        public var imageSize: Int {
            _imageSize ?? 224
        }

        private let _channels: Int?
        public var channels: Int {
            _channels ?? 3
        }

        private let _layerNormEps: Float?
        public var layerNormEps: Float {
            _layerNormEps ?? 1e-6
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case attentionHeads = "num_attention_heads"
            case patchSize = "patch_size"
            case _hiddenLayers = "num_hidden_layers"
            case _intermediateSize = "intermediate_size"
            case _imageSize = "image_size"
            case _channels = "num_channels"
            case _layerNormEps = "layer_norm_eps"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int {
        _imageTokenIndex ?? 255_036
    }

    private let _maxSplitsPerImage: Int?
    public var maxSplitsPerImage: Int {
        _maxSplitsPerImage ?? 12
    }

    private let _downsampleFactor: Int?
    public var downsampleFactor: Int {
        _downsampleFactor ?? 2
    }

    private let _alignmentIntermediateSize: Int?
    public var alignmentIntermediateSize: Int {
        _alignmentIntermediateSize ?? textConfiguration.hiddenSize
    }

    private let _adapterLayerNormEps: Float?
    public var adapterLayerNormEps: Float {
        _adapterLayerNormEps ?? 1e-6
    }

    private let _visionFeatureLayer: Int?
    public var visionFeatureLayer: Int {
        _visionFeatureLayer ?? -1
    }

    private let _visionFeatureSelectStrategy: String?
    public var visionFeatureSelectStrategy: String {
        _visionFeatureSelectStrategy ?? "full"
    }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _imageTokenIndex = "image_token_index"
        case _maxSplitsPerImage = "max_splits_per_img"
        case _downsampleFactor = "downsample_factor"
        case _alignmentIntermediateSize = "alignment_intermediate_size"
        case _adapterLayerNormEps = "adapter_layer_norm_eps"
        case _visionFeatureLayer = "vision_feature_layer"
        case _visionFeatureSelectStrategy = "vision_feature_select_strategy"
    }
}

public struct AyaVisionProcessorConfiguration: Decodable, Sendable {
    public struct Size: Decodable, Sendable {
        private let height: Int?
        private let width: Int?
        private let shortestEdge: Int?

        public var heightValue: Int {
            height ?? shortestEdge ?? 364
        }

        public var widthValue: Int {
            width ?? shortestEdge ?? 364
        }

        enum CodingKeys: String, CodingKey {
            case height
            case width
            case shortestEdge = "shortest_edge"
        }
    }

    private struct NestedImageProcessor: Decodable {
        let size: Size?
        let imageMean: [CGFloat]?
        let imageStd: [CGFloat]?
        let cropToPatches: Bool?
        let maxPatches: Int?

        enum CodingKeys: String, CodingKey {
            case size
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case cropToPatches = "crop_to_patches"
            case maxPatches = "max_patches"
        }
    }

    public let processorClass: String
    public let size: Size
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    private let _patchSize: Int?
    public var patchSize: Int {
        _patchSize ?? 28
    }

    private let _imageSize: Int?
    public var imageSize: Int {
        _imageSize ?? size.heightValue
    }

    private let _imageToken: String?
    public var imageToken: String {
        _imageToken ?? "<image>"
    }

    private let _downsampleFactor: Int?
    public var downsampleFactor: Int {
        _downsampleFactor ?? 1
    }

    private let _startOfImageToken: String?
    public var startOfImageToken: String {
        _startOfImageToken ?? "<|START_OF_IMG|>"
    }

    private let _endOfImageToken: String?
    public var endOfImageToken: String {
        _endOfImageToken ?? "<|END_OF_IMG|>"
    }

    private let _imagePatchToken: String?
    public var imagePatchToken: String {
        _imagePatchToken ?? "<|IMG_PATCH|>"
    }

    private let _imageLineBreakToken: String?
    public var imageLineBreakToken: String {
        _imageLineBreakToken ?? "<|IMG_LINE_BREAK|>"
    }

    private let _tileToken: String?
    public var tileToken: String {
        _tileToken ?? "TILE"
    }

    private let _tileGlobalToken: String?
    public var tileGlobalToken: String {
        _tileGlobalToken ?? "TILE_GLOBAL"
    }

    private let _cropToPatches: Bool?
    public var cropToPatches: Bool {
        _cropToPatches ?? false
    }

    private let _maxPatches: Int?
    public var maxPatches: Int {
        _maxPatches ?? 12
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case size
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageProcessor = "image_processor"
        case _patchSize = "patch_size"
        case _imageSize = "img_size"
        case _imageToken = "image_token"
        case _downsampleFactor = "downsample_factor"
        case _startOfImageToken = "start_of_img_token"
        case _endOfImageToken = "end_of_img_token"
        case _imagePatchToken = "img_patch_token"
        case _imageLineBreakToken = "img_line_break_token"
        case _tileToken = "tile_token"
        case _tileGlobalToken = "tile_global_token"
        case _cropToPatches = "crop_to_patches"
        case _maxPatches = "max_patches"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let nested = try c.decodeIfPresent(NestedImageProcessor.self, forKey: .imageProcessor)
        processorClass =
            try c.decodeIfPresent(String.self, forKey: .processorClass)
            ?? "AyaVisionProcessor"
        size =
            try c.decodeIfPresent(Size.self, forKey: .size)
            ?? nested?.size
            ?? Size(height: 364, width: 364, shortestEdge: nil)
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageMean)
            ?? nested?.imageMean
            ?? [0.5, 0.5, 0.5]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageStd)
            ?? nested?.imageStd
            ?? [0.5, 0.5, 0.5]
        _patchSize = try c.decodeIfPresent(Int.self, forKey: ._patchSize)
        _imageSize = try c.decodeIfPresent(Int.self, forKey: ._imageSize)
        _imageToken = try c.decodeIfPresent(String.self, forKey: ._imageToken)
        _downsampleFactor = try c.decodeIfPresent(Int.self, forKey: ._downsampleFactor)
        _startOfImageToken = try c.decodeIfPresent(String.self, forKey: ._startOfImageToken)
        _endOfImageToken = try c.decodeIfPresent(String.self, forKey: ._endOfImageToken)
        _imagePatchToken = try c.decodeIfPresent(String.self, forKey: ._imagePatchToken)
        _imageLineBreakToken = try c.decodeIfPresent(String.self, forKey: ._imageLineBreakToken)
        _tileToken = try c.decodeIfPresent(String.self, forKey: ._tileToken)
        _tileGlobalToken = try c.decodeIfPresent(String.self, forKey: ._tileGlobalToken)
        _cropToPatches =
            try c.decodeIfPresent(Bool.self, forKey: ._cropToPatches)
            ?? nested?.cropToPatches
        _maxPatches =
            try c.decodeIfPresent(Int.self, forKey: ._maxPatches)
            ?? nested?.maxPatches
    }
}

extension AyaVisionProcessorConfiguration.Size {
    fileprivate init(height: Int, width: Int, shortestEdge: Int?) {
        self.height = height
        self.width = width
        self.shortestEdge = shortestEdge
    }
}

// MARK: - Processor

public struct AyaVisionMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        if message.role == .system {
            return [
                "role": message.role.rawValue,
                "content": message.content,
            ]
        }
        return [
            "role": message.role.rawValue,
            "content":
                message.images.map { _ in ["type": "image"] }
                + [["type": "text", "text": message.content]],
        ]
    }
}

public struct AyaVisionProcessor: UserInputProcessor {
    private let config: AyaVisionProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imagePatchTokenID: Int

    public init(_ config: AyaVisionProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        imagePatchTokenID =
            tokenizer.convertTokenToId(config.imagePatchToken) ?? 255_036
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        let messages = AyaVisionMessageGenerator().generate(from: input)
        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext,
        )

        guard !images.isEmpty else {
            let tokenArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            return LMInput(text: .init(tokens: tokenArray, mask: ones(like: tokenArray)))
        }

        guard images.count == 1 else {
            throw VLMError.singleImageAllowed
        }

        let processed = try preprocess(
            image: images[0].asCIImage(),
            processing: input.processing,
        )
        let expandedTokens = expandImagePlaceholders(
            in: promptTokens,
            patchCounts: [processed.numPatches],
        )
        let tokenArray = MLXArray(expandedTokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)),
            image: .init(pixels: processed.pixels),
        )
    }

    private struct ProcessedImage {
        let pixels: MLXArray
        let numPatches: Int
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?) throws
        -> ProcessedImage
    {
        let canonical = MediaProcessing.zeroOrigin(
            MediaProcessing.apply(image, processing: processing))
        let height = config.size.heightValue
        let width = config.size.widthValue
        let resized = Self.resizeImage(
            image: canonical,
            targetHeight: height,
            targetWidth: width,
            mean: config.imageMean,
            std: config.imageStd,
        )

        let pixels = MLXArray(resized.pixels, [1, 3, height, width])
        return ProcessedImage(pixels: pixels, numPatches: 1)
    }

    private func expandImagePlaceholders(in promptTokens: [Int], patchCounts: [Int]) -> [Int] {
        var decoded = tokenizer.decode(tokenIds: promptTokens, skipSpecialTokens: false)
        var replacements = 0
        for patchCount in patchCounts {
            guard let range = decoded.range(of: config.imageToken) else {
                break
            }
            decoded.replaceSubrange(range, with: promptSplitImage(numPatches: patchCount))
            replacements += 1
        }

        if replacements == patchCounts.count {
            return tokenizer.encode(text: decoded, addSpecialTokens: false)
        }

        let imageTokens = patchCounts.flatMap { patchCount in
            tokenizer.encode(
                text: promptSplitImage(numPatches: patchCount), addSpecialTokens: false)
        }
        return imageTokens + promptTokens
    }

    private func promptSplitImage(numPatches: Int) -> String {
        let effectivePatchSize = config.patchSize * config.downsampleFactor
        let patchesPerSide = max(config.imageSize / max(effectivePatchSize, 1), 1)
        let imagePatchesPerTile = patchesPerSide * patchesPerSide

        var result = config.startOfImageToken
        if numPatches > 1 {
            for index in 1 ..< numPatches {
                result += "\(config.tileToken)_\(index)"
                result += String(repeating: config.imagePatchToken, count: imagePatchesPerTile)
            }
        }
        result += config.tileGlobalToken
        result += String(repeating: config.imagePatchToken, count: imagePatchesPerTile)
        result += config.endOfImageToken
        return result
    }

    private struct ResizedImage {
        let pixels: [Float]
    }

    private static func resizeImage(
        image: CIImage,
        targetHeight: Int,
        targetWidth: Int,
        mean: [CGFloat],
        std: [CGFloat],
    )
        -> ResizedImage
    {
        let resized = MediaProcessing.zeroOrigin(image).resampled(
            to: CGSize(width: targetWidth, height: targetHeight),
            method: .bicubic,
        )
        let raw = MediaProcessing.renderRGB8(resized, width: targetWidth, height: targetHeight)
        return ResizedImage(
            pixels: MediaProcessing.normalizeCHW(
                raw,
                width: targetWidth,
                height: targetHeight,
                mean: mean,
                std: std,
            ),
        )
    }

}

// MARK: - Language

private enum AyaVisionLanguage {
    final class Attention: Module {
        let config: AyaVisionConfiguration.TextConfiguration
        let layerIndex: Int
        let useSlidingWindow: Bool
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        let rope: RoPE

        init(_ config: AyaVisionConfiguration.TextConfiguration, layerIndex: Int) {
            self.config = config
            self.layerIndex = layerIndex
            useSlidingWindow = (layerIndex + 1) % config.slidingWindowPattern != 0
            headDim = config.headDim
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
            rope = RoPE(dimensions: headDim, traditional: true, base: config.ropeTheta)
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

            if useSlidingWindow {
                queries = applyRotaryPosition(rope, to: queries, offset: cache?.ropeOffset)
                keys = applyRotaryPosition(rope, to: keys, offset: cache?.ropeOffset)
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

        init(_ config: AyaVisionConfiguration.TextConfiguration) {
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

    final class TransformerBlock: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm

        init(_ config: AyaVisionConfiguration.TextConfiguration, layerIndex: Int) {
            _attention.wrappedValue = Attention(config, layerIndex: layerIndex)
            mlp = MLP(config)
            _inputLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
                bias: config.layerNormBias,
            )
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let h = inputLayerNorm(x)
            let attentionHidden = attention(h, mask: mask, cache: cache)
            let mlpHidden = mlp(h)
            return attentionHidden + mlpHidden + x
        }
    }

    final class CohereModel: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        let layers: [TransformerBlock]
        @ModuleInfo var norm: LayerNorm
        let config: AyaVisionConfiguration.TextConfiguration

        init(_ config: AyaVisionConfiguration.TextConfiguration) {
            self.config = config
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            layers = (0 ..< config.hiddenLayers).map {
                TransformerBlock(config, layerIndex: $0)
            }
            _norm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
                bias: config.layerNormBias,
            )
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

            let globalMask = createAttentionMask(
                h: h, cache: cache?[safe: config.slidingWindowPattern - 1])
            let slidingMask = createAttentionMask(
                h: h,
                cache: cache?.first,
                windowSize: config.slidingWindow,
            )

            for (index, layer) in layers.enumerated() {
                let isGlobal = (index + 1) % config.slidingWindowPattern == 0
                h = layer(h, mask: isGlobal ? globalMask : slidingMask, cache: cache?[index])
            }
            return norm(h)
        }
    }

    final class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: CohereModel
        let config: AyaVisionConfiguration.TextConfiguration
        let kvHeads: [Int]

        init(_ config: AyaVisionConfiguration.TextConfiguration) {
            self.config = config
            _model.wrappedValue = CohereModel(config)
            kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
        )
            -> LMOutput
        {
            var out = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
            out = model.embedTokens.asLinear(out) * config.logitScale
            return LMOutput(logits: out)
        }
    }
}

// MARK: - Vision

private enum AyaVisionVision {
    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: AyaVisionConfiguration.VisionConfiguration) {
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
            var queries = qProj(x)
            var keys = kProj(x)
            var values = vProj(x)

            queries = queries.reshaped(batchSize, sequenceLength, heads, headDim)
                .transposed(0, 2, 1, 3)
            keys = keys.reshaped(batchSize, sequenceLength, heads, headDim)
                .transposed(0, 2, 1, 3)
            values = values.reshaped(batchSize, sequenceLength, heads, headDim)
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
        let activation = GELU(approximation: .precise)

        init(_ config: AyaVisionConfiguration.VisionConfiguration) {
            _fc1.wrappedValue = Linear(
                config.hiddenSize,
                config.intermediateSize,
                bias: true,
            )
            _fc2.wrappedValue = Linear(
                config.intermediateSize,
                config.hiddenSize,
                bias: true,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    final class EncoderLayer: Module, UnaryLayer {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        let mlp: MLP
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

        init(_ config: AyaVisionConfiguration.VisionConfiguration) {
            _attention.wrappedValue = Attention(config)
            _layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
            mlp = MLP(config)
            _layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let hidden = x + attention(layerNorm1(x))
            return hidden + mlp(layerNorm2(hidden))
        }
    }

    final class Encoder: Module {
        let layers: [EncoderLayer]

        init(_ config: AyaVisionConfiguration.VisionConfiguration) {
            layers = (0 ..< config.hiddenLayers).map { _ in EncoderLayer(config) }
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, [MLXArray]?,
        ) {
            var h = x
            var states: [MLXArray]? = outputHiddenStates ? [h] : nil
            for layer in layers {
                h = layer(h)
                states?.append(h)
            }
            return (h, states)
        }
    }

    final class VisionEmbeddings: Module, UnaryLayer {
        let numPositions: Int

        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        init(_ config: AyaVisionConfiguration.VisionConfiguration) {
            let positionsPerSide = config.imageSize / config.patchSize
            numPositions = positionsPerSide * positionsPerSide
            _patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.channels,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: true,
            )
            _positionEmbedding.wrappedValue = Embedding(
                embeddingCount: numPositions,
                dimensions: config.hiddenSize,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let patchEmbeddings = patchEmbedding(x).flattened(start: 1, end: 2)
            let positionIDs = MLXArray(0 ..< numPositions).expandedDimensions(axis: 0)
            return patchEmbeddings + positionEmbedding(positionIDs)
        }
    }

    final class SigLipVisionModel: Module {
        @ModuleInfo var embeddings: VisionEmbeddings
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        init(_ config: AyaVisionConfiguration.VisionConfiguration) {
            _embeddings.wrappedValue = VisionEmbeddings(config)
            _encoder.wrappedValue = Encoder(config)
            _postLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, [MLXArray]?,
        ) {
            let embeddings = embeddings(x).asType(embeddings.patchEmbedding.weight.dtype)
            let (hidden, states) = encoder(embeddings, outputHiddenStates: outputHiddenStates)
            let poolerOutput = postLayerNorm(hidden)
            return (poolerOutput, embeddings, states)
        }
    }

    final class VisionModel: Module {
        @ModuleInfo(key: "vision_model") var visionModel: SigLipVisionModel

        init(_ config: AyaVisionConfiguration.VisionConfiguration) {
            _visionModel.wrappedValue = SigLipVisionModel(config)
        }

        var patchEmbeddingDtype: DType {
            visionModel.embeddings.patchEmbedding.weight.dtype
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, [MLXArray]?,
        ) {
            visionModel(x, outputHiddenStates: outputHiddenStates)
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

// MARK: - Projector

private final class AyaVisionMultiModalProjector: Module, UnaryLayer {
    let config: AyaVisionConfiguration

    @ModuleInfo var layernorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear

    init(_ config: AyaVisionConfiguration) {
        self.config = config
        let shuffledHiddenSize =
            config.visionConfiguration.hiddenSize
            * config.downsampleFactor
            * config.downsampleFactor
        _layernorm.wrappedValue = LayerNorm(
            dimensions: shuffledHiddenSize,
            eps: config.adapterLayerNormEps,
        )
        _linear1.wrappedValue = Linear(
            shuffledHiddenSize,
            config.alignmentIntermediateSize,
            bias: true,
        )
        _linear2.wrappedValue = Linear(
            config.alignmentIntermediateSize / 2,
            config.textConfiguration.hiddenSize,
            bias: true,
        )
    }

    func callAsFunction(_ imageFeatures: MLXArray) -> MLXArray {
        var hidden = pixelShuffle(imageFeatures)
        hidden = layernorm(hidden)
        let projected = linear1(hidden)
        let parts = split(projected, parts: 2, axis: -1)
        hidden = silu(parts[1]) * parts[0]
        return linear2(hidden)
    }

    private func pixelShuffle(_ imageFeatures: MLXArray) -> MLXArray {
        let batchSize = imageFeatures.dim(0)
        let sequenceLength = imageFeatures.dim(1)
        let side = Int(Float(sequenceLength).squareRoot())
        let downsample = config.downsampleFactor
        var hidden = imageFeatures.reshaped(batchSize, side, side, -1)
        let channels = hidden.dim(-1)
        hidden = hidden.reshaped(
            batchSize,
            side,
            side / downsample,
            channels * downsample,
        )
        hidden = hidden.transposed(0, 2, 1, 3)
        hidden = hidden.reshaped(
            batchSize,
            side / downsample,
            side / downsample,
            -1,
        )
        return hidden.transposed(0, 2, 1, 3)
    }
}

// MARK: - Model

public final class AyaVision: Module, LanguageModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    let config: AyaVisionConfiguration
    @ModuleInfo(key: "vision_tower") private var visionTower: AyaVisionVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: AyaVisionLanguage.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector:
        AyaVisionMultiModalProjector

    public init(_ config: AyaVisionConfiguration) {
        self.config = config
        vocabularySize = config.textConfiguration.vocabularySize
        kvHeads = (0 ..< config.textConfiguration.hiddenLayers).map {
            _ in config.textConfiguration.kvHeads
        }
        _visionTower.wrappedValue = AyaVisionVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = AyaVisionLanguage.LanguageModel(
            config.textConfiguration,
        )
        _multiModalProjector.wrappedValue = AyaVisionMultiModalProjector(config)
    }

    private func inputEmbeddings(inputIDs: MLXArray, pixelValues: MLXArray?) -> MLXArray {
        let embeddings = languageModel.model.embedTokens(inputIDs)
        guard let pixelValues else {
            return embeddings
        }

        let pixels =
            pixelValues
            .asType(visionTower.patchEmbeddingDtype)
            .transposed(0, 2, 3, 1)
        let (_, _, hiddenStates) = visionTower(pixels, outputHiddenStates: true)
        guard let hiddenStates else {
            return embeddings
        }

        let layerIndex =
            config.visionFeatureLayer >= 0
            ? config.visionFeatureLayer
            : hiddenStates.count + config.visionFeatureLayer
        var selectedImageFeature = hiddenStates[layerIndex]
        if config.visionFeatureSelectStrategy == "default" {
            selectedImageFeature = selectedImageFeature[0..., 1..., 0...]
        }

        let imageFeatures = multiModalProjector(selectedImageFeature)
            .asType(embeddings.dtype)
            .reshaped(-1, config.textConfiguration.hiddenSize)
        let ids = inputIDs[0].asArray(Int.self)
        let positions = ids.indices.filter { ids[$0] == config.imageTokenIndex }
        precondition(
            positions.count == imageFeatures.dim(0),
            "Aya Vision image feature/token count mismatch: tokens \(positions.count), features \(imageFeatures.dim(0))",
        )

        embeddings[0..., MLXArray(positions), 0...] = imageFeatures.expandedDimensions(axis: 0)
        return embeddings
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
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

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var renamed: [String: MLXArray] = [:]
        for (key, value) in weights where !key.contains("rotary_emb.inv_freq") {
            var newKey = key
            if newKey.hasPrefix("model.vision_tower.") {
                newKey = newKey.replacingOccurrences(
                    of: "model.vision_tower.",
                    with: "vision_tower.",
                    options: [.anchored],
                )
            } else if newKey.hasPrefix("model.multi_modal_projector.") {
                newKey = newKey.replacingOccurrences(
                    of: "model.multi_modal_projector.",
                    with: "multi_modal_projector.",
                    options: [.anchored],
                )
            } else if newKey.hasPrefix("model.language_model.model.") {
                newKey = newKey.replacingOccurrences(
                    of: "model.language_model.",
                    with: "language_model.",
                    options: [.anchored],
                )
            } else if newKey.hasPrefix("model.language_model.") {
                newKey = newKey.replacingOccurrences(
                    of: "model.language_model.",
                    with: "language_model.model.",
                    options: [.anchored],
                )
            } else if newKey.hasPrefix("lm_head.") {
                newKey = newKey.replacingOccurrences(
                    of: "lm_head.",
                    with: "language_model.lm_head.",
                    options: [.anchored],
                )
            }
            renamed[newKey] = value
        }
        return visionTower.sanitize(weights: renamed)
    }
}

extension Collection {
    fileprivate subscript(safe index: Index) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}
