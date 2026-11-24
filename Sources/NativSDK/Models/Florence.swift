import CoreImage
import Foundation
import MLX
import MLXNN

let florenceCrossAttentionStatesKey = LMOutput.Key<MLXArray>(
    "nativsdk.florence.crossAttentionStates",
)

// MARK: - Configuration

public struct FlorenceConfiguration: Codable, Sendable {
    public struct TextConfiguration: Codable, Sendable {
        public let modelType: String
        public let modelDimension: Int
        public let encoderAttentionHeads: Int
        public let decoderAttentionHeads: Int
        public let encoderFFNDimension: Int
        public let decoderFFNDimension: Int
        public let encoderLayers: Int
        public let decoderLayers: Int
        public let vocabularySize: Int
        private let _maxPositionEmbeddings: Int?
        public var maxPositionEmbeddings: Int {
            _maxPositionEmbeddings ?? 1024
        }

        private let _padTokenID: Int?
        public var padTokenID: Int {
            _padTokenID ?? 1
        }

        private let _bosTokenID: Int?
        public var bosTokenID: Int {
            _bosTokenID ?? 0
        }

        private let _eosTokenID: Int?
        public var eosTokenID: Int {
            _eosTokenID ?? 2
        }

        private let _decoderStartTokenID: Int?
        public var decoderStartTokenID: Int {
            _decoderStartTokenID ?? eosTokenID
        }

        private let _scaleEmbedding: Bool?
        public var scaleEmbedding: Bool {
            _scaleEmbedding ?? false
        }

        private let _dropout: Float?
        public var dropout: Float {
            _dropout ?? 0
        }

        private let _encoderLayerDrop: Float?
        public var encoderLayerDrop: Float {
            _encoderLayerDrop ?? 0
        }

        private let _decoderLayerDrop: Float?
        public var decoderLayerDrop: Float {
            _decoderLayerDrop ?? 0
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case modelDimension = "d_model"
            case encoderAttentionHeads = "encoder_attention_heads"
            case decoderAttentionHeads = "decoder_attention_heads"
            case encoderFFNDimension = "encoder_ffn_dim"
            case decoderFFNDimension = "decoder_ffn_dim"
            case encoderLayers = "encoder_layers"
            case decoderLayers = "decoder_layers"
            case vocabularySize = "vocab_size"
            case _maxPositionEmbeddings = "max_position_embeddings"
            case _padTokenID = "pad_token_id"
            case _bosTokenID = "bos_token_id"
            case _eosTokenID = "eos_token_id"
            case _decoderStartTokenID = "decoder_start_token_id"
            case _scaleEmbedding = "scale_embedding"
            case _dropout = "dropout"
            case _encoderLayerDrop = "encoder_layerdrop"
            case _decoderLayerDrop = "decoder_layerdrop"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public struct EmbeddingSpec: Codable, Sendable {
            public let type: String
            public let maxPositionEmbeddings: Int?
            public let maxTemporalEmbeddings: Int?

            enum CodingKeys: String, CodingKey {
                case type
                case maxPositionEmbeddings = "max_pos_embeddings"
                case maxTemporalEmbeddings = "max_temporal_embeddings"
            }
        }

        private let _modelType: String?
        public var modelType: String {
            _modelType ?? "davit"
        }

        public let depths: [Int]
        public let embedDimensions: [Int]
        public let attentionHeads: [Int]
        public let attentionGroups: [Int]
        private let _windowSize: Int?
        public var windowSize: Int {
            _windowSize ?? 12
        }

        private let _mlpRatio: Float?
        public var mlpRatio: Float {
            _mlpRatio ?? 4
        }

        private let _dropPathRate: Float?
        public var dropPathRate: Float {
            _dropPathRate ?? 0
        }

        private let _patchSizes: [Int]?
        public var patchSizes: [Int] {
            _patchSizes ?? [7, 3, 3, 3]
        }

        private let _patchStrides: [Int]?
        public var patchStrides: [Int] {
            _patchStrides ?? [4, 2, 2, 2]
        }

        private let _patchPadding: [Int]?
        public var patchPadding: [Int] {
            _patchPadding ?? [3, 1, 1, 1]
        }

        private let _patchPreNorm: [Bool]?
        public var patchPreNorm: [Bool] {
            _patchPreNorm ?? Array(repeating: false, count: depths.count)
        }

        private let _qkvBias: Bool?
        public var qkvBias: Bool {
            _qkvBias ?? true
        }

        private let _convAtAttention: Bool?
        public var convAtAttention: Bool {
            _convAtAttention ?? true
        }

        private let _convAtFFN: Bool?
        public var convAtFFN: Bool {
            _convAtFFN ?? true
        }

        private let _channels: Int?
        public var channels: Int {
            _channels ?? 3
        }

        private let _imageSize: Int?
        public var imageSize: Int {
            _imageSize ?? 768
        }

        private let _hiddenSize: Int?
        public var hiddenSize: Int {
            _hiddenSize ?? 768
        }

        public let imagePositionEmbedding: EmbeddingSpec?
        public let visualTemporalEmbedding: EmbeddingSpec?

        enum CodingKeys: String, CodingKey {
            case _modelType = "model_type"
            case depths
            case embedDimensions = "dim_embed"
            case attentionHeads = "num_heads"
            case attentionGroups = "num_groups"
            case _windowSize = "window_size"
            case _mlpRatio = "mlp_ratio"
            case _dropPathRate = "drop_path_rate"
            case _patchSizes = "patch_size"
            case _patchStrides = "patch_stride"
            case _patchPadding = "patch_padding"
            case _patchPreNorm = "patch_prenorm"
            case _qkvBias = "qkv_bias"
            case _convAtAttention = "conv_at_attn"
            case _convAtFFN = "conv_at_ffn"
            case _channels = "in_chans"
            case _imageSize = "image_size"
            case _hiddenSize = "hidden_size"
            case imagePositionEmbedding = "image_pos_embed"
            case visualTemporalEmbedding = "visual_temporal_embedding"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _vocabularySize: Int?
    public var vocabularySize: Int {
        _vocabularySize ?? textConfiguration.vocabularySize
    }

    private let _imageTokenID: Int?
    public var imageTokenID: Int {
        _imageTokenID ?? textConfiguration.vocabularySize
    }

    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int {
        _imageTokenIndex ?? imageTokenID
    }

    private let _padTokenID: Int?
    public var padTokenID: Int {
        _padTokenID ?? textConfiguration.padTokenID
    }

    private let _imageFeatureSource: [String]?
    public var imageFeatureSource: [String] {
        _imageFeatureSource ?? ["temporal_avg_pool", "spatial_avg_pool"]
    }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _vocabularySize = "vocab_size"
        case _imageTokenID = "image_token_id"
        case _imageTokenIndex = "image_token_index"
        case _padTokenID = "pad_token_id"
        case _imageFeatureSource = "image_feature_source"
    }
}

public struct FlorenceProcessorConfiguration: Decodable, Sendable {
    public struct Size: Decodable, Sendable {
        public let width: Int?
        public let height: Int?
        public let shortestEdge: Int?

        public var cgSize: CGSize {
            CGSize(width: width ?? shortestEdge ?? 768, height: height ?? shortestEdge ?? 768)
        }

        enum CodingKeys: String, CodingKey {
            case width
            case height
            case shortestEdge = "shortest_edge"
        }
    }

    public let processorClass: String?
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let size: Size?
    private let _doNormalize: Bool?
    public var doNormalize: Bool {
        _doNormalize ?? true
    }

    private let _imageSequenceLength: Int?
    public var imageSequenceLength: Int {
        _imageSequenceLength ?? 577
    }

    private let _rescaleFactor: CGFloat?
    public var rescaleFactor: CGFloat {
        _rescaleFactor ?? (1 / 255)
    }

    private let _imageToken: String?
    public var imageToken: String {
        _imageToken ?? "<florence_image>"
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
        case size
        case _doNormalize = "do_normalize"
        case _imageSequenceLength = "image_seq_length"
        case _rescaleFactor = "rescale_factor"
        case _imageToken = "image_token"
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
        size = try c.decodeIfPresent(Size.self, forKey: .size)
        _doNormalize = try c.decodeIfPresent(Bool.self, forKey: ._doNormalize)
        _imageSequenceLength = try c.decodeIfPresent(Int.self, forKey: ._imageSequenceLength)
        _rescaleFactor = try c.decodeIfPresent(CGFloat.self, forKey: ._rescaleFactor)
        _imageToken = try c.decodeIfPresent(String.self, forKey: ._imageToken)
    }
}

// MARK: - Processor

public struct FlorenceProcessor: UserInputProcessor {
    private let config: FlorenceProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageTokenID: Int

    public init(_ config: FlorenceProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        if config.imageToken == "<florence_image>" {
            imageTokenID = 51289
        } else {
            imageTokenID = tokenizer.convertTokenToId(config.imageToken) ?? 51289
        }
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        let prompt = Self.normalizedTaskPrompt(Self.prompt(from: input))
        var tokens = tokenizer.encode(text: prompt, addSpecialTokens: false)

        guard !images.isEmpty else {
            let tokensArray = MLXArray(tokens).expandedDimensions(axis: 0)
            return LMInput(text: .init(tokens: tokensArray, mask: ones(like: tokensArray)))
        }

        guard images.count == 1 else {
            throw VLMError.singleImageAllowed
        }

        tokens.insert(imageTokenID, at: 0)
        let tokensArray = MLXArray(tokens).expandedDimensions(axis: 0)
        let pixels = try preprocess(images[0])
        return LMInput(
            text: .init(tokens: tokensArray, mask: ones(like: tokensArray)),
            image: .init(pixels: pixels),
        )
    }

    private func preprocess(_ inputImage: UserInput.Image) throws -> MLXArray {
        var image = try inputImage.asCIImage()
            .toSRGB()
            .resampled(to: config.size?.cgSize ?? CGSize(width: 768, height: 768), method: .bicubic)
        if config.doNormalize {
            image = image.normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
        }
        return image.asMLXArray()
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

    private static func normalizedTaskPrompt(_ prompt: String) -> String {
        let trimmed = prompt.trimmingCharacters(in: .whitespacesAndNewlines)
        let upper = trimmed.uppercased()
        switch upper {
        case "", "<CAPTION>", "<CAP>":
            return "What does the image describe?"
        case "<DETAILED_CAPTION>", "<DCAP>":
            return "Describe in detail what is shown in the image."
        case "<MORE_DETAILED_CAPTION>":
            return "Describe with a paragraph what is shown in the image."
        case "<OD>", "<OBJECT_DETECTION>":
            return "Locate the objects with category name in the image."
        case "<OCR>":
            return "What is the text in the image?"
        default:
            break
        }
        if trimmed.hasPrefix("<") {
            return trimmed
        }
        if upper.contains("DETAIL") {
            return "Describe in detail what is shown in the image."
        }
        if upper.contains("OCR") || upper.contains("TEXT") {
            return "What is the text in the image?"
        }
        return "What does the image describe?"
    }
}

// MARK: - Vision

private enum FlorenceVision {
    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "fc1") var fc1: Linear
        @ModuleInfo(key: "fc2") var fc2: Linear
        let activation = GELU()

        init(inputDimensions: Int, hiddenDimensions: Int, outputDimensions: Int) {
            _fc1.wrappedValue = Linear(inputDimensions, hiddenDimensions, bias: true)
            _fc2.wrappedValue = Linear(hiddenDimensions, outputDimensions, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    final class MLPBlock: Module {
        @ModuleInfo(key: "net") var net: MLP

        init(dimensions: Int, hiddenDimensions: Int) {
            _net.wrappedValue = MLP(
                inputDimensions: dimensions,
                hiddenDimensions: hiddenDimensions,
                outputDimensions: dimensions,
            )
        }

        func callAsFunction(_ x: MLXArray, size: (Int, Int)) -> (MLXArray, (Int, Int)) {
            (net(x), size)
        }
    }

    final class DepthWiseConv2d: Module {
        @ModuleInfo(key: "dw") var depthWise: Conv2d

        init(dimensions: Int, kernelSize: Int, padding: Int, stride: Int) {
            _depthWise.wrappedValue = Conv2d(
                inputChannels: dimensions,
                outputChannels: dimensions,
                kernelSize: IntOrPair(kernelSize),
                stride: IntOrPair(stride),
                padding: IntOrPair(padding),
                groups: dimensions,
                bias: true,
            )
        }

        func callAsFunction(_ x: MLXArray, size: (Int, Int)) -> (MLXArray, (Int, Int)) {
            let batchSize = x.dim(0)
            let channels = x.dim(2)
            let (height, width) = size
            var y = x.reshaped(batchSize, height, width, channels)
            y = depthWise(y)
            let outputSize = (y.dim(1), y.dim(2))
            return (y.flattened(start: 1, end: 2), outputSize)
        }
    }

    final class ConvFunction: Module {
        @ModuleInfo(key: "fn") var function: DepthWiseConv2d

        init(dimensions: Int, kernelSize: Int, padding: Int, stride: Int) {
            _function.wrappedValue = DepthWiseConv2d(
                dimensions: dimensions,
                kernelSize: kernelSize,
                padding: padding,
                stride: stride,
            )
        }

        func callAsFunction(_ x: MLXArray, size: (Int, Int)) -> (MLXArray, (Int, Int)) {
            function(x, size: size)
        }
    }

    final class ConvEmbed: Module {
        let preNorm: Bool
        @ModuleInfo(key: "proj") var projection: Conv2d
        @ModuleInfo var norm: LayerNorm?

        init(
            patchSize: Int,
            inputChannels: Int,
            embedDimensions: Int,
            stride: Int,
            padding: Int,
            preNorm: Bool,
        ) {
            self.preNorm = preNorm
            _projection.wrappedValue = Conv2d(
                inputChannels: inputChannels,
                outputChannels: embedDimensions,
                kernelSize: IntOrPair(patchSize),
                stride: IntOrPair(stride),
                padding: IntOrPair(padding),
                bias: true,
            )
            if preNorm {
                _norm.wrappedValue = LayerNorm(dimensions: inputChannels)
            } else {
                _norm.wrappedValue = LayerNorm(dimensions: embedDimensions)
            }
        }

        func callAsFunction(_ x: MLXArray, size: (Int, Int)) -> (MLXArray, (Int, Int)) {
            var x = x
            if x.ndim == 3 {
                if let norm, preNorm {
                    x = norm(x)
                }
                x = x.reshaped(-1, size.0, size.1, x.dim(-1))
            }
            x = projection(x)
            let outputSize = (x.dim(1), x.dim(2))
            x = x.reshaped(x.dim(0), outputSize.0 * outputSize.1, x.dim(3))
            if let norm, !preNorm {
                x = norm(x)
            }
            return (x, outputSize)
        }
    }

    final class ChannelAttention: Module {
        let groups: Int
        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear

        init(dimensions: Int, groups: Int, qkvBias: Bool) {
            self.groups = groups
            _qkv.wrappedValue = Linear(dimensions, dimensions * 3, bias: qkvBias)
            _proj.wrappedValue = Linear(dimensions, dimensions, bias: true)
        }

        func callAsFunction(_ x: MLXArray, size: (Int, Int)) -> (MLXArray, (Int, Int)) {
            let batchSize = x.dim(0)
            let sequenceLength = x.dim(1)
            let channels = x.dim(2)
            let qkvStates = qkv(x)
                .reshaped(batchSize, sequenceLength, 3, groups, channels / groups)
                .transposed(2, 0, 3, 1, 4)
            let queries = qkvStates[0] * pow(Float(sequenceLength), -0.5)
            let keys = qkvStates[1]
            let values = qkvStates[2]

            let attention = softmax(
                matmul(queries.transposed(0, 1, 3, 2), keys),
                axis: -1,
            )
            let y = matmul(attention, values.transposed(0, 1, 3, 2))
                .transposed(0, 1, 3, 2)
                .transposed(0, 2, 1, 3)
                .reshaped(batchSize, sequenceLength, channels)
            return (proj(y), size)
        }
    }

    static func windowPartition(_ x: MLXArray, windowSize: Int) -> MLXArray {
        let batchSize = x.dim(0)
        let height = x.dim(1)
        let width = x.dim(2)
        let channels = x.dim(3)
        let y = x.reshaped(
            batchSize,
            height / windowSize,
            windowSize,
            width / windowSize,
            windowSize,
            channels,
        )
        return y.transposed(0, 1, 3, 2, 4, 5)
            .reshaped(-1, windowSize, windowSize, channels)
    }

    static func windowReverse(
        _ windows: MLXArray,
        batchSize: Int,
        windowSize: Int,
        height: Int,
        width: Int,
    )
        -> MLXArray
    {
        windows
            .reshaped(
                batchSize,
                height / windowSize,
                width / windowSize,
                windowSize,
                windowSize,
                -1,
            )
            .transposed(0, 1, 3, 2, 4, 5)
            .reshaped(batchSize, height, width, -1)
    }

    final class WindowAttention: Module {
        let dimensions: Int
        let windowSize: Int
        let heads: Int
        let headDimensions: Int
        let scale: Float

        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear

        init(dimensions: Int, heads: Int, windowSize: Int, qkvBias: Bool) {
            self.dimensions = dimensions
            self.windowSize = windowSize
            self.heads = heads
            headDimensions = dimensions / heads
            scale = pow(Float(headDimensions), -0.5)
            _qkv.wrappedValue = Linear(dimensions, dimensions * 3, bias: qkvBias)
            _proj.wrappedValue = Linear(dimensions, dimensions, bias: true)
        }

        func callAsFunction(_ x: MLXArray, size: (Int, Int)) -> (MLXArray, (Int, Int)) {
            let (height, width) = size
            let batchSize = x.dim(0)
            let sequenceLength = x.dim(1)
            let channels = x.dim(2)
            precondition(sequenceLength == height * width, "input feature has wrong size")

            let padRight = (windowSize - width % windowSize) % windowSize
            let padBottom = (windowSize - height % windowSize) % windowSize
            var y = x.reshaped(batchSize, height, width, channels)
            y = padded(
                y,
                widths: [
                    IntOrPair((0, 0)),
                    IntOrPair((0, padBottom)),
                    IntOrPair((0, padRight)),
                    IntOrPair((0, 0)),
                ],
            )

            let paddedHeight = y.dim(1)
            let paddedWidth = y.dim(2)
            y = FlorenceVision.windowPartition(y, windowSize: windowSize)
                .reshaped(-1, windowSize * windowSize, channels)

            let windowBatchSize = y.dim(0)
            let windowTokens = y.dim(1)
            let qkvStates = qkv(y)
                .reshaped(windowBatchSize, windowTokens, 3, heads, channels / heads)
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
            .reshaped(windowBatchSize, windowTokens, channels)

            y = proj(output)
                .reshaped(-1, windowSize, windowSize, channels)
            y = FlorenceVision.windowReverse(
                y,
                batchSize: batchSize,
                windowSize: windowSize,
                height: paddedHeight,
                width: paddedWidth,
            )

            if padRight > 0 || padBottom > 0 {
                y = y[0..., ..<height, ..<width, 0...]
            }
            return (y.reshaped(batchSize, height * width, channels), size)
        }
    }

    final class SpatialBlock: Module {
        @ModuleInfo var conv1: ConvFunction?
        @ModuleInfo(key: "window_attn") var windowAttention: NormedWindowAttention
        @ModuleInfo var conv2: ConvFunction?
        @ModuleInfo var ffn: NormedMLP

        init(
            dimensions: Int,
            heads: Int,
            windowSize: Int,
            mlpRatio: Float,
            qkvBias: Bool,
            convAtAttention: Bool,
            convAtFFN: Bool,
        ) {
            if convAtAttention {
                _conv1.wrappedValue = ConvFunction(
                    dimensions: dimensions,
                    kernelSize: 3,
                    padding: 1,
                    stride: 1,
                )
            }
            _windowAttention.wrappedValue = NormedWindowAttention(
                dimensions: dimensions,
                heads: heads,
                windowSize: windowSize,
                qkvBias: qkvBias,
            )
            if convAtFFN {
                _conv2.wrappedValue = ConvFunction(
                    dimensions: dimensions,
                    kernelSize: 3,
                    padding: 1,
                    stride: 1,
                )
            }
            _ffn.wrappedValue = NormedMLP(
                dimensions: dimensions,
                hiddenDimensions: Int(Float(dimensions) * mlpRatio),
            )
        }

        func callAsFunction(_ x: MLXArray, size: (Int, Int)) -> (MLXArray, (Int, Int)) {
            var x = x
            var size = size
            if let conv1 {
                let result = conv1(x, size: size)
                x = x + result.0
                size = result.1
            }
            var result = windowAttention(x, size: size)
            x = x + result.0
            size = result.1
            if let conv2 {
                result = conv2(x, size: size)
                x = x + result.0
                size = result.1
            }
            result = ffn(x, size: size)
            x = x + result.0
            size = result.1
            return (x, size)
        }
    }

    final class ChannelBlock: Module {
        @ModuleInfo var conv1: ConvFunction?
        @ModuleInfo(key: "channel_attn") var channelAttention: NormedChannelAttention
        @ModuleInfo var conv2: ConvFunction?
        @ModuleInfo var ffn: NormedMLP

        init(
            dimensions: Int,
            groups: Int,
            mlpRatio: Float,
            qkvBias: Bool,
            convAtAttention: Bool,
            convAtFFN: Bool,
        ) {
            if convAtAttention {
                _conv1.wrappedValue = ConvFunction(
                    dimensions: dimensions,
                    kernelSize: 3,
                    padding: 1,
                    stride: 1,
                )
            }
            _channelAttention.wrappedValue = NormedChannelAttention(
                dimensions: dimensions,
                groups: groups,
                qkvBias: qkvBias,
            )
            if convAtFFN {
                _conv2.wrappedValue = ConvFunction(
                    dimensions: dimensions,
                    kernelSize: 3,
                    padding: 1,
                    stride: 1,
                )
            }
            _ffn.wrappedValue = NormedMLP(
                dimensions: dimensions,
                hiddenDimensions: Int(Float(dimensions) * mlpRatio),
            )
        }

        func callAsFunction(_ x: MLXArray, size: (Int, Int)) -> (MLXArray, (Int, Int)) {
            var x = x
            var size = size
            if let conv1 {
                let result = conv1(x, size: size)
                x = x + result.0
                size = result.1
            }
            var result = channelAttention(x, size: size)
            x = x + result.0
            size = result.1
            if let conv2 {
                result = conv2(x, size: size)
                x = x + result.0
                size = result.1
            }
            result = ffn(x, size: size)
            x = x + result.0
            size = result.1
            return (x, size)
        }
    }

    final class NormedWindowAttention: Module {
        @ModuleInfo var norm: LayerNorm
        @ModuleInfo(key: "fn") var function: WindowAttention

        init(dimensions: Int, heads: Int, windowSize: Int, qkvBias: Bool) {
            _norm.wrappedValue = LayerNorm(dimensions: dimensions)
            _function.wrappedValue = WindowAttention(
                dimensions: dimensions,
                heads: heads,
                windowSize: windowSize,
                qkvBias: qkvBias,
            )
        }

        func callAsFunction(_ x: MLXArray, size: (Int, Int)) -> (MLXArray, (Int, Int)) {
            function(norm(x), size: size)
        }
    }

    final class NormedChannelAttention: Module {
        @ModuleInfo var norm: LayerNorm
        @ModuleInfo(key: "fn") var function: ChannelAttention

        init(dimensions: Int, groups: Int, qkvBias: Bool) {
            _norm.wrappedValue = LayerNorm(dimensions: dimensions)
            _function.wrappedValue = ChannelAttention(
                dimensions: dimensions,
                groups: groups,
                qkvBias: qkvBias,
            )
        }

        func callAsFunction(_ x: MLXArray, size: (Int, Int)) -> (MLXArray, (Int, Int)) {
            function(norm(x), size: size)
        }
    }

    final class NormedMLP: Module {
        @ModuleInfo var norm: LayerNorm
        @ModuleInfo(key: "fn") var function: MLPBlock

        init(dimensions: Int, hiddenDimensions: Int) {
            _norm.wrappedValue = LayerNorm(dimensions: dimensions)
            _function.wrappedValue = MLPBlock(
                dimensions: dimensions,
                hiddenDimensions: hiddenDimensions,
            )
        }

        func callAsFunction(_ x: MLXArray, size: (Int, Int)) -> (MLXArray, (Int, Int)) {
            function(norm(x), size: size)
        }
    }

    final class Block: Module {
        @ModuleInfo(key: "spatial_block") var spatialBlock: SpatialBlock
        @ModuleInfo(key: "channel_block") var channelBlock: ChannelBlock

        init(
            dimensions: Int,
            heads: Int,
            groups: Int,
            windowSize: Int,
            mlpRatio: Float,
            qkvBias: Bool,
            convAtAttention: Bool,
            convAtFFN: Bool,
        ) {
            _spatialBlock.wrappedValue = SpatialBlock(
                dimensions: dimensions,
                heads: heads,
                windowSize: windowSize,
                mlpRatio: mlpRatio,
                qkvBias: qkvBias,
                convAtAttention: convAtAttention,
                convAtFFN: convAtFFN,
            )
            _channelBlock.wrappedValue = ChannelBlock(
                dimensions: dimensions,
                groups: groups,
                mlpRatio: mlpRatio,
                qkvBias: qkvBias,
                convAtAttention: convAtAttention,
                convAtFFN: convAtFFN,
            )
        }

        func callAsFunction(_ x: MLXArray, size: (Int, Int)) -> (MLXArray, (Int, Int)) {
            var result = spatialBlock(x, size: size)
            result = channelBlock(result.0, size: result.1)
            return result
        }
    }

    final class VisionModel: Module {
        let blockOffsets: [Int]
        let stageBlockRanges: [Range<Int>]
        @ModuleInfo var convs: [ConvEmbed]
        @ModuleInfo var blocks: [Block]

        init(_ config: FlorenceConfiguration.VisionConfiguration) {
            var convs: [ConvEmbed] = []
            var blocks: [Block] = []
            var blockOffsets: [Int] = []
            var stageBlockRanges: [Range<Int>] = []

            for stage in 0 ..< config.embedDimensions.count {
                convs.append(
                    ConvEmbed(
                        patchSize: config.patchSizes[stage],
                        inputChannels: stage == 0
                            ? config.channels : config.embedDimensions[stage - 1],
                        embedDimensions: config.embedDimensions[stage],
                        stride: config.patchStrides[stage],
                        padding: config.patchPadding[stage],
                        preNorm: config.patchPreNorm[stage],
                    ),
                )

                let start = blocks.count
                blockOffsets.append(start)
                for _ in 0 ..< config.depths[stage] {
                    blocks.append(
                        Block(
                            dimensions: config.embedDimensions[stage],
                            heads: config.attentionHeads[stage],
                            groups: config.attentionGroups[stage],
                            windowSize: config.windowSize,
                            mlpRatio: config.mlpRatio,
                            qkvBias: config.qkvBias,
                            convAtAttention: config.convAtAttention,
                            convAtFFN: config.convAtFFN,
                        ),
                    )
                }
                stageBlockRanges.append(start ..< blocks.count)
            }

            self.blockOffsets = blockOffsets
            self.stageBlockRanges = stageBlockRanges
            _convs.wrappedValue = convs
            _blocks.wrappedValue = blocks
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var x = x
            var size = (x.dim(1), x.dim(2))
            for index in 0 ..< convs.count {
                var result = convs[index](x, size: size)
                x = result.0
                size = result.1
                for blockIndex in stageBlockRanges[index] {
                    result = blocks[blockIndex](x, size: size)
                    x = result.0
                    size = result.1
                }
            }
            return x
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized: [String: MLXArray] = [:]
            for (key, value) in weights {
                let key = remappedBlockKey(key)
                if key.contains("position_ids") {
                    continue
                }
                if key.hasPrefix("vision_tower.convs"), key.hasSuffix("proj.weight"),
                    value.ndim == 4
                {
                    sanitized[key] = Self.sanitizeConvWeight(value)
                } else if key.hasPrefix("vision_tower.blocks"), key.hasSuffix("dw.weight"),
                    value.ndim == 4
                {
                    sanitized[key] =
                        value.shape[1] < value.shape[value.shape.count - 1]
                        ? value.transposed(0, 2, 3, 1)
                        : value
                } else {
                    sanitized[key] = value
                }
            }
            return sanitized
        }

        private func remappedBlockKey(_ key: String) -> String {
            var parts = key.split(separator: ".", omittingEmptySubsequences: false)
                .map(String.init)
            guard parts.count > 4,
                parts[0] == "vision_tower",
                parts[1] == "blocks",
                let stageIndex = Int(parts[2]),
                let blockIndex = Int(parts[3]),
                stageIndex < blockOffsets.count
            else {
                return key
            }
            parts[2] = String(blockOffsets[stageIndex] + blockIndex)
            parts.remove(at: 3)
            return parts.joined(separator: ".")
        }

        private static func sanitizeConvWeight(_ value: MLXArray) -> MLXArray {
            let shape = value.shape
            guard shape.count == 4 else {
                return value
            }
            let outChannels = shape[0]
            let kH = shape[1]
            let kW = shape[2]
            if outChannels >= kH, outChannels >= kW, kH == kW {
                return value
            }
            return value.transposed(0, 2, 3, 1)
        }
    }
}

// MARK: - Language

private enum FlorenceLanguage {
    final class Attention: Module {
        let embedDimensions: Int
        let heads: Int
        let headDimensions: Int
        let scale: Float

        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(embedDimensions: Int, heads: Int) {
            self.embedDimensions = embedDimensions
            self.heads = heads
            headDimensions = embedDimensions / heads
            scale = pow(Float(headDimensions), -0.5)
            _kProj.wrappedValue = Linear(embedDimensions, embedDimensions, bias: true)
            _vProj.wrappedValue = Linear(embedDimensions, embedDimensions, bias: true)
            _qProj.wrappedValue = Linear(embedDimensions, embedDimensions, bias: true)
            _outProj.wrappedValue = Linear(embedDimensions, embedDimensions, bias: true)
        }

        func callAsFunction(
            hiddenStates: MLXArray,
            keyValueStates: MLXArray? = nil,
            cache: KVCache? = nil,
            mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        )
            -> MLXArray
        {
            let batchSize = hiddenStates.dim(0)
            let targetLength = hiddenStates.dim(1)

            let queries = qProj(hiddenStates)
                .reshaped(batchSize, targetLength, heads, headDimensions)
                .transposed(0, 2, 1, 3)

            let sourceStates = keyValueStates ?? hiddenStates
            let sourceLength = sourceStates.dim(1)
            let keys = kProj(sourceStates)
                .reshaped(batchSize, sourceLength, heads, headDimensions)
                .transposed(0, 2, 1, 3)
            let values = vProj(sourceStates)
                .reshaped(batchSize, sourceLength, heads, headDimensions)
                .transposed(0, 2, 1, 3)

            let output: MLXArray =
                if keyValueStates == nil {
                    attentionWithCacheUpdate(
                        queries: queries,
                        keys: keys,
                        values: values,
                        cache: cache,
                        scale: scale,
                        mask: mask,
                    )
                } else {
                    MLXFast.scaledDotProductAttention(
                        queries: queries,
                        keys: keys,
                        values: values,
                        scale: scale,
                        mask: .none,
                    )
                }

            return outProj(
                output
                    .transposed(0, 2, 1, 3)
                    .reshaped(batchSize, targetLength, embedDimensions),
            )
        }
    }

    final class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo(key: "self_attn_layer_norm") var selfAttentionLayerNorm: LayerNorm
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm
        let activation = GELU()

        init(_ config: FlorenceConfiguration.TextConfiguration) {
            _selfAttention.wrappedValue = Attention(
                embedDimensions: config.modelDimension,
                heads: config.encoderAttentionHeads,
            )
            _selfAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: config.modelDimension)
            _fc1.wrappedValue = Linear(
                config.modelDimension,
                config.encoderFFNDimension,
                bias: true,
            )
            _fc2.wrappedValue = Linear(
                config.encoderFFNDimension,
                config.modelDimension,
                bias: true,
            )
            _finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.modelDimension)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var hiddenStates = x + selfAttention(hiddenStates: x)
            hiddenStates = selfAttentionLayerNorm(hiddenStates)
            hiddenStates = hiddenStates + fc2(activation(fc1(hiddenStates)))
            return finalLayerNorm(hiddenStates)
        }
    }

    final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo(key: "self_attn_layer_norm") var selfAttentionLayerNorm: LayerNorm
        @ModuleInfo(key: "encoder_attn") var encoderAttention: Attention
        @ModuleInfo(key: "encoder_attn_layer_norm") var encoderAttentionLayerNorm: LayerNorm
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm
        let activation = GELU()

        init(_ config: FlorenceConfiguration.TextConfiguration) {
            _selfAttention.wrappedValue = Attention(
                embedDimensions: config.modelDimension,
                heads: config.decoderAttentionHeads,
            )
            _selfAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: config.modelDimension)
            _encoderAttention.wrappedValue = Attention(
                embedDimensions: config.modelDimension,
                heads: config.decoderAttentionHeads,
            )
            _encoderAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: config.modelDimension)
            _fc1.wrappedValue = Linear(
                config.modelDimension,
                config.decoderFFNDimension,
                bias: true,
            )
            _fc2.wrappedValue = Linear(
                config.decoderFFNDimension,
                config.modelDimension,
                bias: true,
            )
            _finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.modelDimension)
        }

        func callAsFunction(
            _ x: MLXArray,
            encoderHiddenStates: MLXArray,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let selfMask = createAttentionMask(h: x, cache: cache)
            var hiddenStates =
                x
                + selfAttention(
                    hiddenStates: x,
                    cache: cache,
                    mask: selfMask,
                )
            hiddenStates = selfAttentionLayerNorm(hiddenStates)
            hiddenStates =
                hiddenStates
                + encoderAttention(
                    hiddenStates: hiddenStates,
                    keyValueStates: encoderHiddenStates,
                )
            hiddenStates = encoderAttentionLayerNorm(hiddenStates)
            hiddenStates = hiddenStates + fc2(activation(fc1(hiddenStates)))
            return finalLayerNorm(hiddenStates)
        }
    }

    final class Encoder: Module {
        let config: FlorenceConfiguration.TextConfiguration
        let offset = 2
        @ModuleInfo(key: "embed_positions") var embedPositions: Embedding
        @ModuleInfo var layers: [EncoderLayer]
        @ModuleInfo(key: "layernorm_embedding") var layerNormEmbedding: LayerNorm

        init(_ config: FlorenceConfiguration.TextConfiguration) {
            self.config = config
            _embedPositions.wrappedValue = Embedding(
                embeddingCount: config.maxPositionEmbeddings + offset,
                dimensions: config.modelDimension,
            )
            _layers.wrappedValue = (0 ..< config.encoderLayers).map { _ in
                EncoderLayer(config)
            }
            _layerNormEmbedding.wrappedValue = LayerNorm(dimensions: config.modelDimension)
        }

        func callAsFunction(inputsEmbeds: MLXArray) -> MLXArray {
            let sequenceLength = inputsEmbeds.dim(1)
            let positions = (MLXArray(0 ..< sequenceLength) + offset).expandedDimensions(axis: 0)
            var hiddenStates = inputsEmbeds + embedPositions(positions)
            hiddenStates = layerNormEmbedding(hiddenStates)
            for layer in layers {
                hiddenStates = layer(hiddenStates)
            }
            return hiddenStates
        }
    }

    final class Decoder: Module {
        let config: FlorenceConfiguration.TextConfiguration
        let offset = 2
        @ModuleInfo(key: "embed_positions") var embedPositions: Embedding
        @ModuleInfo var layers: [DecoderLayer]
        @ModuleInfo(key: "layernorm_embedding") var layerNormEmbedding: LayerNorm

        init(_ config: FlorenceConfiguration.TextConfiguration) {
            self.config = config
            _embedPositions.wrappedValue = Embedding(
                embeddingCount: config.maxPositionEmbeddings + offset,
                dimensions: config.modelDimension,
            )
            _layers.wrappedValue = (0 ..< config.decoderLayers).map { _ in
                DecoderLayer(config)
            }
            _layerNormEmbedding.wrappedValue = LayerNorm(dimensions: config.modelDimension)
        }

        func callAsFunction(
            inputsEmbeds: MLXArray,
            encoderHiddenStates: MLXArray,
            cache: [KVCache]?,
        )
            -> MLXArray
        {
            let embeddings = inputsEmbeds
            let sequenceLength = embeddings.dim(1)
            let cacheLength = cache?.first?.offset ?? 0
            let positions =
                (MLXArray(cacheLength ..< (cacheLength + sequenceLength)) + offset)
                .expandedDimensions(axis: 0)
            var hiddenStates = embeddings + embedPositions(positions)
            hiddenStates = layerNormEmbedding(hiddenStates)
            for (index, layer) in layers.enumerated() {
                hiddenStates = layer(
                    hiddenStates,
                    encoderHiddenStates: encoderHiddenStates,
                    cache: cache?[index],
                )
            }
            return hiddenStates
        }
    }

    final class Model: Module {
        let config: FlorenceConfiguration.TextConfiguration
        @ModuleInfo var shared: Embedding
        @ModuleInfo var encoder: Encoder
        @ModuleInfo var decoder: Decoder
        let embedScale: Float

        init(_ config: FlorenceConfiguration.TextConfiguration) {
            self.config = config
            embedScale = config.scaleEmbedding ? sqrt(Float(config.modelDimension)) : 1
            let shared = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.modelDimension,
            )
            _shared.wrappedValue = shared
            _encoder.wrappedValue = Encoder(config)
            _decoder.wrappedValue = Decoder(config)
        }

        func embedTokens(_ inputIDs: MLXArray) -> MLXArray {
            shared(inputIDs) * embedScale
        }

        func encode(inputsEmbeds: MLXArray) -> MLXArray {
            encoder(inputsEmbeds: inputsEmbeds * embedScale)
        }

        func decode(
            inputIDs: MLXArray?,
            inputsEmbeds: MLXArray? = nil,
            encoderHiddenStates: MLXArray,
            cache: [KVCache]?,
        )
            -> MLXArray
        {
            let decoderInputsEmbeds: MLXArray
            if let inputsEmbeds {
                decoderInputsEmbeds = inputsEmbeds
            } else if let inputIDs {
                decoderInputsEmbeds = shared(Self.normalizedDecoderInputIDs(inputIDs))
            } else {
                preconditionFailure("decoder input ids or embeddings are required")
            }
            return decoder(
                inputsEmbeds: decoderInputsEmbeds,
                encoderHiddenStates: encoderHiddenStates,
                cache: cache,
            )
        }

        private static func normalizedDecoderInputIDs(_ inputIDs: MLXArray) -> MLXArray {
            if inputIDs.ndim == 0 {
                return inputIDs[.newAxis, .newAxis]
            }
            if inputIDs.ndim == 1 {
                return inputIDs.expandedDimensions(axis: 0)
            }
            if inputIDs.ndim == 2 {
                return inputIDs
            }
            return inputIDs.reshaped(inputIDs.dim(0), -1)
        }
    }

    final class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?
        let config: FlorenceConfiguration.TextConfiguration

        init(_ config: FlorenceConfiguration.TextConfiguration) {
            self.config = config
            _model.wrappedValue = Model(config)
            _lmHead.wrappedValue = Linear(
                config.modelDimension,
                config.vocabularySize,
                bias: false,
            )
        }

        var kvHeads: [Int] {
            Array(repeating: config.decoderAttentionHeads, count: config.decoderLayers)
        }

        func logits(_ hiddenStates: MLXArray) -> MLXArray {
            if let lmHead {
                return lmHead(hiddenStates)
            }
            return model.shared.asLinear(hiddenStates)
        }
    }
}

// MARK: - Position Embeddings

private final class FlorenceLearnedPositionEmbedding2D: Module {
    @ModuleInfo(key: "row_embeddings") var rowEmbeddings: Embedding
    @ModuleInfo(key: "column_embeddings") var columnEmbeddings: Embedding

    init(embeddingDimensions: Int, positions: Int) {
        _rowEmbeddings.wrappedValue = Embedding(
            embeddingCount: positions,
            dimensions: embeddingDimensions / 2,
        )
        _columnEmbeddings.wrappedValue = Embedding(
            embeddingCount: positions,
            dimensions: embeddingDimensions - (embeddingDimensions / 2),
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batchSize = x.dim(0)
        let height = x.dim(1)
        let width = x.dim(2)
        let channels = x.dim(3)

        let xEmbeddings = columnEmbeddings(MLXArray(0 ..< width))
        let yEmbeddings = rowEmbeddings(MLXArray(0 ..< height))
        let position = concatenated(
            [
                broadcast(
                    xEmbeddings.expandedDimensions(axis: 0),
                    to: [height, width, xEmbeddings.dim(1)],
                ),
                broadcast(
                    yEmbeddings.expandedDimensions(axis: 1),
                    to: [height, width, yEmbeddings.dim(1)],
                ),
            ],
            axis: -1,
        )
        return broadcast(
            position.expandedDimensions(axis: 0),
            to: [batchSize, height, width, channels],
        )
    }
}

private final class FlorencePositionalEmbeddingCosine1D: Module {
    @ModuleInfo(key: "pos_idx_to_embed") var positionIndexToEmbedding: MLXArray

    init(embeddingDimensions: Int, maxSequenceLength: Int) {
        var values: [Float] = []
        values.reserveCapacity(maxSequenceLength * embeddingDimensions)
        let halfDimensions = embeddingDimensions / 2
        for position in 0 ..< maxSequenceLength {
            for index in 0 ..< halfDimensions {
                let denominator = Foundation.exp(
                    -Foundation.log(10000) * Double(index) / Double(embeddingDimensions),
                )
                let frequency = Double(position) * denominator
                values.append(Float(Foundation.sin(frequency)))
                values.append(Float(Foundation.cos(frequency)))
            }
        }
        _positionIndexToEmbedding.wrappedValue = MLXArray(values)
            .reshaped(maxSequenceLength, embeddingDimensions)
    }

    func callAsFunction(_ sequenceEmbeds: MLXArray) -> MLXArray {
        let sequenceLength = sequenceEmbeds.dim(sequenceEmbeds.ndim - 2)
        var embeddings = positionIndexToEmbedding[..<sequenceLength, 0...]
        if sequenceEmbeds.ndim == 3 {
            embeddings = embeddings.expandedDimensions(axis: 0)
        }
        return embeddings
    }
}

// MARK: - Model

public final class Florence: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: FlorenceVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: FlorenceLanguage.LanguageModel
    @ModuleInfo(key: "image_projection") private var imageProjection: MLXArray
    @ModuleInfo(key: "image_proj_norm") private var imageProjectionNorm: LayerNorm
    @ModuleInfo(key: "image_pos_embed") private var imagePositionEmbedding:
        FlorenceLearnedPositionEmbedding2D
    @ModuleInfo(key: "visual_temporal_embed") private var visualTemporalEmbedding:
        FlorencePositionalEmbeddingCosine1D

    public let config: FlorenceConfiguration
    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    private var pendingEncoderHiddenStates: MLXArray?

    public init(_ config: FlorenceConfiguration) {
        self.config = config
        _visionTower.wrappedValue = FlorenceVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = FlorenceLanguage.LanguageModel(
            config.textConfiguration,
        )
        let imageDimensions = config.visionConfiguration.embedDimensions.last ?? 1024
        let textDimensions = config.textConfiguration.modelDimension
        _imageProjection.wrappedValue = MLXArray.zeros([imageDimensions, textDimensions])
        _imageProjectionNorm.wrappedValue = LayerNorm(dimensions: textDimensions)
        _imagePositionEmbedding.wrappedValue = FlorenceLearnedPositionEmbedding2D(
            embeddingDimensions: imageDimensions,
            positions: config.visionConfiguration.imagePositionEmbedding?.maxPositionEmbeddings
                ?? 50,
        )
        _visualTemporalEmbedding.wrappedValue = FlorencePositionalEmbeddingCosine1D(
            embeddingDimensions: imageDimensions,
            maxSequenceLength: config.visionConfiguration.visualTemporalEmbedding?
                .maxTemporalEmbeddings ?? 100,
        )
    }

    public var loraLayers: [Module] {
        languageModel.model.decoder.layers
    }

    private func encodeImage(_ pixelValues: MLXArray) -> MLXArray {
        let temporalLength = 1
        let batchSize = pixelValues.dim(0)
        var x = visionTower(pixelValues.transposed(0, 2, 3, 1))

        x = x.reshaped(batchSize * temporalLength, -1, x.dim(-1))
        let tokenCount = x.dim(1)
        let side = Int(Double(tokenCount).squareRoot())
        precondition(side * side == tokenCount, "Florence only supports square feature maps")
        x = x.reshaped(batchSize * temporalLength, side, side, x.dim(-1))
        x = x + imagePositionEmbedding(x)
        x = x.reshaped(batchSize, temporalLength * side * side, x.dim(-1))

        let temporalView = x.reshaped(batchSize, temporalLength, -1, x.dim(-1))
        let temporalEmbedding = visualTemporalEmbedding(temporalView[0..., 0..., 0, 0...])
            .expandedDimensions(axis: 2)
        x = temporalView + temporalEmbedding

        let spatialAveragePool = mean(x, axis: 2)
        let temporalAveragePool = mean(x, axis: 1)
        let lastFrame = x[0..., -1, 0..., 0...]

        var featureSources: [MLXArray] = []
        for source in config.imageFeatureSource {
            switch source {
            case "spatial_avg_pool":
                featureSources.append(spatialAveragePool)
            case "temporal_avg_pool":
                featureSources.append(temporalAveragePool)
            case "last_frame":
                featureSources.append(lastFrame)
            default:
                preconditionFailure("invalid Florence image feature source \(source)")
            }
        }

        x = concatenated(featureSources, axis: 1)
        x = matmul(x, imageProjection.asType(x.dtype))
        return imageProjectionNorm(x)
    }

    private func inputEmbeddings(inputIDs: MLXArray, pixelValues: MLXArray?) -> MLXArray {
        let ids = inputIDs[0].asArray(Int.self).filter { $0 != config.imageTokenID }
        let taskInputIDs = MLXArray(ids).expandedDimensions(axis: 0)
        let taskEmbeddings =
            ids.isEmpty
            ? MLXArray.zeros([1, 0, config.textConfiguration.modelDimension])
            : languageModel.model.embedTokens(taskInputIDs)

        guard let pixelValues else {
            return taskEmbeddings
        }

        let imageFeatures = encodeImage(pixelValues).asType(taskEmbeddings.dtype)
        return concatenated([imageFeatures, taskEmbeddings], axis: 1)
    }

    public func prepare(
        _ input: LMInput, cache _: [any KVCache], state _: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let embeddings = inputEmbeddings(
            inputIDs: input.text.tokens,
            pixelValues: input.image?.pixels,
        )
        let encoderHiddenStates = languageModel.model.encode(inputsEmbeds: embeddings)
        let decoderStartTokens = MLXArray([
            config.textConfiguration.decoderStartTokenID,
            config.textConfiguration.bosTokenID,
        ])
        pendingEncoderHiddenStates = encoderHiddenStates
        return .tokens(.init(tokens: decoderStartTokens, mask: ones(like: decoderStartTokens)))
    }

    public func callAsFunction(
        _ input: LMInput.Text,
        cache: [any KVCache]?,
        state: LMOutput.State?,
    )
        -> LMOutput
    {
        let encoderHiddenStates: MLXArray
        if let stateEncoderHiddenStates = state?[florenceCrossAttentionStatesKey] {
            encoderHiddenStates = stateEncoderHiddenStates
        } else if let pendingEncoderHiddenStates {
            encoderHiddenStates = pendingEncoderHiddenStates
            self.pendingEncoderHiddenStates = nil
        } else {
            preconditionFailure("Florence decoder requires encoder hidden states")
        }
        let decoderHiddenStates = languageModel.model.decode(
            inputIDs: input.tokens,
            encoderHiddenStates: encoderHiddenStates,
            cache: cache,
        )
        var outputState = LMOutput.State()
        outputState[florenceCrossAttentionStatesKey] = encoderHiddenStates
        return LMOutput(logits: languageModel.logits(decoderHiddenStates), state: outputState)
    }

    public func callAsFunction(_: MLXArray, cache _: [any KVCache]?) -> MLXArray {
        preconditionFailure("Florence requires encoder state; use callAsFunction(_:cache:state:)")
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = visionTower.sanitize(weights: weights).filter { key, _ in
            !key.contains("final_logits_bias")
        }
        if sanitized["language_model.lm_head.weight"] == nil,
            let shared = sanitized["language_model.model.shared.weight"]
        {
            sanitized["language_model.lm_head.weight"] = shared
        }
        return sanitized
    }
}
