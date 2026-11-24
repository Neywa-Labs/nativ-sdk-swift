import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct DeepSeekVLConfiguration: Codable, Sendable {
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

        private let _rmsNormEps: Float?
        public var rmsNormEps: Float {
            _rmsNormEps ?? 1e-6
        }

        public let vocabularySize: Int
        private let _ropeTheta: Float?
        public var ropeTheta: Float {
            _ropeTheta ?? 10000
        }

        private let _ropeTraditional: Bool?
        public var ropeTraditional: Bool {
            _ropeTraditional ?? false
        }

        public let ropeScaling: [String: StringOrNumber]?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case _kvHeads = "num_key_value_heads"
            case _rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case _ropeTheta = "rope_theta"
            case _ropeTraditional = "rope_traditional"
            case ropeScaling = "rope_scaling"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public struct Parameters: Codable, Sendable {
            private let _imageSize: Int?
            public var imageSize: Int {
                _imageSize ?? 384
            }

            public let pixelMean: [CGFloat]?
            public let pixelStd: [CGFloat]?
            public let selectFeature: String?
            public let selectLayer: Int?

            enum CodingKeys: String, CodingKey {
                case _imageSize = "image_size"
                case pixelMean = "pixel_mean"
                case pixelStd = "pixel_std"
                case selectFeature = "select_feature"
                case selectLayer = "select_layer"
            }
        }

        public let modelType: String
        private let _hiddenSize: Int?
        public var hiddenSize: Int {
            _hiddenSize ?? 1024
        }

        private let _hiddenLayers: Int?
        public var hiddenLayers: Int {
            _hiddenLayers ?? 24
        }

        private let _intermediateSize: Int?
        public var intermediateSize: Int {
            _intermediateSize ?? 4096
        }

        private let _attentionHeads: Int?
        public var attentionHeads: Int {
            _attentionHeads ?? 16
        }

        private let _patchSize: Int?
        public var patchSize: Int {
            _patchSize ?? 16
        }

        private let _channels: Int?
        public var channels: Int {
            _channels ?? 3
        }

        private let _layerNormEps: Float?
        public var layerNormEps: Float {
            _layerNormEps ?? 1e-5
        }

        public let cls: String?
        public let parameters: Parameters?
        public var imageSize: Int {
            parameters?.imageSize ?? 384
        }

        public var imageMean: [CGFloat] {
            parameters?.pixelMean ?? [0.5, 0.5, 0.5]
        }

        public var imageStd: [CGFloat] {
            parameters?.pixelStd ?? [0.5, 0.5, 0.5]
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case _hiddenSize = "hidden_size"
            case _hiddenLayers = "num_hidden_layers"
            case _intermediateSize = "intermediate_size"
            case _attentionHeads = "num_attention_heads"
            case _patchSize = "patch_size"
            case _channels = "num_channels"
            case _layerNormEps = "layer_norm_eps"
            case cls
            case parameters = "params"
        }
    }

    public struct ProjectorConfiguration: Codable, Sendable {
        public struct Parameters: Codable, Sendable {
            public let depth: Int?
            public let projectorType: String?

            enum CodingKeys: String, CodingKey {
                case depth
                case projectorType = "projector_type"
            }
        }

        public let cls: String
        public let modelType: String
        public let parameters: Parameters

        enum CodingKeys: String, CodingKey {
            case cls
            case modelType = "model_type"
            case parameters = "params"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    private let _projectorConfiguration: ProjectorConfiguration?
    private let _alignerConfiguration: ProjectorConfiguration?
    public var projectorConfiguration: ProjectorConfiguration {
        _projectorConfiguration ?? _alignerConfiguration!
    }

    public let modelType: String
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int {
        _imageTokenIndex ?? 100_015
    }

    private let _numImageTokens: Int?
    public var numImageTokens: Int {
        _numImageTokens ?? 576
    }

    private let _selectLayer: Int?
    public var selectLayer: Int {
        _selectLayer ?? -1
    }

    private let _visionFeatureSelectStrategy: String?
    public var visionFeatureSelectStrategy: String {
        _visionFeatureSelectStrategy ?? "default"
    }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case _projectorConfiguration = "projector_config"
        case _alignerConfiguration = "aligner_config"
        case modelType = "model_type"
        case _imageTokenIndex = "image_token_index"
        case _numImageTokens = "num_image_tokens"
        case _selectLayer = "select_layer"
        case _visionFeatureSelectStrategy = "vision_feature_select_strategy"
    }
}

public struct DeepSeekVLProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String?
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let backgroundColor: [CGFloat]
    private let _imageSize: Int?
    public var imageSize: Int {
        _imageSize ?? 384
    }

    private let _minSize: Int?
    public var minSize: Int {
        _minSize ?? 14
    }

    private let _rescaleFactor: CGFloat?
    public var rescaleFactor: CGFloat {
        _rescaleFactor ?? (1 / 255)
    }

    private let _doNormalize: Bool?
    public var doNormalize: Bool {
        _doNormalize ?? true
    }

    private let _numImageTokens: Int?
    public var numImageTokens: Int {
        _numImageTokens ?? 576
    }

    private let _imageToken: String?
    public var imageToken: String {
        _imageToken ?? "<image>"
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case backgroundColor = "background_color"
        case _imageSize = "image_size"
        case _minSize = "min_size"
        case _rescaleFactor = "rescale_factor"
        case _doNormalize = "do_normalize"
        case _numImageTokens = "num_image_tokens"
        case _imageToken = "image_token"
        case _imageTag = "image_tag"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass = try c.decodeIfPresent(String.self, forKey: .processorClass)
        imageMean = try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd = try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        backgroundColor =
            try c.decodeIfPresent([CGFloat].self, forKey: .backgroundColor)
            ?? imageMean.map { $0 * 255 }
        _imageSize = try c.decodeIfPresent(Int.self, forKey: ._imageSize)
        _minSize = try c.decodeIfPresent(Int.self, forKey: ._minSize)
        _rescaleFactor = try c.decodeIfPresent(CGFloat.self, forKey: ._rescaleFactor)
        _doNormalize = try c.decodeIfPresent(Bool.self, forKey: ._doNormalize)
        _numImageTokens = try c.decodeIfPresent(Int.self, forKey: ._numImageTokens)
        _imageToken =
            try c.decodeIfPresent(String.self, forKey: ._imageToken)
            ?? c.decodeIfPresent(String.self, forKey: ._imageTag)
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    public var backgroundCIColor: CIColor {
        let scale: CGFloat = backgroundColor.contains { $0 > 1 } ? 255 : 1
        return CIColor(
            red: backgroundColor[0] / scale,
            green: backgroundColor[1] / scale,
            blue: backgroundColor[2] / scale,
        )
    }
}

// MARK: - Processor

public struct DeepSeekVLProcessor: UserInputProcessor {
    private let config: DeepSeekVLProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageTokenID: Int

    public init(_ config: DeepSeekVLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        imageTokenID =
            tokenizer.convertTokenToId(config.imageToken)
            ?? tokenizer.encode(text: config.imageToken, addSpecialTokens: false).first
            ?? 100_015
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        let prompt = Self.prompt(from: input)
        let promptText = try Self.promptWithImagePlaceholders(
            prompt,
            imageCount: images.count,
            imageToken: config.imageToken,
        )
        let tokens = tokenizedPrompt(renderedPrompt(promptText), imageCount: images.count)
        let tokensArray = MLXArray(tokens).expandedDimensions(axis: 0)

        guard !images.isEmpty else {
            return LMInput(text: .init(tokens: tokensArray, mask: ones(like: tokensArray)))
        }

        let imageArrays = try images.map { try preprocess($0) }
        let pixels = concatenated(imageArrays, axis: 0)
        return LMInput(
            text: .init(tokens: tokensArray, mask: ones(like: tokensArray)),
            image: .init(pixels: pixels),
        )
    }

    private func renderedPrompt(_ prompt: String) -> String {
        let system =
            "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language."
        let bos = tokenizer.bosToken ?? "<\u{ff5c}begin\u{2581}of\u{2581}sentence\u{ff5c}>"
        return
            "\(bos)\(system)\n\nUser: \(prompt.trimmingCharacters(in: .whitespacesAndNewlines))\n\nAssistant:"
    }

    private func tokenizedPrompt(_ text: String, imageCount: Int) -> [Int] {
        var remaining = text[...]
        var tokens: [Int] = []

        for _ in 0 ..< imageCount {
            guard let range = remaining.range(of: config.imageToken) else {
                tokens += imageTokenRun()
                continue
            }
            tokens += tokenizer.encode(
                text: String(remaining[..<range.lowerBound]), addSpecialTokens: false)
            tokens += imageTokenRun()
            remaining = remaining[range.upperBound...]
        }

        tokens += tokenizer.encode(text: String(remaining), addSpecialTokens: false)
        return tokens
    }

    private func imageTokenRun() -> [Int] {
        Array(repeating: imageTokenID, count: config.numImageTokens)
    }

    private func preprocess(_ inputImage: UserInput.Image) throws -> MLXArray {
        let image = try inputImage.asCIImage().toSRGB()
        let resized = image.resampled(
            to: targetSize(for: image.extent.size),
            method: .bicubic,
        )
        var processed =
            resized
            .paddingToSquare(backgroundColor: config.backgroundCIColor)
            .resampled(
                to: CGSize(width: config.imageSize, height: config.imageSize),
                method: .bicubic,
            )
        if config.doNormalize {
            processed = processed.normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
        }
        return processed.asMLXArray()
    }

    private func targetSize(for size: CGSize) -> CGSize {
        let width = max(size.width, 1)
        let height = max(size.height, 1)
        let maxSide = max(width, height)
        let imageSize = CGFloat(config.imageSize)
        let minSize = CGFloat(config.minSize)
        let targetWidth = max(floor(width / maxSide * imageSize), minSize)
        let targetHeight = max(floor(height / maxSide * imageSize), minSize)
        return CGSize(width: targetWidth, height: targetHeight)
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

    private static func promptWithImagePlaceholders(
        _ prompt: String,
        imageCount: Int,
        imageToken: String,
    )
        throws -> String
    {
        let existing = prompt.components(separatedBy: imageToken).count - 1
        guard existing <= imageCount else {
            throw VLMError.processing(
                "Prompt contains \(existing) image placeholders but only \(imageCount) image inputs.",
            )
        }
        guard existing < imageCount else {
            return prompt
        }
        return String(repeating: "\(imageToken)\n", count: imageCount - existing) + prompt
    }
}

// MARK: - Language

private enum DeepSeekVLLanguage {
    final class Attention: Module {
        let config: DeepSeekVLConfiguration.TextConfiguration
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        init(_ config: DeepSeekVLConfiguration.TextConfiguration) {
            self.config = config
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _qProj.wrappedValue = Linear(
                config.hiddenSize, config.attentionHeads * headDim, bias: false,
            )
            _kProj.wrappedValue = Linear(
                config.hiddenSize, config.kvHeads * headDim, bias: false,
            )
            _vProj.wrappedValue = Linear(
                config.hiddenSize, config.kvHeads * headDim, bias: false,
            )
            _oProj.wrappedValue = Linear(
                config.attentionHeads * headDim, config.hiddenSize, bias: false,
            )

            let ropeScale: Float =
                if let ropeScaling = config.ropeScaling,
                    ropeScaling["type"] == .string("linear"),
                    let factor = ropeScaling["factor"]?.asFloat()
                {
                    1 / factor
                } else {
                    1
                }

            _rotaryEmbedding.wrappedValue = RoPE(
                dimensions: headDim,
                traditional: config.ropeTraditional,
                base: config.ropeTheta,
                scale: ropeScale,
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

        init(_ config: DeepSeekVLConfiguration.TextConfiguration) {
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

        init(_ config: DeepSeekVLConfiguration.TextConfiguration) {
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
        let config: DeepSeekVLConfiguration.TextConfiguration
        let layers: [DecoderLayer]

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo var norm: RMSNorm

        init(_ config: DeepSeekVLConfiguration.TextConfiguration) {
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
        @ModuleInfo(key: "lm_head") var lmHead: Linear
        let config: DeepSeekVLConfiguration.TextConfiguration

        init(_ config: DeepSeekVLConfiguration.TextConfiguration) {
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

private enum DeepSeekVLVision {
    final class Attention: Module {
        let config: DeepSeekVLConfiguration.VisionConfiguration
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear

        init(_ config: DeepSeekVLConfiguration.VisionConfiguration) {
            self.config = config
            heads = config.attentionHeads
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _qkv.wrappedValue = Linear(
                config.hiddenSize, 3 * config.hiddenSize, bias: true,
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
        let activation = GELU(approximation: .fast)

        init(hiddenSize: Int, intermediateSize: Int, bias: Bool = true) {
            _fc1.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
            _fc2.wrappedValue = Linear(intermediateSize, hiddenSize, bias: bias)
        }

        convenience init(_ config: DeepSeekVLConfiguration.VisionConfiguration) {
            self.init(hiddenSize: config.hiddenSize, intermediateSize: config.intermediateSize)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    final class EncoderLayer: Module {
        @ModuleInfo var attn: Attention
        @ModuleInfo var norm1: LayerNorm
        @ModuleInfo var mlp: MLP
        @ModuleInfo var norm2: LayerNorm

        init(_ config: DeepSeekVLConfiguration.VisionConfiguration) {
            _attn.wrappedValue = Attention(config)
            _norm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps,
            )
            _mlp.wrappedValue = MLP(config)
            _norm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = x + attn(norm1(x))
            return h + mlp(norm2(h))
        }
    }

    final class VisionEmbeddings: Module {
        let config: DeepSeekVLConfiguration.VisionConfiguration
        let numPatches: Int

        @ModuleInfo var proj: Conv2d

        init(_ config: DeepSeekVLConfiguration.VisionConfiguration) {
            self.config = config
            numPatches =
                (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize)
            _proj.wrappedValue = Conv2d(
                inputChannels: config.channels,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: true,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            proj(x).flattened(start: 1, end: 2)
        }
    }

    final class AttentionPoolLatent: Module {
        let config: DeepSeekVLConfiguration.VisionConfiguration
        let heads: Int
        let headDim: Int
        let latentLength = 1

        @ModuleInfo var latent: MLXArray
        @ModuleInfo var q: Linear
        @ModuleInfo var kv: Linear
        @ModuleInfo var proj: Linear
        @ModuleInfo var norm: LayerNorm
        @ModuleInfo var mlp: MLP

        init(_ config: DeepSeekVLConfiguration.VisionConfiguration) {
            self.config = config
            heads = config.attentionHeads
            headDim = config.hiddenSize / config.attentionHeads
            _latent.wrappedValue = MLXArray.zeros([1, latentLength, config.hiddenSize])
            _q.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _kv.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * 2, bias: true)
            _proj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _norm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-5)
            _mlp.wrappedValue = MLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.hiddenSize * 4,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let batchSize = x.dim(0)
            let sequenceLength = x.dim(1)
            let channels = x.dim(2)
            let latentQuery = broadcast(
                latent.asType(x.dtype), to: [batchSize, latentLength, channels])
            var queries = q(latentQuery)
                .reshaped(batchSize, latentLength, heads, headDim)
                .transposed(0, 2, 1, 3)
            let kvStates = kv(x)
                .reshaped(batchSize, sequenceLength, 2, heads, headDim)
                .transposed(2, 0, 3, 1, 4)
            let keys = kvStates[0]
            let values = kvStates[1]
            queries = queries.asType(keys.dtype)

            var output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: pow(Float(headDim), -0.5),
                mask: .none,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, latentLength, channels)
            output = proj(output)
            output = output + mlp(norm(output))
            return output[0..., 0, 0...]
        }
    }

    final class SigLipVisionModel: Module {
        let config: DeepSeekVLConfiguration.VisionConfiguration
        let blocks: [EncoderLayer]

        @ModuleInfo(key: "patch_embed") var patchEmbed: VisionEmbeddings
        @ModuleInfo(key: "pos_embed") var posEmbed: MLXArray
        @ModuleInfo var norm: LayerNorm
        @ModuleInfo(key: "attn_pool") var attentionPool: AttentionPoolLatent

        init(_ config: DeepSeekVLConfiguration.VisionConfiguration) {
            self.config = config
            let numPatches =
                (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize)
            blocks = (0 ..< config.hiddenLayers).map { _ in EncoderLayer(config) }
            _patchEmbed.wrappedValue = VisionEmbeddings(config)
            _posEmbed.wrappedValue = MLXArray.zeros([1, numPatches, config.hiddenSize])
            _norm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
            _attentionPool.wrappedValue = AttentionPoolLatent(config)
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, [MLXArray]?,
        ) {
            var x = patchEmbed(x)
            x = x + posEmbed.asType(x.dtype)

            var encoderStates: [MLXArray]? = outputHiddenStates ? [x] : nil
            for block in blocks {
                x = block(x)
                encoderStates?.append(x)
            }

            return (norm(x), x, encoderStates)
        }
    }

    final class VisionModel: Module {
        let config: DeepSeekVLConfiguration.VisionConfiguration

        @ModuleInfo(key: "vision_tower") var visionTower: SigLipVisionModel

        init(_ config: DeepSeekVLConfiguration.VisionConfiguration) {
            self.config = config
            _visionTower.wrappedValue = SigLipVisionModel(config)
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, [MLXArray]?,
        ) {
            visionTower(x, outputHiddenStates: outputHiddenStates)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized: [String: MLXArray] = [:]
            for (key, value) in weights {
                if key.contains("position_ids") {
                    continue
                }
                if key.hasSuffix("patch_embed.proj.weight"), value.ndim == 4 {
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

private final class DeepSeekVLProjector: Module, UnaryLayer {
    @ModuleInfo var linear0: Linear
    @ModuleInfo var linear2: Linear
    let activation = GELU()

    init(_ config: DeepSeekVLConfiguration) {
        _linear0.wrappedValue = Linear(
            config.visionConfiguration.hiddenSize,
            config.textConfiguration.hiddenSize,
            bias: true,
        )
        _linear2.wrappedValue = Linear(
            config.textConfiguration.hiddenSize,
            config.textConfiguration.hiddenSize,
            bias: true,
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(activation(linear0(x)))
    }
}

public final class DeepSeekVL: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_model") private var visionModel: DeepSeekVLVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: DeepSeekVLLanguage.LanguageModel
    @ModuleInfo private var aligner: DeepSeekVLProjector

    public let config: DeepSeekVLConfiguration
    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public init(_ config: DeepSeekVLConfiguration) {
        self.config = config
        _visionModel.wrappedValue = DeepSeekVLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = DeepSeekVLLanguage.LanguageModel(config.textConfiguration)
        _aligner.wrappedValue = DeepSeekVLProjector(config)
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    private func inputEmbeddings(inputIDs: MLXArray, pixelValues: MLXArray?) -> MLXArray {
        guard let pixelValues else {
            return languageModel.model.embedTokens(inputIDs)
        }

        let safeInputIDs = which(inputIDs .== config.imageTokenIndex, MLXArray(0), inputIDs)
        let inputsEmbeds = languageModel.model.embedTokens(safeInputIDs)

        let (hiddenStates, _, _) = visionModel(
            pixelValues.transposed(0, 2, 3, 1),
            outputHiddenStates: false,
        )
        let imageFeatures = aligner(hiddenStates)
            .asType(inputsEmbeds.dtype)
            .reshaped(-1, config.textConfiguration.hiddenSize)

        let ids = inputIDs[0].asArray(Int.self)
        let positions = ids.indices.filter { ids[$0] == config.imageTokenIndex }
        guard !positions.isEmpty else {
            return inputsEmbeds
        }

        var segments: [MLXArray] = []
        var textStart = 0
        var featureStart = 0
        var index = 0

        while index < positions.count {
            let runStart = positions[index]
            var runEnd = runStart
            while index + 1 < positions.count, positions[index + 1] == runEnd + 1 {
                index += 1
                runEnd = positions[index]
            }

            if textStart < runStart {
                segments.append(inputsEmbeds[0, textStart ..< runStart, 0...])
            }

            let runLength = runEnd - runStart + 1
            segments.append(imageFeatures[featureStart ..< (featureStart + runLength), 0...])
            featureStart += runLength
            textStart = runEnd + 1
            index += 1
        }

        if textStart < ids.count {
            segments.append(inputsEmbeds[0, textStart ..< ids.count, 0...])
        }

        return concatenated(segments, axis: 0).expandedDimensions(axis: 0)
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let embeddings = inputEmbeddings(
            inputIDs: input.text.tokens, pixelValues: input.image?.pixels)

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
        let filtered: [String: MLXArray] = Dictionary(
            uniqueKeysWithValues: weights.compactMap { key, value in
                if key.contains("self_attn.rotary_emb.inv_freq") {
                    return nil
                }
                if key.hasPrefix("aligner.layers.0.") {
                    return (
                        key.replacingOccurrences(of: "aligner.layers.0.", with: "aligner.linear0."),
                        value
                    )
                }
                if key.hasPrefix("aligner.layers.2.") {
                    return (
                        key.replacingOccurrences(of: "aligner.layers.2.", with: "aligner.linear2."),
                        value
                    )
                }
                return (key, value)
            },
        )
        return visionModel.sanitize(weights: filtered)
    }
}
