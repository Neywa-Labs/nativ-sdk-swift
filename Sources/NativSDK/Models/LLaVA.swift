import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct LLaVAConfiguration: Codable, Sendable {
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
        private let _ropeTheta: Float?
        public var ropeTheta: Float {
            _ropeTheta ?? (modelType == "qwen2" ? 1_000_000 : 10000)
        }

        private let _ropeTraditional: Bool?
        public var ropeTraditional: Bool {
            _ropeTraditional ?? false
        }

        public let ropeScaling: [String: StringOrNumber]?
        private let _tieWordEmbeddings: Bool?
        public var tieWordEmbeddings: Bool {
            _tieWordEmbeddings ?? false
        }

        private let _attentionBias: Bool?
        public var attentionBias: Bool {
            _attentionBias ?? (modelType == "qwen2")
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
            case _ropeTheta = "rope_theta"
            case _ropeTraditional = "rope_traditional"
            case ropeScaling = "rope_scaling"
            case _tieWordEmbeddings = "tie_word_embeddings"
            case _attentionBias = "attention_bias"
        }

        public init(
            modelType: String,
            hiddenSize: Int,
            hiddenLayers: Int,
            intermediateSize: Int,
            attentionHeads: Int,
            kvHeads: Int? = nil,
            rmsNormEps: Float,
            vocabularySize: Int,
            ropeTheta: Float? = nil,
            ropeTraditional: Bool? = nil,
            ropeScaling: [String: StringOrNumber]? = nil,
            tieWordEmbeddings: Bool? = nil,
            attentionBias: Bool? = nil,
        ) {
            self.modelType = modelType
            self.hiddenSize = hiddenSize
            self.hiddenLayers = hiddenLayers
            self.intermediateSize = intermediateSize
            self.attentionHeads = attentionHeads
            _kvHeads = kvHeads
            self.rmsNormEps = rmsNormEps
            self.vocabularySize = vocabularySize
            _ropeTheta = ropeTheta
            _ropeTraditional = ropeTraditional
            self.ropeScaling = ropeScaling
            _tieWordEmbeddings = tieWordEmbeddings
            _attentionBias = attentionBias
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let imageSize: Int
        public let patchSize: Int
        private let _channels: Int?
        public var channels: Int {
            _channels ?? 3
        }

        private let _layerNormEps: Float?
        public var layerNormEps: Float {
            _layerNormEps ?? 1e-5
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case _channels = "num_channels"
            case _layerNormEps = "layer_norm_eps"
        }

        public init(
            modelType: String,
            hiddenSize: Int,
            hiddenLayers: Int,
            intermediateSize: Int,
            attentionHeads: Int,
            imageSize: Int,
            patchSize: Int,
            channels: Int? = nil,
            layerNormEps: Float? = nil,
        ) {
            self.modelType = modelType
            self.hiddenSize = hiddenSize
            self.hiddenLayers = hiddenLayers
            self.intermediateSize = intermediateSize
            self.attentionHeads = attentionHeads
            self.imageSize = imageSize
            self.patchSize = patchSize
            _channels = channels
            _layerNormEps = layerNormEps
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int {
        _imageTokenIndex ?? 32000
    }

    private let _visionFeatureLayer: Int?
    public var visionFeatureLayer: Int {
        _visionFeatureLayer ?? -2
    }

    private let _visionFeatureSelectStrategy: String?
    public var visionFeatureSelectStrategy: String {
        _visionFeatureSelectStrategy ?? "default"
    }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _imageTokenIndex = "image_token_index"
        case _visionFeatureLayer = "vision_feature_layer"
        case _visionFeatureSelectStrategy = "vision_feature_select_strategy"
    }

    public init(
        textConfiguration: TextConfiguration,
        visionConfiguration: VisionConfiguration,
        modelType: String,
        imageTokenIndex: Int? = nil,
        visionFeatureLayer: Int? = nil,
        visionFeatureSelectStrategy: String? = nil,
    ) {
        self.textConfiguration = textConfiguration
        self.visionConfiguration = visionConfiguration
        self.modelType = modelType
        _imageTokenIndex = imageTokenIndex
        _visionFeatureLayer = visionFeatureLayer
        _visionFeatureSelectStrategy = visionFeatureSelectStrategy
    }
}

// MARK: - Processor

public struct LLaVAProcessorConfiguration: Codable, Sendable {
    public struct Size: Codable, Sendable {
        public let width: Int?
        public let height: Int?
        public let shortestEdge: Int?

        public var cgSize: CGSize {
            let width = width ?? shortestEdge ?? 384
            let height = height ?? shortestEdge ?? 384
            return CGSize(width: width, height: height)
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
    private let _patchSize: Int?
    public var patchSize: Int {
        _patchSize ?? 14
    }

    private let _visionFeatureSelectStrategy: String?
    public var visionFeatureSelectStrategy: String {
        _visionFeatureSelectStrategy ?? "full"
    }

    private let _imageToken: String?
    public var imageToken: String {
        _imageToken ?? "<image>"
    }

    private let _numAdditionalImageTokens: Int?
    public var numAdditionalImageTokens: Int {
        _numAdditionalImageTokens ?? 0
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
        case _patchSize = "patch_size"
        case _visionFeatureSelectStrategy = "vision_feature_select_strategy"
        case _imageToken = "image_token"
        case _numAdditionalImageTokens = "num_additional_image_tokens"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass = try c.decodeIfPresent(String.self, forKey: .processorClass)
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        size = try c.decodeIfPresent(Size.self, forKey: .size)
        _patchSize = try c.decodeIfPresent(Int.self, forKey: ._patchSize)
        _visionFeatureSelectStrategy = try c.decodeIfPresent(
            String.self, forKey: ._visionFeatureSelectStrategy,
        )
        _imageToken = try c.decodeIfPresent(String.self, forKey: ._imageToken)
        _numAdditionalImageTokens = try c.decodeIfPresent(
            Int.self, forKey: ._numAdditionalImageTokens,
        )
    }
}

public struct LLaVAProcessor: UserInputProcessor {
    private let config: LLaVAProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageTokenIndex: Int
    private let imageTokenCandidates: [Int]

    public init(_ config: LLaVAProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        let encoded = tokenizer.encode(text: config.imageToken)
        let candidates: [Int?] = [
            tokenizer.convertTokenToId(config.imageToken),
            encoded.first,
            151_646,
            32000,
            -200,
        ]
        var uniqueCandidates: [Int] = []
        for candidate in candidates.compactMap(\.self) where !uniqueCandidates.contains(candidate) {
            uniqueCandidates.append(candidate)
        }
        imageTokenCandidates = uniqueCandidates
        imageTokenIndex = uniqueCandidates.first ?? 151_646
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = LLaVAMessageGenerator().generate(from: input)
        let images = MediaProcessing.images(from: input)

        if images.isEmpty {
            let promptTokens = try tokenizer.applyChatTemplate(
                messages: messages, tools: input.tools,
                additionalContext: input.additionalContext,
            )
            let tokensArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            return LMInput(text: .init(tokens: tokensArray, mask: ones(like: tokensArray)))
        }

        guard images.count == 1 else {
            throw VLMError.singleImageAllowed
        }

        let image = try images[0]
            .asCIImage()
            .toSRGB()
            .resampled(
                to: (config.size?.cgSize ?? CGSize(width: 384, height: 384)), method: .bicubic
            )
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
        let pixels = image.asMLXArray()

        let height = pixels.dim(2)
        let width = pixels.dim(3)
        var imageTokenCount = (height / config.patchSize) * (width / config.patchSize)
        imageTokenCount += config.numAdditionalImageTokens
        if config.visionFeatureSelectStrategy == "default" {
            imageTokenCount -= 1
        }

        let promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext,
        )
        let promptImageTokenIndex =
            imageTokenCandidates.first { promptTokens.contains($0) } ?? imageTokenIndex
        let tokenRun = Array(repeating: promptImageTokenIndex, count: max(imageTokenCount, 1))
        var didReplaceImageToken = false
        var tokens = promptTokens.flatMap { token in
            if token == promptImageTokenIndex {
                didReplaceImageToken = true
                return tokenRun
            }
            return [token]
        }
        if !didReplaceImageToken {
            let decoded = tokenizer.decode(tokenIds: promptTokens, skipSpecialTokens: false)
            let promptPieces = decoded.components(separatedBy: config.imageToken)
            if promptPieces.count > 1 {
                tokens = promptPieces.enumerated().flatMap { index, piece in
                    var part = tokenizer.encode(text: piece)
                    if index < promptPieces.count - 1 {
                        part.append(contentsOf: tokenRun)
                    }
                    return part
                }
            }
        }
        let promptArray = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: promptArray, mask: ones(like: promptArray)),
            image: .init(pixels: pixels),
        )
    }

}

public struct LLaVAMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        [
            "role": message.role.rawValue,
            "content": []
                + message.images.map { _ in ["type": "image"] }
                + [["type": "text", "text": message.content]],
        ]
    }
}

// MARK: - Language

enum LLaVALanguage {
    final class Attention: Module {
        let config: LLaVAConfiguration.TextConfiguration
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        init(_ config: LLaVAConfiguration.TextConfiguration) {
            self.config = config
            headDim = config.hiddenSize / config.attentionHeads
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
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        init(_ config: LLaVAConfiguration.TextConfiguration) {
            _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
            _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
            _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: LLaVAConfiguration.TextConfiguration) {
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

    final class TextModel: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        let layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm

        init(_ config: LLaVAConfiguration.TextConfiguration) {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            layers = (0 ..< config.hiddenLayers).map { _ in DecoderLayer(config) }
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

    final class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: TextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear?
        let config: LLaVAConfiguration.TextConfiguration
        let kvHeads: [Int]

        init(_ config: LLaVAConfiguration.TextConfiguration) {
            self.config = config
            _model.wrappedValue = TextModel(config)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.hiddenSize, config.vocabularySize, bias: false,
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
            var out = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }
            return LMOutput(logits: out)
        }
    }
}

// MARK: - Vision

enum LLaVAVision {
    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: LLaVAConfiguration.VisionConfiguration) {
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
        let activation = GELU(approximation: .fast)

        init(_ config: LLaVAConfiguration.VisionConfiguration) {
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

        init(_ config: LLaVAConfiguration.VisionConfiguration) {
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
            var out = x + y
            y = mlp(layerNorm2(out))
            out = out + y
            return out
        }
    }

    final class Encoder: Module {
        let layers: [EncoderLayer]

        init(_ config: LLaVAConfiguration.VisionConfiguration) {
            layers = (0 ..< config.hiddenLayers).map { _ in EncoderLayer(config) }
        }
    }

    final class VisionEmbeddings: Module {
        let config: LLaVAConfiguration.VisionConfiguration
        let numPositions: Int

        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        init(_ config: LLaVAConfiguration.VisionConfiguration) {
            self.config = config
            let numPatches =
                (config.imageSize / config.patchSize) * (config.imageSize / config.patchSize)
            numPositions = config.modelType == "clip_vision_model" ? numPatches + 1 : numPatches
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
            var embeddings = patchEmbedding(x).flattened(start: 1, end: 2)
            if config.modelType == "clip_vision_model" {
                let classEmbedding = MLXArray.zeros(
                    [x.dim(0), 1, config.hiddenSize], dtype: embeddings.dtype)
                embeddings = concatenated([classEmbedding, embeddings], axis: 1)
            }
            let positionIDs = MLXArray(0 ..< numPositions).expandedDimensions(axis: 0)
            return embeddings + positionEmbedding(positionIDs)
        }
    }

    final class VisionModelInner: Module {
        @ModuleInfo var embeddings: VisionEmbeddings
        @ModuleInfo(key: "pre_layrnorm") var preLayerNorm: LayerNorm?
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        init(_ config: LLaVAConfiguration.VisionConfiguration) {
            _embeddings.wrappedValue = VisionEmbeddings(config)
            if config.modelType == "clip_vision_model" {
                _preLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
            }
            _encoder.wrappedValue = Encoder(config)
            _postLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
        }

        func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
            MLXArray, MLXArray, [MLXArray]?,
        ) {
            var x = embeddings(x)
            if let preLayerNorm {
                x = preLayerNorm(x)
            }

            var encoderStates: [MLXArray]? = outputHiddenStates ? [x] : nil
            for layer in encoder.layers {
                x = layer(x)
                encoderStates?.append(x)
            }

            let poolerOutput = postLayerNorm(x[0..., 0, 0...])
            return (poolerOutput, x, encoderStates)
        }
    }

    final class VisionModel: Module {
        @ModuleInfo(key: "vision_model") var visionModel: VisionModelInner

        init(_ config: LLaVAConfiguration.VisionConfiguration) {
            _visionModel.wrappedValue = VisionModelInner(config)
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

// MARK: - Model

final class LLaVAMultiModalProjector: Module, UnaryLayer {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    let activation = GELU()

    init(_ config: LLaVAConfiguration) {
        _linear1.wrappedValue = Linear(
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
        linear2(activation(linear1(x)))
    }
}

public final class LLaVA: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionModel: LLaVAVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: LLaVALanguage.LanguageModel
    @ModuleInfo(key: "multi_modal_projector") private var projector: LLaVAMultiModalProjector

    public let config: LLaVAConfiguration
    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public init(_ config: LLaVAConfiguration) {
        self.config = config
        _visionModel.wrappedValue = LLaVAVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = LLaVALanguage.LanguageModel(config.textConfiguration)
        _projector.wrappedValue = LLaVAMultiModalProjector(config)
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    private func inputEmbeddings(
        inputIDs: MLXArray,
        pixelValues: MLXArray?,
    )
        -> MLXArray
    {
        guard let pixelValues else {
            return languageModel.model.embedTokens(inputIDs)
        }

        let safeInputIDs = which(
            inputIDs .== config.imageTokenIndex,
            MLXArray(0),
            inputIDs,
        )
        let inputsEmbeds = languageModel.model.embedTokens(safeInputIDs)

        let (_, _, hiddenStates) = visionModel(
            pixelValues.transposed(0, 2, 3, 1),
            outputHiddenStates: true,
        )
        guard let hiddenStates else {
            return inputsEmbeds
        }

        let layerIndex =
            config.visionFeatureLayer >= 0
            ? config.visionFeatureLayer
            : hiddenStates.count + config.visionFeatureLayer
        var imageFeatures = hiddenStates[layerIndex]
        if config.visionFeatureSelectStrategy == "default" {
            imageFeatures = imageFeatures[0..., 1..., 0...]
        }
        imageFeatures = projector(imageFeatures).asType(inputsEmbeds.dtype)

        let ids = inputIDs[0].asArray(Int.self)
        let positions = ids.indices.filter { ids[$0] == config.imageTokenIndex }
        guard let first = positions.first, let last = positions.last else {
            return inputsEmbeds
        }

        let before = inputsEmbeds[0, 0 ..< first, 0...]
        let after = inputsEmbeds[0, (last + 1) ..< ids.count, 0...]
        let merged = concatenated([before, imageFeatures[0], after], axis: 0)
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
        return visionModel.sanitize(weights: filtered)
    }
}
