import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct Idefics2Configuration: Decodable, Sendable {
    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        private let _hiddenSize: Int?
        public var hiddenSize: Int {
            _hiddenSize ?? 4096
        }

        private let _hiddenLayers: Int?
        public var hiddenLayers: Int {
            _hiddenLayers ?? 32
        }

        private let _intermediateSize: Int?
        public var intermediateSize: Int {
            _intermediateSize ?? 14336
        }

        private let _attentionHeads: Int?
        public var attentionHeads: Int {
            _attentionHeads ?? 32
        }

        private let _kvHeads: Int?
        public var kvHeads: Int {
            _kvHeads ?? 8
        }

        public let rmsNormEps: Float
        public let vocabularySize: Int
        private let _ropeTheta: Float?
        public var ropeTheta: Float {
            _ropeTheta ?? 1_000_000
        }

        private let _ropeTraditional: Bool?
        public var ropeTraditional: Bool {
            _ropeTraditional ?? false
        }

        private let _maxPositionEmbeddings: Int?
        public var maxPositionEmbeddings: Int {
            _maxPositionEmbeddings ?? 32768
        }

        private let _tieWordEmbeddings: Bool?
        public var tieWordEmbeddings: Bool {
            _tieWordEmbeddings ?? false
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case _hiddenSize = "hidden_size"
            case _hiddenLayers = "num_hidden_layers"
            case _intermediateSize = "intermediate_size"
            case _attentionHeads = "num_attention_heads"
            case _kvHeads = "num_key_value_heads"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case _ropeTheta = "rope_theta"
            case _ropeTraditional = "rope_traditional"
            case _maxPositionEmbeddings = "max_position_embeddings"
            case _tieWordEmbeddings = "tie_word_embeddings"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "mistral"
            _hiddenSize = try c.decodeIfPresent(Int.self, forKey: ._hiddenSize)
            _hiddenLayers = try c.decodeIfPresent(Int.self, forKey: ._hiddenLayers)
            _intermediateSize = try c.decodeIfPresent(Int.self, forKey: ._intermediateSize)
            _attentionHeads = try c.decodeIfPresent(Int.self, forKey: ._attentionHeads)
            _kvHeads = try c.decodeIfPresent(Int.self, forKey: ._kvHeads)
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
            vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 32003
            _ropeTheta = try c.decodeIfPresent(Float.self, forKey: ._ropeTheta)
            _ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: ._ropeTraditional)
            _maxPositionEmbeddings = try c.decodeIfPresent(
                Int.self, forKey: ._maxPositionEmbeddings,
            )
            _tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: ._tieWordEmbeddings)
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let imageSize: Int
        public let patchSize: Int
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
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case _channels = "num_channels"
            case _layerNormEps = "layer_norm_eps"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "idefics2"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4352
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 27
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 980
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            _channels = try c.decodeIfPresent(Int.self, forKey: ._channels)
            _layerNormEps = try c.decodeIfPresent(Float.self, forKey: ._layerNormEps)
        }
    }

    public struct PerceiverConfiguration: Decodable, Sendable {
        public let modelType: String
        private let _kvHeads: Int?
        public var kvHeads: Int {
            _kvHeads ?? 4
        }

        private let _depth: Int?
        public var depth: Int {
            _depth ?? 3
        }

        private let _headDim: Int?
        public var headDim: Int {
            _headDim ?? 96
        }

        private let _heads: Int?
        public var heads: Int {
            _heads ?? 16
        }

        private let _latents: Int?
        public var latents: Int {
            _latents ?? 64
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case _kvHeads = "num_key_value_heads"
            case _depth = "resampler_depth"
            case _headDim = "resampler_head_dim"
            case _heads = "resampler_n_heads"
            case _latents = "resampler_n_latents"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "idefics2"
            _kvHeads = try c.decodeIfPresent(Int.self, forKey: ._kvHeads)
            _depth = try c.decodeIfPresent(Int.self, forKey: ._depth)
            _headDim = try c.decodeIfPresent(Int.self, forKey: ._headDim)
            _heads = try c.decodeIfPresent(Int.self, forKey: ._heads)
            _latents = try c.decodeIfPresent(Int.self, forKey: ._latents)
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let perceiverConfiguration: PerceiverConfiguration
    public let modelType: String
    private let _imageTokenID: Int?
    public var imageTokenID: Int {
        _imageTokenID ?? 32001
    }

    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int {
        _imageTokenIndex ?? imageTokenID
    }

    private let _vocabularySize: Int?
    public var vocabularySize: Int {
        _vocabularySize ?? textConfiguration.vocabularySize
    }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case perceiverConfiguration = "perceiver_config"
        case modelType = "model_type"
        case _imageTokenID = "image_token_id"
        case _imageTokenIndex = "image_token_index"
        case _vocabularySize = "vocab_size"
    }
}

// MARK: - Processor

public struct Idefics2ProcessorConfiguration: Decodable, Sendable {
    public struct Size: Decodable, Sendable {
        public let longestEdge: Int?
        public let shortestEdge: Int?

        enum CodingKeys: String, CodingKey {
            case longestEdge = "longest_edge"
            case shortestEdge = "shortest_edge"
        }
    }

    public let processorClass: String?
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let size: Size
    private let _imageSequenceLength: Int?
    public var imageSequenceLength: Int {
        _imageSequenceLength ?? 64
    }

    private let _doImageSplitting: Bool?
    public var doImageSplitting: Bool {
        _doImageSplitting ?? true
    }

    public let imageTokenID: Int
    public let fakeImageTokenID: Int
    public let endOfUtteranceTokenID: Int

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case size
        case _imageSequenceLength = "image_seq_len"
        case _doImageSplitting = "do_image_splitting"
        case imageTokenID = "image_token_id"
        case fakeImageTokenID = "fake_image_token_id"
        case endOfUtteranceTokenID = "end_of_utterance_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass = try c.decodeIfPresent(String.self, forKey: .processorClass)
        imageMean = try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd = try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        size =
            try c.decodeIfPresent(Size.self, forKey: .size)
            ?? Size(longestEdge: 980, shortestEdge: 378)
        _imageSequenceLength = try c.decodeIfPresent(Int.self, forKey: ._imageSequenceLength)
        _doImageSplitting = try c.decodeIfPresent(Bool.self, forKey: ._doImageSplitting)
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? 32001
        fakeImageTokenID = try c.decodeIfPresent(Int.self, forKey: .fakeImageTokenID) ?? 32000
        endOfUtteranceTokenID =
            try c.decodeIfPresent(Int.self, forKey: .endOfUtteranceTokenID) ?? 32002
    }
}

public struct Idefics2Processor: UserInputProcessor {
    private let config: Idefics2ProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Idefics2ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        let prompt = renderPrompt(from: input, imageCount: images.count)
        guard !images.isEmpty else {
            let tokens = tokenizer.encode(text: prompt, addSpecialTokens: true)
            let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
            return LMInput(text: .init(tokens: tokenArray, mask: ones(like: tokenArray)))
        }

        var allImageBlocks: [[Float]] = []
        for image in images {
            let blocks = try preprocess(image: image.asCIImage(), processing: input.processing)
            allImageBlocks.append(contentsOf: blocks)
        }

        let imageBlockCount = config.doImageSplitting ? 5 : 1
        let tokens = tokenizePrompt(
            prompt,
            expandedImageBlocksPerImage: Array(repeating: imageBlockCount, count: images.count),
        )
        let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)

        return LMInput(
            text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)),
            image: .init(
                pixels: MLXArray(
                    allImageBlocks.flatMap(\.self),
                    [1, allImageBlocks.count, 3, 980, 980],
                ),
            ),
        )
    }

    private func renderPrompt(from input: UserInput, imageCount: Int) -> String {
        switch input.prompt {
        case .chat(let messages):
            var rendered = ""
            for message in messages {
                rendered += message.role.rawValue.capitalized
                rendered += message.images.first == nil ? ": " : ":"
                for _ in message.images {
                    rendered += "<image>"
                }
                rendered += message.content
                rendered += "<end_of_utterance>\n"
            }
            if messages.last?.role != .assistant {
                rendered += "Assistant:"
            }
            return rendered

        case .text(let text):
            let imageTags = String(repeating: "<image>", count: imageCount)
            return "User:\(imageTags)\(text)<end_of_utterance>\nAssistant:"

        case .messages:
            let imageTags = String(repeating: "<image>", count: imageCount)
            return "User:\(imageTags)\(input.prompt.description)<end_of_utterance>\nAssistant:"
        }
    }

    private func tokenizePrompt(_ prompt: String, expandedImageBlocksPerImage: [Int]) -> [Int] {
        let pieces = prompt.components(separatedBy: "<image>")
        var result: [Int] = []
        for index in pieces.indices {
            result.append(contentsOf: encodeTextPiece(pieces[index], addSpecialTokens: index == 0))
            if index < pieces.count - 1 {
                let blockCount =
                    expandedImageBlocksPerImage.indices.contains(index)
                    ? expandedImageBlocksPerImage[index]
                    : 1
                for _ in 0 ..< blockCount {
                    result.append(config.fakeImageTokenID)
                    result.append(
                        contentsOf: Array(
                            repeating: config.imageTokenID,
                            count: config.imageSequenceLength,
                        ),
                    )
                    result.append(config.fakeImageTokenID)
                }
            }
        }
        return result
    }

    private func encodeTextPiece(_ text: String, addSpecialTokens: Bool) -> [Int] {
        let pieces = text.components(separatedBy: "<end_of_utterance>")
        guard pieces.count > 1 else {
            return tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
        }

        var tokens: [Int] = []
        for index in pieces.indices {
            tokens.append(
                contentsOf: tokenizer.encode(
                    text: pieces[index],
                    addSpecialTokens: addSpecialTokens && index == 0,
                ),
            )
            if index < pieces.count - 1 {
                tokens.append(config.endOfUtteranceTokenID)
            }
        }
        return tokens
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?) throws -> [[Float]] {
        let canonical = MediaProcessing.zeroOrigin(
            MediaProcessing.apply(image, processing: processing))
        guard config.doImageSplitting else {
            return [
                Self.processBlock(
                    canonical,
                    mean: config.imageMean,
                    std: config.imageStd,
                )
            ]
        }

        let extent = canonical.extent
        let midX = extent.midX
        let midY = extent.midY
        let quadrants = [
            CGRect(x: extent.minX, y: midY, width: extent.width / 2, height: extent.height / 2),
            CGRect(x: midX, y: midY, width: extent.width / 2, height: extent.height / 2),
            CGRect(
                x: extent.minX, y: extent.minY, width: extent.width / 2, height: extent.height / 2),
            CGRect(x: midX, y: extent.minY, width: extent.width / 2, height: extent.height / 2),
        ]
        var blocks = quadrants.map {
            Self.processBlock(
                MediaProcessing.zeroOrigin(canonical.cropped(to: $0)),
                mean: config.imageMean,
                std: config.imageStd,
            )
        }
        blocks.append(Self.processBlock(canonical, mean: config.imageMean, std: config.imageStd))
        return blocks
    }

    private static func processBlock(_ image: CIImage, mean: [CGFloat], std: [CGFloat]) -> [Float] {
        let resized = MediaProcessing.zeroOrigin(image).resampled(
            to: CGSize(width: 980, height: 980),
            method: .bicubic,
        )
        let raw = MediaProcessing.renderRGB8(resized, width: 980, height: 980)
        return MediaProcessing.normalizeCHW(raw, width: 980, height: 980, mean: mean, std: std)
    }

}

// MARK: - Language

private enum Idefics2Language {
    final class Attention: Module {
        let config: Idefics2Configuration.TextConfiguration
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "rope") var rope: RoPE

        init(_ config: Idefics2Configuration.TextConfiguration) {
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
            _rope.wrappedValue = RoPE(
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
            var queries = qProj(x).reshaped(
                batchSize, sequenceLength, config.attentionHeads, headDim
            )
            .transposed(0, 2, 1, 3)
            var keys = kProj(x).reshaped(batchSize, sequenceLength, config.kvHeads, headDim)
                .transposed(0, 2, 1, 3)
            let values = vProj(x).reshaped(batchSize, sequenceLength, config.kvHeads, headDim)
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
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        init(inputDimensions: Int, hiddenDimensions: Int, outputDimensions: Int) {
            _gate.wrappedValue = Linear(inputDimensions, hiddenDimensions, bias: false)
            _down.wrappedValue = Linear(hiddenDimensions, outputDimensions, bias: false)
            _up.wrappedValue = Linear(inputDimensions, hiddenDimensions, bias: false)
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

        init(_ config: Idefics2Configuration.TextConfiguration) {
            _attention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(
                inputDimensions: config.hiddenSize,
                hiddenDimensions: config.intermediateSize,
                outputDimensions: config.hiddenSize,
            )
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
            var residual = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + residual
            residual = mlp(postAttentionLayerNorm(h))
            return h + residual
        }
    }

    final class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        let config: Idefics2Configuration.TextConfiguration
        var kvHeads: [Int] {
            (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        }

        init(_ config: Idefics2Configuration.TextConfiguration) {
            self.config = config
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in DecoderLayer(config) }
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.hiddenSize,
                    config.vocabularySize,
                    bias: false,
                )
            }
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
        )
            -> LMOutput
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
            h = norm(h)
            let logits = lmHead.map { $0(h) } ?? embedTokens.asLinear(h)
            return LMOutput(logits: logits)
        }
    }
}

// MARK: - Vision

private enum Idefics2Vision {
    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: Idefics2Configuration.VisionConfiguration) {
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

        init(_ config: Idefics2Configuration.VisionConfiguration) {
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

        init(_ config: Idefics2Configuration.VisionConfiguration) {
            _attention.wrappedValue = Attention(config)
            _layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
            _mlp.wrappedValue = MLP(config)
            _layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
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
        @ModuleInfo(key: "layers") var layers: [EncoderLayer]

        init(_ config: Idefics2Configuration.VisionConfiguration) {
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in EncoderLayer(config) }
        }
    }

    final class Embeddings: Module {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding
        let numPositions: Int

        init(_ config: Idefics2Configuration.VisionConfiguration) {
            _patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.channels,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: true,
            )
            let side = config.imageSize / config.patchSize
            numPositions = side * side
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

    final class VisionModel: Module {
        @ModuleInfo var embeddings: Embeddings
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        init(_ config: Idefics2Configuration.VisionConfiguration) {
            _embeddings.wrappedValue = Embeddings(config)
            _encoder.wrappedValue = Encoder(config)
            _postLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = embeddings(x)
            for layer in encoder.layers {
                h = layer(h)
            }
            return postLayerNorm(h)
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
}

// MARK: - Connector

private final class Idefics2PerceiverAttention: Module {
    let config: Idefics2Configuration
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ config: Idefics2Configuration) {
        self.config = config
        let dim = config.textConfiguration.hiddenSize
        let headDim = config.perceiverConfiguration.headDim
        scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(
            dim,
            config.perceiverConfiguration.heads * headDim,
            bias: false,
        )
        _kProj.wrappedValue = Linear(
            dim,
            config.perceiverConfiguration.kvHeads * headDim,
            bias: false,
        )
        _vProj.wrappedValue = Linear(
            dim,
            config.perceiverConfiguration.kvHeads * headDim,
            bias: false,
        )
        _oProj.wrappedValue = Linear(
            config.perceiverConfiguration.heads * headDim,
            dim,
            bias: false,
        )
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        let hiddenStates = concatenated([context, x], axis: -2)
        let batchSize = x.dim(0)
        let queryLength = x.dim(1)
        let keyLength = hiddenStates.dim(1)
        let headDim = config.perceiverConfiguration.headDim

        let queries = qProj(x)
            .reshaped(batchSize, queryLength, config.perceiverConfiguration.heads, headDim)
            .transposed(0, 2, 1, 3)
        let keys = kProj(hiddenStates)
            .reshaped(batchSize, keyLength, config.perceiverConfiguration.kvHeads, headDim)
            .transposed(0, 2, 1, 3)
        let values = vProj(hiddenStates)
            .reshaped(batchSize, keyLength, config.perceiverConfiguration.kvHeads, headDim)
            .transposed(0, 2, 1, 3)

        let output = MLXFast.scaledDotProductAttention(
            queries: queries,
            keys: keys,
            values: values,
            scale: scale,
            mask: .none,
        )
        .transposed(0, 2, 1, 3)
        .reshaped(batchSize, queryLength, -1)
        return oProj(output)
    }
}

private final class Idefics2PerceiverLayer: Module {
    @ModuleInfo(key: "input_latents_norm") var inputLatentsNorm: RMSNorm
    @ModuleInfo(key: "input_context_norm") var inputContextNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attention: Idefics2PerceiverAttention
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo var mlp: Idefics2Language.MLP

    init(_ config: Idefics2Configuration) {
        let hiddenSize = config.textConfiguration.hiddenSize
        _inputLatentsNorm.wrappedValue = RMSNorm(
            dimensions: hiddenSize,
            eps: config.textConfiguration.rmsNormEps,
        )
        _inputContextNorm.wrappedValue = RMSNorm(
            dimensions: hiddenSize,
            eps: config.textConfiguration.rmsNormEps,
        )
        _attention.wrappedValue = Idefics2PerceiverAttention(config)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: hiddenSize,
            eps: config.textConfiguration.rmsNormEps,
        )
        _mlp.wrappedValue = Idefics2Language.MLP(
            inputDimensions: hiddenSize,
            hiddenDimensions: hiddenSize * 4,
            outputDimensions: hiddenSize,
        )
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray) -> MLXArray {
        var latents = attention(inputLatentsNorm(x), context: inputContextNorm(context))
        latents = x + latents
        let residual = latents
        latents = mlp(postAttentionLayerNorm(latents))
        return residual + latents
    }
}

private final class Idefics2PerceiverResampler: Module {
    @ModuleInfo var latents: MLXArray
    @ModuleInfo(key: "layers") var layers: [Idefics2PerceiverLayer]
    @ModuleInfo var norm: RMSNorm

    let config: Idefics2Configuration

    init(_ config: Idefics2Configuration) {
        self.config = config
        _latents.wrappedValue = MLXArray.ones([
            config.perceiverConfiguration.latents,
            config.textConfiguration.hiddenSize,
        ])
        _layers.wrappedValue =
            (0 ..< config.perceiverConfiguration.depth).map { _ in
                Idefics2PerceiverLayer(config)
            }
        _norm.wrappedValue = RMSNorm(
            dimensions: config.textConfiguration.hiddenSize,
            eps: config.textConfiguration.rmsNormEps,
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = broadcast(
            latents.asType(x.dtype)[.newAxis, 0..., 0...],
            to: [
                x.dim(0), config.perceiverConfiguration.latents,
                config.textConfiguration.hiddenSize,
            ],
        )
        for layer in layers {
            h = layer(h, context: x)
        }
        return norm(h)
    }
}

private final class Idefics2Connector: Module, UnaryLayer {
    @ModuleInfo(key: "modality_projection") var modalityProjection: Idefics2Language.MLP
    @ModuleInfo(key: "perceiver_resampler") var perceiverResampler: Idefics2PerceiverResampler

    init(_ config: Idefics2Configuration) {
        _modalityProjection.wrappedValue = Idefics2Language.MLP(
            inputDimensions: config.visionConfiguration.hiddenSize,
            hiddenDimensions: config.textConfiguration.intermediateSize,
            outputDimensions: config.textConfiguration.hiddenSize,
        )
        _perceiverResampler.wrappedValue = Idefics2PerceiverResampler(config)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        perceiverResampler(modalityProjection(x))
    }
}

// MARK: - Model

public final class Idefics2: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_model") private var visionModel: Idefics2Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Idefics2Language.LanguageModel
    @ModuleInfo private var connector: Idefics2Connector

    public let config: Idefics2Configuration
    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public init(_ config: Idefics2Configuration) {
        self.config = config
        _visionModel.wrappedValue = Idefics2Vision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue =
            Idefics2Language.LanguageModel(config.textConfiguration)
        _connector.wrappedValue = Idefics2Connector(config)
    }

    public var loraLayers: [Module] {
        languageModel.layers
    }

    private func inputEmbeddings(inputIDs: MLXArray, pixelValues: MLXArray?) -> MLXArray {
        let inputIDsArray = inputIDs[0].asArray(Int.self)
        let safeInputIDs = MLXArray(
            inputIDsArray.map {
                $0 == config.imageTokenIndex ? 0 : $0
            },
        ).expandedDimensions(axis: 0)
        let inputsEmbeds = languageModel.embedTokens(safeInputIDs)
        guard let pixelValues else {
            return inputsEmbeds
        }

        let imageBatch = pixelValues.reshaped(-1, 3, 980, 980).transposed(0, 2, 3, 1)
        let imageFeatures = connector(
            visionModel(imageBatch).asType(inputsEmbeds.dtype),
        )
        return merge(imageFeatures: imageFeatures, inputsEmbeds: inputsEmbeds, ids: inputIDsArray)
    }

    private func merge(
        imageFeatures: MLXArray,
        inputsEmbeds: MLXArray,
        ids: [Int],
    )
        -> MLXArray
    {
        let positions = ids.indices.filter { ids[$0] == config.imageTokenIndex }
        let features = imageFeatures.reshaped(-1, config.textConfiguration.hiddenSize)
        guard positions.count == features.dim(0) else {
            return inputsEmbeds
        }

        var segments: [MLXArray] = []
        var start = 0
        for (featureIndex, position) in positions.enumerated() {
            if position > start {
                segments.append(inputsEmbeds[0, start ..< position, 0...])
            }
            segments.append(features[featureIndex ..< (featureIndex + 1), 0...])
            start = position + 1
        }
        if start < ids.count {
            segments.append(inputsEmbeds[0, start ..< ids.count, 0...])
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
