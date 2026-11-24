import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct DotsOCRConfiguration: Codable, Sendable {
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
            _ropeTheta ?? 1_000_000
        }

        private let _ropeTraditional: Bool?
        public var ropeTraditional: Bool {
            _ropeTraditional ?? false
        }

        public let ropeScaling: [String: StringOrNumber]?
        private let _attentionBias: Bool?
        public var attentionBias: Bool {
            _attentionBias ?? true
        }

        private let _tieWordEmbeddings: Bool?
        public var tieWordEmbeddings: Bool {
            _tieWordEmbeddings ?? false
        }

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
            case _attentionBias = "attention_bias"
            case _tieWordEmbeddings = "tie_word_embeddings"
        }
    }

    public struct VisionConfiguration: Codable, Sendable {
        private let _modelType: String?
        public var modelType: String {
            _modelType ?? "dots_vit"
        }

        public let embedDimension: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        private let _channels: Int?
        public var channels: Int {
            _channels ?? 3
        }

        public let patchSize: Int
        private let _postNorm: Bool?
        public var postNorm: Bool {
            _postNorm ?? true
        }

        private let _rmsNormEps: Float?
        public var rmsNormEps: Float {
            _rmsNormEps ?? 1e-5
        }

        private let _spatialMergeSize: Int?
        public var spatialMergeSize: Int {
            _spatialMergeSize ?? 2
        }

        private let _temporalPatchSize: Int?
        public var temporalPatchSize: Int {
            _temporalPatchSize ?? 1
        }

        private let _useBias: Bool?
        public var useBias: Bool {
            _useBias ?? false
        }

        enum CodingKeys: String, CodingKey {
            case _modelType = "model_type"
            case embedDimension = "embed_dim"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case _channels = "num_channels"
            case patchSize = "patch_size"
            case _postNorm = "post_norm"
            case _rmsNormEps = "rms_norm_eps"
            case _spatialMergeSize = "spatial_merge_size"
            case _temporalPatchSize = "temporal_patch_size"
            case _useBias = "use_bias"
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _imageTokenID: Int?
    public var imageTokenID: Int {
        _imageTokenID ?? 151_665
    }

    private let _videoTokenID: Int?
    public var videoTokenID: Int {
        _videoTokenID ?? 151_656
    }

    private let _vocabularySize: Int?
    public var vocabularySize: Int {
        _vocabularySize ?? textConfiguration.vocabularySize
    }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _imageTokenID = "image_token_id"
        case _videoTokenID = "video_token_id"
        case _vocabularySize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "dots_ocr"
        visionConfiguration = try container.decode(
            VisionConfiguration.self, forKey: .visionConfiguration,
        )
        textConfiguration =
            try container.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? TextConfiguration(from: decoder)
        _imageTokenID = try container.decodeIfPresent(Int.self, forKey: ._imageTokenID)
        _videoTokenID = try container.decodeIfPresent(Int.self, forKey: ._videoTokenID)
        _vocabularySize = try container.decodeIfPresent(Int.self, forKey: ._vocabularySize)
    }
}

public struct DotsOCRProcessorConfiguration: Decodable, Sendable {
    private struct NestedImageProcessor: Decodable {
        let imageMean: [CGFloat]?
        let imageStd: [CGFloat]?
        let minPixels: Int?
        let maxPixels: Int?
        let mergeSize: Int?
        let patchSize: Int?
        let temporalPatchSize: Int?
        let imageProcessorType: String?
        let size: Size?

        enum CodingKeys: String, CodingKey {
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case minPixels = "min_pixels"
            case maxPixels = "max_pixels"
            case mergeSize = "merge_size"
            case patchSize = "patch_size"
            case temporalPatchSize = "temporal_patch_size"
            case imageProcessorType = "image_processor_type"
            case size
        }
    }

    public struct Size: Decodable, Sendable {
        public let shortestEdge: Int?
        public let longestEdge: Int?

        enum CodingKeys: String, CodingKey {
            case shortestEdge = "shortest_edge"
            case longestEdge = "longest_edge"
        }
    }

    public let processorClass: String?
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let minPixels: Int
    public let maxPixels: Int
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let imageProcessorType: String?

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
        case minPixels = "min_pixels"
        case maxPixels = "max_pixels"
        case mergeSize = "merge_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case imageProcessorType = "image_processor_type"
        case imageProcessor = "image_processor"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let nested = try c.decodeIfPresent(NestedImageProcessor.self, forKey: .imageProcessor)

        processorClass = try c.decodeIfPresent(String.self, forKey: .processorClass)
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageMean)
            ?? nested?.imageMean
            ?? [0.48145466, 0.4578275, 0.40821073]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageStd)
            ?? nested?.imageStd
            ?? [0.26862954, 0.26130258, 0.27577711]
        minPixels =
            try c.decodeIfPresent(Int.self, forKey: .minPixels)
            ?? nested?.minPixels
            ?? nested?.size?.shortestEdge
            ?? 3136
        maxPixels =
            try c.decodeIfPresent(Int.self, forKey: .maxPixels)
            ?? nested?.maxPixels
            ?? nested?.size?.longestEdge
            ?? 11_289_600
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? nested?.mergeSize ?? 2
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? nested?.patchSize ?? 14
        temporalPatchSize =
            try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize)
            ?? nested?.temporalPatchSize
            ?? 1
        imageProcessorType =
            try c.decodeIfPresent(String.self, forKey: .imageProcessorType)
            ?? nested?.imageProcessorType
    }
}

// MARK: - Processor

public struct DotsOCRProcessor: UserInputProcessor {
    private let config: DotsOCRProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: DotsOCRProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW,
    ) {
        let images = images.map { MediaProcessing.apply($0, processing: processing) }
        let size = images[0].extent.size
        let factor = config.patchSize * config.mergeSize
        let (resizedHeight, resizedWidth) = try QwenVL.targetSize(
            height: Int(size.height),
            width: Int(size.width),
            factor: factor,
            minPixels: config.minPixels,
            maxPixels: config.maxPixels,
        )
        let resizedSize = CGSize(width: resizedWidth, height: resizedHeight)

        let processedImages =
            images
            .map { MediaProcessing.inSRGBToneCurveSpace($0) }
            .map { MediaProcessing.resampleBicubic($0, to: resizedSize) }
            .map {
                MediaProcessing.normalize(
                    $0, mean: config.imageMeanTuple, std: config.imageStdTuple,
                )
            }
            .map { MediaProcessing.asMLXArray($0) }

        return try QwenVL.patchify(
            images: processedImages,
            mergeSize: config.mergeSize,
            patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize,
        )
    }

    private func replacePaddingTokens(in promptTokens: [Int], frames: [THW]) throws -> [Int] {
        let paddingToken = "<|imgpad|>"
        let placeholderTokens = tokenizer.encode(
            text: "<|img|>\(paddingToken)<|endofimg|>",
            addSpecialTokens: false,
        )
        let placeholderRanges = promptTokens.ranges(of: placeholderTokens)
        guard placeholderRanges.count == frames.count else {
            throw VLMError.processing(
                "Number of DotsOCR image placeholders (\(placeholderRanges.count)) does not match number of images (\(frames.count))",
            )
        }

        let mergeLength = config.mergeSize * config.mergeSize
        let replacementSequences = frames.map { frame in
            let paddingCount = frame.product / mergeLength
            return tokenizer.encode(
                text:
                    "<|img|>\(Array(repeating: paddingToken, count: paddingCount).joined())<|endofimg|>",
                addSpecialTokens: false,
            )
        }

        var result: [Int] = []
        var currentIndex = promptTokens.startIndex
        for (range, replacement) in zip(placeholderRanges, replacementSequences) {
            result.append(contentsOf: promptTokens[currentIndex ..< range.lowerBound])
            result.append(contentsOf: replacement)
            currentIndex = range.upperBound
        }
        if currentIndex < promptTokens.endIndex {
            result.append(contentsOf: promptTokens[currentIndex...])
        }
        return result
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = DotsOCRMessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext,
        )

        if input.images.isEmpty, input.videos.isEmpty {
            return LMInput(tokens: MLXArray(promptTokens))
        }

        guard input.videos.isEmpty else {
            throw VLMError.singleMediaTypeAllowed
        }

        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imagePixelsAndFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            processedImage = LMInput.ProcessedImage(
                pixels: concatenated(imagePixelsAndFrames.map(\.0)),
                frames: imagePixelsAndFrames.map(\.1),
            )

            if let frames = processedImage?.frames {
                promptTokens = try replacePaddingTokens(in: promptTokens, frames: frames)
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: promptArray, mask: ones(like: promptArray).asType(.int8)),
            image: processedImage,
        )
    }
}

// MARK: - Language

private enum DotsLanguage {
    fileprivate final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear
        @ModuleInfo(key: "rotary_emb") var rotaryEmbedding: RoPE

        init(_ config: DotsOCRConfiguration.TextConfiguration) {
            let dim = config.hiddenSize
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = dim / config.attentionHeads
            scale = pow(Float(headDim), -0.5)

            _wq.wrappedValue = Linear(dim, heads * headDim, bias: config.attentionBias)
            _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.attentionBias)
            _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.attentionBias)
            _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

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
            let (batch, length) = (x.dim(0), x.dim(1))
            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

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
            .reshaped(batch, length, -1)

            return wo(output)
        }
    }

    fileprivate final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "down_proj") var down: Linear
        @ModuleInfo(key: "up_proj") var up: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    fileprivate final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: DotsOCRConfiguration.TextConfiguration) {
            _attention.wrappedValue = Attention(config)
            mlp = MLP(
                dimensions: config.hiddenSize,
                hiddenDimensions: config.intermediateSize,
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
            var r = attention(inputLayerNorm(x), mask: mask, cache: cache)
            let h = x + r
            r = mlp(postAttentionLayerNorm(h))
            return h + r
        }
    }

    fileprivate final class Qwen2Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        fileprivate let layers: [DecoderLayer]
        fileprivate let norm: RMSNorm
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        init(_ config: DotsOCRConfiguration.TextConfiguration) {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            layers = (0 ..< config.hiddenLayers).map { _ in DecoderLayer(config) }
            norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
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
            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }

            h = norm(h)
            if let lmHead {
                return lmHead(h)
            }
            return embedTokens.asLinear(h)
        }
    }

    fileprivate final class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: Qwen2Model
        let kvHeads: [Int]

        init(_ config: DotsOCRConfiguration.TextConfiguration) {
            model = Qwen2Model(config)
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
            LMOutput(
                logits: model(inputs, cache: cache, inputEmbedding: inputEmbedding),
                state: state,
            )
        }
    }
}

// MARK: - Vision

private enum DotsVision {
    fileprivate static func applyRotaryPositionEmbedding(
        _ tensor: MLXArray,
        freqs: MLXArray,
    )
        -> MLXArray
    {
        var cos = MLX.cos(freqs)
        var sin = MLX.sin(freqs)

        cos = expandedDimensions(cos, axis: 1)
        cos = tiled(cos, repetitions: [1, 1, 2])
        cos = expandedDimensions(cos, axis: 0)

        sin = expandedDimensions(sin, axis: 1)
        sin = tiled(sin, repetitions: [1, 1, 2])
        sin = expandedDimensions(sin, axis: 0)

        return ((tensor * cos) + (RotaryEmbeddingOps.rotateHalf(tensor) * sin)).asType(tensor.dtype)
    }

    fileprivate final class RotaryEmbedding {
        let inverseFrequency: MLXArray

        init(dimensions: Int, theta: Float) {
            let p =
                MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
                / Float(dimensions)
            inverseFrequency = 1.0 / pow(theta, p)
        }

        func callAsFunction(sequenceLength: Int) -> MLXArray {
            let seq = MLXArray(0 ..< sequenceLength).asType(inverseFrequency.dtype)
            return outer(seq, inverseFrequency)
        }
    }

    fileprivate final class PatchMerger: Module, UnaryLayer {
        let hiddenSize: Int
        @ModuleInfo(key: "ln_q") var layerNormQ: LayerNorm
        @ModuleInfo var mlp: (Linear, GELU, Linear)

        init(dimensions: Int, contextDimensions: Int, spatialMergeSize: Int) {
            hiddenSize = contextDimensions * spatialMergeSize * spatialMergeSize
            _layerNormQ.wrappedValue = LayerNorm(dimensions: contextDimensions, eps: 1e-6)
            mlp = (
                Linear(hiddenSize, hiddenSize),
                GELU(),
                Linear(hiddenSize, dimensions),
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = layerNormQ(x).reshaped(-1, hiddenSize)
            h = mlp.0(h)
            h = mlp.1(h)
            h = mlp.2(h)
            return h
        }
    }

    fileprivate final class Attention: Module {
        let numHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear

        init(_ config: DotsOCRConfiguration.VisionConfiguration) {
            numHeads = config.attentionHeads
            headDim = config.embedDimension / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _qkv.wrappedValue = Linear(
                config.embedDimension,
                config.embedDimension * 3,
                bias: config.useBias,
            )
            _proj.wrappedValue = Linear(
                config.embedDimension,
                config.embedDimension,
                bias: config.useBias,
            )
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            cuSeqlens: [Int],
            rotaryPositionEmbedding: MLXArray,
        )
            -> MLXArray
        {
            let sequenceLength = hiddenStates.dim(0)
            let qkv = qkv(hiddenStates)
                .reshaped(sequenceLength, 3, numHeads, -1)
                .transposed(1, 0, 2, 3)
            let parts = split(qkv, parts: 3, axis: 0)
            var q = parts[0].squeezed(axis: 0)
            var k = parts[1].squeezed(axis: 0)
            let v = parts[2].squeezed(axis: 0)

            q =
                applyRotaryPositionEmbedding(
                    q[.newAxis, 0..., 0..., 0...],
                    freqs: rotaryPositionEmbedding,
                )[0]
            k =
                applyRotaryPositionEmbedding(
                    k[.newAxis, 0..., 0..., 0...],
                    freqs: rotaryPositionEmbedding,
                )[0]

            let qT = q.transposed(1, 0, 2).expandedDimensions(axis: 0)
            let kT = k.transposed(1, 0, 2).expandedDimensions(axis: 0)
            let vT = v.transposed(1, 0, 2).expandedDimensions(axis: 0)

            var outputs = [MLXArray]()
            for i in 0 ..< (cuSeqlens.count - 1) {
                let start = cuSeqlens[i]
                let end = cuSeqlens[i + 1]
                outputs.append(
                    MLXFast.scaledDotProductAttention(
                        queries: qT[0..., 0..., start ..< end, 0...],
                        keys: kT[0..., 0..., start ..< end, 0...],
                        values: vT[0..., 0..., start ..< end, 0...],
                        scale: scale,
                        mask: .none,
                    ),
                )
            }

            return proj(
                concatenated(outputs, axis: 2)
                    .transposed(0, 2, 1, 3)
                    .reshaped(sequenceLength, -1),
            )
        }
    }

    fileprivate final class SwiGLUFFN: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        @ModuleInfo var fc3: Linear

        init(_ config: DotsOCRConfiguration.VisionConfiguration) {
            _fc1.wrappedValue = Linear(
                config.embedDimension,
                config.intermediateSize,
                bias: config.useBias,
            )
            _fc2.wrappedValue = Linear(
                config.intermediateSize,
                config.embedDimension,
                bias: config.useBias,
            )
            _fc3.wrappedValue = Linear(
                config.embedDimension,
                config.intermediateSize,
                bias: config.useBias,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(silu(fc1(x)) * fc3(x))
        }
    }

    fileprivate final class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo var proj: Conv2d
        @ModuleInfo var norm: RMSNorm

        let channels: Int
        let patchSize: Int
        let temporalPatchSize: Int
        let embedDimension: Int

        init(_ config: DotsOCRConfiguration.VisionConfiguration) {
            channels = config.channels
            patchSize = config.patchSize
            temporalPatchSize = config.temporalPatchSize
            embedDimension = config.embedDimension
            _proj.wrappedValue = Conv2d(
                inputChannels: config.channels,
                outputChannels: config.embedDimension,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: true,
            )
            _norm.wrappedValue = RMSNorm(
                dimensions: config.embedDimension,
                eps: config.rmsNormEps,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = x.reshaped(
                -1,
                channels,
                temporalPatchSize,
                patchSize,
                patchSize,
            )
            h = h[0..., 0..., 0, 0..., 0...]
            h = h.transposed(0, 2, 3, 1)
            h = proj(h).reshaped(-1, embedDimension)
            return norm(h)
        }
    }

    fileprivate final class ViTPreprocessor: Module, UnaryLayer {
        @ModuleInfo var patchifier: PatchEmbed

        init(_ config: DotsOCRConfiguration.VisionConfiguration) {
            _patchifier.wrappedValue = PatchEmbed(config)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            patchifier(x)
        }
    }

    fileprivate final class Block: Module {
        @ModuleInfo var attn: Attention
        @ModuleInfo var norm1: RMSNorm
        @ModuleInfo var mlp: SwiGLUFFN
        @ModuleInfo var norm2: RMSNorm

        init(_ config: DotsOCRConfiguration.VisionConfiguration) {
            _attn.wrappedValue = Attention(config)
            _norm1.wrappedValue = RMSNorm(
                dimensions: config.embedDimension,
                eps: config.rmsNormEps,
            )
            _mlp.wrappedValue = SwiGLUFFN(config)
            _norm2.wrappedValue = RMSNorm(
                dimensions: config.embedDimension,
                eps: config.rmsNormEps,
            )
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            cuSeqlens: [Int],
            rotaryPositionEmbedding: MLXArray,
        )
            -> MLXArray
        {
            var h =
                hiddenStates
                + attn(
                    norm1(hiddenStates),
                    cuSeqlens: cuSeqlens,
                    rotaryPositionEmbedding: rotaryPositionEmbedding,
                )
            h = h + mlp(norm2(h))
            return h
        }
    }

    fileprivate final class VisionModel: Module {
        @ModuleInfo(key: "patch_embed") var patchEmbed: ViTPreprocessor
        @ModuleInfo(key: "blocks") var blocks: [Block]
        @ModuleInfo(key: "post_trunk_norm") var postTrunkNorm: RMSNorm?
        @ModuleInfo var merger: PatchMerger

        let spatialMergeSize: Int
        let rotaryPositionEmbedding: RotaryEmbedding

        init(_ config: DotsOCRConfiguration.VisionConfiguration) {
            spatialMergeSize = config.spatialMergeSize
            _patchEmbed.wrappedValue = ViTPreprocessor(config)
            _blocks.wrappedValue = (0 ..< config.hiddenLayers).map { _ in Block(config) }
            if config.postNorm {
                _postTrunkNorm.wrappedValue = RMSNorm(
                    dimensions: config.embedDimension,
                    eps: config.rmsNormEps,
                )
            }
            _merger.wrappedValue = PatchMerger(
                dimensions: config.hiddenSize,
                contextDimensions: config.embedDimension,
                spatialMergeSize: config.spatialMergeSize,
            )

            let headDimension = config.embedDimension / config.attentionHeads
            rotaryPositionEmbedding = RotaryEmbedding(
                dimensions: headDimension / 2,
                theta: 10000,
            )
        }

        private func rotaryPositionEmbedding(_ frames: [THW]) -> MLXArray {
            var positionIDs = [MLXArray]()
            for frame in frames {
                let (t, h, w) = frame.values

                var hposIDs = expandedDimensions(MLXArray(0 ..< h), axis: 1)
                hposIDs = repeated(hposIDs, count: w, axis: 1)
                hposIDs =
                    hposIDs
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize,
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                var wposIDs = expandedDimensions(MLXArray(0 ..< w), axis: 0)
                wposIDs = repeated(wposIDs, count: h, axis: 0)
                wposIDs =
                    wposIDs
                    .reshaped(
                        h / spatialMergeSize,
                        spatialMergeSize,
                        w / spatialMergeSize,
                        spatialMergeSize,
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()

                let stackedPosIDs = stacked([hposIDs, wposIDs], axis: -1)
                positionIDs.append(tiled(stackedPosIDs, repetitions: [t, 1]))
            }

            let indices = concatenated(positionIDs, axis: 0)
            let maxGridSize = frames.lazy.map { max($0.h, $0.w) }.max() ?? 0
            let rotaryPositionEmbeddingFull = rotaryPositionEmbedding(
                sequenceLength: maxGridSize,
            )[indices]
            return rotaryPositionEmbeddingFull.reshaped(indices.dim(0), -1)
        }

        func callAsFunction(_ hiddenStates: MLXArray, frames: [THW]) -> MLXArray {
            var h = patchEmbed(hiddenStates)
            let rotaryPositionEmbedding = rotaryPositionEmbedding(frames)

            var cuSeqlens = [0]
            for frame in frames {
                for _ in 0 ..< frame.t {
                    cuSeqlens.append(cuSeqlens.last! + frame.h * frame.w)
                }
            }

            for block in blocks {
                h = block(
                    h,
                    cuSeqlens: cuSeqlens,
                    rotaryPositionEmbedding: rotaryPositionEmbedding,
                )
            }

            if let postTrunkNorm {
                h = postTrunkNorm(h)
            }

            return merger(h)
        }

        private func isMLXConv2DWeight(_ array: MLXArray) -> Bool {
            guard array.ndim == 4 else {
                return false
            }
            let (outChannels, kernelHeight, kernelWidth) = (
                array.dim(0), array.dim(1), array.dim(2),
            )
            return outChannels >= kernelHeight && outChannels >= kernelWidth
                && kernelHeight == kernelWidth
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitizedWeights = [String: MLXArray]()
            for (key, value) in weights {
                if key.contains("position_ids") {
                    continue
                }
                if key.contains("vision_tower.patch_embed.patchifier.proj.weight"),
                    !isMLXConv2DWeight(value)
                {
                    sanitizedWeights[key] = value.transposed(0, 2, 3, 1)
                } else {
                    sanitizedWeights[key] = value
                }
            }
            return sanitizedWeights
        }
    }
}

// MARK: - Model

public final class DotsOCR: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionModel: DotsVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: DotsLanguage.LanguageModel

    public let config: DotsOCRConfiguration
    public var vocabularySize: Int {
        config.vocabularySize
    }

    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: DotsOCRConfiguration) {
        self.config = config
        _visionModel.wrappedValue = DotsVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = DotsLanguage.LanguageModel(config.textConfiguration)
    }

    private func inputEmbeddings(inputIDs: MLXArray, pixelValues: MLXArray?, frames: [THW]?)
        -> MLXArray
    {
        guard let pixelValues, let frames else {
            return languageModel.model.embedTokens(inputIDs)
        }

        let inputEmbeds = languageModel.model.embedTokens(inputIDs)
        var hiddenStates = visionModel(pixelValues, frames: frames)
        if hiddenStates.ndim == 2 {
            hiddenStates = hiddenStates[.newAxis, 0..., 0...]
        }

        return MultimodalEmbedding.mergeInputIDsWithImageFeatures(
            inputIDs: inputIDs,
            inputEmbeds: inputEmbeds,
            imageFeatures: hiddenStates,
            imageTokenID: config.imageTokenID,
            videoTokenID: config.videoTokenID,
        )
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let dtype = visionModel.patchEmbed.patchifier.proj.weight.dtype
        let frames = input.image?.frames ?? []
        let pixels = input.image?.pixels.asType(dtype)

        let embeddings = inputEmbeddings(
            inputIDs: input.text.tokens,
            pixelValues: pixels,
            frames: frames.isEmpty ? nil : frames,
        )
        let result = languageModel(nil, cache: cache, inputEmbedding: embeddings)
        prefill.progress?(embeddings.dim(1), embeddings.dim(1))
        return .logits(result)
    }

    public func callAsFunction(
        _ input: LMInput.Text,
        cache: [any KVCache]?,
        state: LMOutput.State?,
    )
        -> LMOutput
    {
        languageModel(input.tokens, cache: cache, state: state)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var transformed = [String: MLXArray]()
        for (key, value) in weights {
            var newKey = key
            if newKey.hasPrefix("model.vision_tower.") {
                newKey = newKey.replacingOccurrences(
                    of: "model.vision_tower.",
                    with: "vision_tower.",
                    options: [.anchored],
                )
            } else if newKey.hasPrefix("model.") {
                newKey = newKey.replacingOccurrences(
                    of: "model.",
                    with: "language_model.model.",
                    options: [.anchored],
                )
            } else if newKey.hasPrefix("lm_head.") {
                newKey = newKey.replacingOccurrences(
                    of: "lm_head.",
                    with: "language_model.model.lm_head.",
                    options: [.anchored],
                )
            }

            if newKey.contains("self_attn.rotary_emb.inv_freq") {
                continue
            }
            transformed[newKey] = value
        }

        if config.textConfiguration.tieWordEmbeddings {
            transformed["language_model.model.lm_head.weight"] = nil
        }

        return visionModel.sanitize(weights: transformed)
    }
}

// MARK: - Message Generator

public struct DotsOCRMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        [
            "role": message.role.rawValue,
            "content":
                message.images.map { _ in ["type": "image"] }
                + [["type": "text", "text": message.content]],
        ]
    }
}
