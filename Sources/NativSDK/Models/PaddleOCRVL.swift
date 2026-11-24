import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct PaddleOCRVLConfiguration: Decodable, Sendable {
    public struct RopeScaling: Decodable, Sendable {
        public let mropeSection: [Int]

        enum CodingKeys: String, CodingKey {
            case mropeSection = "mrope_section"
        }
    }

    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        private let _kvHeads: Int?
        public var kvHeads: Int {
            _kvHeads ?? attentionHeads
        }

        public let vocabularySize: Int
        private let _rmsNormEps: Float?
        public var rmsNormEps: Float {
            _rmsNormEps ?? 1e-5
        }

        private let _maxPositionEmbeddings: Int?
        public var maxPositionEmbeddings: Int {
            _maxPositionEmbeddings ?? 131_072
        }

        private let _ropeTheta: Float?
        public var ropeTheta: Float {
            _ropeTheta ?? 500_000
        }

        private let _useBias: Bool?
        public var useBias: Bool {
            _useBias ?? false
        }

        private let _headDim: Int?
        public var headDim: Int {
            _headDim ?? hiddenSize / attentionHeads
        }

        private let _ropeScaling: RopeScaling?
        public var mropeSection: [Int] {
            _ropeScaling?.mropeSection ?? [16, 24, 24]
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case _kvHeads = "num_key_value_heads"
            case vocabularySize = "vocab_size"
            case _rmsNormEps = "rms_norm_eps"
            case _maxPositionEmbeddings = "max_position_embeddings"
            case _ropeTheta = "rope_theta"
            case _useBias = "use_bias"
            case _headDim = "head_dim"
            case _ropeScaling = "rope_scaling"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "paddleocr_vl"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 18
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            _kvHeads = try c.decodeIfPresent(Int.self, forKey: ._kvHeads)
            vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 103_424
            _rmsNormEps = try c.decodeIfPresent(Float.self, forKey: ._rmsNormEps)
            _maxPositionEmbeddings = try c.decodeIfPresent(
                Int.self, forKey: ._maxPositionEmbeddings)
            _ropeTheta = try c.decodeIfPresent(Float.self, forKey: ._ropeTheta)
            _useBias = try c.decodeIfPresent(Bool.self, forKey: ._useBias)
            _headDim = try c.decodeIfPresent(Int.self, forKey: ._headDim)
            _ropeScaling = try c.decodeIfPresent(RopeScaling.self, forKey: ._ropeScaling)
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        private let _channels: Int?
        public var channels: Int {
            _channels ?? 3
        }

        private let _imageSize: Int?
        public var imageSize: Int {
            _imageSize ?? 384
        }

        public let patchSize: Int
        private let _layerNormEps: Float?
        public var layerNormEps: Float {
            _layerNormEps ?? 1e-6
        }

        private let _spatialMergeSize: Int?
        public var spatialMergeSize: Int {
            _spatialMergeSize ?? 2
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case _channels = "num_channels"
            case _imageSize = "image_size"
            case patchSize = "patch_size"
            case _layerNormEps = "layer_norm_eps"
            case _spatialMergeSize = "spatial_merge_size"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "paddleocr_vl"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 27
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            _channels = try c.decodeIfPresent(Int.self, forKey: ._channels)
            _imageSize = try c.decodeIfPresent(Int.self, forKey: ._imageSize)
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            _layerNormEps = try c.decodeIfPresent(Float.self, forKey: ._layerNormEps)
            _spatialMergeSize = try c.decodeIfPresent(Int.self, forKey: ._spatialMergeSize)
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    public let vocabularySize: Int
    public let imageTokenID: Int
    public let videoTokenID: Int
    public let visionStartTokenID: Int
    public let visionEndTokenID: Int
    public let eosTokenID: Int

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case imageTokenID = "image_token_id"
        case videoTokenID = "video_token_id"
        case visionStartTokenID = "vision_start_token_id"
        case visionEndTokenID = "vision_end_token_id"
        case eosTokenID = "eos_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "paddleocr_vl"
        textConfiguration =
            try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? TextConfiguration(from: decoder)
        visionConfiguration = try c.decode(
            VisionConfiguration.self, forKey: .visionConfiguration,
        )
        vocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .vocabularySize)
            ?? textConfiguration.vocabularySize
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? 100_295
        videoTokenID = try c.decodeIfPresent(Int.self, forKey: .videoTokenID) ?? 101_307
        visionStartTokenID =
            try c.decodeIfPresent(Int.self, forKey: .visionStartTokenID) ?? 101_305
        visionEndTokenID =
            try c.decodeIfPresent(Int.self, forKey: .visionEndTokenID) ?? 101_306
        eosTokenID = try c.decodeIfPresent(Int.self, forKey: .eosTokenID) ?? 2
    }
}

public struct PaddleOCRVLProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String?
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let minPixels: Int
    public let maxPixels: Int
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let imageTokenID: Int

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
        case imageTokenID = "image_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass = try c.decodeIfPresent(String.self, forKey: .processorClass)
        imageMean = try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd = try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        minPixels = try c.decodeIfPresent(Int.self, forKey: .minPixels) ?? 112_896
        maxPixels = try c.decodeIfPresent(Int.self, forKey: .maxPixels) ?? 1_003_520
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        temporalPatchSize =
            try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 1
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? 100_295
    }
}

// MARK: - Shared Helpers

private enum PaddleOCRVLSupport {
    static func targetSize(
        height originalHeight: Int,
        width originalWidth: Int,
        factor: Int,
        minPixels: Int,
        maxPixels: Int,
    )
        throws -> (height: Int, width: Int)
    {
        var height = originalHeight
        var width = originalWidth

        if height < factor {
            width = Int(round((Double(width) * Double(factor)) / Double(max(height, 1))))
            height = factor
        }
        if width < factor {
            height = Int(round((Double(height) * Double(factor)) / Double(max(width, 1))))
            width = factor
        }
        if max(height, width) / min(height, width) > 200 {
            throw VLMError.imageProcessingFailure(
                "Absolute aspect ratio must be smaller than 200: \(width) x \(height)",
            )
        }

        var targetHeight = Int(round(Double(height) / Double(factor))) * factor
        var targetWidth = Int(round(Double(width) / Double(factor))) * factor

        if targetHeight * targetWidth > maxPixels {
            let beta = sqrt(Double(height * width) / Double(maxPixels))
            targetHeight = Int(floor(Double(height) / beta / Double(factor))) * factor
            targetWidth = Int(floor(Double(width) / beta / Double(factor))) * factor
        } else if targetHeight * targetWidth < minPixels {
            let beta = sqrt(Double(minPixels) / Double(height * width))
            targetHeight = Int(ceil(Double(height) * beta / Double(factor))) * factor
            targetWidth = Int(ceil(Double(width) * beta / Double(factor))) * factor
        }

        targetHeight = max(factor, targetHeight)
        targetWidth = max(factor, targetWidth)
        return (targetHeight, targetWidth)
    }

    static func mergeInputIDsWithImageFeatures(
        inputIDs: MLXArray,
        inputEmbeds: MLXArray,
        imageFeatures: MLXArray,
        imageTokenID: Int,
    )
        -> MLXArray
    {
        let batchSize = inputIDs.dim(0)
        let result = inputEmbeds
        var featureStart = 0

        for batchIndex in 0 ..< batchSize {
            let tokens = inputIDs[batchIndex].asArray(Int.self)
            let indices = tokens.enumerated().compactMap { index, token in
                token == imageTokenID ? index : nil
            }
            guard !indices.isEmpty else {
                continue
            }

            let batchFeatures = imageFeatures[featureStart ..< featureStart + indices.count, 0...]
            result[batchIndex, MLXArray(indices), 0...] = batchFeatures
            featureStart += indices.count
        }

        return result
    }
}

// MARK: - Processor

public struct PaddleOCRVLProcessor: UserInputProcessor {
    private let config: PaddleOCRVLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: PaddleOCRVLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        guard input.videos.isEmpty else {
            throw VLMError.singleMediaTypeAllowed
        }

        let prompt = Self.prompt(from: input)

        if images.isEmpty {
            let text = Self.renderPrompt(prompt: prompt, imageTokenCounts: [])
            let tokens = tokenizer.encode(text: text, addSpecialTokens: false)
            let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
            return LMInput(
                text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)))
        }

        let imagePixelsAndFrames = try images.map {
            try preprocess(image: $0.asCIImage(), processing: input.processing)
        }
        let frames = imagePixelsAndFrames.map(\.1)
        let imageTokenCounts = frames.map { frame in
            frame.product / (config.mergeSize * config.mergeSize)
        }
        let text = Self.renderPrompt(prompt: prompt, imageTokenCounts: imageTokenCounts)
        let tokens = tokenizer.encode(text: text, addSpecialTokens: false)
        let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)

        return LMInput(
            text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)),
            image: .init(
                pixels: concatenated(imagePixelsAndFrames.map(\.0)),
                frames: frames,
            ),
        )
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?) throws -> (
        MLXArray, THW,
    ) {
        let image = MediaProcessing.apply(image, processing: processing)
        let size = image.extent.size
        let factor = config.patchSize * config.mergeSize
        let target = try PaddleOCRVLSupport.targetSize(
            height: Int(size.height),
            width: Int(size.width),
            factor: factor,
            minPixels: config.minPixels,
            maxPixels: config.maxPixels,
        )
        let targetSize = CGSize(width: target.width, height: target.height)
        let processed = image.toSRGB().resampled(to: targetSize, method: .bicubic)
        let normalized = normalizedCHWArray(processed)
        let patches = patchify(normalized)

        return (
            patches,
            THW(1, target.height / config.patchSize, target.width / config.patchSize),
        )
    }

    private func normalizedCHWArray(_ image: CIImage) -> MLXArray {
        let width = Int(image.extent.width.rounded())
        let height = Int(image.extent.height.rounded())
        let componentsPerPixel = 4
        let bytesPerPixel = componentsPerPixel * MemoryLayout<Float32>.size
        let bytesPerRow = width * bytesPerPixel
        var data = Data(count: width * height * bytesPerPixel)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)

        data.withUnsafeMutableBytes { pointer in
            CIContext().render(
                image,
                toBitmap: pointer.baseAddress!,
                rowBytes: bytesPerRow,
                bounds: image.extent,
                format: .RGBAf,
                colorSpace: colorSpace,
            )
        }

        let rgba = MLXArray(data, [height, width, 4], type: Float32.self)
        let rgb = rgba[0..., 0..., ..<3]
        let alpha = maximum(rgba[0..., 0..., 3...], MLXArray(Float32(1e-6)))
        let unpremultiplied = clip(rgb / alpha, min: Float32(0), max: Float32(1))
        let mean = MLXArray(config.imageMean.map { Float32($0) })
        let std = MLXArray(config.imageStd.map { Float32($0) })
        return ((unpremultiplied - mean) / std).transposed(2, 0, 1)
    }

    private func patchify(_ image: MLXArray) -> MLXArray {
        let channels = image.dim(0)
        let height = image.dim(1)
        let width = image.dim(2)
        let gridH = height / config.patchSize
        let gridW = width / config.patchSize

        var patches = image.reshaped(
            channels,
            gridH,
            config.patchSize,
            gridW,
            config.patchSize,
        )
        patches = patches.transposed(1, 3, 0, 2, 4)
        return patches.reshaped(gridH * gridW, channels, config.patchSize, config.patchSize)
    }

    private static func renderPrompt(prompt: String, imageTokenCounts: [Int]) -> String {
        let imageContent = imageTokenCounts.map { count in
            "<|IMAGE_START|>\(String(repeating: "<|IMAGE_PLACEHOLDER|>", count: count))<|IMAGE_END|>"
        }.joined()
        return "<|begin_of_sentence|>User: \(imageContent)\(prompt)\nAssistant:\n"
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

}

// MARK: - Language

private enum PaddleOCRVLLanguage {
    fileprivate final class RotaryEmbedding {
        let mropeSection: [Int]
        let invFreq: MLXArray

        init(_ config: PaddleOCRVLConfiguration.TextConfiguration) {
            mropeSection = config.mropeSection
            let dim = config.headDim
            let p =
                MLXArray(stride(from: 0, to: dim, by: 2)).asType(.int64).asType(.float32)
                / Float(dim)
            invFreq = 1.0 / pow(config.ropeTheta, p)
        }

        func callAsFunction(_ x: MLXArray, positionIDs: MLXArray) -> (MLXArray, MLXArray) {
            var chunks = [MLXArray]()
            var start = 0
            for (axis, length) in mropeSection.enumerated() {
                let end = start + length
                let positions = positionIDs[axis, 0..., 0...].asType(.float32)
                let invChunk = invFreq[start ..< end]
                chunks.append(positions[0..., 0..., .newAxis] * invChunk)
                start = end
            }
            let freqs = concatenated(chunks, axis: -1)
            let emb = concatenated([freqs, freqs], axis: -1)
            return (MLX.cos(emb).asType(x.dtype), MLX.sin(emb).asType(x.dtype))
        }
    }

    static func applyRotaryPosEmb(
        q: MLXArray,
        k: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
    )
        -> (MLXArray, MLXArray)
    {
        let cos = cos[0..., .newAxis, 0..., 0...]
        let sin = sin[0..., .newAxis, 0..., 0...]
        let rotaryDim = cos.dim(-1)
        let qRot = q[.ellipsis, ..<rotaryDim]
        let qPass = q[.ellipsis, rotaryDim...]
        let kRot = k[.ellipsis, ..<rotaryDim]
        let kPass = k[.ellipsis, rotaryDim...]

        let qEmbed = (qRot * cos) + (RotaryEmbeddingOps.rotateHalf(qRot) * sin)
        let kEmbed = (kRot * cos) + (RotaryEmbeddingOps.rotateHalf(kRot) * sin)
        return (
            concatenated([qEmbed.asType(q.dtype), qPass], axis: -1),
            concatenated([kEmbed.asType(k.dtype), kPass], axis: -1),
        )
    }

    fileprivate final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        init(_ config: PaddleOCRVLConfiguration.TextConfiguration) {
            let dim = config.hiddenSize
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)

            _wq.wrappedValue = Linear(dim, heads * headDim, bias: config.useBias)
            _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.useBias)
            _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: config.useBias)
            _wo.wrappedValue = Linear(heads * headDim, dim, bias: config.useBias)
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
            positionEmbeddings: (MLXArray, MLXArray),
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

            let (cos, sin) = positionEmbeddings
            (queries, keys) = applyRotaryPosEmb(q: queries, k: keys, cos: cos, sin: sin)

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

        init(_ config: PaddleOCRVLConfiguration.TextConfiguration) {
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
            positionEmbeddings: (MLXArray, MLXArray),
        )
            -> MLXArray
        {
            var r = attention(
                inputLayerNorm(x),
                mask: mask,
                cache: cache,
                positionEmbeddings: positionEmbeddings,
            )
            let h = x + r
            r = mlp(postAttentionLayerNorm(h))
            return h + r
        }
    }

    fileprivate final class TextModel: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        fileprivate let layers: [DecoderLayer]
        fileprivate let norm: RMSNorm
        let rotaryEmbedding: RotaryEmbedding

        init(_ config: PaddleOCRVLConfiguration.TextConfiguration) {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            layers = (0 ..< config.hiddenLayers).map { _ in DecoderLayer(config) }
            norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            rotaryEmbedding = RotaryEmbedding(config)
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
            positionIDs: MLXArray? = nil,
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

            let posIDs: MLXArray
            if let positionIDs {
                posIDs = positionIDs
            } else {
                let offset = cache?.first?.offset ?? 0
                let seqLen = h.dim(h.ndim - 2)
                let positions = MLXArray(Int32(offset) ..< Int32(offset + seqLen))
                    .expandedDimensions(axis: 0)
                posIDs = tiled(positions, repetitions: [3, h.dim(0), 1])
            }

            let positionEmbeddings = rotaryEmbedding(h, positionIDs: posIDs)
            let mask = createAttentionMask(h: h, cache: cache?.first)

            for (i, layer) in layers.enumerated() {
                h = layer(
                    h,
                    mask: mask,
                    cache: cache?[i],
                    positionEmbeddings: positionEmbeddings,
                )
            }

            return norm(h)
        }
    }

    fileprivate final class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: TextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear
        let kvHeads: [Int]
        var positionIDs: MLXArray?
        var ropeDeltas: MLXArray?

        init(_ config: PaddleOCRVLConfiguration.TextConfiguration) {
            model = TextModel(config)
            _lmHead.wrappedValue = Linear(
                config.hiddenSize,
                config.vocabularySize,
                bias: false,
            )
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
            let cacheOffset = cache?.first?.offset ?? 0
            let seqLen =
                inputEmbedding?.dim((inputEmbedding?.ndim ?? 0) - 2)
                ?? inputs?.dim((inputs?.ndim ?? 0) - 1)
                ?? 0
            var currentPositionIDs: MLXArray?

            if let positionIDs {
                let storedLen = positionIDs.dim(2)
                if cacheOffset + seqLen <= storedLen {
                    currentPositionIDs =
                        positionIDs[0..., 0..., cacheOffset ..< (cacheOffset + seqLen)]
                } else {
                    let batchSize = inputEmbedding?.dim(0) ?? inputs?.dim(0) ?? 1
                    let base = MLXArray(Int32(cacheOffset) ..< Int32(cacheOffset + seqLen))
                        .expandedDimensions(axis: 0)
                    let delta = ropeDeltas ?? MLXArray(Int32(0))
                    let positions = tiled(base, repetitions: [batchSize, 1]) + delta
                    currentPositionIDs = tiled(
                        positions.expandedDimensions(axis: 0),
                        repetitions: [3, 1, 1],
                    )
                }
            }

            var out = model(
                inputs,
                cache: cache,
                inputEmbedding: inputEmbedding,
                positionIDs: currentPositionIDs,
            )
            out = lmHead(out)
            return LMOutput(logits: out, state: state)
        }
    }
}

// MARK: - Vision

private enum PaddleOCRVLVision {
    static func applyRotaryPositionEmbedding(_ tensor: MLXArray, freqs: MLXArray) -> MLXArray {
        var cos = MLX.cos(freqs)
        var sin = MLX.sin(freqs)

        cos = expandedDimensions(cos, axis: 1)
        cos = tiled(cos, repetitions: [1, 1, 2])
        cos = expandedDimensions(cos, axis: 0)

        sin = expandedDimensions(sin, axis: 1)
        sin = tiled(sin, repetitions: [1, 1, 2])
        sin = expandedDimensions(sin, axis: 0)

        return ((tensor * cos) + (RotaryEmbeddingOps.rotateHalf(tensor) * sin))
            .asType(tensor.dtype)
    }

    fileprivate final class VisionRotaryEmbedding {
        let inverseFrequency: MLXArray

        init(dimensions: Int, theta: Float = 10000) {
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

    fileprivate final class Embeddings: Module {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        let patchSize: Int
        let embedDim: Int
        let numPositions: Int

        init(_ config: PaddleOCRVLConfiguration.VisionConfiguration) {
            patchSize = config.patchSize
            embedDim = config.hiddenSize
            numPositions =
                (config.imageSize / config.patchSize)
                * (config.imageSize / config.patchSize)

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

        func interpolatedPositionEmbedding(height: Int, width: Int) -> MLXArray {
            let positionIDs = MLXArray(0 ..< numPositions)
            let table = positionEmbedding(positionIDs)
            let sourceSize = Int(Double(numPositions).squareRoot())
            if height == sourceSize, width == sourceSize {
                return table
            }

            var indices00 = [Int]()
            var indices01 = [Int]()
            var indices10 = [Int]()
            var indices11 = [Int]()
            var weights00 = [Float32]()
            var weights01 = [Float32]()
            var weights10 = [Float32]()
            var weights11 = [Float32]()
            indices00.reserveCapacity(height * width)
            indices01.reserveCapacity(height * width)
            indices10.reserveCapacity(height * width)
            indices11.reserveCapacity(height * width)
            weights00.reserveCapacity(height * width)
            weights01.reserveCapacity(height * width)
            weights10.reserveCapacity(height * width)
            weights11.reserveCapacity(height * width)

            for y in 0 ..< height {
                let sourceY =
                    height == 1
                    ? 0
                    : Double(y) * Double(sourceSize - 1) / Double(height - 1)
                let y0 = Int(floor(sourceY))
                let y1 = min(y0 + 1, sourceSize - 1)
                let wy = Float32(sourceY - Double(y0))
                for x in 0 ..< width {
                    let sourceX =
                        width == 1
                        ? 0
                        : Double(x) * Double(sourceSize - 1) / Double(width - 1)
                    let x0 = Int(floor(sourceX))
                    let x1 = min(x0 + 1, sourceSize - 1)
                    let wx = Float32(sourceX - Double(x0))

                    indices00.append(y0 * sourceSize + x0)
                    indices01.append(y0 * sourceSize + x1)
                    indices10.append(y1 * sourceSize + x0)
                    indices11.append(y1 * sourceSize + x1)
                    weights00.append((1 - wy) * (1 - wx))
                    weights01.append((1 - wy) * wx)
                    weights10.append(wy * (1 - wx))
                    weights11.append(wy * wx)
                }
            }

            let w00 = MLXArray(weights00).expandedDimensions(axis: 1)
            let w01 = MLXArray(weights01).expandedDimensions(axis: 1)
            let w10 = MLXArray(weights10).expandedDimensions(axis: 1)
            let w11 = MLXArray(weights11).expandedDimensions(axis: 1)
            return table[MLXArray(indices00)] * w00
                + table[MLXArray(indices01)] * w01
                + table[MLXArray(indices10)] * w10
                + table[MLXArray(indices11)] * w11
        }

        private func patchEmbed(_ hiddenStates: MLXArray) -> MLXArray {
            var states = hiddenStates
            if states.ndim == 5 {
                states = states.reshaped(
                    states.dim(0) * states.dim(1),
                    states.dim(2),
                    states.dim(3),
                    states.dim(4),
                )
            }
            let sequenceLength = states.dim(0)
            states = states.transposed(0, 2, 3, 1)
            return patchEmbedding(states).reshaped(sequenceLength, embedDim)
        }

        func callAsFunction(_ hiddenStates: MLXArray, frames: [THW]) -> MLXArray {
            let embeddings = patchEmbed(hiddenStates)
            var start = 0
            var chunks = [MLXArray]()
            for frame in frames {
                let length = frame.product
                let imageEmbedding = embeddings[start ..< (start + length), 0...]
                let positionEmbedding = interpolatedPositionEmbedding(
                    height: frame.h, width: frame.w)
                chunks.append(imageEmbedding + positionEmbedding)
                start += length
            }
            return concatenated(chunks, axis: 0)
        }
    }

    fileprivate final class Projector: Module {
        let spatialMergeSize: Int
        @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm
        @ModuleInfo(key: "linear_1") var linear1: Linear
        @ModuleInfo var act: GELU
        @ModuleInfo(key: "linear_2") var linear2: Linear

        init(dim: Int, contextDim: Int, spatialMergeSize: Int) {
            self.spatialMergeSize = spatialMergeSize
            let hiddenSize = dim * spatialMergeSize * spatialMergeSize
            _preNorm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
            _linear1.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
            act = GELU()
            _linear2.wrappedValue = Linear(hiddenSize, contextDim, bias: true)
        }

        private func projectImage(_ x: MLXArray, frame: THW) -> MLXArray {
            var x = preNorm(x)
            let dim = x.dim(-1)
            let hBlock = frame.h / spatialMergeSize
            let wBlock = frame.w / spatialMergeSize
            x = x.reshaped(
                frame.t,
                hBlock,
                spatialMergeSize,
                wBlock,
                spatialMergeSize,
                dim,
            )
            x = x.transposed(0, 1, 3, 2, 4, 5)
            x = x.reshaped(
                frame.t * hBlock * wBlock,
                spatialMergeSize * spatialMergeSize * dim,
            )
            return linear2(act(linear1(x)))
        }

        func callAsFunction(_ x: MLXArray, frames: [THW]) -> MLXArray {
            var start = 0
            var chunks = [MLXArray]()
            for frame in frames {
                let length = frame.product
                chunks.append(projectImage(x[start ..< (start + length), 0...], frame: frame))
                start += length
            }
            return concatenated(chunks, axis: 0)
        }
    }

    fileprivate final class Attention: Module {
        let numHeads: Int
        let headDim: Int
        let scale: Float
        @ModuleInfo var qkv: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: PaddleOCRVLConfiguration.VisionConfiguration) {
            numHeads = config.attentionHeads
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _qkv.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * 3, bias: true)
            _outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
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
                let output = MLXFast.scaledDotProductAttention(
                    queries: qT[0..., 0..., start ..< end, 0...],
                    keys: kT[0..., 0..., start ..< end, 0...],
                    values: vT[0..., 0..., start ..< end, 0...],
                    scale: scale,
                    mask: .none,
                )
                outputs.append(output)
            }

            let output = concatenated(outputs, axis: 2)
                .transposed(0, 2, 1, 3)
                .reshaped(sequenceLength, -1)
            return outProj(output)
        }
    }

    fileprivate final class MLP: Module, UnaryLayer {
        @ModuleInfo var activation: GELU
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        init(_ config: PaddleOCRVLConfiguration.VisionConfiguration) {
            activation = GELU()
            fc1 = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            fc2 = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    fileprivate final class EncoderLayer: Module {
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo var mlp: MLP

        init(_ config: PaddleOCRVLConfiguration.VisionConfiguration) {
            _layerNorm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _layerNorm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _selfAttention.wrappedValue = Attention(config)
            mlp = MLP(config)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            cuSeqlens: [Int],
            rotaryPositionEmbedding: MLXArray,
        )
            -> MLXArray
        {
            var hiddenStates =
                hiddenStates
                + selfAttention(
                    layerNorm1(hiddenStates),
                    cuSeqlens: cuSeqlens,
                    rotaryPositionEmbedding: rotaryPositionEmbedding,
                )
            hiddenStates = hiddenStates + mlp(layerNorm2(hiddenStates))
            return hiddenStates
        }
    }

    fileprivate final class VisionModel: Module {
        @ModuleInfo var embeddings: Embeddings
        @ModuleInfo var layers: [EncoderLayer]
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm
        @ModuleInfo var projector: Projector

        let rotaryEmbedding: VisionRotaryEmbedding

        init(_ config: PaddleOCRVLConfiguration.VisionConfiguration) {
            embeddings = Embeddings(config)
            let headDim = config.hiddenSize / config.attentionHeads
            rotaryEmbedding = VisionRotaryEmbedding(dimensions: headDim / 2)
            layers = (0 ..< config.hiddenLayers).map { _ in EncoderLayer(config) }
            _postLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
            projector = Projector(
                dim: config.hiddenSize,
                contextDim: 1024,
                spatialMergeSize: config.spatialMergeSize,
            )
        }

        var patchDType: DType {
            embeddings.patchEmbedding.weight.dtype
        }

        func rotaryPositionEmbedding(frames: [THW]) -> MLXArray {
            var positionIDs = [MLXArray]()
            for frame in frames {
                let count = frame.product
                var hids = [Int32]()
                var wids = [Int32]()
                hids.reserveCapacity(count)
                wids.reserveCapacity(count)
                for position in 0 ..< count {
                    let imagePosition = position % (frame.h * frame.w)
                    hids.append(Int32(imagePosition / frame.w))
                    wids.append(Int32(imagePosition % frame.w))
                }
                positionIDs.append(stacked([MLXArray(hids), MLXArray(wids)], axis: -1))
            }

            let indices = concatenated(positionIDs, axis: 0)
            let maxGridSize = frames.lazy.map { max($0.h, $0.w) }.max() ?? 1
            return rotaryEmbedding(sequenceLength: maxGridSize)[indices]
                .reshaped(indices.dim(0), -1)
        }

        func cuSeqlens(frames: [THW]) -> [Int] {
            var result = [0]
            var total = 0
            for frame in frames {
                for _ in 0 ..< frame.t {
                    total += frame.h * frame.w
                    result.append(total)
                }
            }
            return result
        }

        func callAsFunction(_ hiddenStates: MLXArray, frames: [THW]) -> MLXArray {
            var hiddenStates = embeddings(hiddenStates, frames: frames)
            let rotary = rotaryPositionEmbedding(frames: frames)
            let cuSeqlens = cuSeqlens(frames: frames)

            for layer in layers {
                hiddenStates = layer(
                    hiddenStates,
                    cuSeqlens: cuSeqlens,
                    rotaryPositionEmbedding: rotary,
                )
            }

            hiddenStates = postLayerNorm(hiddenStates)
            return projector(hiddenStates, frames: frames)
        }

        private func isMLXConvWeight(_ array: MLXArray) -> Bool {
            guard array.ndim == 4 else {
                return false
            }
            let outChannels = array.dim(0)
            let kH = array.dim(1)
            let kW = array.dim(2)
            let maybeChannels = array.dim(3)
            return outChannels >= kH && outChannels >= kW && kH == kW && maybeChannels <= kH
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized = [String: MLXArray]()
            for (key, value) in weights {
                if key.contains("position_ids") {
                    continue
                }
                if key.contains("patch_embedding.weight"), !isMLXConvWeight(value), value.ndim == 4
                {
                    sanitized[key] = value.transposed(0, 2, 3, 1)
                } else {
                    sanitized[key] = value
                }
            }
            return sanitized
        }
    }
}

// MARK: - Model

public final class PaddleOCRVL: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo private var visual: PaddleOCRVLVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: PaddleOCRVLLanguage.LanguageModel

    public let config: PaddleOCRVLConfiguration
    public var vocabularySize: Int {
        config.vocabularySize
    }

    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: PaddleOCRVLConfiguration) {
        self.config = config
        visual = PaddleOCRVLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = PaddleOCRVLLanguage.LanguageModel(
            config.textConfiguration,
        )
    }

    public func prepare(
        _ input: LMInput, cache: [any KVCache], state _: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        guard let pixels = input.image?.pixels, let frames = input.image?.frames, !frames.isEmpty
        else {
            languageModel.positionIDs = nil
            languageModel.ropeDeltas = nil
            defer { prefill.progress?(input.text.tokens.size, input.text.tokens.size) }
            return .logits(languageModel(input.text.tokens, cache: cache))
        }

        let pixelValues = pixels.asType(visual.patchDType)
        let inputEmbeds = languageModel.model.embedTokens(input.text.tokens)
        let imageFeatures = visual(pixelValues, frames: frames)
        let embeddings = PaddleOCRVLSupport.mergeInputIDsWithImageFeatures(
            inputIDs: input.text.tokens,
            inputEmbeds: inputEmbeds,
            imageFeatures: imageFeatures,
            imageTokenID: config.imageTokenID,
        )
        let (positionIDs, ropeDeltas) = getRopeIndex(inputIDs: input.text.tokens, frames: frames)
        languageModel.positionIDs = positionIDs
        languageModel.ropeDeltas = ropeDeltas

        defer { prefill.progress?(embeddings.dim(1), embeddings.dim(1)) }
        return .logits(languageModel(nil, cache: cache, inputEmbedding: embeddings))
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

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    private func getRopeIndex(inputIDs: MLXArray, frames: [THW]) -> (MLXArray, MLXArray) {
        let batchSize = inputIDs.dim(0)
        let seqLength = inputIDs.dim(1)
        precondition(batchSize == 1, "PaddleOCRVL getRopeIndex only supports batchSize == 1")

        let positionIDs = zeros([3, batchSize, seqLength], type: Int32.self)
        var frameIndex = 0
        var ropeDelta: Int32 = 0

        for batchIndex in 0 ..< batchSize {
            let tokens = inputIDs[batchIndex].asArray(Int.self)
            var dimT = [Int32]()
            var dimH = [Int32]()
            var dimW = [Int32]()
            dimT.reserveCapacity(tokens.count)
            dimH.reserveCapacity(tokens.count)
            dimW.reserveCapacity(tokens.count)

            var start = 0
            var lastMax: Int32 = -1

            func appendText(_ count: Int) {
                guard count > 0 else {
                    return
                }
                let base = lastMax + 1
                for offset in 0 ..< count {
                    let value = base + Int32(offset)
                    dimT.append(value)
                    dimH.append(value)
                    dimW.append(value)
                }
                lastMax = base + Int32(count) - 1
            }

            while frameIndex < frames.count,
                let imageIndex = tokens[start...].firstIndex(of: config.imageTokenID)
            {
                let frame = frames[frameIndex]
                frameIndex += 1

                let llmGridT = frame.t
                let llmGridH = frame.h / config.visionConfiguration.spatialMergeSize
                let llmGridW = frame.w / config.visionConfiguration.spatialMergeSize
                appendText(imageIndex - start)

                let base = lastMax + 1
                for t in 0 ..< llmGridT {
                    for h in 0 ..< llmGridH {
                        for w in 0 ..< llmGridW {
                            dimT.append(base + Int32(t))
                            dimH.append(base + Int32(h))
                            dimW.append(base + Int32(w))
                        }
                    }
                }
                lastMax = max(
                    base + Int32(llmGridT - 1),
                    max(base + Int32(llmGridH - 1), base + Int32(llmGridW - 1)),
                )
                start = imageIndex + llmGridT * llmGridH * llmGridW
            }

            appendText(tokens.count - start)
            positionIDs[0, batchIndex] = MLXArray(dimT)
            positionIDs[1, batchIndex] = MLXArray(dimH)
            positionIDs[2, batchIndex] = MLXArray(dimW)
            ropeDelta = lastMax + 1 - Int32(tokens.count)
        }

        return (positionIDs, MLXArray([ropeDelta]).reshaped(1, 1))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var transformed = [String: MLXArray]()

        for (key, value) in weights {
            if key.contains("packing_position_embedding")
                || key.contains("vision_model.head")
                || key.contains("position_ids")
            {
                continue
            }

            if key.contains("visual"),
                key.contains(".k_proj.")
                    || key.contains("visual"), key.contains(".v_proj.")
            {
                continue
            }

            if key.contains("visual"), key.contains(".q_proj.") {
                let kKey = key.replacingOccurrences(of: ".q_proj.", with: ".k_proj.")
                let vKey = key.replacingOccurrences(of: ".q_proj.", with: ".v_proj.")
                guard let kValue = weights[kKey], let vValue = weights[vKey] else {
                    continue
                }
                let merged = concatenated([value, kValue, vValue], axis: 0)
                let mergedKey = transformKey(key)
                    .replacingOccurrences(of: ".q_proj.", with: ".qkv.")
                transformed[mergedKey] = merged
                continue
            }

            transformed[transformKey(key)] = value
        }

        return visual.sanitize(weights: transformed)
    }

    private func transformKey(_ key: String) -> String {
        var key = key
        if key.hasPrefix("visual.vision_model.encoder.") {
            key = key.replacingOccurrences(
                of: "visual.vision_model.encoder.",
                with: "visual.",
                options: [.anchored],
            )
        } else if key.hasPrefix("visual.vision_model.") {
            key = key.replacingOccurrences(
                of: "visual.vision_model.",
                with: "visual.",
                options: [.anchored],
            )
        } else if key.hasPrefix("mlp_AR.") {
            key = key.replacingOccurrences(
                of: "mlp_AR.",
                with: "visual.projector.",
                options: [.anchored],
            )
        } else if key.hasPrefix("model.") {
            key = key.replacingOccurrences(
                of: "model.",
                with: "language_model.model.",
                options: [.anchored],
            )
        } else if key.hasPrefix("lm_head.") {
            key = key.replacingOccurrences(
                of: "lm_head.",
                with: "language_model.lm_head.",
                options: [.anchored],
            )
        }
        return key
    }
}
