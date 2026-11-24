import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct MoondreamConfiguration: Decodable, Sendable {
    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let vocabularySize: Int
        public let maxPositionEmbeddings: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let ropeTheta: Float
        public let ropeTraditional: Bool
        public let partialRotaryFactor: Float
        public let layerNormEps: Float

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case vocabularySize = "vocab_size"
            case maxPositionEmbeddings = "max_position_embeddings"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case ropeTheta = "rope_theta"
            case ropeTraditional = "rope_traditional"
            case partialRotaryFactor = "partial_rotary_factor"
            case layerNormEps = "rms_norm_eps"
            case dim
            case ffDim = "ff_dim"
            case layers = "n_layers"
            case context = "max_context"
            case heads = "n_heads"
            case keyValueHeads = "n_kv_heads"
        }

        public init(
            modelType: String = "moondream2",
            hiddenSize: Int = 2048,
            intermediateSize: Int = 8192,
            hiddenLayers: Int = 24,
            vocabularySize: Int = 51200,
            maxPositionEmbeddings: Int = 2048,
            attentionHeads: Int = 32,
            kvHeads: Int = 32,
            ropeTheta: Float = 10000,
            ropeTraditional: Bool = false,
            partialRotaryFactor: Float = 0.5,
            layerNormEps: Float = 1e-5,
        ) {
            self.modelType = modelType
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.hiddenLayers = hiddenLayers
            self.vocabularySize = vocabularySize
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.attentionHeads = attentionHeads
            self.kvHeads = kvHeads
            self.ropeTheta = ropeTheta
            self.ropeTraditional = ropeTraditional
            self.partialRotaryFactor = partialRotaryFactor
            self.layerNormEps = layerNormEps
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "moondream2"
            hiddenSize =
                try c.decodeIfPresent(Int.self, forKey: .hiddenSize)
                ?? c.decodeIfPresent(Int.self, forKey: .dim)
                ?? 2048
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize)
                ?? c.decodeIfPresent(Int.self, forKey: .ffDim)
                ?? 8192
            hiddenLayers =
                try c.decodeIfPresent(Int.self, forKey: .hiddenLayers)
                ?? c.decodeIfPresent(Int.self, forKey: .layers)
                ?? 24
            vocabularySize =
                try c.decodeIfPresent(Int.self, forKey: .vocabularySize)
                ?? 51200
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
                ?? c.decodeIfPresent(Int.self, forKey: .context)
                ?? 2048
            attentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .attentionHeads)
                ?? c.decodeIfPresent(Int.self, forKey: .heads)
                ?? 32
            kvHeads =
                try c.decodeIfPresent(Int.self, forKey: .kvHeads)
                ?? c.decodeIfPresent(Int.self, forKey: .keyValueHeads)
                ?? attentionHeads
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
            ropeTraditional =
                try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
            partialRotaryFactor =
                try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.5
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let patchSize: Int
        public let cropSize: Int
        public let maxCrops: Int
        public let overlapMargin: Int
        public let channels: Int
        public let projectionInnerDimension: Int
        public let projectionOutputDimension: Int
        public let attentionBias: Bool
        public let layerNormEps: Float

        public var visionTokenCount: Int {
            let grid = cropSize / patchSize
            return grid * grid
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case patchSize = "patch_size"
            case cropSize = "crop_size"
            case maxCrops = "max_crops"
            case overlapMargin = "overlap_margin"
            case channels = "in_channels"
            case projectionInnerDimension = "proj_inner_dim"
            case projectionOutputDimension = "proj_out_dim"
            case attentionBias = "attention_bias"
            case layerNormEps = "layer_norm_eps"
            case encoderDimension = "enc_dim"
            case encoderPatchSize = "enc_patch_size"
            case encoderLayers = "enc_n_layers"
            case encoderIntermediate = "enc_ff_dim"
            case encoderHeads = "enc_n_heads"
        }

        public init(
            modelType: String = "moondream2_vision",
            hiddenSize: Int = 1152,
            intermediateSize: Int = 4304,
            hiddenLayers: Int = 27,
            attentionHeads: Int = 16,
            patchSize: Int = 14,
            cropSize: Int = 378,
            maxCrops: Int = 12,
            overlapMargin: Int = 4,
            channels: Int = 3,
            projectionInnerDimension: Int = 8192,
            projectionOutputDimension: Int = 2048,
            attentionBias: Bool = true,
            layerNormEps: Float = 1e-5,
        ) {
            self.modelType = modelType
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.hiddenLayers = hiddenLayers
            self.attentionHeads = attentionHeads
            self.patchSize = patchSize
            self.cropSize = cropSize
            self.maxCrops = maxCrops
            self.overlapMargin = overlapMargin
            self.channels = channels
            self.projectionInnerDimension = projectionInnerDimension
            self.projectionOutputDimension = projectionOutputDimension
            self.attentionBias = attentionBias
            self.layerNormEps = layerNormEps
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType) ?? "moondream2_vision"
            hiddenSize =
                try c.decodeIfPresent(Int.self, forKey: .hiddenSize)
                ?? c.decodeIfPresent(Int.self, forKey: .encoderDimension)
                ?? 1152
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize)
                ?? c.decodeIfPresent(Int.self, forKey: .encoderIntermediate)
                ?? 4304
            hiddenLayers =
                try c.decodeIfPresent(Int.self, forKey: .hiddenLayers)
                ?? c.decodeIfPresent(Int.self, forKey: .encoderLayers)
                ?? 27
            attentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .attentionHeads)
                ?? c.decodeIfPresent(Int.self, forKey: .encoderHeads)
                ?? 16
            patchSize =
                try c.decodeIfPresent(Int.self, forKey: .patchSize)
                ?? c.decodeIfPresent(Int.self, forKey: .encoderPatchSize)
                ?? 14
            cropSize = try c.decodeIfPresent(Int.self, forKey: .cropSize) ?? 378
            maxCrops = try c.decodeIfPresent(Int.self, forKey: .maxCrops) ?? 12
            overlapMargin = try c.decodeIfPresent(Int.self, forKey: .overlapMargin) ?? 4
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            projectionInnerDimension =
                try c.decodeIfPresent(Int.self, forKey: .projectionInnerDimension) ?? 8192
            projectionOutputDimension =
                try c.decodeIfPresent(Int.self, forKey: .projectionOutputDimension) ?? 2048
            attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? true
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
        }
    }

    private struct NestedRuntimeConfiguration: Decodable {
        let text: TextConfiguration?
        let vision: VisionConfiguration?
    }

    public let modelType: String
    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let eosTokenID: Int
    public let bosTokenID: Int

    public var visionTokenCount: Int {
        visionConfiguration.visionTokenCount
    }

    public var visionPrefixLength: Int {
        1 + visionTokenCount
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case runtimeConfiguration = "config"
        case eosTokenID = "eos_token_id"
        case bosTokenID = "bos_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "moondream2"
        let nested = try c.decodeIfPresent(
            NestedRuntimeConfiguration.self, forKey: .runtimeConfiguration)
        textConfiguration =
            try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? nested?.text
            ?? TextConfiguration()
        visionConfiguration =
            try c.decodeIfPresent(VisionConfiguration.self, forKey: .visionConfiguration)
            ?? nested?.vision
            ?? VisionConfiguration()
        eosTokenID = try c.decodeIfPresent(Int.self, forKey: .eosTokenID) ?? 0
        bosTokenID = try c.decodeIfPresent(Int.self, forKey: .bosTokenID) ?? 0
    }
}

public struct MoondreamProcessorConfiguration: Decodable, Sendable {
    private struct NestedRuntimeConfiguration: Decodable {
        let vision: MoondreamConfiguration.VisionConfiguration?
    }

    public let processorClass: String
    public let cropSize: Int
    public let maxCrops: Int
    public let overlapMargin: Int
    public let patchSize: Int
    public let bosTokenID: Int
    public let imagePlaceholderTokenID: Int
    public let queryPrefixTokens: [Int]
    public let querySuffixTokens: [Int]

    public var visionTokenCount: Int {
        let grid = cropSize / patchSize
        return grid * grid
    }

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case cropSize = "crop_size"
        case maxCrops = "max_crops"
        case overlapMargin = "overlap_margin"
        case patchSize = "patch_size"
        case bosTokenID = "bos_token_id"
        case imagePlaceholderTokenID = "image_placeholder_token_id"
        case runtimeConfiguration = "config"
        case visionConfiguration = "vision_config"
        case tokenizer
    }

    enum TokenizerKeys: String, CodingKey {
        case bosID = "bos_id"
        case templates
    }

    enum TemplateKeys: String, CodingKey {
        case query
    }

    enum QueryKeys: String, CodingKey {
        case prefix
        case suffix
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass =
            try c.decodeIfPresent(String.self, forKey: .processorClass) ?? "MoondreamProcessor"

        let runtime = try c.decodeIfPresent(
            NestedRuntimeConfiguration.self, forKey: .runtimeConfiguration)
        let rootVision = try c.decodeIfPresent(
            MoondreamConfiguration.VisionConfiguration.self,
            forKey: .visionConfiguration,
        )
        let vision = rootVision ?? runtime?.vision

        cropSize =
            try c.decodeIfPresent(Int.self, forKey: .cropSize)
            ?? vision?.cropSize
            ?? 378
        maxCrops =
            try c.decodeIfPresent(Int.self, forKey: .maxCrops)
            ?? vision?.maxCrops
            ?? 12
        overlapMargin =
            try c.decodeIfPresent(Int.self, forKey: .overlapMargin)
            ?? vision?.overlapMargin
            ?? 4
        patchSize =
            try c.decodeIfPresent(Int.self, forKey: .patchSize)
            ?? vision?.patchSize
            ?? 14
        imagePlaceholderTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imagePlaceholderTokenID) ?? 0

        var decodedBos = try c.decodeIfPresent(Int.self, forKey: .bosTokenID) ?? 0
        var decodedPrefix = [1, 15381, 2]
        var decodedSuffix = [3, 3]
        if c.contains(.tokenizer) {
            let tokenizer = try c.nestedContainer(keyedBy: TokenizerKeys.self, forKey: .tokenizer)
            decodedBos = try tokenizer.decodeIfPresent(Int.self, forKey: .bosID) ?? decodedBos
            if tokenizer.contains(.templates) {
                let templates = try tokenizer.nestedContainer(
                    keyedBy: TemplateKeys.self, forKey: .templates)
                if templates.contains(.query) {
                    let query = try templates.nestedContainer(
                        keyedBy: QueryKeys.self, forKey: .query)
                    decodedPrefix =
                        try query.decodeIfPresent([Int].self, forKey: .prefix) ?? decodedPrefix
                    let suffix = try query.decodeIfPresent([Int].self, forKey: .suffix) ?? [3]
                    decodedSuffix = suffix + suffix
                }
            }
        }
        bosTokenID = decodedBos
        queryPrefixTokens = decodedPrefix
        querySuffixTokens = decodedSuffix
    }
}

// MARK: - Processor

public struct MoondreamProcessor: UserInputProcessor {
    private let config: MoondreamProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: MoondreamProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        let prompt = Self.prompt(from: input)
        let promptTokens = queryTokens(for: prompt)

        guard !images.isEmpty else {
            let tokens = MLXArray(promptTokens).expandedDimensions(axis: 0)
            return LMInput(text: .init(tokens: tokens, mask: ones(like: tokens).asType(.int8)))
        }
        guard images.count == 1 else {
            throw VLMError.singleImageAllowed
        }

        let processed = try preprocess(
            image: images[0].asCIImage(),
            processing: input.processing,
        )
        var tokens = [config.bosTokenID]
        tokens.append(
            contentsOf: Array(
                repeating: config.imagePlaceholderTokenID,
                count: config.visionTokenCount,
            ),
        )
        tokens.append(contentsOf: promptTokens.dropFirst())

        let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)),
            image: .init(pixels: processed.pixels),
            video: .init(pixels: processed.metadata),
        )
    }

    private func queryTokens(for prompt: String) -> [Int] {
        [config.bosTokenID]
            + config.queryPrefixTokens
            + tokenizer.encode(text: prompt, addSpecialTokens: false)
            + config.querySuffixTokens
    }

    private struct ProcessedImage {
        let pixels: MLXArray
        let metadata: MLXArray
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?) throws
        -> ProcessedImage
    {
        let image = MediaProcessing.apply(image, processing: processing).toSRGB()
        let cropSize = config.cropSize
        let patchSize = config.patchSize
        let marginPixels = patchSize * config.overlapMargin
        let cropWindowSize = cropSize - 2 * marginPixels
        guard cropWindowSize > 0 else {
            throw VLMError.processing("Moondream overlap margin leaves no crop interior")
        }

        let canonical = MediaProcessing.zeroOrigin(image)
        let originalHeight = max(Int(canonical.extent.height.rounded()), 1)
        let originalWidth = max(Int(canonical.extent.width.rounded()), 1)
        let tiling = Self.selectTiling(
            height: originalHeight - 2 * marginPixels,
            width: originalWidth - 2 * marginPixels,
            cropSize: cropWindowSize,
            maxCrops: config.maxCrops,
        )
        let targetHeight = tiling.rows * cropWindowSize + 2 * marginPixels
        let targetWidth = tiling.cols * cropWindowSize + 2 * marginPixels

        var crops = [[Float]]()
        crops.append(
            Self.resizeImage(
                image: canonical,
                targetHeight: cropSize,
                targetWidth: cropSize,
            ),
        )

        let resized = Self.resizeImage(
            image: canonical,
            targetHeight: targetHeight,
            targetWidth: targetWidth,
        )
        for row in 0 ..< tiling.rows {
            for col in 0 ..< tiling.cols {
                crops.append(
                    MediaProcessing.cropPixels(
                        resized,
                        sourceWidth: targetWidth,
                        x: col * cropWindowSize,
                        y: row * cropWindowSize,
                        width: cropSize,
                        height: cropSize,
                        channels: 3,
                    ),
                )
            }
        }

        let flatPixels = crops.flatMap(\.self)
        let pixelArray = MLXArray(
            flatPixels,
            [crops.count, cropSize, cropSize, 3],
        )
        let metadata = MLXArray(
            [Int32(crops.count), Int32(tiling.rows), Int32(tiling.cols)],
            [1, 3],
        )
        return ProcessedImage(pixels: pixelArray, metadata: metadata)
    }

    private static func resizeImage(image: CIImage, targetHeight: Int, targetWidth: Int) -> [Float]
    {
        let resized = MediaProcessing.zeroOrigin(image).resampled(
            to: CGSize(width: targetWidth, height: targetHeight),
            method: .bicubic,
        )
        var pixels = MediaProcessing.renderRGBFloat(
            resized, width: targetWidth, height: targetHeight)
        for index in stride(from: 0, to: pixels.count, by: 3) {
            pixels[index] = (pixels[index] - 0.5) / 0.5
            pixels[index + 1] = (pixels[index + 1] - 0.5) / 0.5
            pixels[index + 2] = (pixels[index + 2] - 0.5) / 0.5
        }
        return pixels
    }

    private static func selectTiling(
        height: Int,
        width: Int,
        cropSize: Int,
        maxCrops: Int,
    )
        -> (rows: Int, cols: Int)
    {
        let height = max(height, 1)
        let width = max(width, 1)
        if height <= cropSize || width <= cropSize {
            return (1, 1)
        }

        let minH = Int(ceil(Float(height) / Float(cropSize)))
        let minW = Int(ceil(Float(width) / Float(cropSize)))
        if minH * minW > maxCrops {
            let ratio = sqrt(Float(maxCrops) / Float(minH * minW))
            return (
                max(1, Int(floor(Float(minH) * ratio))),
                max(1, Int(floor(Float(minW) * ratio))),
            )
        }

        var hTiles = Int(floor(sqrt(Float(maxCrops * height) / Float(width))))
        var wTiles = Int(floor(sqrt(Float(maxCrops * width) / Float(height))))
        hTiles = max(hTiles, minH)
        wTiles = max(wTiles, minW)
        if hTiles * wTiles > maxCrops {
            if wTiles > hTiles {
                wTiles = max(1, maxCrops / hTiles)
            } else {
                hTiles = max(1, maxCrops / wTiles)
            }
        }
        return (max(1, hTiles), max(1, wTiles))
    }

    private static func prompt(from input: UserInput) -> String {
        switch input.prompt {
        case .text(let text):
            text
        case .messages:
            input.prompt.description
        case .chat(let messages):
            messages.last(where: { $0.role == .user })?.content
                ?? messages.map(\.content).joined(separator: "\n")
        }
    }

}

// MARK: - Language

private enum MoondreamLanguage {
    final class Attention: Module {
        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let rope: RoPE
        let fused: (Int, Int)

        init(_ config: MoondreamConfiguration.TextConfiguration) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            let qDim = heads * headDim
            let kvDim = kvHeads * headDim
            fused = (qDim, qDim + kvDim)
            _qkv.wrappedValue = Linear(config.hiddenSize, qDim + 2 * kvDim, bias: true)
            _proj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            let ropeDim = Int(Float(headDim) * config.partialRotaryFactor)
            rope = RoPE(
                dimensions: ropeDim,
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
            let batch = x.dim(0)
            let length = x.dim(1)
            let qkv = split(qkv(x), indices: [fused.0, fused.1], axis: -1)
            var queries = qkv[0].reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            var keys = qkv[1].reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
            let values = qkv[2].reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)
            let offset = cache?.offset ?? 0
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask,
            )
            return proj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let activation = GELU(approximation: .fast)

        init(_ config: MoondreamConfiguration.TextConfiguration) {
            _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    final class Layer: Module {
        @ModuleInfo var ln: LayerNorm
        @ModuleInfo var attn: Attention
        @ModuleInfo var mlp: MLP

        init(_ config: MoondreamConfiguration.TextConfiguration) {
            _ln.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            _attn.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(config)
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let normalized = ln(x)
            return x + attn(normalized, mask: mask, cache: cache) + mlp(normalized)
        }
    }

    final class TextModel: Module {
        @ModuleInfo var wte: Embedding
        let blocks: [Layer]
        @ModuleInfo(key: "post_ln") var postLayerNorm: LayerNorm

        init(_ config: MoondreamConfiguration.TextConfiguration) {
            _wte.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            blocks = (0 ..< config.hiddenLayers).map { _ in Layer(config) }
            _postLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
            prefixAttentionLength: Int? = nil,
        )
            -> MLXArray
        {
            var hidden = inputEmbedding ?? wte(inputs!)
            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                if cache == nil, let prefixAttentionLength,
                    hidden.dim(1) > 1
                {
                    .array(
                        Self.prefixAttentionMask(
                            sequenceLength: hidden.dim(1),
                            prefixLength: prefixAttentionLength,
                            dtype: hidden.dtype,
                        ),
                    )
                } else {
                    createAttentionMask(h: hidden, cache: cache?.first)
                }
            for (index, block) in blocks.enumerated() {
                hidden = block(hidden, mask: mask, cache: cache?[index])
            }
            return postLayerNorm(hidden)
        }

        private static func prefixAttentionMask(
            sequenceLength: Int,
            prefixLength: Int,
            dtype: DType,
        )
            -> MLXArray
        {
            var values = Array(repeating: Float32(0), count: sequenceLength * sequenceLength)
            for row in 0 ..< sequenceLength {
                for col in 0 ..< sequenceLength {
                    let visible = col <= row || (row < prefixLength && col < prefixLength)
                    if !visible {
                        values[row * sequenceLength + col] = -Float32.infinity
                    }
                }
            }
            return MLXArray(values, [sequenceLength, sequenceLength]).asType(dtype)
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo var model: TextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear
        let kvHeads: [Int]

        init(_ config: MoondreamConfiguration.TextConfiguration) {
            _model.wrappedValue = TextModel(config)
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: true)
            kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
            prefixAttentionLength: Int? = nil,
        )
            -> LMOutput
        {
            LMOutput(
                logits: lmHead(
                    model(
                        inputs,
                        cache: cache,
                        inputEmbedding: inputEmbedding,
                        prefixAttentionLength: prefixAttentionLength,
                    ),
                ),
            )
        }
    }
}

// MARK: - Vision

private enum MoondreamVision {
    final class Attention: Module {
        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear

        let heads: Int
        let headDim: Int
        let scale: Float

        init(_ config: MoondreamConfiguration.VisionConfiguration) {
            heads = config.attentionHeads
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _qkv.wrappedValue = Linear(
                config.hiddenSize,
                3 * config.hiddenSize,
                bias: config.attentionBias,
            )
            _proj.wrappedValue = Linear(
                config.hiddenSize,
                config.hiddenSize,
                bias: config.attentionBias,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let batch = x.dim(0)
            let length = x.dim(1)
            let qkv = split(qkv(x), parts: 3, axis: -1)
            let queries = qkv[0].reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let keys = qkv[1].reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let values = qkv[2].reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: nil,
            )
            return proj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let activation = GELU(approximation: .fast)

        init(_ config: MoondreamConfiguration.VisionConfiguration) {
            _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    final class Block: Module {
        @ModuleInfo var ln1: LayerNorm
        @ModuleInfo var attn: Attention
        @ModuleInfo var ln2: LayerNorm
        @ModuleInfo var mlp: MLP

        init(_ config: MoondreamConfiguration.VisionConfiguration) {
            _ln1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            _attn.wrappedValue = Attention(config)
            _ln2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            _mlp.wrappedValue = MLP(config)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var hidden = x + attn(ln1(x))
            hidden = hidden + mlp(ln2(hidden))
            return hidden
        }
    }

    final class Encoder: Module {
        @ModuleInfo(key: "patch_emb") var patchEmbedding: Linear
        @ModuleInfo(key: "pos_emb") var positionEmbedding: MLXArray
        let blocks: [Block]
        @ModuleInfo(key: "post_ln") var postLayerNorm: LayerNorm

        let config: MoondreamConfiguration.VisionConfiguration

        init(_ config: MoondreamConfiguration.VisionConfiguration) {
            self.config = config
            _patchEmbedding.wrappedValue = Linear(
                config.patchSize * config.patchSize * config.channels,
                config.hiddenSize,
                bias: true,
            )
            _positionEmbedding.wrappedValue = MLXArray.zeros(
                [1, config.visionTokenCount, config.hiddenSize],
            )
            blocks = (0 ..< config.hiddenLayers).map { _ in Block(config) }
            _postLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let batch = x.dim(0)
            let height = x.dim(1)
            let width = x.dim(2)
            let channels = x.dim(3)
            let patch = config.patchSize
            let patchH = height / patch
            let patchW = width / patch
            var hidden =
                x
                .reshaped(batch, patchH, patch, patchW, patch, channels)
                .transposed(0, 1, 3, 5, 2, 4)
                .reshaped(batch, patchH * patchW, channels * patch * patch)
            hidden = patchEmbedding(hidden)
            hidden = hidden + positionEmbedding.asType(hidden.dtype)
            for block in blocks {
                hidden = block(hidden)
            }
            return postLayerNorm(hidden)
        }
    }

    final class Projection: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let activation = GELU(approximation: .fast)

        init(_ config: MoondreamConfiguration.VisionConfiguration) {
            _fc1.wrappedValue = Linear(
                2 * config.hiddenSize,
                config.projectionInnerDimension,
                bias: true,
            )
            _fc2.wrappedValue = Linear(
                config.projectionInnerDimension,
                config.projectionOutputDimension,
                bias: true,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    final class VisionModel: Module {
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "proj_mlp") var projection: Projection

        let config: MoondreamConfiguration.VisionConfiguration

        init(_ config: MoondreamConfiguration.VisionConfiguration) {
            self.config = config
            _encoder.wrappedValue = Encoder(config)
            _projection.wrappedValue = Projection(config)
        }

        func callAsFunction(pixelValues: MLXArray, metadata: MLXArray?) -> MLXArray {
            let allFeatures = encoder(pixelValues)
            guard let metadata else {
                let combined = concatenated([allFeatures, allFeatures], axis: -1)
                return projection(combined)
            }

            let values = metadata.asArray(Int32.self).map(Int.init)
            let numCrops = values.first ?? allFeatures.dim(0)
            let rows = values.count > 1 ? values[1] : 1
            let cols = values.count > 2 ? values[2] : max(1, numCrops - 1)
            let globalFeatures = allFeatures[0, 0..., 0...]
            let reconstructed: MLXArray
            if numCrops > 1 {
                let localFeatures = (1 ..< numCrops).map { allFeatures[$0, 0..., 0...] }
                reconstructed = reconstructLocalFeatures(localFeatures, rows: rows, cols: cols)
            } else {
                reconstructed = globalFeatures
            }
            let combined = concatenated([globalFeatures, reconstructed], axis: -1)
            return projection(combined)[.newAxis, 0..., 0...]
        }

        private func reconstructLocalFeatures(
            _ localFeatures: [MLXArray],
            rows: Int,
            cols: Int,
        )
            -> MLXArray
        {
            let gridSize = config.cropSize / config.patchSize
            let margin = config.overlapMargin
            var cropRows = [MLXArray]()
            var index = 0
            for row in 0 ..< rows {
                var rowFeatures = [MLXArray]()
                for col in 0 ..< cols {
                    var feature = localFeatures[min(index, localFeatures.count - 1)]
                        .reshaped(gridSize, gridSize, -1)
                    let top = row > 0 ? margin : 0
                    let bottom = row < rows - 1 ? gridSize - margin : gridSize
                    let left = col > 0 ? margin : 0
                    let right = col < cols - 1 ? gridSize - margin : gridSize
                    feature = feature[top ..< bottom, left ..< right, 0...]
                    rowFeatures.append(feature)
                    index += 1
                }
                cropRows.append(concatenated(rowFeatures, axis: 1))
            }
            let fullGrid = concatenated(cropRows, axis: 0)
            let height = fullGrid.dim(0)
            let width = fullGrid.dim(1)
            var pooled = [MLXArray]()
            pooled.reserveCapacity(gridSize * gridSize)
            for row in 0 ..< gridSize {
                let hStart = min(Int(round(Float(row * height) / Float(gridSize))), height - 1)
                let hEnd = max(
                    hStart + 1,
                    min(Int(round(Float((row + 1) * height) / Float(gridSize))), height),
                )
                for col in 0 ..< gridSize {
                    let wStart = min(Int(round(Float(col * width) / Float(gridSize))), width - 1)
                    let wEnd = max(
                        wStart + 1,
                        min(Int(round(Float((col + 1) * width) / Float(gridSize))), width),
                    )
                    let cell = mean(
                        mean(fullGrid[hStart ..< hEnd, wStart ..< wEnd, 0...], axis: 0),
                        axis: 0,
                    )
                    pooled.append(cell)
                }
            }
            return concatenated(pooled.map { $0[.newAxis, 0...] }, axis: 0)
        }
    }
}

// MARK: - Model

public final class Moondream: Module, LanguageModel, KVCacheDimensionProvider {
    @ModuleInfo private var vision: MoondreamVision.VisionModel
    @ModuleInfo private var text: MoondreamLanguage.LanguageModel

    public let config: MoondreamConfiguration
    public var vocabularySize: Int {
        config.textConfiguration.vocabularySize
    }

    public var kvHeads: [Int] {
        text.kvHeads
    }

    public var loraLayers: [Module] {
        text.model.blocks
    }

    public init(_ config: MoondreamConfiguration) {
        self.config = config
        _vision.wrappedValue = MoondreamVision.VisionModel(config.visionConfiguration)
        _text.wrappedValue = MoondreamLanguage.LanguageModel(config.textConfiguration)
    }

    private func inputEmbeddings(inputIDs: MLXArray, pixelValues: MLXArray?, metadata: MLXArray?)
        -> MLXArray
    {
        let embeddings = text.model.wte(inputIDs)
        guard let pixelValues else {
            return embeddings
        }
        let dtype = vision.encoder.patchEmbedding.weight.dtype
        let imageFeatures = vision(
            pixelValues: pixelValues.asType(dtype),
            metadata: metadata,
        )
        .asType(embeddings.dtype)
        let textStart = 1 + imageFeatures.dim(1)
        let bosEmbedding = embeddings[0..., 0 ..< 1, 0...]
        if embeddings.dim(1) > textStart {
            return concatenated(
                [
                    bosEmbedding,
                    imageFeatures,
                    embeddings[0..., textStart..., 0...],
                ],
                axis: 1,
            )
        }
        return concatenated([bosEmbedding, imageFeatures], axis: 1)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let embeddings = inputEmbeddings(
            inputIDs: input.text.tokens,
            pixelValues: input.image?.pixels,
            metadata: input.video?.pixels,
        )
        let result = text(
            nil,
            cache: cache,
            inputEmbedding: embeddings,
            prefixAttentionLength: input.image == nil ? nil : config.visionPrefixLength,
        )
        prefill.progress?(embeddings.dim(1), embeddings.dim(1))
        return .logits(result)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        text(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.contains("position_ids") || key.contains("freqs_cis") {
                continue
            }
            var newKey = key
            if newKey.hasPrefix("model.") {
                newKey.removeFirst("model.".count)
            }
            if newKey.hasPrefix("region.") || newKey.hasPrefix("region_model.") {
                continue
            }

            if newKey == "text.wte" {
                newKey = "text.model.wte.weight"
            } else if newKey.hasPrefix("text.lm_head.") {
                newKey = "text." + newKey.dropFirst("text.".count)
            } else if newKey.hasPrefix("text.") {
                newKey = "text.model." + newKey.dropFirst("text.".count)
            } else if newKey.hasPrefix("vision."), !newKey.hasPrefix("vision.proj_mlp.") {
                newKey = "vision.encoder." + newKey.dropFirst("vision.".count)
            } else if newKey.hasPrefix("vision_encoder.encoder.model.visual.") {
                newKey =
                    "vision.encoder."
                    + newKey.dropFirst("vision_encoder.encoder.model.visual.".count)
                newKey = newKey.replacingOccurrences(of: "patch_embed.linear.", with: "patch_emb.")
                newKey = newKey.replacingOccurrences(of: "pos_embed", with: "pos_emb")
                newKey = newKey.replacingOccurrences(of: ".norm1.", with: ".ln1.")
                newKey = newKey.replacingOccurrences(of: ".norm2.", with: ".ln2.")
                newKey = newKey.replacingOccurrences(of: "norm.", with: "post_ln.")
            } else if newKey.hasPrefix("vision_encoder.projection.mlp.") {
                newKey =
                    "vision.proj_mlp."
                    + newKey.dropFirst("vision_encoder.projection.mlp.".count)
            } else if newKey == "text_model.transformer.embd.wte.weight" {
                newKey = "text.model.wte.weight"
            } else if newKey.hasPrefix("text_model.transformer.h.") {
                newKey =
                    "text.model.blocks."
                    + newKey.dropFirst("text_model.transformer.h.".count)
                newKey = newKey.replacingOccurrences(of: ".mixer.Wqkv.", with: ".attn.qkv.")
                newKey = newKey.replacingOccurrences(of: ".mixer.out_proj.", with: ".attn.proj.")
            } else if newKey.hasPrefix("text_model.lm_head.ln.") {
                newKey = "text.model.post_ln." + newKey.dropFirst("text_model.lm_head.ln.".count)
            } else if newKey.hasPrefix("text_model.lm_head.linear.") {
                newKey = "text.lm_head." + newKey.dropFirst("text_model.lm_head.linear.".count)
            }

            sanitized[String(newKey)] = value
        }
        return sanitized
    }
}
