import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct BunnyConfiguration: Decodable, Sendable {
    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let intermediateSize: Int
        public let attentionHeads: Int
        public let imageSize: Int
        public let patchSize: Int
        public let channels: Int
        public let layerNormEps: Float

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case channels = "num_channels"
            case layerNormEps = "layer_norm_eps"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType)
                ?? "siglip_vision_model"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 27
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
            attentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 384
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            layerNormEps =
                try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        }

        init(
            modelType: String = "siglip_vision_model",
            hiddenSize: Int = 1152,
            hiddenLayers: Int = 27,
            intermediateSize: Int = 4304,
            attentionHeads: Int = 16,
            imageSize: Int = 384,
            patchSize: Int = 14,
            channels: Int = 3,
            layerNormEps: Float = 1e-6,
        ) {
            self.modelType = modelType
            self.hiddenSize = hiddenSize
            self.hiddenLayers = hiddenLayers
            self.intermediateSize = intermediateSize
            self.attentionHeads = attentionHeads
            self.imageSize = imageSize
            self.patchSize = patchSize
            self.channels = channels
            self.layerNormEps = layerNormEps
        }
    }

    public let modelType: String
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let kvHeads: Int
    public let rmsNormEps: Float
    public let vocabularySize: Int
    public let ropeTheta: Float
    public let ropeScaling: [String: StringOrNumber]?
    public let tieWordEmbeddings: Bool
    public let attentionBias: Bool
    public let imageTokenIndex: Int
    public let imageAspectRatio: String?
    public let tokenizerModelMaxLength: Int?
    public let visionConfiguration: VisionConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case attentionBias = "attention_bias"
        case imageTokenIndex = "image_token_index"
        case imageAspectRatio = "image_aspect_ratio"
        case tokenizerModelMaxLength = "tokenizer_model_max_length"
        case visionConfiguration = "vision_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "bunny-llama"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads =
            try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self,
            forKey: .ropeScaling,
        )
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        attentionBias =
            try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        imageTokenIndex =
            try c.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? -200
        imageAspectRatio = try c.decodeIfPresent(String.self, forKey: .imageAspectRatio)
        tokenizerModelMaxLength =
            try c.decodeIfPresent(Int.self, forKey: .tokenizerModelMaxLength)
        visionConfiguration =
            try c.decodeIfPresent(VisionConfiguration.self, forKey: .visionConfiguration)
            ?? VisionConfiguration()
    }

    var llavaConfiguration: LLaVAConfiguration {
        LLaVAConfiguration(
            textConfiguration: .init(
                modelType: "llama",
                hiddenSize: hiddenSize,
                hiddenLayers: hiddenLayers,
                intermediateSize: intermediateSize,
                attentionHeads: attentionHeads,
                kvHeads: kvHeads,
                rmsNormEps: rmsNormEps,
                vocabularySize: vocabularySize,
                ropeTheta: ropeTheta,
                ropeScaling: ropeScaling,
                tieWordEmbeddings: tieWordEmbeddings,
                attentionBias: attentionBias,
            ),
            visionConfiguration: .init(
                modelType: visionConfiguration.modelType,
                hiddenSize: visionConfiguration.hiddenSize,
                hiddenLayers: visionConfiguration.hiddenLayers,
                intermediateSize: visionConfiguration.intermediateSize,
                attentionHeads: visionConfiguration.attentionHeads,
                imageSize: visionConfiguration.imageSize,
                patchSize: visionConfiguration.patchSize,
                channels: visionConfiguration.channels,
                layerNormEps: visionConfiguration.layerNormEps,
            ),
            modelType: modelType,
            imageTokenIndex: imageTokenIndex,
            visionFeatureLayer: -1,
            visionFeatureSelectStrategy: "full",
        )
    }
}

public struct BunnyProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let imageSize: Int
    public let patchSize: Int
    public let imageTokenIndex: Int
    public let imageAspectRatio: String?

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case imageTokenIndex = "image_token_index"
        case imageAspectRatio = "image_aspect_ratio"
        case visionConfiguration = "vision_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let vision = try c.decodeIfPresent(
            BunnyConfiguration.VisionConfiguration.self,
            forKey: .visionConfiguration,
        )
        processorClass =
            try c.decodeIfPresent(String.self, forKey: .processorClass) ?? "BunnyProcessor"
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        imageSize =
            try c.decodeIfPresent(Int.self, forKey: .imageSize)
            ?? vision?.imageSize
            ?? 384
        patchSize =
            try c.decodeIfPresent(Int.self, forKey: .patchSize)
            ?? vision?.patchSize
            ?? 14
        imageTokenIndex =
            try c.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? -200
        imageAspectRatio =
            try c.decodeIfPresent(String.self, forKey: .imageAspectRatio)
    }

    var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }
}

// MARK: - Processor

public struct BunnyProcessor: UserInputProcessor {
    private let config: BunnyProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: BunnyProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        let prompt = renderPrompt(input: input, externalImageCount: input.images.count)
        let tokens = encodePrompt(prompt)
        let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)

        guard !images.isEmpty else {
            return LMInput(text: .init(tokens: tokenArray, mask: ones(like: tokenArray)))
        }

        var pixelArrays: [MLXArray] = []
        for image in images {
            try pixelArrays.append(preprocess(image.asCIImage(), processing: input.processing))
        }

        return LMInput(
            text: .init(tokens: tokenArray, mask: ones(like: tokenArray)),
            image: .init(pixels: concatenated(pixelArrays, axis: 0)),
        )
    }

    private func renderPrompt(input: UserInput, externalImageCount: Int) -> String {
        let messages: [Chat.Message] =
            switch input.prompt {
            case .text(let text):
                [.user(text)]
            case .chat(let chat):
                chat
            case .messages:
                [.user(input.prompt.description)]
            }

        var result = ""
        var usedExternalImages = false
        for (index, message) in messages.enumerated() {
            let messageImageCount: Int
            if !message.images.isEmpty {
                messageImageCount = message.images.count
            } else if !usedExternalImages, externalImageCount > 0 {
                messageImageCount = externalImageCount
                usedExternalImages = true
            } else {
                messageImageCount = 0
            }

            let imagePrefix = String(repeating: "<image>\n", count: messageImageCount)
            let content = imagePrefix + message.content
            if index == 0 {
                result += "<|begin_of_text|>"
            }
            result += "<|start_header_id|>\(message.role.rawValue)<|end_header_id|>\n\n"
            result += content.trimmingCharacters(in: .whitespacesAndNewlines)
            result += "<|eot_id|>"
        }
        result += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return result
    }

    private func encodePrompt(_ prompt: String) -> [Int] {
        let chunks = prompt.components(separatedBy: "<image>")
        var tokens: [Int] = []
        for (index, chunk) in chunks.enumerated() {
            if index > 0 {
                tokens.append(config.imageTokenIndex)
            }
            tokens.append(contentsOf: tokenizer.encode(text: chunk, addSpecialTokens: false))
        }
        return tokens
    }

    private func preprocess(_ image: CIImage, processing: UserInput.Processing?) throws -> MLXArray
    {
        var image = MediaProcessing.zeroOrigin(MediaProcessing.apply(image, processing: processing))
            .toSRGB()
        if config.imageAspectRatio == "pad" {
            image = Self.padToSquare(image, color: config.imageMeanTuple)
        }
        image =
            image
            .resampled(
                to: CGSize(width: config.imageSize, height: config.imageSize),
                method: .bicubic,
            )
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
        return image.asMLXArray()
    }

    private static func padToSquare(
        _ image: CIImage,
        color: (CGFloat, CGFloat, CGFloat),
    )
        -> CIImage
    {
        let image = MediaProcessing.zeroOrigin(image)
        let width = image.extent.width
        let height = image.extent.height
        guard width != height else {
            return image
        }

        let side = max(width, height)
        let background = CIImage(
            color: CIColor(red: color.0, green: color.1, blue: color.2, alpha: 1),
        )
        .cropped(to: CGRect(x: 0, y: 0, width: side, height: side))
        let translated = image.transformed(
            by: CGAffineTransform(
                translationX: (side - width) / 2,
                y: (side - height) / 2,
            ),
        )
        return translated.composited(over: background)
    }
}

// MARK: - Model

private final class BunnyVisionTower: Module {
    @ModuleInfo(key: "vision_tower") var visionModel: LLaVAVision.VisionModel

    init(_ config: LLaVAConfiguration.VisionConfiguration) {
        _visionModel.wrappedValue = LLaVAVision.VisionModel(config)
    }

    func callAsFunction(_ x: MLXArray, outputHiddenStates: Bool = false) -> (
        MLXArray, MLXArray, [MLXArray]?,
    ) {
        visionModel(x, outputHiddenStates: outputHiddenStates)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        visionModel.sanitize(weights: weights)
    }
}

private final class BunnyTextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [LLaVALanguage.DecoderLayer]
    @ModuleInfo var norm: RMSNorm
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    init(_ config: LLaVAConfiguration.TextConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.hiddenSize,
        )
        layers = (0 ..< config.hiddenLayers).map { _ in
            LLaVALanguage.DecoderLayer(config)
        }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
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
        return lmHead(norm(h))
    }
}

private final class BunnyLanguageModel: Module, KVCacheDimensionProvider {
    @ModuleInfo var model: BunnyTextModel
    let kvHeads: [Int]

    init(_ config: LLaVAConfiguration.TextConfiguration) {
        _model.wrappedValue = BunnyTextModel(config)
        kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
    }

    func callAsFunction(
        _ inputs: MLXArray?,
        cache: [KVCache]? = nil,
        inputEmbedding: MLXArray? = nil,
    )
        -> LMOutput
    {
        LMOutput(logits: model(inputs, cache: cache, inputEmbedding: inputEmbedding))
    }
}

public final class BunnyLlama: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: BunnyVisionTower
    @ModuleInfo(key: "language_model") private var languageModel: BunnyLanguageModel
    @ModuleInfo(key: "mm_projector") private var projector: LLaVAMultiModalProjector

    public let config: BunnyConfiguration
    private let llavaConfig: LLaVAConfiguration
    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public init(_ config: BunnyConfiguration) {
        self.config = config
        llavaConfig = config.llavaConfiguration
        _visionTower.wrappedValue = BunnyVisionTower(llavaConfig.visionConfiguration)
        _languageModel.wrappedValue = BunnyLanguageModel(llavaConfig.textConfiguration)
        _projector.wrappedValue = LLaVAMultiModalProjector(llavaConfig)
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    private func inputEmbeddings(inputIDs: MLXArray, pixelValues: MLXArray?) -> MLXArray {
        guard let pixelValues else {
            return languageModel.model.embedTokens(inputIDs)
        }

        let safeInputIDs = which(
            inputIDs .== config.imageTokenIndex,
            MLXArray(0),
            inputIDs,
        )
        let inputsEmbeds = languageModel.model.embedTokens(safeInputIDs)

        let (_, _, hiddenStates) = visionTower(
            pixelValues.transposed(0, 2, 3, 1),
            outputHiddenStates: true,
        )
        guard let hiddenStates else {
            return inputsEmbeds
        }

        var imageFeatures = hiddenStates[hiddenStates.count - 1]
        imageFeatures = projector(imageFeatures).asType(inputsEmbeds.dtype)

        let ids = inputIDs[0].asArray(Int.self)
        let positions = ids.indices.filter { ids[$0] == config.imageTokenIndex }
        guard !positions.isEmpty else {
            return inputsEmbeds
        }
        precondition(
            positions.count == imageFeatures.dim(0),
            "Bunny image feature/token count mismatch: tokens \(positions.count), features \(imageFeatures.dim(0))",
        )

        var pieces: [MLXArray] = []
        var start = 0
        for (imageIndex, position) in positions.enumerated() {
            if position > start {
                pieces.append(inputsEmbeds[0, start ..< position, 0...])
            }
            pieces.append(imageFeatures[imageIndex])
            start = position + 1
        }
        if start < ids.count {
            pieces.append(inputsEmbeds[0, start ..< ids.count, 0...])
        }

        var merged = concatenated(pieces, axis: 0)
        if let maxLength = config.tokenizerModelMaxLength, merged.dim(0) > maxLength {
            merged = merged[0 ..< maxLength, 0...]
        }
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
                && !key.hasPrefix("vision_tower.vision_tower.vision_model.head.")
        }
        return visionTower.sanitize(weights: filtered)
    }
}
