import MLX
import MLXNN

/// Inkling 2 text generation with optional still images.
public final class Inkling2: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "language_model") var language: Inkling2Language.Model
    @ModuleInfo(key: "vision_tower") var vision: Inkling2Vision.Tower?

    public let config: Inkling2Configuration
    public var vocabularySize: Int { config.text.unpaddedVocabulary ?? config.text.vocabulary }
    public var kvHeads: [Int] { language.model.layers.map { $0.attention.kvHeads } }
    public var toolCallFormat: ToolCallFormat? { .inkling2 }
    public var reasoningConfig: ReasoningConfig? {
        .init(
            startDelimiter: "<|content_thinking|>", endDelimiter: "<|end_message|>",
            promptStrategy: .none, isSpecialToken: true)
    }
    public var loraLayers: [Module] { language.model.layers }

    public init(_ config: Inkling2Configuration) {
        self.config = config
        _language.wrappedValue = Inkling2Language.Model(config.text)
        _vision.wrappedValue = config.vision.map {
            Inkling2Vision.Tower($0, textWidth: config.text.width)
        }
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        language.model.layers.map { layer in
            let window = layer.attention.sliding
            let storage: KVCache = window > 0 ? RotatingKVCache(maxSize: window) : KVCacheSimple()
            return CacheList(MaterializedKVCache(storage), ArraysCache(size: 4))
        }
    }

    func inputEmbeddings(_ input: LMInput) throws -> MLXArray {
        let tokens = input.text.tokens
        let ids = tokens.ndim == 1 ? tokens.expandedDimensions(axis: 0) : tokens
        guard ids.ndim == 2, ids.dim(0) > 0, ids.dim(1) > 0 else {
            throw VLMError.processing("Inkling 2 requires a nonempty token prompt.")
        }
        guard input.video == nil, input.audio == nil else {
            throw VLMError.processing("Inkling 2 accepts text and still images.")
        }
        let initial = language.model.embed(ids)
        guard let image = input.image else { return initial }
        guard let vision else {
            throw VLMError.processing("Image input requires an Inkling 2 vision tower.")
        }
        let features = try vision(image.pixels, grid: image.frames ?? [])
        let slots = ids.asArray(Int32.self).enumerated().compactMap { index, id in
            id == config.imageToken ? Int32(index) : nil
        }
        guard slots.count == features.dim(0) else {
            throw VLMError.processing(
                "Inkling 2 has \(slots.count) image placeholders for \(features.dim(0)) features.")
        }
        let embeddings = initial.reshaped(-1, config.text.width)
        embeddings[MLXArray(slots), 0...] = features.asType(initial.dtype)
        return embeddings.reshaped(initial.shape)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        let embeddings = try inputEmbeddings(input)
        let length = embeddings.dim(1)
        let completed = try prefill.forEachChunk(total: length) { range in
            _ = language.model(embeddings[0..., range, 0...], cache: cache, lastOnly: true)
            asyncEval(cache)
        }
        if completed > 0 { eval(cache) }
        let logits = language(embeddings[0..., completed..., 0...], cache: cache, lastOnly: true)
        prefill.progress?(length, length)
        return .logits(.init(logits: logits))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let ids = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs
        return language(language.model.embed(ids), cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Inkling2Checkpoint.sanitize(weights)
    }
}
