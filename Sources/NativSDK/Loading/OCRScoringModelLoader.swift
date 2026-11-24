import Foundation

extension ModelLoader {
    func loadOCRScoringResources(
        assets: ResolvedModelAssets, configuration: Data, baseConfiguration: BaseConfiguration,
        metadata: ModelMetadata, tokenizerLoader: any TokenizerLoader
    ) async throws -> sending LoadedModel {
        guard MiniCPMOCRJevConfiguration.matches(configuration) else {
            throw ModelFactoryError.unsupportedModelType(
                "\(baseConfiguration.modelType) (OCR scoring)")
        }
        let config = try JSONDecoder.json5().decode(
            MiniCPMOCRJevConfiguration.self, from: configuration)
        let model = try MiniCPMOCRJev(config)
        async let tokenizerTask = tokenizerLoader.load(from: assets.tokenizerDirectory)
        let quantization =
            baseConfiguration.perLayerQuantization == nil
            ? try loadQuantizationConfig(from: assets.modelDirectory) : nil
        let perLayer = baseConfiguration.perLayerQuantization.map { original in
            BaseConfiguration.PerLayerQuantization(
                quantization: original.quantization,
                perLayerQuantization: Dictionary(
                    uniqueKeysWithValues: original.perLayerQuantization.map { key, value in
                        (
                            key == "score" || key.hasPrefix("backbone.") ? key : "backbone." + key,
                            value
                        )
                    }))
        }
        try await CheckpointLoader.load(
            modelDirectory: assets.modelDirectory, model: model,
            quantization: quantization, perLayerQuantization: perLayer)
        var settings: [String: JSONValue] = [:]
        for name in ["processor_config.json", "preprocessor_config.json"] {
            let file = assets.modelDirectory.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: file.path) {
                let data = try JSONDecoder.json5().decode(
                    [String: JSONValue].self, from: Data(contentsOf: file))
                settings.merge(data) { _, new in new }
            }
        }
        if case .object(let image) = settings["image_processor"] {
            settings.merge(image) { current, _ in current }
        }
        let imageConfig =
            settings.isEmpty
            ? Qwen3VLProcessorConfiguration(
                qwen35VisionConfiguration: config.backbone.visionConfiguration)
            : try JSONDecoder.json5().decode(
                Qwen3VLProcessorConfiguration.self, from: JSONEncoder().encode(settings))
        let tokenizer = try await tokenizerTask
        return LoadedModel(
            assets: assets, metadata: metadata, model: model,
            processor: MiniCPMOCRJevProcessor(config: imageConfig, tokenizer: tokenizer),
            tokenizer: tokenizer)
    }
}

extension NativOCRScorer {
    public static func loadModel(
        from source: ModelSource, tokenizerSource: ModelSource? = nil,
        using tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        policy: ModelExecutionPolicy = .init(),
        progress: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> NativOCRScorer {
        let resources = try await ModelLoader().loadResources(
            from: source, kind: .ocrScoring, tokenizerSource: tokenizerSource,
            using: tokenizerLoader, progress: progress)
        try Task.checkCancellation()
        return resources.intoNativOCRScorer(policy: policy)
    }

    public static func loadModel(
        assets: ResolvedModelAssets,
        tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        policy: ModelExecutionPolicy = .init()
    ) async throws -> NativOCRScorer {
        let resources = try await ModelLoader().loadResources(
            assets: assets, kind: .ocrScoring, tokenizerLoader: tokenizerLoader)
        try Task.checkCancellation()
        return resources.intoNativOCRScorer(policy: policy)
    }
}
