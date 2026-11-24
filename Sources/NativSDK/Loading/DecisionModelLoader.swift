import Foundation

extension ModelLoader {
    func loadDecisionResources(
        assets: ResolvedModelAssets, configuration: Data, baseConfiguration: BaseConfiguration,
        metadata: ModelMetadata, tokenizerLoader: any TokenizerLoader
    ) async throws -> sending LoadedModel {
        let model = try registry.createDecisionModel(
            configuration: configuration, modelType: baseConfiguration.modelType)
        async let tokenizer = tokenizerLoader.load(from: assets.tokenizerDirectory)
        let quantization =
            baseConfiguration.perLayerQuantization == nil
            ? try loadQuantizationConfig(from: assets.modelDirectory) : nil
        var perLayer = baseConfiguration.perLayerQuantization
        if model is DeciderModel, let original = perLayer {
            perLayer = .init(
                quantization: original.quantization,
                perLayerQuantization: Dictionary(
                    uniqueKeysWithValues: original.perLayerQuantization.map {
                        (DeciderModel.canonicalPath($0.key), $0.value)
                    }))
        }
        try await CheckpointLoader.load(
            modelDirectory: assets.modelDirectory, model: model,
            quantization: quantization, perLayerQuantization: perLayer)
        if let d1 = model as? LFM3DecisionModel {
            let file = assets.modelDirectory.appendingPathComponent("calibration.json")
            if FileManager.default.fileExists(atPath: file.path) {
                d1.calibration = try JSONDecoder().decode(
                    LFM3DecisionCalibration.self, from: Data(contentsOf: file))
            }
            let data = try Data(
                contentsOf: assets.tokenizerDirectory.appendingPathComponent("tokenizer.json"))
            let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
            d1.specialTokens = (json?["added_tokens"] as? [[String: Any]] ?? []).compactMap {
                $0["special"] as? Bool == true ? $0["content"] as? String : nil
            }
        }
        return try await LoadedModel(
            assets: assets, metadata: metadata, model: model,
            processor: DecisionInputProcessor(), tokenizer: tokenizer)
    }
}

private struct DecisionInputProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput {
        throw ModelExecutionError.unsupportedOperation(
            "Decision models use predict(state:questions:)")
    }
}
