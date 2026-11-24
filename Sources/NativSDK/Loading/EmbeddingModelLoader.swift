import Foundation
import MLX

extension ModelLoader {
    func loadEmbeddingResources(
        assets: ResolvedModelAssets, configuration: Data, baseConfiguration: BaseConfiguration,
        metadata: ModelMetadata, tokenizerLoader: any TokenizerLoader
    ) async throws -> sending LoadedModel {
        let directory = assets.modelDirectory
        let hasPooling = FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("1_Pooling/config.json").path)
        let defaultPooling: EmbeddingPooling =
            switch baseConfiguration.modelType {
            case "qwen3", "qwen3_embedding", "qwen3_vl", "qwen3_vl_embedding": .lasttoken
            case "lfm2", "lfm2_embedding", "lfm2_colbert": .cls
            default: .mean
            }
        let pooling =
            hasPooling ? try EmbeddingPooling.load(from: directory, default: defaultPooling) : nil
        var configuration = configuration
        var modelType = baseConfiguration.modelType
        let denseConfig = directory.appendingPathComponent("1_Dense/config.json")
        if ["lfm2", "lfm2_embedding", "lfm2_colbert"].contains(modelType),
            FileManager.default.fileExists(atPath: denseConfig.path)
        {
            let dense = try JSONDecoder.json5().decode(
                [String: JSONValue].self, from: Data(contentsOf: denseConfig))
            guard case .int(let dimensions) = dense["out_features"], dimensions > 0,
                dense["bias"] != .bool(true)
            else {
                throw ModelExecutionError.invalidOptions(
                    "Invalid LFM2 ColBERT projection configuration")
            }
            var object = try JSONDecoder.json5().decode(
                [String: JSONValue].self, from: configuration)
            object["embedding_dim"] = .int(dimensions)
            configuration = try JSONEncoder().encode(object)
            modelType = "lfm2_colbert"
        }
        // ColQwen checkpoints occur with both flattened and nested Qwen2.5 text configs.
        if modelType == "colqwen2_5" {
            var object = try JSONDecoder.json5().decode(
                [String: JSONValue].self, from: configuration)
            if case .object(let text) = object["text_config"] {
                object.merge(text) { current, _ in current }
                configuration = try JSONEncoder().encode(object)
            }
        }
        let auxiliary = try FileManager.default.contentsOfDirectory(
            at: directory, includingPropertiesForKeys: nil
        )
        .filter {
            $0.lastPathComponent.range(of: #"^\d+_Dense$"#, options: .regularExpression) != nil
        }
        .map(\.lastPathComponent).sorted()
        let model = try registry.createEmbeddingModel(
            configuration: configuration, modelType: modelType, pooling: pooling)
        async let tokenizerTask = tokenizerLoader.load(from: assets.tokenizerDirectory)
        let quantization =
            baseConfiguration.perLayerQuantization == nil
            ? try loadQuantizationConfig(from: directory) : nil
        var perLayerQuantization = baseConfiguration.perLayerQuantization
        if model is EmbeddingGemma2Model, let quantization = perLayerQuantization {
            var mapped: [String: BaseConfiguration.QuantizationOption] = [:]
            for (path, value) in quantization.perLayerQuantization {
                let native = path.hasPrefix("model.") ? String(path.dropFirst(6)) : path
                guard mapped[native] == nil else {
                    throw ModelExecutionError.invalidOptions(
                        "Duplicate EmbeddingGemma 2 quantization path")
                }
                mapped[native] = value
            }
            perLayerQuantization = .init(
                quantization: quantization.quantization,
                perLayerQuantization: mapped)
        }
        if model is Qwen3VLEmbeddingModel || model is ColQwen25EmbeddingModel
            || model is ColIdefics3EmbeddingModel,
            let quantization = perLayerQuantization
        {
            var mapped: [String: BaseConfiguration.QuantizationOption] = [:]
            for (path, value) in quantization.perLayerQuantization {
                // Encoder wrappers own the same backbone under one additional module level.
                let native =
                    path.hasPrefix("backbone.") || path.hasPrefix("custom_text_proj")
                        || path == "linear"
                    ? path : "backbone." + path
                mapped[native] = value
            }
            perLayerQuantization = .init(
                quantization: quantization.quantization, perLayerQuantization: mapped)
        }
        try await CheckpointLoader.load(
            modelDirectory: directory, model: model,
            quantization: quantization, perLayerQuantization: perLayerQuantization,
            auxiliaryWeightDirectories: auxiliary
        )
        let tokenizer = try await tokenizerTask
        var maximumLength = model.maximumSequenceLength
        for (root, file, key) in [
            (directory, "sentence_bert_config.json", "max_seq_length"),
            (assets.tokenizerDirectory, "tokenizer_config.json", "model_max_length"),
        ] {
            let url = root.appendingPathComponent(file)
            guard FileManager.default.fileExists(atPath: url.path) else { continue }
            let config = try JSONDecoder.json5().decode(
                [String: JSONValue].self, from: Data(contentsOf: url))
            // Hugging Face may use a very large sentinel for an unspecified tokenizer limit.
            if case .int(let value) = config[key], value > 0 {
                maximumLength = min(maximumLength, value)
            }
        }
        let textProcessor = EmbeddingTextProcessor(
            tokenizer: tokenizer, maximumLength: maximumLength)
        if let gemma = model as? EmbeddingGemma2Model {
            let processor = try EmbeddingGemma2Processor(
                tokenizer: tokenizer, model: gemma.config,
                settings: embeddingProcessorSettings(directory: directory),
                maximumLength: maximumLength)
            return LoadedModel(
                assets: assets, metadata: metadata, model: model,
                processor: processor, tokenizer: tokenizer)
        }
        let visual: (any UserInputProcessor)?
        if let nemotron = model as? LlamaNemotronEmbeddingModel {
            let settings = try embeddingProcessorSettings(directory: directory)
            visual = LlamaNemotronEmbeddingProcessor(
                tokenizer: tokenizer, config: nemotron.config, settings: settings)
        } else if let colidefics = model as? ColIdefics3EmbeddingModel {
            let config = colidefics.backbone.config
            var settings = try embeddingProcessorSettings(directory: directory)
            let size = config.visionConfig.imageSize
            let sequence = size / config.visionConfig.patchSize / config.scaleFactor
            settings["image_seq_len"] = .int(sequence * sequence)
            settings["max_image_size"] = .object(["longest_edge": .int(size)])
            settings["size"] = settings["size"] ?? .object(["longest_edge": .int(size * 4)])
            settings["video_sampling"] = .object(["fps": .int(1), "max_frames": .int(1)])
            settings["image_mean"] =
                settings["image_mean"] ?? .array([.double(0.5), .double(0.5), .double(0.5)])
            settings["image_std"] =
                settings["image_std"] ?? .array([.double(0.5), .double(0.5), .double(0.5)])
            visual = try ColIdefics3EmbeddingProcessor(
                tokenizer: tokenizer, config: config,
                settings: JSONDecoder().decode(
                    SmolVLMProcessorConfiguration.self, from: JSONEncoder().encode(settings)),
                splitImages: settings["do_image_splitting"] != .bool(false))
        } else if model is Qwen3VLEmbeddingModel || model is ColQwen25EmbeddingModel {
            let family = model is Qwen3VLEmbeddingModel ? "Qwen3VLProcessor" : "Qwen2_5_VLProcessor"
            let settings = try embeddingProcessorSettings(directory: directory)
            visual = try registry.createProcessor(
                configuration: JSONEncoder().encode(settings),
                processorType: family, tokenizer: tokenizer)
        } else {
            visual = nil
        }
        return LoadedModel(
            assets: assets, metadata: metadata, model: model,
            processor: EmbeddingInputProcessor(
                text: textProcessor, visual: visual, maximumLength: maximumLength),
            tokenizer: tokenizer)
    }
}

struct EmbeddingTextProcessor: UserInputProcessor {
    let tokenizer: any Tokenizer
    let maximumLength: Int

    func prepare(input: UserInput) async throws -> LMInput {
        try Task.checkCancellation()
        guard case .text(let text) = input.prompt,
            input.images.isEmpty, input.videos.isEmpty, input.audios.isEmpty,
            input.tools?.isEmpty != false,
            !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw ModelExecutionError.invalidOptions(
                "Text embeddings require nonempty raw text without chat, media or tools")
        }
        let ids = tokenizer.encode(text: text, addSpecialTokens: true)
        guard !ids.isEmpty else {
            throw ModelExecutionError.invalidOptions("Embedding input produced no tokens")
        }
        // Refuse silent truncation: it can discard the document content being indexed.
        guard ids.count <= maximumLength else {
            throw ModelExecutionError.invalidOptions(
                "Embedding input exceeds the checkpoint limit of \(maximumLength) tokens")
        }
        return LMInput(text: .init(tokens: MLXArray(ids).reshaped([1, -1])))
    }
}

private func embeddingProcessorSettings(directory: URL) throws -> [String: JSONValue] {
    var settings: [String: JSONValue] = [:]
    for file in ["processor_config.json", "preprocessor_config.json"] {
        let url = directory.appendingPathComponent(file)
        if FileManager.default.fileExists(atPath: url.path) {
            let data = try JSONDecoder.json5().decode(
                [String: JSONValue].self, from: Data(contentsOf: url))
            settings.merge(data) { _, new in new }
        }
    }
    if case .object(let image) = settings["image_processor"] {
        settings.merge(image) { _, new in new }
    }
    return settings
}

struct EmbeddingInputProcessor: UserInputProcessor {
    let text: EmbeddingTextProcessor
    let visual: (any UserInputProcessor)?
    let maximumLength: Int
    func prepare(input: UserInput) async throws -> LMInput {
        try Task.checkCancellation()
        guard input.tools?.isEmpty != false, input.audios.isEmpty else {
            throw ModelExecutionError.invalidOptions(
                "Embedding inputs cannot contain audio or tools")
        }
        if input.images.isEmpty && input.videos.isEmpty {
            return try await text.prepare(input: input)
        }
        guard let visual, case .text = input.prompt else {
            throw ModelExecutionError.invalidOptions(
                "This embedding model does not support the requested media input")
        }
        let result = try await visual.prepare(input: input)
        guard result.text.tokens.size <= maximumLength else {
            throw ModelExecutionError.invalidOptions(
                "Visual embedding input exceeds the checkpoint token limit")
        }
        return result
    }
}
