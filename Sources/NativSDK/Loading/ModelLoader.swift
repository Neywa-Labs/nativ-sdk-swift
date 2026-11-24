import Foundation

/// Internal resource selection; public loading is specific to each model type.
enum ModelKind: Sendable {
    case language, embedding, decision, ocrScoring
}

/// Loading implementation with immutable constructors that tests can substitute.
struct ModelLoader: Sendable {
    let registry: ModelRegistry

    init(registry: ModelRegistry = .default) {
        self.registry = registry
    }

    /// Loads a model without starting generation or allocating request caches.
    /// Repository sources and missing Moondream tokenizers download through Hugging Face.
    func load(
        from source: ModelSource,
        tokenizerSource: ModelSource? = nil,
        using tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        metadata: ModelMetadata? = nil, policy: ModelExecutionPolicy = .init(),
        expertStreaming: ExpertStreamingOptions? = nil,
        progress: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> NativModel {
        let resources = try await loadResources(
            from: source, tokenizerSource: tokenizerSource, using: tokenizerLoader,
            metadata: metadata, expertStreaming: expertStreaming,
            progress: progress)
        try Task.checkCancellation()
        return resources.intoNativModel(policy: policy)
    }

    /// Loads a model from local assets without resolving or downloading files.
    func load(
        assets: ResolvedModelAssets,
        metadata: ModelMetadata = .init(),
        tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        policy: ModelExecutionPolicy = .init(),
        expertStreaming: ExpertStreamingOptions? = nil
    ) async throws -> NativModel {
        let resources = try await loadResources(
            assets: assets, metadata: metadata, tokenizerLoader: tokenizerLoader,
            expertStreaming: expertStreaming)
        try Task.checkCancellation()
        return resources.intoNativModel(policy: policy)
    }

    /// Loads resources independently of generation. Explicit metadata adds stopping
    /// conditions and overrides the model's tool and reasoning conventions.
    /// Repository sources and missing Moondream tokenizers download through Hugging Face.
    func loadResources(
        from source: ModelSource, kind: ModelKind = .language,
        tokenizerSource: ModelSource? = nil,
        using tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        metadata: ModelMetadata? = nil,
        expertStreaming: ExpertStreamingOptions? = nil,
        progress: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> sending LoadedModel {
        let assets = try await ModelArtifactResolver.resolve(
            model: source,
            tokenizer: tokenizerSource,
            modelPatterns: kind == .embedding
                ? ModelArtifactResolver.modelPatterns + [
                    "*_Dense/*.safetensors", "*_Dense/*.json", "1_Pooling/*.json",
                ]
                : ModelArtifactResolver.modelPatterns + [
                    "audio_tokenizer/*.json", "audio_tokenizer/*.safetensors",
                    "audio_detokenizer/*.json", "audio_detokenizer/*.safetensors",
                ],
            progress: progress)
        return try await loadResources(
            assets: assets, kind: kind, metadata: metadata ?? .init(),
            tokenizerLoader: tokenizerLoader, expertStreaming: expertStreaming)
    }

    /// Load already resolved local assets without resolving or downloading files.
    func loadResources(
        assets: ResolvedModelAssets, kind: ModelKind = .language,
        metadata: ModelMetadata = .init(),
        tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        expertStreaming: ExpertStreamingOptions? = nil
    ) async throws -> sending LoadedModel {
        let modelDirectory = assets.modelDirectory
        var configurationURL = modelDirectory.appending(component: "config.json")
        let configData: Data
        do {
            let nativeURL = modelDirectory.appending(component: "params.json")
            if !FileManager.default.fileExists(atPath: configurationURL.path),
                FileManager.default.fileExists(atPath: nativeURL.path),
                let normalized = try MistralLarge4Configuration.normalizeNative(
                    Data(contentsOf: nativeURL))
            {
                configurationURL = nativeURL
                configData = normalized
            } else {
                configData = try Data(contentsOf: configurationURL)
            }
        } catch {
            throw ModelFactoryError.configurationFileError(
                configurationURL.lastPathComponent, assets.name, error,
            )
        }

        if (try? isTargetBoundDraftConfiguration(configData)) == true {
            throw ModelExecutionError.invalidOptions(
                "This is a target-bound drafter; load it with NativDraftModel.loadModel(target:draft:)"
            )
        }
        let baseConfig: BaseConfiguration
        do {
            baseConfig = try JSONDecoder.json5().decode(BaseConfiguration.self, from: configData)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, assets.name, error,
            )
        }

        if kind == .ocrScoring {
            return try await loadOCRScoringResources(
                assets: assets, configuration: configData, baseConfiguration: baseConfig,
                metadata: metadata, tokenizerLoader: tokenizerLoader)
        }
        if kind == .embedding {
            return try await loadEmbeddingResources(
                assets: assets, configuration: configData, baseConfiguration: baseConfig,
                metadata: metadata, tokenizerLoader: tokenizerLoader)
        }
        if kind == .decision {
            return try await loadDecisionResources(
                assets: assets, configuration: configData, baseConfiguration: baseConfig,
                metadata: metadata, tokenizerLoader: tokenizerLoader)
        }

        // Text-only conversions can retain the original vision configuration.
        // Tensor names also identify optional modules whose weights are included.
        let weightNames: Set<String>?
        if ModelRegistry.needsWeightInspection(baseConfig.modelType) {
            let urls = try safetensorWeightURLs(in: modelDirectory)
            weightNames = try Set(
                urls.flatMap { try safetensorSpansInFileOrder(url: $0).map(\.name) })
        } else {
            weightNames = nil
        }
        let modelConfigData =
            if MuseSparkConfiguration.matches(baseConfig.modelType), let weightNames {
                try MuseSparkConfiguration.resolvingTowers(configData, weightNames: weightNames)
            } else {
                configData
            }

        let model: LanguageModel
        do {
            model = try registry.createModel(
                configuration: modelConfigData,
                modelType: baseConfig.modelType,
                weightNames: weightNames,
            )
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                configurationURL.lastPathComponent, assets.name, error,
            )
        }

        var eosTokenIDs = baseConfig.effectiveEOSTokenIds
        let generationConfigURL = modelDirectory.appending(component: "generation_config.json")
        let generationConfig: GenerationConfigFile? =
            if let generationData = try? Data(contentsOf: generationConfigURL) {
                try? JSONDecoder.json5().decode(GenerationConfigFile.self, from: generationData)
            } else {
                nil
            }
        if let generationEOSTokenIDs = generationConfig?.eosTokenIds?.values {
            eosTokenIDs = Set(generationEOSTokenIDs)
        }

        var mutableConfiguration = metadata
        mutableConfiguration.eosTokenIds.formUnion(eosTokenIDs)
        let supplementalTokens = try ModelConventions.supplementalStopTokens(
            modelType: baseConfig.modelType, configuration: configData)
        mutableConfiguration.extraEOSTokens.formUnion(supplementalTokens)
        mutableConfiguration.stopStrings.formUnion(supplementalTokens)
        mutableConfiguration.stopStrings.formUnion(generationConfig?.stopStrings ?? [])
        if mutableConfiguration.toolCallFormat == nil {
            mutableConfiguration.toolCallFormat = model.toolCallFormat
        }
        if mutableConfiguration.reasoningConfig == nil {
            mutableConfiguration.reasoningConfig =
                ModelConventions.reasoningConfig(
                    modelId: assets.name, modelType: baseConfig.modelType)
                ?? model.reasoningConfig
        }

        async let tokenizerTask = tokenizerLoader.load(from: assets.tokenizerDirectory)
        let sidecarQuantization =
            baseConfig.perLayerQuantization == nil
            ? try loadQuantizationConfig(from: modelDirectory)
            : nil

        let perLayerQuantization = try resolveLanguageModelQuantization(
            for: model, metadata: baseConfig.perLayerQuantization, sidecar: sidecarQuantization)
        try await CheckpointLoader.load(
            modelDirectory: modelDirectory,
            model: model,
            quantization: sidecarQuantization,
            perLayerQuantization: perLayerQuantization,
            expertStreaming: expertStreaming,
        )

        let tokenizer = try await tokenizerTask
        if let lfm3 = model as? LFM3Audio {
            lfm3.codecDirectory = ["audio_tokenizer", "audio_detokenizer"]
                .map { modelDirectory.appendingPathComponent($0) }
                .first {
                    FileManager.default.fileExists(
                        atPath: $0.appendingPathComponent("config.json").path)
                }
        }
        mutableConfiguration.eosTokenIds = StopPolicy.tokenIDs(
            configured: mutableConfiguration.eosTokenIds,
            supplementalTokens: mutableConfiguration.extraEOSTokens, tokenizer: tokenizer)
        let processor = try await loadProcessor(
            from: modelDirectory, modelType: baseConfig.modelType, configData: modelConfigData,
            model: model, tokenizer: tokenizer, messageGenerator: metadata.messageGenerator)

        return LoadedModel(
            assets: assets, metadata: mutableConfiguration,
            model: model, processor: processor, tokenizer: tokenizer)
    }

    func loadProcessor(
        from directory: URL, modelType: String, configData: Data,
        model: any LanguageModel, tokenizer: any Tokenizer,
        messageGenerator: (any MessageGenerator)? = nil
    ) async throws -> any UserInputProcessor {
        let textModel = model as? any LLMModel
        func textProcessor() -> TextProcessor {
            TextProcessor(
                .init(), tokenizer: tokenizer,
                messageGenerator: messageGenerator
                    ?? textModel?.messageGenerator(tokenizer: tokenizer)
                    ?? DefaultMessageGenerator())
        }

        // Text architectures need only tokenizer metadata. A mixed-modality model can
        // opt into processor configuration while retaining its text-only fallback.
        if textModel != nil,
            ModelRegistry.processorPolicies[modelType]?.supportsMultimodalInput != true
        {
            return textProcessor()
        }

        let (data, type) = try await loadProcessorConfig(
            from: directory, modelType: modelType, configData: configData)
        if type == "TextProcessor" {
            return textProcessor()
        }
        return try registry.createProcessor(
            configuration: data, processorType: type, tokenizer: tokenizer)
    }

    /// These processors draw geometry and token counts from model metadata, and
    /// may split image settings and chat settings across two sidecar files.
    private func loadComposedProcessorConfig(
        from directory: URL, modelType: String, configData: Data
    ) throws -> (Data, String) {
        let decoder = JSONDecoder.json5()
        let model = try decoder.decode([String: JSONValue].self, from: configData)
        var settings = model
        for name in ["processor_config.json", "preprocessor_config.json"] {
            let url = directory.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: url.path) {
                let metadata = try decoder.decode(
                    [String: JSONValue].self, from: Data(contentsOf: url))
                settings.merge(metadata) { _, new in new }
            }
        }
        if modelType == "lfm3_audio" || modelType == "lfm3_omni" || modelType == "oxr"
            || modelType == "nemotron_parse_v3_5"
            || DeepSeekOCRConfiguration.isUnlimitedOCR(modelType)
            || MuseSparkConfiguration.matches(modelType) || modelType == "inkling2"
        {
            return (
                try JSONEncoder().encode(settings),
                ModelRegistry.processorPolicies[modelType]!.fallback!
            )
        }
        let vision: [String: JSONValue]
        if case .object(let object) = model["vision_config"] {
            vision = object
        } else {
            vision = [:]
        }
        if modelType == "llmjpvl" {
            if settings["size"] == nil {
                settings["size"] = .object(["height": vision["image_size"] ?? .int(512)])
            }
            if settings["image_seq_length"] == nil {
                let config = try decoder.decode(LLMjpVLConfiguration.self, from: configData)
                let side =
                    Float(
                        config.visionConfiguration.imageSize / config.visionConfiguration.patchSize)
                    * config.downsampleRatio
                settings["image_seq_length"] = .int(Int(side * side))
            }
            let url = directory.appendingPathComponent("tokenizer_config.json")
            if settings["model_max_length"] == nil, FileManager.default.fileExists(atPath: url.path)
            {
                settings["model_max_length"] = try decoder.decode(
                    [String: JSONValue].self, from: Data(contentsOf: url))["model_max_length"]
            }
        } else if modelType == "minicpmv4_7" || modelType == "minicpm_ocr" {
            settings["processor_class"] = .string(
                modelType == "minicpm_ocr" ? "MiniCPMOcrProcessor" : "MiniCPMV4_7Processor")
            var image: [String: JSONValue] = [:]
            if case .object(let nested) = settings["image_processor"] { image = nested }
            for key in [
                "max_slice_nums", "scale_resolution", "image_mean", "image_std", "slice_mode",
                "use_image_id", "pack_image_by_patch",
                "disable_small_image_upsize", "adaptive_small_image_upsize",
                "adaptive_upsize_font_px",
                "adaptive_upsize_dense_per_kpx", "adaptive_upsize_tiny_pixels",
                "adaptive_upsize_tiny_target_resolution", "adaptive_tile_upsize",
            ] {
                if let value = settings[key] { image[key] = value }
            }
            image["downsample_mode"] =
                model["downsample_mode"] ?? .string(modelType == "minicpm_ocr" ? "4x" : "16x")
            image["patch_size"] = vision["patch_size"] ?? .int(14)
            settings["image_processor"] = .object(image)
        } else if modelType == "indic_ocr" {
            if case .object(let image) = settings["image_processor"] {
                settings.merge(image) { _, new in new }
            }
            let defaults: [String: JSONValue] = [
                "image_mean": .array([.double(0.5), .double(0.5), .double(0.5)]),
                "image_std": .array([.double(0.5), .double(0.5), .double(0.5)]),
                "patch_size": vision["patch_size"] ?? .int(16),
                "merge_size": vision["spatial_merge_size"] ?? .int(2),
                "temporal_patch_size": vision["temporal_patch_size"] ?? .int(2),
                "image_processor_type": .string("Qwen3VLImageProcessor"),
            ]
            settings = defaults.merging(settings) { _, new in new }
        } else if [
            "mage_vl", "llava_onevision", "llava_onevision2", "internvl", "mimo_v2", "qwen4_exp",
            "cohere_compass",
            "hunyuan_vl", "glm5_next", "minimax_m3", "minimax_m3_vl", "kimi_k3", "deepseek_v41",
            "nemotron_parse", "phi4-siglip", "phi4_siglip",
        ].contains(modelType) {
            if case .object(let image) = settings["image_processor"] {
                settings.merge(image) { _, new in new }
            }
            for (key, source) in [
                ("patch_size", "patch_size"), ("temporal_patch_size", "temporal_patch_size"),
                ("merge_size", "spatial_merge_size"), ("image_size", "image_size"),
            ] {
                if let value = vision[source] { settings[key] = value }
            }
            if let expand = vision["patch_expand_factor"] {
                settings["patch_expand_factor"] = expand
            }
            if case .object(let media) = model["media_config"] {
                settings.merge(media) { _, new in new }
            }
            if modelType == "llava_onevision", let imageSize = vision["image_size"] {
                settings["size"] = .object(["height": imageSize, "width": imageSize])
            }
            if modelType == "mimo_v2" || modelType == "qwen4_exp" {
                if settings["image_mean"] == nil {
                    settings["image_mean"] = .array([.double(0.5), .double(0.5), .double(0.5)])
                }
                if settings["image_std"] == nil {
                    settings["image_std"] = .array([.double(0.5), .double(0.5), .double(0.5)])
                }
            }
        } else {
            var image: [String: JSONValue] = [
                "image_mean": .array([.double(0.48145466), .double(0.4578275), .double(0.40821073)]
                ),
                "image_std": .array([.double(0.26862954), .double(0.26130258), .double(0.27577711)]
                ),
                "patch_size": vision["patch_size"] ?? .int(14),
                "size": .object(["longest_edge": vision["image_size"] ?? .int(1540)]),
            ]
            for key in image.keys { if let value = settings[key] { image[key] = value } }
            if case .object(let nested) = settings["image_processor"] {
                image.merge(nested) { _, new in new }
            }
            // Checkpoint geometry wins over an image sidecar's generic Pixtral defaults.
            if let patchSize = vision["patch_size"] { image["patch_size"] = patchSize }
            if case .object(let patch) = image["patch_size"] {
                image["patch_size"] = patch["height"] ?? patch["width"] ?? .int(14)
            }
            settings["image_processor"] = .object(image)
            settings["patch_size"] = image["patch_size"]
            if settings["image_token"] == nil { settings["image_token"] = .string("[IMG]") }
        }
        return (
            try JSONEncoder().encode(settings),
            ModelRegistry.processorPolicies[modelType]!.fallback!
        )
    }

    func loadProcessorConfig(
        from modelDirectory: URL,
        modelType: String,
        configData: Data,
    )
        async throws -> (Data, String)
    {
        if NemotronParse35Configuration.matches(configData) {
            return try loadComposedProcessorConfig(
                from: modelDirectory, modelType: "nemotron_parse_v3_5", configData: configData)
        }
        if [
            "llmjpvl", "indic_ocr", "mistral_large3", "mistral_large4", "mage_vl",
            "llava_onevision",
            "llava_onevision2",
            "internvl", "mimo_v2", "qwen4_exp", "cohere_compass", "hunyuan_vl", "glm5_next",
            "minimax_m3", "minimax_m3_vl", "kimi_k3", "deepseek_v41", "nemotron_parse",
            "phi4-siglip", "phi4_siglip", "minicpmv4_7", "minicpm_ocr", "lfm3_audio", "lfm3_omni",
            "oxr",
        ].contains(modelType) || DeepSeekOCRConfiguration.isUnlimitedOCR(modelType)
            || MuseSparkConfiguration.matches(modelType) || modelType == "inkling2"
        {
            return try loadComposedProcessorConfig(
                from: modelDirectory, modelType: modelType, configData: configData)
        }
        let processorConfigURL = modelDirectory.appending(component: "processor_config.json")
        let preprocessorConfigURL = modelDirectory.appending(component: "preprocessor_config.json")
        let fileManager = FileManager.default

        let configURL: URL? =
            if fileManager.fileExists(atPath: preprocessorConfigURL.path) {
                preprocessorConfigURL
            } else if fileManager.fileExists(atPath: processorConfigURL.path) {
                processorConfigURL
            } else {
                nil
            }

        if let configURL {
            do {
                let data = try Data(contentsOf: configURL)
                if let override = ModelRegistry.processorPolicies[modelType]?.override {
                    return (data, override)
                }
                let decoder = JSONDecoder.json5()
                do {
                    let config = try decoder.decode(BaseProcessorConfiguration.self, from: data)
                    return (data, config.processorClass)
                } catch DecodingError.keyNotFound(let key, _)
                    where key.stringValue == "processor_class"
                    && configURL == preprocessorConfigURL
                    && ["qwen3_5", "qwen3_5_moe"].contains(modelType)
                    && fileManager.fileExists(atPath: processorConfigURL.path)
                {
                    let config = try decoder.decode(
                        BaseProcessorConfiguration.self, from: Data(contentsOf: processorConfigURL))
                    return (data, config.processorClass)
                }
            } catch let error as DecodingError {
                throw ModelFactoryError.configurationDecodingError(
                    configURL.lastPathComponent,
                    modelDirectory.lastPathComponent,
                    error,
                )
            } catch {
                throw ModelFactoryError.configurationFileError(
                    configURL.lastPathComponent,
                    modelDirectory.lastPathComponent,
                    error,
                )
            }
        }

        if let fallback = ModelRegistry.processorPolicies[modelType]?.fallback {
            return (configData, fallback)
        }

        throw ModelFactoryError.configurationFileError(
            "processor_config.json",
            modelDirectory.lastPathComponent,
            CocoaError(.fileNoSuchFile),
        )
    }

    func loadQuantizationConfig(from modelDirectory: URL) throws
        -> BaseConfiguration.Quantization?
    {
        let quantizationURL = modelDirectory.appending(component: "quantization.json")
        guard FileManager.default.fileExists(atPath: quantizationURL.path) else {
            return nil
        }

        do {
            let data = try Data(contentsOf: quantizationURL)
            return try JSONDecoder.json5().decode(BaseConfiguration.Quantization.self, from: data)
        } catch let error as DecodingError {
            throw ModelFactoryError.configurationDecodingError(
                quantizationURL.lastPathComponent,
                modelDirectory.lastPathComponent,
                error,
            )
        } catch {
            throw ModelFactoryError.configurationFileError(
                quantizationURL.lastPathComponent,
                modelDirectory.lastPathComponent,
                error,
            )
        }
    }

}

public enum ModelFactoryError: LocalizedError {
    case unsupportedModelType(String)
    case unsupportedProcessorType(String)
    case configurationFileError(String, String, Error)
    case configurationDecodingError(String, String, DecodingError)
    case invalidConfiguration(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedModelType(let type):
            return "Unsupported model type: \(type)"
        case .unsupportedProcessorType(let type):
            return "Unsupported processor type: \(type)"
        case .configurationFileError(let file, let modelName, let error):
            return "Error reading '\(file)' for model '\(modelName)': \(error.localizedDescription)"
        case .invalidConfiguration(let message):
            return "Invalid model configuration: \(message)"
        case .configurationDecodingError(let file, let modelName, let decodingError):
            let errorDetail = extractDecodingErrorDetail(decodingError)
            return "Failed to parse \(file) for model '\(modelName)': \(errorDetail)"
        }
    }

    private func extractDecodingErrorDetail(_ error: DecodingError) -> String {
        switch error {
        case .keyNotFound(let key, let context):
            let path = (context.codingPath + [key]).map { $0.stringValue }.joined(separator: ".")
            return "Missing field '\(path)'"
        case .typeMismatch(_, let context):
            let path = context.codingPath.map { $0.stringValue }.joined(separator: ".")
            return "Type mismatch at '\(path)'"
        case .valueNotFound(_, let context):
            let path = context.codingPath.map { $0.stringValue }.joined(separator: ".")
            return "Missing value at '\(path)'"
        case .dataCorrupted(let context):
            if context.codingPath.isEmpty {
                return "Invalid JSON"
            } else {
                let path = context.codingPath.map { $0.stringValue }.joined(separator: ".")
                return "Invalid data at '\(path)'"
            }
        @unknown default:
            return error.localizedDescription
        }
    }
}

/// Checkpoint conventions are resolved from architecture metadata, independently of
/// repository aliases and quantization variants.
enum ModelConventions {
    static func reasoningConfig(modelId: String, modelType: String) -> ReasoningConfig? {
        let id = modelId.lowercased()
        return id.contains("deepseek-r1") || id.contains("r1-distill") ? .alwaysOnThinking : nil
    }

    private struct Backbone: Decodable {
        let modelType: String?
        enum CodingKeys: String, CodingKey { case modelType = "model_type" }
    }

    private struct Composite: Decodable {
        let text: Backbone?
        let llm: Backbone?
        enum CodingKeys: String, CodingKey {
            case text = "text_config"
            case llm = "llm_config"
        }
    }

    static func supplementalStopTokens(modelType: String, configuration: Data) throws -> Set<String>
    {
        switch modelType {
        case "mistral3":
            let composite = try JSONDecoder.json5().decode(Composite.self, from: configuration)
            return composite.text?.modelType == "mistral4" ? ["</s>"] : []
        case "llava", "llava_next", "llava_onevision", "internvl", "internvl_chat", "llmjpvl":
            let composite = try JSONDecoder.json5().decode(Composite.self, from: configuration)
            let backbone =
                composite.text?.modelType ?? composite.llm?.modelType
                ?? (modelType == "llava_next" ? "mistral" : "")
            switch backbone {
            case "qwen2", "qwen3": return ["<|im_end|>"]
            case "mistral": return ["</s>"]
            case "internlm2": return ["<|im_end|>"]
            default: return []
            }
        default: return stopTokens[modelType] ?? []
        }
    }

    // These are chat turn terminators, including strings that may span multiple tokens.
    // They supplement checkpoint EOS metadata; vocabulary membership alone is not a
    // reason to treat a token as a terminator for another model family.
    private static let stopTokens: [String: Set<String>] = [
        "llama": ["<|eot_id|>"],
        "iquestcoder": ["<|eot_id|>"],
        "llama4_text": ["<|end_of_text|>", "<|eom|>", "<|eot|>"],
        "phi3": ["<|end|>"],
        "indic_ocr": ["<|im_end|>"],
        "phimoe": ["<|end|>"],
        "qwen2": ["<|im_end|>"],
        "qwen": ["<|im_end|>"],
        "qwen2_moe": ["<|im_end|>"],
        "acereason": ["<|im_end|>"],
        "qwen3": ["<|im_end|>"],
        "qwen3_moe": ["<|im_end|>"],
        "qwen3_next": ["<|im_end|>"],
        "qwen3_5_text": ["<|im_end|>"],
        "qwen3_5_moe_text": ["<|im_end|>"],
        "gemma3_text": ["<end_of_turn>"],
        "gemma4_text": ["<turn|>"],
        "qwen2_vl": ["<|im_end|>"],
        "qwen2_5_vl": ["<|im_end|>"],
        "qwen3_vl": ["<|im_end|>"],
        "qwen3_vl_moe": ["<|im_end|>"],
        "qwen3_5": ["<|im_end|>"],
        "prism_hadamard_qwen35": ["<|im_end|>"],
        "qwen3_5_moe": ["<|im_end|>"],
        "qwen3_omni_moe": ["<|im_end|>"],
        "lfm3_audio": ["<|im_end|>", "<|text_end|>", "<|audio_start|>", "<|mixed_start|>"],
        "lfm3_omni": ["<|im_end|>", "<|text_end|>", "<|audio_start|>", "<|mixed_start|>"],
        "gemma3": ["<end_of_turn>"],
        "gemma3n": ["<end_of_turn>"],
        "gemma3n_text": ["<end_of_turn>"],
        "gemma4": ["<turn|>"],
        "gemma4_unified": ["<turn|>"],
        "muse_glimmer": ["<|eot|>", "<|end_of_text|>"],
        "muse_spark": ["<|eot|>", "<|end_of_text|>"],
        "muse_spark_open": ["<|eot|>", "<|end_of_text|>"],
        "phi3_v": ["</s>", "<|end|>", "<|endoftext|>"],
        "idefics2": ["</s>", "<end_of_utterance>"],
        "deepseek_vl_v2": ["<｜end▁of▁sentence｜>"],
        "glm4v": ["<|endoftext|>", "<|user|>", "<|observation|>", "</answer>"],
        "glm4v_moe": ["<|endoftext|>", "<|user|>", "<|observation|>"],
        "dots_ocr": ["<|endoftext|>", "<|endofassistant|>"],
        "falcon_ocr": ["<|end_of_text|>"],
        "falcon_perception": ["<|end_of_text|>"],
        "falcon-perception": ["<|end_of_text|>"],
        "paddleocr_vl": ["</s>"],
        "deepseekocr": ["<｜end▁of▁sentence｜>"],
        "deepseekocr_2": ["<｜end▁of▁sentence｜>"],
        "molmo": ["<|endoftext|>"],
        "molmo2": ["<|im_end|>"],
        "molmo_point": ["<|im_end|>"],
        "moondream1": ["<|endoftext|>"],
        "moondream2": ["<|endoftext|>"],
        "moondream3": ["<|endoftext|>"],
        "minicpmv4_6": ["<|im_end|>"],
        "minicpmv4_7": ["<|im_end|>"],
        "minicpm_ocr": ["<|im_end|>"],
        "oxr": ["<|im_end|>", "<|endoftext|>"],
        "minicpmo": ["<|im_end|>"],
        "granite-vision": ["<|end_of_text|>"],
        "granite4-vision": ["<|end_of_text|>"],
        "granite_vision": ["<|end_of_text|>"],
        "granite4_vision": ["<|end_of_text|>"],
        "aya_vision": ["<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>"],
        "cohere2_vision": ["<|END_RESPONSE|>", "<|END_OF_TURN_TOKEN|>"],
        "locateanything": ["<|im_end|>"],
        "bunny-llama": ["<|end_of_text|>", "<|eot_id|>"],
        "mllama": ["<|end_of_text|>", "<|eom_id|>", "<|eot_id|>"],
        "llama4": ["<|end_of_text|>", "<|eom|>", "<|eot|>"],
        "kimi_vl": ["<|im_end|>", "<|im_user|>"],
        "ernie4_5_moe_vl": ["</s>", "<|end_of_sentence|>"],
        "jvlm": ["<|endoftext|>"],
        "jina_vlm": ["<|endoftext|>"],
        "zaya1_vl": ["<|im_end|>"],
        "step3p7": ["<|im_end|>", "<｜end▁of▁sentence｜>"],
        "youtu_vl": ["<|end_of_text|>"],
        "mistral4": ["</s>"],
        "mistral_large4": ["</s>"],
        "mistral_large4_text": ["</s>"],
        "cohere2_moe": ["<|END_OF_TURN_TOKEN|>"],
        "laguna": ["<eos>", "</s>"],
        "llada2_moe": ["<|endoftext|>"],
        "nemotron_labs_diffusion": ["</s>"],
        "diffusion_gemma": ["<end_of_turn>", "</s>"],
        "deepseek_v4": ["<｜end▁of▁sentence｜>"],
    ]
}
