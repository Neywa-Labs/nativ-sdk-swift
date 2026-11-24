import Foundation

extension NativImageModel {
    /// Loads an image model using checkpoint metadata to select and configure its pipeline.
    public static func loadModel(
        from source: ModelSource, tokenizerSource: ModelSource? = nil,
        using tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        policy: ModelExecutionPolicy = .init(),
        progress: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> NativImageModel {
        try await ImageModelLoader().load(
            from: source, tokenizerSource: tokenizerSource,
            using: tokenizerLoader, policy: policy,
            progress: progress)
    }

    /// Loads supplied local assets without resolving or downloading files.
    public static func loadModel(
        assets: ResolvedModelAssets,
        tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        policy: ModelExecutionPolicy = .init()
    ) async throws -> NativImageModel {
        try await ImageModelLoader().load(
            assets: assets, tokenizerLoader: tokenizerLoader, policy: policy)
    }
}

struct ImageModelLoader: Sendable {
    let provider: any ModelSnapshotProvider

    init(provider: any ModelSnapshotProvider = HuggingFaceDownloader()) {
        self.provider = provider
    }

    func load(
        from source: ModelSource, tokenizerSource: ModelSource? = nil,
        using tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        policy: ModelExecutionPolicy = .init(),
        progress: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> NativImageModel {
        let modelSource: ModelSource
        if case .repository(let name, let revision) = source, !name.contains("/"),
            let variant = ErnieImageVariant.named(name)
        {
            modelSource = .repository(variant.repository, revision: revision)
        } else if case .repository(let name, let revision) = source, !name.contains("/"),
            let variant = MageFlowVariant.named(name)
        {
            modelSource = .repository(variant.repository, revision: revision)
        } else {
            modelSource = source
        }
        let patterns: [String]
        if case .repository(let id, _) = modelSource, ErnieImageVariant.named(id) != nil {
            patterns = [
                "model_index.json", "mlx_ernie_image.json", "scheduler/*.json",
                "transformer/*", "vae/*", "text_encoder/*", "tokenizer/*",
            ]
        } else if case .repository(let id, _) = source,
            ["openmoss-team/moss-image-base", "openmoss-team/moss-image-sft"].contains(
                id.lowercased())
        {
            patterns = [
                "model_index.json", "scheduler/*.json", "tokenizer/*",
                "text_encoder/*", "transformer/*.json", "transformer/*.safetensors", "vae/*",
            ]
        } else if case .repository(let id, _) = source,
            ["black-forest-labs/flux-3-image-klein", "black-forest-labs/flux-3-image-klein-base"]
                .contains(id.lowercased())
        {
            patterns = ["config.json", "model.safetensors", "ae.safetensors", "text_encoder/*"]
        } else if case .repository(let id, _) = source,
            id.lowercased() == "minimaxai/minimax-h3-image"
        {
            patterns = [
                "model_index.json", "tokenizer/*", "text_encoder/*",
                "transformer_diffusers/*", "vae_diffusers/*",
            ]
        } else if case .repository(let id, _) = modelSource,
            id.lowercased().hasPrefix("mage-flow-community/")
                || id.lowercased().hasPrefix("microsoft/mage-flow")
        {
            patterns = [
                "model_index.json", "mlx_mage_flow.json", "scheduler/*.json",
                "transformer/*", "vae/*", "text_encoder/*",
            ]
        } else {
            patterns = Self.downloadPatterns
        }
        let assets = try await ModelArtifactResolver.resolve(
            model: modelSource, tokenizer: tokenizerSource, provider: provider,
            modelPatterns: patterns, progress: progress)
        var conditioner: URL?
        let directory = assets.modelDirectory
        if !FileManager.default.fileExists(
            atPath: directory.appending(path: "text_encoder/config.json").path),
            let index = try ImageCheckpointMetadata.optional("model_index.json", in: directory),
            index.string("_class_name") == "MiniMaxH3ImagePipeline"
                || index.string("model_type") == "minimax_h3_image",
            let entry = index.values["text_encoder"] as? [Any], entry.count > 2,
            let reference = entry[2] as? [String: Any],
            let repository = reference["pretrained_model_name_or_path"] as? String,
            (reference["subfolder"] as? String ?? "text_encoder") == "text_encoder"
        {
            let shared = try await provider.snapshot(
                id: repository, revision: reference["revision"] as? String ?? "main",
                matching: ["text_encoder/*"], progress: progress)
            conditioner = shared.appending(path: "text_encoder")
        }
        var enhancer: (@Sendable () async throws -> URL)?
        if case .repository(let id, let revision) = modelSource {
            let resolvedRevision =
                directory.deletingLastPathComponent().lastPathComponent == "snapshots"
                ? directory.lastPathComponent : revision
            enhancer = {
                try await provider.snapshot(
                    id: id, revision: resolvedRevision,
                    matching: ErnieImagePromptExpansion.bundledPatterns, progress: progress)
            }
        }
        return try await load(
            assets: assets, tokenizerLoader: tokenizerLoader, policy: policy,
            h3Conditioner: conditioner, ernieEnhancer: enhancer)
    }

    func load(
        assets: ResolvedModelAssets,
        tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        policy: ModelExecutionPolicy = .init(),
        h3Conditioner: URL? = nil,
        ernieEnhancer: (@Sendable () async throws -> URL)? = nil
    ) async throws -> NativImageModel {
        let directory = assets.modelDirectory
        var isDirectory = ObjCBool(false)
        guard FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else {
            throw ModelExecutionError.invalidOptions(
                "Image model directory does not exist: \(directory.path)")
        }
        let selected = try ImagePipelineKind.detect(in: directory)
        let tokenizerDirectory: URL
        if assets.tokenizerSource != nil
            || assets.tokenizerDirectory.standardizedFileURL != directory.standardizedFileURL
        {
            tokenizerDirectory = assets.tokenizerDirectory
        } else {
            switch selected {
            case .flux3, .mageFlow: tokenizerDirectory = directory.appending(path: "text_encoder")
            case .qwenImage: tokenizerDirectory = directory.appending(path: "processor")
            case .mingImage: tokenizerDirectory = directory.appending(path: "mllm")
            default: tokenizerDirectory = Self.tokenizerDirectory(in: directory)
            }
        }
        let tokenizer = try await tokenizerLoader.load(from: tokenizerDirectory)
        let pipeline: any PromptImageGenerationModel
        switch selected {
        case .ernieImage:
            pipeline = try ErnieImagePipeline(
                directory: directory, source: assets.source, bundledEnhancer: ernieEnhancer,
                enhancerTokenizerLoader: tokenizerLoader)
        case .flux2:
            pipeline = try Flux2WeightLoader.loadPipeline(
                from: directory)
        case .flux3:
            let metadata = try ImageCheckpointMetadata.required("config.json", in: directory)
            pipeline = try Flux3ImagePipeline(
                directory: directory, config: Flux3Configuration(metadata))
        case .bonsai:
            pipeline = try BonsaiWeightLoader.loadPipeline(from: directory)
        case .ideogram4:
            let ideogram = try Ideogram4WeightLoader.loadPipeline(
                from: directory, loadTransformers: false, loadVAE: false)
            ideogram.componentDirectory = directory
            pipeline = ideogram
        case .zImage: pipeline = try ZImagePipeline(directory: directory)
        case .qwenImage: pipeline = try QwenImagePipeline(directory: directory)
        case .mageFlow:
            pipeline = try MageFlowImagePipeline(directory: directory, source: assets.source)
        case .mingImage: pipeline = MingImagePipeline(directory: directory)
        case .mossImage: pipeline = try MOSSImagePipeline(directory: directory)
        case .minimaxH3Image:
            pipeline = try MiniMaxH3ImagePipeline(
                directory: directory, conditionerDirectory: h3Conditioner)
        case .lladaImage: pipeline = try LLaDAImagePipeline(directory: directory)
        }
        try Task.checkCancellation()
        return NativImageModel(
            resources: ExecutionTransfer((pipeline, tokenizer)), source: assets.source,
            modelDirectory: directory,
            policy: policy)
    }

    static let downloadPatterns = [
        "*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "*.tiktoken", "tokenizer*",
        "**/*.json", "**/*.safetensors", "**/*.txt", "**/*.jinja", "**/*.model", "**/*.tiktoken",
        "**/tokenizer*",
    ]

    static func tokenizerDirectory(in directory: URL) -> URL {
        let tokenizer = directory.appending(path: "tokenizer")
        return FileManager.default.fileExists(atPath: tokenizer.path)
            ? tokenizer : directory.appending(path: "text_encoder")
    }
}
