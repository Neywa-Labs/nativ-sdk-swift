import Foundation
import MLX

extension NativDraftModel {
    /// Loads a target/draft pair into one model without starting generation.
    /// Progress reports each repository download in sequence. Resolve assets separately
    /// to customize either tokenizer source or the downloader.
    public static func loadModel(
        target: ModelSource, draft: ModelSource,
        using tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        metadata: ModelMetadata = .init(), draftMetadata: ModelMetadata = .init(),
        policy: ModelExecutionPolicy = .init(),
        progress: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> NativDraftModel {
        try await ModelLoader().loadSpeculativeExecutor(
            target: target, draft: draft, using: tokenizerLoader,
            metadata: metadata, draftMetadata: draftMetadata, policy: policy, progress: progress)
    }

    /// Loads local assets. Independent drafts need matching tokenizers; supported
    /// target-bound drafters use the target tokenizer and load prediction heads directly.
    public static func loadModel(
        target: ResolvedModelAssets, draft: ResolvedModelAssets,
        metadata: ModelMetadata = .init(), draftMetadata: ModelMetadata = .init(),
        tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        policy: ModelExecutionPolicy = .init()
    ) async throws -> NativDraftModel {
        try await ModelLoader().loadSpeculativeExecutor(
            target: target, draft: draft, metadata: metadata, draftMetadata: draftMetadata,
            tokenizerLoader: tokenizerLoader, policy: policy)
    }
}

extension ModelLoader {
    /// Loads a target/draft pair into one model without starting generation.
    /// Progress reports each repository download in sequence. Resolve assets separately
    /// to customize either tokenizer source or the downloader.
    func loadSpeculativeExecutor(
        target: ModelSource, draft: ModelSource,
        using tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        metadata: ModelMetadata = .init(), draftMetadata: ModelMetadata = .init(),
        policy: ModelExecutionPolicy = .init(),
        progress: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> NativDraftModel {
        let targetAssets = try await ModelArtifactResolver.resolve(
            model: target, progress: progress)
        let draftAssets = try await ModelArtifactResolver.resolve(model: draft, progress: progress)
        return try await loadSpeculativeExecutor(
            target: targetAssets, draft: draftAssets, metadata: metadata,
            draftMetadata: draftMetadata, tokenizerLoader: tokenizerLoader, policy: policy)
    }

    /// Loads local assets. Independent drafts need matching tokenizers; supported
    /// target-bound drafters use the target tokenizer and load prediction heads directly.
    func loadSpeculativeExecutor(
        target: ResolvedModelAssets, draft: ResolvedModelAssets,
        metadata: ModelMetadata = .init(), draftMetadata: ModelMetadata = .init(),
        tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        policy: ModelExecutionPolicy = .init()
    ) async throws -> NativDraftModel {
        try Task.checkCancellation()
        let draftConfigURL = draft.modelDirectory.appendingPathComponent("config.json")
        let draftConfigData =
            FileManager.default.fileExists(atPath: draftConfigURL.path)
            ? try Data(contentsOf: draftConfigURL) : nil
        if let draftConfigData,
            let root = try JSONSerialization.jsonObject(with: draftConfigData) as? [String: Any],
            root["model_type"] as? String == "qwen3_5_mtp"
        {
            guard let text = root["text_config"] as? [String: Any] else {
                throw ModelExecutionError.invalidOptions("Qwen MTP requires text_config")
            }
            let configuration = try JSONDecoder().decode(
                Qwen35TextConfiguration.self,
                from: JSONSerialization.data(withJSONObject: text))
            guard configuration.mtpNumHiddenLayers > 0 else {
                throw ModelExecutionError.invalidOptions(
                    "Qwen MTP requires trained prediction layers")
            }
            let resources = try await loadResources(
                assets: target, metadata: metadata, tokenizerLoader: tokenizerLoader)
            let context = resources.intoModelContext()
            let drafter = Qwen35MTPDraftModel(configuration)
            try drafter.validate(target: context.languageModel)
            let base = try JSONDecoder().decode(BaseConfiguration.self, from: draftConfigData)
            try await CheckpointLoader.load(
                modelDirectory: draft.modelDirectory, model: drafter,
                quantization: base.quantizationContainer?.quantization,
                perLayerQuantization: base.perLayerQuantization.map { settings in
                    .init(
                        quantization: settings.quantization,
                        perLayerQuantization:
                            Dictionary(
                                uniqueKeysWithValues: settings.perLayerQuantization.map {
                                    key, value in
                                    (key.hasPrefix("mtp.") ? key : "mtp." + key, value)
                                }))
                })
            return NativDraftModel(
                target: ExecutionTransfer(context),
                draftResources: ExecutionTransfer(.mtp(drafter)), policy: policy)
        }
        if let draftConfigData, try LagunaDFlashConfiguration.matches(draftConfigData) {
            let configuration = try LagunaDFlashConfiguration(data: draftConfigData)
            let resources = try await loadResources(
                assets: target, metadata: metadata, tokenizerLoader: tokenizerLoader)
            let context = resources.intoModelContext()
            try configuration.validate(target: resolveDFlashTarget(context.languageModel))
            let drafter = LagunaDFlashModel(configuration)
            let base = try JSONDecoder().decode(BaseConfiguration.self, from: draftConfigData)
            try await CheckpointLoader.load(
                modelDirectory: draft.modelDirectory, model: drafter,
                quantization: base.quantizationContainer?.quantization,
                perLayerQuantization: base.perLayerQuantization)
            try Task.checkCancellation()
            return NativDraftModel(
                target: ExecutionTransfer(context),
                draftResources: ExecutionTransfer(.dspark(drafter)), policy: policy)
        }
        if let draftConfigData, try isTargetBoundDraftConfiguration(draftConfigData) {
            let configuration = try DSparkConfiguration(data: draftConfigData)
            let targetResources = try await loadResources(
                assets: target, metadata: metadata, tokenizerLoader: tokenizerLoader)
            let context = targetResources.intoModelContext()
            let targetModel = try resolveDFlashTarget(context.languageModel)
            try configuration.validate(target: targetModel)
            let drafter = DSparkModel(configuration)
            let base = try JSONDecoder().decode(BaseConfiguration.self, from: draftConfigData)
            try await CheckpointLoader.load(
                modelDirectory: draft.modelDirectory, model: drafter,
                quantization: base.quantizationContainer?.quantization,
                perLayerQuantization: base.perLayerQuantization)
            guard
                !configuration.requiresBF16Target
                    || drafter.parameters().flattened().allSatisfy({ $0.1.dtype == .bfloat16 })
            else {
                throw ModelExecutionError.invalidOptions("DSpark requires BF16 draft weights")
            }
            try Task.checkCancellation()
            drafter.prepareContextProjection()
            return NativDraftModel(
                target: ExecutionTransfer(context),
                draftResources: ExecutionTransfer(.dspark(drafter)), policy: policy)
        }
        try validateSpeculativeTokenizers(target.tokenizerDirectory, draft.tokenizerDirectory)
        let targetResources = try await loadResources(
            assets: target, metadata: metadata, tokenizerLoader: tokenizerLoader)
        try Task.checkCancellation()
        let draftResources = try await loadResources(
            assets: draft, metadata: draftMetadata, tokenizerLoader: tokenizerLoader)
        try Task.checkCancellation()
        return NativDraftModel(
            target: ExecutionTransfer(targetResources.intoModelContext()),
            draft: ExecutionTransfer(draftResources.intoModelContext()), policy: policy)
    }
}

extension LoadedModel {
    /// Both models must use identical tokenizer assets and opt into causal chunk verification.
    consuming func intoSpeculativeExecutor(
        draft: consuming LoadedModel,
        policy: ModelExecutionPolicy = .init()
    ) throws -> NativDraftModel {
        try validateSpeculativeTokenizers(
            assets.tokenizerDirectory, draft.assets.tokenizerDirectory)
        return NativDraftModel(
            target: ExecutionTransfer(intoModelContext()),
            draft: ExecutionTransfer(draft.intoModelContext()), policy: policy)
    }
}

func validateSpeculativeTokenizers(_ target: URL, _ draft: URL) throws {
    if target.standardizedFileURL == draft.standardizedFileURL { return }
    let files = [
        "tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt", "merges.txt",
        "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    ]
    var hasVocabulary = false
    for name in files {
        let a = target.appendingPathComponent(name)
        let b = draft.appendingPathComponent(name)
        let aExists = FileManager.default.fileExists(atPath: a.path)
        let bExists = FileManager.default.fileExists(atPath: b.path)
        guard aExists == bExists else {
            throw ModelExecutionError.invalidOptions("target and draft tokenizer assets differ")
        }
        if aExists {
            guard try Data(contentsOf: a) == Data(contentsOf: b) else {
                throw ModelExecutionError.invalidOptions("target and draft tokenizer assets differ")
            }
            if ["tokenizer.json", "tokenizer.model", "vocab.json", "vocab.txt"].contains(name) {
                hasVocabulary = true
            }
        }
    }
    guard hasVocabulary else {
        throw ModelExecutionError.invalidOptions("cannot verify draft tokenizer vocabulary")
    }
}

enum DraftModelResources {
    case autoregressive(ModelContext)
    case dspark(any DFlashDrafter)
    case mtp(Qwen35MTPDraftModel)
}

/// Owns a target/draft pair and serializes whole speculative requests.
/// Rejected inputs rewind attention tails and architecture-specific temporal state.
public actor NativDraftModel {
    private let target: ModelContext
    private let draft: DraftModelResources
    private let policy: ModelExecutionPolicy
    let admission = ExecutionAdmission()
    init(
        target: ExecutionTransfer<ModelContext>, draft: ExecutionTransfer<ModelContext>,
        policy: ModelExecutionPolicy
    ) {
        self.target = target.take()
        self.draft = .autoregressive(draft.take())
        self.policy = policy
    }
    init(
        target: ExecutionTransfer<ModelContext>,
        draftResources: ExecutionTransfer<DraftModelResources>,
        policy: ModelExecutionPolicy
    ) {
        self.target = target.take()
        self.draft = draftResources.take()
        self.policy = policy
    }
    public nonisolated func generate(
        input: sending UserInput,
        options: ModelGenerationOptions = .init(sampling: .init(temperature: 0)),
        speculation: SpeculativeGenerationOptions = .init()
    ) -> ModelGeneration {
        let transfer = ExecutionTransfer(input)
        let (stream, continuation) = AsyncThrowingStream<ModelGenerationEvent, Error>.makeStream(
            bufferingPolicy: .bufferingOldest(max(1, options.output.bufferCapacity)))
        let task = Task {
            do {
                let result = try await self.run(
                    input: transfer, options: options, speculation: speculation, emit: continuation)
                try Task.checkCancellation()
                try GenerationDriver.emit(.completion(result), to: continuation)
                continuation.finish()
                return result
            } catch {
                continuation.finish(throwing: error)
                throw error
            }
        }
        continuation.onTermination = { if case .cancelled = $0 { task.cancel() } }
        return ModelGeneration(stream: stream, task: task)
    }
    private func run(
        input: ExecutionTransfer<UserInput>, options: ModelGenerationOptions,
        speculation: SpeculativeGenerationOptions,
        emit: AsyncThrowingStream<ModelGenerationEvent, Error>.Continuation
    ) async throws -> ModelGenerationSummary {
        try await admission.acquire()
        defer {
            Stream.defaultStream.synchronize()
            admission.release()
        }
        try Task.checkCancellation()
        return try await runGeneration(
            input: input, options: options, speculation: speculation, emit: emit)
    }

    /// Runs under the model admission lock, shared by direct requests and sessions.
    func runGeneration(
        input: ExecutionTransfer<UserInput>, options: ModelGenerationOptions,
        speculation: SpeculativeGenerationOptions,
        emit: AsyncThrowingStream<ModelGenerationEvent, Error>.Continuation,
        observe: @escaping @Sendable (ModelGenerationEvent) -> Void = { _ in },
        sessionCache: DraftCacheVault? = nil,
        cacheReuse: PromptCacheReuseConfiguration = .init()
    ) async throws -> ModelGenerationSummary {
        switch draft {
        case .autoregressive(let draft):
            try SpeculativeIterator.validate(
                options: options, speculation: speculation,
                target: target.languageModel, draft: draft.languageModel)
        case .mtp(let draft):
            try MTPSpeculativeTokenIterator.validate(
                options: options, speculation: speculation,
                target: target.languageModel, draft: draft)
        case .dspark(let draft):
            let model = try resolveDFlashTarget(target.languageModel)
            try DFlashIterator.validate(
                options: options, speculation: speculation, target: model, draft: draft)
        }
        let userInput = input.take()
        let allowsImages: Bool
        if case .dspark = draft {
            allowsImages =
                try target.languageModel is LFM2VL || target.languageModel is Qwen35
                || target.languageModel is any Gemma4DraftTarget
        } else {
            allowsImages = false
        }
        guard userInput.images.isEmpty || allowsImages, userInput.videos.isEmpty,
            userInput.audios.isEmpty
        else {
            throw ModelExecutionError.invalidOptions(
                "drafting supports images with LFM2-VL, Qwen3.5 and Gemma4 targets; video and audio are unsupported"
            )
        }
        let tools = userInput.tools
        let prepared = try await target.processor.prepare(input: userInput)
        let transfer = ExecutionTransfer((prepared, target, draft))
        return try await runModelWorker(policy: policy) {
            let (input, target, draft) = transfer.take()
            return try DraftCacheEngine.run(
                input: input, target: target, draft: draft, tools: tools,
                options: options, speculation: speculation, vault: sessionCache,
                configuration: cacheReuse
            ) {
                observe($0)
                try GenerationDriver.emit($0, to: emit)
            }
        }
    }
}
