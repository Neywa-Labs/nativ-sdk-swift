import Foundation

extension NativModel {
    /// Loads a language or multimodal model without starting generation.
    public static func loadModel(
        from source: ModelSource, tokenizerSource: ModelSource? = nil,
        using tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        metadata: ModelMetadata? = nil, policy: ModelExecutionPolicy = .init(),
        expertStreaming: ExpertStreamingOptions? = nil,
        progress: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> NativModel {
        try await ModelLoader().load(
            from: source, tokenizerSource: tokenizerSource,
            using: tokenizerLoader, metadata: metadata, policy: policy,
            expertStreaming: expertStreaming, progress: progress)
    }

    /// Uses resolved local assets without resolving or downloading files.
    public static func loadModel(
        assets: ResolvedModelAssets, metadata: ModelMetadata = .init(),
        tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        policy: ModelExecutionPolicy = .init(),
        expertStreaming: ExpertStreamingOptions? = nil
    ) async throws -> NativModel {
        try await ModelLoader().load(
            assets: assets, metadata: metadata,
            tokenizerLoader: tokenizerLoader, policy: policy,
            expertStreaming: expertStreaming)
    }
}

extension NativEmbeddingModel {
    /// Loads an embedding encoder, including retrieval projections and pooling metadata.
    public static func loadModel(
        from source: ModelSource, tokenizerSource: ModelSource? = nil,
        using tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        metadata: ModelMetadata? = nil, policy: ModelExecutionPolicy = .init(),
        progress: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> NativEmbeddingModel {
        let resources = try await ModelLoader().loadResources(
            from: source, kind: .embedding,
            tokenizerSource: tokenizerSource, using: tokenizerLoader, metadata: metadata,
            progress: progress)
        try Task.checkCancellation()
        return resources.intoNativEmbeddingModel(policy: policy)
    }

    public static func loadModel(
        assets: ResolvedModelAssets, metadata: ModelMetadata = .init(),
        tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        policy: ModelExecutionPolicy = .init()
    ) async throws -> NativEmbeddingModel {
        let resources = try await ModelLoader().loadResources(
            assets: assets, kind: .embedding,
            metadata: metadata, tokenizerLoader: tokenizerLoader)
        try Task.checkCancellation()
        return resources.intoNativEmbeddingModel(policy: policy)
    }
}

extension NativDecisionModel {
    /// Loads a native decision model without starting prediction.
    public static func loadModel(
        from source: ModelSource, tokenizerSource: ModelSource? = nil,
        using tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        metadata: ModelMetadata? = nil, policy: ModelExecutionPolicy = .init(),
        progress: @Sendable @escaping (Progress) -> Void = { _ in }
    ) async throws -> NativDecisionModel {
        let resources = try await ModelLoader().loadResources(
            from: source, kind: .decision,
            tokenizerSource: tokenizerSource, using: tokenizerLoader, metadata: metadata,
            progress: progress)
        try Task.checkCancellation()
        return resources.intoNativDecisionModel(policy: policy)
    }

    public static func loadModel(
        assets: ResolvedModelAssets, metadata: ModelMetadata = .init(),
        tokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader(),
        policy: ModelExecutionPolicy = .init()
    ) async throws -> NativDecisionModel {
        let resources = try await ModelLoader().loadResources(
            assets: assets, kind: .decision,
            metadata: metadata, tokenizerLoader: tokenizerLoader)
        try Task.checkCancellation()
        return resources.intoNativDecisionModel(policy: policy)
    }
}
