import Foundation
import MLX

extension LoadedModel {
    consuming func intoNativEmbeddingModel(policy: ModelExecutionPolicy = .init())
        -> NativEmbeddingModel
    {
        NativEmbeddingModel(
            assets: assets, metadata: metadata,
            context: ExecutionTransfer(intoModelContext()), policy: policy)
    }
}

/// A loaded embedding encoder with exclusive preparation and inference requests.
public actor NativEmbeddingModel {
    public nonisolated let assets: ResolvedModelAssets
    public nonisolated let metadata: ModelMetadata
    nonisolated let id = UUID()
    let context: ModelContext
    let policy: ModelExecutionPolicy
    let admission = ExecutionAdmission()
    var queuedRequestCount: Int { admission.queuedCount }
    typealias RequestInput = NativModel.RequestInput

    init(
        assets: ResolvedModelAssets, metadata: ModelMetadata,
        context: ExecutionTransfer<ModelContext>, policy: ModelExecutionPolicy
    ) {
        self.assets = assets
        self.metadata = metadata
        self.context = context.take()
        self.policy = policy
    }

    public var capabilities: ModelCapabilities { [.embedding] }

    /// Exclusive low-level access. Do not retain resources or await another request on this model.
    public func withModel<R: Sendable>(
        _ body: @Sendable (isolated NativEmbeddingModel, ModelContext) async throws -> R
    ) async throws -> R {
        try await admission.acquire()
        defer {
            Stream.defaultStream.synchronize()
            admission.release()
        }
        try Task.checkCancellation()
        let result = try await body(self, context)
        try Task.checkCancellation()
        return result
    }

    public func prepare(input: sending UserInput) async throws -> sending PreparedModelInput {
        try await admission.acquire()
        defer {
            Stream.defaultStream.synchronize()
            admission.release()
        }
        try Task.checkCancellation()
        let tools = input.tools
        let prepared = try await context.processor.prepare(input: input)
        try Task.checkCancellation()
        return PreparedModelInput(
            tokenCount: prepared.text.tokens.size, ownerID: id,
            input: prepared, tools: tools)
    }
}
