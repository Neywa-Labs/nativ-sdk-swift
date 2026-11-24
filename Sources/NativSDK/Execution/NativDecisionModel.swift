import Foundation
import MLX

extension LoadedModel {
    consuming func intoNativDecisionModel(policy: ModelExecutionPolicy = .init())
        -> NativDecisionModel
    {
        NativDecisionModel(
            assets: assets, metadata: metadata,
            context: ExecutionTransfer(intoModelContext()), policy: policy)
    }
}

/// A loaded decision model. Complete prediction requests have exclusive resource access.
public actor NativDecisionModel {
    public nonisolated let assets: ResolvedModelAssets
    public nonisolated let metadata: ModelMetadata
    let context: ModelContext
    let policy: ModelExecutionPolicy
    let admission = ExecutionAdmission()
    var queuedRequestCount: Int { admission.queuedCount }

    init(
        assets: ResolvedModelAssets, metadata: ModelMetadata,
        context: ExecutionTransfer<ModelContext>, policy: ModelExecutionPolicy
    ) {
        self.assets = assets
        self.metadata = metadata
        self.context = context.take()
        self.policy = policy
    }

    public var capabilities: ModelCapabilities { [.decision] }

    /// Exclusive low-level access. Do not retain resources or await another request on this model.
    public func withModel<R: Sendable>(
        _ body: @Sendable (isolated NativDecisionModel, ModelContext) async throws -> R
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
}
