import Foundation
import MLX

/// Models with out-of-cache state must explicitly copy every state field.
/// The returned state must remain independent after either continuation mutates.
/// Opaque LMOutput.State values cannot be generically inspected or serialized.
public protocol ContinuationStateCopying: LanguageModel {
    func copyContinuationState(_ state: LMOutput.State) throws -> LMOutput.State
}

public enum ContinuationError: Error, Equatable {
    case incompatibleScope
    case unsupportedCache(Int)
    case unsupportedModelState
    case unsupportedMedia
    case invalidPosition
}

/// An in-process restoration boundary. Model instance identity is checked;
/// callers supply revisions for processor behavior and model/configuration changes.
/// Change these revisions after changing weights, adapters, tokenizer, or preprocessing.
public struct ContinuationScope: Equatable {
    private let model: any LanguageModel
    public let processorRevision: String
    public let configurationRevision: String
    public let cacheConfiguration: KVCacheConfiguration?

    public init(
        model: any LanguageModel, processorRevision: String, configurationRevision: String,
        cacheConfiguration: KVCacheConfiguration? = nil
    ) {
        self.model = model
        self.processorRevision = processorRevision
        self.configurationRevision = configurationRevision
        self.cacheConfiguration = cacheConfiguration
    }

    public static func == (lhs: Self, rhs: Self) -> Bool {
        ObjectIdentifier(lhs.model) == ObjectIdentifier(rhs.model)
            && lhs.processorRevision == rhs.processorRevision
            && lhs.configurationRevision == rhs.configurationRevision
            && lhs.cacheConfiguration == rhs.cacheConfiguration
    }

    fileprivate func contains(_ model: any LanguageModel) -> Bool {
        ObjectIdentifier(self.model) == ObjectIdentifier(model)
    }
}

/// Complete model continuation at a committed token boundary, before sampling.
/// This is not a generation checkpoint: sampler RNG, pending lookahead tokens,
/// output parsing and chat history belong to the request/session, not the model.
/// Capture and restore on the model's serialized execution boundary. Not Sendable.
/// Media reuse and persisted representations are intentionally unsupported.
public struct ContinuationSnapshot {
    public let processedTokenCount: Int
    public let tokens: [Int]
    public let scope: ContinuationScope
    private let caches: [KVCache]
    private let state: LMOutput.State?

    public init(
        model: any LanguageModel, caches: [KVCache], state: LMOutput.State?, tokens: [Int],
        scope: ContinuationScope, mediaIdentities: [String] = []
    ) throws {
        guard scope.contains(model) else { throw ContinuationError.incompatibleScope }
        guard mediaIdentities.isEmpty else { throw ContinuationError.unsupportedMedia }
        for (index, cache) in caches.enumerated() {
            guard cacheCapabilities(cache).contains(.snapshot) else {
                throw ContinuationError.unsupportedCache(index)
            }
        }
        // Recurrent slots have no token offset. Track the logical position separately.
        for leaf in KVCacheTree.leaves(in: caches)
        where cacheCapabilities(leaf.cache).contains(.attention) {
            guard leaf.cache.offset == tokens.count else {
                throw ContinuationError.invalidPosition
            }
        }
        self.scope = scope
        self.tokens = tokens
        self.processedTokenCount = tokens.count
        self.state = try copyModelContinuationState(state, model: model)
        self.caches = caches.map { $0.copy() }
        eval(self.caches.flatMap { $0.innerState() })
    }

    /// Copies again so a snapshot can be restored repeatedly without sharing mutation.
    public func restore(model: any LanguageModel, scope: ContinuationScope) throws
        -> (caches: [KVCache], state: LMOutput.State?)
    {
        guard self.scope == scope, scope.contains(model) else {
            throw ContinuationError.incompatibleScope
        }
        let state = try copyModelContinuationState(self.state, model: model)
        let copies = caches.map { $0.copy() }
        eval(copies.flatMap { $0.innerState() })
        return (copies, state)
    }
}

func copyModelContinuationState(_ state: LMOutput.State?, model: any LanguageModel) throws
    -> LMOutput.State?
{
    guard let state else { return nil }
    guard let model = model as? any ContinuationStateCopying else {
        throw ContinuationError.unsupportedModelState
    }
    return try model.copyContinuationState(state)
}
