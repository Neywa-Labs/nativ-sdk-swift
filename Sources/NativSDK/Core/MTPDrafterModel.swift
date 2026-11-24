// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Protocol for Multi-Token Prediction (MTP) speculative drafter models.
///
/// Refines `BaseLanguageModel` with drafter-specific operations, so
/// implementations inherit weight loading and `sanitize` hooks while defining
/// their own forward signature.
///
/// MTP drafters do **not** conform to `LanguageModel` — their I/O contract is
/// different: a drafter consumes the target's last hidden state and per
/// layer-type shared K/V, produces a block of K-1 candidate tokens in a
/// single call, and does not store transient round-state on the model object.
/// Callers extract shared K/V from the target's `LMOutput.state` and pass
/// it to the drafter as a method argument.
///
/// Conforming types are expected to be stateless with respect to the target
/// model: every per-round input — including the target itself — flows through
/// `draftBlock(...)` as a method parameter. This makes drafter instances safe
/// to share across iterators without per-iterator mutable state on the model
/// instance. Drafters that need their own per-stream state additionally
/// conform to ``StatefulMTPDrafterModel``.
public protocol MTPDrafterModel: BaseLanguageModel {
    /// Largest total verification block the drafter can produce efficiently.
    /// `nil` means the caller may choose any block size.
    var maximumBlockSize: Int? { get }

    /// Whether this drafter consumes target-emitted shared K/V. Qwen owns a
    /// private full-attention cache and therefore only needs target hidden
    /// states; Gemma-style assistants consume the shared target K/V directly.
    var requiresSharedTargetKV: Bool { get }

    /// Whether the drafter must be prefilled over the shifted prompt before
    /// its first proposal. Qwen's private decoder cache requires this.
    var requiresPromptPrefill: Bool { get }

    /// Whether this drafter is currently supported only for greedy decoding.
    /// This keeps stochastic generation on the ordinary target path until a
    /// probability-ratio acceptance sampler is available.
    var requiresGreedySampling: Bool { get }

    /// K-step drafting from a constant position.
    ///
    /// Returns the proposed tokens as a `[B, blockSize - 1]` MLXArray. The
    /// drafter holds no transient round-state on the model instance — every
    /// per-round input is threaded as a method argument.
    ///
    /// - Parameters:
    ///   - target: The main language model this drafter is speculating for.
    ///     Used to look up target-derived constants (input embeddings, embed
    ///     scale, etc.) inline per round; conformers must not retain or
    ///     mutate references derived from `target`.
    ///   - lastToken: Bonus token from the target's last verify pass, shape `[B]`.
    ///   - lastHidden: Target's last hidden state, shape `[B, 1, backbone_hidden_size]`.
    ///   - sharedKV: Dict keyed by `layer_type` (`"full_attention"` /
    ///     `"sliding_attention"`) mapping to `(keys, values)` `MLXArray`s for
    ///     the last layer of that layer-type in the target.
    ///   - positionDeltas: Optional target-emitted position delta state for
    ///     multimodal RoPE models whose continuation positions are not
    ///     derivable from cache length alone.
    ///   - queryOffset: Constant absolute position for the round (the
    ///     position the bonus token sits at in the target's KV cache).
    ///     Passed as a Swift `Int` rather than an `MLXArray` to avoid the
    ///     `.item()` round-trip that would otherwise stall the GPU per
    ///     speculation round.
    ///   - blockSize: Total tokens in the round (the drafter returns
    ///     `blockSize - 1`; the bonus token is implicit).
    ///   - sampler: `LogitSampler` to apply to each step's logits.
    /// - Returns: `[B, blockSize - 1]` token array.
    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        sampler: any LogitSampler
    ) -> MLXArray
}

extension MTPDrafterModel {
    public var maximumBlockSize: Int? { nil }
    public var requiresSharedTargetKV: Bool { true }
    public var requiresPromptPrefill: Bool { false }
    public var requiresGreedySampling: Bool { false }
}

/// Target-side capability for rewinding an in-place speculative verify pass.
///
/// Rollback belongs to the target/cache implementation, not to the drafter.
/// The depth is explicit so a future multi-token drafter cannot accidentally
/// select an in-place path whose recurrent checkpoint is too shallow.
public protocol SpeculativeCacheRewindModel {
    var maximumNativeTargetCacheRewind: Int { get }
}

/// Per-stream state for MTP drafters that need their own transient storage.
///
/// The state is intentionally separate from ``LMOutput/State``: `LMOutput`
/// belongs to the target model's step output, while this value is owned by the
/// MTP iterator and passed only to the drafter.
public struct MTPDrafterState {
    public var cache: [KVCache]

    /// Absolute next position in a drafter-owned autoregressive cache.
    public var nextPosition: Int

    /// A proposal already computed while committing the previous verified
    /// round. Qwen uses this to avoid advancing its cache twice.
    public var seedToken: MLXArray?
    public var seedHidden: MLXArray?

    /// Number of tentative cache entries appended by the current proposal.
    public var proposalAppended: Int

    public init(
        cache: [KVCache],
        nextPosition: Int = 0,
        seedToken: MLXArray? = nil,
        seedHidden: MLXArray? = nil,
        proposalAppended: Int = 0
    ) {
        self.cache = cache
        self.nextPosition = nextPosition
        self.seedToken = seedToken
        self.seedHidden = seedHidden
        self.proposalAppended = proposalAppended
    }
}

/// Optional capability for MTP drafters that maintain per-stream state.
///
/// This keeps the base ``MTPDrafterModel`` surface minimal for stateless
/// drafters such as Gemma 4 assistants while giving Qwen-style MTP predictors
/// an iterator-owned state that can carry and trim their private KV cache. The
/// state must not be stored on the model instance because drafter models can be
/// shared across concurrent generation streams.
public protocol StatefulMTPDrafterModel: MTPDrafterModel {
    /// Create per-stream state owned and trimmed by the iterator.
    func makeState(configuration: KVCacheConfiguration?) -> MTPDrafterState

    /// Prefill drafter-owned state from the target's prompt hidden states.
    func prepareDrafterState(
        target: any LanguageModel,
        promptTokens: MLXArray,
        targetHidden: MLXArray,
        firstBonus: MLXArray,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    )

    /// Stateful variant of ``MTPDrafterModel/draftBlock(target:lastToken:lastHidden:sharedKV:positionDeltas:queryOffset:blockSize:sampler:)``.
    func draftBlock(
        target: any LanguageModel,
        lastToken: MLXArray,
        lastHidden: MLXArray,
        sharedKV: [String: (MLXArray, MLXArray)],
        positionDeltas: MLXArray?,
        queryOffset: Int,
        blockSize: Int,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    ) -> MLXArray

    /// Reconcile tentative proposal writes with the sequence accepted by the
    /// target, then seed the next proposal if the architecture supports it.
    func commitDrafterState(
        target: any LanguageModel,
        targetHidden: MLXArray,
        draftTokens: MLXArray,
        acceptedCount: Int,
        finalToken: MLXArray,
        positionDeltas: MLXArray?,
        state: inout MTPDrafterState,
        sampler: any LogitSampler
    )
}

extension StatefulMTPDrafterModel {
    public func prepareDrafterState(
        target _: any LanguageModel,
        promptTokens _: MLXArray,
        targetHidden _: MLXArray,
        firstBonus _: MLXArray,
        positionDeltas _: MLXArray?,
        state _: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {}

    public func commitDrafterState(
        target _: any LanguageModel,
        targetHidden _: MLXArray,
        draftTokens: MLXArray,
        acceptedCount: Int,
        finalToken _: MLXArray,
        positionDeltas _: MLXArray?,
        state: inout MTPDrafterState,
        sampler _: any LogitSampler
    ) {
        let rejected = draftTokens.dim(-1) - acceptedCount
        if rejected > 0 {
            trimPromptCache(state.cache, numTokens: rejected)
        }
    }
}
