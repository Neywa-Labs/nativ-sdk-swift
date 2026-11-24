import MLX

/// A restorable snapshot of the full per-layer cache at a known token boundary.
///
/// `tokens` is self-contained (not an offset into live history) so a checkpoint stays
/// valid across divergences and resets, and is independent of the live continuation. Persistence is unsupported.
struct PromptCacheCheckpoint {
    let tokens: [Int]
    let caches: [KVCache]
    var state: LMOutput.State? = nil

    func restoreState(model: any LanguageModel) throws -> LMOutput.State? {
        try copyModelContinuationState(state, model: model)
    }

    /// Copies on the way out so the checkpoint remains restorable repeatedly.
    func restore() -> [KVCache] {
        caches.map { $0.copy() }
    }
}

/// Mutable reuse state accessed under the model request lock.
///
/// Not thread-safe by itself: all mutation must happen inside `NativModel.withModel` closures.
final class PromptCacheVault: @unchecked Sendable {
    var model: (any LanguageModel)?
    var state: LMOutput.State?
    var history: [Int] = []
    var caches: [KVCache]?
    var checkpoints: [PromptCacheCheckpoint] = []

    func reset() {
        model = nil
        state = nil
        history = []
        caches = nil
        checkpoints = []
    }
}

enum PromptCacheOps {
    static func isCheckpointSafe(_ cache: KVCache) -> Bool {
        cacheCapabilities(cache).contains(.snapshot)
    }

    static func shouldCheckpoint(_ caches: [KVCache]) -> Bool {
        !caches.isEmpty && caches.allSatisfy(isCheckpointSafe)
            && !caches.allSatisfy { cacheCapabilities($0).contains(.fullRewind) }
    }

    /// Rewinds every layer by exactly `n` tokens, or refuses without touching anything.
    ///
    /// Unlike upstream `trimPromptCache` (silent all-or-nothing no-op), this prechecks
    /// per layer that the trim will succeed exactly, so a partial rewind can never leave
    /// layers inconsistent with each other.
    @discardableResult
    static func trimAll(_ caches: [KVCache], by n: Int) -> Bool {
        guard n > 0 else {
            return true
        }
        guard
            caches.allSatisfy({
                !cacheCapabilities($0).intersection([.fullRewind, .prefixRewind]).isEmpty
                    && $0.isTrimmable && $0.offset >= n
            })
        else {
            return false
        }
        for cache in caches {
            let trimmed = cache.trim(n)
            assert(trimmed == n, "precheck guaranteed a full trim, got \(trimmed) of \(n)")
        }
        return true
    }

    /// Copies every layer and evaluates the copies so the snapshot owns its buffers:
    /// `copy()` returns lazy COW slices that would otherwise pin the live graph and could
    /// observe later in-place writes.
    static func snapshot(of caches: [KVCache], tokens: [Int]) -> PromptCacheCheckpoint {
        precondition(caches.allSatisfy(isCheckpointSafe), "unverified snapshot capability")
        let copies = caches.map { $0.copy() }
        eval(copies.flatMap { $0.innerState() })
        return PromptCacheCheckpoint(tokens: tokens, caches: copies)
    }

    /// The longest checkpoint whose tokens are a strict prefix of `prompt` (strict:
    /// at least one token must remain to feed).
    static func bestCheckpoint(
        in checkpoints: [PromptCacheCheckpoint], for prompt: [Int],
    ) -> (index: Int, checkpoint: PromptCacheCheckpoint)? {
        var best: (index: Int, checkpoint: PromptCacheCheckpoint)?
        for (index, checkpoint) in checkpoints.enumerated() {
            let count = checkpoint.tokens.count
            let bestCount = best?.checkpoint.tokens.count ?? 0
            guard count < prompt.count, count > bestCount,
                checkpoint.tokens.elementsEqual(prompt.prefix(count))
            else {
                continue
            }
            best = (index, checkpoint)
        }
        return best
    }

    /// Drops the oldest checkpoints beyond `maxCount`.
    static func prune(_ checkpoints: inout [PromptCacheCheckpoint], maxCount: Int) {
        if checkpoints.count > maxCount {
            checkpoints.removeFirst(checkpoints.count - maxCount)
        }
    }
}
