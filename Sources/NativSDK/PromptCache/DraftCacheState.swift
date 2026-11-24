import MLX

/// A paired boundary: both decoders (including pending inputs) represent `tokens`.
/// Parallel drafters retain unprojected features; MTP retains one target hidden
/// position while its shifted draft cache lags the stable prefix by one.
struct DraftCacheCheckpoint {
    let tokens: [Int]
    let targetCaches: [KVCache]
    let draftCaches: [KVCache]
    let targetState: LMOutput.State?
    let draftState: LMOutput.State?
    let features: MLXArray?
    let draftPendingTokens: MLXArray?

    static func snapshot(
        tokens: [Int], targetCaches: [KVCache], draftCaches: [KVCache],
        targetState: LMOutput.State? = nil, draftState: LMOutput.State? = nil,
        features: MLXArray? = nil, draftPendingTokens: MLXArray? = nil
    ) throws -> Self {
        let caches = targetCaches + draftCaches
        guard caches.allSatisfy(PromptCacheOps.isCheckpointSafe) else {
            throw ModelExecutionError.invalidOptions(
                "draft session requires snapshot-capable caches")
        }
        let draftLag = (features?.dim(1) ?? 0) + (draftPendingTokens?.size ?? 0)
        guard
            KVCacheTree.leaves(in: targetCaches).allSatisfy({
                !cacheCapabilities($0.cache).contains(.attention) || $0.cache.offset == tokens.count
            }),
            KVCacheTree.leaves(in: draftCaches).allSatisfy({
                !cacheCapabilities($0.cache).contains(.attention)
                    || $0.cache.offset + draftLag == tokens.count
            })
        else {
            throw ModelExecutionError.invalidOptions("draft session cache boundary is inconsistent")
        }
        let target = targetCaches.map { $0.copy() }
        let draft = draftCaches.map { $0.copy() }
        eval(
            (target + draft).flatMap { $0.innerState() } + (features.map { [$0] } ?? [])
                + (draftPendingTokens.map { [$0] } ?? []))
        return Self(
            tokens: tokens, targetCaches: target, draftCaches: draft,
            targetState: targetState, draftState: draftState, features: features,
            draftPendingTokens: draftPendingTokens)
    }
}

/// Protected by the draft model's admission lock, including worker access.
final class DraftCacheVault: @unchecked Sendable {
    var latest: DraftCacheCheckpoint?
    var checkpoints: [DraftCacheCheckpoint] = []
    func reset() {
        latest = nil
        checkpoints = []
    }

    func best(for prompt: [Int]) -> DraftCacheCheckpoint? {
        (checkpoints + (latest.map { [$0] } ?? [])).filter {
            !$0.tokens.isEmpty && $0.tokens.count < prompt.count
                && $0.tokens.elementsEqual(prompt.prefix($0.tokens.count))
        }.max { $0.tokens.count < $1.tokens.count }
    }

    func retain(_ checkpoint: DraftCacheCheckpoint, maxCount: Int, protecting tokens: [Int]? = nil)
    {
        guard maxCount > 0 else { return }
        checkpoints.removeAll { $0.tokens == checkpoint.tokens }
        checkpoints.append(checkpoint)
        while checkpoints.count > maxCount {
            // Keep the current prompt boundary even after a long reasoning span:
            // its generated tokens may disappear when the template is rebuilt.
            let index = checkpoints.firstIndex { $0.tokens != tokens } ?? 0
            checkpoints.remove(at: index)
        }
    }
}
