import MLX
import XCTest

@testable import NativSDK

/// A cache type outside the whitelist: no `CacheCapabilityProviding` conformance.
private final class UnverifiedDummyCache: KVCache {
    var offset = 0
    var maxSize: Int? { nil }
    var state: [MLXArray] {
        get { [] }
        set {}
    }
    var metaState: [String] {
        get { [""] }
        set {}
    }
    var isTrimmable: Bool { false }

    func innerState() -> [MLXArray] { [] }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) { (keys, values) }

    @discardableResult
    func trim(_: Int) -> Int { 0 }

    func makeMask(
        n _: Int, windowSize _: Int?, returnArray _: Bool,
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        .none
    }

    func copy() -> any KVCache { self }
}

final class PromptCacheOpsTests: XCTestCase {
    private func feed(_ cache: KVCache, tokens: Int, value: Float = 1) {
        let keys = MLXArray.ones([1, 1, tokens, 4]) * value
        let values = MLXArray.ones([1, 1, tokens, 4]) * value
        _ = cache.update(keys: keys, values: values)
    }

    func testTrimAllSucceedsOnSimpleCaches() {
        let caches: [KVCache] = [KVCacheSimple(), KVCacheSimple()]
        for cache in caches {
            feed(cache, tokens: 5)
        }

        XCTAssertTrue(PromptCacheOps.trimAll(caches, by: 2))
        XCTAssertEqual(caches.map(\.offset), [3, 3])
    }

    func testTrimAllRefusesMixedTrimmability() {
        let simple = KVCacheSimple()
        feed(simple, tokens: 5)
        let mamba = MambaCache()

        XCTAssertFalse(PromptCacheOps.trimAll([simple, mamba], by: 2))
        XCTAssertEqual(simple.offset, 5, "a refused trim must leave every layer untouched")
    }

    func testTrimAllRefusesWhenRewindExceedsOffset() {
        let simple = KVCacheSimple()
        feed(simple, tokens: 2)

        XCTAssertFalse(PromptCacheOps.trimAll([simple], by: 3))
        XCTAssertEqual(simple.offset, 2)
    }

    func testTrimAllZeroIsNoop() {
        let simple = KVCacheSimple()
        feed(simple, tokens: 2)

        XCTAssertTrue(PromptCacheOps.trimAll([simple], by: 0))
        XCTAssertEqual(simple.offset, 2)
    }

    func testRotatingTrimmableOnlyBeforeWrap() {
        let rotating = RotatingKVCache(maxSize: 4)
        feed(rotating, tokens: 3)
        XCTAssertTrue(rotating.isTrimmable)
        XCTAssertTrue(PromptCacheOps.trimAll([rotating], by: 1))
        XCTAssertEqual(rotating.offset, 2)

        feed(rotating, tokens: 3)
        XCTAssertFalse(rotating.isTrimmable, "ring buffer past maxSize loses temporal order")
        XCTAssertFalse(PromptCacheOps.trimAll([rotating], by: 1))
    }

    func testSnapshotIsDurableAgainstLaterUpdates() {
        let cache = KVCacheSimple()
        feed(cache, tokens: 3, value: 1)

        let checkpoint = PromptCacheOps.snapshot(of: [cache], tokens: [1, 2, 3])
        feed(cache, tokens: 2, value: 9)
        XCTAssertEqual(cache.offset, 5)

        let restored = checkpoint.restore()
        XCTAssertEqual(restored.count, 1)
        XCTAssertEqual(restored[0].offset, 3)
        let keys = restored[0].state[0]
        XCTAssertEqual(keys.dim(2), 3)
        XCTAssertTrue(
            allClose(keys, MLXArray.ones(keys.shape)).item(Bool.self),
            "restored state must reflect snapshot-time contents, not later writes",
        )

        // A checkpoint stays restorable after one restored copy moves on.
        let restoredAgain = checkpoint.restore()
        feed(restored[0], tokens: 1, value: 5)
        XCTAssertEqual(restoredAgain[0].offset, 3)
    }

    func testCheckpointSelectionAndPruning() {
        let checkpoints = [
            PromptCacheCheckpoint(tokens: [1, 2], caches: []),
            PromptCacheCheckpoint(tokens: [1, 2, 3, 4], caches: []),
            PromptCacheCheckpoint(tokens: [9, 9], caches: []),
        ]

        XCTAssertEqual(
            PromptCacheOps.bestCheckpoint(in: checkpoints, for: [1, 2, 3, 4, 5])?.index, 1)
        XCTAssertEqual(
            PromptCacheOps.bestCheckpoint(in: checkpoints, for: [1, 2, 3, 4])?.index, 0,
            "a checkpoint covering the whole prompt is unusable: nothing would be left to feed",
        )
        XCTAssertNil(PromptCacheOps.bestCheckpoint(in: checkpoints, for: [7]))

        var pruned = checkpoints
        PromptCacheOps.prune(&pruned, maxCount: 2)
        XCTAssertEqual(pruned.map(\.tokens), [[1, 2, 3, 4], [9, 9]], "prune keeps the newest")
    }

    func testCapabilityWhitelist() {
        XCTAssertTrue(PromptCacheOps.isCheckpointSafe(KVCacheSimple()))
        XCTAssertTrue(PromptCacheOps.isCheckpointSafe(RotatingKVCache(maxSize: 8)))
        XCTAssertTrue(PromptCacheOps.isCheckpointSafe(QuantizedKVCache()))
        XCTAssertTrue(PromptCacheOps.isCheckpointSafe(ArraysCache(size: 1)))
        XCTAssertTrue(PromptCacheOps.isCheckpointSafe(MambaCache()))
        XCTAssertTrue(
            PromptCacheOps.isCheckpointSafe(CacheList(KVCacheSimple(), MambaCache())))
        XCTAssertFalse(
            PromptCacheOps.isCheckpointSafe(ChunkedKVCache()),
            "ChunkedKVCache subclasses KVCacheSimple but its copy() is unverified — exact type match only",
        )
        XCTAssertFalse(PromptCacheOps.isCheckpointSafe(UnverifiedDummyCache()))
    }

    func testShouldCheckpoint() {
        XCTAssertFalse(
            PromptCacheOps.shouldCheckpoint([KVCacheSimple(), KVCacheSimple()]),
            "forever-trimmable caches never need checkpoints",
        )
        XCTAssertFalse(
            PromptCacheOps.shouldCheckpoint([KVCacheSimple(), QuantizedKVCache()]))
        XCTAssertTrue(
            PromptCacheOps.shouldCheckpoint([KVCacheSimple(), RotatingKVCache(maxSize: 8)]))
        XCTAssertTrue(PromptCacheOps.shouldCheckpoint([MambaCache(), KVCacheSimple()]))
        XCTAssertFalse(
            PromptCacheOps.shouldCheckpoint([KVCacheSimple(), UnverifiedDummyCache()]),
            "an unsafe layer disables checkpointing entirely",
        )
    }
}
