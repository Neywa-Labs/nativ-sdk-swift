import MLX

/// Records only new K/V rows. Rejected writes restore the old ring and append
/// the retained rows, without replaying the target network or losing its window.
private final class AttentionDraftRecordingCache: BaseKVCache {
    let base: RotatingKVCache
    let snapshot: RotatingKVCache
    var writes: [(MLXArray, MLXArray)] = []
    init(_ base: RotatingKVCache) {
        self.base = base
        snapshot = base.copy() as! RotatingKVCache
        super.init()
        offset = base.offset
    }
    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        writes.append((keys, values))
        let result = base.update(keys: keys, values: values)
        offset = base.offset
        return result
    }
    override func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        base.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }
    func commit(_ count: Int) {
        guard count != base.offset - snapshot.offset else { return }
        base.restoreDraftSnapshot(snapshot)
        var remaining = count
        for (keys, values) in writes where remaining > 0 {
            let n = min(remaining, keys.dim(2))
            _ = base.update(
                keys: keys[0..., 0..., ..<n, 0...], values: values[0..., 0..., ..<n, 0...])
            remaining -= n
        }
    }
}

final class AttentionDraftTransaction {
    let caches: [KVCache]
    private let attention: [(KVCacheSimple, Int, Bool)]
    private let rotating: [AttentionDraftRecordingCache]
    private let length: Int
    private var active = true
    init(_ caches: [KVCache], length: Int) throws {
        guard caches.allSatisfy({ $0 is KVCacheSimple || $0 is RotatingKVCache }) else {
            throw ModelExecutionError.invalidOptions(
                "Drafting requires native uncompressed caches")
        }
        self.length = length
        attention = caches.compactMap {
            ($0 as? KVCacheSimple).map { ($0, $0.offset, $0.keys == nil) }
        }
        self.caches = caches.map { cache in
            if let ring = cache as? RotatingKVCache { return AttentionDraftRecordingCache(ring) }
            return cache
        }
        rotating = self.caches.compactMap { $0 as? AttentionDraftRecordingCache }
    }
    func commit(_ count: Int) throws {
        guard active else { return }
        guard (0 ... length).contains(count) else {
            throw ModelExecutionError.invalidOptions("invalid retained draft prefix")
        }
        for (cache, start, empty) in attention {
            _ = cache.trim(cache.offset - start - count)
            if count == 0 && empty {
                cache.keys = nil
                cache.values = nil
            }
        }
        for cache in rotating { cache.commit(count) }
        active = false
    }
    deinit { if active { try? commit(0) } }
}
