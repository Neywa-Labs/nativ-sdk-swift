import Foundation
import MLX

/// Everything one generation owns: the per-layer buffers and the cross-layer handoff.
///
/// Layer state lives here rather than on the modules so two generations can run against
/// one model without overwriting each other, and so the whole thing can be snapshotted
/// and restored. Layers run in order and every index source writes its handoff before
/// its consumers read it.
///
/// One object serves the whole stack, which is what the handoff needs; the model hands
/// it to every layer rather than holding one cache per layer.
public final class DeepSeekV41Cache: KVCache, CacheCapabilityProviding {
    public var capabilities: CacheCapabilities { [.attention, .recurrent, .snapshot] }
    /// The five per-layer slots, in the order `state` serializes them.
    static let slotCount = 5
    /// Buffers grow a block at a time, as the shared caches do.
    public var step = 256

    public var offset = 0
    public var maxSize: Int? { nil }
    public var ropeOffset: RoPEOffset { .scalar(offset) }
    /// The compressor's partial groups cannot be rewound, so neither can this.
    public var isTrimmable: Bool { false }

    public private(set) var layers: Int

    var window: [MLXArray?]
    var compress: [MLXArray?]
    var keys: [MLXArray?]
    private var compressBuffers: [MLXArray?]
    private var keyBuffers: [MLXArray?]
    public var compressorState: DeepSeekV41CompressorState
    var engram: MLXArray?

    var indexKeys: MLXArray?
    var candidates: MLXArray?
    var compressKV: MLXArray?
    var topKIndices: MLXArray?

    public init(layers: Int = 0) {
        self.layers = layers
        window = Array(repeating: nil, count: layers)
        compress = Array(repeating: nil, count: layers)
        keys = Array(repeating: nil, count: layers)
        compressBuffers = Array(repeating: nil, count: layers)
        keyBuffers = Array(repeating: nil, count: layers)
        compressorState = DeepSeekV41CompressorState(layers: layers)
    }

    public func update(keys _: MLXArray, values _: MLXArray) -> (MLXArray, MLXArray) {
        fatalError("DeepSeekV41Cache does not store standard KV pairs.")
    }

    public func advance(_ width: Int) {
        offset += width
    }

    public func resetHandoff() {
        indexKeys = nil
        candidates = nil
        compressKV = nil
        topKIndices = nil
    }

    /// Retain block capacity while publishing only the populated prefix.
    private func append(
        _ arrays: inout [MLXArray?], _ buffers: inout [MLXArray?], layer: Int,
        values: MLXArray
    ) -> MLXArray {
        let previous = arrays[layer]
        let start = previous?.dim(1) ?? 0
        let end = start + values.dim(1)
        var buffer = buffers[layer] ?? previous
        if buffer == nil || end > buffer!.dim(1) {
            let capacity = ((end + step - 1) / step) * step
            let shape = [values.dim(0), capacity] + Array(values.shape.dropFirst(2))
            let dtype = buffer?.dtype ?? values.dtype
            let fresh = MLXArray.zeros(shape, dtype: dtype)
            if let previous {
                fresh[0..., ..<start] = previous
            }
            buffer = fresh
        }
        buffer![0..., start ..< end] = values
        buffers[layer] = buffer
        arrays[layer] = buffer![0..., ..<end]
        return arrays[layer]!
    }

    @discardableResult
    public func appendCompressed(layer: Int, values: MLXArray) -> MLXArray {
        append(&compress, &compressBuffers, layer: layer, values: values)
    }

    @discardableResult
    public func appendIndexKeys(layer: Int, values: MLXArray) -> MLXArray {
        append(&keys, &keyBuffers, layer: layer, values: values)
    }

    private var slots: [[MLXArray?]] {
        [window, compress, keys, compressorState.kvState, compressorState.scoreState]
    }

    public func innerState() -> [MLXArray] {
        state
    }

    /// Every buffer, in a fixed order, with empties standing in for unset slots.
    ///
    /// The generator evaluates this between prefill chunks to keep a long prompt from
    /// becoming one command buffer, so it cannot contain a gap.
    public var state: [MLXArray] {
        get {
            var flat = slots.flatMap { $0 }
            flat.append(engram)
            flat.append(contentsOf: [indexKeys, candidates, compressKV, topKIndices])
            return flat.map { $0 ?? MLXArray.zeros([0]) }
        }
        set {
            let n = (newValue.count - 1) / DeepSeekV41Cache.slotCount
            layers = n
            let restored = (0 ..< DeepSeekV41Cache.slotCount).map { slot in
                newValue[(slot * n) ..< ((slot + 1) * n)].map { $0.size == 0 ? nil : $0 }
            }
            window = restored[0]
            compress = restored[1]
            keys = restored[2]
            compressorState = DeepSeekV41CompressorState(layers: n)
            compressorState.kvState = restored[3]
            compressorState.scoreState = restored[4]
            compressBuffers = compress
            keyBuffers = keys
            let engramState = newValue[DeepSeekV41Cache.slotCount * n]
            engram = engramState.size == 0 ? nil : engramState
            resetHandoff()
            let tail = newValue[(DeepSeekV41Cache.slotCount * n + 1)...]
            let handoff = tail.map { $0.size == 0 ? nil : $0 }
            if handoff.count > 0 { indexKeys = handoff[0] }
            if handoff.count > 1 { candidates = handoff[1] }
            if handoff.count > 2 { compressKV = handoff[2] }
            if handoff.count > 3 { topKIndices = handoff[3] }
        }
    }

    public var metaState: [String] {
        get { [String(offset)] }
        set { offset = Int(newValue.first ?? "0") ?? 0 }
    }

    public func isTrimmable(after _: Int) -> Bool { false }

    @discardableResult
    public func trim(_: Int) -> Int { 0 }

    /// The attention builds its own window mask from the positions it retained, so the
    /// cache has no mask to offer.
    public func makeMask(n _: Int, windowSize _: Int?, returnArray _: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        .none
    }

    /// The block capacity behind a growing slot, which is larger than what the slot
    /// publishes. Exposed so the block growth itself can be checked.
    func compressedCapacity(layer: Int) -> Int? {
        compressBuffers[layer]?.dim(1)
    }

    func indexKeyCapacity(layer: Int) -> Int? {
        keyBuffers[layer]?.dim(1)
    }

    /// A copy carries the block slack across, so it can keep appending without
    /// reallocating, and detaches every buffer while doing so.
    ///
    /// The detaching is belt-and-braces rather than load-bearing: an append does mutate
    /// its buffer in place, but what a slot publishes is a slice, and a slice does not
    /// see a later write to the array it came from. Sharing the buffers is still wrong in
    /// principle and costs nothing to avoid.
    public func copy() -> any KVCache {
        let copy = DeepSeekV41Cache(layers: layers)
        copy.offset = offset
        copy.step = step
        copy.window = window.map { $0.map { $0[.ellipsis] } }
        copy.compressBuffers = compressBuffers.map { $0.map { $0[.ellipsis] } }
        copy.keyBuffers = keyBuffers.map { $0.map { $0[.ellipsis] } }
        copy.compress = zip(compress, copy.compressBuffers).map { published, buffer in
            guard let published else { return nil }
            return buffer.map { $0[0..., ..<published.dim(1)] } ?? published[.ellipsis]
        }
        copy.keys = zip(keys, copy.keyBuffers).map { published, buffer in
            guard let published else { return nil }
            return buffer.map { $0[0..., ..<published.dim(1)] } ?? published[.ellipsis]
        }
        copy.compressorState.kvState = compressorState.kvState.map { $0.map { $0[.ellipsis] } }
        copy.compressorState.scoreState = compressorState.scoreState.map {
            $0.map { $0[.ellipsis] }
        }
        copy.engram = engram.map { $0[.ellipsis] }
        copy.indexKeys = indexKeys.map { $0[.ellipsis] }
        copy.candidates = candidates.map { $0[.ellipsis] }
        copy.compressKV = compressKV.map { $0[.ellipsis] }
        copy.topKIndices = topKIndices.map { $0[.ellipsis] }
        return copy
    }
}
