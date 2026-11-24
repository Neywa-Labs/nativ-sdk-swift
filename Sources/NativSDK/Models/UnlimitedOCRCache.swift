import Foundation
import MLX

/// Retains all prompt keys and only the most recent generated keys.
final class UnlimitedOCRReferenceCache: KVCacheSimple {
    let window: Int
    var promptLength: Int?

    init(window: Int) {
        self.window = window
        super.init()
    }

    override var capabilities: CacheCapabilities { [.attention] }
    override var isTrimmable: Bool { false }
    override func trim(_ n: Int) -> Int { 0 }

    override func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        guard let promptLength, offset >= promptLength + window else {
            return super.update(keys: keys, values: values)
        }
        precondition(keys.dim(2) == 1)
        let slot = promptLength + (offset - promptLength) % window
        self.keys?[.ellipsis, slot ..< (slot + 1), 0...] = keys
        self.values?[.ellipsis, slot ..< (slot + 1), 0...] = values
        offset += 1
        let end = promptLength + window
        return (self.keys![.ellipsis, ..<end, 0...], self.values![.ellipsis, ..<end, 0...])
    }

    override var state: [MLXArray] {
        get {
            guard let keys, let values else { return [] }
            let end = min(offset, (promptLength ?? offset) + window)
            return [keys[.ellipsis, ..<end, 0...], values[.ellipsis, ..<end, 0...]]
        }
        set { super.state = newValue }
    }

    override func copy() -> any KVCache {
        let result = UnlimitedOCRReferenceCache(window: window)
        if !state.isEmpty { result.state = state }
        result.promptLength = promptLength
        result.offset = offset
        return result
    }
}

/// The first layer owns page and prompt state. Every layer owns its own summary keys.
final class UnlimitedOCRDASHCache: KVCacheSimple {
    var prompt: UnlimitedOCRPromptPages?
    var responsePosition: UnlimitedOCRResponsePosition?
    private var pending: MLXArray?
    private var summaries: MLXArray?
    private(set) var summaryCount = 0

    override var capabilities: CacheCapabilities { [.attention] }
    override var isTrimmable: Bool { false }
    override func trim(_ n: Int) -> Int { 0 }

    func startPrompt(_ ids: [Int32], parameters: UnlimitedOCRDASHParameters) {
        let prompt = UnlimitedOCRPromptPages(ids: ids, parameters: parameters)
        self.prompt = prompt
        responsePosition = UnlimitedOCRResponsePosition(prompt)
    }

    override func innerState() -> [MLXArray] {
        super.innerState() + [pending, summaries].compactMap { $0 }
            + (responsePosition?.arrays ?? [])
    }

    override func copy() -> any KVCache {
        let result = UnlimitedOCRDASHCache()
        if !state.isEmpty { result.state = state.map { $0[.ellipsis] } }
        result.prompt = prompt
        result.responsePosition = responsePosition
        result.pending = pending
        result.summaries = summaries?[.ellipsis]
        result.summaryCount = summaryCount
        return result
    }

    func appendRouterKeys(_ keys: MLXArray, head: MLXArray, blockSize: Int) {
        let input = pending.map { concatenated([$0, keys], axis: 0) } ?? keys
        let fullBlocks = input.dim(0) / blockSize
        let complete = fullBlocks * blockSize
        pending = complete < input.dim(0) ? input[complete..., 0...] : nil
        guard fullBlocks > 0 else { return }
        let dim = input.dim(-1)
        let blocks = input[..<complete, 0...].reshaped(fullBlocks, blockSize, dim)
        let scores = clip(
            (blocks * head.asType(.float32)).sum(axis: -1) * pow(Float(dim), -0.5),
            min: -1e4, max: 1e4)
        let summary = (softmax(scores, axis: -1, precise: true)[.ellipsis, .newAxis] * blocks)
            .sum(axis: 1)
        let end = summaryCount + fullBlocks
        _ = summaryBuffer(capacity: UnlimitedOCRDASH.blockCapacity(end), dimension: dim)
        summaries?[summaryCount ..< end, 0...] = summary
        summaryCount = end
    }

    func summaryBuffer(capacity: Int, dimension: Int) -> MLXArray {
        if summaries == nil || summaries!.dim(0) < capacity {
            let buffer = MLXArray.zeros([capacity, dimension], dtype: .float32)
            if let summaries {
                buffer[..<summaryCount, 0...] = summaries[..<summaryCount, 0...]
            }
            summaries = buffer
        }
        return summaries![..<capacity, 0...]
    }
}
