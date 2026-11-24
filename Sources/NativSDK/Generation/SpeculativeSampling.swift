import MLX

/// A request-local target RNG timeline. A speculative block prepares future keys
/// without advancing it; only accepted target draws are committed.
struct SpeculativeRandomCursor {
    struct Block {
        let keys: [MLXArray]
        let continuations: [MLXArray]
    }
    private var state: MLXArray

    init(seed: UInt64?) {
        let random = seed.map { MLXRandom.RandomState(seed: $0) } ?? MLXRandom.RandomState()
        state = random.innerState()[0]
    }

    func prepare(count: Int) -> Block {
        precondition(count > 0)
        var next = state
        var keys: [MLXArray] = []
        var continuations: [MLXArray] = []
        for _ in 0 ..< count {
            let (continuation, key) = MLXRandom.split(key: next)
            keys.append(key)
            continuations.append(continuation)
            next = continuation
        }
        return Block(keys: keys, continuations: continuations)
    }

    mutating func commit(_ block: Block, retaining count: Int) {
        precondition((0 ... block.keys.count).contains(count))
        guard count > 0 else { return }
        state = block.continuations[count - 1]
        // Fence retained state so its lazy split graph stays bounded per round.
        asyncEval(state)
    }
}

/// Singleton shapes preserve the ordinary sampler's random draw layout and
/// reduction numerics. Every row remains on-device until the acceptance read.
func sampleSpeculativeBlock(
    _ logits: MLXArray, sampler: any KeyedLogitSampler,
    keys: [MLXArray]
) -> MLXArray {
    precondition(logits.ndim == 2 && logits.dim(0) == keys.count)
    return concatenated(
        keys.indices.map {
            sampler.sample(logits: logits[$0 ..< ($0 + 1), 0...], key: keys[$0]).asType(.int32)
        }, axis: 0)
}

/// Returns the matching-prefix length and every target sample in one GPU array.
func speculativeSamplingDecision(proposals: MLXArray, predictions: MLXArray) -> MLXArray {
    let count = proposals.size
    let mismatch = concatenated([
        (proposals .!= predictions[..<count]).asType(.int32), MLXArray([Int32(1)]),
    ])
    let accepted = argMax(mismatch).asType(.int32).reshaped(1)
    return concatenated([accepted, predictions])
}
