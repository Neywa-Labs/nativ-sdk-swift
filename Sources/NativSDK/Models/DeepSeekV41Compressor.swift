import Foundation
import MLX
import MLXNN

/// The partial group each compression-source layer is still filling, held across the
/// prefill/decode split. The generation's cache owns one of these.
public final class DeepSeekV41CompressorState {
    public var kvState: [MLXArray?]
    public var scoreState: [MLXArray?]

    public init(layers: Int) {
        kvState = Array(repeating: nil, count: layers)
        scoreState = Array(repeating: nil, count: layers)
    }
}

extension DeepSeekV41Language {
    /// Pools `compressRatio` consecutive tokens into one KV latent with a learned softmax
    /// gate.
    ///
    /// Returns the latent before rope, or nil while a group is still filling up — so
    /// during decode it yields only every `compressRatio` steps, holding the partial
    /// group in state. Pre-rope is deliberate: the indexer needs the unrotated form, so
    /// attention rotates afterwards. Ratio 1 is a plain projection with no gate and no
    /// promotion to FP32.
    final class Compressor: Module {
        let compressRatio: Int
        let headDim: Int
        let layerIndex: Int

        @ModuleInfo(key: "norm") var norm: RMSNorm
        @ModuleInfo(key: "wkv") var wkv: Linear
        @ModuleInfo(key: "wgate") var wgate: Linear?

        init(_ config: DeepSeekV41Configuration, layerIndex: Int) {
            let ratio = config.compressRatios[layerIndex]
            precondition(ratio >= 1, "a compressor needs a ratio of at least one")
            self.compressRatio = ratio
            self.headDim = config.headDim
            self.layerIndex = layerIndex
            _norm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
            _wkv.wrappedValue = Linear(config.hiddenSize, config.headDim, bias: false)
            _wgate.wrappedValue =
                ratio > 1 ? Linear(config.hiddenSize, config.headDim, bias: false) : nil
            super.init()
        }

        func growState(batch: Int, state: DeepSeekV41CompressorState) {
            let index = layerIndex
            guard let existing = state.kvState[index] else {
                state.kvState[index] = MLXArray.zeros(
                    [batch, compressRatio, headDim], dtype: .float32)
                state.scoreState[index] = MLXArray.full(
                    [batch, compressRatio, headDim], values: MLXArray(-Float.infinity))
                return
            }
            guard existing.dim(0) < batch else { return }
            let extra = batch - existing.dim(0)
            state.kvState[index] = concatenated(
                [existing, MLXArray.zeros([extra, compressRatio, headDim], dtype: .float32)],
                axis: 0)
            state.scoreState[index] = concatenated(
                [
                    state.scoreState[index]!,
                    MLXArray.full(
                        [extra, compressRatio, headDim], values: MLXArray(-Float.infinity)),
                ], axis: 0)
        }

        /// Keep the tail of a prefill that did not divide evenly, leaving the rows of any
        /// wider batch already in the state untouched.
        func stowRemainder(_ state: MLXArray, batch: Int, values: MLXArray) -> MLXArray {
            let remainder = values.dim(1)
            let head = concatenated([values, state[0 ..< batch, remainder...]], axis: 1)
            if state.dim(0) > batch {
                return concatenated([head, state[batch...]], axis: 0)
            }
            return head
        }

        func writeSlot(_ state: MLXArray, batch: Int, slot: Int, values: MLXArray)
            -> MLXArray
        {
            var parts: [MLXArray] = []
            if slot > 0 { parts.append(state[0 ..< batch, 0 ..< slot]) }
            parts.append(values)
            if slot + 1 < compressRatio { parts.append(state[0 ..< batch, (slot + 1)...]) }
            let head = parts.count > 1 ? concatenated(parts, axis: 1) : parts[0]
            if state.dim(0) > batch {
                return concatenated([head, state[batch...]], axis: 0)
            }
            return head
        }

        func callAsFunction(
            _ x: MLXArray, startPosition: Int, state: DeepSeekV41CompressorState
        ) -> MLXArray? {
            guard let wgate else { return norm(wkv(x)) }

            let index = layerIndex
            let (batch, length) = (x.dim(0), x.dim(1))
            let dtype = x.dtype
            let promoted = x.asType(.float32)
            var kv = wkv(promoted)
            var score = wgate(promoted)
            var shouldCompress: Bool

            if startPosition == 0 {
                shouldCompress = length >= compressRatio
                let remainder = length % compressRatio
                let cutoff = length - remainder
                if remainder > 0 {
                    growState(batch: batch, state: state)
                    state.kvState[index] = stowRemainder(
                        state.kvState[index]!, batch: batch, values: kv[0..., cutoff...])
                    state.scoreState[index] = stowRemainder(
                        state.scoreState[index]!, batch: batch, values: score[0..., cutoff...])
                }
                kv = kv[0..., ..<cutoff].reshaped(batch, -1, compressRatio, headDim)
                score = score[0..., ..<cutoff].reshaped(batch, -1, compressRatio, headDim)
                kv = (kv * softmax(score, axis: 2)).sum(axis: 2)
            } else {
                growState(batch: batch, state: state)
                var pooled: [MLXArray] = []
                for step in 0 ..< length {
                    let position = startPosition + step
                    let slot = position % compressRatio
                    state.kvState[index] = writeSlot(
                        state.kvState[index]!, batch: batch, slot: slot,
                        values: kv[0..., step ..< (step + 1), 0...])
                    state.scoreState[index] = writeSlot(
                        state.scoreState[index]!, batch: batch, slot: slot,
                        values: score[0..., step ..< (step + 1), 0...])
                    if (position + 1) % compressRatio == 0 {
                        pooled.append(
                            (state.kvState[index]![0 ..< batch]
                                * softmax(state.scoreState[index]![0 ..< batch], axis: 1))
                                .sum(axis: 1, keepDims: true))
                    }
                }
                shouldCompress = !pooled.isEmpty
                if shouldCompress {
                    kv = pooled.count > 1 ? concatenated(pooled, axis: 1) : pooled[0]
                }
            }
            guard shouldCompress else { return nil }
            return norm(kv.asType(dtype))
        }
    }
}
