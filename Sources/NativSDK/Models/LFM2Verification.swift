import MLX
import MLXNN

/// Round-local convolution inputs let us commit any prefix without replaying layers.
struct LFM2VerificationTransaction {
    struct Convolution {
        let cache: MambaCache
        let previous: MLXArray?
        let leftPadding: MLXArray?
        let lengths: MLXArray?
        var inputs: MLXArray?
    }
    let attention: [(cache: KVCacheSimple, offset: Int, wasEmpty: Bool)]
    var convolutions: [Int: Convolution]
    let stateSize: Int

    init(cache: [KVCache], stateSize: Int) {
        self.stateSize = stateSize
        attention = cache.compactMap {
            ($0 as? KVCacheSimple).map { ($0, $0.offset, $0.keys == nil) }
        }
        convolutions = Dictionary(
            uniqueKeysWithValues: cache.enumerated().compactMap { index, value in
                (value as? MambaCache).map {
                    (
                        index,
                        Convolution(
                            cache: $0, previous: $0[0], leftPadding: $0.leftPadding,
                            lengths: $0.lengths, inputs: nil)
                    )
                }
            })
    }

    func commit(retaining count: Int) {
        for (cache, start, wasEmpty) in attention {
            _ = cache.trim(max(0, cache.offset - start - count))
            if count == 0 && wasEmpty {
                cache.keys = nil
                cache.values = nil
            }
        }
        for value in convolutions.values {
            if count == 0 {
                value.cache[0] = value.previous
            } else if let inputs = value.inputs {
                let previous =
                    value.previous
                    ?? MLXArray.zeros(
                        [inputs.dim(0), stateSize, inputs.dim(-1)], dtype: inputs.dtype)
                let combined = concatenated([previous, inputs[0..., ..<count, 0...]], axis: 1)
                value.cache[0] = contiguous(combined[0..., (combined.dim(1) - stateSize)..., 0...])
            }
            value.cache.leftPadding = value.leftPadding.map { $0 - count }
            value.cache.lengths = value.lengths.map { $0 - count }
        }
    }
}

struct LFM2VerificationOutput {
    let logits: MLXArray
    let features: MLXArray
    let transaction: LFM2VerificationTransaction?
}

extension LFM2Model {
    /// Prefill uses the ordinary computation. Verification keeps singleton GEMV and
    /// attention numerics while advancing the block's convolution state once.
    func dflashForward(
        _ tokens: MLXArray, cache: [KVCache], capture: [Int], verify: Bool,
        inputEmbeddings: MLXArray? = nil, recordTransaction: Bool = false
    ) throws -> LFM2VerificationOutput {
        var transaction =
            verify || recordTransaction
            ? LFM2VerificationTransaction(cache: cache, stateSize: configuration.convLCache - 1)
            : nil
        do {
            var hidden = inputEmbeddings ?? model.embedTokens(tokens)
            var features: [MLXArray] = []
            let selected = Set(capture)
            let mask = createAttentionMask(
                h: hidden, cache: cache[configuration.fullAttnIdxs.first ?? 0])
            for (index, layer) in model.layers.enumerated() {
                try Task.checkCancellation()
                if !verify {
                    hidden = layer(
                        hidden, mask: mask, cache: cache[index],
                        recordConvInputs: recordTransaction
                            ? { inputs in
                                transaction?.convolutions[index]?.inputs = inputs
                            } : nil)
                } else {
                    let normed = layer.operatorNorm(hidden)
                    let residual: MLXArray
                    if let attention = layer.attention {
                        residual = attention.exactVerification(normed, cache: cache[index])
                    } else {
                        let conv = layer.conv!
                        let projected = exactSpeculativeLinear(conv.inProj, normed).split(
                            parts: 3, axis: -1)
                        let gated = projected[0] * projected[2]
                        transaction!.convolutions[index]!.inputs = gated
                        let c = cache[index] as! MambaCache
                        let previous =
                            c[0]
                            ?? MLXArray.zeros(
                                [hidden.dim(0), conv.lCache - 1, configuration.hiddenSize],
                                dtype: gated.dtype)
                        let padded = concatenated([previous, gated], axis: 1)
                        c[0] = contiguous(
                            padded[0..., (padded.dim(1) - (conv.lCache - 1))..., 0...])
                        c.advance(tokens.dim(1))
                        residual = exactSpeculativeLinear(
                            conv.outProj, projected[1] * conv.conv(padded))
                    }
                    hidden = hidden + residual
                    let ff = layer.feedForward
                    let normedFF = layer.ffnNorm(hidden)
                    hidden =
                        hidden
                        + exactSpeculativeLinear(
                            ff.w2,
                            fusedSwiGLU(
                                exactSpeculativeLinear(ff.w1, normedFF),
                                exactSpeculativeLinear(ff.w3, normedFF)))
                }
                if selected.contains(index) { features.append(hidden) }
            }
            hidden = model.embeddingNorm(hidden)
            let logits: MLXArray
            if verify,
                let exact = exactSpeculativeProjection(
                    weight: model.embedTokens.weight, input: hidden)
            {
                logits = exact
            } else if verify && hidden.dim(1) > 1 {
                logits = concatenated(
                    (0 ..< hidden.dim(1)).map {
                        model.embedTokens.asLinear(hidden[0..., $0 ..< ($0 + 1), 0...])
                    }, axis: 1)
            } else {
                logits = model.embedTokens.asLinear(hidden)
            }
            return LFM2VerificationOutput(
                logits: logits, features: concatenated(features, axis: -1), transaction: transaction
            )
        } catch {
            transaction?.commit(retaining: 0)
            throw error
        }
    }
}

extension LFM2Attention {
    fileprivate func exactVerification(_ x: MLXArray, cache: KVCache) -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)
        var queries = qLayerNorm(
            exactSpeculativeLinear(qProj, x)
                .reshaped(batch, length, args.attentionHeads, headDim)
        ).transposed(0, 2, 1, 3)
        var keys = kLayerNorm(
            exactSpeculativeLinear(kProj, x)
                .reshaped(batch, length, args.kvHeads, headDim)
        ).transposed(0, 2, 1, 3)
        let values = exactSpeculativeLinear(vProj, x)
            .reshaped(batch, length, args.kvHeads, headDim).transposed(0, 2, 1, 3)
        queries = applyRotaryPosition(rope, to: queries, offset: cache.ropeOffset)
        keys = applyRotaryPosition(rope, to: keys, offset: cache.ropeOffset)
        let (allKeys, allValues) = cache.update(keys: keys, values: values)
        let prefix = allKeys.dim(2) - length
        let output = concatenated(
            (0 ..< length).map { index in
                MLXFast.scaledDotProductAttention(
                    queries: queries[0..., 0..., index ..< (index + 1), 0...],
                    keys: allKeys[0..., 0..., ..<(prefix + index + 1), 0...],
                    values: allValues[0..., 0..., ..<(prefix + index + 1), 0...],
                    scale: scale, mask: .none)
            }, axis: 2
        ).transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        return exactSpeculativeLinear(outProj, output)
    }
}
