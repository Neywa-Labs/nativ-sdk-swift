import MLX
import MLXNN

/// Target operations shared by parallel DFlash and Markov DSpark drafters.
/// Transactions own architecture-specific rollback; the iterator only commits a prefix.
protocol DFlashTarget: LanguageModel {
    var vocabularySize: Int { get }
    var draftHiddenSize: Int { get }
    var draftLayerCount: Int { get }
    var draftEmbedding: Embedding { get }
    var draftEmbeddingScale: Float { get }
    func draftReadout(_ hidden: MLXArray) -> MLXArray
    func draftForward(
        _ tokens: MLXArray, cache: [KVCache], capture: [Int], verify: Bool,
        inputEmbeddings: MLXArray?, recordTransaction: Bool,
        state: LMOutput.State?, positionIds: MLXArray?
    ) throws -> DFlashTargetOutput
}

extension DFlashTarget {
    var draftEmbeddingScale: Float { 1 }
}

struct DFlashTargetOutput {
    let logits: MLXArray
    let features: MLXArray
    let commit: ((Int) throws -> Void)?
    let state: LMOutput.State?
    init(
        logits: MLXArray, features: MLXArray, commit: ((Int) throws -> Void)?,
        state: LMOutput.State? = nil
    ) {
        self.logits = logits
        self.features = features
        self.commit = commit
        self.state = state
    }
}

extension LFM2Model: DFlashTarget {
    var draftHiddenSize: Int { configuration.hiddenSize }
    var draftLayerCount: Int { configuration.hiddenLayers }
    var draftEmbedding: Embedding { model.embedTokens }
    func draftReadout(_ hidden: MLXArray) -> MLXArray { model.embedTokens.asLinear(hidden) }
    func draftForward(
        _ tokens: MLXArray, cache: [KVCache], capture: [Int], verify: Bool,
        inputEmbeddings: MLXArray? = nil, recordTransaction: Bool = false,
        state: LMOutput.State? = nil, positionIds: MLXArray? = nil
    ) throws -> DFlashTargetOutput {
        let output = try dflashForward(
            tokens, cache: cache, capture: capture, verify: verify,
            inputEmbeddings: inputEmbeddings, recordTransaction: recordTransaction)
        return DFlashTargetOutput(
            logits: output.logits, features: output.features,
            commit: output.transaction.map { transaction in { transaction.commit(retaining: $0) } })
    }
}

extension Qwen3Model: DFlashTarget {
    var draftHiddenSize: Int { configuration.hiddenSize }
    var draftLayerCount: Int { configuration.hiddenLayers }
    var draftEmbedding: Embedding { model.embedTokens }
    func draftReadout(_ hidden: MLXArray) -> MLXArray {
        lmHead.map { $0(hidden) } ?? model.embedTokens.asLinear(hidden)
    }
    func draftForward(
        _ tokens: MLXArray, cache: [KVCache], capture: [Int], verify: Bool,
        inputEmbeddings: MLXArray? = nil, recordTransaction: Bool = false,
        state: LMOutput.State? = nil, positionIds: MLXArray? = nil
    ) throws -> DFlashTargetOutput {
        let starts = cache.map(\.offset)
        let commit: (Int) -> Void = { count in
            for (entry, start) in zip(cache, starts) {
                _ = entry.trim(entry.offset - start - count)
            }
        }
        do {
            var hidden = inputEmbeddings ?? model.embedTokens(tokens)
            var features: [MLXArray] = []
            let selected = Set(capture)
            let mask = createAttentionMask(h: hidden, cache: cache.first)
            for (index, layer) in model.layers.enumerated() {
                try Task.checkCancellation()
                if verify {
                    // Singleton projections and attention preserve ordinary decode
                    // numerics, including quantized targets and near-tie logits.
                    let attention = layer.attention
                    let x = layer.inputLayerNorm(hidden)
                    let batch = x.dim(0)
                    let length = x.dim(1)
                    var q = attention.qNorm(
                        exactSpeculativeLinear(attention.wq, x)
                            .reshaped(
                                batch, length, configuration.attentionHeads, configuration.headDim)
                    ).transposed(0, 2, 1, 3)
                    var k = attention.kNorm(
                        exactSpeculativeLinear(attention.wk, x)
                            .reshaped(batch, length, configuration.kvHeads, configuration.headDim)
                    ).transposed(0, 2, 1, 3)
                    let v = exactSpeculativeLinear(attention.wv, x)
                        .reshaped(batch, length, configuration.kvHeads, configuration.headDim)
                        .transposed(0, 2, 1, 3)
                    q = applyRotaryPosition(attention.rope, to: q, offset: cache[index].ropeOffset)
                    k = applyRotaryPosition(attention.rope, to: k, offset: cache[index].ropeOffset)
                    let (keys, values) = cache[index].update(keys: k, values: v)
                    let prefix = keys.dim(2) - length
                    let output = concatenated(
                        (0 ..< length).map { i in
                            MLXFast.scaledDotProductAttention(
                                queries: q[0..., 0..., i ..< (i + 1), 0...],
                                keys: keys[0..., 0..., ..<(prefix + i + 1), 0...],
                                values: values[0..., 0..., ..<(prefix + i + 1), 0...],
                                scale: attention.scale, mask: .none)
                        }, axis: 2
                    ).transposed(0, 2, 1, 3).reshaped(batch, length, -1)
                    hidden = hidden + exactSpeculativeLinear(attention.wo, output)
                    let normed = layer.postAttentionLayerNorm(hidden)
                    hidden =
                        hidden
                        + exactSpeculativeLinear(
                            layer.mlp.down,
                            fusedSwiGLU(
                                exactSpeculativeLinear(layer.mlp.gate, normed),
                                exactSpeculativeLinear(layer.mlp.up, normed)))
                } else {
                    hidden = layer(hidden, mask: mask, cache: cache[index])
                }
                if selected.contains(index) { features.append(hidden) }
            }
            hidden = model.norm(hidden)
            let logits =
                verify
                ? concatenated(
                    (0 ..< tokens.dim(1)).map {
                        draftReadout(hidden[0..., $0 ..< ($0 + 1), 0...])
                    }, axis: 1) : draftReadout(hidden)
            return DFlashTargetOutput(
                logits: logits, features: concatenated(features, axis: -1),
                commit: verify || recordTransaction ? commit : nil)
        } catch {
            commit(0)
            throw error
        }
    }
}
