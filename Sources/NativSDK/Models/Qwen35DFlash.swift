import Foundation
import MLX
import MLXNN

/// Map token rows without changing the target's singleton reductions.
func qwenDraftRows(_ x: MLXArray, _ body: (MLXArray, Int) -> MLXArray) -> MLXArray {
    concatenated((0 ..< x.dim(1)).map { body(x[0..., $0 ..< ($0 + 1)], $0) }, axis: 1)
}

/// Append the block once, but retain singleton SDPA reductions and causal prefixes.
func qwenDraftAttention(
    queries: MLXArray, keys: MLXArray, values: MLXArray, cache: KVCache?, scale: Float
) -> MLXArray {
    let length = queries.dim(2)
    if let cache, !(cache is KVCacheSimple) {
        return concatenated(
            (0 ..< length).map { position in
                attentionWithCacheUpdate(
                    queries: queries[0..., 0..., position ..< (position + 1), 0...],
                    keys: keys[0..., 0..., position ..< (position + 1), 0...],
                    values: values[0..., 0..., position ..< (position + 1), 0...],
                    cache: cache, scale: scale)
            }, axis: 2)
    }
    let (allKeys, allValues) = cache?.update(keys: keys, values: values) ?? (keys, values)
    let prefix = allKeys.dim(2) - length
    return concatenated(
        (0 ..< length).map { position in
            MLXFast.scaledDotProductAttention(
                queries: queries[0..., 0..., position ..< (position + 1), 0...],
                keys: allKeys[0..., 0..., ..<(prefix + position + 1), 0...],
                values: allValues[0..., 0..., ..<(prefix + position + 1), 0...], scale: scale,
                mask: .none)
        }, axis: 2)
}

struct DFlashPreparedPrompt {
    let embeddings: MLXArray?
    let positions: MLXArray?
    let state: LMOutput.State?
}

/// Owns one bounded temporal journal per GDN layer and attention append boundaries.
/// Validation completes before any cache is changed; dropping a live round aborts it.
private final class Qwen35DraftTransaction {
    private var recurrent: [(MambaCache, Int)] = []
    private let attention: [(KVCacheSimple, Int, Bool)]
    private let length: Int
    private var active = true

    init(caches: [KVCache], length: Int, prefixOnly: Bool) throws {
        self.length = length
        attention = caches.compactMap { entry in
            (entry as? KVCacheSimple).map { ($0, $0.offset, $0.keys == nil) }
        }
        guard caches.allSatisfy({ $0 is KVCacheSimple || $0 is MambaCache }) else {
            throw ModelExecutionError.invalidOptions(
                "Qwen DFlash requires native uncompressed caches")
        }
        do {
            for cache in caches.compactMap({ $0 as? MambaCache }) {
                recurrent.append(
                    (cache, try cache.beginSpeculation(length: length, prefixOnly: prefixOnly)))
            }
        } catch {
            abort()
            throw error
        }
    }

    func commit(_ count: Int) throws {
        guard active else { return }
        guard (0 ... length).contains(count) else {
            throw ModelExecutionError.invalidOptions("invalid Qwen DFlash retained prefix")
        }
        for (cache, generation) in recurrent {
            try cache.validateSpeculation(retaining: count, generation: generation)
        }
        for (cache, start, empty) in attention {
            _ = cache.trim(cache.offset - start - count)
            if count == 0 && empty {
                cache.keys = nil
                cache.values = nil
            }
        }
        for (cache, generation) in recurrent {
            try cache.commitSpeculation(retaining: count, generation: generation)
        }
        active = false
    }

    func abort() {
        guard active else { return }
        for (cache, start, empty) in attention {
            _ = cache.trim(cache.offset - start)
            if empty {
                cache.keys = nil
                cache.values = nil
            }
        }
        for (cache, generation) in recurrent { cache.abortSpeculation(generation: generation) }
        active = false
    }

    deinit { abort() }
}

extension Qwen35TextModel: DFlashTarget {
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
        guard inputEmbeddings == nil, positionIds == nil else {
            throw ModelExecutionError.invalidOptions(
                "Qwen text DFlash does not accept image embeddings")
        }
        let transaction =
            try verify || recordTransaction
            ? Qwen35DraftTransaction(caches: cache, length: tokens.dim(1), prefixOnly: !verify)
            : nil
        do {
            try Task.checkCancellation()
            var features: [MLXArray] = []
            let selected = Set(capture)
            let hidden = model.forward(
                tokens, cache: cache, applyFinalNorm: true,
                exactVerification: verify,
                hiddenSink: { index, hidden in
                    if selected.contains(index) { features.append(hidden) }
                })
            let logits: MLXArray
            if verify, let lmHead {
                logits = exactSpeculativeLinear(lmHead, hidden)
            } else if verify {
                logits = qwenDraftRows(hidden) { row, _ in draftReadout(row) }
            } else {
                logits = draftReadout(hidden)
            }
            return DFlashTargetOutput(
                logits: logits, features: concatenated(features, axis: -1),
                commit: transaction.map { value in { try value.commit($0) } }, state: state)
        } catch {
            transaction?.abort()
            throw error
        }
    }
}

extension Qwen35: DFlashTarget {
    var draftHiddenSize: Int { config.textConfiguration.hiddenSize }
    var draftLayerCount: Int { config.textConfiguration.hiddenLayers }
    var draftEmbedding: Embedding { languageModel.model.embedTokens }
    func draftReadout(_ hidden: MLXArray) -> MLXArray {
        languageModel.lmHead.map { $0(hidden) } ?? draftEmbedding.asLinear(hidden)
    }
    func draftForward(
        _ tokens: MLXArray, cache: [KVCache], capture: [Int], verify: Bool,
        inputEmbeddings: MLXArray? = nil, recordTransaction: Bool = false,
        state: LMOutput.State? = nil, positionIds: MLXArray? = nil
    ) throws -> DFlashTargetOutput {
        let transaction =
            try verify || recordTransaction
            ? Qwen35DraftTransaction(caches: cache, length: tokens.dim(1), prefixOnly: !verify)
            : nil
        do {
            try Task.checkCancellation()
            var features: [MLXArray] = []
            let selected = Set(capture)
            var state = state ?? LMOutput.State()
            state[mtpExactVerificationKey] = verify
            let output = languageModel(
                tokens, inputsEmbeds: inputEmbeddings,
                cache: cache, state: state, positionIds: positionIds,
                hiddenSink: { index, hidden in
                    if selected.contains(index) { features.append(hidden) }
                })
            var resultState = output.state
            resultState?[mtpExactVerificationKey] = nil
            return DFlashTargetOutput(
                logits: output.logits, features: concatenated(features, axis: -1),
                commit: transaction.map { value in { try value.commit($0) } }, state: resultState)
        } catch {
            transaction?.abort()
            throw error
        }
    }
}
