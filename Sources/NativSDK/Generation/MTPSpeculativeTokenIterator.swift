import Foundation
import MLX

/// One native Qwen proposal per round. Target recurrent state is checkpointed
/// after the bonus input; rejected proposals never enter the next round.
final class MTPSpeculativeTokenIterator: GenerationTokenIterator {
    private let target: any LanguageModel
    private let draft: Qwen35MTPDraftModel
    private var targetCache: [KVCache]
    private var targetState: LMOutput.State?
    private var draftState: MTPDrafterState
    private var hidden = MLXArray.zeros([1, 0, 0])
    private var seed = MLXArray([] as [Int32])
    private let maxTokens: Int
    private var pending: [Int] = []
    private var pendingIndex = 0
    private var committedCount = 0
    private var history: [Int]
    private let checkpoint: ((DraftCacheCheckpoint) throws -> Void)?
    private(set) var failure: (any Error)?
    private(set) var statistics = SpeculativeGenerationStatistics(strategy: .mtp)
    private(set) var promptPrefillTime: TimeInterval = 0

    static func validate(
        options: ModelGenerationOptions, speculation: SpeculativeGenerationOptions,
        target: any LanguageModel, draft: Qwen35MTPDraftModel
    ) throws {
        try GenerationDriver.validate(options)
        try draft.validate(target: target)
        guard speculation.usesModelDefault || speculation.draftTokenCount == 1,
            options.sampling.temperature == 0,
            options.sampling.repetition == nil, options.sampling.presence == nil,
            options.sampling.frequency == nil, options.sampling.suppressedTokenIDs.isEmpty,
            options.cache?.capacity == nil,
            (options.cache?.strategy ?? .fullPrecision) == .fullPrecision,
            options.diffusion == nil,
            try ModelGenerationStrategy.resolve(options.mode, model: target) == .autoregressive
        else {
            throw ModelExecutionError.invalidOptions(
                "Qwen MTP supports greedy decoding with one draft token, unbounded uncompressed caches and no penalties"
            )
        }
    }

    init(
        input: LMInput, target: any LanguageModel, draft: Qwen35MTPDraftModel,
        options: ModelGenerationOptions, speculation: SpeculativeGenerationOptions,
        reuse: DraftCacheCheckpoint? = nil,
        checkpoint: ((DraftCacheCheckpoint) throws -> Void)? = nil
    ) throws {
        try Self.validate(options: options, speculation: speculation, target: target, draft: draft)
        guard input.image == nil, input.video == nil, input.audio == nil,
            input.text.mask == nil,
            input.text.tokens.ndim == 1
                || (input.text.tokens.ndim == 2 && input.text.tokens.dim(0) == 1),
            input.text.tokens.size > 0
        else {
            throw ModelExecutionError.invalidOptions(
                "Qwen MTP currently requires one unpadded text prompt")
        }
        self.target = target
        self.draft = draft
        self.checkpoint = checkpoint
        maxTokens = options.output.maxTokens ?? Int.max
        history =
            checkpoint == nil ? [] : (reuse?.tokens ?? []) + input.text.tokens.asArray(Int.self)
        targetCache =
            try reuse?.targetCaches.map { $0.copy() } ?? target.newCache(configuration: nil)
        targetState = reuse?.targetState
        draftState = draft.makeState(configuration: nil)
        if let reuse {
            draftState.cache = reuse.draftCaches.map { $0.copy() }
            draftState.nextPosition = reuse.tokens.count - 1
        }
        let start = Date.timeIntervalSinceReferenceDate
        var state = targetState ?? LMOutput.State()
        state[mtpEmitFlagKey] = true
        let tokens = input.text.tokens.flattened()
        let total = tokens.size
        var captured: [MLXArray] = []
        var processed = 0
        let reserve =
            target is Qwen35
            ? 1 : (options.prefill.chunking == .remainder ? options.prefill.resolvedStepSize() : 1)
        if total > options.prefill.resolvedStepSize(),
            let chunk = options.prefill.chunkLength(forChunking: total - reserve)
        {
            while total - processed > reserve {
                try Task.checkCancellation()
                let length = min(chunk, total - processed - 1)
                let output = target(
                    .init(tokens: tokens[processed ..< (processed + length)][.newAxis]),
                    cache: targetCache, state: state)
                state = output.state ?? state
                guard let features = output.state?[mtpLastHiddenStatesKey] else {
                    throw ModelExecutionError.invalidOptions(
                        "target did not emit MTP hidden states")
                }
                captured.append(features)
                asyncEval(features, targetCache.map { $0.state })
                processed += length
                options.prefill.progress?(processed, total)
            }
        }
        try Task.checkCancellation()
        let pinPromptPrefix = checkpoint != nil && total - processed > 1
        state[mtpCacheCheckpointIndexKey] = pinPromptPrefix ? total - processed - 1 : nil
        let output = target(
            .init(tokens: tokens[processed...][.newAxis]), cache: targetCache, state: state)
        targetState = output.state
        guard let tail = output.state?[mtpLastHiddenStatesKey] else {
            throw ModelExecutionError.invalidOptions("target did not emit MTP hidden states")
        }
        captured.append(tail)
        let promptHidden = concatenated(captured, axis: 1)
        hidden = promptHidden[0..., (-1)..., 0...]
        seed = argMax(output.logits[0, -1, 0...]).asType(.int32).reshaped(1)
        if let reuse, let previous = reuse.features {
            draft.prepareContinuation(
                target: target,
                tokens: concatenated([tokens, seed], axis: 0),
                hidden: concatenated([previous, promptHidden], axis: 1), state: &draftState)
        } else {
            draft.prepareDrafterState(
                target: target, promptTokens: tokens,
                targetHidden: promptHidden, firstBonus: seed, positionDeltas: nil,
                state: &draftState, sampler: ArgMaxSampler())
        }
        if pinPromptPrefix, let checkpoint {
            // Keep the boundary before the final generation marker as well:
            // chat templates may omit that marker when rebuilding history.
            let full = targetCache.map { $0.copy() }
            eval(full.flatMap { $0.innerState() })
            for cache in targetCache {
                if cache is KVCacheSimple { _ = cache.trim(1) }
                if let recurrent = cache as? MambaCache {
                    guard recurrent.restoreSpeculativeCheckpoint() else {
                        throw ModelExecutionError.invalidOptions(
                            "target cannot checkpoint the MTP prompt prefix")
                    }
                }
            }
            let privatePrefix = draftState.cache.map { $0.copy() }
            trimPromptCache(privatePrefix, numTokens: 2)
            try checkpoint(
                try DraftCacheCheckpoint.snapshot(
                    tokens: Array(history.dropLast()),
                    targetCaches: targetCache, draftCaches: privatePrefix,
                    targetState: checkpointState(),
                    features: promptHidden[0..., (total - 2) ..< (total - 1), 0...]))
            targetCache = full
        }
        targetState?[mtpCacheCheckpointIndexKey] = nil
        asyncEval(seed, hidden, draftState.cache.map { $0.state })
        options.prefill.progress?(total, total)
        promptPrefillTime = Date.timeIntervalSinceReferenceDate - start
        try saveCheckpoint()
    }

    func next() -> Int? {
        guard failure == nil else { return nil }
        do {
            try Task.checkCancellation()
            if pendingIndex == pending.count {
                if committedCount > 0 { try saveCheckpoint() }
                guard committedCount < maxTokens else { return nil }
                if committedCount == 0 {
                    committedCount = 1
                    let token = seed.item(Int.self)
                    if checkpoint != nil { history.append(token) }
                    return token
                }
                try round()
                pendingIndex = 0
            }
            defer { pendingIndex += 1 }
            let token = pending[pendingIndex]
            if checkpoint != nil { history.append(token) }
            return token
        } catch {
            failure = error
            return nil
        }
    }

    private func saveCheckpoint() throws {
        guard let checkpoint else { return }
        let tokens = committedCount == 0 ? history : Array(history.dropLast())
        // The last drafter entry consumed the carried bonus, which may change
        // when a chat template is rebuilt. Save the stable shifted prefix.
        let caches = draftState.cache.map { $0.copy() }
        trimPromptCache(caches, numTokens: 1)
        try checkpoint(
            try DraftCacheCheckpoint.snapshot(
                tokens: tokens,
                targetCaches: targetCache, draftCaches: caches, targetState: checkpointState(),
                features: hidden
            ))
    }

    private func checkpointState() -> LMOutput.State? {
        var state = targetState
        state?[mtpEmitFlagKey] = nil
        state?[mtpExactVerificationKey] = nil
        state?[mtpCacheCheckpointIndexKey] = nil
        state?[mtpLastHiddenStatesKey] = nil
        state?[mtpSharedKVStatesKey] = nil
        state?[mtpSharedKVOffsetsKey] = nil
        state?[mtpSharedKVSourceIndicesKey] = nil
        return state
    }

    private func round() throws {
        let count = min(1, maxTokens - committedCount - 1)
        let proposals =
            count > 0
            ? draft.draftBlock(
                target: target, lastToken: seed,
                lastHidden: hidden, sharedKV: [:], positionDeltas: nil,
                queryOffset: targetCache.first(where: { $0 is KVCacheSimple })!.offset,
                blockSize: 2, state: &draftState, sampler: ArgMaxSampler())
            : MLXArray.zeros([1, 0], dtype: .int32)
        let inputs = concatenated([seed, proposals.flattened()], axis: 0)
        let starts = targetCache.map(\.offset)
        let recurrent = targetCache.compactMap { $0 as? MambaCache }
        // An abort restores the original state; ordinary rejection uses the
        // native checkpoint after the unconditionally retained bonus input.
        let before = recurrent.map { $0.copy() as! MambaCache }
        var state = targetState ?? LMOutput.State()
        state[mtpEmitFlagKey] = true
        state[mtpCacheCheckpointIndexKey] = count > 0 ? 1 : nil
        state[mtpExactVerificationKey] = true
        let output = target(.init(tokens: inputs[.newAxis]), cache: targetCache, state: state)
        do {
            guard let features = output.state?[mtpLastHiddenStatesKey] else {
                throw ModelExecutionError.invalidOptions("target did not emit MTP hidden states")
            }
            let predictions = argMax(output.logits[0], axis: -1).asType(.int32)
            let decision = speculativeSamplingDecision(
                proposals: proposals.flattened(), predictions: predictions)
            asyncEval(decision, features)
            let row = decision.asArray(Int32.self).map(Int.init)
            try Task.checkCancellation()
            let accepted = row[0]
            let retained = accepted + 1
            for (cache, start) in zip(targetCache, starts) where cache is KVCacheSimple {
                _ = cache.trim(cache.offset - start - retained)
            }
            for cache in recurrent {
                if accepted < count {
                    guard cache.restoreSpeculativeCheckpoint() else {
                        throw ModelExecutionError.invalidOptions(
                            "target cannot rewind rejected MTP proposal")
                    }
                } else {
                    cache.discardSpeculativeCheckpoint()
                }
            }
            targetState = output.state
            targetState?[mtpCacheCheckpointIndexKey] = nil
            hidden = features[0..., accepted ..< (accepted + 1), 0...]
            seed = predictions[accepted ..< (accepted + 1)]
            draft.commitDrafterState(
                target: target, targetHidden: features,
                draftTokens: proposals, acceptedCount: accepted, finalToken: seed,
                positionDeltas: nil, state: &draftState, sampler: ArgMaxSampler())
            pending = Array(row[1 ..< (retained + 1)])
            committedCount += pending.count
            if count > 0 {
                statistics.proposedTokenCount += count
                statistics.acceptedTokenCount += accepted
                statistics.verificationCount += 1
            }
        } catch {
            for (cache, start) in zip(targetCache, starts) where cache is KVCacheSimple {
                _ = cache.trim(cache.offset - start)
            }
            for (cache, saved) in zip(recurrent, before) {
                cache.state = saved.state
                cache.metaState = saved.metaState
                cache.discardSpeculativeCheckpoint()
            }
            throw error
        }
    }
}
