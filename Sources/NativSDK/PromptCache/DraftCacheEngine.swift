import Foundation
import MLX

/// Paired prefix reuse for draft session requests. All access occurs on the model
/// worker under admission; checkpoints never include un-emitted speculative tails.
enum DraftCacheEngine {
    static func run(
        input: LMInput, target: ModelContext, draft: DraftModelResources,
        tools: [[String: any Sendable]]?, options: ModelGenerationOptions,
        speculation: SpeculativeGenerationOptions, vault sessionCache: DraftCacheVault?,
        configuration cacheReuse: PromptCacheReuseConfiguration,
        emit: (ModelGenerationEvent) throws -> Void
    ) throws -> ModelGenerationSummary {
        var succeeded = false
        defer { if !succeeded { sessionCache?.reset() } }
        guard cacheReuse.maxCheckpoints >= 0,
            cacheReuse.checkpointInterval.map({ $0 > 0 }) ?? true
        else {
            throw ModelExecutionError.invalidOptions("invalid session cache configuration")
        }
        let start = Date.timeIntervalSinceReferenceDate
        let prompt = sessionCache == nil ? [] : input.text.tokens.flattened().asArray(Int.self)
        // Gemma's processor supplies an all-ones mask even for one unpadded
        // text prompt. It does not change attention and is safe for prefix reuse.
        let unpaddedGemmaMask =
            try target.languageModel is any Gemma4DraftTarget
            && input.text.mask.map { mask in
                mask.shape == input.text.tokens.shape && all(mask .== 1).item(Bool.self)
            } == true
        let bypass =
            !cacheReuse.reuseEnabled || options.cache != nil
            || input.image != nil || input.video != nil || input.audio != nil
            || (input.text.mask != nil && !unpaddedGemmaMask) || options.prefill.progress != nil
        if bypass { sessionCache?.reset() }
        let vault = bypass ? nil : sessionCache
        let plan =
            prompt.isEmpty
            ? nil
            : vault?.latest.map {
                PromptReusePlanner.plan(history: $0.tokens, prompt: prompt)
            }
        var reuse = vault?.best(for: prompt)
        var trimmed = 0
        // Attention-only independent pairs can rewind directly. Hybrid models
        // restore a paired checkpoint and replay the unmatched suffix instead.
        if let latest = vault?.latest, let plan, plan.keep > 0,
            plan.keep > (reuse?.tokens.count ?? 0), latest.features == nil,
            latest.targetState == nil, latest.draftState == nil
        {
            let targetCopies = latest.targetCaches.map { $0.copy() }
            let draftCopies = latest.draftCaches.map { $0.copy() }
            let draftRewind = plan.rewind - (latest.draftPendingTokens?.size ?? 0)
            if draftRewind >= 0,
                PromptCacheOps.trimAll(targetCopies, by: plan.rewind),
                PromptCacheOps.trimAll(draftCopies, by: draftRewind)
            {
                reuse = DraftCacheCheckpoint(
                    tokens: Array(prompt.prefix(plan.keep)),
                    targetCaches: targetCopies, draftCaches: draftCopies,
                    targetState: nil, draftState: nil, features: nil, draftPendingTokens: nil)
                trimmed = plan.rewind
            }
        }
        let reused = reuse?.tokens.count ?? 0
        var checkpointSuffixCount = 1
        if try target.languageModel is any Gemma4DraftTarget,
            let channel = target.tokenizer.convertTokenToId("<|channel>"),
            let end = target.tokenizer.convertTokenToId("<channel|>"),
            prompt.last == end, let marker = prompt.lastIndex(of: channel), marker > reused
        {
            // Gemma removes its empty thought-channel header from completed turns.
            // Keep the stable prefix before that header for the next request.
            checkpointSuffixCount = prompt.count - marker
        }
        let suffix =
            reused > 0
            ? LMInput(tokens: MLXArray(Array(prompt.dropFirst(reused)))) : input
        var firstBoundary = true
        var retainedAt = reused
        var promptBoundary: [Int]?
        let save: ((DraftCacheCheckpoint) throws -> Void)? = vault.map { vault in
            { checkpoint in
                guard !checkpoint.targetCaches.isEmpty, !checkpoint.draftCaches.isEmpty else {
                    vault.reset()
                    return
                }
                vault.latest = checkpoint
                if firstBoundary {
                    if cacheReuse.checkpointAtTurnBoundaries {
                        promptBoundary = checkpoint.tokens
                        vault.retain(
                            checkpoint, maxCount: cacheReuse.maxCheckpoints,
                            protecting: promptBoundary)
                    }
                    firstBoundary = false
                    retainedAt = checkpoint.tokens.count
                } else if let interval = cacheReuse.checkpointInterval,
                    checkpoint.tokens.count - retainedAt >= interval
                {
                    vault.retain(
                        checkpoint, maxCount: cacheReuse.maxCheckpoints,
                        protecting: promptBoundary)
                    retainedAt = checkpoint.tokens.count
                }
            }
        }
        let iterator: any GenerationTokenIterator
        let statistics: () -> SpeculativeGenerationStatistics
        switch draft {
        case .autoregressive(let draft):
            let stepper = try SpeculativeIterator(
                input: suffix, target: target.languageModel, draft: draft.languageModel,
                options: options, speculation: speculation, reuse: reuse, checkpoint: save)
            iterator = stepper
            statistics = { stepper.statistics }
        case .mtp(let draft):
            let stepper = try MTPSpeculativeTokenIterator(
                input: suffix, target: target.languageModel, draft: draft,
                options: options, speculation: speculation, reuse: reuse, checkpoint: save)
            iterator = stepper
            statistics = { stepper.statistics }
        case .dspark(let draft):
            let stepper = try DFlashIterator(
                input: suffix, target: target.languageModel, draft: draft,
                options: options, speculation: speculation, reuse: reuse, checkpoint: save,
                checkpointSuffixCount: checkpointSuffixCount)
            iterator = stepper
            statistics = { stepper.statistics }
        }
        let planningTime = max(
            0, Date.timeIntervalSinceReferenceDate - start - iterator.promptPrefillTime)
        var result = try AutoregressiveEngine.consume(
            iterator, input: input, context: target, tools: tools, options: options,
            additionalPrefillTime: planningTime
        ) {
            try emit($0)
        }
        if cacheReuse.checkpointAtTurnBoundaries, let latest = vault?.latest {
            vault?.retain(latest, maxCount: cacheReuse.maxCheckpoints, protecting: promptBoundary)
        }
        if sessionCache != nil {
            result.cache = ModelCacheStatistics(
                kind: bypass || vault?.latest == nil
                    ? .bypass : (reused > 0 ? (plan?.kind ?? .extend) : .cold),
                promptTokenCount: prompt.count, reusedTokenCount: reused,
                prefilledTokenCount: prompt.count - reused,
                generatedTokenCount: result.generatedTokenCount,
                usedCheckpoint: reused > 0 && trimmed == 0, trimmedTokenCount: trimmed,
                prefillSeconds: iterator.promptPrefillTime, firstTokenSeconds: result.promptSeconds,
                generateSeconds: result.generationSeconds,
                tokensPerSecond: result.generationSeconds > 0
                    ? Double(result.generatedTokenCount) / result.generationSeconds : 0,
                stopReason: result.stopReason.rawValue)
        }
        result.speculative = statistics()
        try Task.checkCancellation()
        succeeded = true
        return result
    }
}
