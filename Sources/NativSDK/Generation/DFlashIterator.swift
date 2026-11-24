import Foundation
import MLX

/// Resolve the target decoder while preserving multimodal wrappers for image prefill.
func resolveDFlashTarget(_ target: any LanguageModel) throws -> any DFlashTarget {
    if let text = target as? LFM2Model { return text }
    if let vision = target as? LFM2VL { return vision.dflashLanguageModel }
    if let laguna = target as? Laguna { return laguna }
    if let qwen = target as? Qwen3Model { return qwen }
    if let qwen = target as? Qwen35 { return qwen }
    if let qwen = target as? Qwen35TextModel { return qwen }
    if let qwen = target as? Qwen35Model { return qwen.languageModel }
    if let gemma = target as? any Gemma4DraftTarget { return gemma }
    if let gemma = target as? Gemma4TextModel { return gemma }
    if let gemma = target as? Gemma4Model { return gemma.languageModel }
    throw ModelExecutionError.invalidOptions(
        "DFlash/DSpark requires an LFM2, LFM2-VL, dense Qwen3, Qwen3.5-family, Gemma4 or Laguna target"
    )
}

/// Single-request, target-bound block generation. Features describe only committed
/// target inputs; the carried bonus has not entered either cache yet.
final class DFlashIterator: GenerationTokenIterator {
    private let target: any DFlashTarget
    private let draft: any DFlashDrafter
    private var targetCache: [KVCache]
    private let draftCache: [KVCache]
    private var targetState: LMOutput.State?
    private let draftCount: Int
    private let maxTokens: Int
    private let sampler: any KeyedLogitSampler
    private let proposalSampler: any LogitSampler
    private var targetRandom: SpeculativeRandomCursor?
    private var features: MLXArray
    private var seed: MLXArray
    private var pending: [Int] = []
    private var pendingIndex = 0
    private var committedCount = 0
    private var history: [Int]
    private let checkpoint: ((DraftCacheCheckpoint) throws -> Void)?
    private(set) var failure: (any Error)?
    private(set) var statistics = SpeculativeGenerationStatistics(strategy: .dspark)
    private(set) var promptPrefillTime: TimeInterval

    static func validate(
        options: ModelGenerationOptions, speculation: SpeculativeGenerationOptions,
        target: any DFlashTarget, draft: any DFlashDrafter
    ) throws {
        try GenerationDriver.validate(options)
        try draft.config.validate(target: target)
        let count =
            speculation.usesModelDefault
            ? draft.config.defaultProposalCount : speculation.draftTokenCount
        guard (1 ... draft.config.proposalCount).contains(count),
            options.sampling.repetition == nil,
            options.sampling.presence == nil, options.sampling.frequency == nil,
            options.sampling.suppressedTokenIDs.isEmpty, options.diffusion == nil,
            options.cache?.capacity == nil,
            (options.cache?.strategy ?? .fullPrecision) == .fullPrecision,
            try ModelGenerationStrategy.resolve(options.mode, model: target) == .autoregressive
        else {
            throw ModelExecutionError.invalidOptions(
                "DFlash/DSpark requires uncompressed unbounded generation without penalties, and 1...\(draft.config.proposalCount) draft tokens"
            )
        }
    }

    init(
        input: LMInput, target: any LanguageModel, draft: any DFlashDrafter,
        options: ModelGenerationOptions, speculation: SpeculativeGenerationOptions,
        reuse: DraftCacheCheckpoint? = nil,
        checkpoint: ((DraftCacheCheckpoint) throws -> Void)? = nil,
        checkpointSuffixCount: Int = 1
    ) throws {
        guard checkpointSuffixCount > 0 else {
            throw ModelExecutionError.invalidOptions("invalid draft prompt checkpoint boundary")
        }
        let vision = target as? LFM2VL
        let qwenVision = target as? Qwen35
        let gemma = target is any Gemma4DraftTarget
        let hasVision = vision != nil || qwenVision != nil || gemma
        let target = try resolveDFlashTarget(target)
        try Self.validate(options: options, speculation: speculation, target: target, draft: draft)
        guard input.video == nil, input.audio == nil,
            input.image == nil || hasVision,
            input.text.tokens.ndim == 1
                || ((hasVision || target is Gemma4TextModel || target is Qwen35TextModel)
                    && input.text.tokens.ndim == 2
                    && input.text.tokens.dim(0) == 1),
            input.text.tokens.size > 0,
            input.text.mask == nil
                || (hasVision && all(input.text.mask! .== 1).item(Bool.self))
        else {
            throw ModelExecutionError.invalidOptions(
                "DFlash/DSpark requires one unpadded prompt; images require an LFM2-VL, Qwen3.5 or Gemma4 VL target"
            )
        }
        self.target = target
        self.draft = draft
        statistics = SpeculativeGenerationStatistics(strategy: draft.config.strategy)
        sampler = options.sampling.makeKeyedSampler()
        targetRandom =
            options.sampling.temperature == 0
            ? nil : SpeculativeRandomCursor(seed: options.sampling.seed)
        // Proposal draws never advance target randomness, including unseeded requests.
        proposalSampler = ModelSamplingOptions(
            temperature: options.sampling.temperature,
            seed: options.sampling.seed.map { $0 ^ 0x0DFA_5202 }
        ).makeSampler()
        draftCount =
            speculation.usesModelDefault
            ? draft.config.defaultProposalCount : speculation.draftTokenCount
        maxTokens = options.output.maxTokens ?? Int.max
        self.checkpoint = checkpoint
        history =
            checkpoint == nil
            ? [] : (reuse?.tokens ?? []) + input.text.tokens.flattened().asArray(Int.self)
        targetCache =
            try reuse?.targetCaches.map { $0.copy() } ?? target.newCache(configuration: nil)
        draftCache = reuse?.draftCaches.map { $0.copy() } ?? draft.newCache()
        targetState = reuse?.targetState
        let start = Date.timeIntervalSinceReferenceDate
        let tokens = input.text.tokens.flattened()
        let prepared =
            try qwenVision?.prepareDFlashPrompt(
                input, cache: targetCache, state: targetState)
            ?? (target as? any Gemma4DraftTarget)?.prepareGemmaDraftPrompt(input)
        let embeddings = try prepared?.embeddings ?? vision?.dflashInputEmbeddings(input)
        if let prepared { targetState = prepared.state }
        let total = tokens.size
        let prefill = options.prefill
        var processed = 0
        var captured: [MLXArray] = []
        // Match each target's ordinary prefill partition, including VL's single
        // reserved position, so the initial BF16 decision remains identical.
        let checkpointTail = checkpoint == nil ? 1 : checkpointSuffixCount
        let reserve = max(
            checkpointTail,
            !hasVision && prefill.chunking == .remainder ? prefill.resolvedStepSize() : 1)
        if !(gemma && input.image != nil),
            vision != nil || gemma || total > prefill.resolvedStepSize()
                || (qwenVision != nil && reuse != nil),
            let chunk = prefill.chunkLength(forChunking: total - reserve)
        {
            while total - processed > reserve {
                try Task.checkCancellation()
                let length = min(chunk, total - processed - checkpointTail)
                let output = try target.draftForward(
                    tokens[processed ..< (processed + length)][.newAxis],
                    cache: targetCache, capture: draft.config.targetLayers, verify: false,
                    inputEmbeddings: embeddings?[0..., processed ..< (processed + length), 0...],
                    recordTransaction: false, state: targetState,
                    positionIds: prepared?.positions?[
                        0..., 0..., processed ..< (processed + length)])
                targetState = output.state ?? targetState
                captured.append(output.features)
                asyncEval(output.features, targetCache.map { $0.state })
                processed += length
                prefill.progress?(processed, total)
            }
            eval(targetCache.map { $0.state })
        }
        let output = try target.draftForward(
            tokens[processed...][.newAxis],
            cache: targetCache, capture: draft.config.targetLayers, verify: false,
            inputEmbeddings: embeddings?[0..., processed..., 0...],
            recordTransaction: checkpoint != nil && total > checkpointTail,
            state: targetState, positionIds: prepared?.positions?[0..., 0..., processed...])
        targetState = output.state ?? targetState
        captured.append(output.features)
        features = concatenated((reuse?.features.map { [$0] } ?? []) + captured, axis: 1)
        if let checkpoint, total > checkpointTail {
            // Templates can remove generation markers when rebuilding a completed
            // turn. Preserve both the stable prompt prefix and the full prefill.
            let fullCache = targetCache.map { $0.copy() }
            eval(fullCache.flatMap { $0.innerState() })
            try output.commit!(total - processed - checkpointTail)
            try checkpoint(
                try DraftCacheCheckpoint.snapshot(
                    tokens: Array(history.dropLast(checkpointTail)),
                    targetCaches: targetCache, draftCaches: draftCache, targetState: targetState,
                    features: features[0..., ..<(features.dim(1) - checkpointTail), 0...]))
            targetCache = fullCache
        }
        if let block = targetRandom?.prepare(count: 1) {
            seed = sampler.sample(logits: output.logits[0..., -1, 0...], key: block.keys[0]).asType(
                .int32
            ).reshaped(1)
            targetRandom!.commit(block, retaining: 1)
        } else {
            seed = argMax(output.logits[0, -1, 0...]).asType(.int32).reshaped(1)
        }
        asyncEval(seed, features)
        prefill.progress?(total, total)
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
        // Every delivered bonus is still outside the target cache. Checkpoint only
        // before a round (or after it drains), never with un-emitted verified inputs.
        let tokens = committedCount == 0 ? history : Array(history.dropLast())
        try checkpoint(
            try DraftCacheCheckpoint.snapshot(
                tokens: tokens,
                targetCaches: targetCache, draftCaches: draftCache, targetState: targetState,
                features: features))
    }

    private func round() throws {
        let count = min(draftCount, maxTokens - committedCount - 1)
        let proposals =
            count > 0
            ? try draft.propose(
                seed: seed, features: features, count: count, target: target, cache: draftCache,
                sampler: proposalSampler)
            : MLXArray([] as [Int32])
        let input = concatenated([seed, proposals], axis: 0)
        let output = try target.draftForward(
            input[.newAxis], cache: targetCache,
            capture: draft.config.targetLayers, verify: true, inputEmbeddings: nil,
            recordTransaction: false, state: targetState, positionIds: nil)
        do {
            let randomBlock = targetRandom?.prepare(count: count + 1)
            let predictions =
                randomBlock.map {
                    sampleSpeculativeBlock(output.logits[0], sampler: sampler, keys: $0.keys)
                } ?? argMax(output.logits[0], axis: -1).asType(.int32)
            let decision = speculativeSamplingDecision(
                proposals: proposals, predictions: predictions)
            asyncEval(decision, output.features)
            let row = decision.asArray(Int32.self).map(Int.init)
            try Task.checkCancellation()
            let retained = row[0] + 1
            try output.commit!(retained)
            targetState = output.state ?? targetState
            if let randomBlock { targetRandom!.commit(randomBlock, retaining: retained) }
            let retainedFeatures = output.features[0..., ..<retained, 0...]
            // A target-only final step did not project the preceding features into
            // the draft cache. Preserve that lag for the next session turn.
            features =
                count > 0 || checkpoint == nil
                ? retainedFeatures : concatenated([features, retainedFeatures], axis: 1)
            seed = predictions[row[0] ..< (row[0] + 1)]
            pending = Array(row[1 ..< (retained + 1)])
            committedCount += pending.count
            if count > 0 {
                statistics.proposedTokenCount += count
                statistics.acceptedTokenCount += row[0]
                statistics.verificationCount += 1
            }
        } catch {
            try? output.commit!(0)
            throw error
        }
    }
}
