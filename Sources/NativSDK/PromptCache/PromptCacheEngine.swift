import Foundation
import MLX

/// Controls prefix reuse and checkpoint retention across conversation turns.
/// Cache storage, capacity, and compression are configured with `KVCacheConfiguration`.
public struct PromptCacheReuseConfiguration: Sendable {
    /// Master switch; disabled sessions run every request on the cold path.
    public var reuseEnabled: Bool
    /// Memory bound: each checkpoint holds a full copy of the per-layer cache state.
    public var maxCheckpoints: Int
    /// Snapshot every N generated tokens (nil disables mid-generation checkpoints).
    public var checkpointInterval: Int?
    /// Snapshot after prefill and after generation, so the next turn can rewind to a
    /// message boundary (covers the edit/regenerate case for non-trimmable caches).
    public var checkpointAtTurnBoundaries: Bool

    public init(
        reuseEnabled: Bool = true,
        maxCheckpoints: Int = 4,
        checkpointInterval: Int? = 256,
        checkpointAtTurnBoundaries: Bool = true,
    ) {
        self.reuseEnabled = reuseEnabled
        self.maxCheckpoints = maxCheckpoints
        self.checkpointInterval = checkpointInterval
        self.checkpointAtTurnBoundaries = checkpointAtTurnBoundaries
    }
}

public struct ModelCacheStatistics: Sendable, Codable {
    public let kind: PromptReuseKind
    public let promptTokenCount: Int
    /// Cache tokens reused instead of re-prefilled.
    public let reusedTokenCount: Int
    /// Suffix tokens actually fed through the model.
    public let prefilledTokenCount: Int
    public let generatedTokenCount: Int
    public let usedCheckpoint: Bool
    public let trimmedTokenCount: Int
    public let prefillSeconds: TimeInterval
    /// Time from cache planning through the first token, excluding input preparation.
    public let firstTokenSeconds: TimeInterval
    public let generateSeconds: TimeInterval
    public let tokensPerSecond: Double
    /// "stop" | "length" | "cancelled"
    public let stopReason: String
}

/// Reuses prompt prefixes while retaining the ordinary decoder, stopping rules and events.
/// Called on the model worker with exclusive model access.
enum PromptCacheEngine {
    static func run(
        context: ModelContext, prepared: LMInput, tools: [[String: any Sendable]]?,
        options: ModelGenerationOptions, configuration: PromptCacheReuseConfiguration,
        vault: PromptCacheVault,
        emit: (ModelGenerationEvent) throws -> Void
    ) throws -> ModelGenerationSummary {
        guard configuration.maxCheckpoints >= 0,
            configuration.checkpointInterval.map({ $0 > 0 }) ?? true
        else {
            throw ModelExecutionError.invalidOptions("invalid session cache configuration")
        }
        let start = DispatchTime.now()
        var succeeded = false
        defer { if !succeeded { vault.reset() } }
        let result = try runTurn(
            context: context, prepared: prepared, tools: tools,
            options: options, configuration: configuration,
            vault: vault, start: start, emit: emit)
        try Task.checkCancellation()
        succeeded = true
        return result
    }

    private static func runTurn(
        context: ModelContext, prepared: LMInput, tools: [[String: any Sendable]]?,
        options: ModelGenerationOptions, configuration: PromptCacheReuseConfiguration,
        vault: PromptCacheVault, start: DispatchTime,
        emit: (ModelGenerationEvent) throws -> Void
    ) throws -> ModelGenerationSummary {
        let prompt = prepared.text.tokens.asArray(Int.self)
        guard !prompt.isEmpty else {
            throw ModelExecutionError.invalidOptions("empty prompt")
        }

        let multimodal =
            prepared.image != nil || prepared.video != nil || prepared.audio != nil
        var bypass =
            !configuration.reuseEnabled || options.cache != nil || multimodal
            || prepared.text.mask != nil || options.prefill.progress != nil

        let cachedModelIdentity = vault.model.map { ObjectIdentifier($0) }
        let model = try context.languageModel
        if !bypass, cachedModelIdentity != ObjectIdentifier(model) {
            vault.reset()
        }
        if !bypass { vault.model = model }
        var state: LMOutput.State?
        var kind: PromptReuseKind = .bypass
        var keep = 0
        var trimmed = 0
        var usedCheckpoint = false
        let caches: [KVCache]

        if bypass {
            caches = try model.newCache(configuration: options.cache)
        } else {
            let plan = PromptReusePlanner.plan(history: vault.history, prompt: prompt)
            if plan.kind == .extend, let live = vault.caches {
                caches = live
                state = vault.state
                kind = .extend
                keep = plan.keep
            } else if plan.kind == .exact || plan.kind == .diverge, vault.state == nil,
                let live = vault.caches,
                PromptCacheOps.trimAll(live, by: plan.rewind)
            {
                caches = live
                kind = plan.kind
                keep = plan.keep
                trimmed = plan.rewind
            } else if let (index, checkpoint) = PromptCacheOps.bestCheckpoint(
                in: vault.checkpoints, for: prompt)
            {
                // Checkpoints are matched against the prompt itself, so they can rescue
                // any plan — including a cold one after the live history diverged away.
                caches = checkpoint.restore()
                state = try checkpoint.restoreState(model: model)
                kind = plan.kind == .cold ? .diverge : plan.kind
                keep = checkpoint.tokens.count
                usedCheckpoint = true
                // LRU: a hit proves this boundary (e.g. a shared document) is worth
                // keeping — refresh its age so pruning evicts never-used snapshots
                // first, not the checkpoint every subsequent question restores from.
                vault.checkpoints.remove(at: index)
                vault.checkpoints.append(checkpoint)
            } else {
                caches = try model.newCache(configuration: options.cache)
                kind = .cold
            }
            if keep == 0 {
                kind = .cold
            }
        }

        // Models with no reusable cache must always see the complete prompt.
        if caches.isEmpty {
            bypass = true
            kind = .bypass
            keep = 0
            state = nil
            vault.reset()
        }

        let checkpointable =
            !bypass && configuration.maxCheckpoints > 0
            && caches.allSatisfy(PromptCacheOps.isCheckpointSafe)
            && (PromptCacheOps.shouldCheckpoint(caches)
                || model is any ContinuationStateCopying)
        let suffix = Array(prompt[keep...])

        // Snapshot DURING prefill so a divergence inside a long prompt (e.g. a shared
        // document longer than a sliding window) can rewind to a mid-prompt boundary.
        // Without this, non-trimmable caches could only rewind to turn boundaries.
        var tail = suffix
        var manualPrefillSeconds: TimeInterval = 0
        if checkpointable, let interval = configuration.checkpointInterval {
            (tail, manualPrefillSeconds) = try prefillWithCheckpoints(
                model: model, caches: caches, prompt: prompt, keep: keep,
                suffix: suffix, interval: interval, vault: vault,
                configuration: configuration, state: &state,
            )
        }

        // Keep the processor's token layout (e.g. MiniCPM's [1, sequence]) when
        // removing reused or checkpoint-prefilled tokens. Rebuilding a flat array
        // loses the batch dimension expected by model.prepare.
        let suffixInput =
            bypass
            ? prepared
            : LMInput(tokens: prepared.text.tokens[.ellipsis, (prompt.count - tail.count)...])
        let iterator = try AutoregressiveIterator(
            input: suffixInput, model: model, cache: caches, state: state,
            options: options, samplingPrompt: prepared.text.tokens)
        // AutoregressiveIterator.init prefilled the tail: the cache now holds exactly `prompt`.
        if !bypass {
            vault.state = iterator.state
            vault.caches = caches
            vault.history = prompt
            if checkpointable, configuration.checkpointAtTurnBoundaries {
                appendCheckpoint(
                    to: vault, caches: caches, model: model, configuration: configuration)
            }
        }

        var firstToken: DispatchTime?
        var sinceCheckpoint = 0
        var summary = try AutoregressiveEngine.consume(
            iterator, input: prepared, context: context, tools: tools, options: options,
            additionalPrefillTime: manualPrefillSeconds,
            didAdvance: { token in
                if firstToken == nil { firstToken = DispatchTime.now() }
                if !bypass {
                    vault.state = iterator.state
                    vault.history.append(token)
                }
                if checkpointable, let interval = configuration.checkpointInterval {
                    sinceCheckpoint += 1
                    if sinceCheckpoint >= interval {
                        appendCheckpoint(
                            to: vault, caches: caches, model: model,
                            configuration: configuration)
                        sinceCheckpoint = 0
                    }
                }
            }, emit: emit)
        if !bypass, checkpointable, configuration.checkpointAtTurnBoundaries {
            appendCheckpoint(
                to: vault, caches: caches, model: model,
                configuration: configuration)
        }
        summary.cache = ModelCacheStatistics(
            kind: kind, promptTokenCount: prompt.count, reusedTokenCount: keep,
            prefilledTokenCount: suffix.count, generatedTokenCount: summary.generatedTokenCount,
            usedCheckpoint: usedCheckpoint, trimmedTokenCount: trimmed,
            prefillSeconds: summary.promptSeconds,
            firstTokenSeconds: seconds(from: start, to: firstToken ?? DispatchTime.now()),
            generateSeconds: summary.generationSeconds,
            tokensPerSecond: summary.generationSeconds > 0
                ? Double(summary.generatedTokenCount) / summary.generationSeconds : 0,
            stopReason: summary.stopReason.rawValue)
        return summary
    }

    /// Feeds all but the last ≤`interval` suffix tokens through the model in
    /// `interval`-sized chunks, snapshotting after each chunk. Returns the remaining
    /// tail (≥1 token — the iterator needs it to prime generation) and the time spent.
    ///
    /// Threads positional/recurrent model state through every chunk.
    private static func prefillWithCheckpoints(
        model: any LanguageModel,
        caches: [KVCache],
        prompt: [Int],
        keep: Int,
        suffix: [Int],
        interval: Int,
        vault: PromptCacheVault,
        configuration: PromptCacheReuseConfiguration,
        state: inout LMOutput.State?,
    ) throws -> (tail: [Int], seconds: TimeInterval) {
        let start = DispatchTime.now()
        var fed = 0
        while suffix.count - fed > interval {
            try Task.checkCancellation()
            let chunk = suffix[fed ..< (fed + interval)].map(Int32.init)
            let output = model(
                .init(tokens: MLXArray(chunk, [1, interval])), cache: caches, state: state)
            state = output.state
            fed += interval
            let tokens = Array(prompt[0 ..< (keep + fed)])
            if vault.checkpoints.last?.tokens.count != tokens.count {
                // snapshot() evals the copies, which also materializes the shared
                // computation graph of the live cache.
                if let checkpoint = checkpoint(
                    caches: caches, tokens: tokens, state: state, model: model)
                {
                    vault.checkpoints.append(checkpoint)
                }
                PromptCacheOps.prune(&vault.checkpoints, maxCount: configuration.maxCheckpoints)
            }
        }
        return (Array(suffix[fed...]), seconds(from: start, to: DispatchTime.now()))
    }

    private static func appendCheckpoint(
        to vault: PromptCacheVault,
        caches: [KVCache],
        model: any LanguageModel,
        configuration: PromptCacheReuseConfiguration,
    ) {
        guard vault.checkpoints.last?.tokens.count != vault.history.count else {
            return
        }
        if let checkpoint = checkpoint(
            caches: caches, tokens: vault.history, state: vault.state, model: model)
        {
            vault.checkpoints.append(checkpoint)
        }
        PromptCacheOps.prune(&vault.checkpoints, maxCount: configuration.maxCheckpoints)
    }

    private static func checkpoint(
        caches: [KVCache], tokens: [Int], state: LMOutput.State?, model: any LanguageModel
    ) -> PromptCacheCheckpoint? {
        do {
            let copiedState = try copyModelContinuationState(state, model: model)
            var snapshot = PromptCacheOps.snapshot(of: caches, tokens: tokens)
            snapshot.state = copiedState
            return snapshot
        } catch {
            // Opaque model state has no safe generic copy. Continue live or cold;
            // never create an incomplete checkpoint that loses positional state.
            return nil
        }
    }

    private static func seconds(from start: DispatchTime, to end: DispatchTime) -> TimeInterval {
        TimeInterval(end.uptimeNanoseconds - start.uptimeNanoseconds) / 1_000_000_000
    }
}
