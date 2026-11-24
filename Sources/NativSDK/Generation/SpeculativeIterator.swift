import Foundation
import MLX

/// Opt-in contract: text-only causal chunk logits must match single-token forwards,
/// and all continuation state must live in copyable caches or LMOutput.State.
/// Model-specific MTP heads require their own verification strategy.
public protocol SpeculativeDecodingModel: LanguageModel {}

extension Mistral4: SpeculativeDecodingModel {}
extension Qwen3Model: SpeculativeDecodingModel {}

public struct SpeculativeGenerationOptions: Sendable {
    /// Explicit proposal count. An omitted count uses four for independent models,
    /// or the specialized drafter's configured runtime default.
    public var draftTokenCount: Int { didSet { usesModelDefault = false } }
    internal private(set) var usesModelDefault: Bool
    public init() {
        draftTokenCount = 4
        usesModelDefault = true
    }
    public init(draftTokenCount: Int) {
        self.draftTokenCount = draftTokenCount
        usesModelDefault = false
    }
}

public struct SpeculativeGenerationStatistics: Sendable {
    public enum Strategy: String, Sendable { case autoregressive, dspark, dflash, mtp }
    public internal(set) var strategy: Strategy = .autoregressive
    public internal(set) var proposedTokenCount = 0
    public internal(set) var acceptedTokenCount = 0
    public internal(set) var verificationCount = 0
}

/// Greedy block verification with append-only cache trimming. Temporal caches and
/// separate model state retain snapshot/replay rollback.
final class SpeculativeIterator: GenerationTokenIterator {
    private final class Cursor {
        let model: any LanguageModel
        var cache: [KVCache] {
            didSet { leaves = KVCacheTree.leaves(in: cache).map(\.cache) }
        }
        private var leaves: [KVCache]
        var state: LMOutput.State?
        let vocabularySize: Int
        let firstToken: MLXArray
        init(
            model: any LanguageModel, cache: [KVCache], input: LMInput,
            options: ModelGenerationOptions, state: LMOutput.State? = nil
        ) throws {
            self.model = model
            self.state = try copyModelContinuationState(state, model: model)
            self.cache = cache
            self.leaves = KVCacheTree.leaves(in: cache).map(\.cache)
            let logits: MLXArray
            let preparedState = self.state
            switch try model.prepare(
                input, cache: cache, state: preparedState, prefill: options.prefill)
            {
            case .tokens(let remaining):
                guard remaining.tokens.size > 0 else {
                    throw ModelExecutionError.invalidOptions("empty prepared prompt")
                }
                let output = withPreparedCache(cache, lengths: remaining.sequenceLengths) {
                    model(
                        remaining[text: .newAxis], cache: cache.isEmpty ? nil : cache,
                        state: preparedState)
                }
                self.state = output.state
                logits = output.logits[0, -1, 0...]
            case .logits(let output):
                self.state = output.state
                logits = output.logits[0, -1, 0...]
            }
            _ = try copyModelContinuationState(self.state, model: model)
            vocabularySize = logits.size
            firstToken = argMax(logits).asType(.int32).reshaped(1)
            asyncEval(firstToken, cache.map { $0.state })
        }
        private var supportsCausalChunks: Bool {
            leaves.allSatisfy { cacheCapabilities($0).contains([.attention, .fullRewind]) }
        }
        func advance(_ tokens: MLXArray) throws -> MLXArray {
            // Rotating and recurrent cache updates retain their sequential semantics.
            if tokens.size > 1 && !supportsCausalChunks {
                return try concatenated(
                    (0 ..< tokens.size).map { try advanceSingle(tokens[$0 ..< ($0 + 1)]) }, axis: 0)
            }
            return try advanceSingle(tokens)
        }
        private func advanceSingle(_ tokens: MLXArray) throws -> MLXArray {
            try Task.checkCancellation()
            let text = LMInput.Text(tokens: tokens[.newAxis])
            let output = withPreparedCache(cache, lengths: nil) {
                model(text, cache: cache.isEmpty ? nil : cache, state: state)
            }
            state = output.state
            return output.logits[0]
        }
        func schedule(_ tokens: MLXArray? = nil) {
            // A sampled token depends on all attention updates; evaluating the
            // cache again would build redundant state slices for every layer.
            let arrays = tokens.map { [$0] } ?? cache.flatMap { $0.state }
            if !arrays.isEmpty { asyncEval(arrays) }
        }
        struct Checkpoint {
            // A nil cache snapshot denotes a verified append-only rewind boundary.
            let caches: [KVCache]?
            let state: LMOutput.State?
        }
        func checkpoint(for count: Int) throws -> Checkpoint {
            if state == nil && supportsCausalChunks
                && leaves.allSatisfy({ $0.isTrimmable(after: count) })
            {
                return Checkpoint(caches: nil, state: nil)
            }
            let copiedState = try copyModelContinuationState(state, model: model)
            let copies = cache.map { $0.copy() }
            eval(copies.flatMap { $0.innerState() })
            return Checkpoint(caches: copies, state: copiedState)
        }
        func commit(_ checkpoint: Checkpoint, appended: Int, retaining tokens: MLXArray) throws {
            guard tokens.size < appended else { return }
            if checkpoint.caches == nil && state == nil {
                try rewind(appended - tokens.size)
                return
            }
            if let copies = checkpoint.caches {
                cache = copies
            } else {
                // A model may first introduce separate state during this round.
                // Rewind the append-only cache completely, then replay to reconstruct it.
                try rewind(appended)
            }
            state = checkpoint.state
            _ = try advance(tokens)
            schedule()
        }
        private func rewind(_ count: Int) throws {
            for cache in leaves {
                guard cache.isTrimmable(after: count), cache.trim(count) == count else {
                    throw ModelExecutionError.invalidOptions(
                        "cache could not rewind speculative tokens")
                }
            }
        }
    }

    private let target: Cursor
    private let draft: Cursor
    private let draftCount: Int
    private let maxTokens: Int
    private var committedCount = 0
    private var pending: [Int] = []
    private var pendingIndex = 0
    private var seed: MLXArray
    private var deferredDraftToken: MLXArray?
    private var history: [Int]
    private let checkpoint: ((DraftCacheCheckpoint) throws -> Void)?
    private(set) var failure: (any Error)?
    private(set) var promptPrefillTime: TimeInterval
    private(set) var statistics = SpeculativeGenerationStatistics()

    static func validate(
        options: ModelGenerationOptions, speculation: SpeculativeGenerationOptions,
        target: any LanguageModel, draft: any LanguageModel
    ) throws {
        try GenerationDriver.validate(options)
        guard target !== draft else {
            throw ModelExecutionError.invalidOptions(
                "speculation requires distinct target and draft model instances"
            )
        }
        guard target is any SpeculativeDecodingModel else {
            throw ModelExecutionError.invalidOptions(
                "target \(type(of: target)) does not support speculative decoding"
            )
        }
        guard draft is any SpeculativeDecodingModel else {
            throw ModelExecutionError.invalidOptions(
                "draft \(type(of: draft)) does not support speculative decoding"
            )
        }
        guard options.sampling.temperature == 0, options.sampling.repetition == nil,
            options.sampling.presence == nil, options.sampling.frequency == nil,
            options.sampling.suppressedTokenIDs.isEmpty,
            options.diffusion == nil, (options.cache?.strategy ?? .fullPrecision) == .fullPrecision,
            try ModelGenerationStrategy.resolve(options.mode, model: target) == .autoregressive,
            try ModelGenerationStrategy.resolve(options.mode, model: draft) == .autoregressive,
            (1 ... 16).contains(speculation.draftTokenCount)
        else {
            throw ModelExecutionError.invalidOptions(
                "speculation requires greedy uncompressed autoregressive decoding without penalties, and 1...16 draft tokens"
            )
        }
    }

    init(
        input: LMInput, target: any LanguageModel, draft: any LanguageModel,
        options: ModelGenerationOptions, speculation: SpeculativeGenerationOptions,
        reuse: DraftCacheCheckpoint? = nil,
        checkpoint: ((DraftCacheCheckpoint) throws -> Void)? = nil
    ) throws {
        try Self.validate(options: options, speculation: speculation, target: target, draft: draft)
        guard input.image == nil, input.video == nil, input.audio == nil,
            input.text.tokens.size > 0, input.text.tokens.ndim == 1, input.text.mask == nil
        else {
            throw ModelExecutionError.invalidOptions(
                "speculation requires one unbatched text prompt")
        }
        let start = Date.timeIntervalSinceReferenceDate
        self.checkpoint = checkpoint
        history =
            checkpoint == nil ? [] : (reuse?.tokens ?? []) + input.text.tokens.asArray(Int.self)
        let targetCache =
            try reuse?.targetCaches.map { $0.copy() }
            ?? target.newCache(configuration: options.cache)
        let draftCache =
            try reuse?.draftCaches.map { $0.copy() } ?? draft.newCache(configuration: options.cache)
        guard
            KVCacheTree.leaves(in: targetCache + draftCache).allSatisfy({
                cacheCapabilities($0.cache).contains(.snapshot)
            })
        else {
            throw ModelExecutionError.invalidOptions("speculation requires snapshot-capable caches")
        }
        self.target = try Cursor(
            model: target, cache: targetCache, input: input, options: options,
            state: reuse?.targetState)
        let draftInput =
            reuse?.draftPendingTokens.map {
                LMInput(tokens: concatenated([$0, input.text.tokens], axis: 0))
            } ?? input
        self.draft = try Cursor(
            model: draft, cache: draftCache, input: draftInput, options: options,
            state: reuse?.draftState)
        guard self.target.vocabularySize == self.draft.vocabularySize else {
            throw ModelExecutionError.invalidOptions("target and draft vocabularies differ")
        }
        seed = self.target.firstToken
        draftCount = speculation.draftTokenCount
        maxTokens = options.output.maxTokens ?? Int.max
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
                try verifyRound()
                pendingIndex = 0
            }
            let token = pending[pendingIndex]
            pendingIndex += 1
            if checkpoint != nil { history.append(token) }
            return token
        } catch {
            failure = error
            return nil
        }
    }

    private func saveCheckpoint() throws {
        guard let checkpoint else { return }
        // Full acceptance leaves one proposal outside the draft cache. Retain it
        // with the snapshot and fold it into the next prefill on restoration.
        let tokens = committedCount == 0 ? history : Array(history.dropLast())
        try checkpoint(
            try DraftCacheCheckpoint.snapshot(
                tokens: tokens,
                targetCaches: target.cache, draftCaches: draft.cache,
                targetState: copyModelContinuationState(target.state, model: target.model),
                draftState: copyModelContinuationState(draft.state, model: draft.model),
                draftPendingTokens: deferredDraftToken))
    }

    private func verifyRound() throws {
        let count = min(draftCount, maxTokens - committedCount - 1)
        if count == 0 {
            // The final token needs only the target; do not run unused draft work.
            if checkpoint != nil {
                _ = try draft.advance(
                    deferredDraftToken.map { concatenated([$0, seed], axis: 0) } ?? seed)
                deferredDraftToken = nil
            }
            let logits = try target.advance(seed)
            seed = argMax(logits[-1, 0...]).asType(.int32).reshaped(1)
            target.schedule(seed)
            pending = [seed.item(Int.self)]
            committedCount += 1
            return
        }
        let targetCheckpoint = try target.checkpoint(for: count + 1)
        let deferredCount = deferredDraftToken == nil ? 0 : 1
        let draftCheckpoint = try draft.checkpoint(for: count + deferredCount)
        var proposals: [MLXArray] = []
        var previous = deferredDraftToken.map { concatenated([$0, seed], axis: 0) } ?? seed
        for _ in 0 ..< count {
            let logits = try draft.advance(previous)
            previous = argMax(logits[-1, 0...]).asType(.int32).reshaped(1)
            proposals.append(previous)
            draft.schedule(previous)
        }
        let proposed = concatenated(proposals, axis: 0)
        let verificationInput = concatenated([seed, proposed], axis: 0)
        let verified = try target.advance(verificationInput)
        let predictions = argMax(verified, axis: -1).asType(.int32)
        let mismatches = concatenated(
            [(proposed .!= predictions[..<count]).asType(.int32), MLXArray([Int32(1)])], axis: 0)
        let accepted = argMax(mismatches).asType(.int32).reshaped(1)
        let decision = concatenated([accepted, predictions], axis: 0)
        target.schedule(decision)
        // One host read per round carries the accepted length and all emitted IDs.
        let row = decision.asArray(Int32.self).map(Int.init)
        let acceptedCount = row[0]
        pending = Array(row[1 ..< (acceptedCount + 2)])
        seed = predictions[acceptedCount ..< (acceptedCount + 1)]
        let retained = verificationInput[..<(acceptedCount + 1)]
        try target.commit(targetCheckpoint, appended: count + 1, retaining: retained)
        // The draft has not processed its final proposal. On full acceptance,
        // fold that token into the first forward of the next round with the bonus.
        let draftRetained = verificationInput[..<min(acceptedCount + 1, count)]
        let retainedWithDeferred =
            deferredDraftToken.map {
                concatenated([$0, draftRetained], axis: 0)
            } ?? draftRetained
        try draft.commit(
            draftCheckpoint, appended: count + deferredCount, retaining: retainedWithDeferred)
        deferredDraftToken = acceptedCount == count ? proposals.last : nil
        statistics.proposedTokenCount += count
        statistics.acceptedTokenCount += acceptedCount
        statistics.verificationCount += 1
        committedCount += pending.count
    }
}
