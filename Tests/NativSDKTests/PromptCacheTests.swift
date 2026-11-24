import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

/// Deterministic model: always predicts the EOS token (id 2) and advances every cache
/// layer by the fed length, so each turn generates exactly one token and cache offsets
/// mirror token counts.
private final class SessionTestModel: Module, SpeculativeDecodingModel, ContinuationStateCopying {
    static let position = LMOutput.Key<Int>("position")
    var seenPositions: [Int] = []
    let tracksState: Bool
    let nextToken: Int
    let batchedInput: Bool
    enum CacheKind {
        case simple
        case hybrid
    }

    let cacheKind: CacheKind
    let vocabularySize = 128

    init(
        cacheKind: CacheKind, tracksState: Bool = false, nextToken: Int = 2,
        batchedInput: Bool = false
    ) {
        self.batchedInput = batchedInput
        self.nextToken = nextToken
        self.tracksState = tracksState
        self.cacheKind = cacheKind
        super.init()
    }

    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        if batchedInput {
            guard input.text.tokens.ndim == 2, input.text.tokens.dim(0) == 1 else {
                throw ModelExecutionError.invalidOptions("processor batch dimension was lost")
            }
            return .logits(callAsFunction(input.text, cache: cache, state: state))
        }
        return .tokens(input.text)
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let length = inputs.dim(1)
        if let cache {
            for entry in cache {
                if let arrays = entry as? ArraysCache {
                    arrays[0] = MLXArray.zeros([1, 4])
                    arrays[1] = MLXArray.zeros([1, 4])
                    arrays.offset += length
                } else {
                    _ = entry.update(
                        keys: MLXArray.zeros([1, 1, length, 4]),
                        values: MLXArray.zeros([1, 1, length, 4]),
                    )
                }
            }
        }
        let logits = MLXArray.zeros([inputs.dim(0), length, vocabularySize])
        logits[.ellipsis, nextToken] = MLXArray(Float(100))
        return logits
    }

    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let logits = callAsFunction(input.tokens, cache: cache)
        guard tracksState else { return LMOutput(logits: logits) }
        let position = state?[Self.position] ?? 0
        seenPositions.append(position)
        var next = LMOutput.State()
        next[Self.position] = position + input.tokens.dim(-1)
        return LMOutput(logits: logits, state: next)
    }

    func copyContinuationState(_ state: LMOutput.State) throws -> LMOutput.State {
        var copy = LMOutput.State()
        copy[Self.position] = state[Self.position]
        return copy
    }

    func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        let attention = try makeAttentionKVCache(configuration: configuration)
        switch cacheKind {
        case .simple: return [attention]
        case .hybrid: return [MambaCache(), attention]
        }
    }
}

/// Tokenizes text as raw unicode scalars, so token arrays concatenate exactly like the
/// underlying strings. EOS is "\u{02}" (id 2).
private struct SessionTestTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { "\u{02}" }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        text.unicodeScalars.map { Int($0.value) }
    }

    func decode(tokenIds: [Int], skipSpecialTokens _: Bool) -> String {
        String(String.UnicodeScalarView(tokenIds.compactMap { Unicode.Scalar(UInt32($0)) }))
    }

    func convertTokenToId(_ token: String) -> Int? {
        token.unicodeScalars.first.map { Int($0.value) }
    }

    func convertIdToToken(_ id: Int) -> String? {
        Unicode.Scalar(UInt32(id)).map { String($0) }
    }

    func applyChatTemplate(
        messages _: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?,
    ) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }
}

/// Encodes the user prompt verbatim via the scalar tokenizer. A prompt containing
/// "!image" additionally attaches a dummy processed image (multimodal marker).
private struct SessionTestProcessor: UserInputProcessor {
    var batchedInput = false
    func prepare(input: UserInput) async throws -> LMInput {
        let messages = DefaultMessageGenerator().generate(from: input)
        let text = messages.compactMap { $0["content"] as? String }.joined()
        let tokens = text.unicodeScalars.map { Int($0.value) }
        let image: LMInput.ProcessedImage? =
            text.contains("!image")
            ? LMInput.ProcessedImage(pixels: MLXArray.zeros([1, 2, 2, 3])) : nil
        let array = MLXArray(tokens)
        return LMInput(text: .init(tokens: batchedInput ? array[.newAxis] : array), image: image)
    }
}

final class PromptCacheTests: XCTestCase {
    private func makeSession(
        cacheKind: SessionTestModel.CacheKind = .simple,
        batchedInput: Bool = false,
        configuration: PromptCacheReuseConfiguration = PromptCacheReuseConfiguration(),
    ) -> CacheTestSession {
        let context = ModelContext(
            metadata: .init(),
            model: SessionTestModel(cacheKind: cacheKind, batchedInput: batchedInput),
            processor: SessionTestProcessor(batchedInput: batchedInput),
            tokenizer: SessionTestTokenizer(),
        )
        return CacheTestSession(
            loadedModel: makeTestModel(context: context), configuration: configuration)
    }

    private func respond(
        _ session: CacheTestSession,
        _ prompt: String,
        options: ModelGenerationOptions = ModelGenerationOptions(
            sampling: .init(temperature: 0), output: .init(maxTokens: 8)),
    ) async throws -> (text: String, stats: ModelCacheStatistics) {
        let (text, summary) = try await session.run(prompt: prompt, options: options)
        return (text, try XCTUnwrap(summary.cache))
    }

    func testBatchedProcessorShapeSurvivesColdExtendAndCheckpointRestore() async throws {
        for kind: SessionTestModel.CacheKind in [.simple, .hybrid] {
            let session = makeSession(
                cacheKind: kind, batchedInput: true,
                configuration: .init(maxCheckpoints: 16, checkpointInterval: 2))
            let (_, cold) = try await respond(session, "ABCDEFG")
            XCTAssertEqual(cold.kind, .cold)
            XCTAssertEqual(cold.prefilledTokenCount, 7)
            let (_, extended) = try await respond(session, "ABCDEFG\u{02}HI")
            XCTAssertEqual(extended.kind, .extend)
            XCTAssertEqual(extended.reusedTokenCount, 8)
            XCTAssertEqual(extended.prefilledTokenCount, 2)
            let (_, diverged) = try await respond(session, "ABCDXY")
            XCTAssertEqual(diverged.kind, .diverge)
            XCTAssertEqual(diverged.reusedTokenCount, 4)
            XCTAssertEqual(diverged.prefilledTokenCount, 2)
            if case .hybrid = kind { XCTAssertTrue(diverged.usedCheckpoint) }
        }
    }

    func testFirstTurnIsColdAndTracksHistory() async throws {
        let session = makeSession()
        let (text, stats) = try await respond(session, "AB")

        XCTAssertEqual(text, "", "EOS must not be emitted as text")
        XCTAssertEqual(stats.kind, .cold)
        XCTAssertEqual(stats.promptTokenCount, 2)
        XCTAssertEqual(stats.reusedTokenCount, 0)
        XCTAssertEqual(stats.prefilledTokenCount, 2)
        XCTAssertEqual(stats.generatedTokenCount, 0, "model predicts EOS immediately")
        XCTAssertEqual(stats.stopReason, "stop")
    }

    func testSessionExtendAcrossTurns() async throws {
        let session = makeSession()
        let (_, first) = try await respond(session, "AB")
        // History is now [A, B, eos]; a prompt continuing it must reuse all 3 tokens.
        let (_, second) = try await respond(session, "AB\u{02}CD")

        XCTAssertEqual(second.kind, .extend)
        XCTAssertEqual(
            second.reusedTokenCount,
            first.promptTokenCount + 1,
            "reuse must cover the previous prompt AND its generated tokens (incl. EOS)",
        )
        XCTAssertEqual(second.prefilledTokenCount, 2)
        XCTAssertFalse(second.usedCheckpoint)
        XCTAssertEqual(second.trimmedTokenCount, 0)
    }

    func testSessionExactRepeatTrimsOne() async throws {
        let session = makeSession()
        _ = try await respond(session, "AB")
        let (_, second) = try await respond(session, "AB")

        XCTAssertEqual(second.kind, .exact)
        XCTAssertEqual(second.reusedTokenCount, 1)
        XCTAssertEqual(second.prefilledTokenCount, 1, "exact repeat re-feeds only the last token")
        // History was [A, B, eos] (3), keep is 1 → trim 2 from a trimmable cache.
        XCTAssertEqual(second.trimmedTokenCount, 2)
        XCTAssertFalse(second.usedCheckpoint)
    }

    func testSessionDivergeRestoresCheckpointOnHybridModel() async throws {
        let session = makeSession(cacheKind: .hybrid)
        _ = try await respond(session, "AB")
        let (_, second) = try await respond(session, "AB\u{02}XY")
        XCTAssertEqual(second.kind, .extend)

        // History is [A, B, eos, X, Y, eos]; this prompt shares only [A, B, eos] with it.
        // MambaCache is not trimmable, so rewind must go through a checkpoint.
        let (_, third) = try await respond(session, "AB\u{02}QR")
        XCTAssertEqual(third.kind, .diverge)
        XCTAssertTrue(third.usedCheckpoint)
        XCTAssertEqual(third.reusedTokenCount, 3)
        XCTAssertEqual(third.prefilledTokenCount, 2)
        XCTAssertEqual(third.trimmedTokenCount, 0)
    }

    func testPrefillCheckpointsEnableDivergeReuseInsideLongPrompt() async throws {
        // SWA/hybrid case from the issue: the shared prefix is longer than the
        // checkpoint interval, and the caches are not trimmable. A divergence in the
        // MIDDLE of a long prompt must be recoverable from a mid-prefill snapshot,
        // not only from turn boundaries.
        let session = makeSession(cacheKind: .hybrid)
        let shared = String(repeating: "A", count: 300)

        let (_, first) = try await respond(session, shared + String(repeating: "B", count: 300))
        XCTAssertEqual(first.kind, .cold)
        XCTAssertEqual(first.promptTokenCount, 600)

        // Same 300-token document, different continuation: LCP is 300, so the
        // 256-token prefill checkpoint is the best restorable prefix.
        let (_, second) = try await respond(session, shared + String(repeating: "C", count: 200))
        XCTAssertEqual(second.kind, .diverge)
        XCTAssertTrue(second.usedCheckpoint)
        XCTAssertEqual(second.reusedTokenCount, 256)
        XCTAssertEqual(second.prefilledTokenCount, 500 - 256)
    }

    func testUsedCheckpointSurvivesPruning() async throws {
        // Gemma-bench regression: turn 2 restores the shared-document checkpoint, but
        // its own post-prefill snapshots (which include turn-2's question) then push
        // that checkpoint out of the age-based prune window — and turn 3, asking a new
        // question about the SAME document, goes cold. A hit must refresh the
        // checkpoint's age (LRU) so proven boundaries outlive never-used snapshots.
        let session = makeSession(cacheKind: .hybrid)
        let shared = String(repeating: "A", count: 280)

        _ = try await respond(session, shared + String(repeating: "B", count: 320))
        let (_, second) = try await respond(session, shared + String(repeating: "C", count: 200))
        XCTAssertTrue(second.usedCheckpoint)
        XCTAssertEqual(second.reusedTokenCount, 256)

        let (_, third) = try await respond(session, shared + String(repeating: "D", count: 200))
        XCTAssertTrue(
            third.usedCheckpoint,
            "a checkpoint that just produced a hit must not be pruned away",
        )
        XCTAssertEqual(third.reusedTokenCount, 256)
    }

    func testSessionDivergeWithoutCheckpointGoesCold() async throws {
        var configuration = PromptCacheReuseConfiguration()
        configuration.checkpointAtTurnBoundaries = false
        configuration.checkpointInterval = nil
        let session = makeSession(cacheKind: .hybrid, configuration: configuration)
        _ = try await respond(session, "AB")
        _ = try await respond(session, "AB\u{02}XY")

        let (_, third) = try await respond(session, "AB\u{02}QR")
        XCTAssertEqual(third.kind, .cold)
        XCTAssertFalse(third.usedCheckpoint)
        XCTAssertEqual(third.reusedTokenCount, 0)
        XCTAssertEqual(third.prefilledTokenCount, third.promptTokenCount)
    }

    func testMultimodalBypassLeavesSessionStateUntouched() async throws {
        let session = makeSession()
        let (_, first) = try await respond(session, "AB")

        let (_, second) = try await respond(session, "AB!image")
        XCTAssertEqual(second.kind, .bypass)
        XCTAssertEqual(second.reusedTokenCount, 0)

        // The bypass turn must not have polluted history: continuing the ORIGINAL
        // conversation still extends it.
        let (_, third) = try await respond(session, "AB\u{02}CD")
        XCTAssertEqual(third.kind, .extend)
        XCTAssertEqual(
            third.reusedTokenCount, first.promptTokenCount + 1)
    }

    func testExplicitCachePolicyBypassesReuse() async throws {
        let session = makeSession()
        _ = try await respond(session, "AB")
        var options = ModelGenerationOptions(
            sampling: .init(temperature: 0), output: .init(maxTokens: 8))
        options.cache = .init(strategy: .affine(try .init(bits: 4)), compatibility: .allowPartial)
        let (_, compressed) = try await respond(session, "AB\u{02}CD", options: options)
        XCTAssertEqual(compressed.kind, .bypass)
        let (_, compressedAgain) = try await respond(session, "AB\u{02}CD", options: options)
        XCTAssertEqual(compressedAgain.kind, .bypass)
        options.cache = .init(capacity: try .init(maxTokens: 16))
        let (_, changed) = try await respond(session, "AB\u{02}CD", options: options)
        XCTAssertEqual(changed.kind, .bypass)
    }

    func testPositionalStateSurvivesChunkingExtensionAndCheckpointRestore() async throws {
        let loadedModel = makeTestModel(
            context: ModelContext(
                metadata: .init(),
                model: SessionTestModel(cacheKind: .hybrid, tracksState: true),
                processor: SessionTestProcessor(), tokenizer: SessionTestTokenizer()))
        let session = CacheTestSession(
            loadedModel: loadedModel,
            configuration: .init(maxCheckpoints: 16, checkpointInterval: 2))
        _ = try await respond(session, "ABCDEFG")
        let firstPositions = try await loadedModel.withModel { _, context in
            (context.model as! SessionTestModel).seenPositions
        }
        XCTAssertEqual(firstPositions, [0, 2, 4, 6, 7])
        let (_, extended) = try await respond(session, "ABCDEFG\u{02}HI")
        XCTAssertEqual(extended.kind, .extend)
        let secondPositions = try await loadedModel.withModel { _, context in
            (context.model as! SessionTestModel).seenPositions
        }
        XCTAssertEqual(Array(secondPositions.suffix(2)), [8, 10])
        let (_, restored) = try await respond(session, "ABCDXY")
        XCTAssertTrue(restored.usedCheckpoint)
        XCTAssertEqual(restored.reusedTokenCount, 4)
        let lastPositions = try await loadedModel.withModel { _, context in
            (context.model as! SessionTestModel).seenPositions
        }
        XCTAssertEqual(Array(lastPositions.suffix(2)), [4, 6])
    }

}

private struct CacheTestSession: Sendable {
    let loadedModel: NativModel
    let configuration: PromptCacheReuseConfiguration
    let vault = PromptCacheVault()
    func run(prompt: String, options: ModelGenerationOptions) async throws -> (
        String, ModelGenerationSummary
    ) {
        try await loadedModel.withModel { _, context in
            let prepared = try await context.processor.prepare(input: UserInput(prompt: prompt))
            let transfer = ExecutionTransfer((context, prepared))
            return try await runModelWorker(policy: .init()) {
                let (context, prepared) = transfer.take()
                var text = ""
                let summary = try PromptCacheEngine.run(
                    context: context, prepared: prepared, tools: nil, options: options,
                    configuration: configuration, vault: vault,
                    emit: {
                        if case .chunk(let chunk) = $0 { text += chunk }
                    })
                return (text, summary)
            }
        }
    }
}

private actor ConversationRecord {
    var prompts: [[String]] = []
    var waiting = false
    var gate: CheckedContinuation<Void, Never>?
    func prepare(_ messages: [String]) async throws {
        prompts.append(messages)
        if messages.last == "WAIT" {
            waiting = true
            await withCheckedContinuation { gate = $0 }
        }
        if messages.last == "FAIL" { throw CocoaError(.fileReadCorruptFile) }
        try Task.checkCancellation()
    }
    func release() {
        gate?.resume()
        gate = nil
    }
}

private struct ConversationProcessor: UserInputProcessor {
    let record: ConversationRecord
    func prepare(input: UserInput) async throws -> LMInput {
        guard case .chat(let messages) = input.prompt else {
            throw CocoaError(.fileReadCorruptFile)
        }
        try await record.prepare(messages.map(\.content))
        let text = messages.map { $0.content + ($0.role == .assistant ? "\u{02}" : "") }.joined()
        let image: LMInput.ProcessedImage? =
            input.images.isEmpty ? nil : .init(pixels: MLXArray.zeros([1, 2, 2, 3]))
        return LMInput(
            text: .init(tokens: MLXArray(text.unicodeScalars.map { Int($0.value) })), image: image)
    }
}

final class ModelSessionTests: XCTestCase {
    private func loadedModel(_ record: ConversationRecord) -> NativModel {
        makeTestModel(
            context: .init(
                metadata: .init(), model: SessionTestModel(cacheKind: .simple),
                processor: ConversationProcessor(record: record), tokenizer: SessionTestTokenizer())
        )
    }
    private func turn(_ session: ModelSession, _ prompt: String) async throws
        -> ModelGenerationSummary
    {
        let request = try await session.respond(to: prompt)
        for try await _ in request {}
        return try await request.metrics()
    }
    private func transcript(_ session: ModelSession) async throws -> [String] {
        try await session.withTranscript { $0.map { "\($0.role.rawValue):\($0.content)" } }
    }

    func testSessionBuildsTranscriptAndReusesItsOwnCache() async throws {
        let record = ConversationRecord()
        let session = ModelSession(model: loadedModel(record), instructions: "rules")
        let first = try await turn(session, "A")
        let second = try await turn(session, "B")
        XCTAssertEqual(first.cache?.kind, .cold)
        XCTAssertEqual(second.cache?.kind, .extend)
        XCTAssertEqual(second.cache?.reusedTokenCount, "rulesA".count + 1)
        let messages = try await transcript(session)
        XCTAssertEqual(messages, ["system:rules", "user:A", "assistant:", "user:B", "assistant:"])
        let prompts = await record.prompts
        XCTAssertEqual(prompts, [["rules", "A"], ["rules", "A", "", "B"]])
        await session.close()
    }

    func testCacheReleaseResetAndTranscriptReplacement() async throws {
        let record = ConversationRecord()
        let session = ModelSession(model: loadedModel(record), instructions: "rules")
        _ = try await turn(session, "A")
        try await session.clearCache()
        let second = try await turn(session, "B")
        XCTAssertEqual(second.cache?.kind, .cold)
        let retained = try await transcript(session)
        XCTAssertEqual(retained.count, 5)
        try await session.replaceTranscript([.user("edited"), .assistant("answer")])
        _ = try await turn(session, "C")
        let prompts = await record.prompts
        XCTAssertEqual(prompts.last, ["rules", "edited", "answer", "C"])
        try await session.reset()
        let cleared = try await transcript(session)
        XCTAssertEqual(cleared, ["system:rules"])
        let afterReset = try await turn(session, "D")
        XCTAssertEqual(afterReset.cache?.kind, .cold)
        await session.close()
    }

    func testFailureRollsBackWholeTurnAndDiscardsCache() async throws {
        let session = ModelSession(model: loadedModel(ConversationRecord()))
        _ = try await turn(session, "A")
        let before = try await transcript(session)
        do {
            _ = try await turn(session, "FAIL")
            XCTFail("Expected preparation failure")
        } catch is CocoaError {}
        let after = try await transcript(session)
        XCTAssertEqual(after, before)
        let retried = try await turn(session, "B")
        XCTAssertEqual(retried.cache?.kind, .cold)
        await session.close()
    }

    func testQueuedTurnUsesCommittedHistoryAndCancelledTurnIsOmitted() async throws {
        let record = ConversationRecord()
        let session = ModelSession(model: loadedModel(record))
        _ = try await turn(session, "A")
        let cancelled = try await session.respond(to: "WAIT")
        for _ in 0 ..< 10_000 {
            if await record.waiting { break }
            await Task.yield()
        }
        let waiting = await record.waiting
        XCTAssertTrue(waiting)
        let next = try await session.respond(to: "B")
        cancelled.cancel()
        await record.release()
        do {
            _ = try await cancelled.metrics()
            XCTFail("Expected cancellation")
        } catch { XCTAssertTrue(error is CancellationError) }
        for try await _ in next {}
        _ = try await next.metrics()
        let prompts = await record.prompts
        XCTAssertEqual(prompts.last, ["A", "", "B"])
        let messages = try await transcript(session)
        XCTAssertEqual(messages, ["user:A", "assistant:", "user:B", "assistant:"])
        await session.close()
    }

    func testConcurrentTurnsCommitInSubmissionOrder() async throws {
        let record = ConversationRecord()
        let session = ModelSession(model: loadedModel(record))
        let first = try await session.respond(to: "A")
        let second = try await session.respond(to: "B")
        for try await _ in first {}
        for try await _ in second {}
        let prompts = await record.prompts
        XCTAssertEqual(prompts, [["A"], ["A", "", "B"]])
        await session.close()
    }

    func testSharedExecutorDoesNotShareConversationAndDefaultsCanChange() async throws {
        let record = ConversationRecord()
        let loadedModel = loadedModel(record)
        let a = ModelSession(model: loadedModel, instructions: "first")
        let b = ModelSession(model: loadedModel, instructions: "second")
        _ = try await turn(a, "A")
        _ = try await turn(b, "B")
        await a.setOptions(.init(output: .init(maxTokens: 0)))
        let zero = try await turn(a, "C")
        XCTAssertEqual(zero.generatedTokenCount, 0)
        let prompts = await record.prompts
        XCTAssertEqual(prompts.last, ["first", "A", "", "C"])
        await a.close()
        _ = try await turn(b, "D")
        let remaining = try await transcript(b)
        XCTAssertEqual(
            remaining, ["system:second", "user:B", "assistant:", "user:D", "assistant:"])
        await b.close()
    }

    func testMediaRetainedInTranscriptUsesColdExecution() async throws {
        let session = ModelSession(model: loadedModel(ConversationRecord()))
        let request = try await session.generate(
            input: UserInput(
                prompt: "image", images: [.array(MLXArray.zeros([2, 2, 3]))]))
        for try await _ in request {}
        let first = try await request.metrics()
        XCTAssertEqual(first.cache?.kind, .bypass)
        let second = try await turn(session, "followup")
        XCTAssertEqual(second.cache?.kind, .bypass)
        let counts = try await session.withTranscript { $0.map { $0.images.count } }
        XCTAssertEqual(counts, [1, 0, 0, 0])
        await session.close()
    }

    func testInvalidCacheSettingsAndRawTokenOutputFailWithoutPublishingHistory() async throws {
        for configuration in [
            PromptCacheReuseConfiguration(maxCheckpoints: -1), .init(checkpointInterval: 0),
        ] {
            let session = ModelSession(
                model: loadedModel(ConversationRecord()), cacheReuse: configuration)
            do {
                _ = try await turn(session, "A")
                XCTFail("Expected invalid cache settings")
            } catch ModelExecutionError.invalidOptions {}
            let messages = try await transcript(session)
            XCTAssertTrue(messages.isEmpty)
            await session.close()
        }
        let session = ModelSession(model: loadedModel(ConversationRecord()))
        let request = try await session.respond(
            to: "A", options: .init(output: .init(maxTokens: 1, format: .tokens)))
        do {
            _ = try await request.metrics()
            XCTFail("Expected text-only session")
        } catch ModelExecutionError.invalidOptions {}
        let messages = try await transcript(session)
        XCTAssertTrue(messages.isEmpty)
        await session.close()
    }
}

final class DraftModelSessionTests: XCTestCase {
    private func model(
        _ record: ConversationRecord, nextToken: Int = 2,
        cacheKind: SessionTestModel.CacheKind = .simple,
        tracksState: Bool = false
    ) -> NativDraftModel {
        let target = ModelContext(
            metadata: .init(),
            model: SessionTestModel(
                cacheKind: cacheKind, tracksState: tracksState, nextToken: nextToken),
            processor: ConversationProcessor(record: record), tokenizer: SessionTestTokenizer())
        let draft = ModelContext(
            metadata: .init(),
            model: SessionTestModel(
                cacheKind: cacheKind, tracksState: tracksState, nextToken: nextToken),
            processor: ConversationProcessor(record: ConversationRecord()),
            tokenizer: SessionTestTokenizer())
        return NativDraftModel(
            target: ExecutionTransfer(target), draft: ExecutionTransfer(draft), policy: .init())
    }

    private func turn(_ session: ModelSession, _ prompt: String) async throws
        -> ModelGenerationSummary
    {
        let request = try await session.respond(to: prompt)
        for try await _ in request {}
        return try await request.metrics()
    }

    private func transcript(_ session: ModelSession) async throws -> [String] {
        try await session.withTranscript { $0.map { "\($0.role.rawValue):\($0.content)" } }
    }

    func testDraftSessionDefaultsBackendAndFullHistory() async throws {
        let record = ConversationRecord()
        let loaded = model(record)
        let session = ModelSession(
            model: loaded, instructions: "rules", history: [.user("initial")],
            speculation: .init(draftTokenCount: 7))
        let loadedBackend = await session.model
        guard case .draft(let backend) = loadedBackend else { return XCTFail("Wrong backend") }
        XCTAssertTrue(backend === loaded)
        let defaults = await session.options
        XCTAssertEqual(defaults.sampling.temperature, 0)
        let speculation = await session.speculation
        let cacheReuse = await session.cacheReuse
        XCTAssertEqual(speculation?.draftTokenCount, 7)
        XCTAssertTrue(cacheReuse.reuseEnabled)
        let first = try await turn(session, "A")
        let second = try await turn(session, "B")
        XCTAssertEqual(first.cache?.kind, .cold)
        XCTAssertEqual(second.cache?.kind, .extend)
        XCTAssertEqual(second.cache?.reusedTokenCount, "rulesinitialA".count)
        XCTAssertEqual(second.speculative?.strategy, .autoregressive)
        let prompts = await record.prompts
        XCTAssertEqual(prompts, [["rules", "initial", "A"], ["rules", "initial", "A", "", "B"]])
        let messages = try await transcript(session)
        XCTAssertEqual(
            messages,
            ["system:rules", "user:initial", "user:A", "assistant:", "user:B", "assistant:"])
        await session.close()
    }

    func testDraftCachesExtendAndRewindBothModels() async throws {
        for kind: SessionTestModel.CacheKind in [.simple, .hybrid] {
            for state in [false, true] {
                let session = ModelSession(
                    model: model(
                        ConversationRecord(), nextToken: 65,
                        cacheKind: kind, tracksState: state), instructions: "rules",
                    options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 9)))
                let first = try await turn(session, "AB")
                XCTAssertEqual(first.generatedTokenCount, 9)
                let second = try await turn(session, "CD")
                XCTAssertEqual(second.cache?.kind, .extend)
                // The bonus token is deliberately re-fed, so the retained prefix
                // includes all but the final generated token of the first turn.
                XCTAssertEqual(second.cache?.reusedTokenCount, "rulesAB".count + 8)
                XCTAssertEqual(second.cache?.prefilledTokenCount, 4)
                try await session.replaceTranscript([.user("AB"), .assistant("edited")])
                let edited = try await turn(session, "EF")
                XCTAssertEqual(edited.cache?.kind, .diverge)
                XCTAssertGreaterThan(edited.cache?.reusedTokenCount ?? 0, 0)
                try await session.clearCache()
                let cleared = try await turn(session, "G")
                XCTAssertEqual(cleared.cache?.kind, .cold)
                try await session.reset()
                let reset = try await turn(session, "H")
                XCTAssertEqual(reset.cache?.kind, .cold)
                await session.close()
            }
        }
    }

    func testDraftExactRegenerationTrimsDeferredProposalWithBothCaches() async throws {
        let session = ModelSession(
            model: model(ConversationRecord(), nextToken: 65),
            options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 9)))
        _ = try await turn(session, "AB")
        try await session.replaceTranscript([])
        let regenerated = try await turn(session, "AB")
        XCTAssertEqual(regenerated.cache?.kind, .exact)
        XCTAssertEqual(regenerated.cache?.reusedTokenCount, 1)
        XCTAssertGreaterThan(regenerated.cache?.trimmedTokenCount ?? 0, 0)
        XCTAssertEqual(regenerated.generatedTokenCount, 9)
        await session.close()
    }

    func testDraftDisabledReuseAndInvalidConfiguration() async throws {
        let disabled = ModelSession(
            model: model(ConversationRecord()),
            cacheReuse: .init(reuseEnabled: false))
        let a = try await turn(disabled, "A")
        let b = try await turn(disabled, "B")
        XCTAssertEqual(a.cache?.kind, .bypass)
        XCTAssertEqual(b.cache?.kind, .bypass)
        await disabled.close()
        let invalid = ModelSession(
            model: model(ConversationRecord()),
            cacheReuse: .init(maxCheckpoints: -1))
        do {
            _ = try await turn(invalid, "A")
            XCTFail("Expected invalid cache configuration")
        } catch ModelExecutionError.invalidOptions {}
        let history = try await transcript(invalid)
        XCTAssertTrue(history.isEmpty)
        await invalid.close()
    }

    func testFailureAndInvalidInputsLeaveTranscriptIntact() async throws {
        let session = ModelSession(model: model(ConversationRecord()), instructions: "rules")
        _ = try await turn(session, "A")
        let before = try await transcript(session)
        do {
            _ = try await turn(session, "FAIL")
            XCTFail("Expected processor failure")
        } catch is CocoaError {}
        for rawDictionary in [false, true] {
            let request: ModelGeneration
            if rawDictionary {
                request = try await session.generate(
                    input: UserInput(messages: [["role": "user", "content": "B"]]))
            } else {
                request = try await session.generate(input: UserInput(chat: []))
            }
            do {
                _ = try await request.metrics()
                XCTFail("Expected invalid input")
            } catch ModelExecutionError.invalidOptions {}
        }
        let raw = try await session.respond(
            to: "B",
            options: .init(
                sampling: .init(temperature: 0),
                output: .init(maxTokens: 1, format: .tokens)))
        do {
            _ = try await raw.metrics()
            XCTFail("Expected text-only output")
        } catch ModelExecutionError.invalidOptions {}
        let after = try await transcript(session)
        XCTAssertEqual(after, before)
        let retried = try await turn(session, "C")
        XCTAssertEqual(retried.cache?.kind, .cold)
        await session.close()
    }

    func testQueuedCancellationAndConcurrentTurnsUseCommittedHistory() async throws {
        let record = ConversationRecord()
        let session = ModelSession(model: model(record))
        _ = try await turn(session, "A")
        let cancelled = try await session.respond(to: "WAIT")
        for _ in 0 ..< 10_000 {
            if await record.waiting { break }
            await Task.yield()
        }
        let waiting = await record.waiting
        XCTAssertTrue(waiting)
        let second = try await session.respond(to: "B")
        let third = try await session.respond(to: "C")
        cancelled.cancel()
        await record.release()
        do {
            _ = try await cancelled.metrics()
            XCTFail("Expected cancellation")
        } catch { XCTAssertTrue(error is CancellationError) }
        for try await _ in second {}
        for try await _ in third {}
        _ = try await third.metrics()
        let messages = try await transcript(session)
        XCTAssertEqual(
            messages, ["user:A", "assistant:", "user:B", "assistant:", "user:C", "assistant:"])
        await session.close()
    }

    func testReplaceClearResetAndIndependentSessions() async throws {
        let record = ConversationRecord()
        let loaded = model(record)
        let a = ModelSession(model: loaded, instructions: "first")
        let b = ModelSession(model: loaded, instructions: "second")
        _ = try await turn(a, "A")
        _ = try await turn(b, "B")
        try await a.clearCache()
        try await a.replaceTranscript([.user("edited"), .assistant("answer")])
        _ = try await turn(a, "C")
        let prompts = await record.prompts
        XCTAssertEqual(prompts.last, ["first", "edited", "answer", "C"])
        await a.setOptions(.init(sampling: .init(temperature: 0), output: .init(maxTokens: 0)))
        let zero = try await turn(a, "D")
        XCTAssertEqual(zero.generatedTokenCount, 0)
        try await a.reset()
        let reset = try await transcript(a)
        XCTAssertEqual(reset, ["system:first"])
        await a.close()
        do {
            _ = try await a.respond(to: "closed")
            XCTFail("Expected closed session")
        } catch ModelExecutionError.sessionClosed {}
        _ = try await turn(b, "E")
        let remaining = try await transcript(b)
        XCTAssertEqual(
            remaining, ["system:second", "user:B", "assistant:", "user:E", "assistant:"])
        await b.close()
        let direct = loaded.generate(input: UserInput(chat: [.user("direct")]))
        for try await _ in direct {}
        _ = try await direct.metrics()
    }

    func testCompletionOverflowDoesNotCommitAndOverrideDoesNotPersist() async throws {
        let session = ModelSession(
            model: model(ConversationRecord(), nextToken: 65),
            options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 4)))
        let overflow = try await session.respond(
            to: "A",
            options: .init(
                sampling: .init(temperature: 0),
                output: .init(maxTokens: 1, bufferCapacity: 1)))
        // Leave the single-slot stream undrained: the completion event must overflow.
        do {
            _ = try await overflow.metrics()
            XCTFail("Expected overflow")
        } catch ModelExecutionError.outputBufferOverflow {}
        let messages = try await transcript(session)
        XCTAssertTrue(messages.isEmpty)
        let response = try await session.respond(to: "B")
        let text = try await response.response()
        XCTAssertEqual(text, "AAAA")
        let metrics = try await response.metrics()
        XCTAssertEqual(metrics.speculative?.acceptedTokenCount, 2)
        let committed = try await transcript(session)
        XCTAssertEqual(committed, ["user:B", "assistant:AAAA"])
        await session.close()
    }

    func testCloseCancelsActiveAndQueuedRequestsAndClearsHistory() async throws {
        let record = ConversationRecord()
        let session = ModelSession(model: model(record))
        _ = try await turn(session, "A")
        let first = try await session.respond(to: "WAIT")
        for _ in 0 ..< 10_000 {
            if await record.waiting { break }
            await Task.yield()
        }
        let next = try await session.respond(to: "B")
        for _ in 0 ..< 10_000 {
            if await session.queuedRequestCount == 1 { break }
            await Task.yield()
        }
        let closing = Task { await session.close() }
        for _ in 0 ..< 10_000 {
            if await session.queuedRequestCount == 0 { break }
            await Task.yield()
        }
        await record.release()
        await closing.value
        for request in [first, next] {
            do {
                _ = try await request.metrics()
                XCTFail("Expected cancellation")
            } catch { XCTAssertTrue(error is CancellationError) }
        }
        let messages = try await transcript(session)
        XCTAssertTrue(messages.isEmpty)
    }
}
