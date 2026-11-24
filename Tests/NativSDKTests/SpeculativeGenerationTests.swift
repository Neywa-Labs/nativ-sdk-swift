import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private final class TrackingSpeculativeCache: KVCacheSimple {
    var copyCount = 0
    var trimmedCount = 0
    override func copy() -> any KVCache {
        copyCount += 1
        return super.copy()
    }
    override func trim(_ n: Int) -> Int {
        let count = super.trim(n)
        trimmedCount += count
        return count
    }
}

private final class CountingSpeculativeModel: Module, SpeculativeDecodingModel {
    enum CacheKind { case simple, rotating, recurrent, unsupported }
    let kind: CacheKind
    let wrongAt: Set<Int>
    var forwardCount = 0
    var forwardLengths: [Int] = []
    var simpleCache = TrackingSpeculativeCache()
    init(kind: CacheKind = .simple, wrongAt: Set<Int> = []) {
        self.kind = kind
        self.wrongAt = wrongAt
    }
    func newCache(configuration: KVCacheConfiguration?) -> [KVCache] {
        switch kind {
        case .simple:
            simpleCache = TrackingSpeculativeCache()
            return [simpleCache]
        case .rotating: return [RotatingKVCache(maxSize: 4, keep: 1)]
        case .recurrent: return [ArraysCache(size: 1)]
        case .unsupported: return [ChunkedKVCache()]
        }
    }
    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult { .tokens(input.text) }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        forwardCount += 1
        let tokens = inputs.asArray(Int.self)
        forwardLengths.append(tokens.count)
        var position = cache![0].offset
        if let recurrent = cache![0] as? ArraysCache {
            position = recurrent[0]?.item(Int.self) ?? 0
        }
        var values: [Float] = []
        for token in tokens {
            position += 1
            let next = (token + position + (wrongAt.contains(position) ? 1 : 0)) % 31 + 1
            values += (0 ..< 33).map { $0 == next ? Float(10) : -10 }
        }
        if let recurrent = cache![0] as? ArraysCache {
            recurrent[0] = MLXArray(position)
        } else {
            let kv = MLXArray(tokens.map(Float.init), [1, 1, tokens.count, 1])
            _ = cache![0].update(keys: kv, values: kv)
        }
        return MLXArray(values, [1, tokens.count, 33])
    }
}

private class StatefulSpeculativeFixture: Module, SpeculativeDecodingModel {
    static let position = LMOutput.Key<MLXArray>("position")
    let wrong: Bool
    init(wrong: Bool = false) { self.wrong = wrong }
    func newCache(configuration: KVCacheConfiguration?) -> [KVCache] { [] }
    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult { .tokens(input.text) }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(.init(tokens: inputs), cache: cache, state: nil).logits
    }
    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        var position = state?[Self.position]?.item(Int.self) ?? 0
        var values: [Float] = []
        let tokens = input.tokens.asArray(Int.self)
        for token in tokens {
            position += 1
            let next = (position + token + (wrong ? 1 : 0)) % 15
            values += (0 ..< 16).map { $0 == next ? Float(10) : -10 }
        }
        var next = LMOutput.State()
        next[Self.position] = MLXArray(position)
        return LMOutput(logits: MLXArray(values, [1, tokens.count, 16]), state: next)
    }
}
private final class CopyingSpeculativeFixture: StatefulSpeculativeFixture,
    ContinuationStateCopying
{
    func copyContinuationState(_ state: LMOutput.State) throws -> LMOutput.State {
        var copy = LMOutput.State()
        copy[Self.position] = state[Self.position]?[.ellipsis]
        if let position = copy[Self.position] { eval(position) }
        return copy
    }
}

private final class LateStateSpeculativeFixture: Module, SpeculativeDecodingModel,
    ContinuationStateCopying
{
    let wrong: Bool
    init(wrong: Bool = false) { self.wrong = wrong }
    func newCache(configuration: KVCacheConfiguration?) -> [KVCache] { [KVCacheSimple()] }
    func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult { .tokens(input.text) }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(.init(tokens: inputs), cache: cache, state: nil).logits
    }
    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let cache = cache![0]
        let introduceState = cache.offset >= 3
        var position = state?[StatefulSpeculativeFixture.position]?.item(Int.self) ?? cache.offset
        let tokens = input.tokens.asArray(Int.self)
        var values: [Float] = []
        for token in tokens {
            position += 1
            let next = (position + token + (wrong ? 1 : 0)) % 15
            values += (0 ..< 16).map { $0 == next ? Float(10) : -10 }
        }
        let kv = MLXArray(tokens.map(Float.init), [1, 1, tokens.count, 1])
        _ = cache.update(keys: kv, values: kv)
        var next: LMOutput.State?
        if introduceState {
            next = LMOutput.State()
            next![StatefulSpeculativeFixture.position] = MLXArray(position)
        }
        return LMOutput(logits: MLXArray(values, [1, tokens.count, 16]), state: next)
    }
    func copyContinuationState(_ state: LMOutput.State) throws -> LMOutput.State {
        var next = LMOutput.State()
        next[StatefulSpeculativeFixture.position] =
            state[StatefulSpeculativeFixture.position]?[.ellipsis]
        return next
    }
}

final class SpeculativeGenerationTests: XCTestCase {
    private let options = ModelGenerationOptions(
        sampling: .init(temperature: 0), output: .init(maxTokens: 12, format: .tokens))
    private func drain(_ iterator: SpeculativeIterator) -> [Int] {
        var tokens: [Int] = []
        while let token = iterator.next() { tokens.append(token) }
        return tokens
    }
    func testPairedCheckpointsResumeAfterFullAcceptanceAndMidBlockStops() throws {
        let pairs: [(any LanguageModel, any LanguageModel)] = [
            (CountingSpeculativeModel(), CountingSpeculativeModel()),
            (CountingSpeculativeModel(kind: .rotating), CountingSpeculativeModel(kind: .rotating)),
            (
                CountingSpeculativeModel(kind: .recurrent),
                CountingSpeculativeModel(kind: .recurrent)
            ),
            (LateStateSpeculativeFixture(), LateStateSpeculativeFixture()),
        ]
        for (target, draft) in pairs {
            for deliveredLimit in [1, 2, 5, 12] {
                let prompt = [1, 2, 3]
                var checkpoint: DraftCacheCheckpoint?
                let iterator = try SpeculativeIterator(
                    input: LMInput(tokens: MLXArray(prompt)),
                    target: target, draft: draft, options: options, speculation: .init(),
                    checkpoint: { checkpoint = $0 })
                var delivered: [Int] = []
                for _ in 0 ..< deliveredLimit { delivered.append(try XCTUnwrap(iterator.next())) }
                XCTAssertNil(iterator.failure)
                let saved = try XCTUnwrap(checkpoint)
                let nextPrompt = prompt + delivered + [4, 5]
                XCTAssertTrue(saved.tokens.elementsEqual(nextPrompt.prefix(saved.tokens.count)))
                let resumed = try SpeculativeIterator(
                    input: LMInput(
                        tokens: MLXArray(Array(nextPrompt.dropFirst(saved.tokens.count)))),
                    target: target, draft: draft, options: options, speculation: .init(),
                    reuse: saved)
                let cold = try SpeculativeIterator(
                    input: LMInput(tokens: MLXArray(nextPrompt)),
                    target: target, draft: draft, options: options, speculation: .init())
                XCTAssertEqual(
                    drain(resumed), drain(cold), "\(type(of: target)), limit=\(deliveredLimit)")
                XCTAssertNil(resumed.failure)
                XCTAssertNil(cold.failure)
            }
        }
    }

    func testQwen3CausalChunksMatchSingleTokenForwards() throws {
        for quantized in [false, true] {
            let model = try qwen3(quantized: quantized)
            let prompt = MLXArray([1, 2, 3], [1, 3])
            let tokens = [4, 5, 6, 7, 8]
            let chunkCache = try model.newCache(configuration: nil)
            let sequentialCache = try model.newCache(configuration: nil)
            eval(model(prompt, cache: chunkCache), model(prompt, cache: sequentialCache))
            let chunk = model(MLXArray(tokens, [1, tokens.count]), cache: chunkCache)
            let sequential = concatenated(
                tokens.map { model(MLXArray([$0], [1, 1]), cache: sequentialCache) }, axis: 1
            )
            // Batched and single-token GPU kernels can differ slightly in rounding.
            XCTAssertLessThan(abs(chunk - sequential).max().item(Float.self), 0.01)
            XCTAssertEqual(
                argMax(chunk, axis: -1).asArray(Int.self),
                argMax(sequential, axis: -1).asArray(Int.self)
            )
            XCTAssertEqual(chunkCache.map(\.offset), sequentialCache.map(\.offset))
        }
    }

    func testQwen3SpeculationMatchesGreedyWithQuantizedWeightsAndRotatingCaches() throws {
        for quantized in [false, true] {
            let target = try qwen3(quantized: quantized)
            let draft = try qwen3(quantized: quantized)
            draft.update(parameters: target.parameters())
            for bounded in [false, true] {
                var options = options
                if bounded {
                    options.cache = try .init(
                        capacity: .init(maxTokens: 6, preservedPrefixTokens: 1)
                    )
                }
                let input = LMInput(tokens: MLXArray([1, 2, 3]))
                let expected = try Array(
                    AutoregressiveIterator(input: input, model: target, options: options)
                )
                let iterator = try SpeculativeIterator(
                    input: input, target: target, draft: draft, options: options,
                    speculation: .init(draftTokenCount: 4)
                )
                var actual: [Int] = []
                while let token = iterator.next() {
                    actual.append(token)
                }
                XCTAssertNil(iterator.failure)
                XCTAssertEqual(actual, expected, "quantized=\(quantized), bounded=\(bounded)")
                XCTAssertEqual(
                    iterator.statistics.acceptedTokenCount, iterator.statistics.proposedTokenCount
                )
            }
        }
    }

    private func qwen3(quantized: Bool) throws -> Qwen3Model {
        let json =
            #"{"hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"vocab_size":64,"rms_norm_eps":0.000001,"max_position_embeddings":128,"tie_word_embeddings":false}"#
        let config = try JSONDecoder().decode(Qwen3Configuration.self, from: Data(json.utf8))
        let model = withRandomState(MLXRandom.RandomState(seed: 1)) { Qwen3Model(config) }
        if quantized { quantize(model: model, groupSize: 32, bits: 4) }
        eval(model.parameters())
        return model
    }

    func testAcceptanceAndRollbackMatchGreedyForSimpleRotatingAndRecurrentCaches() throws {
        for kind in [CountingSpeculativeModel.CacheKind.simple, .rotating, .recurrent] {
            for wrongAt: Set<Int> in [[], [4], [5], Set(1 ... 30)] {
                let input = LMInput(tokens: MLXArray([1, 2, 3]))
                let target = CountingSpeculativeModel(kind: kind)
                let expected = Array(
                    try AutoregressiveIterator(input: input, model: target, options: options))
                let iterator = try SpeculativeIterator(
                    input: input, target: target,
                    draft: CountingSpeculativeModel(kind: kind, wrongAt: wrongAt), options: options,
                    speculation: .init(draftTokenCount: 4))
                var actual: [Int] = []
                while let token = iterator.next() { actual.append(token) }
                XCTAssertNil(iterator.failure)
                XCTAssertEqual(actual, expected, "\(kind), \(wrongAt)")
                if wrongAt.isEmpty {
                    XCTAssertEqual(
                        iterator.statistics.acceptedTokenCount,
                        iterator.statistics.proposedTokenCount)
                } else {
                    XCTAssertLessThan(
                        iterator.statistics.acceptedTokenCount,
                        iterator.statistics.proposedTokenCount)
                }
                XCTAssertGreaterThan(iterator.statistics.verificationCount, 0)
            }
        }
    }
    func testAppendOnlyRoundsTrimWithoutSnapshotsOrTargetReplay() throws {
        var options = options
        options.output.maxTokens = 6
        let input = LMInput(tokens: MLXArray([1, 2, 3]))
        let target = CountingSpeculativeModel()
        let draft = CountingSpeculativeModel(wrongAt: [4])
        let iterator = try SpeculativeIterator(
            input: input, target: target, draft: draft, options: options,
            speculation: .init(draftTokenCount: 4))
        let actual = drain(iterator)
        let expected = Array(
            try AutoregressiveIterator(
                input: input, model: CountingSpeculativeModel(), options: options))
        XCTAssertNil(iterator.failure)
        XCTAssertEqual(actual, expected)
        XCTAssertEqual(target.forwardLengths, [3, 5, 4])
        XCTAssertEqual(draft.forwardLengths, [3, 1, 1, 1, 1, 1, 1, 1])
        XCTAssertEqual(target.simpleCache.copyCount, 0)
        XCTAssertEqual(draft.simpleCache.copyCount, 0)
        XCTAssertEqual(target.simpleCache.trimmedCount, 4)
        XCTAssertEqual(draft.simpleCache.trimmedCount, 3)
        XCTAssertEqual(target.simpleCache.keys?.dim(2), 256)
        XCTAssertEqual(draft.simpleCache.keys?.dim(2), 256)
        XCTAssertEqual(iterator.statistics.verificationCount, 2)
    }

    func testFullAcceptanceDefersLastDraftTokenIntoNextForward() throws {
        let target = CountingSpeculativeModel()
        // A disagreement at prefill does not matter: the target supplies the seed.
        let draft = CountingSpeculativeModel(wrongAt: [3])
        let iterator = try SpeculativeIterator(
            input: LMInput(tokens: MLXArray([1, 2, 3])), target: target, draft: draft,
            options: options, speculation: .init(draftTokenCount: 4))
        XCTAssertEqual(drain(iterator).count, 12)
        XCTAssertNil(iterator.failure)
        XCTAssertEqual(target.forwardLengths, [3, 5, 5, 1])
        XCTAssertEqual(draft.forwardLengths, [3, 1, 1, 1, 1, 2, 1, 1, 1])
        XCTAssertEqual(target.simpleCache.copyCount, 0)
        XCTAssertEqual(draft.simpleCache.copyCount, 0)
        XCTAssertEqual(target.simpleCache.trimmedCount, 0)
        XCTAssertEqual(draft.simpleCache.trimmedCount, 0)
        XCTAssertEqual(iterator.statistics.acceptedTokenCount, 8)
    }

    func testUnsupportedOptionsAndCachesFailBeforeForward() throws {
        let target = CountingSpeculativeModel()
        let draft = CountingSpeculativeModel()
        var invalid = options
        invalid.sampling.temperature = 1
        XCTAssertThrowsError(
            try SpeculativeIterator(
                input: LMInput(tokens: MLXArray([1])), target: target, draft: draft,
                options: invalid, speculation: .init()))
        XCTAssertEqual(target.forwardCount, 0)
        XCTAssertEqual(draft.forwardCount, 0)
        let unsupported = CountingSpeculativeModel(kind: .unsupported)
        XCTAssertThrowsError(
            try SpeculativeIterator(
                input: LMInput(tokens: MLXArray([1])), target: unsupported, draft: draft,
                options: options, speculation: .init()))
        XCTAssertEqual(unsupported.forwardCount, 0)
        XCTAssertThrowsError(
            try SpeculativeIterator(
                input: LMInput(tokens: MLXArray([1])), target: target, draft: unsupported,
                options: options, speculation: .init()))
        XCTAssertEqual(target.forwardCount, 0)

        XCTAssertThrowsError(
            try SpeculativeIterator(
                input: LMInput(tokens: MLXArray([1])), target: target, draft: target,
                options: options, speculation: .init()))
    }
    func testOutOfCacheStateRequiresExplicitCopyAndSurvivesRollback() throws {
        let input = LMInput(tokens: MLXArray([1, 2]))
        XCTAssertThrowsError(
            try SpeculativeIterator(
                input: input, target: StatefulSpeculativeFixture(),
                draft: CopyingSpeculativeFixture(), options: options, speculation: .init())
        ) {
            XCTAssertEqual($0 as? ContinuationError, .unsupportedModelState)
        }
        let target = CopyingSpeculativeFixture()
        let expected = Array(
            try AutoregressiveIterator(input: input, model: target, options: options))
        let iterator = try SpeculativeIterator(
            input: input, target: target, draft: CopyingSpeculativeFixture(wrong: true),
            options: options, speculation: .init())
        var actual: [Int] = []
        while let token = iterator.next() { actual.append(token) }
        XCTAssertNil(iterator.failure)
        XCTAssertEqual(actual, expected)
        XCTAssertLessThan(
            iterator.statistics.acceptedTokenCount, iterator.statistics.proposedTokenCount)
    }

    func testStateFirstIntroducedDuringVerificationSurvivesRollback() throws {
        let input = LMInput(tokens: MLXArray([1, 2, 3]))
        let expected = Array(
            try AutoregressiveIterator(
                input: input, model: LateStateSpeculativeFixture(), options: options))
        let iterator = try SpeculativeIterator(
            input: input, target: LateStateSpeculativeFixture(),
            draft: LateStateSpeculativeFixture(wrong: true), options: options,
            speculation: .init())
        XCTAssertEqual(drain(iterator), expected)
        XCTAssertNil(iterator.failure)
        XCTAssertLessThan(
            iterator.statistics.acceptedTokenCount, iterator.statistics.proposedTokenCount)
    }

    func testZeroAndShortTokenLimits() throws {
        for count in [0, 1, 2, 4, 5] {
            var options = options
            options.output.maxTokens = count
            let iterator = try SpeculativeIterator(
                input: LMInput(tokens: MLXArray([1, 2])),
                target: CountingSpeculativeModel(), draft: CountingSpeculativeModel(),
                options: options,
                speculation: .init(draftTokenCount: 4))
            var tokens: [Int] = []
            while let token = iterator.next() { tokens.append(token) }
            XCTAssertNil(iterator.failure)
            XCTAssertEqual(tokens.count, count)
        }
    }
    func testOwnedSpeculationStopsAndPublishesStatisticsAfterCleanup() async throws {
        let directory = URL(fileURLWithPath: "/tmp/speculative-fixture")
        func loaded(_ wrong: Set<Int> = [], eos: Set<Int> = [14]) -> LoadedModel {
            LoadedModel(
                assets: .init(
                    source: .directory(directory), modelDirectory: directory,
                    tokenizerDirectory: directory),
                metadata: .init(eosTokenIds: eos),
                model: CountingSpeculativeModel(wrongAt: wrong),
                processor: ExecutionFixtureProcessor(), tokenizer: ExecutionFixtureTokenizer())
        }
        let loadedModel = try loaded().intoSpeculativeExecutor(draft: loaded([4]))
        let request = loadedModel.generate(input: UserInput(prompt: "hello"), options: options)
        var tokens: [Int] = []
        var completions = 0
        for try await event in request {
            switch event {
            case .token(let token): tokens.append(token)
            case .completion: completions += 1
            default: break
            }
        }
        let result = try await request.metrics()
        XCTAssertEqual(tokens, [5, 9])
        XCTAssertEqual(result.stopReason, .stop)
        XCTAssertEqual(completions, 1)
        XCTAssertNotNil(result.speculative)
        let immediatelyStopped = try loaded(eos: [5]).intoSpeculativeExecutor(draft: loaded())
        let stopped = try await immediatelyStopped.generate(
            input: UserInput(prompt: "hello"), options: options
        ).metrics()
        XCTAssertEqual(stopped.generatedTokenCount, 0)
        XCTAssertEqual(stopped.stopReason, .stop)
        XCTAssertEqual(stopped.speculative?.proposedTokenCount, 0)
        XCTAssertEqual(stopped.speculative?.verificationCount, 0)
        let cancelled = loadedModel.generate(input: UserInput(prompt: "hello"), options: options)
        cancelled.cancel()
        do {
            _ = try await cancelled.metrics()
            XCTFail("Expected cancellation")
        } catch { XCTAssertTrue(error is CancellationError) }
    }

    func testTokenizerIdentityRequiresMatchingAssets() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let a = root.appendingPathComponent("a")
        let b = root.appendingPathComponent("b")
        for directory in [a, b] {
            try FileManager.default.createDirectory(
                at: directory, withIntermediateDirectories: true)
        }
        XCTAssertThrowsError(try validateSpeculativeTokenizers(a, b))
        for directory in [a, b] {
            try Data("vocabulary".utf8).write(
                to: directory.appendingPathComponent("tokenizer.json"))
        }
        XCTAssertNoThrow(try validateSpeculativeTokenizers(a, b))
        try Data("different".utf8).write(to: b.appendingPathComponent("tokenizer.json"))
        XCTAssertThrowsError(try validateSpeculativeTokenizers(a, b))
    }
}
