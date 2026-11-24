import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private final class GenerationFixture: Module, LLMModel {
    static let position = LMOutput.Key<Int>("position")
    var loraLayers: [Module] { [] }
    var forwards: [Int] = []
    var fixedToken: Int?
    init(fixedToken: Int? = nil) { self.fixedToken = fixedToken }
    func newCache(configuration: KVCacheConfiguration?) -> [KVCache] { [KVCacheSimple()] }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        callAsFunction(.init(tokens: inputs), cache: cache, state: nil).logits
    }
    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let length = input.tokens.dim(-1)
        forwards.append(length)
        let start = state?[Self.position] ?? 0
        for entry in cache ?? [] {
            let keys = MLXArray.zeros([1, 1, length, 64])
            if let quantized = entry as? any QuantizedKVCacheProtocol {
                _ = quantized.updateQuantized(keys: keys, values: keys)
            } else {
                _ = entry.update(keys: keys, values: keys)
            }
        }
        var values = Array(repeating: Float(-5), count: length * 8)
        for index in 0 ..< length {
            values[index * 8 + (fixedToken ?? ((start + index + 1) % 8))] = 5
        }
        var outputState = LMOutput.State()
        outputState[Self.position] = start + length
        return LMOutput(logits: MLXArray(values, [1, length, 8]), state: outputState)
    }
}

private struct GenerationTokenizer: Tokenizer {
    var pieces: [Int: String] = [:]
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1, 2] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { pieces[$0] ?? String($0) }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { token == "extra-stop" ? 3 : nil }
    func convertIdToToken(_ id: Int) -> String? { String(id) }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [1, 2] }
}

private struct GenerationProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput { LMInput(tokens: MLXArray([1, 2])) }
}

private final class PrefillRecord: @unchecked Sendable {
    private let lock = NSLock()
    private var values: [(Int, Int)] = []
    func append(_ done: Int, _ total: Int) { lock.withLock { values.append((done, total)) } }
    var completed: [Int] { lock.withLock { values.map(\.0) } }
}

final class AutoregressiveGenerationTests: XCTestCase {
    func testCollectedResponseMatchesStreamingAndCommitsSessionHistory() async throws {
        let options = ModelGenerationOptions(
            sampling: .init(temperature: 0), output: .init(maxTokens: 128, bufferCapacity: 16))
        let streamed = loadedModel().generate(input: UserInput(prompt: "test"), options: options)
        var expected = ""
        for try await event in streamed {
            if case .chunk(let text) = event { expected += text }
        }
        let session = ModelSession(model: loadedModel(), options: options)
        let collected = try await session.respond(to: "test")
        let text = try await collected.response()
        XCTAssertEqual(text, expected)
        XCTAssertEqual(text, String(repeating: "3", count: 128))
        let metrics = try await collected.metrics()
        XCTAssertEqual(metrics.generatedTokenCount, 128)
        let transcript = try await session.withTranscript { $0.map(\.content) }
        XCTAssertEqual(transcript, ["test", text])
        await session.close()
    }

    func testOpaqueContinuationStateRequiresAnExplicitCopyAdapter() throws {
        let model = GenerationFixture()
        let scope = ContinuationScope(
            model: model,
            processorRevision: "fixture", configurationRevision: "v1")
        XCTAssertThrowsError(
            try ContinuationSnapshot(
                model: model,
                caches: model.newCache(configuration: nil), state: .init(), tokens: [], scope: scope
            )
        ) {
            XCTAssertEqual($0 as? ContinuationError, .unsupportedModelState)
        }
    }

    func testRawEnginePreservesStopTokenPolicyAndCompression() async throws {
        for includeStop in [false, true] {
            let context = ModelContext(
                metadata: .init(eosTokenIds: [3]),
                model: GenerationFixture(fixedToken: 3), processor: GenerationProcessor(),
                tokenizer: GenerationTokenizer())
            let input = LMInput(tokens: MLXArray([1, 2]))
            let options = ModelGenerationOptions(
                cache: .init(strategy: .affine(try .init(bits: 4, groupSize: 32))),
                sampling: .init(temperature: 0),
                output: .init(maxTokens: 3, format: .tokens))
            let iterator = try AutoregressiveIterator(
                input: input, model: context.languageModel, options: options)
            var tokens: [Int] = []
            let info = try AutoregressiveEngine.consume(
                iterator, input: input, context: context, tools: nil,
                options: options, includeStopToken: includeStop
            ) { if case .token(let token) = $0 { tokens.append(token) } }
            XCTAssertEqual(tokens, includeStop ? [3] : [])
            XCTAssertEqual(info.generatedTokenCount, tokens.count)
            XCTAssertEqual(info.stopReason, .stop)
        }
    }

    func testSeededSamplersRespectFiltersDistributionAndIndependentRNG() {
        let values: [Float] = [-3, -1, 0, 1, 2, 3, 4, 5]
        for dtype in [DType.float32, .bfloat16] {
            let logits = MLXArray(values, [1, 8]).asType(dtype)
            let options = ModelSamplingOptions(
                temperature: 0.7, topP: 0.9, topK: 3, minP: 0.05, seed: 42)
            let owned = options.makeSampler()
            let independent = options.makeSampler()
            for _ in 0 ..< 32 {
                let token = owned.sample(logits: logits).item(Int.self)
                XCTAssertEqual(token, independent.sample(logits: logits).item(Int.self))
                XCTAssertTrue((5 ... 7).contains(token))
            }
        }
        let logits = MLXArray(values, [1, 8])
        XCTAssertEqual(ArgMaxSampler().sample(logits: logits).item(Int.self), 7)
        let owned = CategoricalSampler(temperature: 0.8, seed: 12)
        let independent = CategoricalSampler(temperature: 0.8, seed: 12)
        for _ in 0 ..< 24 {
            XCTAssertEqual(
                owned.sample(logits: logits).item(Int.self),
                independent.sample(logits: logits).item(Int.self))
        }
        let probabilities: [Float] = [0.1, 0.3, 0.6]
        let population = broadcast(log(MLXArray(probabilities)), to: [4096, 3])
        let samples = CategoricalSampler(temperature: 1, seed: 99)
            .sample(logits: population).asArray(Int.self)
        for (token, probability) in probabilities.enumerated() {
            let observed = Float(samples.filter { $0 == token }.count) / Float(samples.count)
            XCTAssertEqual(observed, probability, accuracy: 0.025)
        }
    }

    func testPenaltyOrderCountsAndRingRolloverMatchPinnedFixture() throws {
        let options = ModelSamplingOptions(
            repetition: .init(1.2, contextSize: 4),
            presence: .init(0.3, contextSize: 3), frequency: .init(0.7, contextSize: 5))
        var owned = options.makeProcessor()
        let prompt = MLXArray([1, 1, 2, 3, 4, 1, 2])
        owned.prompt(prompt)
        let logits = MLXArray([Float(-2), -1, 0, 1, 2, 3, 4, 5], [1, 8])
        for (index, token) in [2, 2, 3, 0, 7, 6, 5, 5, 4].enumerated() {
            XCTAssertTrue(
                allClose(
                    owned.process(logits: logits),
                    try RuntimeReferenceFixtures.tensor("penalty", index: index, shape: [1, 8])
                ).item(
                    Bool.self))
            owned.didSample(token: MLXArray([token]))
        }
    }

    func testGreedyIterationCachesAndWarmStateMatchPinnedFixtures() throws {
        for compressed in [false, true] {
            let options = ModelGenerationOptions(
                cache: compressed
                    ? .init(
                        strategy: .affine(try .init(bits: 4, groupSize: 32, compressionStart: 4)))
                    : nil,
                sampling: .init(temperature: 0), output: .init(maxTokens: 4))
            let model = GenerationFixture()
            let input = LMInput(tokens: MLXArray([1, 2, 3]))
            var state = LMOutput.State()
            state[GenerationFixture.position] = 10
            let owned = try AutoregressiveIterator(
                input: input, model: model, state: state, options: options)
            let tokens = Array(owned)
            XCTAssertEqual(
                tokens, try RuntimeReferenceFixtures.values("greedy-\(compressed)", as: [Int].self))
            XCTAssertEqual(tokens.count, 4)
            XCTAssertNil(owned.failure)
            XCTAssertEqual(owned.state?[GenerationFixture.position], 17)
            XCTAssertEqual(owned.cache[0].offset, 7)
            XCTAssertEqual(owned.cache[0] is QuantizedKVCache, compressed)
            if !compressed {
                let suffix = LMInput(tokens: MLXArray([4, 5]))
                let continued = try AutoregressiveIterator(
                    input: suffix, model: model, cache: owned.cache,
                    state: owned.state, options: options)
                XCTAssertEqual(
                    Array(continued), try RuntimeReferenceFixtures.values("warm", as: [Int].self))
            }
        }
        Stream.defaultStream.synchronize()
    }

    func testChunkedPrefillPreservesContinuationStateAndProgress() throws {
        for (chunking, expected): (ModelPrefillOptions.Chunking, [Int]) in [
            (.balanced, [3, 3, 1]), (.remainder, [3, 3, 1]), (.unchunked, [7]),
        ] {
            let model = GenerationFixture()
            let progress = PrefillRecord()
            var state = LMOutput.State()
            state[GenerationFixture.position] = 10
            let iterator = try AutoregressiveIterator(
                input: LMInput(tokens: MLXArray([1, 2, 3, 4, 5, 6, 7])),
                model: model, state: state,
                options: .init(
                    prefill: .init(
                        stepSize: 3, chunking: chunking,
                        progress: { progress.append($0, $1) }), sampling: .init(temperature: 0),
                    output: .init(maxTokens: 1)))
            XCTAssertEqual(model.forwards, expected)
            XCTAssertEqual(iterator.state?[GenerationFixture.position], 17)
            XCTAssertEqual(progress.completed.last, 7)
            XCTAssertEqual(progress.completed.filter { $0 == 7 }.count, 1)
            XCTAssertEqual(iterator.next(), 1)
            XCTAssertEqual(iterator.state?[GenerationFixture.position], 18)
        }
        Stream.defaultStream.synchronize()
    }

    func testPlatformPrefillDefaultAndExplicitOverrides() throws {
        #if os(iOS)
            let defaultForwards = [256, 256, 88]
        #else
            let defaultForwards = [512, 88]
        #endif
        for (stepSize, expected): (Int?, [Int]) in [
            (nil, defaultForwards), (256, [256, 256, 88]), (512, [512, 88]),
        ] {
            let model = GenerationFixture()
            let progress = PrefillRecord()
            let iterator = try AutoregressiveIterator(
                input: LMInput(tokens: MLXArray(Array(repeating: 1, count: 600))),
                model: model,
                options: .init(
                    prefill: .init(
                        stepSize: stepSize, chunking: .remainder,
                        progress: { progress.append($0, $1) }),
                    sampling: .init(temperature: 0), output: .init(maxTokens: 1)))
            XCTAssertEqual(model.forwards, expected)
            XCTAssertEqual(iterator.cache[0].offset, 600)
            XCTAssertEqual(iterator.state?[GenerationFixture.position], 600)
            XCTAssertEqual(progress.completed.last, 600)
            XCTAssertEqual(iterator.next(), 0)
        }
        Stream.defaultStream.synchronize()
    }

    func testChunkDriverRanges() throws {
        for (chunking, expected): (ModelPrefillOptions.Chunking, [Range<Int>]) in [
            (.balanced, [0 ..< 4, 4 ..< 8, 8 ..< 12, 12 ..< 16]),
            (.remainder, [0 ..< 4, 4 ..< 8, 8 ..< 12, 12 ..< 16]),
            (.unchunked, []),
        ] {
            let parameters = ModelPrefillOptions(stepSize: 4, chunking: chunking)
            var actual: [Range<Int>] = []
            let count = try parameters.forEachChunk(total: 17) { actual.append($0) }
            XCTAssertEqual(actual, expected)
            XCTAssertEqual(count, expected.last?.upperBound ?? 0)
            for total in [0, 1] {
                XCTAssertEqual(try parameters.forEachChunk(total: total) { _ in XCTFail() }, 0)
            }
        }
        var balanced: [Range<Int>] = []
        XCTAssertEqual(
            try ModelPrefillOptions(stepSize: 4).forEachChunk(total: 10) { balanced.append($0) }, 9)
        XCTAssertEqual(balanced, [0 ..< 3, 3 ..< 6, 6 ..< 9])
    }

    func testTokenSuppressionAndInvalidMask() throws {
        let model = GenerationFixture(fixedToken: 3)
        let options = ModelGenerationOptions(
            sampling: .init(temperature: 0, suppressedTokenIDs: [3]),
            output: .init(maxTokens: 3))
        let iterator = try AutoregressiveIterator(
            input: LMInput(tokens: MLXArray([1, 2])), model: model, options: options)
        while let token = iterator.next() { XCTAssertNotEqual(token, 3) }
        for mask: Set<Int> in [[-1], [9], Set(0 ..< 8)] {
            var invalid = options
            invalid.sampling.suppressedTokenIDs = mask
            XCTAssertThrowsError(
                try AutoregressiveIterator(
                    input: LMInput(tokens: MLXArray([1, 2])),
                    model: GenerationFixture(), options: invalid))
        }
        Stream.defaultStream.synchronize()
    }

    func testCancellationBetweenPrefillChunks() async throws {
        let record = PrefillRecord()
        let task = Task {
            try ModelPrefillOptions(stepSize: 2).forEachChunk(total: 20) { range in
                record.append(range.upperBound, 20)
                withUnsafeCurrentTask { $0?.cancel() }
            }
        }
        do {
            _ = try await task.value
            XCTFail("Cancellation must prevent a second chunk")
        } catch is CancellationError {
            XCTAssertEqual(record.completed, [2])
        }
    }

    func testTextAndToolEventOrderMatchesPinnedFixture() async throws {
        let tokenizer = GenerationTokenizer(pieces: [
            2: "before ", 3: "<tool_call>",
            4: #"{"name":"weather","arguments":{"city":"Paris"}}"#,
            5: "</tool_call>", 6: " after",
        ])
        let schemas: [[String: any Sendable]] = [
            [
                "type": "function",
                "function": [
                    "name": "weather",
                    "parameters": [
                        "type": "object", "properties": ["city": ["type": "string"]],
                        "required": ["city"],
                    ] as [String: any Sendable],
                ] as [String: any Sendable],
            ]
        ]
        let context = ModelContext(
            metadata: .init(toolCallFormat: .json),
            model: GenerationFixture(), processor: GenerationProcessor(), tokenizer: tokenizer)
        let input = LMInput(tokens: MLXArray([1, 2]))
        let options = ModelGenerationOptions(
            sampling: .init(temperature: 0), output: .init(maxTokens: 5))
        let captured: [[String]] = try RuntimeReferenceFixtures.values("events")
        let expected = captured[0]
        let iterator = try AutoregressiveIterator(
            input: input, model: context.languageModel, options: options)
        var actual: [String] = []
        let summary = try AutoregressiveEngine.consume(
            iterator, input: input, context: context, tools: schemas, options: options
        ) { event in
            switch event {
            case .chunk(let text): actual.append("text:" + text)
            case .toolCall(let call): actual.append("tool:" + call.function.name)
            case .rejectedToolCall: actual.append("rejected")
            case .token, .completion, .reasoning, .protocolError: break
            }
        }
        XCTAssertEqual(actual, expected)
        XCTAssertEqual(actual, ["text:before ", "tool:weather", "text: after"])
        XCTAssertEqual(summary.generatedTokenCount, 5)
        XCTAssertEqual(summary.rejectedToolCallCount, 0)
        Stream.defaultStream.synchronize()
    }

    func testTypedCachePolicyCompressesDuringGeneration() throws {
        let options = ModelGenerationOptions(
            cache: .init(strategy: .affine(try .init(bits: 4, groupSize: 32))),
            sampling: .init(temperature: 0),
            output: .init(maxTokens: 2))
        let iterator = try AutoregressiveIterator(
            input: LMInput(tokens: MLXArray([1, 2])), model: GenerationFixture(), options: options)
        XCTAssertEqual(Array(iterator).count, 2)
        XCTAssertTrue(iterator.cache[0] is QuantizedKVCache)
        Stream.defaultStream.synchronize()
    }

    private func loadedModel(metadata: ModelMetadata = .init(), fixedToken: Int = 3) -> NativModel {
        let dir = URL(fileURLWithPath: "/tmp/generation-fixture")
        let loaded = LoadedModel(
            assets: .init(source: .directory(dir), modelDirectory: dir, tokenizerDirectory: dir),
            metadata: metadata, model: GenerationFixture(fixedToken: fixedToken),
            processor: GenerationProcessor(), tokenizer: GenerationTokenizer())
        return loaded.intoNativModel()
    }

    func testRawTokensLimitsEOSAndStopStringBypass() async throws {
        for limit in [0, 1, 5] {
            let run = loadedModel(metadata: .init(stopStrings: ["33"])).generate(
                input: UserInput(prompt: "test"),
                options: .init(
                    sampling: .init(temperature: 0),
                    output: .init(maxTokens: limit, format: .tokens)))
            var tokens: [Int] = []
            var completions = 0
            for try await event in run {
                switch event {
                case .token(let token): tokens.append(token)
                case .completion(let summary):
                    completions += 1
                    XCTAssertEqual(summary.generatedTokenCount, limit)
                    XCTAssertEqual(summary.stopReason, .length)
                default: XCTFail("Raw output must bypass text interpretation")
                }
            }
            XCTAssertEqual(tokens, Array(repeating: 3, count: limit))
            XCTAssertEqual(completions, 1)
        }
        for metadata in [
            ModelMetadata(eosTokenIds: [3]), ModelMetadata(extraEOSTokens: ["extra-stop"]),
        ] {
            let run = loadedModel(metadata: metadata).generate(
                input: UserInput(prompt: "test"),
                options: .init(
                    sampling: .init(temperature: 0), output: .init(maxTokens: 5, format: .tokens)))
            var tokens = 0
            for try await event in run { if case .token = event { tokens += 1 } }
            XCTAssertEqual(tokens, 0)
            let summary = try await run.metrics()
            XCTAssertEqual(summary.stopReason, .stop)
        }
    }

    func testTextStopSpanningTokensAndFramedFallbackOutput() async throws {
        for format: ToolCallFormat? in [nil, .gptOSS, .atem] {
            let run = loadedModel(metadata: .init(stopStrings: ["33"], toolCallFormat: format))
                .generate(
                    input: UserInput(prompt: "test"),
                    options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 8)))
            var text = ""
            for try await event in run { if case .chunk(let chunk) = event { text += chunk } }
            XCTAssertEqual(text, "")
            let summary = try await run.metrics()
            XCTAssertEqual(summary.stopReason, .stop)
            XCTAssertEqual(summary.generatedTokenCount, 2)
        }
    }
}

extension AutoregressiveGenerationTests {
    func testSessionPreservesToolCallsAndAcceptsToolResultTurns() async throws {
        let tokenizer = GenerationTokenizer(pieces: [
            2: "before ", 3: "<tool_call>",
            4: #"{"name":"weather","arguments":{"city":"Paris"}}"#,
            5: "</tool_call>", 6: " after",
        ])
        let context = ModelContext(
            metadata: .init(toolCallFormat: .json), model: GenerationFixture(),
            processor: GenerationProcessor(), tokenizer: tokenizer)
        let session = ModelSession(
            model: makeTestModel(context: context),
            options: .init(
                sampling: .init(temperature: 0), output: .init(maxTokens: 5)))
        let request = try await session.respond(to: "weather?")
        var events: [String] = []
        for try await event in request {
            switch event {
            case .chunk(let text): events.append(text)
            case .toolCall(let call): events.append("tool:" + call.function.name)
            default: break
            }
        }
        XCTAssertEqual(events, ["before ", "tool:weather", " after"])
        let assistant = try await session.withTranscript { messages in
            (messages.last?.content, messages.last?.tool?.calls?.first?.function.name)
        }
        XCTAssertEqual(assistant.0, "before  after")
        XCTAssertEqual(assistant.1, "weather")
        let result = try await session.generate(
            input: UserInput(chat: [.tool("sunny", id: "1", name: "weather")]),
            options: .init(output: .init(maxTokens: 0)))
        for try await _ in result {}
        let roles = try await session.withTranscript { $0.map(\.role) }
        XCTAssertEqual(roles, [.user, .assistant, .tool, .assistant])
        await session.close()
    }

    func testSessionUsesTextStopsAndRollsBackOutputOverflow() async throws {
        let stopped = ModelSession(
            model: loadedModel(metadata: .init(stopStrings: ["33"])),
            options: .init(
                sampling: .init(temperature: 0), output: .init(maxTokens: 8)))
        let request = try await stopped.respond(to: "stop")
        for try await _ in request {}
        let summary = try await request.metrics()
        XCTAssertEqual(summary.stopReason, .stop)
        let response = try await stopped.withTranscript { $0.last?.content }
        XCTAssertEqual(response, "")
        await stopped.close()

        let session = ModelSession(
            model: loadedModel(),
            options: .init(
                sampling: .init(temperature: 0), output: .init(maxTokens: 64, bufferCapacity: 1)))
        let overflow = try await session.respond(to: "overflow")
        do {
            _ = try await overflow.metrics()
            XCTFail("Expected overflow")
        } catch { XCTAssertEqual(error as? ModelExecutionError, .outputBufferOverflow) }
        let count = try await session.withTranscript { $0.count }
        XCTAssertEqual(count, 0)
        let retry = try await session.respond(
            to: "retry", options: .init(output: .init(maxTokens: 0)))
        for try await _ in retry {}
        let retried = try await retry.metrics()
        XCTAssertEqual(retried.cache?.kind, .cold)
        await session.close()
    }
}
