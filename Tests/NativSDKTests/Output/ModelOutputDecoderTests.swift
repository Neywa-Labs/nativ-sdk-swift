import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct ReplayTokenizer: Tokenizer {
    var pieces: [String]
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        pieces.firstIndex(of: text).map { [$0] } ?? []
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { pieces[$0] }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? { pieces.firstIndex(of: token) }
    func convertIdToToken(_ id: Int) -> String? { pieces[id] }
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [0] }
}

private final class ReplayModel: Module, LLMModel {
    let script: [Int]
    let vocabulary: Int
    var index = 0
    init(script: [Int], vocabulary: Int) {
        self.script = script
        self.vocabulary = vocabulary
    }
    var loraLayers: [Module] { [] }
    func newCache(configuration: KVCacheConfiguration?) -> [KVCache] { [] }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var logits = Array(repeating: Float(-10), count: vocabulary)
        logits[script[min(index, script.count - 1)]] = 10
        index += 1
        return MLXArray(logits, [1, 1, vocabulary])
    }
}

private struct ReplayProcessor: UserInputProcessor {
    func prepare(input: UserInput) async throws -> LMInput { LMInput(tokens: MLXArray([0])) }
}

final class ModelOutputDecoderTests: XCTestCase {
    func testRecordedPythonReferenceFixturesAtEverySplit() throws {
        struct Fixture: Decodable {
            let format: ToolCallFormat
            let text: String
            let calls: [ToolCall.Function]
        }
        let fixtures = try JSONDecoder().decode(
            [Fixture].self, from: Data(referenceToolFixtures.utf8))
        for fixture in fixtures {
            let scalars = Array(fixture.text.unicodeScalars)
            for split in 0 ... scalars.count {
                let tokenizer = ReplayTokenizer(pieces: [
                    String(String.UnicodeScalarView(scalars.prefix(split))),
                    String(String.UnicodeScalarView(scalars.dropFirst(split))),
                ])
                let decoder = ModelOutputDecoder(
                    tokenizer: tokenizer, format: fixture.format,
                    toolCallPolicy: .init(recovery: .disabled))
                let events = decoder.push(0) + decoder.push(1) + decoder.finish()
                let calls: [ToolCall.Function] = events.compactMap {
                    if case .toolCall(let call) = $0 { return call.function }
                    return nil
                }
                XCTAssertEqual(calls, fixture.calls, "\(fixture.format) split \(split)")
                XCTAssertEqual(response(events), "")
            }
        }
    }

    private let controls = [
        "<|start|>", "<|channel|>", "<|message|>", "<|end|>", "<|call|>", "<|return|>",
        "<|constrain|>", "<|eom|>", "<|eot|>",
    ]
    private let tools: [[String: any Sendable]] = [
        ["function": ["name": "weather"] as [String: any Sendable]]
    ]

    private func response(_ events: [ModelOutputEvent]) -> String {
        events.compactMap {
            if case .response(let text) = $0 { return text }
            return nil
        }.joined()
    }
    private func reasoning(_ events: [ModelOutputEvent]) -> String {
        events.compactMap {
            if case .reasoning(let text) = $0 { return text }
            return nil
        }.joined()
    }

    func testReasoningIsLosslessAcrossEveryScalarSplit() {
        let text = "before <think>\n e\u{301} 🏳️\n</think> after"
        let scalars = Array(text.unicodeScalars)
        for split in 0 ... scalars.count {
            let pieces = [
                String(String.UnicodeScalarView(scalars.prefix(split))),
                String(String.UnicodeScalarView(scalars.dropFirst(split))),
            ]
            let decoder = ModelOutputDecoder(
                tokenizer: ReplayTokenizer(pieces: pieces), reasoning: .thinkTagsWithEnableThinking)
            let events = decoder.push(0) + decoder.push(1) + decoder.finish()
            XCTAssertEqual(response(events), "before  after", "split \(split)")
            XCTAssertEqual(reasoning(events), "\n e\u{301} 🏳️\n", "split \(split)")
            XCTAssertTrue(decoder.finish().isEmpty)
            XCTAssertTrue(decoder.push(0).isEmpty)
        }
    }

    func testPrimedReasoningImplicitExitAndNoToolPromotionInsideReasoning() {
        let hidden = #"<tool_call>{"name":"weather","arguments":{}}</tool_call>"#
        let pieces = [hidden, "</think>", " answer"]
        let decoder = ModelOutputDecoder(
            tokenizer: ReplayTokenizer(pieces: pieces), tools: tools,
            reasoning: .thinkTagsWithEnableThinking, reasoningPrimed: true)
        let events = pieces.indices.flatMap { decoder.push($0) } + decoder.finish()
        XCTAssertEqual(reasoning(events), hidden)
        XCTAssertEqual(response(events), " answer")
        XCTAssertFalse(
            events.contains {
                if case .toolCall = $0 { return true }
                return false
            })

        let config = ReasoningConfig(
            startDelimiter: "<think>", endDelimiter: "</think>", promptStrategy: .none,
            implicitEndDelimiters: ["<tool_call>"])
        let implicit = ModelOutputDecoder(
            tokenizer: ReplayTokenizer(pieces: ["private", hidden]), tools: tools,
            reasoning: config, reasoningPrimed: true)
        let result = implicit.push(0) + implicit.push(1) + implicit.finish()
        XCTAssertEqual(reasoning(result), "private")
        XCTAssertEqual(
            result.compactMap {
                if case .toolCall(let call) = $0 { return call.function.name }
                return nil
            }, ["weather"])
    }

    func testReasoningControlIDsDoNotInterpretLiteralMarkersInArguments() {
        let payload =
            #"<tool_call>{"name":"weather","arguments":{"text":"literal <think> and </think>"}}</tool_call>"#
        let tokenizer = ReplayTokenizer(pieces: ["<think>", "</think>", "private", payload])
        let decoder = ModelOutputDecoder(
            tokenizer: tokenizer, tools: tools, reasoning: .thinkTagsWithEnableThinking)
        let events = [0, 2, 1, 3].flatMap { decoder.push($0) } + decoder.finish()
        let calls: [ToolCall] = events.compactMap {
            if case .toolCall(let call) = $0 { return call }
            return nil
        }
        XCTAssertEqual(reasoning(events), "private")
        XCTAssertEqual(
            calls.first?.function.arguments["text"], .string("literal <think> and </think>"))
    }

    func testCancellationDiscardsPendingRepairAndFinishIsIdempotent() {
        let partial = #"<tool_call>{"name":"weather","arguments":{}}"#
        let decoder = ModelOutputDecoder(
            tokenizer: ReplayTokenizer(pieces: [partial]), format: .lfm2, tools: tools,
            toolCallPolicy: .init(recovery: .permissive))
        XCTAssertTrue(decoder.push(0).isEmpty)
        decoder.cancel()
        XCTAssertTrue(decoder.finish().isEmpty)
        XCTAssertTrue(decoder.push(0).isEmpty)
        XCTAssertEqual(decoder.recoveredToolCallCount, 0)
        let completed = ModelOutputDecoder(
            tokenizer: ReplayTokenizer(pieces: [partial]), format: .lfm2, tools: tools,
            toolCallPolicy: .init(recovery: .permissive))
        _ = completed.push(0)
        XCTAssertTrue(
            completed.finish().contains {
                if case .toolCall = $0 { return true }
                return false
            })
        XCTAssertTrue(completed.finish().isEmpty)
    }

    func testStopsAndIncompleteDelimiters() {
        let pieces = ["hello ST", "OP ignored", "more"]
        let decoder = ModelOutputDecoder(
            tokenizer: ReplayTokenizer(pieces: pieces), stopStrings: ["STOP"])
        let events = decoder.push(0) + decoder.push(1) + decoder.push(2) + decoder.finish()
        XCTAssertEqual(response(events), "hello ")
        XCTAssertEqual(events.filter { $0 == .stop }.count, 1)
        XCTAssertTrue(decoder.isStopped)
        let incomplete = ModelOutputDecoder(
            tokenizer: ReplayTokenizer(pieces: ["hello <thi"]),
            reasoning: .thinkTagsWithEnableThinking)
        XCTAssertEqual(response(incomplete.push(0) + incomplete.finish()), "hello <thi")
    }

    func testHarmonyAndOnyxReplayAndOwnedGeneration() async throws {
        for (format, pieces): (ToolCallFormat, [String]) in [
            (
                .gptOSS,
                [
                    "<|channel|>", "analysis", "<|message|>", "private", "<|end|>", "<|start|>",
                    "assistant", "<|channel|>", "final", "<|message|>", "answer", "<|return|>",
                ]
            ),
            (
                .atem,
                [
                    " to=self", "<|message|>", "private", "<|eom|>", "<|start|>",
                    "assistant to=user", "<|message|>", "answer", "<|eot|>",
                ]
            ),
        ] {
            let tokenizer = ReplayTokenizer(pieces: controls + pieces)
            let tokens = pieces.map { tokenizer.convertTokenToId($0)! }
            let decoder = ModelOutputDecoder(tokenizer: tokenizer, format: format)
            let events = tokens.flatMap { decoder.push($0) } + decoder.finish()
            XCTAssertEqual(response(events), "answer")
            XCTAssertEqual(reasoning(events), "private")
            XCTAssertTrue(decoder.isStopped)
            let directory = URL(fileURLWithPath: "/tmp/replay-model")
            let loaded = LoadedModel(
                assets: .init(
                    source: .directory(directory), modelDirectory: directory,
                    tokenizerDirectory: directory), metadata: .init(toolCallFormat: format),

                model: ReplayModel(script: tokens, vocabulary: tokenizer.pieces.count),
                processor: ReplayProcessor(), tokenizer: tokenizer)
            let run = loaded.intoNativModel().generate(
                input: UserInput(prompt: "test"),
                options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 40)))
            var answer = ""
            var thought = ""
            for try await event in run {
                if case .chunk(let text) = event { answer += text }
                if case .reasoning(let text) = event { thought += text }
            }
            XCTAssertEqual(answer, "answer")
            XCTAssertEqual(thought, "private")
            let summary = try await run.metrics()
            XCTAssertEqual(summary.stopReason, .stop)
            XCTAssertEqual(summary.generatedTokenCount, tokens.count - 1)
        }
    }

    func testOnyxToolCommitAndMalformedFrames() {
        let pieces = [
            " to=weather", "<|message|>",
            #"<atem:function_calls><atem:invoke name="weather"><atem:parameter name="city">Paris</atem:parameter></atem:invoke></atem:function_calls>"#,
            "<|eot|>",
        ]
        let tokenizer = ReplayTokenizer(pieces: controls + pieces)
        let decoder = ModelOutputDecoder(tokenizer: tokenizer, format: .atem, tools: tools)
        let events =
            pieces.flatMap { decoder.push(tokenizer.convertTokenToId($0)!) } + decoder.finish()
        let cities: [JSONValue] = events.compactMap {
            if case .toolCall(let call) = $0 { return call.function.arguments["city"] }
            return nil
        }
        XCTAssertEqual(cities, [.string("Paris")])
        let malformed = ModelOutputDecoder(tokenizer: tokenizer, format: .atem, tools: tools)
        _ = malformed.push(tokenizer.convertTokenToId(" to=weather")!)
        XCTAssertTrue(
            malformed.finish().contains {
                if case .protocolError = $0 { return true }
                return false
            })
    }

    func testOwnedGenerationPrimesReasoningAndRawOutputBypassesIt() async throws {
        for raw in [false, true] {
            let tokenizer = ReplayTokenizer(pieces: ["<think>\n", "hidden", "</think>", "answer"])
            let directory = URL(fileURLWithPath: "/tmp/replay-model")
            let loaded = LoadedModel(
                assets: .init(
                    source: .directory(directory), modelDirectory: directory,
                    tokenizerDirectory: directory),
                metadata: .init(reasoningConfig: .thinkTagsWithEnableThinking),
                model: ReplayModel(script: [1, 2, 3], vocabulary: 4), processor: ReplayProcessor(),
                tokenizer: tokenizer)
            let run = loaded.intoNativModel().generate(
                input: UserInput(prompt: "test"),
                options: .init(
                    sampling: .init(temperature: 0),
                    output: .init(maxTokens: 3, format: raw ? .tokens : .text)))
            var tokens: [Int] = []
            var answer = ""
            var thought = ""
            for try await event in run {
                switch event {
                case .token(let token): tokens.append(token)
                case .reasoning(let text): thought += text
                case .chunk(let text): answer += text
                default: break
                }
            }
            XCTAssertEqual(tokens, raw ? [1, 2, 3] : [])
            XCTAssertEqual(answer, raw ? "" : "answer")
            XCTAssertEqual(thought, raw ? "" : "hidden")
        }
    }
}
