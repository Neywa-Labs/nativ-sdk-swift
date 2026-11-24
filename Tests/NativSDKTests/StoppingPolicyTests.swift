import Foundation
import XCTest

@testable import NativSDK

private struct StopTokenizer: Tokenizer {
    let tokens = ["<unk>": 0, "<eos>": 1, "<|im_end|>": 2]
    var bosToken: String? { nil }
    var eosToken: String? { "<eos>" }
    var unknownToken: String? { "<unk>" }
    func convertTokenToId(_ token: String) -> Int? { tokens[token] ?? 0 }
    func convertIdToToken(_ id: Int) -> String? { tokens.first { $0.value == id }?.key }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map(Int.init) }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(decoding: tokenIds.map(UInt8.init), as: UTF8.self)
    }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

final class StoppingPolicyTests: XCTestCase {
    func testCheckpointSupplementalTerminators() throws {
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "stopping-defaults", withExtension: "json", subdirectory: "Fixtures"))
        let fixtures = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [[String: Any]])
        XCTAssertEqual(fixtures.count, 71)
        for fixture in fixtures {
            let configuration = try XCTUnwrap(fixture["configuration"] as? [String: Any])
            let modelType = try XCTUnwrap(configuration["model_type"] as? String)
            let expected = Set(try XCTUnwrap(fixture["supplementalTokens"] as? [String]))
            let actual = try ModelConventions.supplementalStopTokens(
                modelType: modelType,
                configuration: JSONSerialization.data(withJSONObject: configuration))
            XCTAssertEqual(actual, expected, fixture["checkpoint"] as? String ?? modelType)
        }
    }

    func testCompositeBackboneControlsTerminators() throws {
        for type in ["llava", "llava_next", "llava_onevision", "internvl", "internvl_chat"] {
            for key in ["text_config", "llm_config"] {
                let qwen = try ModelConventions.supplementalStopTokens(
                    modelType: type,
                    configuration: Data("{\"\(key)\":{\"model_type\":\"qwen2\"}}".utf8))
                XCTAssertEqual(qwen, ["<|im_end|>"])
                let mistral = try ModelConventions.supplementalStopTokens(
                    modelType: type,
                    configuration: Data("{\"\(key)\":{\"model_type\":\"mistral\"}}".utf8))
                XCTAssertEqual(mistral, ["</s>"])
            }
        }
        XCTAssertEqual(
            try ModelConventions.supplementalStopTokens(
                modelType: "mistral3",
                configuration: Data(#"{"text_config":{"model_type":"mistral4"}}"#.utf8)), ["</s>"])
        XCTAssertEqual(
            try ModelConventions.supplementalStopTokens(
                modelType: "custom", configuration: Data("{}".utf8)), [])
    }

    func testArchitectureAliasesRetainSupplementalTerminators() throws {
        for (alias, canonical) in [
            ("granite-vision", "granite_vision"),
            ("granite4-vision", "granite4_vision"),
            ("qwen3_5_moe_text", "qwen3_5_text"),
        ] {
            let expected = try ModelConventions.supplementalStopTokens(
                modelType: canonical, configuration: Data("{}".utf8))
            XCTAssertFalse(expected.isEmpty)
            XCTAssertEqual(
                try ModelConventions.supplementalStopTokens(
                    modelType: alias, configuration: Data("{}".utf8)), expected)
        }
    }

    func testEOSUnionExcludesUnknownLookupsAndRetainsCallerIDs() {
        XCTAssertEqual(
            StopPolicy.tokenIDs(
                configured: [8, 9], supplementalTokens: ["<|im_end|>", "</answer>", "missing"],
                tokenizer: StopTokenizer()), [1, 2, 8, 9])
    }

    func testMultiTokenStopStringAcrossChunks() {
        var filter = StopStringFilter(stopStrings: ["</answer>"])
        // A terminator absent as a single vocabulary entry must still stop text output.
        let first = filter.process("A complete answer</ans")
        XCTAssertEqual(first.text, "A complete answer")
        XCTAssertFalse(first.stopped)
        let second = filter.process("wer>ignored")
        XCTAssertNil(second.text)
        XCTAssertTrue(second.stopped)
        XCTAssertNil(filter.finish())
    }

    func testReasoningDependsOnCheckpointIdentityAndKeepsBaseModelsUnchanged() {
        XCTAssertEqual(
            ModelConventions.reasoningConfig(
                modelId: "someone/DeepSeek-R1-Distill-Qwen-4bit", modelType: "qwen2"),
            .alwaysOnThinking)
        XCTAssertEqual(
            ModelConventions.reasoningConfig(
                modelId: "local/R1-Distill-Llama", modelType: "llama"), .alwaysOnThinking)
        XCTAssertNil(ModelConventions.reasoningConfig(modelId: "someone/Qwen2", modelType: "qwen2"))
        XCTAssertNil(
            ModelConventions.reasoningConfig(modelId: "DeepSeek-V3", modelType: "deepseek_v3"))
    }
}
