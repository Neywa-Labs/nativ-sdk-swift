import Foundation
import XCTest

@testable import NativSDK

final class TokenizerLoadingTests: XCTestCase {
    func testUnigramRoutingPreservesEncodingDecodingAndChatTemplate() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let metaspace: [String: Any] = [
            "type": "Metaspace", "replacement": "▁", "prepend_scheme": "always",
        ]
        try write(
            [
                "model": [
                    "type": "Unigram", "unk_id": 0,
                    "vocab": [
                        ["<unk>", 0.0], ["</s>", 0.0], ["▁he", -1.0],
                        ["llo", -2.0], ["▁world", -1.5],
                    ],
                ],
                "added_tokens": [], "pre_tokenizer": metaspace, "decoder": metaspace,
            ], to: directory.appending(path: "tokenizer.json"))
        try "{{ messages[0]['content'] }}".write(
            to: directory.appending(path: "chat_template.jinja"), atomically: true, encoding: .utf8)

        for declaredClass in [
            nil, "PreTrainedTokenizerFast", "CustomTokenizerFast", "T5TokenizerFast",
            "XLMRobertaTokenizerFast",
        ] as [String?] {
            var config = ["unk_token": "<unk>", "eos_token": "</s>"]
            config["tokenizer_class"] = declaredClass
            try write(config, to: directory.appending(path: "tokenizer_config.json"))
            let tokenizer = try await DefaultTokenizerLoader().load(from: directory)
            XCTAssertEqual(
                tokenizer.encode(text: "hello world", addSpecialTokens: false), [2, 3, 4])
            XCTAssertEqual(tokenizer.decode(tokenIds: [2, 3, 4]), "hello world")
            XCTAssertEqual(
                try tokenizer.applyChatTemplate(messages: [
                    ["role": "user", "content": "hello world"]
                ]),
                [2, 3, 4])
        }
    }

    func testBPEGenericAndUnknownClassesStillApplyMerges() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try write(
            [
                "model": [
                    "type": "BPE", "vocab": ["<unk>": 0, "a": 1, "b": 2, "ab": 3],
                    "merges": ["a b"],
                ],
                "added_tokens": [],
            ], to: directory.appending(path: "tokenizer.json"))
        for declaredClass in ["PreTrainedTokenizerFast", "CustomTokenizerFast"] {
            try write(
                ["tokenizer_class": declaredClass, "unk_token": "<unk>"],
                to: directory.appending(path: "tokenizer_config.json"))
            let tokenizer = try await DefaultTokenizerLoader().load(from: directory)
            XCTAssertEqual(tokenizer.encode(text: "abab", addSpecialTokens: false), [3, 3])
            XCTAssertEqual(tokenizer.decode(tokenIds: [3, 3]), "abab")
        }
    }

    func testDecisionSpecialTokensPreserveCustomNamesAndObjectContent() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try write(
            [
                "model": [
                    "type": "BPE",
                    "vocab": [
                        "<unk>": 0, "a": 1, "b": 2, "ab": 3,
                        "<C>": 4, "<S>": 5, "<M>": 6, "<P>": 7,
                    ],
                    "merges": ["a b"],
                ],
                "added_tokens": [],
            ], to: directory.appending(path: "tokenizer.json"))
        try write(
            [
                "tokenizer_class": "PreTrainedTokenizerFast", "unk_token": "<unk>",
                "cls_token": "<C>", "sep_token": "<S>", "pad_token": "<P>",
                "mask_token": ["content": "<M>", "special": true],
            ], to: directory.appending(path: "tokenizer_config.json"))
        let tokenizer = try await DefaultTokenizerLoader().load(from: directory)
        XCTAssertEqual(tokenizer.clsToken, "<C>")
        XCTAssertEqual(tokenizer.sepToken, "<S>")
        XCTAssertEqual(tokenizer.maskToken, "<M>")
        XCTAssertEqual(tokenizer.padToken, "<P>")
        XCTAssertEqual(tokenizer.convertTokenToId(try XCTUnwrap(tokenizer.maskToken)), 6)
        XCTAssertEqual(tokenizer.encode(text: "ab", addSpecialTokens: false), [3])
    }

    private func write(_ value: [String: Any], to url: URL) throws {
        try JSONSerialization.data(withJSONObject: value).write(to: url)
    }
}
