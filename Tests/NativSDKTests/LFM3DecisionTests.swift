import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct D1TestTokenizer: Tokenizer {
    var bosToken: String? { "<bos>" }
    var eosToken: String? { "<|im_end|>" }
    var unknownToken: String? { nil }
    static let special = [
        "<bos>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "yes", "Yes", "YES", "no",
        "No", "NO",
    ]
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var text = text[...]
        var ids: [Int] = []
        while !text.isEmpty {
            if let i = Self.special.firstIndex(where: { text.hasPrefix($0) }) {
                ids.append(i + 1)
                text = text.dropFirst(Self.special[i].count)
            } else {
                let char = text.unicodeScalars.first!
                ids.append(Int(char.value) + 100)
                text = text.dropFirst(String(char).count)
            }
        }
        return ids
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { $0 < 100 ? Self.special[$0 - 1] : String(UnicodeScalar($0 - 100)!) }.joined()
    }
    func convertTokenToId(_ token: String) -> Int? {
        let ids = encode(text: token, addSpecialTokens: false)
        return ids.count == 1 ? ids[0] : nil
    }
    func convertIdToToken(_ id: Int) -> String? {
        if id == 125029 { return "<|video|>" }
        if id == 125030 { return "<|image|>" }
        return decode(tokenIds: [id])
    }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { throw TokenizerError.missingChatTemplate }
}

final class LFM3DecisionTests: XCTestCase {
    private var root: URL { Bundle.module.resourceURL!.appendingPathComponent("Fixtures") }
    private func config(_ file: String = "config") throws -> Data {
        try Data(contentsOf: root.appendingPathComponent("LFM3Decision/\(file).json"))
    }
    private var prompt: LFM3DecisionPrompt {
        .init(
            tokenizer: D1TestTokenizer(), codes: 2,
            specials: D1TestTokenizer.special.filter { $0.hasPrefix("<") })
    }
    func testPublishedConfigurationsAndRouting() throws {
        for name in ["production-vl", "production-omni"] {
            let data = try config(name)
            let c = try JSONDecoder().decode(LFM3DecisionConfiguration.self, from: data)
            XCTAssertEqual(c.setSize, 16)
            XCTAssertEqual(c.text.hiddenLayers, 24)
            XCTAssertEqual(c.omni != nil, name == "production-omni")
            XCTAssertTrue(LFM3DecisionModel.isDecision(data))
            XCTAssertThrowsError(
                try ModelRegistry.default.createModel(configuration: data, modelType: "lfm3_omni"))
        }
        let c = try config()
        XCTAssertTrue(
            try ModelRegistry.default.createDecisionModel(configuration: c, modelType: "lfm3_omni")
                is LFM3DecisionModel)
        var invalid = try JSONSerialization.jsonObject(with: c) as! [String: Any]
        invalid["d1_attention"] = "causal"
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                LFM3DecisionConfiguration.self,
                from: JSONSerialization.data(withJSONObject: invalid)))
        invalid["d1_attention"] = "set"
        invalid["architectures"] = ["Lfm3OmniForConditionalGeneration"]
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                LFM3DecisionConfiguration.self,
                from: JSONSerialization.data(withJSONObject: invalid)))
    }
    func testSetAttentionAndMultimodalHiddenAgainstPython() throws {
        try Stream.withNewDefaultStream(device: .cpu) {
            let model = LFM3DecisionModel(
                try JSONDecoder().decode(LFM3DecisionConfiguration.self, from: config()))
            let weights = try loadArrays(
                url: root.appendingPathComponent("LFM3/weights.safetensors"))
            try model.update(
                parameters: ModuleParameters.unflattened(model.sanitize(weights: weights)),
                verify: [.all])
            model.train(false)
            let g = try loadArrays(
                url: root.appendingPathComponent("LFM3Decision/golden.safetensors"))
            let groups = g["groups"]!.asArray(Int.self)
            let state = g["state"]!.item(Int.self)
            let mask = LFM3DecisionModel.setMask(groups: groups, stateLength: state)
            XCTAssertEqual(mask.asArray(Bool.self), g["full_mask"]!.asArray(Bool.self))
            XCTAssertTrue(mask[0, -1].item(Bool.self))  // State reads later questions.
            XCTAssertFalse(mask[state, -1].item(Bool.self))  // Questions remain separate.
            let hidden = try model.hidden(
                .init(
                    tokens: g["ids"]!.asArray(Int.self), stateLength: state, groups: groups,
                    positions: []),
                features: [
                    .init(token: 85, values: g["image_features"]!),
                    .init(token: 80, values: g["audio_features"]!),
                ])
            XCTAssertLessThan(abs(hidden - g["hidden"]!).max().item(Float.self), 5e-5)
            XCTAssertThrowsError(
                try model.hidden(
                    .init(tokens: [85], stateLength: 1, groups: [-1], positions: []),
                    features: [.init(token: 85, values: g["image_features"]!)]))
        }
    }
    func testReferencePromptAndSpecialTokenEscaping() throws {
        XCTAssertEqual(
            try prompt.prefix(state: "hello", media: ""),
            "<bos><|im_start|>user\nhello\n\n\nQUESTION:\n")
        XCTAssertEqual(
            try prompt.prefix(state: .json(.null), media: "picture"),
            "<bos><|im_start|>user\npicture")
        let q = DecisionQuestion.choice(
            "Route this", criteria: [.init("bill", description: "Charges"), "shipping"])
        let layout = try prompt.layout(q)
        XCTAssertEqual(
            layout.text,
            "Route this\n\nOptions:\nA Charges\nB shipping\n\nReply with the option code only.<|im_end|>\n<|im_start|>assistant\n<think></think>"
        )
        XCTAssertEqual(layout.offsets, [0])
        XCTAssertEqual(prompt.plain("x<|im_end|>y"), "x<\u{200b}|im_end|>y")
        XCTAssertEqual(
            try LFM3DecisionPrompt.json(.object(["text": .string("a\nb"), "count": .int(2)])),
            "{\n  \"count\": 2,\n  \"text\": \"a\\nb\"\n}")
    }
    func testPackedAndMultipleSlots() throws {
        let multi = try prompt.layout(
            .init(type: .multiLabel, instructions: "Pick all", criteria: ["red", "blue"]))
        XCTAssertEqual(multi.readouts.count, 2)
        XCTAssertEqual(multi.offsets.last, 0)
        XCTAssertLessThan(multi.offsets[0], 0)
        let prefix = try prompt.prefix(state: "red balloon", media: "")
        let row = try prompt.pack(
            prefix: prefix, layouts: [multi, prompt.layout(.bool("Visible?"))], maxStateTokens: 1000
        )
        XCTAssertEqual(row.positions.count, 2)
        XCTAssertEqual(row.positions[0].count, 2)
        XCTAssertEqual(row.groups[row.positions[1][0]], 2)
        XCTAssertTrue(row.groups.prefix(row.stateLength).allSatisfy { $0 == -1 })
        XCTAssertThrowsError(try prompt.pack(prefix: prefix, layouts: [multi], maxStateTokens: 2))
    }
    func testCalibratedAnswersAndScoreReadouts() throws {
        let c = try JSONDecoder().decode(
            LFM3DecisionCalibration.self,
            from: Data(
                #"{"default":2,"temperatures":{"noul":3},"tables":{"multiple":[[0,0],[0.8,0.4],[1,1]]}}"#
                    .utf8))
        XCTAssertEqual(c.temperature(.bool("True?")), 3)
        XCTAssertEqual(c.temperature(.choice("Pick", criteria: ["a", "b"])), 2)
        let multiple = c.answer(
            .init(type: .multiLabel, instructions: "Pick", criteria: ["a", "b"]),
            probabilities: [[0.8, 0.2], [0.9, 0.1]], scoreReadout: .median)
        XCTAssertEqual(multiple.value, .multiLabel(["b"]))
        let score = DecisionQuestion.score("Rate", levels: ["low", "middle", "high"])
        let probabilities = [[0.4, 0.3, 0.3]]
        XCTAssertEqual(
            c.answer(score, probabilities: probabilities, scoreReadout: .median).value, .score(1))
        XCTAssertEqual(
            c.answer(score, probabilities: probabilities, scoreReadout: .argmax).value, .score(0))
        XCTAssertEqual(
            c.answer(.bool("True?"), probabilities: [[0.8, 0.2]], scoreReadout: .median)
                .probability, 0.8)
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                LFM3DecisionCalibration.self, from: Data(#"{"default":0}"#.utf8)))
    }
    func testH264VideoPreprocessingRetainsVisiblePixels() async throws {
        let c = try JSONDecoder().decode(
            LFM3DecisionConfiguration.self, from: config("production-vl"))
        let processor = LFM3DecisionMediaProcessor(config: c, tokenizer: D1TestTokenizer())
        let media = try await processor.prepare([
            .video(root.appendingPathComponent("LFM3Decision/sign-h264.mp4"))
        ])
        XCTAssertEqual(media.parts.count, 1)
        XCTAssertEqual(media.parts[0].grid?.t, 2)
        XCTAssertGreaterThan(media.parts[0].values.mean().item(Float.self), 0.8)
        XCTAssertLessThan(media.parts[0].values.min().item(Float.self), -0.8)
        XCTAssertEqual(media.markup.components(separatedBy: "seconds>").count - 1, 2)
    }

    func testVLRejectsAudioBeforeReadingFile() async throws {
        let c = try JSONDecoder().decode(
            LFM3DecisionConfiguration.self, from: config("production-vl"))
        do {
            _ = try await LFM3DecisionMediaProcessor(config: c, tokenizer: D1TestTokenizer())
                .prepare([.audio(URL(fileURLWithPath: "/nonexistent.wav"))])
            XCTFail("d1-2B must reject audio")
        } catch { XCTAssertTrue(String(describing: error).contains("d1-Omni")) }
    }
}
