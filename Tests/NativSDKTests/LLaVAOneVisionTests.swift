import CoreImage
import MLX
import XCTest

@testable import NativSDK

/// The 36 AnyRes resolutions every published onevision checkpoint ships.
private let onevisionPinpoints: [[Int]] = {
    let steps = [384, 768, 1152, 1536, 1920, 2304]
    return steps.flatMap { height in steps.map { width in [height, width] } }
}()

private struct OneVisionTestTokenizer: Tokenizer {
    let imageTokenID = 151_646

    var bosToken: String? { "<|im_start|>" }
    var eosToken: String? { "<|im_end|>" }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        var output: [Int] = []
        var index = text.startIndex
        while index < text.endIndex {
            if text[index...].hasPrefix("<image>") {
                output.append(imageTokenID)
                index = text.index(index, offsetBy: "<image>".count)
            } else {
                output.append(Int(text[index].unicodeScalars.first!.value))
                index = text.index(after: index)
            }
        }
        return output
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map { $0 == imageTokenID ? "<image>" : String(UnicodeScalar($0)!) }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        token == "<image>" ? imageTokenID : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        id == imageTokenID ? "<image>" : nil
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?,
    )
        throws -> [Int]
    {
        let images = messages.reduce(into: 0) { total, message in
            guard let content = message["content"] as? [[String: any Sendable]] else { return }
            total += content.count { ($0["type"] as? String) == "image" }
        }
        return encode(
            text: String(repeating: "<image>", count: images) + "ok", addSpecialTokens: false)
    }
}

private func makeProcessor(tile: Int = 384, pinpoints: [[Int]] = onevisionPinpoints) throws
    -> LLaVAOneVisionProcessor
{
    let grid = pinpoints.map { "[\($0[0]), \($0[1])]" }.joined(separator: ", ")
    let json = """
        {
          "processor_class": "LlavaOnevisionProcessor",
          "image_processor_type": "LlavaOnevisionImageProcessor",
          "size": {"height": \(tile), "width": \(tile)},
          "image_mean": [0.5, 0.5, 0.5],
          "image_std": [0.5, 0.5, 0.5],
          "image_grid_pinpoints": [\(grid)]
        }
        """.data(using: .utf8)!
    let config = try JSONDecoder.json5().decode(
        LLaVAOneVisionProcessorConfiguration.self, from: json)
    return LLaVAOneVisionProcessor(config, tokenizer: OneVisionTestTokenizer())
}

final class LLaVAOneVisionTests: XCTestCase {

    func testMaxTileBudgetParsing() {
        XCTAssertEqual(llavaOneVisionMaxTiles("anyres_max_9"), 9)
        XCTAssertEqual(llavaOneVisionMaxTiles("anyres_max_16"), 16)
        XCTAssertEqual(llavaOneVisionMaxTiles("anyres"), 9)
    }

    /// Counts captured from mlx-vlm on `llava-onevision-qwen2-0.5b-ov-hf`; the 2000x2000
    /// case is the only one that trips the `ratio > 1.1` downsample.
    func testImageTokenCountsMatchReferenceProcessor() throws {
        let processor = try makeProcessor()
        let cases: [(height: Int, width: Int, expected: Int)] = [
            (448, 448, 3699),
            (400, 800, 4009),
            (900, 300, 2997),
            (2000, 2000, 7371),
            (384, 384, 1485),
        ]
        for testCase in cases {
            XCTAssertEqual(
                processor.imageTokenCount(
                    originalHeight: testCase.height, originalWidth: testCase.width),
                testCase.expected,
                "\(testCase.height)x\(testCase.width)",
            )
        }
    }

    func testProcessorEmitsAnyresTilesAndOriginalSize() async throws {
        let processor = try makeProcessor(tile: 28, pinpoints: [[28, 28], [28, 56], [56, 28]])
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 40, height: 20))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)

        XCTAssertEqual(prepared.image?.pixels.shape, [1, 3, 3, 28, 28])
        XCTAssertEqual(prepared.image?.frames?.count, 1)
        XCTAssertEqual(prepared.image?.frames?.first?.h, 20)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 40)
        XCTAssertNil(prepared.video)

        let tokens = prepared.text.tokens.asArray(Int.self)
        let expected = processor.imageTokenCount(originalHeight: 20, originalWidth: 40)
        XCTAssertEqual(tokens.count(where: { $0 == 151_646 }), expected)
    }

    func testMultipleImagesPadTheTileAxisToTheBatchMaximum() async throws {
        let processor = try makeProcessor(tile: 28, pinpoints: [[28, 28], [28, 56], [56, 28]])
        let square = CIImage(color: .green).cropped(to: CGRect(x: 0, y: 0, width: 20, height: 20))
        let wide = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 40, height: 20))
        let input = UserInput(
            chat: [.user("Compare.", images: [.ciImage(square), .ciImage(wide)])])

        let prepared = try await processor.prepare(input: input)

        XCTAssertEqual(prepared.image?.pixels.shape, [2, 3, 3, 28, 28])
        XCTAssertEqual(prepared.image?.frames?.map(\.w), [20, 40])

        let tokens = prepared.text.tokens.asArray(Int.self)
        let expected =
            processor.imageTokenCount(originalHeight: 20, originalWidth: 20)
            + processor.imageTokenCount(originalHeight: 20, originalWidth: 40)
        XCTAssertEqual(tokens.count(where: { $0 == 151_646 }), expected)
    }

    func testSanitizeMapsTransformersLayoutAndDropsTiedHead() throws {
        let config = try JSONDecoder.json5().decode(
            LLaVAOneVisionConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "llava_onevision",
                  "text_config": {
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "intermediate_size": 16,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "vocab_size": 16,
                    "tie_word_embeddings": true
                  },
                  "vision_config": {
                    "hidden_size": 6,
                    "num_hidden_layers": 1,
                    "intermediate_size": 12,
                    "num_attention_heads": 2,
                    "image_size": 4,
                    "patch_size": 2
                  }
                }
                """.data(using: .utf8)),
        )
        let model = LLaVAOneVision(config)

        let sanitized = model.sanitize(weights: [
            "model.vision_tower.embeddings.patch_embedding.weight": MLXArray.zeros([6, 3, 2, 2]),
            "model.multi_modal_projector.linear_1.weight": MLXArray.zeros([8, 6]),
            "model.language_model.norm.weight": MLXArray.zeros([8]),
            "model.image_newline": MLXArray.zeros([8]),
            "lm_head.weight": MLXArray.zeros([16, 8]),
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray([1, 2]),
            "vision_tower.vision_model.embeddings.position_ids": MLXArray.zeros([1, 4]),
        ])

        XCTAssertEqual(
            sanitized["vision_tower.vision_model.embeddings.patch_embedding.weight"]?.shape,
            [6, 2, 2, 3],
        )
        XCTAssertNotNil(sanitized["multi_modal_projector.linear_1.weight"])
        XCTAssertNotNil(sanitized["language_model.model.norm.weight"])
        XCTAssertNotNil(sanitized["image_newline"])
        XCTAssertNil(sanitized["language_model.lm_head.weight"])
        XCTAssertNil(sanitized["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNil(sanitized["vision_tower.vision_model.embeddings.position_ids"])
    }
}
