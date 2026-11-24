import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct InternVLTestTokenizer: Tokenizer {
    var bosToken: String? {
        nil
    }

    var eosToken: String? {
        "<|im_end|>"
    }

    var unknownToken: String? {
        nil
    }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        var output: [Int] = []
        var index = text.startIndex
        while index < text.endIndex {
            if text[index...].hasPrefix("<IMG_CONTEXT>") {
                output.append(151_667)
                index = text.index(index, offsetBy: "<IMG_CONTEXT>".count)
            } else if text[index...].hasPrefix("</img>") {
                output.append(151_666)
                index = text.index(index, offsetBy: "</img>".count)
            } else if text[index...].hasPrefix("<img>") {
                output.append(151_665)
                index = text.index(index, offsetBy: "<img>".count)
            } else {
                output.append(Int(text[index].unicodeScalars.first!.value))
                index = text.index(after: index)
            }
        }
        return output
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map { id in
            switch id {
            case 151_665: "<img>"
            case 151_666: "</img>"
            case 151_667: "<IMG_CONTEXT>"
            default: String(UnicodeScalar(id)!)
            }
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<img>": 151_665
        case "</img>": 151_666
        case "<IMG_CONTEXT>": 151_667
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case 151_665: "<img>"
        case 151_666: "</img>"
        case 151_667: "<IMG_CONTEXT>"
        default: nil
        }
    }

    func applyChatTemplate(
        messages _: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?,
    )
        throws -> [Int]
    {
        encode(text: "User: Describe.\nAssistant:\n", addSpecialTokens: false)
    }
}

final class InternVLTests: XCTestCase {
    func testProcessorBuildsSingleTilePrompt() async throws {
        let data = """
            {
              "processor_class": "InternVLChatProcessor",
              "image_mean": [0.485, 0.456, 0.406],
              "image_std": [0.229, 0.224, 0.225],
              "size": 448
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(InternVLProcessorConfiguration.self, from: data)
        let processor = InternVLProcessor(config, tokenizer: InternVLTestTokenizer())
        let image = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 32, height: 32))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 151_667 }), 256)
        XCTAssertEqual(tokens.count(where: { $0 == 151_665 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 151_666 }), 1)
        XCTAssertEqual(prepared.image?.pixels.shape, [1, 1, 3, 448, 448])
    }

    func testProcessorDynamicallyTilesWideImageWithThumbnail() async throws {
        let data = """
            {
              "processor_class": "InternVLChatProcessor",
              "image_mean": [0.485, 0.456, 0.406],
              "image_std": [0.229, 0.224, 0.225],
              "size": 448,
              "dynamic_use_thumbnail": true
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(InternVLProcessorConfiguration.self, from: data)
        let processor = InternVLProcessor(config, tokenizer: InternVLTestTokenizer())
        let image = CIImage(color: .green).cropped(to: CGRect(x: 0, y: 0, width: 896, height: 448))
        let input = UserInput(chat: [.user("<image>\nDescribe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 151_667 }), 768)
        XCTAssertEqual(tokens.count(where: { $0 == 151_665 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 151_666 }), 1)
        XCTAssertEqual(prepared.image?.pixels.shape, [1, 3, 3, 448, 448])
    }

    func testProcessorSupportsMultipleImages() async throws {
        let data = """
            {
              "processor_class": "InternVLChatProcessor",
              "image_mean": [0.485, 0.456, 0.406],
              "image_std": [0.229, 0.224, 0.225],
              "size": 448
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(InternVLProcessorConfiguration.self, from: data)
        let processor = InternVLProcessor(config, tokenizer: InternVLTestTokenizer())
        let first = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 32, height: 32))
        let second = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 64, height: 64))
        let input = UserInput(chat: [
            .user("Compare these images.", images: [.ciImage(first), .ciImage(second)])
        ])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 151_667 }), 512)
        XCTAssertEqual(tokens.count(where: { $0 == 151_665 }), 2)
        XCTAssertEqual(tokens.count(where: { $0 == 151_666 }), 2)
        XCTAssertEqual(prepared.image?.pixels.shape, [1, 2, 3, 448, 448])
    }

    func testSanitizeDropsRotaryAndTransposesPatchEmbedding() throws {
        let data = """
            {
              "model_type": "internvl_chat",
              "text_config": {
                "model_type": "qwen2",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "vocab_size": 16,
                "tie_word_embeddings": true
              },
              "vision_config": {
                "model_type": "intern_vit_6b",
                "hidden_size": 6,
                "num_hidden_layers": 1,
                "intermediate_size": 12,
                "num_attention_heads": 2,
                "image_size": 4,
                "patch_size": 2,
                "num_channels": 3,
                "layer_norm_eps": 1e-6
              }
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(InternVLConfiguration.self, from: data)
        let model = InternVL(config)

        let weights = model.sanitize(weights: [
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray([1, 2]),
            "vision_model.embeddings.patch_embedding.weight": MLXArray.zeros([
                6, 3, 2, 2,
            ]),
        ])

        XCTAssertNil(weights["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertEqual(
            weights["vision_model.embeddings.patch_embedding.weight"]?.shape,
            [6, 2, 2, 3],
        )
    }
}
