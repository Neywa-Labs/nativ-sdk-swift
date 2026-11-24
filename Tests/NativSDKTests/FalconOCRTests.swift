import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct FalconOCRTestTokenizer: Tokenizer {
    var bosToken: String? {
        nil
    }

    var eosToken: String? {
        "<|end_of_text|>"
    }

    var unknownToken: String? {
        nil
    }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        var output: [Int] = []
        var index = text.startIndex
        while index < text.endIndex {
            if text[index...].hasPrefix("<|OCR_PLAIN|>") {
                output.append(249)
                index = text.index(index, offsetBy: "<|OCR_PLAIN|>".count)
            } else if text[index...].hasPrefix("<|image|>") {
                output.append(227)
                index = text.index(index, offsetBy: "<|image|>".count)
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
            case 227: "<|image|>"
            case 244: "<|image_cls|>"
            case 245: "<|image_reg_1|>"
            case 246: "<|image_reg_2|>"
            case 247: "<|image_reg_3|>"
            case 248: "<|image_reg_4|>"
            case 249: "<|OCR_PLAIN|>"
            case 230: "<|img_end|>"
            default: String(UnicodeScalar(id)!)
            }
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<|image|>": 227
        case "<|image_cls|>": 244
        case "<|image_reg_1|>": 245
        case "<|image_reg_2|>": 246
        case "<|image_reg_3|>": 247
        case "<|image_reg_4|>": 248
        case "<|OCR_PLAIN|>": 249
        case "<|img_end|>": 230
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case 227: "<|image|>"
        case 244: "<|image_cls|>"
        case 245: "<|image_reg_1|>"
        case 246: "<|image_reg_2|>"
        case 247: "<|image_reg_3|>"
        case 248: "<|image_reg_4|>"
        case 249: "<|OCR_PLAIN|>"
        case 230: "<|img_end|>"
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
        encode(
            text: "<|image|>Extract the text content from this image.\n<|OCR_PLAIN|>",
            addSpecialTokens: false,
        )
    }
}

final class FalconOCRTests: XCTestCase {
    func testProcessorExpandsFalconImageToken() async throws {
        let data = """
            {
              "spatial_patch_size": 16,
              "img_id": 227,
              "img_end_id": 230,
              "image_cls_token_id": 244,
              "image_reg_1_token_id": 245,
              "image_reg_2_token_id": 246,
              "image_reg_3_token_id": 247,
              "image_reg_4_token_id": 248
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(
            FalconOCRProcessorConfiguration.self, from: data)
        let processor = FalconOCRProcessor(config, tokenizer: FalconOCRTestTokenizer())
        let image = CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 64, height: 64))
        let input = UserInput(chat: [.user("plain", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 227 }), 16)
        XCTAssertEqual(tokens.count(where: { $0 == 244 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 245 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 246 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 247 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 248 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 230 }), 1)
        XCTAssertEqual(prepared.image?.pixels.shape, [1, 64, 64, 3])
        XCTAssertEqual(prepared.image?.frames?.first?.t, 1)
        XCTAssertEqual(prepared.image?.frames?.first?.h, 4)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 4)
    }

    func testSanitizeMapsFalconKeysAndSplitsW13() throws {
        let model = try FalconOCR(Self.smallConfig())
        let w13 = MLXArray(0 ..< 24).reshaped(6, 4)

        let weights = model.sanitize(weights: [
            "tok_embeddings.weight": MLXArray.zeros([32, 8]),
            "img_projector.weight": MLXArray.zeros([8, 12]),
            "output.weight": MLXArray.zeros([32, 8]),
            "freqs_cis_golden": MLXArray.zeros([2, 2, 2]),
            "layers.0.attention.wqkv.weight": MLXArray.zeros([24, 8]),
            "layers.0.feed_forward.w13.weight": w13,
            "language_model.model.layers.1.mlp.w13.weight": w13,
        ])

        XCTAssertEqual(weights["language_model.model.embed_tokens.weight"]?.shape, [32, 8])
        XCTAssertEqual(weights["language_model.model.img_projector.weight"]?.shape, [8, 12])
        XCTAssertEqual(weights["language_model.lm_head.weight"]?.shape, [32, 8])
        XCTAssertEqual(weights["language_model.model.freqs_cis_golden"]?.shape, [2, 2, 2])
        XCTAssertEqual(
            weights["language_model.model.layers.0.self_attn.wqkv.weight"]?.shape,
            [24, 8],
        )
        XCTAssertEqual(
            weights["language_model.model.layers.0.mlp.w13.weight"]?.asArray(Int.self),
            [
                0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 4, 5, 6, 7, 12, 13, 14, 15, 20,
                21, 22, 23,
            ],
        )
        XCTAssertEqual(
            weights["language_model.model.layers.1.mlp.w13.weight"]?.asArray(Int.self),
            Array(0 ..< 24),
        )
    }

    private static func smallConfig() throws -> FalconOCRConfiguration {
        let data = """
            {
              "model_type": "falcon_ocr",
              "dim": 8,
              "n_layers": 1,
              "n_heads": 2,
              "head_dim": 4,
              "n_kv_heads": 1,
              "vocab_size": 32,
              "ffn_dim": 16,
              "norm_eps": 1e-5,
              "max_seq_len": 64,
              "spatial_patch_size": 2,
              "img_id": 9,
              "eos_id": 11,
              "image_cls_token_id": 12,
              "image_reg_1_token_id": 13,
              "image_reg_2_token_id": 14,
              "image_reg_3_token_id": 15,
              "image_reg_4_token_id": 16,
              "img_end_id": 17
            }
            """.data(using: .utf8)!
        return try JSONDecoder.json5().decode(FalconOCRConfiguration.self, from: data)
    }
}
