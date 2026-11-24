import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct DotsOCRTestTokenizer: Tokenizer {
    var bosToken: String? {
        nil
    }

    var eosToken: String? {
        "<|endoftext|>"
    }

    var unknownToken: String? {
        nil
    }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        var output: [Int] = []
        var index = text.startIndex
        while index < text.endIndex {
            if text[index...].hasPrefix("<|imgpad|>") {
                output.append(151_665)
                index = text.index(index, offsetBy: "<|imgpad|>".count)
            } else if text[index...].hasPrefix("<|img|>") {
                output.append(151_664)
                index = text.index(index, offsetBy: "<|img|>".count)
            } else if text[index...].hasPrefix("<|endofimg|>") {
                output.append(151_666)
                index = text.index(index, offsetBy: "<|endofimg|>".count)
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
            case 151_664: "<|img|>"
            case 151_665: "<|imgpad|>"
            case 151_666: "<|endofimg|>"
            default: String(UnicodeScalar(id)!)
            }
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<|img|>": 151_664
        case "<|imgpad|>": 151_665
        case "<|endofimg|>": 151_666
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case 151_664: "<|img|>"
        case 151_665: "<|imgpad|>"
        case 151_666: "<|endofimg|>"
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
            text: "<|user|><|img|><|imgpad|><|endofimg|>OCR:<|assistant|>",
            addSpecialTokens: false,
        )
    }
}

final class DotsOCRTests: XCTestCase {
    func testProcessorExpandsDotsImagePlaceholder() async throws {
        let data = """
            {
              "processor_class": "DotsVLProcessor",
              "min_pixels": 3136,
              "max_pixels": 3136,
              "patch_size": 14,
              "temporal_patch_size": 1,
              "merge_size": 2,
              "image_mean": [0.48145466, 0.4578275, 0.40821073],
              "image_std": [0.26862954, 0.26130258, 0.27577711]
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(DotsOCRProcessorConfiguration.self, from: data)
        let processor = DotsOCRProcessor(config, tokenizer: DotsOCRTestTokenizer())
        let image = CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 56, height: 56))
        let input = UserInput(chat: [.user("OCR:", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 151_665 }), 4)
        XCTAssertEqual(tokens.count(where: { $0 == 151_664 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 151_666 }), 1)
        XCTAssertEqual(prepared.image?.pixels.shape, [16, 588])
        XCTAssertEqual(prepared.image?.frames?.first?.t, 1)
        XCTAssertEqual(prepared.image?.frames?.first?.h, 4)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 4)
    }

    func testSanitizeMapsDotsKeysAndTransposesPatchEmbedding() throws {
        let config = try Self.smallConfig()
        let model = DotsOCR(config)

        let weights = model.sanitize(weights: [
            "model.vision_tower.patch_embed.patchifier.proj.weight": MLXArray.zeros([
                6, 3, 2, 2,
            ]),
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray([1, 2]),
            "lm_head.weight": MLXArray.zeros([32, 8]),
        ])

        XCTAssertEqual(
            weights["vision_tower.patch_embed.patchifier.proj.weight"]?.shape,
            [6, 2, 2, 3],
        )
        XCTAssertNil(weights["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertEqual(weights["language_model.model.lm_head.weight"]?.shape, [32, 8])
    }

    private static func smallConfig() throws -> DotsOCRConfiguration {
        let data = """
            {
              "model_type": "dots_ocr",
              "image_token_id": 9,
              "video_token_id": 10,
              "text_config": {
                "model_type": "dots_ocr",
                "hidden_size": 8,
                "num_hidden_layers": 1,
                "intermediate_size": 16,
                "num_attention_heads": 2,
                "num_key_value_heads": 2,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000,
                "attention_bias": true,
                "tie_word_embeddings": false,
                "vocab_size": 32
              },
              "vision_config": {
                "model_type": "dots_vit",
                "embed_dim": 6,
                "hidden_size": 8,
                "intermediate_size": 12,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_channels": 3,
                "patch_size": 2,
                "post_norm": true,
                "rms_norm_eps": 1e-5,
                "spatial_merge_size": 2,
                "temporal_patch_size": 1,
                "use_bias": false
              },
              "vocab_size": 32
            }
            """.data(using: .utf8)!
        return try JSONDecoder.json5().decode(DotsOCRConfiguration.self, from: data)
    }
}
