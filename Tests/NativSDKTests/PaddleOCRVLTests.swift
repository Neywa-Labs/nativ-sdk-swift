import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct PaddleOCRVLTestTokenizer: Tokenizer {
    var bosToken: String? {
        "<|begin_of_sentence|>"
    }

    var eosToken: String? {
        "</s>"
    }

    var unknownToken: String? {
        nil
    }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        var output: [Int] = []
        var index = text.startIndex
        while index < text.endIndex {
            if text[index...].hasPrefix("<|begin_of_sentence|>") {
                output.append(100_273)
                index = text.index(index, offsetBy: "<|begin_of_sentence|>".count)
            } else if text[index...].hasPrefix("<|IMAGE_START|>") {
                output.append(101_305)
                index = text.index(index, offsetBy: "<|IMAGE_START|>".count)
            } else if text[index...].hasPrefix("<|IMAGE_PLACEHOLDER|>") {
                output.append(100_295)
                index = text.index(index, offsetBy: "<|IMAGE_PLACEHOLDER|>".count)
            } else if text[index...].hasPrefix("<|IMAGE_END|>") {
                output.append(101_306)
                index = text.index(index, offsetBy: "<|IMAGE_END|>".count)
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
            case 100_273: "<|begin_of_sentence|>"
            case 101_305: "<|IMAGE_START|>"
            case 100_295: "<|IMAGE_PLACEHOLDER|>"
            case 101_306: "<|IMAGE_END|>"
            default: String(UnicodeScalar(id)!)
            }
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<|begin_of_sentence|>": 100_273
        case "<|IMAGE_START|>": 101_305
        case "<|IMAGE_PLACEHOLDER|>": 100_295
        case "<|IMAGE_END|>": 101_306
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case 100_273: "<|begin_of_sentence|>"
        case 101_305: "<|IMAGE_START|>"
        case 100_295: "<|IMAGE_PLACEHOLDER|>"
        case 101_306: "<|IMAGE_END|>"
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
            text:
                "<|begin_of_sentence|>User: <|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>Describe.\nAssistant:\n",
            addSpecialTokens: false,
        )
    }
}

final class PaddleOCRVLTests: XCTestCase {
    func testProcessorExpandsPaddleImagePlaceholder() async throws {
        let data = """
            {
              "processor_class": "PaddleOCRVLProcessor",
              "min_pixels": 3136,
              "max_pixels": 3136,
              "patch_size": 14,
              "temporal_patch_size": 1,
              "merge_size": 2,
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5]
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(
            PaddleOCRVLProcessorConfiguration.self,
            from: data,
        )
        let processor = PaddleOCRVLProcessor(config, tokenizer: PaddleOCRVLTestTokenizer())
        let image = CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 56, height: 56))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 100_295 }), 4)
        XCTAssertEqual(tokens.count(where: { $0 == 101_305 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 101_306 }), 1)
        XCTAssertEqual(prepared.image?.pixels.shape, [16, 3, 14, 14])
        XCTAssertEqual(prepared.image?.frames?.first?.t, 1)
        XCTAssertEqual(prepared.image?.frames?.first?.h, 4)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 4)
    }

    func testSanitizePreservesConvertedQKVAndMapsRawKeys() throws {
        let model = try PaddleOCRVL(Self.smallConfig())
        let rawQ = MLXArray.zeros([6, 4])
        let rawK = MLXArray.ones([6, 4])
        let rawV = MLXArray.ones([6, 4]) * Float32(2)

        let weights = model.sanitize(weights: [
            "visual.layers.0.self_attn.qkv.weight": MLXArray.zeros([18, 4]),
            "visual.layers.0.self_attn.out_proj.weight": MLXArray.zeros([4, 4]),
            "visual.vision_model.encoder.layers.0.self_attn.q_proj.weight": rawQ,
            "visual.vision_model.encoder.layers.0.self_attn.k_proj.weight": rawK,
            "visual.vision_model.encoder.layers.0.self_attn.v_proj.weight": rawV,
            "model.embed_tokens.weight": MLXArray.zeros([32, 8]),
            "lm_head.weight": MLXArray.zeros([32, 8]),
            "visual.embeddings.patch_embedding.weight": MLXArray.zeros([6, 3, 2, 2]),
        ])

        XCTAssertEqual(weights["visual.layers.0.self_attn.qkv.weight"]?.shape, [18, 4])
        XCTAssertEqual(
            weights["visual.layers.0.self_attn.out_proj.weight"]?.shape,
            [4, 4],
        )
        XCTAssertEqual(
            weights["visual.layers.0.self_attn.qkv.weight"]?.shape,
            [18, 4],
        )
        XCTAssertEqual(weights["language_model.model.embed_tokens.weight"]?.shape, [32, 8])
        XCTAssertEqual(weights["language_model.lm_head.weight"]?.shape, [32, 8])
        XCTAssertEqual(weights["visual.embeddings.patch_embedding.weight"]?.shape, [6, 2, 2, 3])
    }

    private static func smallConfig() throws -> PaddleOCRVLConfiguration {
        let data = """
            {
              "model_type": "paddleocr_vl",
              "hidden_size": 8,
              "num_hidden_layers": 1,
              "intermediate_size": 16,
              "num_attention_heads": 2,
              "num_key_value_heads": 1,
              "vocab_size": 32,
              "head_dim": 4,
              "use_bias": false,
              "rope_theta": 500000,
              "rope_scaling": {"mrope_section": [1, 1, 0], "type": "default"},
              "image_token_id": 9,
              "video_token_id": 10,
              "vision_start_token_id": 11,
              "vision_end_token_id": 12,
              "vision_config": {
                "model_type": "paddleocr_vl",
                "hidden_size": 6,
                "intermediate_size": 12,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_channels": 3,
                "image_size": 28,
                "patch_size": 2,
                "layer_norm_eps": 1e-6,
                "spatial_merge_size": 2
              },
              "vocab_size": 32
            }
            """.data(using: .utf8)!
        return try JSONDecoder.json5().decode(PaddleOCRVLConfiguration.self, from: data)
    }
}
