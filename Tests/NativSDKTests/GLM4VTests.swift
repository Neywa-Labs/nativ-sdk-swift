import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct GLM4VTestTokenizer: Tokenizer {
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
            if text[index...].hasPrefix("[gMASK]") {
                output.append(151_330)
                index = text.index(index, offsetBy: "[gMASK]".count)
            } else if text[index...].hasPrefix("<sop>") {
                output.append(151_331)
                index = text.index(index, offsetBy: "<sop>".count)
            } else if text[index...].hasPrefix("<|user|>") {
                output.append(151_336)
                index = text.index(index, offsetBy: "<|user|>".count)
            } else if text[index...].hasPrefix("<|assistant|>") {
                output.append(151_337)
                index = text.index(index, offsetBy: "<|assistant|>".count)
            } else if text[index...].hasPrefix("<|begin_of_image|>") {
                output.append(151_339)
                index = text.index(index, offsetBy: "<|begin_of_image|>".count)
            } else if text[index...].hasPrefix("<|end_of_image|>") {
                output.append(151_340)
                index = text.index(index, offsetBy: "<|end_of_image|>".count)
            } else if text[index...].hasPrefix("<|image|>") {
                output.append(151_343)
                index = text.index(index, offsetBy: "<|image|>".count)
            } else if text[index...].hasPrefix("<|endoftext|>") {
                output.append(151_329)
                index = text.index(index, offsetBy: "<|endoftext|>".count)
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
            case 151_329: "<|endoftext|>"
            case 151_330: "[gMASK]"
            case 151_331: "<sop>"
            case 151_336: "<|user|>"
            case 151_337: "<|assistant|>"
            case 151_339: "<|begin_of_image|>"
            case 151_340: "<|end_of_image|>"
            case 151_343: "<|image|>"
            default: String(UnicodeScalar(id)!)
            }
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<|endoftext|>": 151_329
        case "[gMASK]": 151_330
        case "<sop>": 151_331
        case "<|user|>": 151_336
        case "<|assistant|>": 151_337
        case "<|begin_of_image|>": 151_339
        case "<|end_of_image|>": 151_340
        case "<|image|>": 151_343
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case 151_329: "<|endoftext|>"
        case 151_330: "[gMASK]"
        case 151_331: "<sop>"
        case 151_336: "<|user|>"
        case 151_337: "<|assistant|>"
        case 151_339: "<|begin_of_image|>"
        case 151_340: "<|end_of_image|>"
        case 151_343: "<|image|>"
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
                "[gMASK]<sop>\n<|user|>\n<|begin_of_image|><|image|><|end_of_image|>Describe.\n<|assistant|>\n",
            addSpecialTokens: false,
        )
    }
}

final class GLM4VTests: XCTestCase {
    func testProcessorExpandsGLM4VImagePlaceholder() async throws {
        let data = """
            {
              "processor_class": "Glm4vProcessor",
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "merge_size": 2,
              "patch_size": 14,
              "temporal_patch_size": 2,
              "size": {"shortest_edge": 3136, "longest_edge": 3136}
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(GLM4VProcessorConfiguration.self, from: data)
        let processor = GLM4VProcessor(config, tokenizer: GLM4VTestTokenizer())
        let image = CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 56, height: 56))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 151_343 }), 4)
        XCTAssertEqual(tokens.count(where: { $0 == 151_339 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 151_340 }), 1)
        XCTAssertEqual(prepared.image?.pixels.shape, [16, 1176])
        XCTAssertEqual(prepared.image?.frames?.first?.t, 1)
        XCTAssertEqual(prepared.image?.frames?.first?.h, 4)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 4)
    }

    func testSanitizeMapsGLM4VKeysAndTransposesConvs() throws {
        let model = try GLM4V(Self.smallConfig())
        let weights = model.sanitize(weights: [
            "model.visual.embeddings.position_embedding.weight": MLXArray.zeros([4, 4]),
            "model.visual.downsample.weight": MLXArray.zeros([8, 4, 2, 2]),
            "model.language_model.layers.0.self_attn.q_proj.bias": MLXArray.zeros([4]),
            "model.language_model.layers.1.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
            "lm_head.weight": MLXArray.zeros([32, 8]),
        ])

        XCTAssertEqual(
            weights["vision_tower.embeddings.position_embedding.weight"]?.shape,
            [4, 4],
        )
        XCTAssertEqual(weights["vision_tower.downsample.weight"]?.shape, [8, 2, 2, 4])
        XCTAssertEqual(
            weights["language_model.model.layers.0.self_attn.q_proj.bias"]?.shape,
            [4],
        )
        XCTAssertNil(weights["language_model.model.layers.1.self_attn.q_proj.weight"])
        XCTAssertEqual(weights["language_model.lm_head.weight"]?.shape, [32, 8])
    }

    private static func smallConfig() throws -> GLM4VConfiguration {
        let data = """
            {
              "model_type": "glm4v",
              "image_token_id": 9,
              "video_token_id": 10,
              "text_config": {
                "vocab_size": 32,
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "num_key_value_heads": 1,
                "rms_norm_eps": 1e-5,
                "rope_theta": 10000,
                "attention_bias": true,
                "partial_rotary_factor": 0.5,
                "rope_scaling": {"rope_type": "default", "mrope_section": [1, 0, 0]},
                "tie_word_embeddings": false
              },
              "vision_config": {
                "hidden_size": 4,
                "intermediate_size": 10,
                "depth": 1,
                "num_heads": 2,
                "image_size": 4,
                "patch_size": 2,
                "out_hidden_size": 8,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2,
                "in_channels": 3,
                "rms_norm_eps": 1e-5
              }
            }
            """.data(using: .utf8)!
        return try JSONDecoder.json5().decode(GLM4VConfiguration.self, from: data)
    }
}
