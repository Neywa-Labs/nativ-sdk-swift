import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct GraniteVisionTestTokenizer: Tokenizer {
    let imageTokenID = 49155

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
        tokenIDs.map { id in
            id == imageTokenID ? "<image>" : String(UnicodeScalar(id)!)
        }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        token == "<image>" ? imageTokenID : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        id == imageTokenID ? "<image>" : nil
    }

    func applyChatTemplate(
        messages _: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?,
    )
        throws -> [Int]
    {
        encode(text: "<image>\nDescribe.", addSpecialTokens: false)
    }
}

final class GraniteVisionTests: XCTestCase {
    func testProcessorBuildsAnyresCropsAndExpandedImageTokens() async throws {
        let data = """
            {
              "processor_class": "LlavaNextProcessor",
              "crop_size": {"height": 28, "width": 28},
              "size": {"height": 28, "width": 28},
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "image_grid_pinpoints": [[28, 28]]
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(
            GraniteVisionProcessorConfiguration.self,
            from: data,
        )
        let processor = GraniteVisionProcessor(config, tokenizer: GraniteVisionTestTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 20, height: 12))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 49155 }), 10)
        XCTAssertEqual(prepared.image?.pixels.shape, [2, 3, 28, 28])
        XCTAssertEqual(prepared.video?.pixels.shape, [1, 2])
        XCTAssertEqual(prepared.video?.pixels.asArray(Int32.self).map(Int.init), [12, 20])
    }

    func testSanitizeDropsRotaryMapsVisionWeightsAndCopiesTiedHead() throws {
        let config = try JSONDecoder.json5().decode(
            GraniteVisionConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "granite_vision",
                  "image_token_index": 9,
                  "vision_feature_layer": [-1, -2],
                  "text_config": {
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "intermediate_size": 16,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "rms_norm_eps": 1e-5,
                    "vocab_size": 16
                  },
                  "vision_config": {
                    "hidden_size": 6,
                    "num_hidden_layers": 2,
                    "intermediate_size": 12,
                    "num_attention_heads": 2,
                    "image_size": 4,
                    "patch_size": 2,
                    "num_channels": 3
                  }
                }
                """.data(using: .utf8)),
        )
        let model = GraniteVision(config)

        let sanitized = model.sanitize(weights: [
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray([1, 2]),
            "language_model.model.embed_tokens.weight": MLXArray.zeros([16, 8]),
            "vision_tower.vision_model.embeddings.patch_embedding.weight": MLXArray.zeros([
                6, 3, 2, 2,
            ]),
            "vision_tower.vision_model.head.attention.in_proj_weight": MLXArray.zeros([18, 6]),
            "vision_tower.vision_model.head.attention.in_proj_bias": MLXArray.zeros([18]),
        ])

        XCTAssertNil(sanitized["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertEqual(
            sanitized["vision_tower.vision_model.embeddings.patch_embedding.weight"]?.shape,
            [6, 2, 2, 3],
        )
        XCTAssertNotNil(sanitized["language_model.lm_head.weight"])
        XCTAssertNotNil(sanitized["vision_tower.vision_model.head.attention.in_proj.weight"])
        XCTAssertNotNil(sanitized["vision_tower.vision_model.head.attention.in_proj.bias"])
    }
}
