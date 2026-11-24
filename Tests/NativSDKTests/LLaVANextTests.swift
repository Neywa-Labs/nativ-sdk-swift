import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct LLaVANextTestTokenizer: Tokenizer {
    let imageTokenID = 32000

    var bosToken: String? {
        "<s>"
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
        encode(text: "<s>[INST] <image>\nDescribe. [/INST]", addSpecialTokens: false)
    }
}

final class LLaVANextTests: XCTestCase {
    func testConfigurationDecodesMistralDefaultsFromCheckpointShape() throws {
        let data = """
            {
              "model_type": "llava_next",
              "image_token_index": 32000,
              "vision_feature_layer": -2,
              "vision_feature_select_strategy": "default",
              "image_grid_pinpoints": [[336, 672], [672, 336]],
              "use_image_newline_parameter": true,
              "text_config": {
                "model_type": "mistral",
                "intermediate_size": 14336,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-5,
                "vocab_size": 32064,
                "rope_theta": 1000000
              },
              "vision_config": {
                "model_type": "clip_vision_model",
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "intermediate_size": 4096,
                "num_attention_heads": 16,
                "image_size": 336,
                "patch_size": 14
              }
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder.json5().decode(LLaVANextConfiguration.self, from: data)

        XCTAssertEqual(config.modelType, "llava_next")
        XCTAssertEqual(config.imageTokenIndex, 32000)
        XCTAssertEqual(config.visionFeatureLayer, -2)
        XCTAssertEqual(config.visionFeatureSelectStrategy, "default")
        XCTAssertEqual(config.textConfiguration.modelType, "mistral")
        XCTAssertEqual(config.textConfiguration.hiddenSize, 4096)
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 32)
        XCTAssertEqual(config.textConfiguration.attentionHeads, 32)
        XCTAssertEqual(config.textConfiguration.kvHeads, 8)
        XCTAssertEqual(config.textConfiguration.vocabularySize, 32064)
        XCTAssertEqual(config.visionConfiguration.modelType, "clip_vision_model")
        XCTAssertTrue(config.useImageNewlineParameter)
    }

    func testProcessorBuildsAnyresCropsAndExpandedImageTokens() async throws {
        let data = """
            {
              "processor_class": "LlavaNextProcessor",
              "crop_size": {"height": 28, "width": 28},
              "size": {"height": 28, "width": 28},
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "image_grid_pinpoints": [[28, 28]],
              "num_additional_image_tokens": 1,
              "patch_size": 14,
              "vision_feature_select_strategy": "default"
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(
            LLaVANextProcessorConfiguration.self,
            from: data,
        )
        let processor = LLaVANextProcessor(config, tokenizer: LLaVANextTestTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 20, height: 12))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 32000 }), 10)
        XCTAssertEqual(prepared.image?.pixels.shape, [2, 3, 28, 28])
        XCTAssertEqual(prepared.video?.pixels.shape, [1, 2])
        XCTAssertEqual(prepared.video?.pixels.asArray(Int32.self).map(Int.init), [12, 20])
    }

    func testSanitizeDropsRotaryAndMapsClipPatchWeights() throws {
        let config = try JSONDecoder.json5().decode(
            LLaVANextConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "llava_next",
                  "image_token_index": 9,
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
        let model = LLaVANext(config)

        let sanitized = model.sanitize(weights: [
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray([1, 2]),
            "vision_tower.vision_model.embeddings.class_embedding": MLXArray.zeros([6]),
            "vision_tower.vision_model.embeddings.patch_embedding.weight": MLXArray.zeros([
                6, 3, 2, 2,
            ]),
            "vision_tower.vision_model.embeddings.position_ids": MLXArray.zeros([1, 5]),
        ])

        XCTAssertNil(sanitized["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNil(sanitized["vision_tower.vision_model.embeddings.position_ids"])
        XCTAssertNotNil(sanitized["vision_tower.vision_model.embeddings.class_embedding"])
        XCTAssertEqual(
            sanitized["vision_tower.vision_model.embeddings.patch_embedding.weight"]?.shape,
            [6, 2, 2, 3],
        )
    }
}
