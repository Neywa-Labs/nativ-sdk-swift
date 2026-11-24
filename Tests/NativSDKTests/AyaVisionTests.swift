import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct AyaVisionTestTokenizer: Tokenizer {
    private let tokenIDs: [String: Int] = [
        "<image>": 42,
        "<BOS_TOKEN>": 5,
        "<|START_OF_TURN_TOKEN|>": 255_000,
        "<|END_OF_TURN_TOKEN|>": 255_001,
        "<|USER_TOKEN|>": 255_006,
        "<|CHATBOT_TOKEN|>": 255_007,
        "<|START_OF_IMG|>": 255_033,
        "<|END_OF_IMG|>": 255_034,
        "<|IMG_LINE_BREAK|>": 255_035,
        "<|IMG_PATCH|>": 255_036,
        "TILE_GLOBAL": 255_100,
        "TILE_1": 255_101,
    ]

    var bosToken: String? {
        "<BOS_TOKEN>"
    }

    var eosToken: String? {
        "<|END_OF_TURN_TOKEN|>"
    }

    var unknownToken: String? {
        nil
    }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        let orderedTokens = tokenIDs.keys.sorted { $0.count > $1.count }
        var result: [Int] = []
        var index = text.startIndex
        while index < text.endIndex {
            if let token = orderedTokens.first(where: { text[index...].hasPrefix($0) }),
                let id = tokenIDs[token]
            {
                result.append(id)
                index = text.index(index, offsetBy: token.count)
            } else {
                result.append(Int(text[index].unicodeScalars.first!.value))
                index = text.index(after: index)
            }
        }
        return result
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map { id in
            if let token = convertIdToToken(id) {
                token
            } else if let scalar = UnicodeScalar(id) {
                String(scalar)
            } else {
                "\(id)"
            }
        }
        .joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        tokenIDs[token]
    }

    func convertIdToToken(_ id: Int) -> String? {
        tokenIDs.first { $0.value == id }?.key
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
                "<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|><image>\nDescribe.<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>",
            addSpecialTokens: false,
        )
    }
}

final class AyaVisionTests: XCTestCase {
    func testConfigurationDecodesCheckpointShapeWithDefaults() throws {
        let data = """
            {
              "model_type": "aya_vision",
              "image_token_index": 255036,
              "max_splits_per_img": 12,
              "downsample_factor": 2,
              "alignment_intermediate_size": 28672,
              "vision_feature_layer": -1,
              "vision_feature_select_strategy": "full",
              "text_config": {
                "model_type": "cohere2",
                "hidden_size": 4096,
                "intermediate_size": 14336,
                "logit_scale": 0.25,
                "num_attention_heads": 32,
                "num_hidden_layers": 32,
                "num_key_value_heads": 8,
                "rope_theta": 50000
              },
              "vision_config": {
                "model_type": "siglip_vision_model",
                "hidden_size": 1152,
                "image_size": 364,
                "intermediate_size": 4304,
                "num_attention_heads": 16,
                "num_hidden_layers": 27,
                "patch_size": 14
              }
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder.json5().decode(AyaVisionConfiguration.self, from: data)

        XCTAssertEqual(config.modelType, "aya_vision")
        XCTAssertEqual(config.imageTokenIndex, 255_036)
        XCTAssertEqual(config.downsampleFactor, 2)
        XCTAssertEqual(config.alignmentIntermediateSize, 28672)
        XCTAssertEqual(config.textConfiguration.modelType, "cohere2")
        XCTAssertEqual(config.textConfiguration.vocabularySize, 256_000)
        XCTAssertEqual(config.textConfiguration.headDim, 128)
        XCTAssertEqual(config.textConfiguration.slidingWindowPattern, 4)
        XCTAssertEqual(config.textConfiguration.logitScale, 0.25, accuracy: 0.0001)
        XCTAssertEqual(config.visionConfiguration.modelType, "siglip_vision_model")
        XCTAssertEqual(config.visionConfiguration.imageSize, 364)
    }

    func testProcessorBuildsStructuredImageTokens() async throws {
        let data = """
            {
              "processor_class": "AyaVisionProcessor",
              "img_size": 56,
              "patch_size": 28,
              "size": {"height": 56, "width": 56},
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5]
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(
            AyaVisionProcessorConfiguration.self,
            from: data,
        )
        let processor = AyaVisionProcessor(config, tokenizer: AyaVisionTestTokenizer())
        let image = CIImage(color: .green).cropped(to: CGRect(x: 0, y: 0, width: 12, height: 10))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 255_036 }), 4)
        XCTAssertTrue(tokens.contains(255_033))
        XCTAssertTrue(tokens.contains(255_034))
        XCTAssertTrue(tokens.contains(255_100))
        XCTAssertEqual(prepared.image?.pixels.shape, [1, 3, 56, 56])
    }

    func testProcessorDefaultsFromPreprocessorSidecar() throws {
        let data = """
            {
              "processor_class": "AyaVisionProcessor",
              "size": {"height": 364, "width": 364},
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5]
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder.json5().decode(
            AyaVisionProcessorConfiguration.self,
            from: data,
        )

        XCTAssertEqual(config.processorClass, "AyaVisionProcessor")
        XCTAssertEqual(config.imageSize, 364)
        XCTAssertEqual(config.patchSize, 28)
        XCTAssertEqual(config.imagePatchToken, "<|IMG_PATCH|>")
    }

    func testSanitizeRenamesPrefixesDropsUnusedAndTransposesTorchConv() throws {
        let config = try JSONDecoder.json5().decode(
            AyaVisionConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "aya_vision",
                  "image_token_index": 9,
                  "downsample_factor": 2,
                  "alignment_intermediate_size": 16,
                  "text_config": {
                    "model_type": "cohere2",
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "head_dim": 4,
                    "vocab_size": 16
                  },
                  "vision_config": {
                    "model_type": "siglip_vision_model",
                    "hidden_size": 6,
                    "num_attention_heads": 2,
                    "patch_size": 2,
                    "num_hidden_layers": 1,
                    "intermediate_size": 12,
                    "image_size": 4
                  }
                }
                """.data(using: .utf8)),
        )
        let model = AyaVision(config)

        let sanitized = model.sanitize(weights: [
            "model.vision_tower.vision_model.embeddings.patch_embedding.weight": MLXArray.zeros([
                6, 3, 2, 2,
            ]),
            "model.vision_tower.vision_model.embeddings.position_ids": MLXArray.zeros([4]),
            "model.multi_modal_projector.linear_1.weight": MLXArray.zeros([16, 24]),
            "model.language_model.embed_tokens.weight": MLXArray.zeros([16, 8]),
            "model.language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([
                4
            ]),
        ])

        XCTAssertEqual(
            sanitized["vision_tower.vision_model.embeddings.patch_embedding.weight"]?.shape,
            [6, 2, 2, 3],
        )
        XCTAssertNil(sanitized["vision_tower.vision_model.embeddings.position_ids"])
        XCTAssertNotNil(sanitized["multi_modal_projector.linear_1.weight"])
        XCTAssertNotNil(sanitized["language_model.model.embed_tokens.weight"])
        XCTAssertNil(sanitized["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
    }
}
