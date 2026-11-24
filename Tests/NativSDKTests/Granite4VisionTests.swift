import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct Granite4VisionTestTokenizer: Tokenizer {
    let imageTokenID = 100_352

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
        throw TokenizerError.missingChatTemplate
    }
}

final class Granite4VisionTests: XCTestCase {
    func testConfigurationDecodesGranite4VisionShape() throws {
        let data = """
            {
              "model_type": "granite4_vision",
              "image_token_index": 100352,
              "downsample_rate": "4/8",
              "deepstack_layer_map": [[-19, 9], [-13, 6], [-7, 3], [-1, 0]],
              "spatial_target_layers": [12, 15, 18, 21],
              "text_config": {
                "model_type": "granitemoehybrid",
                "hidden_size": 2560,
                "num_hidden_layers": 40,
                "intermediate_size": 8192,
                "shared_intermediate_size": 8192,
                "num_attention_heads": 40,
                "num_key_value_heads": 8,
                "rms_norm_eps": 1e-5,
                "vocab_size": 100353,
                "rope_theta": 10000000,
                "logits_scaling": 10.0
              },
              "vision_config": {
                "model_type": "siglip_vision_model",
                "hidden_size": 1152,
                "num_hidden_layers": 27,
                "intermediate_size": 4304,
                "num_attention_heads": 16,
                "image_size": 384,
                "patch_size": 16
              }
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder.json5().decode(Granite4VisionConfiguration.self, from: data)

        XCTAssertEqual(config.modelType, "granite4_vision")
        XCTAssertEqual(config.imageTokenIndex, 100_352)
        XCTAssertEqual(config.downsampleRate, "4/8")
        XCTAssertEqual(config.deepstackLayerMap.count, 4)
        XCTAssertEqual(config.spatialTargetLayers, [12, 15, 18, 21])
        XCTAssertEqual(config.textConfiguration.modelType, "granitemoehybrid")
        XCTAssertEqual(config.textConfiguration.sharedIntermediateSize, 8192)
        XCTAssertTrue(config.textConfiguration.useSharedMLP)
        XCTAssertEqual(config.textConfiguration.logitsScaling, 10)
        XCTAssertEqual(config.visionConfiguration.patchSize, 16)
    }

    func testProcessorBuildsDownsampledAnyresImageTokens() async throws {
        let data = """
            {
              "processor_class": "Granite4VisionProcessor",
              "crop_size": {"height": 32, "width": 32},
              "size": {"height": 32, "width": 32},
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "image_grid_pinpoints": [[32, 32]],
              "patch_size": 16,
              "downsample_rate": "1/2"
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(
            Granite4VisionProcessorConfiguration.self,
            from: data,
        )
        let processor = Granite4VisionProcessor(config, tokenizer: Granite4VisionTestTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 20, height: 12))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 100_352 }), 3)
        XCTAssertEqual(prepared.image?.pixels.shape, [2, 3, 32, 32])
        XCTAssertEqual(prepared.video?.pixels.shape, [1, 2])
        XCTAssertEqual(prepared.video?.pixels.asArray(Int32.self).map(Int.init), [12, 20])
    }

    func testSanitizeMapsGranite4ReferenceKeys() throws {
        let config = try JSONDecoder.json5().decode(
            Granite4VisionConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "granite4_vision",
                  "image_token_index": 9,
                  "downsample_rate": "1/2",
                  "deepstack_layer_map": [[-1, 0]],
                  "use_spatial_sampling": false,
                  "text_config": {
                    "hidden_size": 16,
                    "num_hidden_layers": 1,
                    "intermediate_size": 32,
                    "shared_intermediate_size": 32,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "rms_norm_eps": 1e-5,
                    "vocab_size": 16
                  },
                  "vision_config": {
                    "hidden_size": 128,
                    "num_hidden_layers": 1,
                    "intermediate_size": 256,
                    "num_attention_heads": 2,
                    "image_size": 32,
                    "patch_size": 16,
                    "num_channels": 3
                  }
                }
                """.data(using: .utf8)),
        )
        let model = Granite4Vision(config)

        let sanitized = model.sanitize(weights: [
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray([1, 2]),
            "model.language_model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([16, 16]),
            "lm_head.weight": MLXArray.zeros([16, 16]),
            "vision_tower.vision_model.embeddings.patch_embedding.weight": MLXArray.zeros([
                128, 3, 16, 16,
            ]),
            "vision_tower.vision_model.head.attention.in_proj_weight": MLXArray.zeros([384, 128]),
            "vision_tower.vision_model.head.attention.in_proj_bias": MLXArray.zeros([384]),
        ])

        XCTAssertNil(sanitized["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(sanitized["language_model.model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(sanitized["language_model.lm_head.weight"])
        XCTAssertEqual(
            sanitized["vision_tower.vision_model.embeddings.patch_embedding.weight"]?.shape,
            [128, 16, 16, 3],
        )
        XCTAssertNotNil(sanitized["vision_tower.vision_model.head.attention.in_proj.weight"])
        XCTAssertNotNil(sanitized["vision_tower.vision_model.head.attention.in_proj.bias"])
    }
}
