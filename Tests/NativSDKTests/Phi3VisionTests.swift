import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct Phi3VisionTestTokenizer: Tokenizer {
    var bosToken: String? {
        "<s>"
    }

    var eosToken: String? {
        "</s>"
    }

    var unknownToken: String? {
        nil
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var tokens = addSpecialTokens ? [1] : []
        tokens.append(contentsOf: text.unicodeScalars.map { Int($0.value) })
        return tokens
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.compactMap { UnicodeScalar($0).map(String.init) }.joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        token == "<s>" ? 1 : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        id == 1 ? "<s>" : nil
    }

    func applyChatTemplate(
        messages _: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?,
    )
        throws -> [Int]
    {
        encode(text: "<|user|>\nDescribe.<|end|>\n<|assistant|>\n", addSpecialTokens: true)
    }
}

final class Phi3VisionTests: XCTestCase {
    func testConfigurationDecodesPhi35VisionShape() throws {
        let data = """
            {
              "model_type": "phi3_v",
              "hidden_size": 3072,
              "num_hidden_layers": 32,
              "intermediate_size": 8192,
              "num_attention_heads": 32,
              "num_key_value_heads": 32,
              "rms_norm_eps": 1e-5,
              "vocab_size": 32064,
              "rope_theta": 10000.0,
              "max_position_embeddings": 131072,
              "original_max_position_embeddings": 4096,
              "rope_scaling": {
                "type": "su",
                "short_factor": [1.08, 1.1],
                "long_factor": [1.08, 1.11]
              },
              "vision_config": {
                "image_dim_out": 1024,
                "name": "clip_vision_model",
                "num_img_tokens": 144,
                "intermediate_size": 4096
              }
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder.json5().decode(Phi3VisionConfiguration.self, from: data)

        XCTAssertEqual(config.modelType, "phi3_v")
        XCTAssertEqual(config.hiddenSize, 3072)
        XCTAssertEqual(config.kvHeads, 32)
        XCTAssertEqual(config.vocabularySize, 32064)
        XCTAssertEqual(config.ropeScaling?["type"], .string("su"))
        XCTAssertEqual(config.visionConfiguration.modelType, "clip_vision_model")
        XCTAssertEqual(config.visionConfiguration.hiddenLayers, 24)
        XCTAssertEqual(config.visionConfiguration.imageSize, 336)
        XCTAssertEqual(config.visionConfiguration.imageDimOut, 1024)
        XCTAssertEqual(config.visionConfiguration.numImageTokens, 144)
    }

    func testProcessorBuildsHDImageTokensAndNegativePlaceholders() async throws {
        let config = try JSONDecoder.json5().decode(
            Phi3VisionProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "Phi3VProcessor",
                  "image_mean": [0.5, 0.5, 0.5],
                  "image_std": [0.5, 0.5, 0.5],
                  "num_crops": 1,
                  "num_img_tokens": 144
                }
                """.data(using: .utf8)),
        )
        let processor = Phi3VisionProcessor(config, tokenizer: Phi3VisionTestTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 20, height: 12))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == -1 }), 313)
        XCTAssertEqual(prepared.image?.pixels.shape, [1, 2, 3, 336, 336])
        XCTAssertEqual(prepared.video?.pixels.shape, [1, 2])
        XCTAssertEqual(prepared.video?.pixels.asArray(Int32.self).map(Int.init), [336, 336])
    }

    func testSanitizeDropsRotaryAndTransposesClipPatchWeights() throws {
        let config = try JSONDecoder.json5().decode(
            Phi3VisionConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "phi3_v",
                  "hidden_size": 8,
                  "num_hidden_layers": 1,
                  "intermediate_size": 16,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 2,
                  "rms_norm_eps": 1e-5,
                  "vocab_size": 16,
                  "max_position_embeddings": 4096,
                  "original_max_position_embeddings": 4096,
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
        let model = Phi3Vision(config)

        let sanitized = model.sanitize(weights: [
            "model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray([1, 2]),
            "model.vision_embed_tokens.img_processor.vision_model.embeddings.class_embedding":
                MLXArray.zeros([6]),
            "model.vision_embed_tokens.img_processor.vision_model.embeddings.patch_embedding.weight":
                MLXArray.zeros([6, 3, 2, 2]),
            "model.vision_embed_tokens.img_processor.vision_model.embeddings.position_ids":
                MLXArray.zeros([1, 5]),
        ])

        XCTAssertNil(sanitized["model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNil(
            sanitized[
                "model.vision_embed_tokens.img_processor.vision_model.embeddings.position_ids"],
        )
        XCTAssertNotNil(
            sanitized[
                "model.vision_embed_tokens.img_processor.vision_model.embeddings.class_embedding",
            ],
        )
        XCTAssertEqual(
            sanitized[
                "model.vision_embed_tokens.img_processor.vision_model.embeddings.patch_embedding.weight",
            ]?.shape,
            [6, 2, 2, 3],
        )
    }
}
