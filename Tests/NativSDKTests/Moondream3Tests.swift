import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct Moondream3TestTokenizer: Tokenizer {
    var bosToken: String? {
        "<|endoftext|>"
    }

    var eosToken: String? {
        "<|endoftext|>"
    }

    var unknownToken: String? {
        nil
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let payload = text.unicodeScalars.map { Int($0.value) % 100 + 10 }
        return addSpecialTokens ? [0] + payload : payload
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        token == "<|endoftext|>" ? 0 : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        id == 0 ? "<|endoftext|>" : nil
    }

    func applyChatTemplate(
        messages _: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?,
    )
        throws -> [Int]
    {
        encode(text: "Describe.", addSpecialTokens: true)
    }
}

final class Moondream3Tests: XCTestCase {
    func testConfigurationDecodesPreviewConfigWithDefaults() throws {
        let data = """
            {
              "architectures": ["HfMoondream"],
              "config": {
                "skills": ["query", "caption", "detect", "point"]
              },
              "model_type": "moondream3",
              "quantization": {
                "group_size": 64,
                "bits": 4,
                "mode": "affine"
              }
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder.json5().decode(Moondream3Configuration.self, from: data)

        XCTAssertEqual(config.modelType, "moondream3")
        XCTAssertEqual(config.textConfiguration.hiddenSize, 2048)
        XCTAssertEqual(config.textConfiguration.numExperts, 64)
        XCTAssertEqual(config.textConfiguration.numExpertsPerToken, 8)
        XCTAssertEqual(config.textConfiguration.moeStartLayer, 4)
        XCTAssertEqual(config.textConfiguration.ropeTheta, 1_500_000)
        XCTAssertEqual(config.visionConfiguration.cropSize, 378)
        XCTAssertEqual(config.visionTokenCount, 729)
        XCTAssertEqual(config.visionPrefixLength, 730)
    }

    func testProcessorBuildsVisionPrefixAndCropMetadata() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            Moondream3ProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "Moondream3Processor",
                  "crop_size": 28,
                  "patch_size": 14,
                  "max_crops": 2,
                  "overlap_margin": 0,
                  "bos_token_id": 0,
                  "answer_token_id": 3
                }
                """.data(using: .utf8)),
        )
        let processor = Moondream3Processor(processorConfig, tokenizer: Moondream3TestTokenizer())
        let image = CIImage(color: .green).cropped(to: CGRect(x: 0, y: 0, width: 20, height: 12))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(Array(tokens.prefix(5)), [0, 0, 0, 0, 0])
        XCTAssertEqual(tokens.last, 3)
        XCTAssertGreaterThan(tokens.count, 10)
        XCTAssertEqual(prepared.image?.pixels.shape, [2, 28, 28, 3])
        XCTAssertEqual(prepared.video?.pixels.asArray(Int32.self).map(Int.init), [2, 1, 1])
    }

    func testSanitizeMapsOfficialAndMLXCheckpointKeys() throws {
        let config = try JSONDecoder.json5().decode(
            Moondream3Configuration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "moondream3",
                  "text_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 2,
                    "vocab_size": 32,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "head_dim": 4,
                    "rope_dim": 2,
                    "num_experts": 4,
                    "num_experts_per_tok": 2,
                    "moe_intermediate_size": 4,
                    "moe_start_layer": 1
                  },
                  "vision_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "patch_size": 14,
                    "crop_size": 28,
                    "proj_inner_dim": 16,
                    "proj_out_dim": 8
                  }
                }
                """.data(using: .utf8)),
        )
        let model = Moondream3(config)
        let weights: [String: MLXArray] = [
            "model.text.wte": MLXArray.zeros([32, 8]),
            "model.text.blocks.1.mlp.router.weight": MLXArray.zeros([4, 8]),
            "model.vision.blocks.0.attn.qkv.weight": MLXArray.zeros([24, 8]),
            "vision.encoder.patch_emb.weight": MLXArray.zeros([8, 588]),
            "text.model.blocks.1.mlp.fc1.scales": MLXArray.zeros([4, 1, 1]),
            "model.region.coord_encoder.weight": MLXArray.zeros([4, 4]),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertNotNil(sanitized["text.model.wte.weight"])
        XCTAssertNotNil(sanitized["text.model.blocks.1.mlp.router.weight"])
        XCTAssertNotNil(sanitized["vision.encoder.blocks.0.attn.qkv.weight"])
        XCTAssertNotNil(sanitized["vision.encoder.patch_emb.weight"])
        XCTAssertNotNil(sanitized["text.model.blocks.1.mlp.fc1.scales"])
        XCTAssertNil(sanitized["region.coord_encoder.weight"])
        XCTAssertNil(sanitized["model.region.coord_encoder.weight"])
    }
}
