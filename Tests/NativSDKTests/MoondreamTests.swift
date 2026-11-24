import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct MoondreamTestTokenizer: Tokenizer {
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

final class MoondreamTests: XCTestCase {
    func testConfigurationDecodesCurrentOfficialConfigWithDefaults() throws {
        let data = """
            {
              "architectures": ["HfMoondream"],
              "config": {},
              "model_type": "moondream1",
              "torch_dtype": "bfloat16"
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder.json5().decode(MoondreamConfiguration.self, from: data)

        XCTAssertEqual(config.modelType, "moondream1")
        XCTAssertEqual(config.textConfiguration.hiddenSize, 2048)
        XCTAssertEqual(config.textConfiguration.vocabularySize, 51200)
        XCTAssertEqual(config.visionConfiguration.cropSize, 378)
        XCTAssertEqual(config.visionTokenCount, 729)
        XCTAssertEqual(config.visionPrefixLength, 730)
    }

    func testProcessorBuildsVisionPrefixAndCropMetadata() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            MoondreamProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "MoondreamProcessor",
                  "crop_size": 28,
                  "patch_size": 14,
                  "max_crops": 2,
                  "overlap_margin": 0,
                  "tokenizer": {
                    "bos_id": 0,
                    "templates": {
                      "query": {
                        "prefix": [1, 15381, 2],
                        "suffix": [3]
                      }
                    }
                  }
                }
                """.data(using: .utf8)),
        )
        let processor = MoondreamProcessor(processorConfig, tokenizer: MoondreamTestTokenizer())
        let image = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 20, height: 12))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(Array(tokens.prefix(5)), [0, 0, 0, 0, 0])
        XCTAssertEqual(Array(tokens[5 ..< 8]), [1, 15381, 2])
        XCTAssertEqual(Array(tokens.suffix(2)), [3, 3])
        XCTAssertEqual(prepared.image?.pixels.shape, [2, 28, 28, 3])
        XCTAssertEqual(prepared.video?.pixels.asArray(Int32.self).map(Int.init), [2, 1, 1])
    }

    func testSanitizeMapsCurrentAndLegacyCheckpointKeys() throws {
        let config = try JSONDecoder.json5().decode(
            MoondreamConfiguration.self,
            from: XCTUnwrap(#"{"model_type": "moondream1", "config": {}}"#.data(using: .utf8)),
        )
        let model = Moondream(config)
        let weights: [String: MLXArray] = [
            "model.text.wte": MLXArray.zeros([8, 4]),
            "model.text.blocks.0.attn.qkv.weight": MLXArray.zeros([12, 4]),
            "model.text.lm_head.weight": MLXArray.zeros([8, 4]),
            "model.vision.patch_emb.weight": MLXArray.zeros([4, 4]),
            "model.vision.proj_mlp.fc1.weight": MLXArray.zeros([4, 8]),
            "model.region.coord_encoder.weight": MLXArray.zeros([4, 4]),
            "text_model.transformer.h.0.mixer.Wqkv.weight": MLXArray.zeros([12, 4]),
            "vision_encoder.encoder.model.visual.patch_embed.linear.weight": MLXArray.zeros([4, 4]),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertNotNil(sanitized["text.model.wte.weight"])
        XCTAssertNotNil(sanitized["text.model.blocks.0.attn.qkv.weight"])
        XCTAssertNotNil(sanitized["text.lm_head.weight"])
        XCTAssertNotNil(sanitized["vision.encoder.patch_emb.weight"])
        XCTAssertNotNil(sanitized["vision.proj_mlp.fc1.weight"])
        XCTAssertNotNil(sanitized["vision.encoder.patch_emb.weight"])
        XCTAssertNil(sanitized["model.region.coord_encoder.weight"])
        XCTAssertNil(sanitized["region.coord_encoder.weight"])
    }
}
