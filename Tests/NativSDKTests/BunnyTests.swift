import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct BunnyTestTokenizer: Tokenizer {
    private let tokenIDs: [String: Int] = [
        "<|begin_of_text|>": 128_000,
        "<|end_of_text|>": 128_001,
        "<|start_header_id|>": 128_006,
        "<|end_header_id|>": 128_007,
        "<|eot_id|>": 128_009,
    ]

    var bosToken: String? {
        "<|begin_of_text|>"
    }

    var eosToken: String? {
        "<|end_of_text|>"
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
        []
    }
}

final class BunnyTests: XCTestCase {
    func testConfigurationDecodesRootLevelBunnyShape() throws {
        let data = """
            {
              "model_type": "bunny-llama",
              "hidden_size": 4096,
              "num_hidden_layers": 32,
              "intermediate_size": 14336,
              "num_attention_heads": 32,
              "num_key_value_heads": 8,
              "rms_norm_eps": 0.00001,
              "vocab_size": 128256,
              "rope_theta": 500000,
              "attention_bias": false,
              "tie_word_embeddings": false,
              "image_aspect_ratio": "pad",
              "tokenizer_model_max_length": 2048,
              "mm_hidden_size": 1152
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder.json5().decode(BunnyConfiguration.self, from: data)

        XCTAssertEqual(config.modelType, "bunny-llama")
        XCTAssertEqual(config.hiddenSize, 4096)
        XCTAssertEqual(config.kvHeads, 8)
        XCTAssertEqual(config.vocabularySize, 128_256)
        XCTAssertEqual(config.ropeTheta, 500_000, accuracy: 0.01)
        XCTAssertFalse(config.attentionBias)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertEqual(config.imageTokenIndex, -200)
        XCTAssertEqual(config.imageAspectRatio, "pad")
        XCTAssertEqual(config.tokenizerModelMaxLength, 2048)
        XCTAssertEqual(config.visionConfiguration.modelType, "siglip_vision_model")
        XCTAssertEqual(config.visionConfiguration.hiddenSize, 1152)
        XCTAssertEqual(config.visionConfiguration.imageSize, 384)
    }

    func testProcessorInsertsImageSentinelAndPreprocessesImage() async throws {
        let config = try JSONDecoder.json5().decode(
            BunnyProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "bunny-llama",
                  "image_token_index": -200,
                  "image_aspect_ratio": "pad",
                  "vision_config": {
                    "image_size": 56,
                    "patch_size": 14
                  }
                }
                """.data(using: .utf8)),
        )
        let processor = BunnyProcessor(config, tokenizer: BunnyTestTokenizer())
        let image = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 16, height: 10))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == -200 }), 1)
        XCTAssertTrue(tokens.contains(128_000))
        XCTAssertTrue(tokens.contains(128_006))
        XCTAssertEqual(prepared.image?.pixels.shape, [1, 3, 56, 56])
    }

    func testSanitizeKeepsBunnyKeyLayoutDropsUnusedHeadAndTransposesPatchEmbedding() throws {
        let config = try JSONDecoder.json5().decode(
            BunnyConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "bunny-llama",
                  "hidden_size": 8,
                  "num_hidden_layers": 1,
                  "intermediate_size": 16,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 2,
                  "rms_norm_eps": 0.00001,
                  "vocab_size": 16,
                  "rope_theta": 500000,
                  "attention_bias": false,
                  "tie_word_embeddings": false,
                  "vision_config": {
                    "model_type": "siglip_vision_model",
                    "hidden_size": 6,
                    "num_hidden_layers": 1,
                    "intermediate_size": 12,
                    "num_attention_heads": 2,
                    "image_size": 4,
                    "patch_size": 2,
                    "num_channels": 3,
                    "layer_norm_eps": 0.000001
                  }
                }
                """.data(using: .utf8)),
        )
        let model = BunnyLlama(config)

        let sanitized = model.sanitize(weights: [
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([4]),
            "language_model.model.lm_head.weight": MLXArray.zeros([16, 8]),
            "mm_projector.linear_1.weight": MLXArray.zeros([8, 6]),
            "vision_tower.vision_tower.vision_model.head.probe": MLXArray.zeros([1, 1, 6]),
            "vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight":
                MLXArray.zeros([6, 3, 2, 2]),
        ])

        XCTAssertNil(sanitized["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNotNil(sanitized["language_model.model.lm_head.weight"])
        XCTAssertNotNil(sanitized["mm_projector.linear_1.weight"])
        XCTAssertNil(sanitized["vision_tower.vision_tower.vision_model.head.probe"])
        XCTAssertEqual(
            sanitized["vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight"]?
                .shape,
            [6, 2, 2, 3],
        )
    }
}
