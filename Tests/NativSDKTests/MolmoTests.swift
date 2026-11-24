import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct MolmoTestTokenizer: Tokenizer {
    private let tokenIDs: [String: Int] = [
        "<im_start>": 152_064,
        "<im_end>": 152_065,
        "<im_patch>": 152_066,
        "<im_col>": 152_067,
        "<|endoftext|>": 151_643,
    ]

    var bosToken: String? {
        nil
    }

    var eosToken: String? {
        "<|endoftext|>"
    }

    var unknownToken: String? {
        nil
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var result = addSpecialTokens ? [151_643] : []
        result.append(contentsOf: text.unicodeScalars.map { Int($0.value) % 1000 + 10 })
        return result
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map { id in
            convertIdToToken(id) ?? "?"
        }.joined()
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
        encode(text: "Describe.", addSpecialTokens: true)
    }
}

final class MolmoTests: XCTestCase {
    func testConfigurationDecodesMolmoDefaultsFromConvertedCheckpointConfig() throws {
        let data = """
            {
              "model_type": "molmo",
              "vocab_size": 152064,
              "vision_config": {
                "skip_vision_non_divisible": true
              },
              "tie_word_embeddings": false
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder.json5().decode(MolmoConfiguration.self, from: data)

        XCTAssertEqual(config.modelType, "molmo")
        XCTAssertEqual(config.textConfiguration.modelDimension, 3584)
        XCTAssertEqual(config.textConfiguration.vocabularySize, 152_064)
        XCTAssertEqual(config.textConfiguration.additionalVocabularySize, 128)
        XCTAssertEqual(config.visionConfiguration.imageEmbeddingDimension, 1024)
        XCTAssertEqual(config.visionConfiguration.vitLayers, [-2, -9])
        XCTAssertEqual(config.imagePatchTokenID, 152_066)
    }

    func testProcessorBuildsMolmoPatchTokensAndPackedMetadata() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            MolmoProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "MolmoProcessor",
                  "max_crops": 1,
                  "overlap_margins": [0, 0],
                  "base_image_input_size": [28, 28],
                  "image_token_length_w": 1,
                  "image_token_length_h": 1,
                  "image_patch_size": 14,
                  "image_padding_mask": true,
                  "do_normalize": true,
                  "image_mean": [0.5, 0.5, 0.5],
                  "image_std": [0.5, 0.5, 0.5]
                }
                """.data(using: .utf8)),
        )
        let processor = MolmoProcessor(processorConfig, tokenizer: MolmoTestTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 20, height: 12))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 152_066 }), 2)
        XCTAssertEqual(prepared.image?.pixels.shape, [2, 4, 588])
        XCTAssertEqual(prepared.video?.pixels.shape, [2, 5])
    }

    func testSanitizeMapsMolmoReferenceKeys() throws {
        let config = try JSONDecoder.json5().decode(
            MolmoConfiguration.self,
            from: XCTUnwrap(#"{"model_type": "molmo"}"#.data(using: .utf8)),
        )
        let model = Molmo(config)
        let weights: [String: MLXArray] = [
            "model.transformer.blocks.0.att_proj.weight": MLXArray.zeros([8, 8]),
            "model.vision_backbone.image_vit.patch_embedding.weight": MLXArray.zeros([8, 8]),
            "language_model.model.blocks.0.rotary_emb.inv_freq": MLXArray.zeros([4]),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertNotNil(sanitized["language_model.model.blocks.0.att_proj.weight"])
        XCTAssertNotNil(sanitized["vision_tower.image_vit.patch_embedding.weight"])
        XCTAssertNil(sanitized["language_model.model.blocks.0.rotary_emb.inv_freq"])
    }
}
