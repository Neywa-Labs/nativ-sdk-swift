import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct Molmo2TestTokenizer: Tokenizer {
    private let tokenIDs: [String: Int] = [
        "<im_start>": 151_936,
        "<im_end>": 151_937,
        "<im_patch>": 151_938,
        "<im_col>": 151_939,
        "<low_res_im_start>": 151_940,
        "<|image|>": 151_941,
        "<|im_start|>": 151_644,
        "<|im_end|>": 151_645,
    ]

    var bosToken: String? {
        nil
    }

    var eosToken: String? {
        "<|im_end|>"
    }

    var unknownToken: String? {
        nil
    }

    func encode(text _: String, addSpecialTokens _: Bool) -> [Int] {
        [151_941, 151_644, 872, 198]
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map { id in
            convertIdToToken(id) ?? "\(id)"
        }.joined(separator: " ")
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
        [151_941, 151_644, 872, 198]
    }
}

final class Molmo2Tests: XCTestCase {
    func testConfigurationDecodesMolmo2CheckpointShape() throws {
        let data = """
            {
              "model_type": "molmo2",
              "image_patch_id": 151938,
              "vit_config": {
                "num_hidden_layers": 27,
                "image_num_pos": 729,
                "image_default_input_size": [378, 378]
              },
              "adapter_config": {
                "vit_layers": [-3, -9],
                "pooling_attention_mask": true
              },
              "text_config": {
                "vocab_size": 151936,
                "additional_vocab_size": 128
              }
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder.json5().decode(Molmo2Configuration.self, from: data)

        XCTAssertEqual(config.modelType, "molmo2")
        XCTAssertEqual(config.textConfiguration.vocabularySize, 151_936)
        XCTAssertEqual(config.textConfiguration.additionalVocabularySize, 128)
        XCTAssertEqual(config.vitConfiguration.hiddenLayers, 25)
        XCTAssertEqual(config.vitConfiguration.imageNumPatch.0, 27)
        XCTAssertEqual(config.adapterConfiguration.vitLayers, [-3, -9])
        XCTAssertEqual(config.imagePatchTokenID, 151_938)
    }

    func testProcessorConfigurationDecodesTopLevelPreprocessorConfig() throws {
        let data = """
            {
              "processor_class": "Molmo2ImageProcessor",
              "size": {"height": 28, "width": 42},
              "image_mean": [0.1, 0.2, 0.3],
              "image_std": [0.4, 0.5, 0.6],
              "max_crops": 3,
              "overlap_margins": [1, 2],
              "patch_size": 14,
              "pooling_size": [2, 3]
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder.json5().decode(Molmo2ProcessorConfiguration.self, from: data)

        XCTAssertEqual(config.imageProcessor.size.height, 28)
        XCTAssertEqual(config.imageProcessor.size.width, 42)
        XCTAssertEqual(config.imageProcessor.imageMean.map(Double.init), [0.1, 0.2, 0.3])
        XCTAssertEqual(config.imageProcessor.imageStd.map(Double.init), [0.4, 0.5, 0.6])
        XCTAssertEqual(config.imageProcessor.maxCrops, 3)
        XCTAssertEqual(config.imageProcessor.overlapMargins, [1, 2])
        XCTAssertEqual(config.imageProcessor.poolingSize, [2, 3])
    }

    func testProcessorBuildsMolmo2ImageTokensAndPoolingMetadata() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            Molmo2ProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "Molmo2Processor",
                  "image_processor": {
                    "size": {"height": 28, "width": 28},
                    "max_crops": 1,
                    "overlap_margins": [0, 0],
                    "patch_size": 14,
                    "pooling_size": [2, 2],
                    "image_mean": [0.5, 0.5, 0.5],
                    "image_std": [0.5, 0.5, 0.5]
                  }
                }
                """.data(using: .utf8)),
        )
        let processor = Molmo2Processor(processorConfig, tokenizer: Molmo2TestTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 20, height: 12))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 151_938 }), 2)
        XCTAssertEqual(tokens.prefix(4).map(\.self), [151_940, 151_938, 151_939, 151_937])
        XCTAssertEqual(prepared.image?.pixels.shape, [2, 4, 588])
        XCTAssertEqual(prepared.video?.pixels.shape, [2, 4])
        XCTAssertEqual(
            prepared.video?.pixels.asArray(Int32.self).map(Int.init), [0, 1, 2, 3, 4, 5, 6, 7])
    }

    func testSanitizeMapsMolmo2ReferenceKeys() throws {
        let config = try JSONDecoder.json5().decode(
            Molmo2Configuration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "molmo2",
                  "text_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "head_dim": 4,
                    "vocab_size": 16,
                    "additional_vocab_size": 2
                  },
                  "vit_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "head_dim": 4,
                    "image_patch_size": 14,
                    "image_num_pos": 4,
                    "image_default_input_size": [28, 28]
                  },
                  "adapter_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "text_hidden_size": 8,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "head_dim": 4,
                    "vit_layers": [-1]
                  }
                }
                """.data(using: .utf8)),
        )
        let model = Molmo2(config)
        let weights: [String: MLXArray] = [
            "model.transformer.blocks.0.self_attn.att_proj.weight": MLXArray.zeros([8, 8]),
            "model.vision_backbone.image_vit.transformer.resblocks.0.attention.wq.weight":
                MLXArray.zeros([8, 8]),
            "lm_head.weight": MLXArray.zeros([8, 8]),
            "language_model.model.blocks.0.rotary_emb.inv_freq": MLXArray.zeros([4]),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertNotNil(sanitized["language_model.model.blocks.0.self_attn.att_proj.weight"])
        XCTAssertNotNil(sanitized["vision_tower.image_vit.transformer.0.attention.wq.weight"])
        XCTAssertNotNil(sanitized["language_model.lm_head.weight"])
        XCTAssertNil(sanitized["language_model.model.blocks.0.rotary_emb.inv_freq"])
    }
}
