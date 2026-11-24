import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct MolmoPointTestTokenizer: Tokenizer {
    private let tokenIDs: [String: Int] = [
        "<im_start>": 151_936,
        "<im_end>": 151_937,
        "<im_patch>": 151_938,
        "<im_col>": 151_939,
        "<low_res_im_start>": 151_940,
        "<|image|>": 151_941,
        "<im_low>": 151_942,
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

final class MolmoPointTests: XCTestCase {
    func testConfigurationDecodesCheckpointShape() throws {
        let data = """
            {
              "model_type": "molmo_point",
              "image_patch_id": 151938,
              "image_non_indexable_patch_id": 151942,
              "patch_token_id": 151947,
              "subpatch_token_id": 151948,
              "location_token_id": 151949,
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
                "hidden_size": 4096,
                "vocab_size": 151936,
                "additional_vocab_size": 128
              }
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder.json5().decode(MolmoPointConfiguration.self, from: data)

        XCTAssertEqual(config.modelType, "molmo_point")
        XCTAssertEqual(config.textConfiguration.hiddenSize, 4096)
        XCTAssertEqual(config.totalVocabularySize, 152_064)
        XCTAssertEqual(config.vitConfiguration.hiddenLayers, 27)
        XCTAssertEqual(config.adapterConfiguration.vitLayers, [-3, -9])
        XCTAssertEqual(config.imagePatchTokenID, 151_938)
        XCTAssertEqual(config.imageNonIndexablePatchTokenID, 151_942)
        XCTAssertEqual(config.patchTokenID, 151_947)
        XCTAssertEqual(config.subpatchTokenID, 151_948)
        XCTAssertEqual(config.locationTokenID, 151_949)
    }

    func testSharedProcessorBuildsMolmoPointLowResTokens() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            Molmo2ProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "Molmo2Processor",
                  "image_use_col_tokens": true,
                  "use_single_crop_col_tokens": false,
                  "use_single_crop_start_token": true,
                  "use_low_res_token_for_global_crops": true,
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
        let processor = Molmo2Processor(processorConfig, tokenizer: MolmoPointTestTokenizer())
        let image = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 20, height: 12))
        let input = UserInput(chat: [.user("Point.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 151_942 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 151_938 }), 1)
        XCTAssertEqual(
            tokens.prefix(8).map(\.self),
            [151_645, 151_940, 151_942, 151_937, 151_936, 151_938, 151_939, 151_937],
        )
        XCTAssertEqual(prepared.image?.pixels.shape, [2, 4, 588])
        XCTAssertEqual(prepared.video?.pixels.shape, [2, 4])
    }

    func testSanitizeDropsRotaryFrequenciesOnly() throws {
        let config = try JSONDecoder.json5().decode(
            MolmoPointConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "molmo_point",
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
                  },
                  "patch_embed_dim": 4
                }
                """.data(using: .utf8)),
        )
        let model = MolmoPoint(config)
        let weights: [String: MLXArray] = [
            "lm.model.blocks.0.self_attn.att_proj.weight": MLXArray.zeros([8, 8]),
            "lm.model.blocks.0.rotary_emb.inv_freq": MLXArray.zeros([4]),
        ]

        let sanitized = model.sanitize(weights: weights)

        XCTAssertNotNil(sanitized["lm.model.blocks.0.self_attn.att_proj.weight"])
        XCTAssertNil(sanitized["lm.model.blocks.0.rotary_emb.inv_freq"])
    }
}
