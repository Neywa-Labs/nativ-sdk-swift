import CoreImage
import Foundation
import MLX
import XCTest

@testable import NativSDK

private struct LocateAnythingTestTokenizer: Tokenizer {
    private let tokenIDs: [String: Int] = [
        "<|im_start|>": 151_644,
        "<|im_end|>": 151_645,
        "<IMG_CONTEXT>": 151_665,
        "<img>": 151_666,
        "</img>": 151_667,
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
            convertIdToToken(id) ?? "\(id)"
        }
        .joined(separator: " ")
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
                "<|im_start|>user\n<img><IMG_CONTEXT></img>Locate.<|im_end|>\n<|im_start|>assistant\n",
            addSpecialTokens: false,
        )
    }
}

final class LocateAnythingTests: XCTestCase {
    func testConfigurationDecodesCheckpointShape() throws {
        let data = """
            {
              "model_type": "locateanything",
              "image_token_index": 151665,
              "eos_token_id": 151645,
              "box_start_token_id": 151668,
              "box_end_token_id": 151669,
              "coord_start_token_id": 151677,
              "coord_end_token_id": 152677,
              "mlp_connector_layers": 2,
              "n_future_tokens": 6,
              "text_config": {
                "model_type": "qwen2",
                "hidden_size": 2048,
                "intermediate_size": 11008,
                "num_hidden_layers": 36,
                "num_attention_heads": 16,
                "num_key_value_heads": 2,
                "vocab_size": 152681,
                "rms_norm_eps": 0.000001,
                "rope_theta": 1000000,
                "tie_word_embeddings": true
              },
              "vision_config": {
                "model_type": "moonvit",
                "hidden_size": 1152,
                "intermediate_size": 4304,
                "num_hidden_layers": 27,
                "num_attention_heads": 16,
                "patch_size": 14,
                "init_pos_emb_height": 64,
                "init_pos_emb_width": 64,
                "merge_kernel_size": [2, 2]
              }
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder.json5().decode(LocateAnythingConfiguration.self, from: data)

        XCTAssertEqual(config.modelType, "locateanything")
        XCTAssertEqual(config.imageTokenIndex, 151_665)
        XCTAssertEqual(config.eosTokenIDs, [151_645])
        XCTAssertEqual(config.boxStartTokenID, 151_668)
        XCTAssertEqual(config.coordEndTokenID, 152_677)
        XCTAssertEqual(config.futureTokens, 6)
        XCTAssertEqual(config.textConfiguration.modelType, "qwen2")
        XCTAssertEqual(config.textConfiguration.hiddenSize, 2048)
        XCTAssertEqual(config.textConfiguration.kvHeads, 2)
        XCTAssertTrue(config.textConfiguration.tieWordEmbeddings)
        XCTAssertEqual(config.visionConfiguration.modelType, "moonvit")
        XCTAssertEqual(config.visionConfiguration.hiddenSize, 1152)
        XCTAssertEqual(config.visionConfiguration.mergeKernelSize, [2, 2])
    }

    func testProcessorPatchPacksAndExpandsImageContextTokens() async throws {
        let data = """
            {
              "processor_class": "LocateAnythingProcessor",
              "image_processor": {
                "patch_size": 14,
                "in_token_limit": 25600,
                "merge_kernel_size": [2, 2],
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.5, 0.5, 0.5]
              }
            }
            """.data(using: .utf8)!
        let config = try JSONDecoder.json5().decode(
            LocateAnythingProcessorConfiguration.self,
            from: data,
        )
        let processor = LocateAnythingProcessor(config, tokenizer: LocateAnythingTestTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 12, height: 10))
        let input = UserInput(chat: [.user("Locate.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 151_665 }), 1)
        XCTAssertTrue(tokens.contains(151_666))
        XCTAssertTrue(tokens.contains(151_667))
        XCTAssertEqual(prepared.image?.pixels.shape, [4, 3, 14, 14])
        XCTAssertEqual(prepared.image?.frames?.first?.t, 1)
        XCTAssertEqual(prepared.image?.frames?.first?.h, 2)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 2)
    }

    func testProcessorDefaultsFromPreprocessorSidecar() throws {
        let data = """
            {
              "processor_class": "LocateAnythingProcessor",
              "patch_size": 14,
              "in_token_limit": 25600,
              "merge_kernel_size": [2, 2],
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5]
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder.json5().decode(
            LocateAnythingProcessorConfiguration.self,
            from: data,
        )

        XCTAssertEqual(config.processorClass, "LocateAnythingProcessor")
        XCTAssertEqual(config.patchSize, 14)
        XCTAssertEqual(config.inTokenLimit, 25600)
        XCTAssertEqual(config.mergeKernelSize, [2, 2])
        XCTAssertEqual(config.imageToken, "<IMG_CONTEXT>")
        XCTAssertEqual(config.imageStartToken, "<img>")
        XCTAssertEqual(config.imageEndToken, "</img>")
    }

    func testSanitizeMapsLegacyKeysDropsTiedLmHeadAndTransposesTorchConv() throws {
        let config = try JSONDecoder.json5().decode(
            LocateAnythingConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "locateanything",
                  "image_token_index": 9,
                  "text_config": {
                    "model_type": "qwen2",
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "vocab_size": 16,
                    "tie_word_embeddings": true
                  },
                  "vision_config": {
                    "model_type": "moonvit",
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "patch_size": 2,
                    "init_pos_emb_height": 2,
                    "init_pos_emb_width": 2,
                    "merge_kernel_size": [2, 2]
                  }
                }
                """.data(using: .utf8)),
        )
        let model = LocateAnything(config)

        let sanitized = model.sanitize(weights: [
            "language_model.lm_head.weight": MLXArray.zeros([16, 8]),
            "vision_model.encoder.patch_embed.proj.weight": MLXArray.zeros([8, 3, 2, 2]),
            "vision_model.encoder.patch_embed.position_ids": MLXArray.zeros([4]),
            "mlp1.0.weight": MLXArray.zeros([32]),
            "mlp1.1.weight": MLXArray.zeros([8, 32]),
            "mlp1.3.weight": MLXArray.zeros([8, 8]),
            "language_model.model.embed_tokens.weight": MLXArray.zeros([16, 8]),
        ])

        XCTAssertNil(sanitized["language_model.lm_head.weight"])
        XCTAssertEqual(sanitized["vision_tower.patch_embed.proj.weight"]?.shape, [8, 2, 2, 3])
        XCTAssertNil(sanitized["vision_tower.patch_embed.position_ids"])
        XCTAssertNotNil(sanitized["multi_modal_projector.layer_norm.weight"])
        XCTAssertNotNil(sanitized["multi_modal_projector.linear_1.weight"])
        XCTAssertNotNil(sanitized["multi_modal_projector.linear_2.weight"])
        XCTAssertNotNil(sanitized["language_model.model.embed_tokens.weight"])
    }
}
