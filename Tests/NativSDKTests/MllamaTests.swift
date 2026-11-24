import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct MllamaTestTokenizer: Tokenizer {
    private let tokenIDs: [String: Int] = [
        "<|begin_of_text|>": 128_000,
        "<|end_of_text|>": 128_001,
        "<|start_header_id|>": 128_006,
        "<|end_header_id|>": 128_007,
        "<|eom_id|>": 128_008,
        "<|eot_id|>": 128_009,
        "<|image|>": 128_256,
    ]

    var bosToken: String? {
        "<|begin_of_text|>"
    }

    var eosToken: String? {
        "<|eot_id|>"
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

final class MllamaTests: XCTestCase {
    func testConfigurationDecodesMllamaShape() throws {
        let config = try JSONDecoder.json5().decode(
            MllamaConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "mllama",
                  "image_token_index": 128256,
                  "text_config": {
                    "model_type": "mllama_text_model",
                    "vocab_size": 128256,
                    "hidden_size": 4096,
                    "intermediate_size": 14336,
                    "num_hidden_layers": 40,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 8,
                    "rms_norm_eps": 0.00001,
                    "rope_theta": 500000,
                    "rope_scaling": {
                      "rope_type": "llama3",
                      "factor": 8,
                      "low_freq_factor": 1,
                      "high_freq_factor": 4,
                      "original_max_position_embeddings": 8192
                    },
                    "cross_attention_layers": [3, 8, 13, 18, 23, 28, 33, 38]
                  },
                  "vision_config": {
                    "image_size": 560,
                    "patch_size": 14,
                    "num_channels": 3,
                    "hidden_size": 1280,
                    "intermediate_size": 5120,
                    "num_hidden_layers": 32,
                    "attention_heads": 16,
                    "max_num_tiles": 4,
                    "max_aspect_ratio_id": 8,
                    "num_global_layers": 8,
                    "norm_eps": 0.00001,
                    "vision_output_dim": 7680,
                    "intermediate_layers_indices": [3, 7, 15, 23, 30]
                  }
                }
                """.data(using: .utf8)),
        )

        XCTAssertEqual(config.modelType, "mllama")
        XCTAssertEqual(config.imageTokenIndex, 128_256)
        XCTAssertEqual(config.textConfiguration.vocabularySize, 128_256)
        XCTAssertEqual(config.textConfiguration.kvHeads, 8)
        XCTAssertEqual(
            config.textConfiguration.crossAttentionLayers, [3, 8, 13, 18, 23, 28, 33, 38])
        XCTAssertEqual(config.visionConfiguration.imageSize, 560)
        XCTAssertEqual(config.visionConfiguration.attentionHeads, 16)
        XCTAssertEqual(config.visionConfiguration.patchesPerTile, 1601)
        XCTAssertEqual(config.visionConfiguration.outputDimension, 7680)
    }

    func testProcessorBuildsTilesAspectMetadataAndCrossAttentionMask() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            MllamaProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "MllamaProcessor",
                  "image_mean": [0.5, 0.5, 0.5],
                  "image_std": [0.5, 0.5, 0.5],
                  "max_image_tiles": 4,
                  "image_token_index": 128256,
                  "size": {"height": 56, "width": 56}
                }
                """.data(using: .utf8)),
        )
        let processor = MllamaProcessor(processorConfig, tokenizer: MllamaTestTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 24, height: 16))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 128_256 }), 1)
        XCTAssertEqual(prepared.image?.pixels.shape, [1, 1, 4, 3, 56, 56])
        XCTAssertEqual(prepared.image?.frames?.first?.t, 1)
        XCTAssertEqual(prepared.image?.frames?.first?.h, 1)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 4)
        XCTAssertEqual(prepared.video?.pixels.shape, [1, tokens.count, 1, 4])
        XCTAssertGreaterThan(prepared.video?.pixels.asArray(Int32.self).reduce(0, +) ?? 0, 0)
    }

    func testSmallModelInstantiatesAndSanitizeMapsKeys() throws {
        let model = try Mllama(Self.smallConfig())
        let weights = model.sanitize(weights: [
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray.zeros([4]),
            "vision_model.patch_embedding.weight": MLXArray.zeros([8, 3, 2, 2]),
            "language_model.model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([8, 8]),
            "multi_modal_projector.weight": MLXArray.zeros([8, 16]),
        ])

        XCTAssertNil(weights["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertEqual(weights["vision_tower.patch_embedding.weight"]?.shape, [8, 2, 2, 3])
        XCTAssertEqual(
            weights["language_model.model.layers.0.self_attn.q_proj.weight"]?.shape,
            [8, 8],
        )
        XCTAssertEqual(weights["multi_modal_projector.weight"]?.shape, [8, 16])
    }

    private static func smallConfig() throws -> MllamaConfiguration {
        try JSONDecoder.json5().decode(
            MllamaConfiguration.self,
            from: """
                {
                  "model_type": "mllama",
                  "image_token_index": 128256,
                  "text_config": {
                    "vocab_size": 32,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "rms_norm_eps": 0.00001,
                    "rope_theta": 10000,
                    "cross_attention_layers": [1]
                  },
                  "vision_config": {
                    "image_size": 4,
                    "patch_size": 2,
                    "num_channels": 3,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "max_num_tiles": 1,
                    "max_aspect_ratio_id": 1,
                    "num_global_layers": 1,
                    "norm_eps": 0.00001,
                    "vision_output_dim": 16,
                    "intermediate_layers_indices": [0],
                    "supported_aspect_ratios": [[1, 1]]
                  }
                }
                """.data(using: .utf8)!,
        )
    }
}
