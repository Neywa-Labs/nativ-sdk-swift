import CoreImage
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct JinaVLMTestTokenizer: Tokenizer {
    private let tokenIDs: [String: Int] = [
        "<|image|>": 151_940,
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
        tokenIDs.map { id in convertIdToToken(id) ?? "?" }.joined()
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
        encode(text: "Describe.", addSpecialTokens: false)
    }
}

final class JinaVLMTests: XCTestCase {
    func testConfigurationDecodesCheckpointShape() throws {
        let config = try JSONDecoder.json5().decode(
            JinaVLMConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "jvlm",
                  "text_config": {
                    "hidden_size": 2048,
                    "n_layers": 28,
                    "vocab_size": 151936,
                    "additional_vocab_size": 128,
                    "max_sequence_length": 40960,
                    "rope_theta": 1000000,
                    "block_config": {
                      "attn_config": {
                        "n_heads": 16,
                        "n_kv_heads": 8,
                        "head_dim": 128,
                        "q_lnorm": true
                      },
                      "ffn_config": {"size": 6144},
                      "lnorm_config": {"eps": 0.000001}
                    }
                  },
                  "vision_config": {
                    "hidden_size": 1152,
                    "n_layers": 27,
                    "input_size": [378, 378],
                    "patch_size": 14,
                    "n_channels": 3,
                    "output_size": 2048,
                    "vit_layers": [-4, -10],
                    "post_lnorm": true,
                    "block_config": {
                      "attn_config": {"n_heads": 16, "head_dim": 72, "q_bias": true},
                      "ffn_config": {"size": 4304, "activation_type": "gelu_pytorch_tanh"},
                      "lnorm_config": {"eps": 0.000001}
                    },
                    "vl_connector_config": {
                      "pooling_h": 2,
                      "pooling_w": 2,
                      "mlp_projector_config": {"size": 6144}
                    }
                  }
                }
                """.data(using: .utf8)),
        )

        XCTAssertEqual(config.modelType, "jvlm")
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 28)
        XCTAssertEqual(config.textConfiguration.kvHeads, 8)
        XCTAssertEqual(config.visionConfiguration.hiddenLayers, 27)
        XCTAssertEqual(config.visionConfiguration.tokensPerImage, 196)
        XCTAssertEqual(config.imagePatchTokenID, 151_938)
        XCTAssertEqual(config.imageTokenIndex, 151_940)
    }

    func testProcessorBuildsPatchifiedCropsAndMetadata() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            JinaVLMProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "JinaVLMProcessor",
                  "base_input_size": [4, 4],
                  "patch_size": 2,
                  "max_crops": 1,
                  "min_pixels": 4,
                  "max_pixels": 16,
                  "overlap_margins": [0, 0],
                  "pooling_h": 2,
                  "pooling_w": 2,
                  "use_column_tokens": true,
                  "start_token_id": 151936,
                  "end_token_id": 151937,
                  "patch_token_id": 151938,
                  "column_token_id": 151939
                }
                """.data(using: .utf8)),
        )
        let processor = JinaVLMProcessor(processorConfig, tokenizer: JinaVLMTestTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 151_938 }), 2)
        XCTAssertEqual(prepared.image?.pixels.shape, [2, 4, 12])
        XCTAssertEqual(prepared.video?.pixels.shape, [2, 5])
    }

    func testSmallModelRunsVisionToLanguagePrepareAndSanitizesKeys() throws {
        let config = try Self.smallConfig()
        let model = JinaVLM(config)
        let tokens = MLXArray([1, 2, 3]).reshaped(1, 3)
        let metadata = MLXArray([
            Float(1), 1, 1, 1, 1,
        ]).reshaped(1, 5)
        let input = LMInput(
            text: .init(tokens: tokens, mask: ones(like: tokens).asType(.int8)),
            image: .init(pixels: MLXArray.zeros([1, 4, 12])),
            video: .init(pixels: metadata),
        )
        let cache = try model.newCache(configuration: nil)
        let prepared = try model.prepare(input, cache: cache, state: nil, prefill: .init())

        switch prepared {
        case .logits(let output):
            XCTAssertEqual(output.logits.shape, [1, 3, 64])
        case .tokens:
            XCTFail("JinaVLM.prepare should return logits")
        }

        let weights = model.sanitize(weights: [
            "lm_head.weight": MLXArray.zeros([64, 32]),
            "language_model.layers.0.attn.qkv.weight": MLXArray.zeros([64, 32]),
            "language_model.layers.0.attn.rotary_emb.inv_freq": MLXArray.zeros([4]),
        ])

        XCTAssertNotNil(weights["language_model.lm_head.weight"])
        XCTAssertNotNil(weights["language_model.layers.0.attn.qkv.weight"])
        XCTAssertNil(weights["language_model.layers.0.attn.rotary_emb.inv_freq"])
    }

    func testLeafModulePathsExposeLayerLinearsForQuantization() throws {
        let model = try JinaVLM(Self.smallConfig())
        let paths = Set(model.leafModules().flattened().map(\.0))

        XCTAssertTrue(paths.contains("language_model.layers.0.attn.qkv"))
        XCTAssertTrue(paths.contains("language_model.layers.0.ffn.gate_up"))
        XCTAssertTrue(paths.contains("vision_model.layers.0.attn.qkv"))
        XCTAssertTrue(paths.contains("vl_connector.projector.gate_up"))
    }

    func testQuantizationAllowsFullPrecisionFirstLanguageLayer() throws {
        let model = try JinaVLM(Self.smallConfig())

        quantize(model: model) { path, _ in
            path.hasPrefix("language_model.layers.1.")
                ? (groupSize: 32, bits: 4, mode: .affine) : nil
        }

        let paths = Set(model.leafModules().flattened().map(\.0))
        XCTAssertTrue(paths.contains("language_model.layers.0.attn.qkv"))
        XCTAssertTrue(paths.contains("language_model.layers.1.attn.qkv"))
    }

    private static func smallConfig() throws -> JinaVLMConfiguration {
        try JSONDecoder.json5().decode(
            JinaVLMConfiguration.self,
            from: """
                {
                  "model_type": "jvlm",
                  "text_config": {
                    "hidden_size": 32,
                    "n_layers": 2,
                    "vocab_size": 64,
                    "additional_vocab_size": 8,
                    "max_sequence_length": 16,
                    "rope_theta": 10000,
                    "block_config": {
                      "attn_config": {
                        "n_heads": 4,
                        "n_kv_heads": 2,
                        "head_dim": 8,
                        "q_lnorm": true
                      },
                      "ffn_config": {"size": 64},
                      "lnorm_config": {"eps": 0.000001}
                    }
                  },
                  "vision_config": {
                    "hidden_size": 32,
                    "n_layers": 1,
                    "input_size": [4, 4],
                    "patch_size": 2,
                    "n_channels": 3,
                    "output_size": 32,
                    "vit_layers": [-1],
                    "post_lnorm": true,
                    "block_config": {
                      "attn_config": {"n_heads": 4, "head_dim": 8, "q_bias": true},
                      "ffn_config": {"size": 64, "activation_type": "gelu_pytorch_tanh"},
                      "lnorm_config": {"eps": 0.000001}
                    },
                    "vl_connector_config": {
                      "pooling_h": 2,
                      "pooling_w": 2,
                      "mlp_projector_config": {"size": 64}
                    }
                  }
                }
                """.data(using: .utf8)!,
        )
    }
}
