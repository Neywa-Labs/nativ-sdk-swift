import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct Idefics2TestTokenizer: Tokenizer {
    var bosToken: String? {
        "<s>"
    }

    var eosToken: String? {
        "</s>"
    }

    var unknownToken: String? {
        "<unk>"
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
        switch token {
        case "<fake_token_around_image>": 32000
        case "<image>": 32001
        case "<end_of_utterance>": 32002
        case "<s>": 1
        case "</s>": 2
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case 32000: "<fake_token_around_image>"
        case 32001: "<image>"
        case 32002: "<end_of_utterance>"
        case 1: "<s>"
        case 2: "</s>"
        default: nil
        }
    }

    func applyChatTemplate(
        messages _: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?,
    )
        throws -> [Int]
    {
        encode(text: "User:<image>Describe.<end_of_utterance>\nAssistant:", addSpecialTokens: true)
    }
}

final class Idefics2Tests: XCTestCase {
    func testConfigurationDecodesCheckpointDefaults() throws {
        let data = """
            {
              "image_token_id": 32001,
              "model_type": "idefics2",
              "perceiver_config": {
                "model_type": "idefics2"
              },
              "text_config": {
                "max_position_embeddings": 32768,
                "model_type": "mistral",
                "pad_token_id": 0,
                "rms_norm_eps": 1e-5,
                "vocab_size": 32003
              },
              "vision_config": {
                "hidden_size": 1152,
                "image_size": 980,
                "intermediate_size": 4352,
                "model_type": "idefics2",
                "num_attention_heads": 16,
                "num_hidden_layers": 27,
                "patch_size": 14
              }
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder.json5().decode(Idefics2Configuration.self, from: data)

        XCTAssertEqual(config.modelType, "idefics2")
        XCTAssertEqual(config.imageTokenIndex, 32001)
        XCTAssertEqual(config.textConfiguration.hiddenSize, 4096)
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 32)
        XCTAssertEqual(config.textConfiguration.intermediateSize, 14336)
        XCTAssertEqual(config.textConfiguration.kvHeads, 8)
        XCTAssertEqual(config.visionConfiguration.hiddenSize, 1152)
        XCTAssertEqual(config.visionConfiguration.imageSize, 980)
        XCTAssertEqual(config.perceiverConfiguration.latents, 64)
        XCTAssertEqual(config.perceiverConfiguration.depth, 3)
    }

    func testProcessorExpandsSplitImageIntoFiveImageBlocks() async throws {
        let config = try JSONDecoder.json5().decode(
            Idefics2ProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "Idefics2Processor",
                  "image_seq_len": 2,
                  "do_image_splitting": true,
                  "image_mean": [0.5, 0.5, 0.5],
                  "image_std": [0.5, 0.5, 0.5]
                }
                """.data(using: .utf8)),
        )
        let processor = Idefics2Processor(config, tokenizer: Idefics2TestTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 20, height: 12))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 32001 }), 10)
        XCTAssertEqual(tokens.count(where: { $0 == 32000 }), 10)
        XCTAssertEqual(tokens.count(where: { $0 == 32002 }), 1)
        XCTAssertEqual(prepared.image?.pixels.shape, [1, 5, 3, 980, 980])
    }

    func testSanitizeDropsRotaryAndTransposesPatchEmbedding() throws {
        let config = try JSONDecoder.json5().decode(
            Idefics2Configuration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "idefics2",
                  "image_token_id": 9,
                  "perceiver_config": {
                    "model_type": "idefics2",
                    "resampler_depth": 1,
                    "resampler_head_dim": 2,
                    "resampler_n_heads": 2,
                    "num_key_value_heads": 1,
                    "resampler_n_latents": 2
                  },
                  "text_config": {
                    "hidden_size": 8,
                    "num_hidden_layers": 1,
                    "intermediate_size": 16,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "rms_norm_eps": 1e-5,
                    "vocab_size": 16
                  },
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
        let model = Idefics2(config)

        let sanitized = model.sanitize(weights: [
            "language_model.layers.0.self_attn.rotary_emb.inv_freq": MLXArray([1, 2]),
            "vision_model.embeddings.patch_embedding.weight": MLXArray.zeros([6, 3, 2, 2]),
            "vision_model.embeddings.patch_embedding.bias": MLXArray.zeros([6]),
        ])

        XCTAssertNil(sanitized["language_model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertEqual(
            sanitized["vision_model.embeddings.patch_embedding.weight"]?.shape,
            [6, 2, 2, 3],
        )
        XCTAssertNotNil(sanitized["vision_model.embeddings.patch_embedding.bias"])
    }
}
