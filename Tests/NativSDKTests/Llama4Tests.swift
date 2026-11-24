import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct Llama4TestTokenizer: Tokenizer {
    private let tokenIDs: [String: Int] = [
        "<|begin_of_text|>": 200_000,
        "<|end_of_text|>": 200_001,
        "<|header_start|>": 200_005,
        "<|header_end|>": 200_006,
        "<|eom|>": 200_007,
        "<|eot|>": 200_008,
        "<|image_start|>": 200_080,
        "<|image_end|>": 200_081,
        "<|tile_x_separator|>": 200_084,
        "<|tile_y_separator|>": 200_085,
        "<|image|>": 200_090,
        "<|patch|>": 200_092,
    ]

    var bosToken: String? {
        "<|begin_of_text|>"
    }

    var eosToken: String? {
        "<|eot|>"
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
        tokenIDs.map { convertIdToToken($0) ?? "\($0)" }.joined(separator: " ")
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

final class Llama4Tests: XCTestCase {
    func testConfigurationDecodesCheckpointShape() throws {
        let config = try JSONDecoder.json5().decode(
            Llama4Configuration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "llama4",
                  "image_token_index": 200092,
                  "text_config": {
                    "model_type": "llama4_text",
                    "vocab_size": 202048,
                    "hidden_size": 5120,
                    "intermediate_size": 8192,
                    "intermediate_size_mlp": 16384,
                    "num_hidden_layers": 48,
                    "num_attention_heads": 40,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "rms_norm_eps": 0.00001,
                    "rope_theta": 500000,
                    "rope_scaling": {
                      "factor": 16.0,
                      "rope_type": "llama3",
                      "original_max_position_embeddings": 8192
                    },
                    "num_experts_per_tok": 1,
                    "num_local_experts": 16,
                    "no_rope_layers": [1, 1, 1, 0],
                    "attention_chunk_size": 8192
                  },
                  "vision_config": {
                    "model_type": "llama4_vision_model",
                    "hidden_size": 1408,
                    "intermediate_size": 5632,
                    "num_hidden_layers": 34,
                    "num_attention_heads": 16,
                    "num_channels": 3,
                    "image_size": 336,
                    "patch_size": 14,
                    "pixel_shuffle_ratio": 0.5,
                    "projector_input_dim": 4096,
                    "projector_output_dim": 4096,
                    "vision_output_dim": 4096,
                    "rope_theta": 10000
                  }
                }
                """.data(using: .utf8)),
        )

        XCTAssertEqual(config.modelType, "llama4")
        XCTAssertEqual(config.imageTokenIndex, 200_092)
        XCTAssertEqual(config.textConfiguration.modelType, "llama4_text")
        XCTAssertEqual(config.textConfiguration.hiddenSize, 5120)
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 48)
        XCTAssertEqual(config.textConfiguration.kvHeads, 8)
        XCTAssertTrue(config.textConfiguration.usesRope(layerIndex: 0))
        XCTAssertFalse(config.textConfiguration.usesRope(layerIndex: 3))
        XCTAssertEqual(config.visionConfiguration.modelType, "llama4_vision_model")
        XCTAssertEqual(config.visionConfiguration.patchesPerChunk, 144)
        XCTAssertEqual(config.visionConfiguration.visionOutputDim, 4096)
    }

    func testProcessorTilesImageAndExpandsPatchTokens() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            Llama4ProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "Llama4Processor",
                  "patch_size": 2,
                  "pixel_shuffle_ratio": 0.5,
                  "max_patches": 4,
                  "image_mean": [0.5, 0.5, 0.5],
                  "image_std": [0.5, 0.5, 0.5],
                  "size": {"height": 4, "width": 4}
                }
                """.data(using: .utf8)),
        )
        let processor = Llama4Processor(processorConfig, tokenizer: Llama4TestTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 20, height: 4))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 200_092 }), 5)
        XCTAssertTrue(tokens.contains(200_080))
        XCTAssertTrue(tokens.contains(200_081))
        XCTAssertTrue(tokens.contains(200_084))
        XCTAssertTrue(tokens.contains(200_085))
        XCTAssertEqual(prepared.image?.pixels.shape, [5, 3, 4, 4])
    }

    func testSmallModelInstantiatesRunsPrepareAndSanitizesKeys() throws {
        let config = try Self.smallConfig()
        let model = Llama4(config)

        let tokens = MLXArray([1, config.imageTokenIndex, 2]).reshaped(1, 3)
        let input = LMInput(
            text: .init(tokens: tokens, mask: ones(like: tokens).asType(.int8)),
            image: .init(pixels: MLXArray.zeros([1, 3, 4, 4])),
        )
        let cache = try model.newCache(configuration: nil)
        let prepared = try model.prepare(input, cache: cache, state: nil, prefill: .init())
        switch prepared {
        case .logits(let output):
            XCTAssertEqual(output.logits.shape, [1, 1, 32])
        case .tokens:
            XCTFail("Llama4.prepare should return logits")
        }

        let weights = model.sanitize(weights: [
            "model.language_model.model.embed_tokens.weight": MLXArray.zeros([32, 16]),
            "model.vision_model.class_embedding": MLXArray.zeros([8]),
            "model.multi_modal_projector.linear_1.weight": MLXArray.zeros([16, 16]),
            "lm_head.weight": MLXArray.zeros([32, 16]),
            "language_model.model.rotary_emb.inv_freq": MLXArray.zeros([4]),
        ])

        XCTAssertNotNil(weights["language_model.model.embed_tokens.weight"])
        XCTAssertNotNil(weights["vision_model.class_embedding"])
        XCTAssertNotNil(weights["multi_modal_projector.linear_1.weight"])
        XCTAssertNotNil(weights["language_model.lm_head.weight"])
        XCTAssertNil(weights["language_model.model.rotary_emb.inv_freq"])
    }

    private static func smallConfig() throws -> Llama4Configuration {
        try JSONDecoder.json5().decode(
            Llama4Configuration.self,
            from: """
                {
                  "model_type": "llama4",
                  "image_token_id": 9,
                  "image_token_index": 9,
                  "text_config": {
                    "model_type": "llama4_text",
                    "vocab_size": 32,
                    "hidden_size": 16,
                    "intermediate_size": 8,
                    "intermediate_size_mlp": 16,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 8,
                    "rms_norm_eps": 0.00001,
                    "rope_theta": 10000,
                    "num_experts_per_tok": 1,
                    "num_local_experts": 2,
                    "no_rope_layers": [1, 0],
                    "attention_chunk_size": 4
                  },
                  "vision_config": {
                    "model_type": "llama4_vision_model",
                    "hidden_size": 8,
                    "intermediate_size": 32,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_channels": 3,
                    "image_size": 4,
                    "patch_size": 2,
                    "pixel_shuffle_ratio": 0.5,
                    "projector_input_dim": 16,
                    "projector_output_dim": 16,
                    "vision_output_dim": 16,
                    "rope_theta": 10000
                  }
                }
                """.data(using: .utf8)!,
        )
    }
}
