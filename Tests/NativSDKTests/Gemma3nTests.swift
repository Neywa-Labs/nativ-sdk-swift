import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct Gemma3nTestTokenizer: Tokenizer {
    var bosToken: String? {
        "<bos>"
    }

    var eosToken: String? {
        "<eos>"
    }

    var unknownToken: String? {
        nil
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var output: [Int] = []
        var index = text.startIndex
        while index < text.endIndex {
            if text[index...].hasPrefix("<image_soft_token>") {
                output.append(262_145)
                index = text.index(index, offsetBy: "<image_soft_token>".count)
            } else if text[index...].hasPrefix("<start_of_image>") {
                output.append(262_144)
                index = text.index(index, offsetBy: "<start_of_image>".count)
            } else if text[index...].hasPrefix("<end_of_image>") {
                output.append(262_146)
                index = text.index(index, offsetBy: "<end_of_image>".count)
            } else if text[index...].hasPrefix("<audio_soft_token>") {
                output.append(262_273)
                index = text.index(index, offsetBy: "<audio_soft_token>".count)
            } else if text[index...].hasPrefix("<start_of_audio>") {
                output.append(256_000)
                index = text.index(index, offsetBy: "<start_of_audio>".count)
            } else if text[index...].hasPrefix("<end_of_audio>") {
                output.append(262_272)
                index = text.index(index, offsetBy: "<end_of_audio>".count)
            } else if text[index...].hasPrefix("<bos>") {
                output.append(2)
                index = text.index(index, offsetBy: "<bos>".count)
            } else if text[index...].hasPrefix("<end_of_turn>") {
                output.append(106)
                index = text.index(index, offsetBy: "<end_of_turn>".count)
            } else if text[index...].hasPrefix("<start_of_turn>") {
                output.append(105)
                index = text.index(index, offsetBy: "<start_of_turn>".count)
            } else {
                output.append(Int(text[index].unicodeScalars.first!.value))
                index = text.index(after: index)
            }
        }
        return addSpecialTokens ? [2] + output : output
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<image_soft_token>": 262_145
        case "<start_of_image>": 262_144
        case "<end_of_image>": 262_146
        case "<audio_soft_token>": 262_273
        case "<start_of_audio>": 256_000
        case "<end_of_audio>": 262_272
        case "<bos>": 2
        case "<end_of_turn>": 106
        case "<start_of_turn>": 105
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case 262_145: "<image_soft_token>"
        case 262_144: "<start_of_image>"
        case 262_146: "<end_of_image>"
        case 262_273: "<audio_soft_token>"
        case 256_000: "<start_of_audio>"
        case 262_272: "<end_of_audio>"
        case 2: "<bos>"
        case 106: "<end_of_turn>"
        case 105: "<start_of_turn>"
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
        encode(text: "Describe this image.", addSpecialTokens: false)
    }
}

final class Gemma3nTests: XCTestCase {
    func testProcessorUsesPreprocessorConfigAndExpandsImageTokens() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            Gemma3nProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "image_seq_length": 4,
                  "audio_seq_length": 3,
                  "size": {"height": 8, "width": 8},
                  "do_resize": true,
                  "do_rescale": true,
                  "do_normalize": false,
                  "rescale_factor": 0.00392156862745098,
                  "image_mean": [0.5, 0.5, 0.5],
                  "image_std": [0.5, 0.5, 0.5]
                }
                """.data(using: .utf8)),
        )
        let processor = Gemma3nProcessor(processorConfig, tokenizer: Gemma3nTestTokenizer())
        let image = CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 6))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(Array(tokens.prefix(2)), [2, 2])
        XCTAssertEqual(tokens.count(where: { $0 == 262_145 }), 4)
        XCTAssertEqual(tokens.count(where: { $0 == 262_144 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 262_146 }), 1)
        XCTAssertEqual(prepared.image?.pixels.shape, [1, 3, 8, 8])
        XCTAssertEqual(prepared.image?.frames?.first?.h, 8)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 8)
        let pixelValues = try XCTUnwrap(prepared.image?.pixels.asArray(Float.self))
        XCTAssertGreaterThan(pixelValues.max() ?? 0, 0.99)
    }

    func testProcessorBuildsAudioFeaturesAndExpandsAudioTokens() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            Gemma3nProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "image_seq_length": 4,
                  "audio_seq_length": 3,
                  "size": {"height": 8, "width": 8},
                  "feature_size": 8,
                  "sampling_rate": 16000,
                  "padding_value": 0
                }
                """.data(using: .utf8)),
        )
        let processor = Gemma3nProcessor(processorConfig, tokenizer: Gemma3nTestTokenizer())
        let input = UserInput(chat: [.user("Transcribe.", audios: [.array(MLXArray.zeros([1600]))])]
        )

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 262_273 }), 3)
        XCTAssertEqual(tokens.count(where: { $0 == 256_000 }), 1)
        XCTAssertEqual(tokens.count(where: { $0 == 262_272 }), 1)
        XCTAssertEqual(prepared.audio?.features.shape, [1, 8, 8])
        XCTAssertEqual(prepared.audio?.mask?.shape, [1, 8])
    }

    func testSanitizeMapsVisionAndAudioKeys() {
        let sanitized = Gemma3n.sanitizeWeights([
            "model.language_model.embed_tokens.weight": MLXArray.zeros([4, 2]),
            "language_model.model.embed_tokens_per_layer.weight": MLXArray.zeros([4, 3]),
            "model.vision_tower.timm_model.conv_stem.conv.weight": MLXArray.zeros([2, 3, 4, 5]),
            "model.embed_vision.embedding.weight": MLXArray.zeros([8, 4]),
            "model.audio_tower.subsample_conv_projection.conv_0.conv.weight": MLXArray.zeros([
                2, 1, 3, 3,
            ]),
            "model.audio_tower.conformer.0.lconv1d.depthwise_conv1d.weight": MLXArray.zeros([
                4, 1, 5,
            ]),
            "model.embed_audio.embedding.weight": MLXArray.zeros([2, 2]),
        ])

        XCTAssertNotNil(sanitized["model.language_model.embed_tokens.weight"])
        XCTAssertNotNil(sanitized["model.language_model.embed_tokens_per_layer.weight"])
        XCTAssertEqual(
            sanitized["vision_tower.timm_model.conv_stem.conv.weight"]?.shape,
            [2, 4, 5, 3],
        )
        XCTAssertNotNil(sanitized["embed_vision.embedding.weight"])
        XCTAssertEqual(
            sanitized["audio_tower.subsample_conv_projection.conv_0.conv.weight"]?.shape,
            [2, 3, 3, 1],
        )
        XCTAssertEqual(
            sanitized["audio_tower.conformer.0.lconv1d.depthwise_conv1d.weight"]?.shape,
            [4, 5, 1],
        )
        XCTAssertNotNil(sanitized["embed_audio.embedding.weight"])
    }

    func testTextOnlyPrepareReturnsLogitsForBatchedProcessorTokens() throws {
        let config = try JSONDecoder.json5().decode(
            Gemma3nConfiguration.self,
            from: Self.tinyTextConfigData,
        )
        let model = Gemma3n(config)
        let input = LMInput(
            text: .init(tokens: MLXArray([2, 65, 106]).expandedDimensions(axis: 0)),
        )

        let prepared = try model.prepare(
            input,
            cache: model.newCache(configuration: nil),
            state: nil, prefill: .init(),
        )

        guard case .logits(let output) = prepared else {
            return XCTFail("Gemma3n text-only prepare should prefill and return logits.")
        }
        XCTAssertEqual(output.logits.shape, [1, 1, 256])
    }

    func testReducedAudioPrepareReturnsLogits() throws {
        let config = try JSONDecoder.json5().decode(
            Gemma3nConfiguration.self,
            from: Self.tinyAudioConfigData,
        )
        let model = Gemma3n(config)
        let input = LMInput(
            text: .init(tokens: MLXArray([2, 41, 41, 41, 41, 3]).expandedDimensions(axis: 0)),
            audio: .init(
                features: MLXArray.zeros([1, 16, 8]),
                mask: MLXArray(Array(repeating: true, count: 16), [1, 16]),
            ),
        )

        let prepared = try model.prepare(
            input,
            cache: model.newCache(configuration: nil),
            state: nil, prefill: .init(),
        )

        guard case .logits(let output) = prepared else {
            return XCTFail("Gemma3n audio prepare should prefill and return logits.")
        }
        XCTAssertEqual(output.logits.shape, [1, 1, 64])
    }

    private static let tinyTextConfigData = """
        {
          "model_type": "gemma3n",
          "image_token_id": 262145,
          "audio_token_id": 262273,
          "vision_soft_tokens_per_image": 256,
          "audio_soft_tokens_per_image": 188,
          "hidden_size": 16,
          "eos_token_id": [1, 106],
          "text_config": {
            "model_type": "gemma3n_text",
            "hidden_size": 16,
            "num_hidden_layers": 2,
            "intermediate_size": [32, 32],
            "num_attention_heads": 2,
            "head_dim": 8,
            "rms_norm_eps": 1e-6,
            "vocab_size": 256,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 0,
            "query_pre_attn_scalar": null,
            "vocab_size_per_layer_input": 256,
            "sliding_window": 8,
            "max_position_embeddings": 128,
            "rope_local_base_freq": 10000.0,
            "rope_theta": 1000000.0,
            "final_logit_softcapping": 30.0,
            "layer_types": ["sliding_attention", "full_attention"],
            "activation_sparsity_pattern": [0.0, 0.0],
            "hidden_size_per_layer_input": 8,
            "altup_num_inputs": 2,
            "altup_coef_clip": null,
            "altup_correct_scale": false,
            "altup_active_idx": 0,
            "laurel_rank": 4,
            "rope_scaling": null,
            "sliding_window_pattern": null
          },
          "vision_config": {
            "model_type": "gemma3n_vision",
            "hidden_size": 2048,
            "image_size": null,
            "vocab_size": 128,
            "vocab_offset": 262144,
            "rms_norm_eps": 1e-6
          },
          "audio_config": {
            "hidden_size": 1536,
            "vocab_size": 128,
            "vocab_offset": 262272
          }
        }
        """.data(using: .utf8)!

    private static let tinyAudioConfigData = """
        {
          "model_type": "gemma3n",
          "image_token_id": 33,
          "audio_token_id": 41,
          "vision_soft_tokens_per_image": 4,
          "audio_soft_tokens_per_image": 4,
          "hidden_size": 8,
          "eos_token_id": [1, 3],
          "text_config": {
            "model_type": "gemma3n_text",
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "intermediate_size": [16, 16],
            "num_attention_heads": 2,
            "head_dim": 4,
            "rms_norm_eps": 0.000001,
            "vocab_size": 64,
            "num_key_value_heads": 1,
            "num_kv_shared_layers": 0,
            "query_pre_attn_scalar": null,
            "vocab_size_per_layer_input": 32,
            "sliding_window": 8,
            "max_position_embeddings": 128,
            "rope_local_base_freq": 10000.0,
            "rope_theta": 1000000.0,
            "final_logit_softcapping": 30.0,
            "layer_types": ["sliding_attention", "full_attention"],
            "activation_sparsity_pattern": [0.0, 0.0],
            "hidden_size_per_layer_input": 4,
            "altup_num_inputs": 2,
            "altup_coef_clip": null,
            "altup_correct_scale": false,
            "altup_active_idx": 0,
            "laurel_rank": 4,
            "rope_scaling": null,
            "sliding_window_pattern": null
          },
          "vision_config": {
            "model_type": "gemma3n_vision",
            "hidden_size": 8,
            "image_size": null,
            "vocab_size": 4,
            "vocab_offset": 32,
            "rms_norm_eps": 0.000001
          },
          "audio_config": {
            "input_feat_size": 8,
            "hidden_size": 8,
            "vocab_size": 4,
            "vocab_offset": 40,
            "rms_norm_eps": 0.000001,
            "gradient_clipping": 10000000000.0,
            "conf_attention_chunk_size": 2,
            "conf_attention_context_left": 2,
            "conf_attention_context_right": 0,
            "conf_attention_logit_cap": 50.0,
            "conf_attention_invalid_logits_value": -1000000000.0,
            "conf_num_attention_heads": 2,
            "conf_num_hidden_layers": 1,
            "conf_conv_kernel_size": 3,
            "conf_reduction_factor": 1,
            "conf_residual_weight": 0.5,
            "sscp_conv_channel_size": [2, 2],
            "sscp_conv_group_norm_eps": 0.001,
            "sscp_conv_kernel_size": [[3, 3], [3, 3]],
            "sscp_conv_stride_size": [[2, 2], [2, 2]]
          }
        }
        """.data(using: .utf8)!
}
