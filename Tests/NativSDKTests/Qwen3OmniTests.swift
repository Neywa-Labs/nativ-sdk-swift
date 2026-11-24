import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct Qwen3OmniTestTokenizer: Tokenizer {
    private let tokenIDs: [String: Int] = [
        "<|im_start|>": 10,
        "<|im_end|>": 11,
        "<|vision_start|>": 12,
        "<|vision_end|>": 13,
        "<|image_pad|>": 99,
        "<|video_pad|>": 100,
        "<|audio_start|>": 101,
        "<|audio_end|>": 102,
        "<|audio_pad|>": 103,
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

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let specials = tokenIDs.keys.sorted { $0.count > $1.count }
        var result: [Int] = addSpecialTokens ? [10] : []
        var index = text.startIndex
        while index < text.endIndex {
            if let token = specials.first(where: { text[index...].hasPrefix($0) }),
                let id = tokenIDs[token]
            {
                result.append(id)
                index = text.index(index, offsetBy: token.count)
            } else {
                result.append(Int(text[index].unicodeScalars.first!.value) % 64)
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
        messages: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?,
    )
        throws -> [Int]
    {
        var text = ""
        for message in messages {
            if let content = message["content"] as? [[String: String]] {
                for item in content {
                    switch item["type"] {
                    case "image":
                        text += "<|vision_start|><|image_pad|><|vision_end|>"
                    case "video":
                        text += "<|vision_start|><|video_pad|><|vision_end|>"
                    case "audio":
                        text += "<|audio_start|><|audio_pad|><|audio_end|>"
                    case "text":
                        text += item["text"] ?? ""
                    default:
                        break
                    }
                }
            } else if let content = message["content"] as? String {
                text += content
            }
        }
        return encode(text: text, addSpecialTokens: false)
    }
}

private struct Qwen3OmniZeroSampler: LogitSampler {
    func sample(logits: MLXArray) -> MLXArray {
        MLXArray.zeros([logits.dim(0)], dtype: .int32)
    }
}

final class Qwen3OmniTests: XCTestCase {
    func testConfigurationDecodesNestedThinkerShape() throws {
        let config = try JSONDecoder.json5().decode(
            Qwen3OmniMoEConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "qwen3_omni_moe",
                  "eos_token_id": 151645,
                  "thinker_config": {
                    "model_type": "qwen3_omni_moe_thinker",
                    "image_token_id": 151655,
                    "video_token_id": 151656,
                    "vision_start_token_id": 151652,
                    "vision_end_token_id": 151653,
                    "text_config": {
                      "model_type": "qwen3_omni_moe_thinker_text",
                      "hidden_size": 2048,
                      "intermediate_size": 768,
                      "num_hidden_layers": 48,
                      "num_attention_heads": 32,
                      "num_key_value_heads": 4,
                      "head_dim": 128,
                      "max_position_embeddings": 32768,
                      "num_experts": 128,
                      "num_experts_per_tok": 8,
                      "decoder_sparse_step": 1,
                      "mlp_only_layers": [],
                      "moe_intermediate_size": 768,
                      "norm_topk_prob": true,
                      "tie_word_embeddings": false,
                      "vocab_size": 152064,
                      "rope_scaling": {"mrope_section": [24, 20, 20]}
                    },
                    "vision_config": {
                      "model_type": "qwen3_omni_moe_thinker_vision",
                      "depth": 27,
                      "hidden_size": 1152,
                      "intermediate_size": 4304,
                      "out_hidden_size": 2048,
                      "num_heads": 16,
                      "patch_size": 16,
                      "spatial_merge_size": 2,
                      "temporal_patch_size": 2,
                      "deepstack_visual_indexes": [8, 16, 24]
                    },
                    "audio_config": {
                      "model_type": "qwen3_omni_moe_audio_encoder",
                      "d_model": 1280,
                      "encoder_layers": 32,
                      "encoder_attention_heads": 20,
                      "encoder_ffn_dim": 5120,
                      "num_mel_bins": 128,
                      "output_dim": 2048,
                      "downsample_hidden_size": 480,
                      "activation_function": "gelu",
                      "conv_chunksize": 500,
                      "n_window": 50,
                      "n_window_infer": 800,
                      "max_source_positions": 1500
                    }
                  },
                  "code2wav_config": {
                    "hidden_size": 1024,
                    "intermediate_size": 3072,
                    "num_hidden_layers": 8,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 16,
                    "decoder_dim": 1536,
                    "codebook_size": 2048,
                    "num_quantizers": 16,
                    "upsample_rates": [8, 5, 4, 3],
                    "upsampling_ratios": [2, 2]
                  },
                  "talker_config": {
                    "model_type": "qwen3_omni_moe_talker",
                    "thinker_hidden_size": 2048,
                    "num_code_groups": 16,
                    "codec_eos_token_id": 2150,
                    "text_config": {
                      "model_type": "qwen3_omni_moe_talker_text",
                      "hidden_size": 2048,
                      "intermediate_size": 768,
                      "num_hidden_layers": 24,
                      "num_attention_heads": 32,
                      "num_key_value_heads": 4,
                      "head_dim": 128,
                      "max_position_embeddings": 32768,
                      "num_experts": 128,
                      "num_experts_per_tok": 8,
                      "decoder_sparse_step": 1,
                      "mlp_only_layers": [],
                      "moe_intermediate_size": 768,
                      "shared_expert_intermediate_size": 768,
                      "norm_topk_prob": true,
                      "tie_word_embeddings": false,
                      "vocab_size": 152064,
                      "rope_scaling": {"mrope_section": [24, 20, 20]}
                    },
                    "code_predictor_config": {
                      "model_type": "qwen3_omni_moe_talker_code_predictor",
                      "num_hidden_layers": 5,
                      "hidden_size": 1024,
                      "intermediate_size": 3072,
                      "num_attention_heads": 16,
                      "num_key_value_heads": 8,
                      "head_dim": 128,
                      "hidden_act": "silu",
                      "rms_norm_eps": 0.000001,
                      "rope_theta": 1000000,
                      "max_position_embeddings": 32768,
                      "vocab_size": 2048,
                      "num_code_groups": 16
                    }
                  }
                }
                """.data(using: .utf8)),
        )

        XCTAssertEqual(config.modelType, "qwen3_omni_moe")
        XCTAssertEqual(config.imageTokenID, 151_655)
        XCTAssertEqual(config.videoTokenID, 151_656)
        XCTAssertEqual(config.eosTokenID, [151_645])
        XCTAssertEqual(config.textConfiguration.numExperts, 128)
        XCTAssertEqual(config.textConfiguration.numExpertsPerToken, 8)
        XCTAssertFalse(config.textConfiguration.tieWordEmbeddings)
        XCTAssertEqual(config.textConfiguration.ropeScaling?.mropeSection, [24, 20, 20])
        XCTAssertEqual(config.visionConfiguration.numPositionEmbeddings, 2304)
        XCTAssertEqual(config.visionConfiguration.deepstackVisualIndexes, [8, 16, 24])
        XCTAssertEqual(config.audioConfiguration?.modelDimensions, 1280)
        XCTAssertEqual(config.audioConfiguration?.encoderLayers, 32)
        XCTAssertEqual(config.audioConfiguration?.melBins, 128)
        XCTAssertEqual(config.audioTokenID, 151_675)
        XCTAssertEqual(config.audioStartTokenID, 151_669)
        XCTAssertEqual(config.audioEndTokenID, 151_670)
        XCTAssertEqual(config.imStartTokenID, 151_644)
        XCTAssertEqual(config.systemTokenID, 8948)
        XCTAssertEqual(config.userTokenID, 872)
        XCTAssertEqual(config.assistantTokenID, 77091)
        XCTAssertTrue(config.enableAudioOutput)
        XCTAssertEqual(config.code2WavConfiguration?.hiddenSize, 1024)
        XCTAssertEqual(config.code2WavConfiguration?.numQuantizers, 16)
        XCTAssertEqual(config.code2WavConfiguration?.totalUpsampleFactor, 1920)
        XCTAssertEqual(config.talkerConfiguration?.thinkerHiddenSize, 2048)
        XCTAssertEqual(
            config.talkerConfiguration?.textConfiguration.sharedExpertIntermediateSize, 768)
        XCTAssertEqual(config.talkerConfiguration?.codePredictorConfiguration.numCodeGroups, 16)
    }

    func testProcessorPacksQwenPatchesAndExpandsImageToken() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            Qwen3OmniMoEProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "Qwen3OmniMoeProcessor",
                  "patch_size": 2,
                  "merge_size": 2,
                  "temporal_patch_size": 1,
                  "min_pixels": 16,
                  "max_pixels": 16,
                  "image_mean": [0.5, 0.5, 0.5],
                  "image_std": [0.5, 0.5, 0.5]
                }
                """.data(using: .utf8)),
        )
        let processor = Qwen3OmniMoEProcessor(
            processorConfig,
            tokenizer: Qwen3OmniTestTokenizer(),
        )
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)

        XCTAssertEqual(prepared.image?.pixels.shape, [4, 12])
        XCTAssertEqual(prepared.image?.frames?.first?.t, 1)
        XCTAssertEqual(prepared.image?.frames?.first?.h, 2)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 2)
        let tokens = prepared.text.tokens.asArray(Int.self)
        XCTAssertEqual(tokens.count(where: { $0 == 99 }), 1)
        XCTAssertTrue(tokens.contains(12))
        XCTAssertTrue(tokens.contains(13))
    }

    func testProcessorExtractsQwenAudioFeaturesAndExpandsAudioToken() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            Qwen3OmniMoEProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "Qwen3OmniMoeProcessor",
                  "feature_extractor": {
                    "feature_size": 8,
                    "sampling_rate": 16000,
                    "hop_length": 160,
                    "n_fft": 400,
                    "chunk_length": 1,
                    "n_samples": 16000
                  }
                }
                """.data(using: .utf8)),
        )
        let processor = Qwen3OmniMoEProcessor(
            processorConfig,
            tokenizer: Qwen3OmniTestTokenizer(),
        )
        let input = UserInput(
            chat: [.user("Transcribe.", audios: [.array(MLXArray.zeros([1600]))])],
        )

        let prepared = try await processor.prepare(input: input)

        XCTAssertEqual(prepared.audio?.features.shape, [1, 10, 8])
        XCTAssertEqual(prepared.audio?.mask?.shape, [1, 10])
        let tokens = prepared.text.tokens.asArray(Int.self)
        XCTAssertEqual(tokens.count(where: { $0 == 103 }), 2)
        XCTAssertTrue(tokens.contains(101))
        XCTAssertTrue(tokens.contains(102))
    }

    func testSanitizerMapsThinkerKeysKeepsAudioTowerAndStacksExperts() throws {
        let model = try Qwen3OmniMoE(Self.smallConfig())
        let sanitized = model.sanitize(weights: [
            "thinker.language_model.model.embed_tokens.weight": zeros([128, 8]),
            "thinker.language_model.model.layers.0.mlp.experts.0.gate_proj.weight": zeros([4, 8]),
            "thinker.language_model.model.layers.0.mlp.experts.1.gate_proj.weight": zeros([4, 8]),
            "thinker.language_model.model.layers.0.mlp.experts.0.down_proj.weight": zeros([8, 4]),
            "thinker.language_model.model.layers.0.mlp.experts.1.down_proj.weight": zeros([8, 4]),
            "thinker.vision_tower.patch_embed.proj.weight": zeros([8, 3, 1, 2, 2]),
            "thinker.audio_tower.conv2d1.weight": zeros([4, 1, 3, 3]),
            "thinker.audio_tower.conv_out.weight": zeros([8, 4]),
            "talker.model.embed_tokens.weight": zeros([1]),
            "talker.model.codec_embedding.weight": zeros([32, 8]),
            "talker.model.layers.0.mlp.experts.0.gate_proj.weight": zeros([4, 8]),
            "talker.model.layers.0.mlp.experts.1.gate_proj.weight": zeros([4, 8]),
            "talker.model.layers.0.mlp.experts.0.down_proj.weight": zeros([8, 4]),
            "talker.model.layers.0.mlp.experts.1.down_proj.weight": zeros([8, 4]),
            "code2wav.upsample.0.0.conv.weight": zeros([8, 4, 3]),
            "code2wav.decoder.0.conv.weight": zeros([1, 8, 7]),
            "code2wav.decoder.1.block.1.conv.weight": zeros([8, 4, 4]),
        ])

        XCTAssertNotNil(sanitized["language_model.model.embed_tokens.weight"])
        XCTAssertEqual(
            sanitized["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape,
            [2, 4, 8],
        )
        XCTAssertEqual(
            sanitized["language_model.model.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape,
            [2, 8, 4],
        )
        XCTAssertNil(
            sanitized["language_model.model.layers.0.mlp.experts.0.gate_proj.weight"],
        )
        XCTAssertEqual(sanitized["vision_tower.patch_embed.proj.weight"]?.shape, [8, 1, 2, 2, 3])
        XCTAssertEqual(sanitized["audio_tower.conv2d1.weight"]?.shape, [4, 3, 3, 1])
        XCTAssertNotNil(sanitized["audio_tower.conv_out.weight"])
        XCTAssertNil(sanitized["talker.model.embed_tokens.weight"])
        XCTAssertNotNil(sanitized["talker.model.codec_embedding.weight"])
        XCTAssertEqual(
            sanitized["talker.model.layers.0.mlp.switch_mlp.gate_proj.weight"]?.shape,
            [2, 4, 8],
        )
        XCTAssertEqual(
            sanitized["talker.model.layers.0.mlp.switch_mlp.down_proj.weight"]?.shape,
            [2, 8, 4],
        )
        XCTAssertEqual(sanitized["code2wav.upsample.0.0.conv.weight"]?.shape, [4, 3, 8])
        XCTAssertEqual(sanitized["code2wav.decoder.0.conv.weight"]?.shape, [1, 7, 8])
        XCTAssertEqual(sanitized["code2wav.decoder.1.block.1.conv.weight"]?.shape, [4, 4, 8])
    }

    func testReducedImageForwardProducesLogits() throws {
        let model = try Qwen3OmniMoE(Self.smallConfig())
        let input = LMInput(
            text: .init(tokens: MLXArray([1, 12, 99, 13, 2], [1, 5])),
            image: .init(
                pixels: zeros([4, 12]),
                frames: [THW(1, 2, 2)],
            ),
        )

        let result = try model.prepare(
            input, cache: model.newCache(configuration: nil), state: nil, prefill: .init())

        guard case .logits(let output) = result else {
            return XCTFail("Expected logits")
        }
        XCTAssertEqual(output.logits.shape, [1, 5, 128])
    }

    func testReducedAudioForwardProducesLogits() throws {
        let model = try Qwen3OmniMoE(Self.smallConfig())
        let input = LMInput(
            text: .init(tokens: MLXArray([1, 103, 103, 2], [1, 4])),
            audio: .init(
                features: zeros([1, 16, 8]),
                mask: MLXArray(Array(repeating: true, count: 16), [1, 16]),
            ),
        )

        let result = try model.prepare(
            input, cache: model.newCache(configuration: nil), state: nil, prefill: .init())

        guard case .logits(let output) = result else {
            return XCTFail("Expected logits")
        }
        XCTAssertEqual(output.logits.shape, [1, 4, 128])
    }

    func testReducedCode2WavForwardProducesWaveform() throws {
        let model = try Qwen3OmniMoE(Self.smallConfig())
        let codes = MLXArray(
            [
                Int32(0), Int32(1), Int32(2), Int32(3),
                Int32(0), Int32(1), Int32(2), Int32(3),
            ],
            [1, 2, 4],
        )

        let waveform = try model.decodeAudio(codes: codes)

        XCTAssertEqual(waveform.shape, [1, 1, 16])
        let values = waveform.asArray(Float.self)
        XCTAssertFalse(values.isEmpty)
        XCTAssertTrue(values.allSatisfy { $0.isFinite && $0 >= -1.0001 && $0 <= 1.0001 })
    }

    func testReducedTalkerGenerateCodesProducesCodecTensor() throws {
        let config = try Self.smallConfig()
        let talker = try Qwen3OmniMoETalker.Talker(XCTUnwrap(config.talkerConfiguration))
        let codes = try talker.generateCodes(
            inputEmbeddings: zeros([1, 1, 8]),
            trailingTextHidden: zeros([1, 2, 8]),
            ttsPadEmbedding: zeros([1, 1, 8]),
            maxNewTokens: 2,
            sampler: Qwen3OmniZeroSampler(),
        )

        XCTAssertEqual(codes.shape, [1, 2, 2])
        XCTAssertEqual(codes.asArray(Int32.self), [0, 0, 0, 0])
    }

    func testReducedTalkerInputAssemblyFromThinkerHiddenStates() throws {
        let model = try Qwen3OmniMoE(Self.smallConfig())
        let sequences = MLXArray(
            [10, 12, 1, 103, 10, 13, 2, 3, 4, 5, 6],
            [1, 11],
        )
        let talkerInput = try XCTUnwrap(
            model.makeTalkerInput(
                sequences: sequences,
                thinkerHiddenStates: MLXArray.ones([1, 11, 8]),
                thinkerInputEmbeddings: MLXArray.zeros([1, 11, 8]),
                speaker: "Ethan",
            ),
        )

        XCTAssertEqual(talkerInput.inputEmbeddings.shape, [1, 13, 8])
        XCTAssertEqual(talkerInput.inputIDs.shape, [1, 13])
        XCTAssertEqual(
            talkerInput.inputIDs.asArray(Int32.self),
            [10, 12, 1, 103] + Array(repeating: Int32(5), count: 9),
        )
        XCTAssertEqual(talkerInput.trailingTextHidden.shape, [1, 4, 8])
        XCTAssertEqual(talkerInput.ttsPadEmbedding.shape, [1, 1, 8])
    }

    func testExtractThinkerTraceReplaysAcceptedHiddenLayer() throws {
        let model = try Qwen3OmniMoE(Self.smallConfig())
        let sequences = MLXArray(
            [10, 12, 1, 103, 10, 13, 2, 3, 4, 5, 6],
            [1, 11],
        )
        let trace = try model.extractThinkerTrace(
            input: LMInput(text: .init(tokens: sequences)),
            sequences: sequences,
        )

        XCTAssertEqual(trace.sequences.shape, [1, 11])
        XCTAssertEqual(trace.hiddenStates.shape, [1, 11, 8])
        XCTAssertEqual(trace.inputEmbeddings.shape, [1, 11, 8])
        XCTAssertTrue(trace.hiddenStates.asArray(Float.self).allSatisfy(\.isFinite))
    }

    func testQwenAudioExecutorReturnsWaveform() async throws {
        struct AudioProcessor: UserInputProcessor {
            func prepare(input: UserInput) async throws -> LMInput {
                LMInput(tokens: MLXArray([10, 12, 1, 103, 10, 13, 2, 3, 4, 5, 6], [1, 11]))
            }
        }
        let directory = URL(fileURLWithPath: "/tmp/qwen-audio-fixture")
        let loaded = LoadedModel(
            assets: .init(
                source: .directory(directory), modelDirectory: directory,
                tokenizerDirectory: directory),
            metadata: .init(eosTokenIds: [11]),
            model: try Qwen3OmniMoE(Self.smallConfig()),
            processor: AudioProcessor(), tokenizer: Qwen3OmniTestTokenizer())
        let loadedModel = loaded.intoNativModel()
        let operation = loadedModel.generateAudio(
            input: UserInput(prompt: "fixture"),
            options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 0)),
            audio: .init(maxTokens: 2))
        let result = try await operation.result()
        XCTAssertEqual(result.shape, [1, 1, 8])
        XCTAssertEqual(result.sampleRate, 24000)
        XCTAssertTrue(result.samples.allSatisfy(\.isFinite))
        XCTAssertEqual(result.tokenIDs, [])
        XCTAssertEqual(result.text, "")
        XCTAssertEqual(result.summary.generatedTokenCount, 0)
    }

    private static func smallConfig() throws -> Qwen3OmniMoEConfiguration {
        try JSONDecoder.json5().decode(
            Qwen3OmniMoEConfiguration.self,
            from: """
                {
                  "model_type": "qwen3_omni_moe",
                  "im_start_token_id": 10,
                  "im_end_token_id": 11,
                  "system_token_id": 14,
                  "user_token_id": 12,
                  "assistant_token_id": 13,
                  "image_token_id": 99,
                  "video_token_id": 100,
                  "audio_token_id": 103,
                  "audio_start_token_id": 101,
                  "audio_end_token_id": 102,
                  "vision_start_token_id": 12,
                  "vision_end_token_id": 13,
                  "tts_bos_token_id": 3,
                  "tts_eos_token_id": 4,
                  "tts_pad_token_id": 5,
                  "text_config": {
                    "model_type": "qwen3_omni_moe_thinker_text",
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "head_dim": 4,
                    "max_position_embeddings": 64,
                    "rms_norm_eps": 0.000001,
                    "rope_theta": 10000,
                    "num_experts": 2,
                    "num_experts_per_tok": 1,
                    "decoder_sparse_step": 1,
                    "mlp_only_layers": [],
                    "moe_intermediate_size": 4,
                    "norm_topk_prob": true,
                    "tie_word_embeddings": true,
                    "vocab_size": 128
                  },
                  "vision_config": {
                    "model_type": "qwen3_omni_moe_thinker_vision",
                    "depth": 1,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "out_hidden_size": 8,
                    "num_heads": 2,
                    "patch_size": 2,
                    "spatial_merge_size": 2,
                    "temporal_patch_size": 1,
                    "num_position_embeddings": 16,
                    "deepstack_visual_indexes": []
                  },
                  "audio_config": {
                    "model_type": "qwen3_omni_moe_audio_encoder",
                    "d_model": 8,
                    "encoder_layers": 1,
                    "encoder_attention_heads": 2,
                    "encoder_ffn_dim": 16,
                    "num_mel_bins": 8,
                    "output_dim": 8,
                    "downsample_hidden_size": 4,
                    "activation_function": "gelu",
                    "conv_chunksize": 8,
                    "n_window": 8,
                    "n_window_infer": 16,
                    "max_source_positions": 16
                  },
                  "code2wav_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "decoder_dim": 8,
                    "hidden_act": "silu",
                    "attention_bias": false,
                    "attention_dropout": 0.0,
                    "rms_norm_eps": 0.000001,
                    "rope_theta": 10000,
                    "max_position_embeddings": 64,
                    "sliding_window": 4,
                    "codebook_size": 16,
                    "num_quantizers": 2,
                    "num_semantic_quantizers": 1,
                    "semantic_codebook_size": 16,
                    "layer_scale_initial_scale": 0.01,
                    "upsample_rates": [2],
                    "upsampling_ratios": [2]
                  },
                  "talker_config": {
                    "model_type": "qwen3_omni_moe_talker",
                    "accept_hidden_layer": 0,
                    "thinker_hidden_size": 8,
                    "num_code_groups": 2,
                    "codec_bos_id": 7,
                    "codec_eos_token_id": 99,
                    "codec_nothink_id": 8,
                    "codec_pad_id": 9,
                    "codec_think_bos_id": 10,
                    "codec_think_eos_id": 11,
                    "speaker_id": {
                      "ethan": 12
                    },
                    "text_config": {
                      "model_type": "qwen3_omni_moe_talker_text",
                      "hidden_size": 8,
                      "intermediate_size": 16,
                      "num_hidden_layers": 1,
                      "num_attention_heads": 2,
                      "num_key_value_heads": 1,
                      "head_dim": 4,
                      "max_position_embeddings": 64,
                      "rms_norm_eps": 0.000001,
                      "rope_theta": 10000,
                      "num_experts": 2,
                      "num_experts_per_tok": 1,
                      "decoder_sparse_step": 1,
                      "mlp_only_layers": [],
                      "moe_intermediate_size": 4,
                      "shared_expert_intermediate_size": 4,
                      "norm_topk_prob": true,
                      "tie_word_embeddings": false,
                      "vocab_size": 32
                    },
                    "code_predictor_config": {
                      "model_type": "qwen3_omni_moe_talker_code_predictor",
                      "num_hidden_layers": 1,
                      "hidden_size": 8,
                      "intermediate_size": 16,
                      "num_attention_heads": 2,
                      "num_key_value_heads": 1,
                      "head_dim": 4,
                      "hidden_act": "silu",
                      "attention_bias": false,
                      "attention_dropout": 0.0,
                      "rms_norm_eps": 0.000001,
                      "rope_theta": 10000,
                      "max_position_embeddings": 64,
                      "vocab_size": 16,
                      "num_code_groups": 2,
                      "layer_types": ["full_attention"]
                    }
                  }
                }
                """.data(using: .utf8)!,
        )
    }
}
