import CoreImage
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct Phi4MultimodalTestTokenizer: Tokenizer {
    let promptTokens: [Int]

    var bosToken: String? {
        nil
    }

    var eosToken: String? {
        nil
    }

    var unknownToken: String? {
        nil
    }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        text.utf8.map(Int.init)
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_: String) -> Int? {
        nil
    }

    func convertIdToToken(_: Int) -> String? {
        nil
    }

    func applyChatTemplate(
        messages _: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?,
    )
        throws -> [Int]
    {
        promptTokens
    }
}

final class Phi4MultimodalTests: XCTestCase {
    func testAudioSanitizerMapsNestedKeysAndConvolutionLayouts() {
        let sanitized = Phi4MultimodalAudio.sanitizeWeights([
            "model.embed_tokens_extend.audio_embed.encoder.embed.conv.0.weight": MLXArray.zeros([
                4, 1, 3, 3,
            ]),
            "model.embed_tokens_extend.audio_embed.encoder.embed.conv.1.weight": MLXArray.zeros([1]
            ),
            "model.embed_tokens_extend.audio_embed.encoder.embed.conv.2.weight": MLXArray.zeros([
                4, 1, 3, 3,
            ]),
            "model.embed_tokens_extend.audio_embed.encoder.embed.conv.3.weight": MLXArray.zeros([
                4, 4, 1, 1,
            ]),
            "model.embed_tokens_extend.audio_embed.encoder.encoders.0.feed_forward_in.net.0.linear.weight":
                MLXArray
                .zeros([
                    12,
                    8,
                ]),
            "model.embed_tokens_extend.audio_embed.encoder.encoders.0.conv.glu.ext_pw_conv_1d.weight":
                MLXArray.zeros([
                    16,
                    8,
                    1,
                ]),
            "model.embed_tokens_extend.audio_embed.encoder.encoders.0.conv.glu.b1": MLXArray.zeros([
                1, 8, 1,
            ]),
            "model.embed_tokens_extend.audio_embed.encoder.encoders.0.conv.dw_sep_conv_1d.dw_conv.weight":
                MLXArray
                .zeros([
                    8,
                    1,
                    3,
                ]),
            "model.embed_tokens_extend.audio_embed.audio_projection.speech.0.weight":
                MLXArray.zeros([16, 8]),
            "model.embed_tokens_extend.audio_embed.audio_projection.vision.2.bias": MLXArray.zeros([
                16
            ]),
        ])

        XCTAssertNil(sanitized[""])
        XCTAssertEqual(sanitized["audio_encoder.embed.conv_0.weight"]?.shape, [4, 3, 3, 1])
        XCTAssertEqual(
            sanitized["audio_encoder.embed.dw_pw_layers.0.dw.weight"]?.shape,
            [4, 3, 3, 1],
        )
        XCTAssertEqual(
            sanitized["audio_encoder.embed.dw_pw_layers.0.pw.weight"]?.shape,
            [4, 1, 1, 4],
        )
        XCTAssertEqual(
            sanitized["audio_encoder.encoders.0.feed_forward_in.net_0.linear.weight"]?.shape,
            [12, 8],
        )
        XCTAssertEqual(
            sanitized["audio_encoder.encoders.0.conv.glu.ext_pw_conv_1d.weight"]?.shape,
            [16, 8],
        )
        XCTAssertEqual(sanitized["audio_encoder.encoders.0.conv.glu.b1"]?.shape, [8])
        XCTAssertEqual(
            sanitized["audio_encoder.encoders.0.conv.dw_sep_conv_1d.dw_conv.weight"]?.shape,
            [8, 3, 1],
        )
        XCTAssertEqual(sanitized["audio_projection.speech.proj_0.weight"]?.shape, [16, 8])
        XCTAssertEqual(sanitized["audio_projection.vision.proj_2.bias"]?.shape, [16])
    }

    func testAudioFeatureExtractorBuildsEmbedsSizesAndMask() throws {
        let extractor = Phi4MultimodalAudio.AudioFeatureExtractor()
        let first = MLXArray(
            (0 ..< 3200).map { Float($0 % 100) / 100 - 0.5 },
            [3200],
        )
        let second = MLXArray(
            (0 ..< 960).map { Float($0 % 60) / 60 - 0.5 },
            [960],
        )

        let output = try extractor([first, second])
        eval(output.inputAudioEmbeds)
        eval(output.audioEmbedSizes)
        if let mask = output.audioAttentionMask {
            eval(mask)
        }

        XCTAssertEqual(output.inputAudioEmbeds.shape, [2, 18, 80])
        XCTAssertEqual(output.audioEmbedSizes.asArray(Int32.self), [3, 1])
        XCTAssertEqual(output.audioAttentionMask?.shape, [2, 18])
        XCTAssertEqual(
            Array(output.audioAttentionMask!.asArray(Int32.self)[0 ..< 18]),
            Array(repeating: Int32(1), count: 18),
        )
        XCTAssertEqual(
            Array(output.audioAttentionMask!.asArray(Int32.self)[18 ..< 36]),
            Array(repeating: Int32(1), count: 4) + Array(repeating: Int32(0), count: 14),
        )
        XCTAssertFalse(output.inputAudioEmbeds.asArray(Float.self).contains { !$0.isFinite })
    }

    func testReducedAudioEncoderAndProjectionForward() {
        let config = Phi4MultimodalConfiguration.AudioConfiguration(
            attentionDimensions: 8,
            attentionHeads: 2,
            numBlocks: 1,
            linearUnits: 6,
            inputSize: 8,
            timeReduction: 2,
            kernelSize: 3,
            extPointwiseOutChannels: 8,
            depthwiseSeparableOutChannels: 8,
            t5BiasMaxDistance: 8,
            convChannels: 4,
        )
        let encoder = Phi4MultimodalAudio.ConformerEncoder(config)
        let projection = Phi4MultimodalAudio.AudioProjection(
            audioDimensions: config.attentionDimensions,
            hiddenSize: 10,
        )
        let features = MLXArray.zeros([1, 8, 8])
        let mask = MLXArray([Int32](repeating: 1, count: 8), [1, 8])

        let (encoded, encodedMask) = encoder(features, masks: mask)
        let projected = projection(encoded, mode: "speech")
        eval(projected)

        XCTAssertEqual(encoded.shape, [1, 4, 8])
        XCTAssertEqual(encodedMask?.shape, [1, 1, 4])
        XCTAssertEqual(projected.shape, [1, 4, 10])
    }

    func testProcessorBuildsAudioFeaturesAndExpandsAudioToken() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            Phi4MultimodalProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "Phi4MMProcessor",
                  "audio_token_index": 31,
                  "audio_processor": {
                    "config": {
                      "time_reduction": 2
                    }
                  }
                }
                """.data(using: .utf8)),
        )
        let processor = Phi4MultimodalProcessor(
            processorConfig,
            tokenizer: Phi4MultimodalTestTokenizer(promptTokens: [1, 31, 2]),
        )
        let samples = MLXArray((0 ..< 1600).map { Float($0 % 80) / 80 - 0.5 }, [1600])

        let prepared = try await processor.prepare(
            input: UserInput(prompt: "Transcribe.", audios: [.array(samples)]),
        )

        XCTAssertEqual(
            prepared.text.tokens.asArray(Int.self), [1] + Array(repeating: 31, count: 4) + [2])
        XCTAssertEqual(prepared.audio?.features.shape, [1, 8, 80])
        XCTAssertEqual(prepared.audio?.mask?.shape, [1, 1])
        XCTAssertEqual(prepared.audio?.mask?.asArray(Int32.self), [4])

        // The same 100 ms recorded at 8 kHz must occupy the same number of audio tokens.
        let lowRate = try await processor.prepare(
            input: UserInput(
                prompt: "Transcribe.",
                audios: [
                    .array(MLXArray.zeros([800]), sampleRate: 8_000)
                ]))
        XCTAssertEqual(lowRate.audio?.features.shape, prepared.audio?.features.shape)
        XCTAssertEqual(
            lowRate.text.tokens.asArray(Int.self), prepared.text.tokens.asArray(Int.self))
    }

    func testProcessorBuildsImagePatches() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            Phi4MultimodalProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "Phi4MMProcessor",
                  "image_token_index": 30,
                  "image_processor": {
                    "patch_size": 2,
                    "min_num_patches": 4,
                    "max_num_patches": 4
                  }
                }
                """.data(using: .utf8)),
        )
        let processor = Phi4MultimodalProcessor(
            processorConfig,
            tokenizer: Phi4MultimodalTestTokenizer(promptTokens: [1, 30, 2]),
        )
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4))

        let prepared = try await processor.prepare(
            input: UserInput(chat: [.user("Describe.", images: [.ciImage(image)])]),
        )

        XCTAssertEqual(prepared.text.tokens.asArray(Int.self), [1, 30, 2])
        XCTAssertEqual(prepared.image?.pixels.shape, [1, 4, 12])
        XCTAssertEqual(prepared.image?.positionIds?.shape, [1, 6])
        XCTAssertEqual(prepared.image?.positionIds?.asArray(Int32.self), [2, 2, 1, 1, 1, 1])
    }

    func testReducedAudioPrepareReturnsLogits() throws {
        let config = try Self.reducedConfig()
        let model = Phi4Multimodal(config)
        let tokens = MLXArray([1, 31, 31, 31, 31, 2]).expandedDimensions(axis: 0)
        let input = LMInput(
            text: .init(tokens: tokens, mask: ones(like: tokens).asType(.int8)),
            audio: .init(
                features: MLXArray.zeros([1, 8, 8], dtype: .float32),
                mask: MLXArray([Int32(4)], [1, 1]),
            ),
        )

        let result = try model.prepare(
            input, cache: model.newCache(configuration: nil), state: nil, prefill: .init())

        guard case .logits(let output) = result else {
            return XCTFail("Phi-4 Multimodal audio prepare should prefill and return logits.")
        }
        XCTAssertEqual(output.logits.shape, [1, 6, 64])
    }

    func testReducedImagePrepareReturnsExpandedLogits() throws {
        let config = try Self.reducedConfig()
        let model = Phi4Multimodal(config)
        let tokens = MLXArray([1, 30, 2]).expandedDimensions(axis: 0)
        let input = LMInput(
            text: .init(tokens: tokens, mask: ones(like: tokens).asType(.int8)),
            image: .init(
                pixels: MLXArray.zeros([1, 4, 12], dtype: .float32),
                positionIds: MLXArray([Int32(2), 2, 1, 1, 1, 1], [1, 6]),
            ),
        )

        let result = try model.prepare(
            input, cache: model.newCache(configuration: nil), state: nil, prefill: .init())

        guard case .logits(let output) = result else {
            return XCTFail("Phi-4 Multimodal image prepare should prefill and return logits.")
        }
        eval(output.logits)
        XCTAssertEqual(output.logits.shape, [1, 6, 64])
        XCTAssertFalse(output.logits.asArray(Float.self).contains { !$0.isFinite })
    }

    func testModelSanitizerMapsLanguageAudioAndSpeechLoRA() throws {
        let model = try Phi4Multimodal(Self.reducedConfig())
        let sanitized = model.sanitize(weights: [
            "model.layers.0.self_attn.qkv_proj.base_layer.weight": MLXArray.zeros([16, 8]),
            "model.layers.0.self_attn.qkv_proj.lora_A.speech.weight": MLXArray.ones([2, 8]),
            "model.layers.0.self_attn.qkv_proj.lora_B.speech.weight": MLXArray.ones([16, 2]),
            "model.embed_tokens.weight": MLXArray.zeros([64, 8]),
            "model.embed_tokens_extend.audio_embed.encoder.embed.conv.0.weight":
                MLXArray.zeros([4, 1, 3, 3]),
            "model.embed_tokens_extend.audio_embed.audio_projection.speech.0.weight":
                MLXArray.zeros([8, 8]),
            "model.embed_tokens_extend.image_embed.img_processor.embeddings.patch_embedding.weight":
                MLXArray.zeros([8, 3, 2, 2]),
            "model.embed_tokens_extend.image_embed.img_projection.0.weight":
                MLXArray.zeros([8, 8]),
        ])

        XCTAssertEqual(
            sanitized["language_model.model.layers.0.self_attn.qkv_proj.weight"]?.shape,
            [16, 8],
        )
        XCTAssertNotNil(sanitized["language_model.model.embed_tokens.weight"])
        XCTAssertEqual(sanitized["audio_encoder.embed.conv_0.weight"]?.shape, [4, 3, 3, 1])
        XCTAssertEqual(sanitized["audio_projection.speech.proj_0.weight"]?.shape, [8, 8])
        XCTAssertEqual(
            sanitized["vision_tower.vision_tower.embeddings.patch_embedding.weight"]?.shape,
            [8, 12],
        )
        XCTAssertEqual(sanitized["mm_projector.0.weight"]?.shape, [8, 8])
        XCTAssertNil(sanitized["model.layers.0.self_attn.qkv_proj.lora_A.speech.weight"])
    }

    func testModelSwitchesVisionAndSpeechLoRAAtPrepare() throws {
        let config = try Self.reducedConfig()
        let model = Phi4Multimodal(config)
        let sanitized = model.sanitize(weights: [
            "model.layers.0.self_attn.qkv_proj.base_layer.weight": MLXArray.zeros([16, 8]),
            "model.layers.0.self_attn.qkv_proj.lora_A.vision.weight": MLXArray.ones([2, 8]),
            "model.layers.0.self_attn.qkv_proj.lora_B.vision.weight": MLXArray.ones([16, 2]),
            "model.layers.0.self_attn.qkv_proj.lora_A.speech.weight": MLXArray.ones([2, 8]) * 2,
            "model.layers.0.self_attn.qkv_proj.lora_B.speech.weight": MLXArray.ones([16, 2]) * 2,
        ])

        let qkv = try XCTUnwrap(
            sanitized["language_model.model.layers.0.self_attn.qkv_proj.weight"])
        XCTAssertEqual(qkv.asArray(Float.self).first, 4)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .noUnusedKeys)

        let imageTokens = MLXArray([1, 30, 2]).expandedDimensions(axis: 0)
        let imageInput = LMInput(
            text: .init(tokens: imageTokens),
            image: .init(
                pixels: MLXArray.zeros([1, 4, 12], dtype: .float32),
                positionIds: MLXArray([Int32(2), 2, 1, 1, 1, 1], [1, 6]),
            ),
        )
        _ = try model.prepare(
            imageInput, cache: model.newCache(configuration: nil), state: nil, prefill: .init())

        let audioTokens = MLXArray([1, 31, 31, 31, 31, 2]).expandedDimensions(axis: 0)
        let audioInput = LMInput(
            text: .init(tokens: audioTokens),
            audio: .init(
                features: MLXArray.zeros([1, 8, 8], dtype: .float32),
                mask: MLXArray([Int32(4)], [1, 1]),
            ),
        )
        _ = try model.prepare(
            audioInput, cache: model.newCache(configuration: nil), state: nil, prefill: .init())
    }

    private static func reducedConfig() throws -> Phi4MultimodalConfiguration {
        try JSONDecoder.json5().decode(
            Phi4MultimodalConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "phi4mm",
                  "vocab_size": 64,
                  "hidden_size": 8,
                  "num_hidden_layers": 1,
                  "intermediate_size": 16,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1,
                  "partial_rotary_factor": 0.5,
                  "image_token_index": 30,
                  "audio_token_index": 31,
                  "tie_word_embeddings": true,
                  "mm_hidden_size": 8,
                  "vision_config": {
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_attention_heads": 2,
                    "num_hidden_layers": 1,
                    "patch_size": 2,
                    "image_size": 4,
                    "num_channels": 3,
                    "layer_norm_eps": 0.000001
                  },
                  "vision_lora": {
                    "r": 2,
                    "lora_alpha": 4
                  },
                  "speech_lora": {
                    "r": 2,
                    "lora_alpha": 4
                  },
                  "audio_processor": {
                    "config": {
                      "attention_dim": 8,
                      "attention_heads": 2,
                      "num_blocks": 1,
                      "linear_units": 6,
                      "input_size": 8,
                      "time_reduction": 2,
                      "kernel_size": 3,
                      "ext_pw_out_channel": 8,
                      "depthwise_seperable_out_channel": 8,
                      "t5_bias_max_distance": 8,
                      "conv_channels": 4
                    }
                  }
                }
                """.data(using: .utf8)),
        )
    }
}
