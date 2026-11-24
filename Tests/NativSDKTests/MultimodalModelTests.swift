import CoreImage
import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct FixedPromptTokenizer: Tokenizer {
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

final class MultimodalModelTests: XCTestCase {
    // MARK: - Input processing

    func testAudioProcessorDecodesFamilyDefaultsAndCheckpointOverrides() throws {
        for data in ["{}", #"{"feature_extractor": {}}"#] {
            let config = try JSONDecoder().decode(
                Gemma3nProcessorConfiguration.self, from: Data(data.utf8))
            let audio = config.audioFeatureExtractor
            XCTAssertEqual(audio.frameLengthMs, 32)
            XCTAssertEqual(audio.minFrequency, 125)
            XCTAssertEqual(audio.maxFrequency, 7600)
            XCTAssertEqual(audio.preemphasis, 0.97)
            XCTAssertTrue(audio.fftOverdrive)
            XCTAssertEqual(audio.melFloor, 1e-5)
            XCTAssertEqual(audio.maxLength, 480_000)
        }
        let overrides =
            #"{"frame_length_ms": 24, "preemphasis": 0.5, "fft_overdrive": false, "min_frequency": 80, "max_frequency": 7000, "mel_floor": 0.0001, "max_length": 16000}"#
        for data in [overrides, "{\"feature_extractor\": \(overrides)}"] {
            let config = try JSONDecoder().decode(
                Gemma3nProcessorConfiguration.self, from: Data(data.utf8))
            let audio = config.audioFeatureExtractor
            XCTAssertEqual(audio.frameLengthMs, 24)
            XCTAssertEqual(audio.preemphasis, 0.5)
            XCTAssertFalse(audio.fftOverdrive)
            XCTAssertEqual(audio.minFrequency, 80)
            XCTAssertEqual(audio.maxFrequency, 7000)
            XCTAssertEqual(audio.melFloor, 0.0001)
            XCTAssertEqual(audio.maxLength, 16000)
        }
        let originalDefaults = try JSONDecoder().decode(
            Gemma4AudioFeatureExtractorConfiguration.self, from: Data("{}".utf8))
        XCTAssertEqual(originalDefaults.frameLengthMs, 20)
        XCTAssertEqual(originalDefaults.preemphasis, 0)
        XCTAssertFalse(originalDefaults.fftOverdrive)
    }

    func testUnpaddedAudioFeaturesMatchPythonAndFollowPrompt() async throws {
        struct Reference: Decodable {
            let lengths: [Int]
            let shape: [Int]
            let features: [Float]
            let mask: [Bool]
        }
        let url = try XCTUnwrap(
            Bundle.module.url(
                forResource: "audio-preprocessing", withExtension: "json", subdirectory: "Fixtures")
        )
        let reference = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))
        let config = try JSONDecoder().decode(
            Gemma3nProcessorConfiguration.self,
            from: Data(
                #"{"feature_size": 8, "image_seq_length": 4, "audio_seq_length": 3, "size": {"height": 8, "width": 8}}"#
                    .utf8))
        let processor = Gemma3nProcessor(config, tokenizer: FixedPromptTokenizer(promptTokens: []))
        let samples = reference.lengths.map { count in
            UserInput.Audio.array(MLXArray((0 ..< count).map { Float($0 % 31) / 31 - 0.5 }))
        }
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 8, height: 8))
        let prepared = try await processor.prepare(
            input: UserInput(chat: [
                .user("  Transcribe  ", images: [.ciImage(image)], audios: samples)
            ]))
        let audio = try XCTUnwrap(prepared.audio)
        XCTAssertEqual(audio.features.shape, reference.shape)
        XCTAssertEqual(audio.mask?.asArray(Bool.self), reference.mask)
        let values = audio.features.asArray(Float.self)
        XCTAssertEqual(values.count, reference.features.count)
        for (actual, expected) in zip(values, reference.features) {
            // M5's default TF32 matmul differs slightly from NumPy's float64 mel projection.
            XCTAssertEqual(actual, expected, accuracy: 0.001)
        }
        let text = String(decoding: prepared.text.tokens.asArray(UInt8.self), as: UTF8.self)
        let imageBlock =
            "\n\n<start_of_image>" + String(repeating: "<image_soft_token>", count: 4)
            + "<end_of_image>\n\n"
        let audioBlock =
            "\n\n<start_of_audio>" + String(repeating: "<audio_soft_token>", count: 3)
            + "<end_of_audio>\n\n"
        XCTAssertEqual(
            text,
            "<bos><start_of_turn>user\n\(imageBlock)Transcribe\(audioBlock)\(audioBlock)<end_of_turn>\n<start_of_turn>model\n"
        )
    }

    func testAudioConvolutionSanitizationPreservesConvertedWeights() throws {
        let weights = [
            "audio_tower.subsample_conv_projection.conv_0.conv.weight": MLXArray(0 ..< 36).reshaped(
                4, 1, 3, 3),
            "audio_tower.subsample_conv_projection.conv_1.conv.weight": MLXArray(0 ..< 288)
                .reshaped(8, 4, 3, 3),
            "audio_tower.conformer.0.lconv1d.depthwise_conv1d.weight": MLXArray(0 ..< 40).reshaped(
                8, 1, 5),
        ]
        for prefix in ["", "model."] {
            let raw = Dictionary(uniqueKeysWithValues: weights.map { (prefix + $0.key, $0.value) })
            let converted = Gemma3n.sanitizeWeights(raw)
            let again = Gemma3n.sanitizeWeights(converted)
            for (key, original) in weights {
                let expected =
                    original.ndim == 4
                    ? original.transposed(0, 2, 3, 1) : original.transposed(0, 2, 1)
                let actual = try XCTUnwrap(converted[key])
                XCTAssertEqual(actual.shape, expected.shape)
                XCTAssertEqual(actual.asArray(Int.self), expected.asArray(Int.self))
                XCTAssertEqual(again[key]?.shape, expected.shape)
                XCTAssertEqual(again[key]?.asArray(Int.self), expected.asArray(Int.self))
            }
        }
    }

    private func processorConfigurationData() throws -> Data {
        try XCTUnwrap(
            """
            {
              "processor_class": "Gemma4UnifiedProcessor",
              "image_token_id": 30,
              "audio_token_id": 31,
              "boi_token_id": 40,
              "eoi_token_id": 41,
              "boa_token_id": 42,
              "eoa_token_id": 43,
              "audio_seq_length": 8,
              "audio_ms_per_token": 40,
              "patch_size": 2,
              "pooling_kernel_size": 1,
              "model_patch_size": 2,
              "max_soft_tokens": 4,
              "feature_extractor": {
                "feature_size": 4,
                "sampling_rate": 100,
                "frame_length_ms": 40,
                "hop_length_ms": 20,
                "max_frequency": 50,
                "max_length": 16,
                "pad_to_multiple_of": 4
              }
            }
            """.data(using: .utf8),
        )
    }

    func testProcessorExtractsAudioFeaturesAndExpandsPromptToken() async throws {
        let configuration = try JSONDecoder.json5().decode(
            Gemma4UnifiedAudioProcessorConfiguration.self,
            from: processorConfigurationData(),
        )
        let processor = Gemma4UnifiedAudioProcessor(
            configuration,
            tokenizer: FixedPromptTokenizer(promptTokens: [1, 31, 2]),
        )
        let samples = MLXArray((0 ..< 9).map { Float($0) / 10 }, [9])

        let prepared = try await processor.prepare(
            input: UserInput(prompt: "Transcribe", audios: [.array(samples)]),
        )

        XCTAssertEqual(prepared.text.tokens.asArray(Int.self), [1, 42, 31, 31, 31, 43, 2])
        XCTAssertEqual(prepared.audio?.features.shape, [1, 5, 4])
        XCTAssertEqual(prepared.audio?.mask?.asArray(Bool.self), [true, true, true, true, false])
        XCTAssertNil(prepared.image)

        let featureValues = try XCTUnwrap(prepared.audio?.features.asArray(Float.self))
        XCTAssertTrue(featureValues.allSatisfy(\.isFinite))
    }

    private func processor(tokens: [Int]) throws -> Gemma4Processor {
        let data = Data(
            """
            {
              "processor_class": "Gemma4Processor",
              "image_token_id": 30, "audio_token_id": 31,
              "boi_token_id": 40, "eoi_token_id": 41,
              "boa_token_id": 42, "eoa_token_id": 43,
              "patch_size": 2, "pooling_kernel_size": 1, "max_soft_tokens": 4
            }
            """.utf8)
        let config = try JSONDecoder().decode(Gemma4ProcessorConfiguration.self, from: data)
        return Gemma4Processor(config, tokenizer: FixedPromptTokenizer(promptTokens: tokens))
    }

    func testMessageIncludesImageTextAndAudioInOrder() {
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4))
        let message = Gemma4MessageGenerator().generate(
            message: .user(
                "Follow the spoken instruction.", images: [.ciImage(image)],
                audios: [.array(MLXArray.zeros([3200]))]))
        let content = message["content"] as? [[String: String]]
        XCTAssertEqual(content?.compactMap { $0["type"] }, ["image", "text", "audio"])
        XCTAssertEqual(content?[1]["text"], "Follow the spoken instruction.")
    }

    func testProcessorRetainsAudioWithAndWithoutImages() async throws {
        for withImage in [false, true] {
            let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4))
            let prepared = try await processor(tokens: withImage ? [1, 30, 31, 2] : [1, 31, 2])
                .prepare(
                    input: UserInput(
                        prompt: "Listen", images: withImage ? [.ciImage(image)] : [],
                        audios: [.array(MLXArray.zeros([3200]))]))
            let expected =
                [1] + (withImage ? [40, 30, 30, 30, 30, 41] : [])
                + [42, 31, 31, 31, 31, 31, 43, 2]
            XCTAssertEqual(prepared.text.tokens.asArray(Int.self), expected)
            XCTAssertEqual(prepared.audio?.features.shape, [1, 19, 128])
            XCTAssertTrue(
                try XCTUnwrap(prepared.audio).features.asArray(Float.self).allSatisfy(\.isFinite))
            XCTAssertEqual(
                prepared.audio?.mask?.asArray(Bool.self), Array(repeating: true, count: 19))
            XCTAssertEqual(prepared.image != nil, withImage)
        }
    }

    func testProcessorCountsPaddedClipsAtEncoderFrameBoundaries() async throws {
        let prepared = try await processor(tokens: [1, 31, 31, 2]).prepare(
            input: UserInput(
                prompt: "Listen",
                audios: [
                    .array(MLXArray.zeros([641])), .array(MLXArray.zeros([3200])),
                ]))
        // 641 samples round to two nominal 40 ms tokens, but yield only one
        // valid encoder row. Padding to the longer clip must not add tokens.
        XCTAssertEqual(
            prepared.text.tokens.asArray(Int.self),
            [1, 42, 31, 43, 42, 31, 31, 31, 31, 31, 43, 2])
        XCTAssertEqual(prepared.audio?.features.shape, [2, 19, 128])
    }

    func testProcessorRejectsMissingAndExtraAudioPlaceholders() async throws {
        for tokens in [[1, 2], [1, 31, 31, 2]] {
            do {
                _ = try await processor(tokens: tokens).prepare(
                    input: UserInput(
                        prompt: "Listen", audios: [.array(MLXArray.zeros([3200]))]))
                XCTFail("Mismatched audio placeholders should fail.")
            } catch {
                XCTAssertTrue(error.localizedDescription.contains("Audio prompt tokens"))
            }
        }
    }

    private func model(audio: Bool = true, clipped: Bool = true, encoderLayers: Int = 1) throws
        -> Gemma4
    {
        var root: [String: Any] = [
            "model_type": "gemma4", "image_token_id": 30, "audio_token_id": 31,
            "text_config": [
                "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "num_global_key_value_heads": 1, "head_dim": 8, "global_head_dim": 8,
                "vocab_size": 64, "vocab_size_per_layer_input": 64,
                "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
                "sliding_window": 8, "use_double_wide_mlp": false,
                "layer_types": ["sliding_attention"],
            ],
            "vision_config": [
                "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": encoderLayers,
                "num_attention_heads": 2, "head_dim": 4, "patch_size": 2,
                "pooling_kernel_size": 1, "position_embedding_size": 16,
                "use_clipped_linears": false,
            ],
        ]
        if audio {
            root["audio_config"] = [
                "model_type": "gemma4_audio", "hidden_size": 8, "num_hidden_layers": encoderLayers,
                "num_attention_heads": 2, "subsampling_conv_channels": [2, 2],
                "conv_kernel_size": 3, "attention_chunk_size": 2,
                "attention_context_left": 2, "attention_context_right": 0,
                "output_proj_dims": 8, "use_clipped_linears": clipped,
            ]
        }
        return Gemma4(
            try JSONDecoder().decode(
                Gemma4Configuration.self, from: JSONSerialization.data(withJSONObject: root)))
    }

    func testSanitizerRetainsAudioAndHandlesBothConvolutionLayouts() throws {
        let model = try model()
        let clipKey = "audio_tower.layers.0.self_attn.q_proj.input_min"
        let weights: [String: MLXArray] = [
            "model.audio_tower.subsample_conv_projection.layer0.conv.weight": .zeros([2, 1, 3, 3]),
            "model.audio_tower.subsample_conv_projection.layer1.conv.weight": .zeros([2, 2, 3, 3]),
            "model.audio_tower.layers.0.lconv1d.depthwise_conv1d.weight": .zeros([8, 1, 3]),
            "model.embed_audio.embedding_projection.weight": .zeros([16, 8]),
            "model.\(clipKey)": MLXArray(-1.0),
        ]
        let sanitized = model.sanitize(weights: weights)
        XCTAssertEqual(sanitized.count, weights.count)
        XCTAssertEqual(
            sanitized["audio_tower.subsample_conv_projection.layer0.conv.weight"]?.shape,
            [2, 3, 3, 1])
        XCTAssertEqual(
            sanitized["audio_tower.subsample_conv_projection.layer1.conv.weight"]?.shape,
            [2, 3, 3, 2])
        XCTAssertEqual(
            sanitized["audio_tower.layers.0.lconv1d.depthwise_conv1d.weight"]?.shape,
            [8, 3, 1])
        XCTAssertNotNil(sanitized[clipKey])
        let second = model.sanitize(weights: sanitized)
        for (key, tensor) in sanitized { XCTAssertEqual(second[key]?.shape, tensor.shape) }
        XCTAssertNil(try self.model(clipped: false).sanitize(weights: weights)[clipKey])
    }

    func testAudioChangesEmbeddingsAndPrefillsWithAndWithoutImage() throws {
        // Exercise multiple evaluation boundaries and the final encoder block.
        let model = try model(encoderLayers: 5)
        let audio = LMInput.ProcessedAudio(
            features: MLXArray((0 ..< 1024).map { sin(Float($0)) }, [1, 8, 128]),
            mask: ones([1, 8], dtype: .bool))
        for withImage in [false, true] {
            let tokens = MLXArray(withImage ? [1, 30, 30, 30, 30, 31, 31, 2] : [1, 31, 31, 2])
                .expandedDimensions(axis: 0)
            let image: LMInput.ProcessedImage? =
                withImage
                ? .init(pixels: ones([1, 3, 4, 4]), frames: [THW(1, 4, 4)]) : nil
            let (baseline, _) = try model.getInputEmbeddings(inputIds: tokens, image: image)
            model.train(true)
            let (referenceEmbeddings, _) = try model.getInputEmbeddings(
                inputIds: tokens, image: image, audio: audio)
            eval(referenceEmbeddings)
            model.train(false)
            let (embedded, perLayerInputs) = try model.getInputEmbeddings(
                inputIds: tokens, image: image, audio: audio)
            XCTAssertEqual(abs(embedded - referenceEmbeddings).max().item(Float.self), 0)
            let audioStart = withImage ? 5 : 1
            XCTAssertGreaterThan(
                abs(
                    embedded[0, audioStart ..< audioStart + 2]
                        - baseline[0, audioStart ..< audioStart + 2]
                ).sum().item(Float.self), 0)
            XCTAssertEqual(
                abs(embedded[0, ..<audioStart] - baseline[0, ..<audioStart]).sum().item(Float.self),
                0)
            let cache = try model.newCache(configuration: nil)
            let referenceCache = try model.newCache(configuration: nil)
            let full = model.languageModel(
                cache: referenceCache, inputsEmbeds: embedded, perLayerInputs: perLayerInputs,
                tokenTypeIds: gemma4TokenTypeIds(
                    inputIds: tokens, imageTokenId: 30, videoTokenId: nil, audioTokenId: 31))
            let result = try model.prepare(
                LMInput(text: .init(tokens: tokens), image: image, audio: audio),
                cache: cache, state: nil, prefill: .init())
            guard case .logits(let output) = result else {
                return XCTFail("Expected multimodal prefill")
            }
            XCTAssertEqual(output.logits.shape, [1, 1, 64])
            XCTAssertTrue(output.logits.asArray(Float.self).allSatisfy(\.isFinite))
            // A one-row readout selects GEMV instead of GEMM on Metal, so allow
            // its accumulation rounding while requiring the same greedy token.
            let expected = full.logits[0..., (-1)..., 0...]
            XCTAssertTrue(
                allClose(output.logits, expected, rtol: 0.001, atol: 0.005).item(Bool.self))
            XCTAssertEqual(argMax(output.logits).item(Int.self), argMax(expected).item(Int.self))
            XCTAssertEqual(cache.map(\.offset), referenceCache.map(\.offset))
            let next = MLXArray([3])[.newAxis]
            XCTAssertLessThan(
                abs(model(next, cache: cache) - model(next, cache: referenceCache))
                    .max().item(Float.self), 0.00001)
        }
        XCTAssertThrowsError(
            try self.model(audio: false).getInputEmbeddings(
                inputIds: MLXArray([1, 31, 31, 2])[.newAxis], audio: audio))
        XCTAssertThrowsError(
            try model.getInputEmbeddings(
                inputIds: MLXArray([1, 31, 2])[.newAxis], audio: audio))
    }

    // MARK: - Audio encoding and checkpoint layouts

    func testAudioEncoderDownsamplesFeaturesAndMask() {
        let encoder = Gemma4AudioEncoder(config: Self.tinyAudioConfig)
        let features = MLXArray.zeros([1, 8, 128], dtype: .float32)
        let validMask = MLXArray(
            [true, true, true, true, true, false, false, false],
            [1, 8],
        )

        let (encoded, invalidMask) = encoder(features, mask: .!validMask)

        XCTAssertEqual(encoded.shape, [1, 2, 8])
        XCTAssertEqual(invalidMask.shape, [1, 2])
        XCTAssertEqual(invalidMask.asArray(Bool.self), [false, false])
    }

    func testUnifiedSanitizerPrefixesBaseAndMapsAudioTowerWeights() throws {
        let model = try Gemma4UnifiedAudioModel(Self.tinyModelConfig())
        let sanitized = model.sanitize(weights: [
            "model.language_model.model.embed_tokens.weight": MLXArray.zeros([64, 16]),
            "model.audio_tower.subsample_conv_projection.layer0.conv.weight":
                MLXArray.zeros([2, 1, 3, 3]),
            "model.audio_tower.layers.0.lconv1d.depthwise_conv1d.weight":
                MLXArray.zeros([8, 1, 3]),
        ])

        XCTAssertNotNil(sanitized["base.language_model.model.embed_tokens.weight"])
        XCTAssertEqual(
            sanitized["audio_tower.subsample_conv_projection.layer0.conv.weight"]?.shape,
            [2, 3, 3, 1],
        )
        XCTAssertEqual(
            sanitized["audio_tower.layers.0.lconv1d.depthwise_conv1d.weight"]?.shape,
            [8, 3, 1],
        )
    }

    func testPrepareRoutesAudioThroughTower() throws {
        let model = try Gemma4UnifiedAudioModel(Self.tinyModelConfig())
        _ = model.sanitize(weights: [
            "model.audio_tower.subsample_conv_projection.layer0.conv.weight":
                MLXArray.zeros([2, 1, 3, 3])
        ])

        let tokens = MLXArray([1, 42, 31, 31, 43, 2]).expandedDimensions(axis: 0)
        let input = LMInput(
            text: .init(tokens: tokens, mask: ones(like: tokens).asType(.int8)),
            audio: .init(
                features: MLXArray.zeros([1, 8, 128], dtype: .float32),
                mask: MLXArray(
                    [true, true, true, true, true, false, false, false],
                    [1, 8],
                ),
            ),
        )

        let result = try model.prepare(
            input,
            cache: model.newCache(configuration: nil),
            state: nil, prefill: .init(),
        )

        guard case .logits(let output) = result else {
            return XCTFail("Gemma4 unified prepare should prefill and return logits.")
        }
        XCTAssertEqual(output.logits.shape, [1, 1, 64])
    }

    private static let tinyAudioConfig = Gemma4AudioEncoderConfiguration(
        hiddenSize: 8,
        hiddenLayers: 1,
        attentionHeads: 2,
        subsamplingConvChannels: [2, 2],
        convKernelSize: 3,
        attentionChunkSize: 2,
        attentionContextLeft: 2,
        attentionContextRight: 0,
        useClippedLinears: false,
        outputProjectionDimensions: 8,
    )

    private static func tinyModelConfig() throws -> Gemma4UnifiedAudioModelConfiguration {
        try JSONDecoder.json5().decode(
            Gemma4UnifiedAudioModelConfiguration.self,
            from: tinyConfigData,
        )
    }

    private static let tinyConfigData = """
        {
          "model_type": "gemma4_unified",
          "vocab_size": 64,
          "hidden_size": 16,
          "image_token_id": 30,
          "audio_token_id": 31,
          "video_token_id": 32,
          "boi_token_id": 40,
          "eoi_token_id": 41,
          "boa_token_id": 42,
          "eoa_token_id": 43,
          "text_config": {
            "model_type": "gemma4_unified_text",
            "hidden_size": 16,
            "num_hidden_layers": 1,
            "intermediate_size": 32,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_global_key_value_heads": 1,
            "head_dim": 8,
            "global_head_dim": 8,
            "vocab_size": 64,
            "vocab_size_per_layer_input": 64,
            "num_kv_shared_layers": 0,
            "hidden_size_per_layer_input": 0,
            "sliding_window": 8,
            "sliding_window_pattern": 2,
            "max_position_embeddings": 64,
            "rms_norm_eps": 1e-6,
            "use_double_wide_mlp": false,
            "attention_k_eq_v": true,
            "use_bidirectional_attention": "vision",
            "layer_types": ["sliding_attention"]
          },
          "audio_config": {
            "model_type": "gemma4_unified_audio",
            "audio_samples_per_token": 640,
            "audio_embed_dim": 8,
            "hidden_size": 8,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "subsampling_conv_channels": [2, 2],
            "conv_kernel_size": 3,
            "attention_chunk_size": 2,
            "attention_context_left": 2,
            "attention_context_right": 0,
            "use_clipped_linears": false,
            "output_proj_dims": 8,
            "rms_norm_eps": 1e-6
          }
        }
        """.data(using: .utf8)!

    // MARK: - Prefill and cache preservation

    func testFusedElementwiseOperationsPreserveDTypeAndNumericalAccuracy() {
        for dtype in [DType.float32, .float16, .bfloat16] {
            for length in [1, 7] {
                let gate = MLXArray((0 ..< length * 64).map { sin(Float($0)) * 4 })
                    .reshaped(1, length, 64).asType(dtype)
                let up = cos(gate)
                let activated = geluApproximate(gate)
                eval(activated)
                let expectedGate = activated * up
                let actualGate = gemma4GatedGELU(gate, up)
                let sum = gate + up
                eval(sum)
                let scale = MLXArray([Float(0.7)]).asType(dtype)
                let expectedResidual = sum * scale
                let actualResidual = gemma4ScaledResidual(gate, up, scale)
                let tolerance = dtype == .float32 ? 0.00001 : 0.01
                for (actual, expected) in [
                    (actualGate, expectedGate), (actualResidual, expectedResidual),
                ] {
                    XCTAssertEqual(actual.dtype, expected.dtype)
                    XCTAssertEqual(actual.shape, expected.shape)
                    XCTAssertTrue(
                        allClose(actual, expected, rtol: tolerance, atol: tolerance).item(Bool.self)
                    )
                }
            }
        }
    }

    func testFusedElementwiseGradientsMatchReference() {
        for dtype in [DType.float32, .float16, .bfloat16] {
            for length in [1, 7] {
                let gate = MLXArray((0 ..< length * 64).map { sin(Float($0)) * 4 })
                    .reshaped(1, length, 64).asType(dtype)
                let up = cos(gate)
                let scale = MLXArray([Float(0.7)]).asType(dtype)
                let operations:
                    [(
                        inputs: [MLXArray], reference: ([MLXArray]) -> MLXArray,
                        fused: ([MLXArray]) -> MLXArray
                    )] = [
                        (
                            [gate, up], { sum(geluApproximate($0[0]) * $0[1]) },
                            { sum(gemma4GatedGELU($0[0], $0[1])) }
                        ),
                        (
                            [gate, up, scale], { sum(($0[0] + $0[1]) * $0[2]) },
                            { sum(gemma4ScaledResidual($0[0], $0[1], $0[2])) }
                        ),
                    ]
                for operation in operations {
                    // Check autodiff after an inference call has populated the compile cache.
                    eval(operation.fused(operation.inputs))
                    let reference = valueAndGrad(
                        { [operation.reference($0)] }, argumentNumbers: operation.inputs.indices)
                    let fused = valueAndGrad(
                        { [operation.fused($0)] }, argumentNumbers: operation.inputs.indices)
                    let (expectedLoss, expectedGradients) = reference(operation.inputs)
                    let (actualLoss, actualGradients) = fused(operation.inputs)
                    let tolerance = dtype == .float32 ? 0.00001 : 0.01
                    for (actual, expected) in zip(
                        actualLoss + actualGradients, expectedLoss + expectedGradients)
                    {
                        XCTAssertEqual(actual.dtype, expected.dtype)
                        XCTAssertEqual(actual.shape, expected.shape)
                        XCTAssertTrue(
                            allClose(actual, expected, rtol: tolerance, atol: tolerance)
                                .item(Bool.self))
                    }
                }
            }
        }
    }

    func testFusedDecodePreservesCacheContinuationAndWeightUpdates() throws {
        for sharedLayers in [0, 2] {
            for perLayerWidth in [0, 8] {
                let data = try JSONSerialization.data(withJSONObject: [
                    "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
                    "num_attention_heads": 2, "num_key_value_heads": 1,
                    "num_global_key_value_heads": 1, "head_dim": 16, "global_head_dim": 16,
                    "vocab_size": 64, "vocab_size_per_layer_input": 64,
                    "num_kv_shared_layers": sharedLayers,
                    "hidden_size_per_layer_input": perLayerWidth,
                    "sliding_window": 4, "use_double_wide_mlp": false,
                    "tie_word_embeddings": true,
                    "layer_types": [
                        "sliding_attention", "full_attention", "sliding_attention",
                        "full_attention",
                    ],
                ])
                let config = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: data)
                let model = withRandomState(MLXRandom.RandomState(seed: 27)) {
                    Gemma4TextLanguageModel(config)
                }
                let referenceCache = try model.newCache(configuration: nil)
                let compiledCache = try model.newCache(configuration: nil)
                let prefix = MLXArray([1, 3, 5])[.newAxis]
                eval(model(prefix, cache: referenceCache).logits)
                eval(model(prefix, cache: compiledCache).logits)
                // Cross the sliding-window boundary and update a weight after
                // fusion to catch stale constants in the traced operations.
                for step in 0 ..< 8 {
                    if step == 3 {
                        let mlp = model.model.layers[0].mlp
                        mlp.downProj.update(
                            parameters: .unflattened([
                                "weight": mlp.downProj.weight * 0.5
                            ]))
                    }
                    let token = MLXArray([step + 7])[.newAxis]
                    model.train(true)
                    let expected: MLXArray = {
                        // Training uses the same helpers; disable compilation for an
                        // independent reference and restore it before the inference call.
                        compile(enable: false)
                        defer { compile(enable: true) }
                        let logits = model(token, cache: referenceCache).logits
                        eval(logits)
                        return logits
                    }()
                    model.train(false)
                    let actual = model(token, cache: compiledCache).logits
                    XCTAssertTrue(
                        allClose(actual, expected, rtol: 0.0001, atol: 0.0001).item(Bool.self))
                    XCTAssertEqual(argMax(actual).item(Int.self), argMax(expected).item(Int.self))
                    XCTAssertEqual(compiledCache.map(\.offset), referenceCache.map(\.offset))
                    for (compiledEntry, referenceEntry) in zip(compiledCache, referenceCache) {
                        for (actualState, expectedState) in zip(
                            compiledEntry.state, referenceEntry.state)
                        {
                            XCTAssertTrue(
                                allClose(actualState, expectedState, rtol: 0.0001, atol: 0.0001)
                                    .item(Bool.self))
                        }
                    }
                }
            }
        }
    }

    func testSharedTailLastQueryMatchesFullPrefillAndCacheContinuation() throws {
        for sharedLayers in [0, 2] {
            let data = try JSONSerialization.data(withJSONObject: [
                "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 4,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "num_global_key_value_heads": 1, "head_dim": 16, "global_head_dim": 16,
                "vocab_size": 64, "vocab_size_per_layer_input": 64,
                "num_kv_shared_layers": sharedLayers, "hidden_size_per_layer_input": 8,
                "sliding_window": 4, "use_double_wide_mlp": false,
                "tie_word_embeddings": true, "use_bidirectional_attention": "vision",
                "layer_types": [
                    "sliding_attention", "full_attention", "sliding_attention", "full_attention",
                ],
            ])
            let config = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: data)
            let model = withRandomState(MLXRandom.RandomState(seed: 19)) {
                Gemma4TextLanguageModel(config)
            }
            // Text, audio, image (including a final image query), and mixed spans.
            for types in [
                [0, 0, 0, 0, 0, 0], [0, 3, 3, 3, 3, 0],
                [0, 1, 1, 0, 1, 1], [0, 1, 1, 3, 3, 0],
            ] {
                for prefix in [[], [2, 4, 6]] {
                    let fullCache = try model.newCache(configuration: nil)
                    let lastCache = try model.newCache(configuration: nil)
                    if !prefix.isEmpty {
                        eval(model(MLXArray(prefix)[.newAxis], cache: fullCache).logits)
                        eval(model(MLXArray(prefix)[.newAxis], cache: lastCache).logits)
                    }
                    let tokens = MLXArray([1, 3, 5, 7, 9, 2])[.newAxis]
                    let tokenTypes = MLXArray(types)[.newAxis]
                    let full = model(tokens, cache: fullCache, tokenTypeIds: tokenTypes)
                    let last = model(
                        tokens, cache: lastCache, tokenTypeIds: tokenTypes, logitsToKeep: 1)
                    let expected = full.logits[0..., (-1)..., 0...]
                    XCTAssertEqual(last.logits.shape, [1, 1, 64])
                    XCTAssertTrue(
                        allClose(last.logits, expected, rtol: 0.001, atol: 0.005).item(Bool.self))
                    XCTAssertEqual(
                        argMax(last.logits).item(Int.self), argMax(expected).item(Int.self))
                    XCTAssertEqual(fullCache.map(\.offset), lastCache.map(\.offset))
                    for (fullEntry, lastEntry) in zip(fullCache, lastCache) {
                        for (fullState, lastState) in zip(fullEntry.state, lastEntry.state) {
                            XCTAssertEqual(abs(fullState - lastState).max().item(Float.self), 0)
                        }
                    }
                    let next = MLXArray([11])[.newAxis]
                    XCTAssertLessThan(
                        abs(
                            model(next, cache: fullCache).logits
                                - model(next, cache: lastCache).logits
                        )
                        .max().item(Float.self), 0.00001)
                }
            }
        }
    }

    func testTokenScatterPreservesOrderAcrossSeparatedSpansAndBatches() {
        let input = MLXArray(0 ..< 24).reshaped(2, 4, 3).asType(.bfloat16)
        let mask = MLXArray([false, true, false, true, true, false, true, false], [2, 4])
        let source = MLXArray(100 ..< 112).reshaped(1, 4, 3).asType(.bfloat16)
        let output = gemma4ScatterTokenFeatures(
            inputTensor: input, tokenMask: mask, source: source)
        XCTAssertEqual(output.dtype, input.dtype)
        XCTAssertEqual(output.shape, input.shape)
        XCTAssertEqual(
            output.asArray(Float.self),
            [
                0, 1, 2, 100, 101, 102, 6, 7, 8, 103, 104, 105,
                106, 107, 108, 15, 16, 17, 109, 110, 111, 21, 22, 23,
            ])
        XCTAssertEqual(input.asArray(Int.self), Array(0 ..< 24))
        let unchanged = gemma4ScatterTokenFeatures(
            inputTensor: input, tokenMask: zeros([2, 4], dtype: .bool),
            source: zeros([0, 3], dtype: input.dtype))
        XCTAssertEqual(unchanged.asArray(Float.self), input.asArray(Float.self))
    }

    func testAltupLogitsToKeepPreservesFullForwardAndCacheContinuation() throws {
        for sharedLayers in [0, 2] {
            let data = try JSONSerialization.data(withJSONObject: [
                "model_type": "gemma3n_text", "hidden_size": 16,
                "intermediate_size": [32, 32, 32, 32], "num_hidden_layers": 4,
                "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 8,
                "vocab_size": 64, "vocab_size_per_layer_input": 64,
                "num_kv_shared_layers": sharedLayers, "hidden_size_per_layer_input": 8,
                "sliding_window": 8, "max_position_embeddings": 128,
                "rms_norm_eps": 0.000001, "rope_local_base_freq": 10000,
                "rope_theta": 1000000, "final_logit_softcapping": 30,
                "altup_num_inputs": 2, "altup_correct_scale": false,
                "altup_active_idx": 0, "laurel_rank": 4,
                "activation_sparsity_pattern": [0.0, 0.0, 0.0, 0.0],
                "layer_types": [
                    "sliding_attention", "full_attention", "sliding_attention", "full_attention",
                ],
            ])
            let config = try JSONDecoder().decode(Gemma3nTextConfiguration.self, from: data)
            let model = withRandomState(MLXRandom.RandomState(seed: 19)) {
                Gemma3nTextModel(config: config)
            }
            let tokens = MLXArray([1, 3, 5, 7, 9, 2])[.newAxis]
            for (keep, expectedCount) in [(0, 6), (1, 1), (2, 2), (6, 6), (9, 6)] {
                for prefix in [[], [2, 4, 6]] {
                    let fullCache = try model.newCache(configuration: nil)
                    let keptCache = try model.newCache(configuration: nil)
                    if !prefix.isEmpty {
                        let prefixTokens = MLXArray(prefix)[.newAxis]
                        eval(model(prefixTokens, cache: fullCache))
                        eval(model(prefixTokens, cache: keptCache))
                    }
                    let full = model(tokens, cache: fullCache)
                    let kept = model(tokens, cache: keptCache, logitsToKeep: keep)
                    let expected = full[0..., (-expectedCount)..., 0...]
                    XCTAssertEqual(full.shape, [1, 6, 64])
                    XCTAssertEqual(kept.shape, [1, expectedCount, 64])
                    XCTAssertTrue(
                        allClose(kept, expected, rtol: 0.001, atol: 0.005).item(Bool.self))
                    XCTAssertTrue(
                        all(argMax(kept, axis: -1) .== argMax(expected, axis: -1)).item(Bool.self))
                    XCTAssertEqual(fullCache.map(\.offset), keptCache.map(\.offset))
                    for (fullEntry, keptEntry) in zip(fullCache, keptCache) {
                        for (fullState, keptState) in zip(fullEntry.state, keptEntry.state) {
                            XCTAssertEqual(abs(fullState - keptState).max().item(Float.self), 0)
                        }
                    }
                    let next = MLXArray([11])[.newAxis]
                    XCTAssertLessThan(
                        abs(model(next, cache: fullCache) - model(next, cache: keptCache))
                            .max().item(Float.self), 0.00001)
                }
            }
        }
    }

    func testLogitsToKeepMatchesFullForwardAndKeepsDrafterState() throws {
        for tied in [false, true] {
            let data = try JSONSerialization.data(withJSONObject: [
                "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 4,
                "num_attention_heads": 2, "num_key_value_heads": 1,
                "num_global_key_value_heads": 1, "head_dim": 8, "global_head_dim": 8,
                "vocab_size": 64, "vocab_size_per_layer_input": 64,
                "num_kv_shared_layers": 2, "hidden_size_per_layer_input": 0,
                "sliding_window": 8, "use_double_wide_mlp": false,
                "tie_word_embeddings": tied, "final_logit_softcapping": 30,
                "layer_types": [
                    "sliding_attention", "full_attention", "sliding_attention", "full_attention",
                ],
            ])
            let config = try JSONDecoder().decode(Gemma4TextConfiguration.self, from: data)
            let model = withRandomState(MLXRandom.RandomState(seed: 7)) {
                Gemma4TextLanguageModel(config)
            }
            let tokens = MLXArray([1, 3, 5, 7, 9, 2])[.newAxis]
            for (keep, expectedCount) in [(0, 6), (1, 1), (2, 2), (6, 6), (9, 6)] {
                for emitDrafterState in [false, true] {
                    let fullCache = try model.newCache(configuration: nil)
                    let keptCache = try model.newCache(configuration: nil)
                    let full = model(tokens, cache: fullCache, emitDrafterState: emitDrafterState)
                    let kept = model(
                        tokens, cache: keptCache, logitsToKeep: keep,
                        emitDrafterState: emitDrafterState)
                    XCTAssertEqual(full.logits.shape, [1, 6, 64])
                    XCTAssertEqual(kept.logits.shape, [1, expectedCount, 64])
                    // GEMV and GEMM may accumulate differently; the readout must remain
                    // numerically close and choose the same greedy tokens.
                    let expected = full.logits[0..., (-expectedCount)..., 0...]
                    XCTAssertTrue(
                        allClose(kept.logits, expected, rtol: 0.001, atol: 0.005).item(Bool.self))
                    XCTAssertTrue(
                        all(argMax(kept.logits, axis: -1) .== argMax(expected, axis: -1))
                            .item(Bool.self))
                    if emitDrafterState {
                        let hidden = try XCTUnwrap(kept.state?[mtpLastHiddenStatesKey])
                        XCTAssertEqual(hidden.shape, [1, 6, 16])
                        let fullHidden = try XCTUnwrap(full.state?[mtpLastHiddenStatesKey])
                        XCTAssertEqual(abs(hidden - fullHidden).max().item(Float.self), 0)
                    } else {
                        XCTAssertNil(kept.state)
                    }
                    XCTAssertEqual(fullCache.map(\.offset), keptCache.map(\.offset))
                    let next = MLXArray([11])[.newAxis]
                    XCTAssertLessThan(
                        abs(
                            model(next, cache: fullCache).logits
                                - model(next, cache: keptCache).logits
                        ).max().item(Float.self), 0.00001)
                }
            }
        }
    }
}
