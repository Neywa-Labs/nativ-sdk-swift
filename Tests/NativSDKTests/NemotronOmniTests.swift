import AVFoundation
import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct NemotronOmniTestTokenizer: Tokenizer {
    let promptTokens: [Int]

    private let tokenIDs: [String: Int] = [
        "<image>": 41,
        "<video>": 42,
        "<img>": 43,
        "</img>": 44,
        "<so_embedding>": 31,
        "<so_start>": 32,
        "<so_end>": 33,
    ]

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
        promptTokens
    }
}

final class NemotronOmniTests: XCTestCase {
    private func tinyTextConfiguration(
        hiddenSize: Int = 24,
        pattern: String = "*",
    )
        -> NemotronHConfiguration
    {
        NemotronHConfiguration(
            vocabSize: 128,
            hiddenSize: hiddenSize,
            numHiddenLayers: pattern.count,
            numAttentionHeads: 4,
            numKeyValueHeads: 4,
            mambaNumHeads: 1,
            mambaHeadDim: 8,
            ssmStateSize: 8,
            convKernel: 3,
            nGroups: 1,
            intermediateSize: 32,
            moeIntermediateSize: 16,
            moeSharedExpertIntermediateSize: 16,
            nRoutedExperts: 2,
            numExpertsPerTok: 1,
            hybridOverridePattern: pattern,
            useConvBias: false,
        )
    }

    private func tinySoundConfiguration(hiddenSize: Int = 16)
        -> NemotronOmniConfiguration.SoundConfiguration
    {
        NemotronOmniConfiguration.SoundConfiguration(
            hiddenSize: hiddenSize,
            numAttentionHeads: 4,
            numHiddenLayers: 1,
            intermediateSize: 32,
            convKernelSize: 3,
            subsamplingFactor: 8,
            subsamplingConvChannels: 4,
            numMelBins: 16,
            maxPositionEmbeddings: 64,
            projectionHiddenSize: 32,
        )
    }

    private func tinyVisionConfiguration(hiddenSize: Int = 8)
        -> NemotronOmniConfiguration.VisionConfiguration
    {
        NemotronOmniConfiguration.VisionConfiguration(
            hiddenSize: hiddenSize,
            numHiddenLayers: 1,
            numAttentionHeads: 2,
            intermediateSize: 16,
            imageSize: 4,
            patchSize: 2,
            maxResolution: 4,
            downsampleRatio: 0.5,
            dynamicImageMinNumPatches: 4,
            dynamicImageMaxNumPatches: 4,
            videoTargetNumPatches: 4,
            videoTemporalPatchSize: 2,
        )
    }

    private func processorConfigurationData() throws -> Data {
        try XCTUnwrap(
            """
            {
              "processor_class": "NemotronHNanoOmniProcessor",
              "audio_sampling_rate": 16000,
              "audio_subsampling_factor": 8,
              "audio_hop_length": 160,
              "audio_subsampling_conv_kernel_size": 3,
              "audio_subsampling_conv_stride": 2,
              "sound_config": {
                "num_mel_bins": 8,
                "feat_in": 8,
                "sampling_rate": 16000,
                "hop_length": 160,
                "n_fft": 512,
                "win_length": 400
              }
            }
            """.data(using: .utf8),
        )
    }

    func testConfigurationDecodesNestedNemotronHTextConfig() throws {
        let config = try JSONDecoder.json5().decode(
            NemotronOmniConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "nemotron_h_nano_omni",
                  "text_config": {
                    "model_type": "nemotron_h",
                    "vocab_size": 128,
                    "hidden_size": 24,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 4,
                    "num_key_value_heads": 4,
                    "mamba_num_heads": 1,
                    "mamba_head_dim": 8,
                    "ssm_state_size": 8,
                    "conv_kernel": 3,
                    "n_groups": 1,
                    "intermediate_size": 32,
                    "moe_intermediate_size": 16,
                    "moe_shared_expert_intermediate_size": 16,
                    "n_routed_experts": 2,
                    "num_experts_per_tok": 1,
                    "hybrid_override_pattern": ["*"]
                  },
                  "sound_context_token_id": 99
                }
                """.data(using: .utf8)),
        )

        XCTAssertEqual(config.textConfiguration.hiddenSize, 24)
        XCTAssertEqual(config.languageConfiguration?.vocabSize, 128)
        XCTAssertEqual(config.languageConfiguration?.hybridOverridePattern, "*")
        XCTAssertEqual(config.soundContextTokenID, 99)
    }

    func testAudioSanitizerDropsFeatureExtractorAndTransposesConvolutions() {
        let sanitized = NemotronOmniAudio.sanitizeWeights([
            "sound_encoder.encoder.feature_extractor.window": MLXArray.zeros([4]),
            "sound_encoder.encoder.subsampling.layers.0.weight": MLXArray.zeros([4, 1, 3, 3]),
            "sound_encoder.encoder.subsampling.layers.2.weight": MLXArray.zeros([4, 1, 3, 3]),
            "sound_encoder.encoder.subsampling.layers.3.weight": MLXArray.zeros([4, 4, 1, 1]),
            "sound_encoder.encoder.layers.0.conv.depthwise_conv.weight": MLXArray.zeros([8, 1, 9]),
            "sound_encoder.encoder.layers.0.conv.pointwise_conv1.weight": MLXArray.zeros([16, 8, 1]
            ),
            "sound_encoder.encoder.layers.0.conv.norm.num_batches_tracked": MLXArray.zeros([1]),
            "sound_projection.linear1.weight": MLXArray.zeros([16, 8]),
        ])

        XCTAssertNil(sanitized["sound_encoder.encoder.feature_extractor.window"])
        XCTAssertNil(sanitized["sound_encoder.encoder.layers.0.conv.norm.num_batches_tracked"])
        XCTAssertEqual(
            sanitized["sound_encoder.encoder.subsampling.conv0.weight"]?.shape,
            [4, 3, 3, 1],
        )
        XCTAssertEqual(
            sanitized["sound_encoder.encoder.subsampling.layers.0.depthwise_conv.weight"]?.shape,
            [4, 3, 3, 1],
        )
        XCTAssertEqual(
            sanitized["sound_encoder.encoder.subsampling.layers.0.pointwise_conv.weight"]?.shape,
            [4, 1, 1, 4],
        )
        XCTAssertEqual(
            sanitized["sound_encoder.encoder.layers.0.conv.depthwise_conv.weight"]?.shape,
            [8, 9, 1],
        )
        XCTAssertEqual(
            sanitized["sound_encoder.encoder.layers.0.conv.pointwise_conv1.weight"]?.shape,
            [16, 1, 8],
        )
        XCTAssertEqual(sanitized["sound_projection.linear1.weight"]?.shape, [16, 8])
    }

    func testSoundFeatureExtractorBuildsPaddedNormalizedFeatures() throws {
        let config = NemotronOmniConfiguration.SoundConfiguration(
            numMelBins: 8,
            samplingRate: 16000,
            hopLength: 160,
            fftLength: 512,
            windowLength: 400,
        )
        let extractor = NemotronOmniAudio.SoundFeatureExtractor(config)
        let first = MLXArray(
            (0 ..< 1600).map { Float($0 % 80) / 80 - 0.5 },
            [1600],
        )
        let second = MLXArray(
            (0 ..< 800).map { Float($0 % 40) / 40 - 0.5 },
            [800],
        )

        let output = try extractor([first, second])
        eval(output.inputFeatures)
        eval(output.attentionMask)
        eval(output.fullLengths)

        XCTAssertEqual(output.inputFeatures.shape, [2, 11, 8])
        XCTAssertEqual(output.attentionMask.shape, [2, 11])
        XCTAssertEqual(output.fullLengths.asArray(Int32.self), [11, 6])
        XCTAssertEqual(
            Array(output.attentionMask.asArray(Int32.self)[0 ..< 11]),
            Array(repeating: Int32(1), count: 10) + [0],
        )
        XCTAssertEqual(
            Array(output.attentionMask.asArray(Int32.self)[11 ..< 22]),
            Array(repeating: Int32(1), count: 5) + Array(repeating: Int32(0), count: 6),
        )
        XCTAssertFalse(output.inputFeatures.asArray(Float.self).contains { !$0.isFinite })
    }

    func testReducedSoundEncoderAndProjectionForward() {
        let config = NemotronOmniConfiguration.SoundConfiguration(
            hiddenSize: 8,
            numAttentionHeads: 2,
            numHiddenLayers: 1,
            intermediateSize: 16,
            convKernelSize: 3,
            subsamplingFactor: 2,
            subsamplingConvChannels: 4,
            numMelBins: 8,
            maxPositionEmbeddings: 64,
            projectionHiddenSize: 12,
        )
        let encoder = NemotronOmniAudio.SoundEncoder(config)
        let projection = NemotronOmniAudio.SoundProjection(config, llmHiddenSize: 10)

        let inputFeatures = MLXArray.zeros([1, 8, 8])
        let attentionMask = MLXArray([Int32](repeating: 1, count: 8), [1, 8])
        let (encoded, outputMask) = encoder.encoded(inputFeatures, attentionMask: attentionMask)
        let projected = projection(encoded)
        eval(projected)

        XCTAssertEqual(encoded.shape, [1, 4, 8])
        XCTAssertEqual(outputMask?.shape, [1, 4])
        XCTAssertEqual(projected.shape, [1, 4, 10])
    }

    func testReducedModelAudioPrefillProducesLogits() throws {
        let model = NemotronOmni(
            .init(
                languageConfiguration: tinyTextConfiguration(),
                soundConfiguration: tinySoundConfiguration(),
                soundContextTokenID: 99,
            ),
        )
        let inputIDs = MLXArray([Int32(1), 99, 99, 99, 2], [1, 5])
        let inputFeatures = MLXArray.zeros([1, 17, 16])
        let metadata = MLXArray(
            [Int32(3), 17] + Array(repeating: Int32(1), count: 17),
            [1, 19],
        )
        let input = LMInput(
            text: .init(tokens: inputIDs),
            audio: .init(features: inputFeatures, mask: metadata),
        )

        let result = try model.prepare(
            input, cache: model.newCache(configuration: nil), state: nil, prefill: .init())
        guard case .logits(let output) = result else {
            return XCTFail("Nemotron Omni audio prepare should prefill and return logits.")
        }
        eval(output.logits)

        XCTAssertEqual(output.logits.shape, [1, 5, 128])
        XCTAssertFalse(output.logits.asArray(Float.self).contains { !$0.isFinite })
    }

    func testReducedModelRejectsSoundTokenFeatureCountMismatch() throws {
        let model = NemotronOmni(
            .init(
                languageConfiguration: tinyTextConfiguration(),
                soundConfiguration: tinySoundConfiguration(),
                soundContextTokenID: 99,
            ),
        )
        let inputIDs = MLXArray([Int32(1), 99, 99, 2], [1, 4])
        let metadata = MLXArray(
            [Int32(3), 17] + Array(repeating: Int32(1), count: 17),
            [1, 19],
        )

        XCTAssertThrowsError(
            try model.getInputEmbeddings(
                inputIDs: inputIDs,
                audio: .init(features: MLXArray.zeros([1, 17, 16]), mask: metadata),
            ),
        )
    }

    func testModelSanitizerMapsLanguageAudioAndKeepsVisionWeights() {
        let model = NemotronOmni(
            .init(
                languageConfiguration: tinyTextConfiguration(pattern: "M"),
                visionConfiguration: tinyVisionConfiguration(),
                soundConfiguration: tinySoundConfiguration(),
                imageContextTokenID: 98,
                soundContextTokenID: 99,
            ),
        )
        let convDim = 1 * 8 + 2 * 1 * 8
        let sanitized = model.sanitize(weights: [
            "language_model.backbone.layers.0.mixer.conv1d.weight":
                MLXArray.zeros([convDim, 1, 3]),
            "sound_encoder.encoder.subsampling.layers.0.weight":
                MLXArray.zeros([4, 1, 3, 3]),
            "sound_projection.linear1.weight": MLXArray.zeros([32, 16]),
            "vision_model.radio_model.model.patch_generator.embedder.weight":
                MLXArray.zeros([8, 12]),
            "mlp1.1.weight": MLXArray.zeros([16, 32]),
        ])

        XCTAssertEqual(
            sanitized["language_model.backbone.layers.0.mixer.conv1d.weight"]?.shape,
            [convDim, 3, 1],
        )
        XCTAssertEqual(
            sanitized["sound_encoder.encoder.subsampling.conv0.weight"]?.shape,
            [4, 3, 3, 1],
        )
        XCTAssertNotNil(sanitized["sound_projection.linear1.weight"])
        XCTAssertEqual(
            sanitized["vision_model.radio_model.model.patch_generator.embedder.weight"]?.shape,
            [8, 12],
        )
        XCTAssertEqual(sanitized["mlp1.layers.1.weight"]?.shape, [16, 32])
    }

    func testProcessorExpandsImageAndVideoTokens() async throws {
        let configuration = try JSONDecoder.json5().decode(
            NemotronOmniProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "NemotronHNanoOmniProcessor",
                  "image_token_id": 41,
                  "video_token_id": 42,
                  "image_processor": {
                    "patch_size": 2,
                    "downsample_ratio": 0.5,
                    "min_num_patches": 4,
                    "max_num_patches": 4,
                    "max_model_len": 8,
                    "video_target_num_patches": 4,
                    "video_maintain_aspect_ratio": true
                  },
                  "video_temporal_patch_dim": 2,
                  "video_fps": 1,
                  "max_num_frames": 2,
                  "sound_config": {
                    "num_mel_bins": 8,
                    "feat_in": 8
                  }
                }
                """.data(using: .utf8)),
        )
        let processor = NemotronOmniProcessor(
            configuration,
            tokenizer: NemotronOmniTestTokenizer(promptTokens: [1, 41, 42, 2]),
        )
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4))
        let video = UserInput.Video.frames([
            .init(
                image: .ciImage(
                    CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4))),
                timeStamp: .zero,
            ),
            .init(
                image: .ciImage(
                    CIImage(color: .green).cropped(to: CGRect(x: 0, y: 0, width: 4, height: 4))),
                timeStamp: CMTime(value: 2, timescale: 1),
            ),
        ])

        let prepared = try await processor.prepare(
            input: UserInput(chat: [.user("Describe.", images: [.ciImage(image)], videos: [video])]
            ),
        )

        XCTAssertEqual(prepared.text.tokens.asArray(Int.self), [1, 43, 41, 44, 43, 41, 44, 2])
        XCTAssertEqual(prepared.image?.pixels.shape, [1, 3, 4, 4])
        XCTAssertEqual(prepared.video?.pixels.shape, [2, 3, 4, 4])
        XCTAssertEqual(prepared.image?.frames?.map { [$0.t, $0.h, $0.w] }, [[1, 4, 4]])
        XCTAssertEqual(
            prepared.video?.frames?.map { [$0.t, $0.h, $0.w] }, [[1, 4, 4], [1, 4, 4]])
    }

    func testReducedModelImageAndVideoPrefillProducesLogits() throws {
        let model = NemotronOmni(
            .init(
                languageConfiguration: tinyTextConfiguration(),
                visionConfiguration: tinyVisionConfiguration(),
                projectorHiddenSize: 16,
                vitHiddenSize: 8,
                videoTemporalPatchSize: 2,
                imageContextTokenID: 41,
            ),
        )
        let inputIDs = MLXArray([Int32(1), 41, 41, 2], [1, 4])
        let input = LMInput(
            text: .init(tokens: inputIDs),
            image: .init(
                pixels: MLXArray.zeros([1, 3, 4, 4], dtype: .float32),
                frames: [THW(1, 4, 4)],
            ),
            video: .init(
                pixels: MLXArray.zeros([2, 3, 4, 4], dtype: .float32),
                frames: [THW(1, 4, 4), THW(1, 4, 4)],
            ),
        )

        let result = try model.prepare(
            input, cache: model.newCache(configuration: nil), state: nil, prefill: .init())
        guard case .logits(let output) = result else {
            return XCTFail("Nemotron Omni image/video prepare should prefill and return logits.")
        }
        eval(output.logits)

        XCTAssertEqual(output.logits.shape, [1, 4, 128])
        XCTAssertFalse(output.logits.asArray(Float.self).contains { !$0.isFinite })
    }

    func testProcessorBuildsSoundFeaturesAndExpandsAudioToken() async throws {
        let configuration = try JSONDecoder.json5().decode(
            NemotronOmniProcessorConfiguration.self,
            from: processorConfigurationData(),
        )
        let processor = NemotronOmniProcessor(
            configuration,
            tokenizer: NemotronOmniTestTokenizer(promptTokens: [1, 31, 2]),
        )
        let samples = MLXArray((0 ..< 1600).map { Float($0 % 80) / 80 - 0.5 }, [1600])

        let prepared = try await processor.prepare(
            input: UserInput(prompt: "Transcribe.", audios: [.array(samples)]),
        )

        XCTAssertEqual(prepared.text.tokens.asArray(Int.self), [1, 32, 31, 31, 33, 2])
        XCTAssertEqual(prepared.audio?.features.shape, [1, 11, 8])
        XCTAssertEqual(prepared.audio?.mask?.shape, [1, 13])
        XCTAssertEqual(
            prepared.audio?.mask?.asArray(Int32.self),
            [2, 11] + Array(repeating: Int32(1), count: 10) + [0],
        )
    }
}
