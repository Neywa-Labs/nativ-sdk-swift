import CoreImage
import MLX
import XCTest

@testable import NativSDK

private struct MiniCPMOTestTokenizer: Tokenizer {
    private let tokenIDs: [String: Int] = [
        "<|im_start|>": 10,
        "<|im_end|>": 11,
        "<image>": 12,
        "</image>": 13,
        "<|audio_start|>": 14,
        "<|audio_end|>": 15,
        "<|listen|>": 16,
        "<unk>": 0,
    ]

    var bosToken: String? {
        nil
    }

    var eosToken: String? {
        "<|im_end|>"
    }

    var unknownToken: String? {
        "<unk>"
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var result: [Int] = addSpecialTokens ? [10] : []
        var index = text.startIndex
        let specials = tokenIDs.keys.sorted { $0.count > $1.count }
        while index < text.endIndex {
            if let match = specials.first(where: { text[index...].hasPrefix($0) }),
                let id = tokenIDs[match]
            {
                result.append(id)
                index = text.index(index, offsetBy: match.count)
            } else {
                let scalar = text[index].unicodeScalars.first?.value ?? 0
                result.append(Int(scalar) % 50 + 20)
                index = text.index(after: index)
            }
        }
        return result
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map { convertIdToToken($0) ?? String($0) }.joined(separator: " ")
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

final class MiniCPMOTests: XCTestCase {
    func testProcessorBuildsImageBoundsAndCHWPixels() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            MiniCPMOProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "MiniCPMOProcessor",
                  "scale_resolution": 56,
                  "patch_size": 14,
                  "image_feature_size": 64,
                  "norm_mean": [0.5, 0.5, 0.5],
                  "norm_std": [0.5, 0.5, 0.5]
                }
                """.data(using: .utf8)),
        )
        let processor = MiniCPMOProcessor(processorConfig, tokenizer: MiniCPMOTestTokenizer())
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 28, height: 28))
        let input = UserInput(chat: [.user("Describe.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)
        let bounds = prepared.video?.pixels.asArray(Int.self)

        XCTAssertEqual(tokens.count(where: { $0 == 0 }), 64)
        XCTAssertEqual(prepared.image?.pixels.shape, [9408])
        XCTAssertEqual(prepared.image?.frames?.first?.h, 4)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 4)
        XCTAssertEqual(prepared.video?.pixels.shape, [1, 2])
        XCTAssertEqual((bounds?[1] ?? 0) - (bounds?[0] ?? 0), 64)
    }

    func testProcessorBuildsAudioFeaturesAndMetadata() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            MiniCPMOProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "MiniCPMOProcessor",
                  "audio_pool_step": 5,
                  "audio_processor": {
                    "feature_size": 80,
                    "sampling_rate": 16000,
                    "hop_length": 160,
                    "n_fft": 400,
                    "chunk_length": 1
                  }
                }
                """.data(using: .utf8)),
        )
        let processor = MiniCPMOProcessor(processorConfig, tokenizer: MiniCPMOTestTokenizer())
        let input = UserInput(chat: [
            .user("Transcribe.", audios: [.array(MLXArray.zeros([1600]))])
        ])

        let prepared = try await processor.prepare(input: input)
        let tokens = prepared.text.tokens.asArray(Int.self)
        let metadata = prepared.audio?.mask?.asArray(Int.self)

        XCTAssertEqual(prepared.audio?.features.shape, [1, 80, 10])
        XCTAssertTrue(tokens.contains(14))
        XCTAssertTrue(tokens.contains(15))
        XCTAssertEqual(
            metadata,
            try [
                10, XCTUnwrap(tokens.firstIndex(of: 14)) + 1, XCTUnwrap(tokens.firstIndex(of: 15)),
            ],
        )
    }

    func testSanitizeMapsAudioAndReferenceKeys() throws {
        let config = try JSONDecoder.json5().decode(
            MiniCPMOConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "minicpmo",
                  "hidden_size": 8,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1,
                  "head_dim": 4,
                  "vocab_size": 32,
                  "audio_config": {
                    "d_model": 4,
                    "encoder_layers": 1,
                    "encoder_attention_heads": 1,
                    "encoder_ffn_dim": 16,
                    "num_mel_bins": 80,
                    "max_source_positions": 8
                  },
                  "vision_config": {"hidden_size": 4, "num_hidden_layers": 1, "num_attention_heads": 1, "image_size": 4, "patch_size": 2}
                }
                """.data(using: .utf8)),
        )
        let model = MiniCPMO(config)
        let sanitized = model.sanitize(weights: [
            "llm.model.embed_tokens.weight": MLXArray.zeros([32, 8]),
            "vpm.embeddings.patch_embedding.weight": MLXArray.zeros([4, 3, 2, 2]),
            "resampler.attn.in_proj_weight": MLXArray.zeros([24, 8]),
            "resampler.attn.in_proj_bias": MLXArray.zeros([24]),
            "apm.conv1.weight": MLXArray.zeros([4, 80, 3]),
            "audio_projection_layer.linear1.weight": MLXArray.zeros([8, 4]),
            "tts.head.weight": MLXArray.zeros([1]),
            "unused.weight": MLXArray.zeros([1]),
        ])

        XCTAssertNotNil(sanitized["language_model.model.embed_tokens.weight"])
        XCTAssertEqual(
            sanitized["vision_tower.embeddings.patch_embedding.weight"]?.shape, [4, 2, 2, 3])
        XCTAssertEqual(sanitized["resampler.attn.q_proj.weight"]?.shape, [8, 8])
        XCTAssertEqual(sanitized["resampler.attn.k_proj.bias"]?.shape, [8])
        XCTAssertEqual(sanitized["audio_tower.conv1.weight"]?.shape, [4, 3, 80])
        XCTAssertNotNil(sanitized["audio_projection_layer.linear1.weight"])
        XCTAssertNil(sanitized["tts.head.weight"])
        XCTAssertNil(sanitized["unused.weight"])
    }

    func testReducedImageForwardProducesLogits() throws {
        let config = try JSONDecoder.json5().decode(
            MiniCPMOConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "minicpmo",
                  "hidden_size": 8,
                  "intermediate_size": 16,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1,
                  "head_dim": 4,
                  "rms_norm_eps": 0.000001,
                  "vocab_size": 64,
                  "tie_word_embeddings": true,
                  "query_num": 4,
                  "vision_config": {
                    "hidden_size": 4,
                    "intermediate_size": 8,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "num_channels": 3,
                    "image_size": 4,
                    "patch_size": 2
                  }
                }
                """.data(using: .utf8)),
        )
        let model = MiniCPMO(config)
        let tokens = MLXArray([1, 12, 0, 0, 0, 0, 13, 2], [1, 8])
        let pixels = MLXArray.zeros([48])
        let bounds = MLXArray([2, 6], [1, 2])
        let input = LMInput(
            text: .init(tokens: tokens),
            image: .init(pixels: pixels, frames: [THW(1, 2, 2)]),
            video: .init(pixels: bounds),
        )

        let result = try model.prepare(
            input, cache: model.newCache(configuration: nil), state: nil, prefill: .init())

        guard case .logits(let output) = result else {
            XCTFail("Expected logits")
            return
        }
        XCTAssertEqual(output.logits.shape, [1, 8, 64])
    }

    func testReducedAudioForwardProducesLogits() throws {
        let config = try JSONDecoder.json5().decode(
            MiniCPMOConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "minicpmo",
                  "hidden_size": 8,
                  "intermediate_size": 16,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1,
                  "head_dim": 4,
                  "rms_norm_eps": 0.000001,
                  "vocab_size": 64,
                  "tie_word_embeddings": true,
                  "query_num": 4,
                  "audio_pool_step": 2,
                  "audio_config": {
                    "d_model": 4,
                    "encoder_layers": 1,
                    "encoder_attention_heads": 1,
                    "encoder_ffn_dim": 16,
                    "num_mel_bins": 80,
                    "max_source_positions": 8
                  },
                  "vision_config": {
                    "hidden_size": 4,
                    "intermediate_size": 8,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 1,
                    "num_channels": 3,
                    "image_size": 4,
                    "patch_size": 2
                  }
                }
                """.data(using: .utf8)),
        )
        let model = MiniCPMO(config)
        let tokens = MLXArray([1, 14, 0, 15, 2], [1, 5])
        let audio = LMInput.ProcessedAudio(
            features: MLXArray.zeros([1, 80, 4]),
            mask: MLXArray([4, 2, 3], [1, 3]),
        )
        let input = LMInput(text: .init(tokens: tokens), audio: audio)

        let result = try model.prepare(
            input, cache: model.newCache(configuration: nil), state: nil, prefill: .init())

        guard case .logits(let output) = result else {
            XCTFail("Expected logits")
            return
        }
        XCTAssertEqual(output.logits.shape, [1, 5, 64])
    }
}
