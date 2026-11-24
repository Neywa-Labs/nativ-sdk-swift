import CoreImage
import CoreMedia
import MLX
import XCTest

@testable import NativSDK

private struct DiffusionGemmaTestTokenizer: Tokenizer {
    var bosToken: String? {
        "<bos>"
    }

    var eosToken: String? {
        "<eos>"
    }

    var unknownToken: String? {
        nil
    }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        text.unicodeScalars.map { Int($0.value) }
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<bos>": 2
        case "<eos>": 1
        case "<image>": 30
        case "<video>": 31
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case 1: "<eos>"
        case 2: "<bos>"
        case 30: "<image>"
        case 31: "<video>"
        default: nil
        }
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?,
    )
        throws -> [Int]
    {
        var tokens = [7]
        for message in messages {
            if let content = message["content"] as? [[String: String]] {
                appendMediaTokens(from: content, to: &tokens)
            } else if let content = message["content"] as? [[String: any Sendable]] {
                appendMediaTokens(from: content, to: &tokens)
            }
        }
        tokens.append(8)
        return tokens
    }

    private func appendMediaTokens(from content: [[String: String]], to tokens: inout [Int]) {
        for item in content {
            switch item["type"] {
            case "image":
                tokens.append(30)
            case "video":
                tokens.append(31)
            default:
                break
            }
        }
    }

    private func appendMediaTokens(
        from content: [[String: any Sendable]],
        to tokens: inout [Int],
    ) {
        for item in content {
            switch item["type"] as? String {
            case "image":
                tokens.append(30)
            case "video":
                tokens.append(31)
            default:
                break
            }
        }
    }
}

final class DiffusionGemmaTests: XCTestCase {
    func testConfigurationDecodesDefaultsAndLayerPattern() throws {
        let config = try JSONDecoder.json5().decode(
            DiffusionGemmaConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "diffusion_gemma",
                  "canvas_length": 256,
                  "text_config": {
                    "model_type": "diffusion_gemma_text",
                    "vocab_size": 262144,
                    "hidden_size": 2816,
                    "num_hidden_layers": 7,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "num_global_key_value_heads": 2,
                    "head_dim": 256,
                    "global_head_dim": 512,
                    "eos_token_id": [1, 107],
                    "top_k_experts": 8
                  }
                }
                """.data(using: .utf8)),
        )

        XCTAssertEqual(config.modelType, "diffusion_gemma")
        XCTAssertEqual(config.canvasLength, 256)
        XCTAssertEqual(config.textConfiguration.modelType, "diffusion_gemma_text")
        XCTAssertEqual(config.textConfiguration.layerTypes.count, 7)
        XCTAssertEqual(config.textConfiguration.layerTypes[5], "full_attention")
        XCTAssertEqual(config.textConfiguration.layerTypes[6], "full_attention")
        XCTAssertEqual(config.textConfiguration.eosTokenID?.values, [1, 107])
    }

    func testVisionConfigurationDecodesDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            DiffusionGemmaConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "diffusion_gemma",
                  "text_config": {"num_hidden_layers": 1},
                  "vision_config": {
                    "model_type": "gemma4_vision",
                    "hidden_size": 1024,
                    "num_hidden_layers": 12,
                    "num_attention_heads": 8,
                    "num_key_value_heads": 4,
                    "patch_size": 16,
                    "default_output_length": 280,
                    "use_clipped_linears": true
                  }
                }
                """.data(using: .utf8)),
        )

        let visionConfig = try XCTUnwrap(config.visionConfiguration)
        XCTAssertEqual(visionConfig.modelType, "gemma4_vision")
        XCTAssertEqual(visionConfig.hiddenSize, 1024)
        XCTAssertEqual(visionConfig.hiddenLayers, 12)
        XCTAssertEqual(visionConfig.keyValueHeads, 4)
        XCTAssertEqual(visionConfig.defaultOutputLength, 280)
        XCTAssertTrue(visionConfig.useClippedLinears)
        XCTAssertEqual(visionConfig.ropeParameters["rope_theta"]?.asFloat(), 100)
    }

    func testSanitizerKeepsTiedDecoderAndEncoderLayerScalars() throws {
        let model = try DiffusionGemma(Self.smallConfig())
        let sanitized = model.sanitize(weights: [
            "model.decoder.embed_tokens.weight": zeros([32, 8]),
            "model.encoder.language_model.layers.0.layer_scalar": zeros([1]),
            "model.encoder.language_model.layers.0.self_attn.q_proj.weight": zeros([8, 8]),
            "model.decoder.layers.0.experts.down_proj": zeros([4, 8, 4]),
            "model.decoder.layers.0.experts.gate_up_proj": zeros([4, 8, 8]),
            "model.encoder.vision_tower.embeddings.patch_embedding.weight": zeros([8, 3, 2, 2]),
            "lm_head.weight": zeros([32, 8]),
            "model.decoder.layers.0.self_attn.rotary_emb.inv_freq": zeros([1]),
        ])

        XCTAssertEqual(sanitized["model.decoder.embed_tokens.weight"]?.shape, [32, 8])
        XCTAssertEqual(
            sanitized["model.encoder.language_model.layers.0.layer_scalar"]?.shape,
            [1],
        )
        XCTAssertNil(sanitized["model.encoder.language_model.layers.0.self_attn.q_proj.weight"])
        XCTAssertEqual(
            sanitized["model.decoder.layers.0.experts.down_proj.weight"]?.shape,
            [4, 8, 4],
        )
        XCTAssertEqual(
            sanitized["model.decoder.layers.0.experts.gate_up_proj.weight"]?.shape,
            [4, 8, 8],
        )
        XCTAssertNil(sanitized["model.encoder.vision_tower.embeddings.patch_embedding.weight"])
        XCTAssertNil(sanitized["lm_head.weight"])
        XCTAssertNil(sanitized["model.decoder.layers.0.self_attn.rotary_emb.inv_freq"])
    }

    func testSanitizerKeepsVisionWeightsWhenConfigured() throws {
        let model = try DiffusionGemma(Self.smallVisionConfig())
        let sanitized = model.sanitize(weights: [
            "model.encoder.vision_tower.patch_embedder.input_proj.linear.weight": zeros([8, 12]),
            "model.encoder.vision_tower.encoder.layers.0.self_attn.q_proj.input_min": zeros([1]),
            "model.encoder.embed_vision.embedding_projection.weight": zeros([8, 8]),
        ])

        XCTAssertEqual(
            sanitized["model.encoder.vision_tower.patch_embedder.input_proj.linear.weight"]?.shape,
            [8, 12],
        )
        XCTAssertEqual(
            sanitized["model.encoder.embed_vision.embedding_projection.weight"]?.shape,
            [8, 8],
        )
        XCTAssertNil(
            sanitized["model.encoder.vision_tower.encoder.layers.0.self_attn.q_proj.input_min"],
        )
    }

    func testReducedPrefillDecoderSelfConditioningAndCacheUpdate() throws {
        let model = try DiffusionGemma(Self.smallConfig())
        let cache = model.diffusionMakeCache(parameters: nil)
        XCTAssertEqual(cache.count, 2)

        try model.diffusionPrefill(
            input: LMInput(tokens: MLXArray([1, 2, 3], [1, 3])),
            cache: cache,
        )
        XCTAssertEqual(cache.first?.offset, 3)

        let canvas = MLXArray([4, 5], [1, 2])
        let logits = model.diffusionDecoderLogits(
            canvas: canvas,
            cache: cache,
            selfConditioning: nil,
        )
        XCTAssertEqual(logits.shape, [1, 2, 32])

        let selfConditioning = try XCTUnwrap(model.diffusionSelfConditioning(logits: logits))
        XCTAssertEqual(selfConditioning.shape, [1, 2, 8])

        let conditionedLogits = model.diffusionDecoderLogits(
            canvas: canvas,
            cache: cache,
            selfConditioning: selfConditioning,
        )
        XCTAssertEqual(conditionedLogits.shape, [1, 2, 32])

        model.diffusionUpdateCache(tokens: MLXArray([6, 7], [1, 2]), cache: cache)
        XCTAssertEqual(cache.first?.offset, 5)
        XCTAssertEqual(model.diffusionCanvasLength, 4)
        XCTAssertEqual(model.diffusionEOSTokenIDs, [0])
    }

    func testReducedVisionPrefillReplacesImageTokens() throws {
        let model = try DiffusionGemma(Self.smallVisionConfig())
        let cache = model.diffusionMakeCache(parameters: nil)
        let tokens = MLXArray([1, 30, 30, 2], [1, 4])
        let mask = ones([1, 4], dtype: .int8)
        let pixels = MLXArray.ones([1, 3, 2, 4]) * 0.5

        try model.diffusionPrefill(
            input: LMInput(
                text: .init(tokens: tokens, mask: mask),
                image: .init(pixels: pixels),
            ),
            cache: cache,
        )
        XCTAssertEqual(cache.first?.offset, 4)

        let logits = model.diffusionDecoderLogits(
            canvas: MLXArray([4], [1, 1]),
            cache: cache,
            selfConditioning: nil,
        )
        XCTAssertEqual(logits.shape, [1, 1, 32])
    }

    func testReducedVisionPrefillAcceptsVideoTokens() throws {
        let model = try DiffusionGemma(Self.smallVisionConfig())
        let cache = model.diffusionMakeCache(parameters: nil)
        let tokens = MLXArray([1, 31, 31, 2], [1, 4])
        let mask = ones([1, 4], dtype: .int8)
        let pixels = MLXArray.ones([1, 3, 2, 4]) * 0.5

        try model.diffusionPrefill(
            input: LMInput(
                text: .init(tokens: tokens, mask: mask),
                video: .init(pixels: pixels),
            ),
            cache: cache,
        )
        XCTAssertEqual(cache.first?.offset, 4)
    }

    func testProcessorExpandsImageTokensAndPreprocessesPixels() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            DiffusionGemmaProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "DiffusionGemma4Processor",
                  "image_seq_length": 2,
                  "image_token_id": 30,
                  "boi_token_id": 28,
                  "eoi_token_id": 29,
                  "size": {"height": 2, "width": 2}
                }
                """.data(using: .utf8)),
        )
        let processor = DiffusionGemmaProcessor(
            processorConfig,
            tokenizer: DiffusionGemmaTestTokenizer(),
        )
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 2, height: 2))

        let prepared = try await processor.prepare(
            input: UserInput(chat: [.user("Describe.", images: [.ciImage(image)])]),
        )
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens, [7, 28, 30, 30, 29, 8])
        XCTAssertEqual(prepared.image?.pixels.shape, [1, 3, 2, 2])
        XCTAssertEqual(prepared.image?.frames?.first?.t, 1)
        XCTAssertNil(prepared.video)
    }

    func testProcessorExpandsVideoTokensAndPreprocessesFrames() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            DiffusionGemmaProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "DiffusionGemma4Processor",
                  "image_seq_length": 2,
                  "image_token_id": 30,
                  "video_token_id": 31,
                  "boi_token_id": 28,
                  "eoi_token_id": 29,
                  "video_fps": 2,
                  "size": {"height": 2, "width": 2}
                }
                """.data(using: .utf8)),
        )
        let processor = DiffusionGemmaProcessor(
            processorConfig,
            tokenizer: DiffusionGemmaTestTokenizer(),
        )
        let firstFrame = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 2, height: 2))
        let secondFrame = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 2, height: 2))
        let video = UserInput.Video.frames([
            .init(image: .ciImage(firstFrame), timeStamp: .zero),
            .init(
                image: .ciImage(secondFrame), timeStamp: CMTime(seconds: 1, preferredTimescale: 600)
            ),
        ])

        let prepared = try await processor.prepare(
            input: UserInput(chat: [.user("Describe.", videos: [video])]),
        )
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens, [7, 28, 31, 31, 31, 31, 29, 8])
        XCTAssertNil(prepared.image)
        XCTAssertEqual(prepared.video?.pixels.shape, [2, 3, 2, 2])
        XCTAssertEqual(prepared.video?.frames?.first?.t, 2)
    }

    func testProcessorExpandsMixedImageAndVideoTokens() async throws {
        let processorConfig = try JSONDecoder.json5().decode(
            DiffusionGemmaProcessorConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "processor_class": "DiffusionGemma4Processor",
                  "image_seq_length": 2,
                  "image_token_id": 30,
                  "video_token_id": 31,
                  "boi_token_id": 28,
                  "eoi_token_id": 29,
                  "video_fps": 2,
                  "size": {"height": 2, "width": 2}
                }
                """.data(using: .utf8)),
        )
        let processor = DiffusionGemmaProcessor(
            processorConfig,
            tokenizer: DiffusionGemmaTestTokenizer(),
        )
        let image = CIImage(color: .green).cropped(to: CGRect(x: 0, y: 0, width: 2, height: 2))
        let firstFrame = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 2, height: 2))
        let secondFrame = CIImage(color: .blue).cropped(to: CGRect(x: 0, y: 0, width: 2, height: 2))
        let video = UserInput.Video.frames([
            .init(image: .ciImage(firstFrame), timeStamp: .zero),
            .init(
                image: .ciImage(secondFrame), timeStamp: CMTime(seconds: 1, preferredTimescale: 600)
            ),
        ])

        let prepared = try await processor.prepare(
            input: UserInput(chat: [.user("Describe.", images: [.ciImage(image)], videos: [video])]
            ),
        )
        let tokens = prepared.text.tokens.asArray(Int.self)

        XCTAssertEqual(tokens, [7, 28, 30, 30, 29, 28, 31, 31, 31, 31, 29, 8])
        XCTAssertEqual(prepared.image?.pixels.shape, [1, 3, 2, 2])
        XCTAssertEqual(prepared.video?.pixels.shape, [2, 3, 2, 2])
    }

    private static func smallConfig() throws -> DiffusionGemmaConfiguration {
        try JSONDecoder.json5().decode(
            DiffusionGemmaConfiguration.self,
            from: """
                {
                  "model_type": "diffusion_gemma",
                  "canvas_length": 4,
                  "eos_token_id": 0,
                  "text_config": {
                    "model_type": "diffusion_gemma_text",
                    "vocab_size": 32,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "moe_intermediate_size": 4,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "num_global_key_value_heads": 1,
                    "head_dim": 4,
                    "global_head_dim": 4,
                    "rms_norm_eps": 0.000001,
                    "max_position_embeddings": 64,
                    "pad_token_id": 0,
                    "eos_token_id": 0,
                    "tie_word_embeddings": true,
                    "sliding_window": 4,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "num_experts": 4,
                    "top_k_experts": 2,
                    "rope_parameters": {
                      "sliding_attention": {
                        "rope_type": "default",
                        "rope_theta": 10000
                      },
                      "full_attention": {
                        "rope_type": "proportional",
                        "partial_rotary_factor": 0.5,
                        "rope_theta": 10000
                      }
                    }
                  }
                }
                """.data(using: .utf8)!,
        )
    }

    private static func smallVisionConfig() throws -> DiffusionGemmaConfiguration {
        try JSONDecoder.json5().decode(
            DiffusionGemmaConfiguration.self,
            from: """
                {
                  "model_type": "diffusion_gemma",
                  "canvas_length": 4,
                  "eos_token_id": 0,
                  "image_token_id": 30,
                  "video_token_id": 31,
                  "text_config": {
                    "model_type": "diffusion_gemma_text",
                    "vocab_size": 32,
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "moe_intermediate_size": 4,
                    "num_hidden_layers": 2,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 1,
                    "num_global_key_value_heads": 1,
                    "head_dim": 4,
                    "global_head_dim": 4,
                    "rms_norm_eps": 0.000001,
                    "max_position_embeddings": 64,
                    "pad_token_id": 0,
                    "eos_token_id": 0,
                    "tie_word_embeddings": true,
                    "sliding_window": 4,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "num_experts": 4,
                    "top_k_experts": 2,
                    "rope_parameters": {
                      "sliding_attention": {
                        "rope_type": "default",
                        "rope_theta": 10000
                      },
                      "full_attention": {
                        "rope_type": "proportional",
                        "partial_rotary_factor": 0.5,
                        "rope_theta": 10000
                      }
                    }
                  },
                  "vision_config": {
                    "model_type": "gemma4_vision",
                    "hidden_size": 8,
                    "intermediate_size": 16,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 2,
                    "num_key_value_heads": 2,
                    "head_dim": 4,
                    "patch_size": 2,
                    "rms_norm_eps": 0.000001,
                    "default_output_length": 2,
                    "position_embedding_size": 8,
                    "pooling_kernel_size": 1,
                    "use_clipped_linears": false,
                    "standardize": false,
                    "rope_parameters": {
                      "rope_type": "default",
                      "rope_theta": 100
                    }
                  }
                }
                """.data(using: .utf8)!,
        )
    }
}
