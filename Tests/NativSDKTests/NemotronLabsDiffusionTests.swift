import MLX
import XCTest

@testable import NativSDK

final class NemotronLabsDiffusionTests: XCTestCase {
    func testConfigurationDecodesDefaultsAndDiffusionControls() throws {
        let config = try JSONDecoder.json5().decode(
            NemotronLabsDiffusionConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "nemotron_labs_diffusion",
                  "vocab_size": 131072,
                  "hidden_size": 4096,
                  "intermediate_size": 14336,
                  "num_hidden_layers": 34,
                  "num_attention_heads": 32,
                  "num_key_value_heads": 8,
                  "head_dim": 128,
                  "rope_theta": 1000000.0,
                  "mask_token_id": 100,
                  "default_generation_mode": "ar",
                  "default_diffusion_steps": 32,
                  "default_diffusion_threshold": 0.9,
                  "default_diffusion_min_threshold": 0.45,
                  "block_size": 32
                }
                """.data(using: .utf8)),
        )

        XCTAssertEqual(config.modelType, "nemotron_labs_diffusion")
        XCTAssertEqual(config.vocabularySize, 131_072)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.ropeTheta, 1_000_000)
        XCTAssertEqual(config.maskTokenID, 100)
        XCTAssertEqual(config.defaultGenerationMode, "ar")
        XCTAssertEqual(config.defaultDiffusionSteps, 32)
        XCTAssertEqual(config.eosTokenID.values, [11])
    }

    func testSanitizerRemapsEncoderAndDiffusionHead() throws {
        let model = try NemotronLabsDiffusion(Self.smallConfig())
        let sanitized = model.sanitize(weights: [
            "encoder.embed_tokens.weight": zeros([32, 8]),
            "encoder.layers.0.self_attn.q_proj.weight": zeros([8, 8]),
            "diffusion_head.weight": zeros([32, 8]),
            "encoder.layers.0.self_attn.rotary_emb.inv_freq": zeros([1]),
            "encoder.position_ids": zeros([1]),
        ])

        XCTAssertEqual(sanitized["language_model.model.embed_tokens.weight"]?.shape, [32, 8])
        XCTAssertEqual(
            sanitized["language_model.model.layers.0.self_attn.q_proj.weight"]?.shape,
            [8, 8],
        )
        XCTAssertEqual(sanitized["language_model.diffusion_head.weight"]?.shape, [32, 8])
        XCTAssertNil(sanitized["language_model.model.layers.0.self_attn.rotary_emb.inv_freq"])
        XCTAssertNil(sanitized["language_model.model.position_ids"])
    }

    func testReducedForwardAndDiffusionLogitsProduceLogits() throws {
        let model = try NemotronLabsDiffusion(Self.smallConfig())
        let cache = try model.newCache(configuration: nil)
        XCTAssertEqual(cache.count, 2)

        let tokens = MLXArray([1, 2, 3], [1, 3])
        let logits = model(tokens, cache: cache)
        XCTAssertEqual(logits.shape, [1, 3, 32])

        let diffusionLogits = model.diffusionLogits(tokens)
        XCTAssertEqual(diffusionLogits.shape, [1, 3, 32])
        XCTAssertFalse(model.diffusionPrefersDiffusionGeneration)
        XCTAssertEqual(model.diffusionMaskTokenID, 7)
    }

    private static func smallConfig() throws -> NemotronLabsDiffusionConfiguration {
        try JSONDecoder.json5().decode(
            NemotronLabsDiffusionConfiguration.self,
            from: """
                {
                  "model_type": "nemotron_labs_diffusion",
                  "vocab_size": 32,
                  "hidden_size": 8,
                  "intermediate_size": 16,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1,
                  "head_dim": 4,
                  "max_position_embeddings": 64,
                  "rms_norm_eps": 0.000001,
                  "rope_theta": 10000,
                  "attention_bias": false,
                  "mlp_bias": false,
                  "mask_token_id": 7,
                  "eos_token_id": 0,
                  "tie_word_embeddings": false,
                  "default_generation_mode": "ar"
                }
                """.data(using: .utf8)!,
        )
    }
}
