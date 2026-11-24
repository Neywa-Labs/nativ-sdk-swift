import MLX
import XCTest

@testable import NativSDK

final class LagunaTests: XCTestCase {
    func testConfigurationDecodesLagunaXSShape() throws {
        let config = try JSONDecoder.json5().decode(
            LagunaConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "laguna",
                  "vocab_size": 100352,
                  "hidden_size": 2048,
                  "intermediate_size": 8192,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 48,
                  "num_key_value_heads": 8,
                  "head_dim": 128,
                  "max_position_embeddings": 131072,
                  "gating": true,
                  "layer_types": ["full_attention", "sliding_attention"],
                  "mlp_layer_types": ["dense", "sparse"],
                  "moe_intermediate_size": 512,
                  "moe_routed_scaling_factor": 2.5,
                  "num_attention_heads_per_layer": [48, 64],
                  "num_experts": 256,
                  "num_experts_per_tok": 8,
                  "partial_rotary_factor": 0.5,
                  "rope_parameters": {
                    "full_attention": {
                      "rope_theta": 500000.0,
                      "rope_type": "yarn",
                      "factor": 32.0,
                      "original_max_position_embeddings": 4096
                    },
                    "sliding_attention": {
                      "rope_type": "default",
                      "rope_theta": 10000.0,
                      "partial_rotary_factor": 1.0
                    }
                  },
                  "rms_norm_eps": 0.000001,
                  "shared_expert_intermediate_size": 512,
                  "sliding_window": 512,
                  "tie_word_embeddings": false,
                  "eos_token_id": [2, 24],
                  "pad_token_id": 9
                }
                """.data(using: .utf8)),
        )

        XCTAssertEqual(config.modelType, "laguna")
        XCTAssertEqual(config.hiddenLayers, 2)
        XCTAssertEqual(config.gating, "per-head")
        XCTAssertEqual(config.attentionHeadsPerLayer, [48, 64])
        XCTAssertEqual(config.mlpOnlyLayers, [0])
        XCTAssertFalse(config.usesSparseMLP(0))
        XCTAssertTrue(config.usesSparseMLP(1))
        XCTAssertEqual(config.eosTokenID?.values, [2, 24])
    }

    func testSanitizerRemapsRouterStacksExpertsAndPrefixesWrapperKeys() throws {
        let model = try Laguna(Self.smallConfig())
        let sanitized = model.sanitize(weights: [
            "lm_head.weight": zeros([32, 8]),
            "model.layers.1.mlp.gate.weight": zeros([2, 8]),
            "model.layers.1.mlp.experts.e_score_correction_bias": zeros([2]),
            "model.layers.1.mlp.experts.0.gate_proj.weight": zeros([4, 8]),
            "model.layers.1.mlp.experts.1.gate_proj.weight": ones([4, 8]),
            "model.layers.1.mlp.experts.0.up_proj.weight": zeros([4, 8]),
            "model.layers.1.mlp.experts.1.up_proj.weight": ones([4, 8]),
            "model.layers.1.mlp.experts.0.down_proj.weight": zeros([8, 4]),
            "model.layers.1.mlp.experts.1.down_proj.weight": ones([8, 4]),
            "model.layers.1.self_attn.rotary_emb.inv_freq": zeros([2]),
            "model.layers.1.self_attn.k_scale": zeros([1]),
        ])

        XCTAssertEqual(sanitized["language_model.lm_head.weight"]?.shape, [32, 8])
        XCTAssertEqual(
            sanitized["language_model.model.layers.1.mlp.gate.proj.weight"]?.shape,
            [2, 8],
        )
        XCTAssertEqual(
            sanitized["language_model.model.layers.1.mlp.gate.e_score_correction_bias"]?.shape,
            [2],
        )
        XCTAssertEqual(
            sanitized["language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape,
            [2, 4, 8],
        )
        XCTAssertEqual(
            sanitized["language_model.model.layers.1.mlp.switch_mlp.down_proj.weight"]?.shape,
            [2, 8, 4],
        )
        XCTAssertNil(sanitized["language_model.model.layers.1.mlp.experts.0.gate_proj.weight"])
        XCTAssertNil(sanitized["language_model.model.layers.1.self_attn.rotary_emb.inv_freq"])
        XCTAssertNil(sanitized["language_model.model.layers.1.self_attn.k_scale"])
    }

    private static func smallConfig() throws -> LagunaConfiguration {
        try JSONDecoder.json5().decode(
            LagunaConfiguration.self,
            from: """
                {
                  "model_type": "laguna",
                  "vocab_size": 32,
                  "hidden_size": 8,
                  "intermediate_size": 16,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1,
                  "head_dim": 4,
                  "max_position_embeddings": 64,
                  "gating": true,
                  "layer_types": ["full_attention", "sliding_attention"],
                  "num_attention_heads_per_layer": [2, 2],
                  "mlp_layer_types": ["dense", "sparse"],
                  "moe_intermediate_size": 4,
                  "num_experts": 2,
                  "num_experts_per_tok": 1,
                  "partial_rotary_factor": 1.0,
                  "rope_parameters": {
                    "full_attention": {"rope_type": "default", "rope_theta": 10000.0},
                    "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}
                  },
                  "rms_norm_eps": 0.000001,
                  "shared_expert_intermediate_size": 4,
                  "sliding_window": 8,
                  "tie_word_embeddings": false
                }
                """.data(using: .utf8)!,
        )
    }
}
