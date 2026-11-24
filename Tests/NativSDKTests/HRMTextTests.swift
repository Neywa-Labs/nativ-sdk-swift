import MLX
import XCTest

@testable import NativSDK

final class HRMTextTests: XCTestCase {
    func testConfigurationExpandsStackLayersIntoRecurrentCacheLayers() throws {
        let config = try JSONDecoder.json5().decode(
            HRMTextConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "hrm_text",
                  "vocab_size": 65536,
                  "hidden_size": 1536,
                  "intermediate_size": 4096,
                  "num_hidden_layers": 16,
                  "num_attention_heads": 12,
                  "num_key_value_heads": 12,
                  "head_dim": 128,
                  "H_cycles": 2,
                  "L_cycles": 3,
                  "L_bp_cycles": [0, 3],
                  "max_position_embeddings": 4096,
                  "rms_norm_eps": 0.000001,
                  "rope_theta": 10000.0,
                  "tie_word_embeddings": false,
                  "initializer_range": 0.025515518153991442,
                  "embedding_scale": 39.191835884530846,
                  "prefix_lm": true,
                  "pad_token_id": 5,
                  "bos_token_id": 6,
                  "eos_token_id": 11
                }
                """.data(using: .utf8)),
        )

        XCTAssertEqual(config.modelType, "hrm_text")
        XCTAssertEqual(config.layersPerStack, 16)
        XCTAssertEqual(config.hiddenLayers, 128)
        XCTAssertEqual(config.highCycles, 2)
        XCTAssertEqual(config.lowCycles, 3)
        XCTAssertEqual(config.lowBackpropCycles, [0, 3])
        XCTAssertEqual(config.eosTokenID?.values, [11])
    }

    func testSanitizerSplitsFusedAttentionAndMLPWeights() throws {
        let model = try HRMText(Self.smallConfig())
        let sanitized = model.sanitize(weights: [
            "model.L_module.layers.0.attn.gqkv_proj.weight": zeros([32, 8]),
            "model.L_module.layers.0.attn.gqkv_proj.scales": zeros([32, 1]),
            "model.L_module.layers.0.attn.gqkv_proj.bias": zeros([32]),
            "model.L_module.layers.0.mlp.gate_up_proj.weight": zeros([32, 8]),
            "model.L_module.layers.0.mlp.gate_up_proj.scales": zeros([32, 1]),
            "model.L_module.layers.0.mlp.gate_up_proj.bias": zeros([32]),
            "lm_head.weight": zeros([32, 8]),
        ])

        XCTAssertEqual(
            sanitized["language_model.model.L_module.layers.0.self_attn.gate_proj.weight"]?
                .shape,
            [8, 8],
        )
        XCTAssertEqual(
            sanitized["language_model.model.L_module.layers.0.self_attn.q_proj.weight"]?
                .shape,
            [8, 8],
        )
        XCTAssertEqual(
            sanitized["language_model.model.L_module.layers.0.self_attn.k_proj.weight"]?
                .shape,
            [8, 8],
        )
        XCTAssertEqual(
            sanitized["language_model.model.L_module.layers.0.self_attn.v_proj.weight"]?
                .shape,
            [8, 8],
        )
        XCTAssertEqual(
            sanitized["language_model.model.L_module.layers.0.self_attn.gate_proj.bias"]?
                .shape,
            [8],
        )
        XCTAssertEqual(
            sanitized["language_model.model.L_module.layers.0.self_attn.gate_proj.scales"]?
                .shape,
            [8, 1],
        )
        XCTAssertEqual(
            sanitized["language_model.model.L_module.layers.0.self_attn.q_proj.scales"]?
                .shape,
            [8, 1],
        )
        XCTAssertEqual(
            sanitized["language_model.model.L_module.layers.0.mlp.gate_proj.weight"]?
                .shape,
            [16, 8],
        )
        XCTAssertEqual(
            sanitized["language_model.model.L_module.layers.0.mlp.up_proj.weight"]?.shape,
            [16, 8],
        )
        XCTAssertEqual(
            sanitized["language_model.model.L_module.layers.0.mlp.gate_proj.scales"]?.shape,
            [16, 1],
        )
        XCTAssertEqual(
            sanitized["language_model.model.L_module.layers.0.mlp.up_proj.scales"]?.shape,
            [16, 1],
        )
        XCTAssertEqual(
            sanitized["language_model.model.L_module.layers.0.mlp.gate_proj.bias"]?.shape,
            [16],
        )
        XCTAssertEqual(sanitized["language_model.lm_head.weight"]?.shape, [32, 8])
    }

    func testReducedForwardProducesLogitsAndExpandedCache() throws {
        let model = try HRMText(Self.smallConfig())
        let cache = try model.newCache(configuration: nil)

        XCTAssertEqual(cache.count, 2)
        XCTAssertTrue(cache[0] is KVCacheSimple)
        XCTAssertTrue(cache[1] is KVCacheSimple)

        let logits = model(MLXArray([1, 2, 3], [1, 3]), cache: cache)
        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    private static func smallConfig() throws -> HRMTextConfiguration {
        try JSONDecoder.json5().decode(
            HRMTextConfiguration.self,
            from: """
                {
                  "model_type": "hrm_text",
                  "vocab_size": 32,
                  "hidden_size": 8,
                  "intermediate_size": 16,
                  "num_hidden_layers": 1,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 2,
                  "head_dim": 4,
                  "max_position_embeddings": 64,
                  "H_cycles": 1,
                  "L_cycles": 1,
                  "embedding_scale": 1.0,
                  "rms_norm_eps": 0.000001,
                  "tie_word_embeddings": false
                }
                """.data(using: .utf8)!,
        )
    }
}
