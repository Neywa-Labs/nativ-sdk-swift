import MLX
import XCTest

@testable import NativSDK

final class DeepSeekV4Tests: XCTestCase {
    func testConfigurationDecodesFlashShapeAndTruncatesCompressRatios() throws {
        let config = try JSONDecoder.json5().decode(
            DeepSeekV4Configuration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "deepseek_v4",
                  "vocab_size": 129280,
                  "hidden_size": 4096,
                  "intermediate_size": 18432,
                  "moe_intermediate_size": 2048,
                  "num_hidden_layers": 43,
                  "num_attention_heads": 64,
                  "num_key_value_heads": 1,
                  "n_shared_experts": 1,
                  "n_routed_experts": 256,
                  "num_experts_per_tok": 6,
                  "q_lora_rank": 1024,
                  "qk_rope_head_dim": 64,
                  "head_dim": 512,
                  "hc_mult": 4,
                  "sliding_window": 128,
                  "o_groups": 8,
                  "o_lora_rank": 1024,
                  "num_hash_layers": 3,
                  "scoring_func": "sqrtsoftplus",
                  "topk_method": "noaux_tc",
                  "compress_ratios": [
                    0, 0, 4, 128, 4, 128, 4, 128, 4, 128, 4,
                    128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 128,
                    4, 128, 4, 128, 4, 128, 4, 128, 4, 128, 4,
                    128, 4, 128, 4, 128, 4, 128, 4, 128, 4, 0
                  ],
                  "rope_scaling": {
                    "type": "yarn",
                    "factor": 32.0,
                    "original_max_position_embeddings": 32768
                  },
                  "eos_token_id": 129001
                }
                """.data(using: .utf8)),
        )

        XCTAssertEqual(config.modelType, "deepseek_v4")
        XCTAssertEqual(config.hiddenLayers, 43)
        XCTAssertEqual(config.attentionHeads, 64)
        XCTAssertEqual(config.kvHeads, 1)
        XCTAssertEqual(config.headDim, 512)
        XCTAssertEqual(config.compressRatios.count, 43)
        XCTAssertEqual(Array(config.compressRatios.prefix(4)), [0, 0, 4, 128])
        XCTAssertEqual(config.compressRatios.last, 4)
        XCTAssertEqual(config.ropeScaling?["type"]?.asString(), "yarn")
        XCTAssertEqual(config.eosTokenID?.values, [129_001])
    }

    func testSanitizerPrefixesRemapsAndStacksLegacyExperts() throws {
        let model = try DeepSeekV4(Self.smallConfig(nRoutedExperts: 2))
        let sanitized = model.sanitize(weights: [
            "embed.weight": zeros([32, 8]),
            "head.weight": zeros([32, 8]),
            "hc_head_fn": zeros([2, 16]),
            "hc_head_base": zeros([2]),
            "layers.0.hc_attn_fn": zeros([8, 16]),
            "layers.0.hc_ffn_scale": ones([3]),
            "model.layers.0.ffn.gate.bias": zeros([2]),
            "model.layers.0.ffn.gate.bias_vl": MLXArray([Float(2), 3]),
            "model.layers.0.ffn.gate.tid2eid": ones([32, 1], dtype: .float32),
            "model.layers.0.ffn.experts.0.w1.weight": zeros([4, 8]),
            "model.layers.0.ffn.experts.1.w1.weight": ones([4, 8]),
            "model.layers.0.ffn.experts.0.w2.weight": zeros([8, 4]),
            "model.layers.0.ffn.experts.1.w2.weight": ones([8, 4]),
            "model.layers.0.ffn.experts.0.w3.weight": zeros([4, 8]),
            "model.layers.0.ffn.experts.1.w3.weight": ones([4, 8]),
            "model.layers.0.attn.wo_a.weight": zeros([2, 8]),
            "model.layers.0.attn.rotary_emb.inv_freq": zeros([1]),
            "mtp.layers.0.weight": zeros([1]),
        ])

        XCTAssertEqual(sanitized["language_model.model.embed_tokens.weight"]?.shape, [32, 8])
        XCTAssertEqual(sanitized["language_model.lm_head.weight"]?.shape, [32, 8])
        XCTAssertEqual(sanitized["language_model.model.hc_head.fn"]?.shape, [2, 16])
        XCTAssertEqual(sanitized["language_model.model.hc_head.base"]?.shape, [2])
        XCTAssertEqual(
            sanitized["language_model.model.layers.0.attn_hc.fn"]?.shape,
            [8, 16],
        )
        XCTAssertEqual(
            sanitized["language_model.model.layers.0.ffn_hc.scale"]?.shape,
            [3],
        )
        XCTAssertEqual(
            sanitized["language_model.model.layers.0.ffn.gate.e_score_correction_bias"]?.shape,
            [2],
        )
        XCTAssertEqual(
            sanitized["language_model.model.layers.0.ffn.gate.bias_vl"]?.asArray(Float.self),
            [2, 3])
        XCTAssertNil(sanitized["language_model.model.layers.0.ffn.gate.e_score_correction_bias_vl"])
        XCTAssertEqual(
            sanitized["language_model.model.layers.0.ffn.gate.tid2eid"]?.dtype,
            .int32,
        )
        XCTAssertEqual(
            sanitized["language_model.model.layers.0.ffn.switch_mlp.gate_proj.weight"]?.shape,
            [2, 4, 8],
        )
        XCTAssertEqual(
            sanitized["language_model.model.layers.0.ffn.switch_mlp.down_proj.weight"]?.shape,
            [2, 8, 4],
        )
        XCTAssertEqual(
            sanitized["language_model.model.layers.0.attn.wo_a.weight"]?.shape,
            [2, 2, 4],
        )
        XCTAssertNil(
            sanitized["language_model.model.layers.0.ffn.experts.0.w1.weight"],
        )
        XCTAssertNil(
            sanitized["language_model.model.layers.0.attn.rotary_emb.inv_freq"],
        )
        XCTAssertNil(sanitized["language_model.mtp.layers.0.weight"])
    }

    func testReducedForwardProducesLogitsForLocalAndCompressedAttention() throws {
        let localModel = try DeepSeekV4(Self.smallConfig(compressRatios: [0, 0]))
        let localLogits = localModel(
            MLXArray([1, 2, 3], [1, 3]), cache: localModel.newCache(configuration: nil))
        XCTAssertEqual(localLogits.shape, [1, 3, 32])

        let compressedModel = try DeepSeekV4(Self.smallConfig(compressRatios: [0, 4]))
        let cache = compressedModel.newCache(configuration: nil)
        XCTAssertEqual(cache.count, 2)
        XCTAssertTrue(cache[0] is RotatingKVCache)
        XCTAssertTrue(cache[1] is CacheList)
        let compressedLogits = compressedModel(
            MLXArray([1, 2, 3, 4], [1, 4]),
            cache: cache,
        )
        XCTAssertEqual(compressedLogits.shape, [1, 4, 32])
    }

    private static func smallConfig(
        compressRatios: [Int] = [0, 0],
        nRoutedExperts: Int = 4,
    )
        throws -> DeepSeekV4Configuration
    {
        try JSONDecoder.json5().decode(
            DeepSeekV4Configuration.self,
            from: """
                {
                  "model_type": "deepseek_v4",
                  "vocab_size": 32,
                  "hidden_size": 8,
                  "intermediate_size": 16,
                  "moe_intermediate_size": 4,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1,
                  "n_shared_experts": 1,
                  "n_routed_experts": \(nRoutedExperts),
                  "routed_scaling_factor": 1.0,
                  "q_lora_rank": 4,
                  "qk_rope_head_dim": 2,
                  "num_experts_per_tok": 1,
                  "norm_topk_prob": true,
                  "hidden_act": "silu",
                  "max_position_embeddings": 64,
                  "rms_norm_eps": 0.000001,
                  "rope_theta": 10000,
                  "attention_bias": false,
                  "head_dim": 4,
                  "scoring_func": "sqrtsoftplus",
                  "compress_ratios": \(jsonArray(compressRatios)),
                  "compress_rope_theta": 10000,
                  "hc_mult": 2,
                  "hc_sinkhorn_iters": 2,
                  "hc_eps": 0.000001,
                  "num_hash_layers": 1,
                  "swiglu_limit": 0,
                  "sliding_window": 8,
                  "o_groups": 2,
                  "o_lora_rank": 2,
                  "index_n_heads": 2,
                  "index_head_dim": 4,
                  "index_topk": 4,
                  "index_block": 0,
                  "index_keep": 0,
                  "tie_word_embeddings": false
                }
                """.data(using: .utf8)!,
        )
    }

    private static func jsonArray(_ values: [Int]) -> String {
        "[" + values.map(String.init).joined(separator: ", ") + "]"
    }
}

extension StringOrNumber {
    fileprivate func asString() -> String? {
        if case .string(let value) = self {
            return value
        }
        return nil
    }
}
