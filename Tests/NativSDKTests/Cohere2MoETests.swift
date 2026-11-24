import MLX
import XCTest

@testable import NativSDK

private struct Cohere2MoETestTokenizer: Tokenizer {
    let hasChatTemplate: Bool

    var bosToken: String? {
        "<BOS_TOKEN>"
    }

    var eosToken: String? {
        "<|END_OF_TURN_TOKEN|>"
    }

    var unknownToken: String? {
        "<UNK>"
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var tokens = text.unicodeScalars.map { Int($0.value) % 64 }
        if addSpecialTokens {
            tokens.insert(2, at: 0)
        }
        return tokens
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map(String.init).joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        switch token {
        case "<BOS_TOKEN>": 2
        case "<|END_OF_TURN_TOKEN|>": 255_001
        case "<UNK>": 1
        default: nil
        }
    }

    func convertIdToToken(_ id: Int) -> String? {
        switch id {
        case 2: "<BOS_TOKEN>"
        case 255_001: "<|END_OF_TURN_TOKEN|>"
        case 1: "<UNK>"
        default: nil
        }
    }

    func applyChatTemplate(
        messages _: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?,
    )
        throws -> [Int]
    {
        guard hasChatTemplate else {
            throw TokenizerError.missingChatTemplate
        }
        return [2, 10, 11, 255_001]
    }
}

final class Cohere2MoETests: XCTestCase {
    func testAliasCheckpointLoadsAndGeneratesWithoutProcessorMetadata() async throws {
        try await assertTextCheckpoint(
            configuration: JSONEncoder().encode(Self.smallConfig()), modelType: "cohere2moe")
    }

    func testConfigurationDecodesNorthMiniCodeShape() throws {
        let config = try JSONDecoder.json5().decode(
            Cohere2MoEConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "architectures": ["Cohere2MoeForCausalLM"],
                  "attention_bias": false,
                  "bos_token_id": 2,
                  "eos_token_id": 255001,
                  "expert_selection_fn": "sigmoid",
                  "first_k_dense_replace": 1,
                  "head_dim": 128,
                  "hidden_size": 2048,
                  "intermediate_size": 768,
                  "layer_types": ["full_attention", "sliding_attention"],
                  "logit_scale": 1.0,
                  "model_type": "cohere2_moe",
                  "norm_topk_prob": false,
                  "num_attention_heads": 32,
                  "num_experts": 128,
                  "num_experts_per_tok": 8,
                  "num_hidden_layers": 49,
                  "num_key_value_heads": 4,
                  "num_shared_experts": 0,
                  "pad_token_id": 0,
                  "prefix_dense_intermediate_size": 3072,
                  "prefix_dense_sliding_window_pattern": 1,
                  "rms_norm_eps": 0.000001,
                  "rope_theta": 50000,
                  "shared_expert_combination_strategy": "average",
                  "sliding_window": 4096,
                  "vocab_size": 262144
                }
                """.data(using: .utf8)),
        )

        XCTAssertEqual(config.modelType, "cohere2_moe")
        XCTAssertEqual(config.hiddenSize, 2048)
        XCTAssertEqual(config.hiddenLayers, 49)
        XCTAssertEqual(config.prefixDenseIntermediateSize, 3072)
        XCTAssertEqual(config.moeNumSharedExperts, 0)
        XCTAssertEqual(config.moeGateActivation, "sigmoid")
        XCTAssertEqual(config.eosTokenID?.values, [255_001])
        XCTAssertFalse(config.isSlidingLayer(0))
        XCTAssertTrue(config.isSlidingLayer(1))
    }

    func testTextProcessorUsesChatTemplateAndPlainTextFallback() async throws {
        let templateProcessor = TextProcessor(
            .init(),
            tokenizer: Cohere2MoETestTokenizer(hasChatTemplate: true),
        )
        let templated = try await templateProcessor.prepare(input: UserInput(prompt: "hello"))
        XCTAssertEqual(templated.text.tokens.asArray(Int.self), [2, 10, 11, 255_001])

        let fallbackProcessor = TextProcessor(
            .init(),
            tokenizer: Cohere2MoETestTokenizer(hasChatTemplate: false),
        )
        let fallback = try await fallbackProcessor.prepare(
            input: UserInput(chat: [.system("system"), .user("hello")]),
        )

        let fallbackTokens = fallback.text.tokens.asArray(Int.self)
        XCTAssertFalse(fallbackTokens.isEmpty)
        XCTAssertEqual(fallbackTokens.first, 2)
    }

    func testSanitizerStacksExpertsAndKeepsPublishedPrefixedKeys() throws {
        let model = try Cohere2MoE(Self.smallConfig())
        let sanitized = model.sanitize(weights: [
            "language_model.model.embed_tokens.weight": zeros([32, 8]),
            "model.layers.1.mlp.experts.0.gate_proj.weight": zeros([4, 8]),
            "model.layers.1.mlp.experts.1.gate_proj.weight": ones([4, 8]),
            "model.layers.1.mlp.experts.0.up_proj.weight": zeros([4, 8]),
            "model.layers.1.mlp.experts.1.up_proj.weight": ones([4, 8]),
            "model.layers.1.mlp.experts.0.down_proj.weight": zeros([8, 4]),
            "model.layers.1.mlp.experts.1.down_proj.weight": ones([8, 4]),
            "model.layers.1.self_attn.q_proj.bias": zeros([8]),
            "model.layers.1.input_layernorm.bias": zeros([8]),
            "model.layers.1.self_attn.rotary_emb.inv_freq": zeros([2]),
        ])

        XCTAssertNotNil(sanitized["language_model.model.embed_tokens.weight"])
        XCTAssertEqual(
            sanitized["language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape,
            [2, 4, 8],
        )
        XCTAssertEqual(
            sanitized["language_model.model.layers.1.mlp.switch_mlp.up_proj.weight"]?.shape,
            [2, 4, 8],
        )
        XCTAssertEqual(
            sanitized["language_model.model.layers.1.mlp.switch_mlp.down_proj.weight"]?.shape,
            [2, 8, 4],
        )
        XCTAssertNil(sanitized["language_model.model.layers.1.mlp.experts.0.gate_proj.weight"])
        XCTAssertNil(sanitized["language_model.model.layers.1.self_attn.q_proj.bias"])
        XCTAssertNil(sanitized["language_model.model.layers.1.input_layernorm.bias"])
        XCTAssertNil(sanitized["language_model.model.layers.1.self_attn.rotary_emb.inv_freq"])
    }

    func testReducedForwardProducesLogitsAndSlidingCache() throws {
        let model = try Cohere2MoE(Self.smallConfig())
        let cache = try model.newCache(configuration: nil)
        XCTAssertTrue(cache[0] is KVCacheSimple)
        XCTAssertTrue(cache[1] is RotatingKVCache)

        let logits = model(MLXArray([1, 2, 3], [1, 3]), cache: cache)
        XCTAssertEqual(logits.shape, [1, 3, 32])
    }

    private static func smallConfig() throws -> Cohere2MoEConfiguration {
        try JSONDecoder.json5().decode(
            Cohere2MoEConfiguration.self,
            from: """
                {
                  "model_type": "cohere2_moe",
                  "vocab_size": 32,
                  "hidden_size": 8,
                  "intermediate_size": 4,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1,
                  "head_dim": 4,
                  "layer_norm_eps": 0.00001,
                  "rms_norm_eps": 0.000001,
                  "sliding_window": 8,
                  "layer_types": ["full_attention", "sliding_attention"],
                  "first_k_dense_replace": 1,
                  "prefix_dense_intermediate_size": 16,
                  "num_experts": 2,
                  "num_experts_per_tok": 1,
                  "norm_topk_prob": true,
                  "moe_num_shared_experts": 1,
                  "logit_scale": 1.0
                }
                """.data(using: .utf8)!,
        )
    }
}
