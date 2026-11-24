import MLX
import XCTest

@testable import NativSDK

final class LLaDA2Tests: XCTestCase {
    func testTinyLLaDADiffusionMatchesEngineAndExecutor() async throws {
        let config = try Self.smallConfig(numExperts: 2)
        let model = withRandomState(MLXRandom.RandomState(seed: 6)) { LLaDA2(config) }
        model.train(false)
        let context = ModelContext(
            metadata: .init(),
            model: model, processor: ExecutionFixtureProcessor(),
            tokenizer: ExecutionFixtureTokenizer())
        let diffusion = DiffusionGenerateParameters(
            maxTokens: 4, blockLength: 2, steps: 2, eosEarlyStop: false, skipSpecialTokens: false)
        var expected: [String] = []
        _ = try runDiffusion(
            input: LMInput(tokens: MLXArray([1, 2])),
            options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 4)),
            diffusionParameters: diffusion, context: context,
            mode: .diffusion
        ) {
            expected.append(
                context.tokenizer.decode(
                    tokenIds: $0, skipSpecialTokens: diffusion.skipSpecialTokens))
        }
        let directory = URL(fileURLWithPath: "/tmp/llada-fixture")
        let loaded = LoadedModel(
            assets: .init(
                source: .directory(directory), modelDirectory: directory,
                tokenizerDirectory: directory),
            metadata: .init(), model: model,
            processor: ExecutionFixtureProcessor(), tokenizer: ExecutionFixtureTokenizer())
        let request = loaded.intoNativModel().generate(
            input: UserInput(prompt: "fixture"),
            options: .init(
                sampling: .init(temperature: 0), output: .init(maxTokens: 4), diffusion: diffusion))
        var actual: [String] = []
        for try await event in request {
            if case .chunk(let chunk) = event { actual.append(chunk) }
        }
        XCTAssertEqual(actual, expected)
        let summary = try await request.metrics()
        XCTAssertEqual(summary.generatedTokenCount, 4)
    }

    func testConfigurationDecodesMiniShapeAndDefaults() throws {
        let config = try JSONDecoder.json5().decode(
            LLaDA2Configuration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "llada2_moe",
                  "vocab_size": 157184,
                  "hidden_size": 2048,
                  "intermediate_size": 5120,
                  "num_hidden_layers": 20,
                  "num_attention_heads": 16,
                  "num_key_value_heads": 4,
                  "head_dim": 128,
                  "use_qkv_bias": false,
                  "use_qk_norm": true,
                  "use_bias": false,
                  "max_position_embeddings": 32768,
                  "rope_theta": 600000,
                  "partial_rotary_factor": 0.5,
                  "num_experts": 256,
                  "num_shared_experts": 1,
                  "num_experts_per_tok": 8,
                  "n_group": 8,
                  "topk_group": 4,
                  "routed_scaling_factor": 2.5,
                  "moe_intermediate_size": 512,
                  "first_k_dense_replace": 1,
                  "pad_token_id": 156892,
                  "score_function": "sigmoid",
                  "moe_router_enable_expert_bias": true
                }
                """.data(using: .utf8)),
        )

        XCTAssertEqual(config.modelType, "llada2_moe")
        XCTAssertEqual(config.vocabularySize, 157_184)
        XCTAssertEqual(config.hiddenLayers, 20)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.rotaryDim, 64)
        XCTAssertEqual(config.eosTokenID.values, [156_892])
        XCTAssertEqual(config.maskTokenID, 156_895)
        XCTAssertFalse(config.usesSparseMLP(0))
        XCTAssertTrue(config.usesSparseMLP(1))
    }

    func testSanitizerPrefixesAndStacksLegacyExperts() throws {
        let model = try LLaDA2(Self.smallConfig(numExperts: 2))
        let sanitized = model.sanitize(weights: [
            "model.word_embeddings.weight": zeros([32, 8]),
            "lm_head.weight": zeros([32, 8]),
            "model.layers.1.mlp.gate.weight": zeros([2, 8]),
            "model.layers.1.mlp.gate.expert_bias": zeros([2]),
            "model.layers.1.mlp.experts.0.gate_proj.weight": zeros([4, 8]),
            "model.layers.1.mlp.experts.1.gate_proj.weight": ones([4, 8]),
            "model.layers.1.mlp.experts.0.up_proj.weight": zeros([4, 8]),
            "model.layers.1.mlp.experts.1.up_proj.weight": ones([4, 8]),
            "model.layers.1.mlp.experts.0.down_proj.weight": zeros([8, 4]),
            "model.layers.1.mlp.experts.1.down_proj.weight": ones([8, 4]),
            "model.layers.1.attention.rotary_emb.inv_freq": zeros([2]),
            "model.layers.1.position_ids": zeros([1]),
        ])

        XCTAssertEqual(sanitized["language_model.model.word_embeddings.weight"]?.shape, [32, 8])
        XCTAssertEqual(sanitized["language_model.lm_head.weight"]?.shape, [32, 8])
        XCTAssertEqual(
            sanitized["language_model.model.layers.1.mlp.gate.weight"]?.shape,
            [2, 8],
        )
        XCTAssertEqual(
            sanitized["language_model.model.layers.1.mlp.gate.expert_bias"]?.shape,
            [2],
        )
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
        XCTAssertNil(sanitized["language_model.model.layers.1.attention.rotary_emb.inv_freq"])
        XCTAssertNil(sanitized["language_model.model.layers.1.position_ids"])
    }

    private static func smallConfig(numExperts: Int = 4) throws -> LLaDA2Configuration {
        try JSONDecoder.json5().decode(
            LLaDA2Configuration.self,
            from: """
                {
                  "model_type": "llada2_moe",
                  "vocab_size": 32,
                  "hidden_size": 8,
                  "intermediate_size": 16,
                  "num_hidden_layers": 2,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 1,
                  "head_dim": 4,
                  "use_qkv_bias": false,
                  "use_qk_norm": true,
                  "use_bias": false,
                  "rms_norm_eps": 0.000001,
                  "max_position_embeddings": 64,
                  "rope_theta": 10000,
                  "partial_rotary_factor": 0.5,
                  "rotary_dim": 2,
                  "num_experts": \(numExperts),
                  "num_shared_experts": 1,
                  "num_experts_per_tok": 2,
                  "n_group": 2,
                  "topk_group": 1,
                  "routed_scaling_factor": 1.0,
                  "moe_intermediate_size": 4,
                  "first_k_dense_replace": 1,
                  "norm_topk_prob": true,
                  "score_function": "sigmoid",
                  "moe_router_enable_expert_bias": true,
                  "pad_token_id": 0
                }
                """.data(using: .utf8)!,
        )
    }
}
