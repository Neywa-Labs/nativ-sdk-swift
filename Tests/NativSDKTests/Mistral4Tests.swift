import MLX
import XCTest

@testable import NativSDK

final class Mistral4Tests: XCTestCase {
    func testOwnedGreedyDecoderMatchesPinnedTinyMistralFixture() throws {
        let config = try Self.smallConfig()
        let model = withRandomState(MLXRandom.RandomState(seed: 1)) { Mistral4(config) }
        let input = LMInput(tokens: MLXArray([1, 2, 3, 4, 5, 6, 7, 8, 9]))
        let options = ModelGenerationOptions(
            prefill: .init(stepSize: 3), sampling: .init(temperature: 0),
            output: .init(maxTokens: 16))
        let expected: [[Int]] = try RuntimeReferenceFixtures.values("mistral")
        let iterator = try AutoregressiveIterator(input: input, model: model, options: options)
        XCTAssertEqual(Array(iterator), expected[0])
        XCTAssertNil(iterator.failure)
        XCTAssertTrue(iterator.cache.allSatisfy { $0.offset == 25 })
        Stream.defaultStream.synchronize()
    }

    func testTinyMistralSpeculationMatchesGreedy() throws {
        let config = try Self.smallConfig()
        for maxCache: Int? in [nil, 8] {
            for draftSeed: UInt64 in [1, 2] {
                let target = withRandomState(MLXRandom.RandomState(seed: 1)) { Mistral4(config) }
                let draft = withRandomState(MLXRandom.RandomState(seed: draftSeed)) {
                    Mistral4(config)
                }
                let input = LMInput(tokens: MLXArray([1, 2, 3, 4, 5]))
                let options = ModelGenerationOptions(
                    cache: try maxCache.map { .init(capacity: try .init(maxTokens: $0)) },
                    sampling: .init(temperature: 0), output: .init(maxTokens: 12))
                let expected = Array(
                    try AutoregressiveIterator(input: input, model: target, options: options))
                let iterator = try SpeculativeIterator(
                    input: input, target: target, draft: draft,
                    options: options, speculation: .init(draftTokenCount: 3))
                var actual: [Int] = []
                while let token = iterator.next() { actual.append(token) }
                XCTAssertNil(iterator.failure)
                XCTAssertEqual(
                    actual, expected, "cache=\(String(describing: maxCache)), draft=\(draftSeed)")
            }
        }
    }

    func testLocalLLMPrefillMatchesManualChunkedForwards() throws {
        let config = try Self.smallConfig()
        let model: any LLMModel = withRandomState(MLXRandom.RandomState(seed: 1)) {
            Mistral4(config)
        }
        let tokens = MLXArray([1, 2, 3, 4, 5, 6, 7, 8, 9])
        for (chunking, ranges, remainingCount) in [
            (ModelPrefillOptions.Chunking.balanced, [0 ..< 3, 3 ..< 6, 6 ..< 8], 1),
            (.remainder, [0 ..< 3, 3 ..< 6], 3), (.unchunked, [], 9),
        ] {
            let referenceCache = try model.newCache(configuration: nil)
            for range in ranges {
                _ = model(tokens[range].expandedDimensions(axis: 0), cache: referenceCache)
                eval(referenceCache)
            }
            let expected = model(
                tokens[(tokens.size - remainingCount)...].expandedDimensions(axis: 0),
                cache: referenceCache)[0, -1, 0...]
            eval(expected)

            let cache = try model.newCache(configuration: nil)
            let prepared = try model.prepare(
                LMInput(tokens: tokens), cache: cache, state: nil,
                prefill: .init(stepSize: 3, chunking: chunking))
            let actual: MLXArray
            switch prepared {
            case .tokens(let remaining):
                XCTAssertEqual(remaining.tokens.size, remainingCount)
                actual =
                    model(remaining.tokens.expandedDimensions(axis: 0), cache: cache)[
                        0, -1, 0...]
            case .logits(let output):
                actual = output.logits[0, -1, 0...]
            }
            XCTAssertTrue(
                allClose(actual, expected, rtol: 1e-4, atol: 1e-5).item(Bool.self),
                "\(chunking): max error \(abs(actual - expected).max().item(Float.self))")
            XCTAssertTrue(cache.allSatisfy { $0.offset == tokens.size })
        }
    }

    func testRegistryRoutesMistral3AndNestedMistral4() async throws {
        let textData = try JSONEncoder().encode(Self.smallConfig())
        let nestedData = try JSONSerialization.data(withJSONObject: [
            "model_type": "mistral3",
            "text_config": JSONSerialization.jsonObject(with: textData),
        ])
        for (type, data) in [("mistral4", textData), ("mistral3", nestedData)] {
            let model = try ModelRegistry.default.createModel(
                configuration: data, modelType: type)
            XCTAssertTrue(model is Mistral4)
        }

        let mistral3Data = Data(
            """
            {
              "model_type": "mistral3",
              "text_config": {
                "model_type": "mistral", "hidden_size": 8, "num_hidden_layers": 1,
                "intermediate_size": 16, "num_attention_heads": 2,
                    "rms_norm_eps": 0.00001, "vocab_size": 32,
                    "rope_parameters": {"rope_theta": 10000.0}
              },
              "vision_config": {
                "model_type": "pixtral", "hidden_size": 8, "num_hidden_layers": 1,
                "num_attention_heads": 2, "intermediate_size": 16,
                "patch_size": 2, "image_size": 8
              }
            }
            """.utf8)
        let model = try ModelRegistry.default.createModel(
            configuration: mistral3Data, modelType: "mistral3")
        XCTAssertTrue(model is Mistral3VLM)
    }

    func testConfigurationDecodesNestedMistral3TextConfig() throws {
        let config = try JSONDecoder.json5().decode(
            Mistral4Configuration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "mistral3",
                  "text_config": {
                    "model_type": "mistral4",
                    "hidden_size": 4096,
                    "num_hidden_layers": 36,
                    "intermediate_size": 12288,
                    "num_attention_heads": 32,
                    "num_key_value_heads": 32,
                    "rms_norm_eps": 0.000001,
                    "vocab_size": 131072,
                    "max_position_embeddings": 1048576,
                    "tie_word_embeddings": false,
                    "q_lora_rank": 1024,
                    "kv_lora_rank": 256,
                    "qk_rope_head_dim": 64,
                    "qk_nope_head_dim": 64,
                    "v_head_dim": 128,
                    "rope_interleave": true,
                    "rope_parameters": {
                      "rope_theta": 10000.0,
                      "llama_4_scaling_beta": 0.1,
                      "original_max_position_embeddings": 8192
                    },
                    "n_routed_experts": 128,
                    "n_shared_experts": 1,
                    "num_experts_per_tok": 4,
                    "moe_intermediate_size": 2048,
                    "first_k_dense_replace": 0,
                    "eos_token_id": 2
                  }
                }
                """.data(using: .utf8)),
        )

        XCTAssertEqual(config.modelType, "mistral4")
        XCTAssertEqual(config.hiddenSize, 4096)
        XCTAssertEqual(config.hiddenLayers, 36)
        XCTAssertEqual(config.vocabularySize, 131_072)
        XCTAssertTrue(config.ropeTraditional)
        XCTAssertEqual(config.llama4ScalingBeta, 0.1, accuracy: 0.00001)
        XCTAssertEqual(config.originalMaxPositionEmbeddings, 8192)
        XCTAssertEqual(config.eosTokenID?.values, [2])
    }

    func testSanitizerDequantizesAndSplitsFusedExpertWeights() throws {
        let model = try Mistral4(Self.smallConfig())
        let gateUp = MLXArray(0 ..< 64, [4, 8, 2]).asType(.float32)
        let down = MLXArray(0 ..< 32, [4, 4, 2]).asType(.float32)
        let qWeight = ones([4, 8])
        let scale = MLXArray(2.0)

        let sanitized = model.sanitize(weights: [
            "language_model.model.layers.0.self_attn.q_a_proj.weight": qWeight,
            "language_model.model.layers.0.self_attn.q_a_proj.weight_scale_inv": scale,
            "language_model.model.layers.0.self_attn.q_a_proj.activation_scale": ones([1]),
            "language_model.model.layers.0.mlp.experts.gate_up_proj": gateUp,
            "language_model.model.layers.0.mlp.experts.gate_up_proj_scale_inv": scale,
            "language_model.model.layers.0.mlp.experts.down_proj": down,
            "language_model.model.layers.0.mlp.experts.down_proj_scale_inv": scale,
            "language_model.model.layers.0.self_attn.rotary_emb.inv_freq": zeros([2]),
            "vision_tower.vision_model.patch_conv.weight": zeros([1]),
        ])

        XCTAssertEqual(
            sanitized["language_model.model.layers.0.self_attn.q_a_proj.weight"]?
                .asArray(Float.self),
            Array(repeating: 2.0, count: 32),
        )
        XCTAssertNil(
            sanitized[
                "language_model.model.layers.0.self_attn.q_a_proj.weight_scale_inv",
            ],
        )
        XCTAssertNil(
            sanitized[
                "language_model.model.layers.0.self_attn.q_a_proj.activation_scale",
            ],
        )
        XCTAssertNil(
            sanitized[
                "language_model.model.layers.0.self_attn.rotary_emb.inv_freq",
            ],
        )
        XCTAssertNil(sanitized["vision_tower.vision_model.patch_conv.weight"])
        XCTAssertEqual(
            sanitized["language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"]?
                .shape,
            [4, 4, 2],
        )
        XCTAssertEqual(
            sanitized["language_model.model.layers.0.mlp.switch_mlp.up_proj.weight"]?
                .shape,
            [4, 4, 2],
        )
        XCTAssertEqual(
            sanitized["language_model.model.layers.0.mlp.switch_mlp.down_proj.weight"]?
                .shape,
            [4, 4, 2],
        )
        XCTAssertNil(sanitized["language_model.model.layers.0.mlp.experts.gate_up_proj"])
        XCTAssertNil(sanitized["language_model.model.layers.0.mlp.experts.down_proj"])
    }

    private static func smallConfig() throws -> Mistral4Configuration {
        try JSONDecoder.json5().decode(
            Mistral4Configuration.self,
            from: """
                {
                  "model_type": "mistral4",
                  "hidden_size": 8,
                  "num_hidden_layers": 2,
                  "intermediate_size": 16,
                  "num_attention_heads": 2,
                  "num_key_value_heads": 2,
                  "rms_norm_eps": 0.000001,
                  "vocab_size": 32,
                  "max_position_embeddings": 64,
                  "tie_word_embeddings": false,
                  "q_lora_rank": 4,
                  "kv_lora_rank": 4,
                  "qk_rope_head_dim": 2,
                  "qk_nope_head_dim": 2,
                  "v_head_dim": 4,
                  "rope_parameters": {
                    "rope_theta": 10000.0,
                    "llama_4_scaling_beta": 0.1,
                    "original_max_position_embeddings": 32
                  },
                  "n_routed_experts": 4,
                  "n_shared_experts": 1,
                  "num_experts_per_tok": 2,
                  "moe_intermediate_size": 4,
                  "first_k_dense_replace": 0
                }
                """.data(using: .utf8)!,
        )
    }
}
