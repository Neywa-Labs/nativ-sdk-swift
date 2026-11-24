import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class Gemma4DFlashTests: XCTestCase {
    private func configuration(shared: Bool = false, ple: Bool = false) -> [String: Any] {
        [
            "model_type": "gemma4_text", "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 4, "num_attention_heads": 4, "num_key_value_heads": 2,
            "num_global_key_value_heads": 1, "head_dim": 16, "global_head_dim": 32,
            "vocab_size": 128, "vocab_size_per_layer_input": 128,
            "rms_norm_eps": 1e-6, "sliding_window": 8,
            "num_kv_shared_layers": shared ? 2 : 0, "hidden_size_per_layer_input": ple ? 8 : 0,
            "attention_k_eq_v": true, "use_double_wide_mlp": false,
            "tie_word_embeddings": false, "final_logit_softcapping": 30,
            "layer_types": [
                "sliding_attention", "full_attention", "sliding_attention", "full_attention",
            ],
            "rope_parameters": [
                "full_attention": [
                    "rope_type": "proportional", "rope_theta": 1e6, "partial_rotary_factor": 0.25,
                ],
                "sliding_attention": ["rope_type": "default", "rope_theta": 10000],
            ],
        ]
    }
    private func target(shared: Bool = false, ple: Bool = false, unified: Bool = false) throws
        -> any DFlashTarget
    {
        let root = configuration(shared: shared, ple: ple)
        let model: any DFlashTarget
        if unified {
            let data = try JSONSerialization.data(withJSONObject: [
                "model_type": "gemma4", "text_config": root,
                "vision_config": [
                    "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 1,
                    "num_attention_heads": 2, "head_dim": 16, "patch_size": 4,
                ],
            ])
            let config = try JSONDecoder().decode(Gemma4Configuration.self, from: data)
            model = withRandomState(MLXRandom.RandomState(seed: 7)) { Gemma4(config) }
        } else {
            let config = try JSONDecoder().decode(
                Gemma4TextModelConfiguration.self,
                from: JSONSerialization.data(withJSONObject: root))
            model = withRandomState(MLXRandom.RandomState(seed: 7)) { Gemma4TextModel(config) }
        }
        eval(model)
        return model
    }
    private func draftConfig(dspark: Bool) -> [String: Any] {
        var root = configuration()
        root["architectures"] = [dspark ? "Gemma4DSparkModel" : "DFlashDraftModel"]
        root["model_type"] = dspark ? "gemma4_text" : "qwen3"
        root["num_hidden_layers"] = 2
        root["layer_types"] = ["full_attention", "sliding_attention"]
        root["num_target_layers"] = 4
        root["target_layer_ids"] = [0, 2]
        root["block_size"] = dspark ? 3 : 4
        root["mask_token_id"] = 4
        root["markov_rank"] = 8
        root["enable_confidence_head"] = true
        if !dspark { root["rope_parameters"] = ["rope_type": "default", "rope_theta": 10000] }
        return root
    }
    private func draft(dspark: Bool) throws -> DSparkModel {
        let config = try DSparkConfiguration(
            data: JSONSerialization.data(withJSONObject: draftConfig(dspark: dspark)))
        let model = withRandomState(MLXRandom.RandomState(seed: 11)) { DSparkModel(config) }
        model.prepareContextProjection()
        eval(model)
        return model
    }
    private func forward(_ target: any DFlashTarget, tokens: [Int], caches: [KVCache], verify: Bool)
        throws -> DFlashTargetOutput
    {
        try target.draftForward(
            MLXArray(tokens)[.newAxis], cache: caches, capture: [0, 2], verify: verify,
            inputEmbeddings: nil, recordTransaction: false, state: nil, positionIds: nil)
    }
    private func drain(_ iterator: any GenerationTokenIterator) -> [Int] {
        var tokens: [Int] = []
        while let token = iterator.next() { tokens.append(token) }
        XCTAssertNil(iterator.failure)
        return tokens
    }
    // Generated from mlx-vlm's Gemma4 DSpark implementation at 06ff63d6.
    // Its fused context helper is disabled because it assumes a v_proj for K=V layers.
    func testPythonDSparkFixture() throws {
        let directory = try XCTUnwrap(
            Bundle.module.url(
                forResource: "gemma4-dspark", withExtension: nil, subdirectory: "Fixtures"))
        let config = try DSparkConfiguration(
            data: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let model = DSparkModel(config)
        let weights = try MLX.loadArrays(
            url: directory.appendingPathComponent("weights.safetensors"))
        try model.update(parameters: .unflattened(model.sanitize(weights: weights)), verify: .all)
        var missing = weights
        missing["lm_head.weight"] = nil
        XCTAssertThrowsError(
            try DSparkModel(config).update(parameters: .unflattened(missing), verify: .all))
        struct Expected: Decodable {
            let tokens: [Int]
            let sampling_logits: [Float]
        }
        let expected = try JSONDecoder().decode(
            Expected.self, from: Data(contentsOf: directory.appendingPathComponent("expected.json"))
        )
        final class RecordingSampler: LogitSampler {
            var logits: [MLXArray] = []
            func sample(logits: MLXArray) -> MLXArray {
                self.logits.append(logits)
                return argMax(logits)
            }
        }
        let sampler = RecordingSampler()
        let caches = model.newCache()
        let features = sin(MLXArray(0 ..< 160).asType(.float32)).reshaped(1, 5, 32) * 0.1
        let tokens = try model.propose(
            seed: MLXArray([Int32(2)]), features: features, count: 3,
            target: target(), cache: caches, sampler: sampler)
        XCTAssertEqual(tokens.asArray(Int.self), expected.tokens)
        let actual = concatenated(sampler.logits, axis: 0)
        XCTAssertLessThan(
            abs(actual - MLXArray(expected.sampling_logits)).max().item(Float.self), 0.0001)
        XCTAssertEqual(caches.map(\.offset), [5, 5])
    }
    func testRotatingSnapshotRestoresEvictedRows() {
        for prefix in [0, 6, 8, 13] {
            for retained in 0 ... 4 {
                let ring = RotatingKVCache(maxSize: 8)
                let fresh = RotatingKVCache(maxSize: 8)
                func append(_ cache: KVCache, _ token: Int) {
                    let row = MLXArray([Float(token)]).reshaped(1, 1, 1, 1)
                    _ = cache.update(keys: row, values: row * 2)
                }
                for token in 0 ..< prefix {
                    append(ring, token)
                    append(fresh, token)
                }
                let snapshot = ring.copy() as! RotatingKVCache
                for token in prefix ..< (prefix + 4) { append(ring, token) }
                eval(ring.state)
                ring.restoreDraftSnapshot(snapshot)
                for token in prefix ..< (prefix + retained) {
                    append(ring, token)
                    append(fresh, token)
                }
                XCTAssertEqual(ring.metaState, fresh.metaState)
                XCTAssertEqual(
                    ring.state.map { $0.asArray(Float.self) },
                    fresh.state.map { $0.asArray(Float.self) })
            }
        }
    }
    func testPublishedDraftLayoutsAndValidation() throws {
        for dspark in [false, true] {
            let root = draftConfig(dspark: dspark)
            let data = try JSONSerialization.data(withJSONObject: root)
            XCTAssertTrue(try isTargetBoundDraftConfiguration(data))
            let config = try DSparkConfiguration(data: data)
            XCTAssertEqual(config.proposalCount, 3)
            XCTAssertEqual(config.finalLogitSoftcapping, 30)
            let model = DSparkModel(config)
            let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            XCTAssertEqual(weights["layers.0.self_attn.v_proj.weight"] == nil, dspark)
            XCTAssertEqual(weights["layers.0.layer_scalar"] != nil, dspark)
            XCTAssertEqual(weights["embed_tokens.weight"] != nil, dspark)
            XCTAssertEqual(weights["lm_head.weight"] != nil, dspark)
            XCTAssertNotNil(weights["layers.1.self_attn.v_proj.weight"])
            try config.validate(target: target())
            for (key, value) in [
                ("mask_token_id", 128), ("num_target_layers", 2), ("markov_rank", 0),
            ] {
                if !dspark && key == "markov_rank" { continue }
                var bad = root
                bad[key] = value
                XCTAssertThrowsError(
                    try DSparkConfiguration(data: JSONSerialization.data(withJSONObject: bad)))
            }
            var nested = root
            let keys = [
                "mask_token_id", "target_layer_ids", "num_target_layers", "markov_rank",
                "enable_confidence_head",
            ]
            nested["dflash_config"] = Dictionary(
                uniqueKeysWithValues: keys.map { ($0, nested.removeValue(forKey: $0)!) })
            XCTAssertEqual(
                try DSparkConfiguration(data: JSONSerialization.data(withJSONObject: nested))
                    .targetLayers, [0, 2])
        }
    }
    func testRollbackAcrossSlidingWindowAndSharedKV() throws {
        for unified in [false, true] {
            for shared in [false, true] {
                let model = try target(shared: shared, ple: true, unified: unified)
                for prefix in [0, 6, 8, 13] {
                    for retained in [0, 1, 3, 4] {
                        let caches = try model.newCache(configuration: nil)
                        let fresh = try model.newCache(configuration: nil)
                        if prefix > 0 {
                            for c in [caches, fresh] {
                                eval(
                                    try forward(
                                        model, tokens: Array(1 ... prefix), caches: c, verify: false
                                    ).logits)
                            }
                        }
                        let output = try forward(
                            model, tokens: [20, 21, 22, 23], caches: caches, verify: true)
                        eval(output.logits, output.features)
                        XCTAssertEqual(output.features.shape, [1, 4, 128])
                        try output.commit?(retained)
                        if retained > 0 {
                            eval(
                                try forward(
                                    model, tokens: Array([20, 21, 22, 23].prefix(retained)),
                                    caches: fresh, verify: false
                                ).logits)
                        }
                        XCTAssertEqual(caches.map(\.offset), fresh.map(\.offset))
                        let a = try forward(model, tokens: [24], caches: caches, verify: false)
                            .logits
                        let b = try forward(model, tokens: [24], caches: fresh, verify: false)
                            .logits
                        XCTAssertLessThan(
                            abs(a - b).max().item(Float.self), 0.002,
                            "unified=\(unified), shared=\(shared), prefix=\(prefix), retained=\(retained)"
                        )
                    }
                }
            }
        }
    }
    func testPromptCheckpointBeforeRewrittenHeader() throws {
        let model = try target(shared: true, ple: true, unified: true)
        let draft = try draft(dspark: true)
        let options = ModelGenerationOptions(
            prefill: .init(stepSize: 4), sampling: .init(temperature: 0),
            output: .init(maxTokens: 2))
        var checkpoints: [DraftCacheCheckpoint] = []
        let iterator = try DFlashIterator(
            input: LMInput(tokens: MLXArray(Array(1 ... 15))), target: model, draft: draft,
            options: options, speculation: .init(), checkpoint: { checkpoints.append($0) },
            checkpointSuffixCount: 4)
        _ = drain(iterator)
        let prefix = try XCTUnwrap(checkpoints.first)
        XCTAssertEqual(prefix.tokens, Array(1 ... 11))
        XCTAssertTrue(prefix.targetCaches.allSatisfy { $0.offset == 11 })
        let warm = try DFlashIterator(
            input: LMInput(tokens: MLXArray([22, 23, 24])),
            target: model, draft: draft, options: options, speculation: .init(), reuse: prefix)
        let cold = try DFlashIterator(
            input: LMInput(tokens: MLXArray(Array(1 ... 11) + [22, 23, 24])),
            target: model, draft: draft, options: options, speculation: .init())
        XCTAssertEqual(drain(warm), drain(cold))
    }
    func testAllAcceptedRoundsAndOutputLimits() throws {
        let model = try target(shared: true, unified: true)
        model.update(parameters: model.parameters().mapValues { MLXArray.zeros(like: $0) })
        for dspark in [false, true] {
            let draft = try draft(dspark: dspark)
            draft.update(parameters: draft.parameters().mapValues { MLXArray.zeros(like: $0) })
            draft.prepareContextProjection()
            for limit in [0, 1, 2, 7, 17] {
                let options = ModelGenerationOptions(
                    sampling: .init(temperature: 0), output: .init(maxTokens: limit))
                let input = LMInput(tokens: MLXArray([1, 2, 3, 4, 5, 6, 7]))
                let expected = drain(
                    try AutoregressiveIterator(input: input, model: model, options: options))
                let iterator = try DFlashIterator(
                    input: input, target: model, draft: draft,
                    options: options, speculation: .init(draftTokenCount: 3))
                XCTAssertEqual(drain(iterator), expected)
                XCTAssertEqual(
                    iterator.statistics.acceptedTokenCount, iterator.statistics.proposedTokenCount)
            }
        }
    }
    func testAbandonedVerificationRestoresCaches() throws {
        let model = try target(shared: true, ple: true, unified: true)
        let caches = try model.newCache(configuration: nil)
        eval(try forward(model, tokens: Array(1 ... 13), caches: caches, verify: false).logits)
        let states = caches.map { $0.copy() }
        var output: DFlashTargetOutput? = try forward(
            model, tokens: [20, 21, 22, 23], caches: caches, verify: true)
        eval(output!.logits)
        XCTAssertThrowsError(try output!.commit?(5))
        output = nil
        XCTAssertEqual(caches.map(\.offset), states.map(\.offset))
        for (actual, expected) in zip(caches, states) {
            XCTAssertEqual(
                actual.state.map { $0.asArray(Float.self) },
                expected.state.map { $0.asArray(Float.self) })
        }
    }
    func testBothDraftsSamplingAndSessionReuse() throws {
        for unified in [false, true] {
            let model = try target(shared: true, ple: true, unified: unified)
            for dspark in [false, true] {
                let draft = try draft(dspark: dspark)
                for temperature: Float in [0, 0.7] {
                    let options = ModelGenerationOptions(
                        prefill: .init(stepSize: 4),
                        sampling: .init(temperature: temperature, topP: 0.9, seed: 42),
                        output: .init(maxTokens: 15))
                    let vault = DraftCacheVault()
                    let input = LMInput(tokens: MLXArray([1, 2, 3, 4, 5, 6, 7]))
                    let iterator = try DFlashIterator(
                        input: input, target: model, draft: draft, options: options,
                        speculation: .init(draftTokenCount: 3), checkpoint: { vault.latest = $0 })
                    let result = drain(iterator)
                    XCTAssertEqual(result.count, 15)
                    XCTAssertGreaterThan(iterator.statistics.proposedTokenCount, 0)
                    let saved = try XCTUnwrap(vault.latest)
                    let warm = try DFlashIterator(
                        input: LMInput(tokens: MLXArray([8, 9])), target: model, draft: draft,
                        options: options, speculation: .init(), reuse: saved)
                    let cold = try DFlashIterator(
                        input: LMInput(tokens: MLXArray(saved.tokens + [8, 9])), target: model,
                        draft: draft,
                        options: options, speculation: .init())
                    XCTAssertEqual(drain(warm), drain(cold))
                }
            }
        }
    }
}
