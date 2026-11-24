import MLX
import XCTest

@testable import NativSDK

/// Correctness proof for prompt-cache reuse: logits after a trim-rewind or a
/// checkpoint-restore must match a cold full prefill on real (tiny, random-weight)
/// models covering every cache family the reuse layer handles.
final class PromptCacheEquivalenceTests: XCTestCase {
    /// Feeds tokens through the model, returning the logits of the last position.
    private func feed(_ model: any LanguageModel, _ tokens: [Int], cache: [KVCache])
        -> MLXArray
    {
        let logits = model(
            MLXArray(tokens.map(Int32.init), [1, tokens.count]), cache: cache)
        eval(logits)
        return logits[0..., -1, 0...]
    }

    private func assertLogitsClose(
        _ actual: MLXArray, _ expected: MLXArray,
        _ message: String,
        file: StaticString = #filePath, line: UInt = #line,
    ) {
        XCTAssertEqual(actual.shape, expected.shape, message, file: file, line: line)
        XCTAssertTrue(
            allClose(actual, expected, rtol: 1e-4, atol: 1e-5).item(Bool.self),
            message, file: file, line: line,
        )
    }

    /// Cold reference and reuse path use the same [prefix][suffix] chunking, isolating
    /// the checkpoint snapshot/restore as the only difference.
    private func assertCheckpointRestoreMatchesCold(
        model: any LanguageModel,
        prefix: [Int], junk: [Int], suffix: [Int],
        file: StaticString = #filePath, line: UInt = #line,
    ) throws {
        let coldCache = try model.newCache(configuration: nil)
        _ = feed(model, prefix, cache: coldCache)
        let expected = feed(model, suffix, cache: coldCache)

        let liveCache = try model.newCache(configuration: nil)
        _ = feed(model, prefix, cache: liveCache)
        let checkpoint = PromptCacheOps.snapshot(of: liveCache, tokens: prefix)
        _ = feed(model, junk, cache: liveCache)

        let restored = checkpoint.restore()
        let actual = feed(model, suffix, cache: restored)
        assertLogitsClose(
            actual, expected,
            "checkpoint restore + suffix prefill must equal cold prefill",
            file: file, line: line,
        )
    }

    // MARK: - Laguna: KVCacheSimple + RotatingKVCache (sliding window 8)

    func testLagunaTrimWithinWindowMatchesCold() throws {
        let model = try Laguna(Self.lagunaConfig())
        let prefix = [1, 2, 3]
        let junk = [9, 8]
        let suffix = [4, 5]

        let expected = feed(model, prefix + suffix, cache: try model.newCache(configuration: nil))

        let liveCache = try model.newCache(configuration: nil)
        _ = feed(model, prefix + junk, cache: liveCache)
        XCTAssertTrue(PromptCacheOps.trimAll(liveCache, by: junk.count))
        let actual = feed(model, suffix, cache: liveCache)

        assertLogitsClose(
            actual, expected, "trim-rewind + suffix prefill must equal cold prefill")
    }

    func testLagunaCheckpointRestoreMatchesCold() throws {
        let model = try Laguna(Self.lagunaConfig())
        try assertCheckpointRestoreMatchesCold(
            model: model, prefix: [1, 2, 3, 4], junk: [9, 8, 7], suffix: [5, 6])
    }

    func testLagunaWrappedRotatingRestoreMatchesCold() throws {
        let model = try Laguna(Self.lagunaConfig())
        // 12 tokens overflow the sliding window (8): the rotating layer wraps and loses
        // trimmability; checkpoint restore must still reproduce cold logits exactly.
        let prefix = Array(1 ... 12)
        let probeCache = try model.newCache(configuration: nil)
        _ = feed(model, prefix, cache: probeCache)
        XCTAssertFalse(
            PromptCacheOps.trimAll(probeCache, by: 1),
            "a wrapped RotatingKVCache must refuse to trim",
        )

        try assertCheckpointRestoreMatchesCold(
            model: model, prefix: prefix, junk: [20, 21], suffix: [13, 14])
    }

    // MARK: - NemotronOmni: hybrid MambaCache + KVCacheSimple (pattern "M*")

    func testNemotronHybridCheckpointRestoreMatchesCold() throws {
        let model = NemotronOmni(
            .init(
                languageConfiguration: Self.nemotronTextConfig(),
                visionConfiguration: Self.nemotronVisionConfig(),
                projectorHiddenSize: 16,
                vitHiddenSize: 8,
                videoTemporalPatchSize: 2,
                imageContextTokenID: 41,
            ),
        )
        let cache = model.newCache(configuration: nil)
        XCTAssertTrue(cache.contains { $0 is MambaCache })
        XCTAssertTrue(cache.contains { type(of: $0) == KVCacheSimple.self })
        XCTAssertFalse(
            cache.allSatisfy(\.isTrimmable),
            "the Mamba layer must not be trimmable",
        )

        try assertCheckpointRestoreMatchesCold(
            model: model, prefix: [1, 2, 3, 4], junk: [9, 8], suffix: [5, 6])
    }

    // MARK: - Zaya1VL: CacheList(KVCacheSimple, ArraysCache) per layer

    func testZayaCacheListCheckpointRestoreMatchesCold() throws {
        let model = try Zaya1VL(Self.zayaConfig())
        let cache = try model.newCache(configuration: nil)
        XCTAssertTrue(cache.first is CacheList)
        XCTAssertFalse(
            cache.allSatisfy(\.isTrimmable),
            "a CacheList holding an ArraysCache must not be trimmable",
        )

        try assertCheckpointRestoreMatchesCold(
            model: model, prefix: [1, 2, 3, 4], junk: [9, 8], suffix: [5, 6])
    }

    // MARK: - Tiny configurations (cribbed from the per-model test suites)

    private static func lagunaConfig() throws -> LagunaConfiguration {
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

    private static func nemotronTextConfig() -> NemotronHConfiguration {
        NemotronHConfiguration(
            vocabSize: 128,
            hiddenSize: 24,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 4,
            mambaNumHeads: 1,
            mambaHeadDim: 8,
            ssmStateSize: 8,
            convKernel: 3,
            nGroups: 1,
            intermediateSize: 32,
            moeIntermediateSize: 16,
            moeSharedExpertIntermediateSize: 16,
            nRoutedExperts: 2,
            numExpertsPerTok: 1,
            hybridOverridePattern: "M*",
            useConvBias: false,
        )
    }

    private static func nemotronVisionConfig() -> NemotronOmniConfiguration.VisionConfiguration {
        NemotronOmniConfiguration.VisionConfiguration(
            hiddenSize: 8,
            numHiddenLayers: 1,
            numAttentionHeads: 2,
            intermediateSize: 16,
            imageSize: 4,
            patchSize: 2,
            maxResolution: 4,
            downsampleRatio: 0.5,
            dynamicImageMinNumPatches: 4,
            dynamicImageMaxNumPatches: 4,
            videoTargetNumPatches: 4,
            videoTemporalPatchSize: 2,
        )
    }

    private static func zayaConfig() throws -> Zaya1VLConfiguration {
        try JSONDecoder.json5().decode(
            Zaya1VLConfiguration.self,
            from: """
                {
                  "model_type": "zaya1_vl",
                  "attention_bias": false,
                  "cca_time0": 2,
                  "cca_time1": 2,
                  "ffn_hidden_size": 16,
                  "hidden_size": 8,
                  "image_token_id": 99,
                  "max_position_embeddings": 32,
                  "moe_router_topk": 1,
                  "norm_epsilon": 0.00001,
                  "num_attention_heads": 2,
                  "num_experts": 2,
                  "num_hidden_layers": 1,
                  "num_key_value_heads": 1,
                  "num_query_groups": 1,
                  "rope_pct": 0.5,
                  "rotary_base": 10000,
                  "scale_residual_merge": true,
                  "tie_word_embeddings": true,
                  "vision_lora": true,
                  "vision_lora_rank_attn": 2,
                  "vision_lora_rank_mlp": 2,
                  "vocab_size": 128,
                  "zaya_mlp_expansion": 4,
                  "zaya_use_eda": true,
                  "zaya_use_mod": true,
                  "vision_config": {
                    "depth": 1,
                    "hidden_size": 8,
                    "intermediate_size": 12,
                    "in_chans": 3,
                    "model_type": "qwen2_5_vl",
                    "num_heads": 2,
                    "out_hidden_size": 8,
                    "spatial_merge_size": 2,
                    "spatial_patch_size": 14,
                    "temporal_patch_size": 1,
                    "window_size": 28,
                    "fullatt_block_indexes": [0]
                  }
                }
                """.data(using: .utf8)!,
        )
    }
}
