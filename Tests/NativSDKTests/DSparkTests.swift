import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class DSparkTests: XCTestCase {
    private func target(vocabularySize: Int = 128) throws -> LFM2Model {
        let json =
            #"{"model_type":"lfm2","hidden_size":64,"num_hidden_layers":4,"num_attention_heads":4,"num_key_value_heads":2,"vocab_size":128,"norm_eps":0.00001,"block_ff_dim":128,"block_auto_adjust_ff_dim":false,"conv_L_cache":3,"full_attn_idxs":[1,3]}"#
        let config = try JSONDecoder().decode(
            LFM2Configuration.self,
            from: Data(
                json.replacingOccurrences(
                    of: "\"vocab_size\":128", with: "\"vocab_size\":\(vocabularySize)"
                ).utf8))
        let model = withRandomState(MLXRandom.RandomState(seed: 7)) { LFM2Model(config) }
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        eval(model.parameters())
        return model
    }

    private func configuration(_ changes: [String: Any] = [:]) throws -> DSparkConfiguration {
        var root: [String: Any] = [
            "architectures": ["Lfm2DSparkDraftModel"], "model_type": "qwen3",
            "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
            "vocab_size": 128, "block_size": 9, "markov_rank": 8,
            "rope_is_neox_style": false, "enable_confidence_head": true,
            "dflash_config": [
                "mask_token_id": 127, "target_layer_ids": [0, 2], "num_target_layers": 4,
            ],
        ]
        root.merge(changes) { _, new in new }
        return try DSparkConfiguration(data: JSONSerialization.data(withJSONObject: root))
    }

    private func draft() throws -> DSparkModel {
        let config = try configuration()
        let model = withRandomState(MLXRandom.RandomState(seed: 11)) { DSparkModel(config) }
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        model.prepareContextProjection()
        return model
    }

    private func visionTarget() throws -> LFM2VL {
        let text = try JSONSerialization.jsonObject(
            with: JSONEncoder().encode(target().configuration))
        let root: [String: Any] = [
            "model_type": "lfm2_vl", "text_config": text,
            "vision_config": [
                "model_type": "siglip2", "hidden_size": 16, "intermediate_size": 32,
                "num_hidden_layers": 1, "num_attention_heads": 2, "patch_size": 2, "num_patches": 4,
            ],
            "downsample_factor": 2, "image_token_id": 127, "projector_hidden_size": 32,
            "projector_use_layernorm": false,
        ]
        let config = try JSONDecoder().decode(
            LFM2VLConfiguration.self,
            from: JSONSerialization.data(withJSONObject: root))
        let model = withRandomState(MLXRandom.RandomState(seed: 7)) { LFM2VL(config) }
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        eval(model.parameters())
        return model
    }

    private func tokens(_ iterator: DFlashIterator) -> [Int] {
        var result: [Int] = []
        while let token = iterator.next() { result.append(token) }
        XCTAssertNil(iterator.failure)
        return result
    }

    func testDSparkPairedPrefixReuseMatchesColdGeneration() throws {
        let model = try target()
        let drafter = try draft()
        for temperature: Float in [0, 0.7] {
            for limit in [0, 1, 2, 9, 18] {
                let options = ModelGenerationOptions(
                    prefill: .init(stepSize: 1, chunking: .balanced),
                    sampling: .init(temperature: temperature, seed: 42),
                    output: .init(maxTokens: limit, format: .tokens))
                let vault = DraftCacheVault()
                let prompt = [1, 3, 5, 7]
                let first = try DFlashIterator(
                    input: LMInput(tokens: MLXArray(prompt)),
                    target: model, draft: drafter, options: options, speculation: .init(),
                    checkpoint: { vault.latest = $0 })
                let generated = tokens(first)
                XCTAssertNil(first.failure)
                let nextPrompt = prompt + generated + [11, 13]
                let checkpoint = try XCTUnwrap(vault.best(for: nextPrompt))
                XCTAssertEqual(checkpoint.tokens, Array(nextPrompt.prefix(checkpoint.tokens.count)))
                XCTAssertEqual(
                    checkpoint.targetCaches.filter { cacheCapabilities($0).contains(.attention) }
                        .map(\.offset),
                    Array(repeating: checkpoint.tokens.count, count: 2))
                XCTAssertEqual(
                    checkpoint.draftCaches[0].offset + (checkpoint.features?.dim(1) ?? 0),
                    checkpoint.tokens.count)
                let nextOptions = ModelGenerationOptions(
                    prefill: options.prefill,
                    sampling: options.sampling, output: .init(maxTokens: 12, format: .tokens))
                let reused = try DFlashIterator(
                    input: LMInput(
                        tokens: MLXArray(Array(nextPrompt.dropFirst(checkpoint.tokens.count)))),
                    target: model, draft: drafter, options: nextOptions, speculation: .init(),
                    reuse: checkpoint)
                let cold = try DFlashIterator(
                    input: LMInput(tokens: MLXArray(nextPrompt)),
                    target: model, draft: drafter, options: nextOptions, speculation: .init())
                XCTAssertEqual(
                    tokens(reused), tokens(cold), "temperature=\(temperature), limit=\(limit)")
                XCTAssertNil(reused.failure)
                // Restoration copies caches: reusing the checkpoint repeatedly is safe.
                let repeated = try DFlashIterator(
                    input: LMInput(
                        tokens: MLXArray(Array(nextPrompt.dropFirst(checkpoint.tokens.count)))),
                    target: model, draft: drafter, options: nextOptions, speculation: .init(),
                    reuse: checkpoint)
                XCTAssertEqual(
                    tokens(repeated),
                    tokens(
                        try DFlashIterator(
                            input: LMInput(tokens: MLXArray(nextPrompt)), target: model,
                            draft: drafter,
                            options: nextOptions, speculation: .init())))
            }
        }
    }

    func testDSparkReusesPrefixWhenTemplateDropsThinkingMarker() throws {
        let model = try target()
        let drafter = try draft()
        let vault = DraftCacheVault()
        let options = ModelGenerationOptions(
            prefill: .init(stepSize: 1, chunking: .balanced),
            sampling: .init(temperature: 0.7, seed: 42),
            output: .init(maxTokens: 12, format: .tokens))
        let prompt = [1, 3, 5, 127]  // Final token stands in for <think>.
        let iterator = try DFlashIterator(
            input: LMInput(tokens: MLXArray(prompt)),
            target: model, draft: drafter, options: options, speculation: .init(),
            checkpoint: {
                vault.latest = $0
                vault.retain($0, maxCount: 4, protecting: Array(prompt.dropLast()))
            })
        _ = tokens(iterator)
        let nextPrompt = [1, 3, 5, 11, 13, 17]
        let checkpoint = try XCTUnwrap(vault.best(for: nextPrompt))
        XCTAssertEqual(checkpoint.tokens, [1, 3, 5])
        let reused = try DFlashIterator(
            input: LMInput(tokens: MLXArray(Array(nextPrompt.dropFirst(3)))), target: model,
            draft: drafter, options: options, speculation: .init(), reuse: checkpoint)
        let cold = try DFlashIterator(
            input: LMInput(tokens: MLXArray(nextPrompt)),
            target: model, draft: drafter, options: options, speculation: .init())
        XCTAssertEqual(tokens(reused), tokens(cold))
    }

    func testRecordingOrdinaryPrefillPreservesNumericsAndRestoresPrefix() throws {
        let model = try target()
        let input = MLXArray([[1, 3, 5, 7]])
        let cache = try model.newCache(configuration: nil)
        let expected = try model.dflashForward(
            input, cache: model.newCache(configuration: nil),
            capture: [0, 2], verify: false)
        let recorded = try model.dflashForward(
            input, cache: cache,
            capture: [0, 2], verify: false, recordTransaction: true)
        XCTAssertEqual(recorded.logits.asArray(Float.self), expected.logits.asArray(Float.self))
        XCTAssertEqual(recorded.features.asArray(Float.self), expected.features.asArray(Float.self))
        recorded.transaction!.commit(retaining: 3)
        let prefixCache = try model.newCache(configuration: nil)
        _ = try model.dflashForward(
            input[0..., ..<3], cache: prefixCache, capture: [0, 2], verify: false)
        let next = MLXArray([[11]])
        let actual = model(next, cache: cache)
        let reference = model(next, cache: prefixCache)
        XCTAssertEqual(actual.asArray(Float.self), reference.asArray(Float.self))
    }

    func testDSparkStopInsideVerifiedBlockReusesOnlyDeliveredPrefix() throws {
        let model = try target()
        let drafter = try draft()
        let vault = DraftCacheVault()
        // Force full acceptance so stopping after five outputs leaves verified
        // inputs ahead of the delivered prefix, including recurrent state.
        model.update(parameters: model.parameters().mapValues { MLXArray.zeros(like: $0) })
        drafter.update(parameters: drafter.parameters().mapValues { MLXArray.zeros(like: $0) })
        drafter.prepareContextProjection()
        let options = ModelGenerationOptions(
            prefill: .init(stepSize: 1, chunking: .balanced),
            sampling: .init(temperature: 0), output: .init(maxTokens: 30, format: .tokens))
        let prompt = [1, 3, 5, 7]
        let iterator = try DFlashIterator(
            input: LMInput(tokens: MLXArray(prompt)),
            target: model, draft: drafter, options: options, speculation: .init(),
            checkpoint: { vault.latest = $0 })
        var delivered: [Int] = []
        for _ in 0 ..< 5 { delivered.append(try XCTUnwrap(iterator.next())) }
        XCTAssertEqual(iterator.statistics.acceptedTokenCount, 7)
        let nextPrompt = prompt + delivered + [11, 13]
        let checkpoint = try XCTUnwrap(vault.best(for: nextPrompt))
        XCTAssertLessThan(checkpoint.tokens.count, prompt.count + delivered.count)
        let reused = try DFlashIterator(
            input: LMInput(tokens: MLXArray(Array(nextPrompt.dropFirst(checkpoint.tokens.count)))),
            target: model, draft: drafter, options: options, speculation: .init(), reuse: checkpoint
        )
        let cold = try DFlashIterator(
            input: LMInput(tokens: MLXArray(nextPrompt)),
            target: model, draft: drafter, options: options, speculation: .init())
        XCTAssertEqual(tokens(reused), tokens(cold))
        XCTAssertNil(reused.failure)
    }

    func testVisionIteratorMatchesGreedyWithImagesAndPrefillPolicies() throws {
        let model = try visionTarget()
        let drafter = try draft()
        for imageCount in [0, 1, 2] {
            let tokens = MLXArray([1] + Array(repeating: 127, count: imageCount) + [2, 3, 4, 5])
            let input = LMInput(
                text: .init(
                    tokens: imageCount == 0 ? tokens : tokens[.newAxis],
                    mask: imageCount == 0 ? nil : MLXArray.ones([1, tokens.size], dtype: .int8)),
                image: imageCount == 0
                    ? nil
                    : .init(
                        pixels: cos(arange(imageCount * 48).asType(.float32)).reshaped(
                            imageCount, 4, 12),
                        frames: Array(repeating: THW(1, 2, 2), count: imageCount)))
            for policy: ModelPrefillOptions.Chunking in [.unchunked, .remainder, .balanced] {
                for limit in [0, 1, 2, 7, 12] {
                    let options = ModelGenerationOptions(
                        prefill: .init(stepSize: 3, chunking: policy),
                        sampling: .init(temperature: 0),
                        output: .init(maxTokens: limit, format: .tokens))
                    let expected = Array(
                        try AutoregressiveIterator(input: input, model: model, options: options))
                    for count in [1, 7, 9] {
                        let iterator = try DFlashIterator(
                            input: input, target: model, draft: drafter,
                            options: options, speculation: .init(draftTokenCount: count))
                        var actual: [Int] = []
                        while let token = iterator.next() { actual.append(token) }
                        XCTAssertNil(iterator.failure)
                        XCTAssertEqual(
                            actual, expected,
                            "images=\(imageCount), policy=\(policy), limit=\(limit), proposals=\(count)"
                        )
                    }
                }
            }
        }
    }

    func testVisionDraftRejectsPaddingAndMismatchedImageFeatures() throws {
        let model = try visionTarget()
        let drafter = try draft()
        let options = ModelGenerationOptions(sampling: .init(temperature: 0))
        let padded = LMInput(tokens: MLXArray([[1, 2]]), mask: MLXArray([[1, 0]]))
        XCTAssertThrowsError(
            try DFlashIterator(
                input: padded, target: model, draft: drafter,
                options: options, speculation: .init()))
        let image = LMInput(
            text: .init(tokens: MLXArray([[1, 2]])),
            image: .init(
                pixels: MLXArray.zeros([1, 4, 12]), frames: [THW(1, 2, 2)]))
        XCTAssertThrowsError(
            try DFlashIterator(
                input: image, target: model, draft: drafter,
                options: options, speculation: .init()))
        XCTAssertThrowsError(
            try DFlashIterator(
                input: image, target: target(), draft: drafter,
                options: options, speculation: .init()))
    }

    func testExactBlockProjectionMatchesSingletonBF16() throws {
        for (k, o) in [(64, 128), (128, 256), (2048, 6144), (6144, 2048)] {
            let w = sin(arange(k * o).asType(.float32)).reshaped(o, k).asType(.bfloat16)
            for length in [2, 4, 10] {
                let x = cos(arange(length * k).asType(.float32)).reshaped(1, length, k).asType(
                    .bfloat16)
                let actual = try XCTUnwrap(exactSpeculativeProjection(weight: w, input: x))
                let expected = concatenated(
                    (0 ..< length).map {
                        matmul(x[0..., $0 ..< ($0 + 1), 0...], w.T)
                    }, axis: 1)
                XCTAssertEqual(
                    actual.asArray(Float.self), expected.asArray(Float.self),
                    "K=\(k), O=\(o), L=\(length)")
            }
        }
    }

    func testVerificationAndEveryCommitPrefixMatchSingletonForwards() throws {
        let model = try target()
        for prefill in [[], [1, 2, 3, 4]] as [[Int]] {
            for count in 0 ... 5 {
                let actualCache = try model.newCache(configuration: nil)
                let expectedCache = try model.newCache(configuration: nil)
                if !prefill.isEmpty {
                    eval(
                        model(MLXArray(prefill)[.newAxis], cache: actualCache),
                        model(MLXArray(prefill)[.newAxis], cache: expectedCache))
                }
                let block = [5, 6, 7, 8, 9]
                let verified = try model.dflashForward(
                    MLXArray(block)[.newAxis],
                    cache: actualCache, capture: [0, 2], verify: true)
                let fullCache = expectedCache.map { $0.copy() }
                let expectedLogits = concatenated(
                    block.map {
                        model(MLXArray([$0])[.newAxis], cache: fullCache)
                    }, axis: 1)
                XCTAssertEqual(
                    verified.logits.asArray(Float.self), expectedLogits.asArray(Float.self))
                XCTAssertEqual(verified.features.shape, [1, 5, 128])
                verified.transaction!.commit(retaining: count)
                for token in block.prefix(count) {
                    eval(model(MLXArray([token])[.newAxis], cache: expectedCache))
                }
                for (a, b) in zip(actualCache, expectedCache) {
                    XCTAssertEqual(a.offset, b.offset)
                    XCTAssertEqual(a.state.count, b.state.count)
                    for (x, y) in zip(a.state, b.state) {
                        XCTAssertEqual(x.asArray(Float.self), y.asArray(Float.self))
                    }
                }
                let next = MLXArray([10])[.newAxis]
                XCTAssertEqual(
                    model(next, cache: actualCache).asArray(Float.self),
                    model(next, cache: expectedCache).asArray(Float.self))
            }
        }
    }

    func testBlockIteratorMatchesGreedyForLimitsCountsAndPrefillPolicies() throws {
        let model = try target()
        let drafter = try draft()
        for policy: ModelPrefillOptions.Chunking in [.unchunked, .remainder, .balanced] {
            for limit in [0, 1, 2, 4, 12] {
                let options = ModelGenerationOptions(
                    prefill: .init(stepSize: 3, chunking: policy),
                    sampling: .init(temperature: 0),
                    output: .init(maxTokens: limit, format: .tokens))
                let input = LMInput(tokens: MLXArray([1, 2, 3, 4, 5, 6, 7]))
                let expected = Array(
                    try AutoregressiveIterator(input: input, model: model, options: options))
                for count in [1, 4, 7, 9] {
                    let iterator = try DFlashIterator(
                        input: input, target: model, draft: drafter,
                        options: options, speculation: .init(draftTokenCount: count))
                    var actual: [Int] = []
                    while let token = iterator.next() { actual.append(token) }
                    XCTAssertNil(iterator.failure)
                    XCTAssertEqual(
                        actual, expected, "policy=\(policy), limit=\(limit), count=\(count)")
                    XCTAssertEqual(iterator.statistics.strategy, .dspark)
                }
            }
        }
    }

    private func assertSampledParity(
        input: LMInput, model: any LanguageModel, drafter: DSparkModel,
        sampling: ModelSamplingOptions, policy: ModelPrefillOptions.Chunking,
        limit: Int = 16, count: Int
    ) throws {
        let options = ModelGenerationOptions(
            prefill: .init(stepSize: 3, chunking: policy),
            sampling: sampling, output: .init(maxTokens: limit, format: .tokens))
        let ordinary = try AutoregressiveIterator(input: input, model: model, options: options)
        let expected = Array(ordinary)
        XCTAssertNil(ordinary.failure)
        let iterator = try DFlashIterator(
            input: input, target: model, draft: drafter,
            options: options, speculation: .init(draftTokenCount: count))
        var actual: [Int] = []
        while let token = iterator.next() { actual.append(token) }
        XCTAssertNil(iterator.failure)
        XCTAssertEqual(
            actual, expected,
            "temperature=\(sampling.temperature), seed=\(String(describing: sampling.seed)), policy=\(policy), count=\(count), limit=\(limit)"
        )
    }

    func testSampledTextMatchesOrdinaryAcrossSeedsFiltersAndBlockWidths() throws {
        let model = try target()
        let drafter = try draft()
        let input = LMInput(tokens: MLXArray([1, 2, 3, 4, 5, 6, 7]))
        for seed: UInt64 in [0, 42, UInt64.max] {
            for sampling in [
                ModelSamplingOptions(temperature: 0.2, seed: seed),
                .init(temperature: 0.7, topP: 0.9, seed: seed),
                .init(temperature: 1, topK: 20, seed: seed),
                .init(temperature: 0.7, topP: 0.95, topK: 20, minP: 0.05, seed: seed),
            ] {
                for policy: ModelPrefillOptions.Chunking in [.unchunked, .remainder, .balanced] {
                    for count in [1, 4, 7, 9] {
                        try assertSampledParity(
                            input: input, model: model, drafter: drafter,
                            sampling: sampling, policy: policy, count: count)
                    }
                }
            }
        }
        for limit in [0, 1, 2, 7] {
            for count in [1, 4, 7, 9] {
                try assertSampledParity(
                    input: input, model: model, drafter: drafter,
                    sampling: .init(temperature: 0.7, seed: 42), policy: .remainder, limit: limit,
                    count: count)
            }
        }
    }

    func testSampledVisionMatchesOrdinaryAndRequestsReset() throws {
        let model = try visionTarget()
        let drafter = try draft()
        for imageCount in [0, 1, 2] {
            let tokens = MLXArray([1] + Array(repeating: 127, count: imageCount) + [2, 3, 4, 5])
            let input = LMInput(
                text: .init(
                    tokens: imageCount == 0 ? tokens : tokens[.newAxis],
                    mask: imageCount == 0 ? nil : MLXArray.ones([1, tokens.size], dtype: .int8)),
                image: imageCount == 0
                    ? nil
                    : .init(
                        pixels: cos(arange(imageCount * 48).asType(.float32)).reshaped(
                            imageCount, 4, 12),
                        frames: Array(repeating: THW(1, 2, 2), count: imageCount)))
            for _ in 0 ..< 2 {
                for policy: ModelPrefillOptions.Chunking in [.unchunked, .remainder, .balanced] {
                    for count in [1, 4, 7, 9] {
                        try assertSampledParity(
                            input: input, model: model, drafter: drafter,
                            sampling: .init(temperature: 0.7, topP: 0.9, topK: 20, seed: 42),
                            policy: policy, count: count)
                    }
                }
            }
        }
    }

    func testSampledCancellationLeavesNextRequestIndependent() async throws {
        let resources = ExecutionTransfer((try target(), try draft()))
        let task = Task.detached {
            let (model, drafter) = resources.take()
            let iterator = try DFlashIterator(
                input: LMInput(tokens: MLXArray([1, 2, 3, 4])),
                target: model, draft: drafter,
                options: .init(
                    sampling: .init(temperature: 0.7, seed: 42), output: .init(maxTokens: 16)),
                speculation: .init())
            let first = iterator.next()
            withUnsafeCurrentTask { $0?.cancel() }
            let next = iterator.next()
            let cancelled = first != nil && next == nil && iterator.failure is CancellationError
            return (cancelled, ExecutionTransfer((model, drafter)))
        }
        let (cancelled, returned) = try await task.value
        XCTAssertTrue(cancelled)
        let (model, drafter) = returned.take()
        try assertSampledParity(
            input: LMInput(tokens: MLXArray([1, 2, 3, 4])),
            model: model, drafter: drafter, sampling: .init(temperature: 0.7, seed: 42),
            policy: .remainder, count: 7)
    }

    func testConfigurationAndSharedCheckpointStructure() throws {
        let config = try configuration()
        XCTAssertEqual(config.proposalCount, 9)
        XCTAssertEqual(config.defaultProposalCount, 7)
        XCTAssertTrue(config.traditionalRoPE)
        XCTAssertTrue(SpeculativeGenerationOptions().usesModelDefault)
        var options = SpeculativeGenerationOptions()
        options.draftTokenCount = 4
        XCTAssertFalse(options.usesModelDefault)
        let model = try draft()
        let keys = Set(model.parameters().flattened().map(\.0))
        XCTAssertEqual(keys.count, 29)
        XCTAssertTrue(keys.contains("confidence_head.proj.weight"))
        XCTAssertTrue(keys.contains("layers.0.self_attn.q_norm.weight"))
        XCTAssertFalse(keys.contains { $0.contains("embed") || $0.contains("contextProjection") })
        try config.validate(target: target())
        for invalid: [String: Any] in [
            ["block_size": 0], ["is_causal": true], ["rope_scaling": ["type": "linear"]],
            ["markov_head_type": "other"], ["attention_bias": true], ["quantization": ["bits": 4]],
            ["block_size": 1.5], ["hidden_size": true], ["is_causal": "false"],
            ["confidence_head_with_markov": false], ["attention_sink_bias": true],
            ["draft_window_size": 32], ["block_size_policy": "adaptive"], ["runtime_block_size": 1],
        ] { XCTAssertThrowsError(try configuration(invalid)) }
        XCTAssertEqual(try configuration(["runtime_block_size": 5]).defaultProposalCount, 4)
        XCTAssertNoThrow(
            try DFlashIterator.validate(
                options: .init(sampling: .init(temperature: 0.7)), speculation: .init(),
                target: target(), draft: model))
        XCTAssertThrowsError(
            try DFlashIterator.validate(
                options: .init(sampling: .init(temperature: 0)),
                speculation: .init(draftTokenCount: 10), target: target(), draft: model))
        let data = Data(#"{"model_type":"qwen3","dflash_config":{}}"#.utf8)
        XCTAssertTrue(try isTargetBoundDraftConfiguration(data))
        XCTAssertThrowsError(try DSparkConfiguration(data: data))
    }

    func testMarkovTransitionsAndOnlyContextKVPersists() throws {
        let model = try draft()
        let target = try target()
        var weights = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map {
                ($0.0, MLXArray.zeros($0.1.shape, dtype: .bfloat16))
            })
        let embedding = MLXArray.zeros([128, 8], dtype: .bfloat16)
        let projection = MLXArray.zeros([128, 8], dtype: .bfloat16)
        for token in 1 ... 6 {
            embedding[token, token] = MLXArray(1).asType(.bfloat16)
            projection[token + 1, token] = MLXArray(1).asType(.bfloat16)
        }
        weights["markov_head.markov_w1.weight"] = embedding
        weights["markov_head.markov_w2.weight"] = projection
        try model.update(parameters: .unflattened(weights), verify: .all)
        model.prepareContextProjection()
        let cache = model.newCache()
        let first = try model.propose(
            seed: MLXArray([Int32(1)]),
            features: MLXArray.zeros([1, 3, 128], dtype: .bfloat16), count: 4, target: target,
            cache: cache)
        XCTAssertEqual(first.asArray(Int.self), [2, 3, 4, 5])
        XCTAssertEqual(cache.map(\.offset), [3, 3])
        let second = try model.propose(
            seed: MLXArray([Int32(5)]),
            features: MLXArray.zeros([1, 2, 128], dtype: .bfloat16), count: 1, target: target,
            cache: cache)
        XCTAssertEqual(second.asArray(Int.self), [6])
        XCTAssertEqual(cache.map(\.offset), [5, 5])
    }

    func testCheckpointRequiresEveryDraftWeightAndRejectsExtras() throws {
        let model = try draft()
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var missing = weights
        missing.removeValue(forKey: "confidence_head.proj.bias")
        XCTAssertThrowsError(
            try DSparkModel(configuration()).update(parameters: .unflattened(missing), verify: .all)
        )
        var extra = weights
        extra["lm_head.weight"] = MLXArray.zeros([128, 64], dtype: .bfloat16)
        XCTAssertThrowsError(
            try DSparkModel(configuration()).update(parameters: .unflattened(extra), verify: .all))
        try DSparkModel(configuration()).update(parameters: .unflattened(weights), verify: .all)
    }

    func testReadoutFallbackPreservesSingletonNumerics() throws {
        let model = try target(vocabularySize: 129)
        let actual = try model.newCache(configuration: nil)
        let expected = try model.newCache(configuration: nil)
        let tokens = [1, 2, 3, 4]
        let block = try model.dflashForward(
            MLXArray(tokens)[.newAxis], cache: actual,
            capture: [0, 2], verify: true)
        let single = concatenated(
            tokens.map { model(MLXArray([$0])[.newAxis], cache: expected) }, axis: 1)
        XCTAssertEqual(block.logits.asArray(Float.self), single.asArray(Float.self))
    }
}
