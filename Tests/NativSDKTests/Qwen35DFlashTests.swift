import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class Qwen35DFlashTests: XCTestCase {
    private func textConfig(wide: Bool = false) throws -> Data {
        try JSONSerialization.data(withJSONObject: [
            "model_type": "qwen3_5_text", "hidden_size": wide ? 512 : 64, "num_hidden_layers": 4,
            "intermediate_size": wide ? 1024 : 128, "num_attention_heads": wide ? 4 : 2,
            "num_key_value_heads": wide ? 2 : 1,
            "head_dim": wide ? 128 : 32, "linear_num_value_heads": wide ? 4 : 2,
            "linear_num_key_heads": wide ? 2 : 1,
            "linear_key_head_dim": wide ? 128 : 32, "linear_value_head_dim": wide ? 128 : 32,
            "linear_conv_kernel_dim": 4, "full_attention_interval": 2,
            "partial_rotary_factor": 1.0, "rms_norm_eps": 0.000001,
            "vocab_size": 128, "tie_word_embeddings": false,
            "rope_parameters": [
                "rope_type": "default", "rope_theta": 100000.0,
                "partial_rotary_factor": 1.0, "mrope_section": wide ? [32, 16, 16] : [8, 4, 4],
            ],
        ])
    }
    private func target(vision: Bool, quantized: Bool, wide: Bool = false) throws
        -> any DFlashTarget
    {
        let model: any DFlashTarget
        if vision {
            let text = try JSONSerialization.jsonObject(with: textConfig(wide: wide))
            let config = try JSONDecoder().decode(
                Qwen35Configuration.self,
                from:
                    JSONSerialization.data(withJSONObject: [
                        "model_type": "qwen3_5", "text_config": text,
                        "image_token_id": 120, "video_token_id": 121,
                        "vision_start_token_id": 122, "vision_end_token_id": 123,
                        "vision_config": [
                            "model_type": "qwen3_vl", "depth": 2, "hidden_size": 32,
                            "intermediate_size": 64, "out_hidden_size": 64, "num_heads": 2,
                            "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
                            "num_position_embeddings": 64,
                        ],
                    ]))
            model = withRandomState(MLXRandom.RandomState(seed: 7)) { Qwen35(config) }
        } else {
            let config = try JSONDecoder().decode(
                Qwen35TextConfiguration.self, from: textConfig(wide: wide))
            model = withRandomState(MLXRandom.RandomState(seed: 7)) { Qwen35TextModel(config) }
        }
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        if quantized {
            if let vl = model as? Qwen35 {
                quantize(model: vl.languageModel, groupSize: 32, bits: 4)
            } else {
                quantize(model: model, groupSize: 32, bits: 4)
            }
        }
        try model.prepare()
        model.train(false)
        eval(model)
        return model
    }
    private func draft(window: Int = 4) throws -> DSparkModel {
        let data = try JSONSerialization.data(withJSONObject: [
            "architectures": ["DFlashDraftModel"], "model_type": "qwen3",
            "hidden_size": 64, "num_hidden_layers": 2, "intermediate_size": 128,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 32,
            "rms_norm_eps": 0.000001, "vocab_size": 128,
            "layer_types": ["sliding_attention", "full_attention"], "sliding_window": window,
            "num_target_layers": 4,
            "rope_parameters": ["rope_type": "default", "rope_theta": 100000.0],
            "dflash_config": ["block_size": 16, "mask_token_id": 127, "target_layer_ids": [0, 2]],
        ])
        let config = try DSparkConfiguration(data: data)
        let model = withRandomState(MLXRandom.RandomState(seed: 11)) { DSparkModel(config) }
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        model.prepareContextProjection()
        eval(model)
        return model
    }
    private func drain(_ iterator: any GenerationTokenIterator) -> [Int] {
        var result: [Int] = []
        while let token = iterator.next() { result.append(token) }
        XCTAssertNil(iterator.failure)
        return result
    }
    private func forward(
        _ target: any DFlashTarget, tokens: [Int], cache: [KVCache],
        verify: Bool, state: LMOutput.State? = nil, record: Bool = false
    ) throws -> DFlashTargetOutput {
        try target.draftForward(
            MLXArray(tokens)[.newAxis], cache: cache,
            capture: [0, 2], verify: verify, inputEmbeddings: nil, recordTransaction: record,
            state: state, positionIds: nil)
    }

    func testTemporalKernelHistoryMatchesSingletonStates() throws {
        for width in [16, 32, 128] {
            for kernel in [false, true] {
                let arrays = withRandomState(MLXRandom.RandomState(seed: 17)) {
                    (0 ..< 5).map { index in
                        MLXRandom.normal(
                            index < 2
                                ? [1, 7, 1, width] : (index == 2 ? [1, 7, 2, 16] : [1, 7, 2]))
                    }
                }
                for dtype in [DType.bfloat16, .float16, .float32] {
                    let (q, k, v, a, b) = (
                        arrays[0].asType(dtype), arrays[1].asType(dtype),
                        arrays[2].asType(dtype), arrays[3].asType(dtype), arrays[4].asType(dtype)
                    )
                    let aLog = MLXArray.zeros([2])
                    let bias = MLXArray.zeros([2])
                    let mask = MLXArray([true, true, false, true, true, false, true])[.newAxis]
                    let actual = gatedDeltaUpdateWithHistory(
                        q: q, k: k, v: v, a: a, b: b,
                        aLog: aLog, dtBias: bias, state: nil, mask: mask, useKernel: kernel,
                        positions: [2, 4])
                    var state: MLXArray?
                    var ys: [MLXArray] = []
                    var states: [MLXArray] = []
                    for i in 0 ..< 7 {
                        let (y, next) = gatedDeltaUpdate(
                            q: q[0..., i ..< (i + 1)],
                            k: k[0..., i ..< (i + 1)], v: v[0..., i ..< (i + 1)],
                            a: a[0..., i ..< (i + 1)], b: b[0..., i ..< (i + 1)],
                            aLog: aLog, dtBias: bias, state: state,
                            mask: mask[0..., i ..< (i + 1)], useKernel: kernel)
                        ys.append(y)
                        state = next
                        if [2, 4].contains(i + 1) { states.append(next) }
                    }
                    XCTAssertEqual(
                        actual.output.asArray(Float.self),
                        concatenated(ys, axis: 1).asArray(Float.self))
                    XCTAssertEqual(actual.state.asArray(Float.self), state!.asArray(Float.self))
                    XCTAssertEqual(
                        actual.history!.asArray(Float.self),
                        stacked(states, axis: 1).asArray(Float.self))
                }
            }
        }
    }

    func testQuantizedBlockKernelsPreserveLogitsAndRollback() throws {
        for vision in [false, true] {
            let model = try target(vision: vision, quantized: true, wide: true)
            for length in [2, 8, 16] {
                let actual = try model.newCache(configuration: nil)
                let expected = try model.newCache(configuration: nil)
                let prefix = LMInput.Text(tokens: MLXArray([1, 2, 3])[.newAxis])
                let first = model(prefix, cache: actual, state: nil)
                var state = model(prefix, cache: expected, state: nil).state
                let tokens = Array(4 ..< (4 + length))
                let verified = try forward(
                    model, tokens: tokens, cache: actual, verify: true, state: first.state)
                var logits: [MLXArray] = []
                for token in tokens {
                    let output = model(
                        .init(tokens: MLXArray([[Int32(token)]])), cache: expected, state: state)
                    state = output.state ?? state
                    logits.append(output.logits)
                }
                XCTAssertEqual(
                    verified.logits.asArray(Float.self),
                    concatenated(logits, axis: 1).asArray(Float.self),
                    "vision=\(vision), length=\(length)")
                let keep = length / 2
                try verified.commit?(keep)
                let fresh = try model.newCache(configuration: nil)
                state = model(prefix, cache: fresh, state: nil).state
                for token in tokens.prefix(keep) {
                    state =
                        model(.init(tokens: MLXArray([[Int32(token)]])), cache: fresh, state: state)
                        .state ?? state
                }
                let next = LMInput.Text(tokens: MLXArray([[Int32(25)]]))
                XCTAssertEqual(
                    model(next, cache: actual, state: verified.state).logits.asArray(Float.self),
                    model(next, cache: fresh, state: state).logits.asArray(Float.self))
            }
        }
    }

    func testVerificationCommitsEveryPrefixAndAbortsColdCaches() throws {
        for vision in [false, true] {
            for quantized in [false, true] {
                let model = try target(vision: vision, quantized: quantized)
                for keep in [0, 1, 3, 7] {
                    let actual = try model.newCache(configuration: nil)
                    let expected = try model.newCache(configuration: nil)
                    let prefix = [1, 2, 3]
                    let a = model(
                        .init(tokens: MLXArray(prefix)[.newAxis]), cache: actual, state: nil)
                    var state = model(
                        .init(tokens: MLXArray(prefix)[.newAxis]), cache: expected, state: nil
                    ).state
                    let inputs = [4, 5, 6, 7, 8, 9, 10]
                    let result = try forward(
                        model, tokens: inputs, cache: actual, verify: true, state: a.state)
                    var logits: [MLXArray] = []
                    for token in inputs {
                        let output = model(
                            .init(tokens: MLXArray([[Int32(token)]])), cache: expected, state: state
                        )
                        state = output.state ?? state
                        logits.append(output.logits)
                    }
                    XCTAssertEqual(
                        result.logits.asArray(Float.self),
                        concatenated(logits, axis: 1).asArray(Float.self),
                        "vision=\(vision), quantized=\(quantized)")
                    try result.commit?(keep)
                    let fresh = try model.newCache(configuration: nil)
                    state =
                        model(.init(tokens: MLXArray(prefix)[.newAxis]), cache: fresh, state: nil)
                        .state
                    for token in inputs.prefix(keep) {
                        state =
                            model(
                                .init(tokens: MLXArray([[Int32(token)]])), cache: fresh,
                                state: state
                            ).state ?? state
                    }
                    let resumed = model(
                        .init(tokens: MLXArray([[Int32(11)]])), cache: actual, state: result.state)
                    let cold = model(
                        .init(tokens: MLXArray([[Int32(11)]])), cache: fresh, state: state)
                    XCTAssertEqual(
                        resumed.logits.asArray(Float.self), cold.logits.asArray(Float.self))
                }
                let cache = try model.newCache(configuration: nil)
                let output = try forward(model, tokens: [1, 2, 3], cache: cache, verify: true)
                try output.commit?(0)
                XCTAssertTrue(cache.allSatisfy { $0.state.isEmpty && $0.offset == 0 })
                let resumed = try forward(model, tokens: [4, 5], cache: cache, verify: false)
                let fresh = try forward(
                    model, tokens: [4, 5], cache: model.newCache(configuration: nil), verify: false)
                XCTAssertEqual(resumed.logits.asArray(Float.self), fresh.logits.asArray(Float.self))
            }
        }
    }

    func testDFlashParitySamplingLimitsAndPairedCacheReuse() throws {
        for vision in [false, true] {
            let model = try target(vision: vision, quantized: true)
            let draft = try draft()
            for temperature: Float in [0, 0.7] {
                for count in [1, 4, 15] {
                    for limit in [0, 1, 2, 7, 17] {
                        let options = ModelGenerationOptions(
                            sampling: .init(
                                temperature: temperature,
                                topP: 0.9, seed: 42), output: .init(maxTokens: limit))
                        let prompt = LMInput(tokens: MLXArray([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
                        let baseline = drain(
                            try AutoregressiveIterator(
                                input: prompt, model: model, options: options))
                        let vault = DraftCacheVault()
                        let iterator = try DFlashIterator(
                            input: prompt, target: model, draft: draft,
                            options: options, speculation: .init(draftTokenCount: count),
                            checkpoint: { vault.latest = $0 })
                        XCTAssertEqual(drain(iterator), baseline)
                        let saved = try XCTUnwrap(vault.latest)
                        let warm = try DFlashIterator(
                            input: LMInput(tokens: MLXArray([12, 13])),
                            target: model, draft: draft, options: options,
                            speculation: .init(draftTokenCount: count), reuse: saved)
                        let cold = try DFlashIterator(
                            input: LMInput(tokens: MLXArray(saved.tokens + [12, 13])),
                            target: model, draft: draft, options: options,
                            speculation: .init(draftTokenCount: count))
                        XCTAssertEqual(drain(warm), drain(cold))
                    }
                }
            }
        }
    }

    func testImagePrefillAndPositionAnchorMatchOrdinaryDecode() throws {
        let model = try target(vision: true, quantized: true)
        let draft = try draft()
        for step in [3, 512] {
            for policy: ModelPrefillOptions.Chunking in [.unchunked, .balanced, .remainder] {
                for temperature: Float in [0, 0.7] {
                    let tokens = MLXArray([[Int32(1), 122, 120, 120, 120, 120, 123, 2, 3, 4]])
                    let input = LMInput(
                        text: .init(tokens: tokens),
                        image: .init(
                            pixels: sin(arange(16 * 1536).asType(.float32)).reshaped(16, 1536),
                            frames: [THW(1, 4, 4)]))
                    let options = ModelGenerationOptions(
                        prefill: .init(stepSize: step, chunking: policy),
                        sampling: .init(temperature: temperature, topP: 0.9, seed: 42),
                        output: .init(maxTokens: 17))
                    let expected = drain(
                        try AutoregressiveIterator(input: input, model: model, options: options))
                    for count in [1, 4, 15] {
                        let iterator = try DFlashIterator(
                            input: input, target: model, draft: draft,
                            options: options, speculation: .init(draftTokenCount: count))
                        XCTAssertEqual(drain(iterator), expected)
                    }
                }
            }
        }
    }

    func testSlidingDraftKeepsAbsolutePositionsAndBoundedHistory() throws {
        let model = try target(vision: false, quantized: false)
        let draft = try draft()
        let full = MLXArray.zeros([1, 10, 128], dtype: .bfloat16)
        let caches = draft.newCache()
        let first = try draft.propose(
            seed: MLXArray([Int32(1)]), features: full, count: 7,
            target: model, cache: caches)
        XCTAssertEqual(caches.map(\.offset), [10, 10])
        XCTAssertLessThanOrEqual(caches[0].state[0].dim(2), 3)
        XCTAssertEqual(first.size, 7)
        _ = try draft.propose(
            seed: MLXArray([Int32(2)]),
            features: MLXArray.zeros([1, 9, 128], dtype: .bfloat16), count: 1, target: model,
            cache: caches)
        XCTAssertEqual(caches.map(\.offset), [19, 19])
        // Native rotating caches keep at most one bulk-write's extra rows.
        XCTAssertLessThanOrEqual(caches[0].state[0].dim(2), 5)
        XCTAssertTrue(caches[0].copy() is RotatingKVCache)
        XCTAssertThrowsError(try self.draft(window: 1))
    }

    func testCacheTransactionsRejectStaleIncompleteAndOverlappingHistory() throws {
        let cache = MambaCache(leftPadding: [2])
        let generation = try cache.beginSpeculation(length: 3)
        XCTAssertThrowsError(try cache.beginSpeculation(length: 1))
        XCTAssertThrowsError(try cache.commitSpeculation(retaining: 1, generation: generation))
        cache.abortSpeculation(generation: generation + 1)
        XCTAssertThrowsError(try cache.beginSpeculation(length: 1))
        cache.abortSpeculation(generation: generation)
        XCTAssertNil(cache.speculativeStatePositions)
        XCTAssertEqual(cache.leftPadding?.asArray(Int.self), [2])
        let next = try cache.beginSpeculation(length: 4, prefixOnly: true)
        XCTAssertEqual(cache.speculativeStatePositions, [3])
        cache.recordSpeculativeWindow(MLXArray.ones([1, 7, 2]), width: 3)
        cache.recordSpeculativeRecurrent(
            MLXArray.ones([1, 1, 1, 2, 2]), final: MLXArray.full([1, 1, 2, 2], values: MLXArray(2)))
        XCTAssertThrowsError(try cache.commitSpeculation(retaining: 2, generation: next))
        try cache.commitSpeculation(retaining: 3, generation: next)
        XCTAssertEqual(cache.leftPadding?.asArray(Int.self), [-1])
        XCTAssertNil(cache.speculativeStatePositions)
    }

    func testAbandonedVerificationRestoresAttentionAndRecurrentCaches() throws {
        let model = try target(vision: true, quantized: true)
        let cache = try model.newCache(configuration: nil)
        var abandoned: DFlashTargetOutput? = try forward(
            model, tokens: [1, 2, 3], cache: cache, verify: true)
        eval(try XCTUnwrap(abandoned).logits)
        XCTAssertTrue(cache.contains { !$0.state.isEmpty })
        abandoned = nil
        XCTAssertTrue(cache.allSatisfy { $0.state.isEmpty && $0.offset == 0 })
        let output = try forward(model, tokens: [4, 5], cache: cache, verify: true)
        let fresh = try forward(
            model, tokens: [4, 5], cache: model.newCache(configuration: nil), verify: true)
        XCTAssertEqual(output.logits.asArray(Float.self), fresh.logits.asArray(Float.self))
        try output.commit?(2)
        try fresh.commit?(2)
    }

    func testPublicDraftSessionExtendsAndRegeneratesQwenCaches() async throws {
        for vision in [false, true] {
            let context = ModelContext(
                metadata: .init(), model: try target(vision: vision, quantized: true),
                processor: QwenDraftSessionProcessor(batched: vision),
                tokenizer: QwenDraftSessionTokenizer())
            let loaded = NativDraftModel(
                target: ExecutionTransfer(context),
                draftResources: ExecutionTransfer(.dspark(try draft())), policy: .init())
            let session = ModelSession(
                model: loaded,
                options: .init(
                    sampling: .init(temperature: 0.7, topP: 0.9, seed: 42),
                    output: .init(maxTokens: 17)))
            let first = try await session.respond(to: "first")
            let original = try await first.response()
            let firstMetrics = try await first.metrics()
            XCTAssertEqual(firstMetrics.speculative?.strategy, .dflash)
            XCTAssertEqual(firstMetrics.cache?.kind, .cold)
            let second = try await session.respond(to: "second")
            _ = try await second.response()
            let secondMetrics = try await second.metrics()
            XCTAssertEqual(secondMetrics.cache?.kind, .extend)
            XCTAssertGreaterThan(secondMetrics.cache?.reusedTokenCount ?? 0, 0)
            try await session.replaceTranscript([])
            let repeated = try await session.respond(to: "first")
            let regenerated = try await repeated.response()
            XCTAssertEqual(regenerated, original)
            let repeatedMetrics = try await repeated.metrics()
            XCTAssertEqual(repeatedMetrics.cache?.kind, .exact)
            await session.close()
        }
    }
}

private struct QwenDraftSessionProcessor: UserInputProcessor {
    let batched: Bool
    func prepare(input: UserInput) async throws -> LMInput {
        let messages = DefaultMessageGenerator().generate(from: input)
        let text = messages.compactMap { $0["content"] as? String }.joined()
        let tokens = MLXArray(text.unicodeScalars.map { Int32($0.value) })
        return LMInput(tokens: batched ? tokens[.newAxis] : tokens)
    }
}

/// Preserve token IDs through the session transcript, unlike numeric fixture text.
private struct QwenDraftSessionTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        text.unicodeScalars.map { Int($0.value) }
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        String(String.UnicodeScalarView(tokenIds.compactMap { Unicode.Scalar($0) }))
    }
    func convertTokenToId(_ token: String) -> Int? {
        token.unicodeScalars.first.map { Int($0.value) }
    }
    func convertIdToToken(_ id: Int) -> String? { Unicode.Scalar(id).map(String.init) }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { throw TokenizerError.missingChatTemplate }
}
