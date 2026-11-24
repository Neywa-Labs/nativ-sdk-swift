import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class DraftExpansionTests: XCTestCase {
    private func qwenConfiguration() throws -> Data {
        Data(
            #"{"model_type":"qwen3","hidden_size":64,"num_hidden_layers":4,"intermediate_size":128,"num_attention_heads":8,"num_key_value_heads":2,"head_dim":16,"rms_norm_eps":0.000001,"vocab_size":128,"tie_word_embeddings":false}"#
                .utf8)
    }
    private func qwen(quantized: Bool = false) throws -> Qwen3Model {
        let config = try JSONDecoder().decode(Qwen3Configuration.self, from: qwenConfiguration())
        let model = withRandomState(MLXRandom.RandomState(seed: 7)) { Qwen3Model(config) }
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        if quantized { quantize(model: model, groupSize: 32, bits: 4) }
        eval(model)
        return model
    }
    private func blockConfiguration(dspark: Bool) throws -> Data {
        var root: [String: Any] = [
            "architectures": [dspark ? "Qwen3DSparkModel" : "DFlashDraftModel"],
            "model_type": "qwen3", "hidden_size": 64, "num_hidden_layers": 2,
            "intermediate_size": 128, "num_attention_heads": 8, "num_key_value_heads": 2,
            "head_dim": 16, "rms_norm_eps": 0.000001, "vocab_size": 128,
            "block_size": 7, "markov_rank": 32, "mask_token_id": 127,
            "target_layer_ids": [0, 2], "num_target_layers": 4,
            "rope_parameters": ["rope_type": "default", "rope_theta": 1_000_000],
            "sliding_window": NSNull(), "rope_scaling": NSNull(), "tie_word_embeddings": false,
        ]
        if !dspark { root["dflash_config"] = ["mask_token_id": 127, "target_layer_ids": [0, 2]] }
        return try JSONSerialization.data(withJSONObject: root)
    }
    private func blockDraft(dspark: Bool) throws -> DSparkModel {
        let configuration = try DSparkConfiguration(data: blockConfiguration(dspark: dspark))
        let draft = withRandomState(MLXRandom.RandomState(seed: 11)) { DSparkModel(configuration) }
        draft.update(parameters: draft.parameters().mapValues { $0.asType(.bfloat16) })
        draft.prepareContextProjection()
        return draft
    }
    private func mtpConfiguration() throws -> Data {
        Data(
            #"{"model_type":"qwen3_5_text","hidden_size":64,"num_hidden_layers":4,"intermediate_size":128,"num_attention_heads":4,"num_key_value_heads":2,"head_dim":16,"linear_num_value_heads":4,"linear_num_key_heads":2,"linear_key_head_dim":16,"linear_value_head_dim":16,"linear_conv_kernel_dim":3,"full_attention_interval":2,"partial_rotary_factor":0.5,"rms_norm_eps":0.000001,"vocab_size":128,"mtp_num_hidden_layers":1,"tie_word_embeddings":false}"#
                .utf8)
    }
    private func mtpPair(zeros: Bool = false) throws -> (Qwen35TextModel, Qwen35MTPDraftModel) {
        let config = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: mtpConfiguration())
        let target = withRandomState(MLXRandom.RandomState(seed: 7)) { Qwen35TextModel(config) }
        let draft = withRandomState(MLXRandom.RandomState(seed: 11)) { Qwen35MTPDraftModel(config) }
        if zeros {
            target.update(parameters: target.parameters().mapValues { MLXArray.zeros(like: $0) })
            draft.update(parameters: draft.parameters().mapValues { MLXArray.zeros(like: $0) })
        }
        target.train(false)
        draft.train(false)
        eval(target, draft)
        return (target, draft)
    }
    private func drain(_ iterator: any GenerationTokenIterator) -> [Int] {
        var tokens: [Int] = []
        while let token = iterator.next() { tokens.append(token) }
        XCTAssertNil(iterator.failure)
        return tokens
    }
    func testQwenBlockVerificationMatchesSingletonAndRewinds() throws {
        for quantized in [false, true] {
            let target = try qwen(quantized: quantized)
            let cache = try target.newCache(configuration: nil)
            let ordinary = try target.newCache(configuration: nil)
            let inputs = [1, 2, 3, 4]
            let output = try target.draftForward(
                MLXArray(inputs)[.newAxis], cache: cache,
                capture: [0, 2], verify: true)
            let expected = concatenated(
                inputs.map { target(MLXArray([$0])[.newAxis], cache: ordinary) }, axis: 1)
            XCTAssertEqual(output.logits.asArray(Float.self), expected.asArray(Float.self))
            try output.commit?(2)
            XCTAssertEqual(cache.map(\.offset), [2, 2, 2, 2])
            let fresh = try target.newCache(configuration: nil)
            for token in inputs.prefix(2) {
                eval(target(MLXArray([token])[.newAxis], cache: fresh))
            }
            XCTAssertEqual(
                target(MLXArray([9])[.newAxis], cache: cache).asArray(Float.self),
                target(MLXArray([9])[.newAxis], cache: fresh).asArray(Float.self))
        }
    }
    func testQwenDFlashAndDSparkParityIncludingSamplingAndSessionReuse() throws {
        for dspark in [false, true] {
            let target = try qwen(quantized: true)
            let draft = try blockDraft(dspark: dspark)
            let input = LMInput(tokens: MLXArray([1, 2, 3, 4, 5, 6, 7]))
            for temperature: Float in [0, 0.7] {
                for limit in [0, 1, 2, 7, 17] {
                    let options = ModelGenerationOptions(
                        prefill: .init(stepSize: 3),
                        sampling: .init(temperature: temperature, topP: 0.9, seed: 42),
                        output: .init(maxTokens: limit))
                    let expected = drain(
                        try AutoregressiveIterator(input: input, model: target, options: options))
                    for count in [1, 4, 6] {
                        let vault = DraftCacheVault()
                        let iterator = try DFlashIterator(
                            input: input, target: target, draft: draft,
                            options: options, speculation: .init(draftTokenCount: count),
                            checkpoint: { vault.latest = $0 })
                        XCTAssertEqual(drain(iterator), expected)
                        XCTAssertEqual(iterator.statistics.strategy, dspark ? .dspark : .dflash)
                        let saved = try XCTUnwrap(vault.latest)
                        let suffix = [8, 9]
                        let warm = try DFlashIterator(
                            input: LMInput(tokens: MLXArray(suffix)),
                            target: target, draft: draft, options: options, speculation: .init(),
                            reuse: saved)
                        let cold = try DFlashIterator(
                            input: LMInput(tokens: MLXArray(saved.tokens + suffix)),
                            target: target, draft: draft, options: options, speculation: .init())
                        XCTAssertEqual(drain(warm), drain(cold))
                    }
                }
            }
        }
    }
    func testMTPGreedyParityRejectionAcceptanceLimitsAndSessionReuse() throws {
        for zeros in [false, true] {
            let (target, draft) = try mtpPair(zeros: zeros)
            for policy: ModelPrefillOptions.Chunking in [.unchunked, .balanced, .remainder] {
                for limit in [0, 1, 2, 3, 7, 17] {
                    let options = ModelGenerationOptions(
                        prefill: .init(stepSize: 3, chunking: policy),
                        sampling: .init(temperature: 0), output: .init(maxTokens: limit))
                    let prompt = [1, 2, 3, 4, 5, 6, 7]
                    let input = LMInput(tokens: MLXArray(prompt))
                    let expected = drain(
                        try AutoregressiveIterator(input: input, model: target, options: options))
                    let vault = DraftCacheVault()
                    let iterator = try MTPSpeculativeTokenIterator(
                        input: input, target: target, draft: draft,
                        options: options, speculation: .init(), checkpoint: { vault.latest = $0 })
                    XCTAssertEqual(
                        drain(iterator), expected,
                        "zeros=\(zeros), limit=\(limit), policy=\(policy)")
                    if limit > 2 {
                        XCTAssertGreaterThan(iterator.statistics.proposedTokenCount, 0)
                        if zeros {
                            XCTAssertEqual(
                                iterator.statistics.acceptedTokenCount,
                                iterator.statistics.proposedTokenCount)
                        } else {
                            XCTAssertLessThan(
                                iterator.statistics.acceptedTokenCount,
                                iterator.statistics.proposedTokenCount)
                        }
                    }
                    let saved = try XCTUnwrap(vault.latest)
                    XCTAssertEqual(saved.draftCaches[0].offset + 1, saved.tokens.count)
                    let warm = try MTPSpeculativeTokenIterator(
                        input: LMInput(tokens: MLXArray([8, 9])),
                        target: target, draft: draft, options: options, speculation: .init(),
                        reuse: saved)
                    let cold = try MTPSpeculativeTokenIterator(
                        input: LMInput(tokens: MLXArray(saved.tokens + [8, 9])),
                        target: target, draft: draft, options: options, speculation: .init())
                    XCTAssertEqual(drain(warm), drain(cold))
                }
            }
            XCTAssertThrowsError(
                try MTPSpeculativeTokenIterator.validate(
                    options: .init(sampling: .init(temperature: 0.7)),
                    speculation: .init(), target: target, draft: draft))
            XCTAssertThrowsError(
                try MTPSpeculativeTokenIterator.validate(
                    options: .init(sampling: .init(temperature: 0)),
                    speculation: .init(draftTokenCount: 2), target: target, draft: draft))
            XCTAssertThrowsError(try draft.validate(target: qwen()))
        }
    }
    func testQuantizedDraftBackbonesAndInterruptedRoundReuse() throws {
        let (target, mtp) = try mtpPair(zeros: true)
        quantize(model: mtp, groupSize: 32, bits: 4)
        let options = ModelGenerationOptions(
            sampling: .init(temperature: 0), output: .init(maxTokens: 12))
        let vault = DraftCacheVault()
        let iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray([1, 2, 3])),
            target: target, draft: mtp, options: options, speculation: .init(),
            checkpoint: { vault.latest = $0 })
        XCTAssertEqual(iterator.next(), 0)
        XCTAssertEqual(iterator.next(), 0)  // Stop before delivering the round's bonus.
        let saved = try XCTUnwrap(vault.latest)
        XCTAssertEqual(saved.tokens, [1, 2, 3])
        let warm = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray([8, 9])),
            target: target, draft: mtp, options: options, speculation: .init(), reuse: saved)
        XCTAssertEqual(drain(warm), Array(repeating: 0, count: 12))
        for dspark in [false, true] {
            let qwen = try qwen(quantized: true)
            let draft = try blockDraft(dspark: dspark)
            quantize(model: draft, groupSize: 32, bits: 4)
            draft.prepareContextProjection()
            let input = LMInput(tokens: MLXArray([1, 2, 3]))
            XCTAssertEqual(
                drain(
                    try DFlashIterator(
                        input: input, target: qwen, draft: draft,
                        options: options, speculation: .init())),
                drain(try AutoregressiveIterator(input: input, model: qwen, options: options)))
        }
    }

    func testMTPReusesPrefixWhenTemplateDropsGenerationMarker() throws {
        let (target, draft) = try mtpPair()
        let options = ModelGenerationOptions(
            sampling: .init(temperature: 0), output: .init(maxTokens: 12))
        var checkpoints: [DraftCacheCheckpoint] = []
        let iterator = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray([1, 2, 3, 4])),
            target: target, draft: draft, options: options, speculation: .init(),
            checkpoint: { checkpoints.append($0) })
        _ = drain(iterator)
        let saved = try XCTUnwrap(checkpoints.first { $0.tokens == [1, 2, 3] })
        let warm = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray([8, 9])),
            target: target, draft: draft, options: options, speculation: .init(), reuse: saved)
        let cold = try MTPSpeculativeTokenIterator(
            input: LMInput(tokens: MLXArray([1, 2, 3, 8, 9])),
            target: target, draft: draft, options: options, speculation: .init())
        XCTAssertEqual(drain(warm), drain(cold))
    }

    func testMTPQuantizedVisionLanguageTargetMatchesDecodeAndRewinds() throws {
        var text = try XCTUnwrap(
            JSONSerialization.jsonObject(with: mtpConfiguration()) as? [String: Any])
        text["model_type"] = "qwen3_5"
        text["rope_parameters"] = [
            "type": "default", "mrope_section": [2, 1, 1],
            "rope_theta": 100000.0, "partial_rotary_factor": 0.5,
        ]
        let config = try JSONDecoder().decode(
            Qwen35Configuration.self,
            from:
                JSONSerialization.data(withJSONObject: [
                    "model_type": "qwen3_5", "text_config": text,
                    "image_token_id": 500, "video_token_id": 501,
                    "vision_start_token_id": 502, "vision_end_token_id": 503,
                    "vision_config": [
                        "model_type": "qwen3_vl", "depth": 2, "hidden_size": 32,
                        "intermediate_size": 64, "out_hidden_size": 64, "num_heads": 2,
                        "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
                        "num_position_embeddings": 64,
                    ],
                ]))
        for quantized in [false, true] {
            let target = withRandomState(MLXRandom.RandomState(seed: 7)) { Qwen35(config) }
            target.update(parameters: target.parameters().mapValues { $0.asType(.bfloat16) })
            if quantized { quantize(model: target.languageModel, groupSize: 32, bits: 4) }
            try target.prepare()
            target.train(false)
            let draft = try mtpPair().1
            let options = ModelGenerationOptions(
                sampling: .init(temperature: 0), output: .init(maxTokens: 17))
            let prompt = LMInput(tokens: MLXArray([1, 2, 3, 4, 5]))
            let ordinary = drain(
                try AutoregressiveIterator(
                    input: prompt, model: target, options: options))
            let vault = DraftCacheVault()
            let iterator = try MTPSpeculativeTokenIterator(
                input: prompt, target: target, draft: draft, options: options,
                speculation: .init(), checkpoint: { vault.latest = $0 })
            XCTAssertEqual(drain(iterator), ordinary)
            XCTAssertLessThan(
                iterator.statistics.acceptedTokenCount,
                iterator.statistics.proposedTokenCount)
            let saved = try XCTUnwrap(vault.latest)
            let prefix = saved.tokens
            let reused = saved
            let warm = try MTPSpeculativeTokenIterator(
                input: LMInput(tokens: MLXArray([8, 9])), target: target, draft: draft,
                options: options, speculation: .init(), reuse: reused)
            let cold = try MTPSpeculativeTokenIterator(
                input: LMInput(tokens: MLXArray(prefix + [8, 9])), target: target, draft: draft,
                options: options, speculation: .init())
            XCTAssertEqual(drain(warm), drain(cold))
        }
    }

    func testMTPFlatSplitWeightsPreserveNormsAndQuantization() throws {
        let (_, draft) = try mtpPair()
        let native = ["mtp.norm.weight": MLXArray([Float(0), 1])]
        XCTAssertEqual(
            draft.sanitize(weights: native)["mtp.norm.weight"]!.asArray(Float.self), [1, 2])
        let flat = ["norm.weight": MLXArray([Float(0), 1]), "fc.scales": MLXArray([Float(2)])]
        let result = draft.sanitize(weights: flat)
        XCTAssertEqual(result["mtp.norm.weight"]!.asArray(Float.self), [0, 1])
        XCTAssertNotNil(result["mtp.fc.scales"])
    }
    func testDraftLoadersAcceptTokenizerlessSplitHeads() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        func write(_ name: String, config: Data, weights: [String: MLXArray]) throws -> URL {
            let path = root.appendingPathComponent(name)
            try FileManager.default.createDirectory(at: path, withIntermediateDirectories: true)
            try config.write(to: path.appendingPathComponent("config.json"))
            try MLX.save(arrays: weights, url: path.appendingPathComponent("model.safetensors"))
            return path
        }
        let (target, draft) = try mtpPair()
        let targetURL = try write(
            "mtp-target", config: mtpConfiguration(),
            weights: Dictionary(uniqueKeysWithValues: target.parameters().flattened()))
        let draftConfig = try JSONSerialization.data(withJSONObject: [
            "model_type": "qwen3_5_mtp",
            "text_config": JSONSerialization.jsonObject(with: mtpConfiguration()),
        ])
        let draftURL = try write(
            "mtp-draft", config: draftConfig,
            weights: Dictionary(
                uniqueKeysWithValues: draft.parameters().flattened().map {
                    (String($0.0.dropFirst(4)), $0.1)
                }))
        let loaded = try await NativDraftModel.loadModel(
            target: .directory(targetURL), draft: .directory(draftURL),
            using: TextFixtureTokenizerLoader())
        let request = loaded.generate(
            input: UserInput(prompt: "hello"),
            options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 7)))
        _ = try await request.response()
        let summary = try await request.metrics()
        XCTAssertEqual(summary.speculative?.strategy, .mtp)
        let session = ModelSession(
            model: loaded,
            options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 5)))
        let response = try await session.respond(to: "hello")
        _ = try await response.response()
        let sessionSummary = try await response.metrics()
        XCTAssertEqual(sessionSummary.speculative?.strategy, .mtp)
        await session.close()
        let qwen = try qwen()
        let qwenURL = try write(
            "qwen-target", config: qwenConfiguration(),
            weights: Dictionary(uniqueKeysWithValues: qwen.parameters().flattened()))
        for dspark in [false, true] {
            let drafter = try blockDraft(dspark: dspark)
            let path = try write(
                dspark ? "dspark" : "dflash", config: blockConfiguration(dspark: dspark),
                weights: Dictionary(uniqueKeysWithValues: drafter.parameters().flattened()))
            let pair = try await NativDraftModel.loadModel(
                target: .directory(qwenURL), draft: .directory(path),
                using: TextFixtureTokenizerLoader())
            let result = pair.generate(
                input: UserInput(prompt: "hello"),
                options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 7)))
            _ = try await result.response()
            let summary = try await result.metrics()
            XCTAssertEqual(summary.speculative?.strategy, dspark ? .dspark : .dflash)
        }
    }
}
