import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct TextFixtureTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [1, 2, 3] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map(String.init).joined(separator: " ")
    }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { String(id) }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [1, 2, 3] }
}

struct TextFixtureTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer { TextFixtureTokenizer() }
}

final class TextModelLoadingTests: XCTestCase {
    func testSmallTextCheckpointsLoadAndGenerate() async throws {
        for name in ModelTestConfigurations.textModels
            + ModelTestConfigurations.additionalTextModels
            + ModelTestConfigurations.extendedModels
        {
            let data = try ModelTestConfigurations.configuration(name)
            let config = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
            try await assertTextCheckpoint(
                configuration: data,
                modelType: XCTUnwrap(config["model_type"] as? String))
        }
    }

    func testTextAliasesLoadExistingArchitecturesAndGenerate() async throws {
        for (alias, canonical) in [
            ("minimax_m2", "minimax"), ("iquestcoder", "llama"),
            ("solar_open", "glm4_moe"), ("kimi_k2", "deepseek_v3"),
            ("joyai_llm_flash", "deepseek_v3"),
        ] {
            var config = try XCTUnwrap(
                JSONSerialization.jsonObject(
                    with: ModelTestConfigurations.textConfiguration(canonical)) as? [String: Any])
            config["model_type"] = alias
            if canonical == "deepseek_v3" {
                for key in [
                    "routed_scaling_factor", "norm_topk_prob", "moe_layer_freq", "attention_bias",
                ] {
                    config.removeValue(forKey: key)
                }
            }
            try await assertTextCheckpoint(
                configuration: JSONSerialization.data(withJSONObject: config), modelType: alias)
        }
    }

    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: url) }
        return url
    }

    func testTextCheckpointsLoadAndGenerateWithoutProcessorMetadata() async throws {
        for modelType in [
            "llama", "qwen3", "gemma3_text", "gemma4", "qwen3_5", "qwen3_5_moe", "mistral3",
        ] {
            let data = try ModelTestConfigurations.textConfiguration(modelType)
            let model = try ModelRegistry.default.createModel(
                configuration: data, modelType: modelType)
            model.train(false)
            eval(model)
            let expected = model(MLXArray([1, 2, 3], [1, 3]), cache: nil).asArray(Float.self)
            let url = try directory()
            try MLX.save(
                arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
                url: url.appendingPathComponent("model.safetensors"))
            var config = try JSONSerialization.jsonObject(with: data) as! [String: Any]
            if ["gemma4", "qwen3_5", "qwen3_5_moe", "mistral3"].contains(modelType) {
                // Text conversions may retain the source model's unused vision configuration.
                config["vision_config"] = ["model_type": "unused_vision_tower"]
            }
            try JSONSerialization.data(withJSONObject: config).write(
                to: url.appendingPathComponent("config.json"))
            let loadedModel = try await NativModel.loadModel(
                from: .directory(url), using: TextFixtureTokenizerLoader())
            let actual = try await loadedModel.withModel { _, context in
                XCTAssertTrue(context.processor is TextProcessor, modelType)
                return try context.languageModel(MLXArray([1, 2, 3], [1, 3]), cache: nil).asArray(
                    Float.self)
            }
            XCTAssertEqual(actual, expected, modelType)
            let request = loadedModel.generate(
                input: UserInput(prompt: "hello"),
                options: .init(
                    sampling: .init(temperature: 0), output: .init(maxTokens: 2)))
            for try await _ in request {}
            let summary = try await request.metrics()
            XCTAssertEqual(summary.generatedTokenCount, 2, modelType)
        }
    }

    func testJinaArchitectureLoadsProjectorSidecarAndScoresDocuments() async throws {
        var config =
            try JSONSerialization.jsonObject(
                with: ModelTestConfigurations.textConfiguration("qwen3"))
            as! [String: Any]
        config["architectures"] = ["JinaForRanking"]
        let data = try JSONSerialization.data(withJSONObject: config)
        let model = try XCTUnwrap(
            ModelRegistry.default.createModel(configuration: data, modelType: "qwen3")
                as? JinaRerankerModel)
        let input = RerankerInput(
            tokenIds: [1, 7, 2, 8, 3, 8], markerTokenIds: .init(query: 7, document: 8))
        eval(model)
        let expected = try model.score(input: input, documentCount: 2)
        XCTAssertTrue(expected.allSatisfy { $0.isFinite && (-1 ... 1).contains($0) })
        XCTAssertThrowsError(try model.score(input: input, documentCount: 3))
        let url = try directory()
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try MLX.save(
            arrays: weights.filter { !$0.key.hasPrefix("projector.") },
            url: url.appendingPathComponent("model.safetensors"))
        let projector = Dictionary(
            uniqueKeysWithValues: weights.filter { $0.key.hasPrefix("projector.") }.map {
                (String($0.key.dropFirst("projector.".count)), $0.value)
            })
        try MLX.save(arrays: projector, url: url.appendingPathComponent("projector.safetensors"))
        try data.write(to: url.appendingPathComponent("config.json"))
        let loadedModel = try await NativModel.loadModel(
            from: .directory(url), using: TextFixtureTokenizerLoader())
        let scores = try await loadedModel.withModel { _, context in
            let reranker = try XCTUnwrap(context.model as? any ListwiseRerankerModel)
            return try reranker.score(input: input, documentCount: 2)
        }
        XCTAssertEqual(scores, expected)
    }

    func testQwenDrafterCommitsAcceptedTokensAndRollsBackRejectedTokens() throws {
        var config =
            try JSONSerialization.jsonObject(
                with: ModelTestConfigurations.textConfiguration("qwen3_5_text")) as! [String: Any]
        config["mtp_num_hidden_layers"] = 1
        let cfg = try JSONDecoder().decode(
            Qwen35TextConfiguration.self, from: JSONSerialization.data(withJSONObject: config))
        let target = Qwen35TextModel(cfg)
        let drafter = Qwen35MTPDraftModel(cfg)
        eval(target, drafter)
        let prompt = MLXArray([1, 2, 3], [1, 3])
        var targetState = LMOutput.State()
        targetState[mtpEmitFlagKey] = true
        let output = target(.init(tokens: prompt), cache: nil, state: targetState)
        let hidden = try XCTUnwrap(output.state?[mtpLastHiddenStatesKey])
        for pattern in [[0, 0], [0, 1], [1, 0], [1, 1]] {
            var state = drafter.makeState(configuration: nil)
            var bonus = MLXArray([4])
            drafter.prepareDrafterState(
                target: target, promptTokens: prompt, targetHidden: hidden,
                firstBonus: bonus, positionDeltas: nil, state: &state, sampler: ArgMaxSampler())
            var expectedPosition = 3
            for accepted in pattern {
                let proposal = drafter.draftBlock(
                    target: target, lastToken: bonus,
                    lastHidden: hidden[0..., (-1)..., 0...], sharedKV: [:], positionDeltas: nil,
                    queryOffset: expectedPosition, blockSize: 2, state: &state,
                    sampler: ArgMaxSampler())
                eval(proposal)
                let finalToken = MLXArray([8 + accepted])
                drafter.commitDrafterState(
                    target: target, targetHidden: zeros([1, 2, cfg.hiddenSize]),
                    draftTokens: proposal, acceptedCount: accepted, finalToken: finalToken,
                    positionDeltas: nil, state: &state, sampler: ArgMaxSampler())
                eval(state.cache)
                expectedPosition += accepted + 1
                XCTAssertEqual(state.nextPosition, expectedPosition)
                XCTAssertTrue(state.cache.allSatisfy { $0.offset == expectedPosition })
                bonus = finalToken
            }
        }
    }

    func testNemotronLinearSpeculationMatchesGreedyGeneration() throws {
        let model = NemotronLabsDiffusion(
            .init(
                vocabularySize: 64, hiddenSize: 32,
                intermediateSize: 64, hiddenLayers: 2, attentionHeads: 4, kvHeads: 2,
                maskTokenID: 63))
        let weights = model.parameters().mapValues { zeros(like: $0) }
        try model.update(parameters: weights, verify: .all)
        let prompt = [1, 2, 3]
        let expected = model.arGenerate(promptIds: prompt, maxNewTokens: 8).tokens
        XCTAssertEqual(expected, prompt + Array(repeating: 0, count: 8))
        XCTAssertEqual(
            model.linearSpecGenerate(
                promptIds: prompt, maxNewTokens: 8, blockLength: 4, threshold: 0
            ).tokens, expected)
        XCTAssertEqual(
            model.diffusionGenerate(
                promptIds: prompt, maxNewTokens: 8, blockLength: 4, threshold: 0
            ).tokens, expected)
    }
}
