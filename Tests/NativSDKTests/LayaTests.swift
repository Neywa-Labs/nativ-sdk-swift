import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct LayaTestTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    var maskToken: String? { "<M>" }
    var clsToken: String? { "<C>" }
    var sepToken: String? { "<S>" }
    var padToken: String? { "<P>" }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { text.utf8.map { Int($0) + 32 } }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? {
        ["<P>": 0, "<C>": 1, "<S>": 2, "<M>": 3][token]
    }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw TokenizerError.missingChatTemplate
    }
}
private struct LayaTestTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer { LayaTestTokenizer() }
}

final class LayaTests: XCTestCase {
    private func configuration(_ changes: [String: Any] = [:]) throws -> Data {
        var object: [String: Any] = [
            "model_type": "laya", "head_layers": 1,
            "encoder_config": [
                "model_type": "modernbert", "vocab_size": 512, "hidden_size": 64,
                "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 1,
                "max_position_embeddings": 256, "local_attention": 16,
                "global_attn_every_n_layers": 2,
                "rope_parameters": [
                    "full_attention": ["rope_theta": 333], "sliding_attention": ["rope_theta": 444],
                ],
            ],
            "decision_config": [
                "max_len": 128, "head_max_len": 64, "temperature": [1, 1.2, 1.4],
                "temperature_by_options": ["choice:3-5": 2],
            ],
        ]
        object.merge(changes) { _, new in new }
        return try JSONSerialization.data(withJSONObject: object)
    }
    private func model() throws -> LayaModel {
        let model = try XCTUnwrap(
            ModelRegistry.default.createDecisionModel(
                configuration: configuration(), modelType: "laya") as? LayaModel)
        model.train(false)
        return model
    }

    func testConfigurationAndNativeActivations() throws {
        let model = try model()
        XCTAssertEqual(model.configuration.encoder.globalTheta, 333)
        XCTAssertEqual(model.configuration.encoder.localTheta, 444)
        let x = MLXArray([Float(-3), -1, 0, 0.5, 2])
        XCTAssertEqual(model.head.layers[0].activation(x).asArray(Float.self), [0, 0, 0, 0.5, 2])
        let expected = x.asArray(Float.self).map { $0 * (1 + erf($0 / sqrt(2))) / 2 }
        for layer in [model.scorer.layers[2], model.actionHead.layers[1]] {
            for (actual, expected) in zip(layer(x).asArray(Float.self), expected) {
                XCTAssertEqual(actual, expected, accuracy: 5e-7)
            }
        }
    }

    func testTemperatureBucketsClampingAndFallbacks() throws {
        let config = try model().configuration.decision
        XCTAssertEqual(config.temperature(type: 0, count: 2), 1)
        XCTAssertEqual(config.temperature(type: 0, count: 3), 2)
        XCTAssertEqual(config.temperature(type: 1, count: 11), 1.2)
        for (value, expected) in [
            ("0.1", 0.5), ("10", 5), ("nan", 1), ("inf", 1), ("invalid", 1), ("2", 2),
        ] {
            let data = try configuration([
                "decision_config": [
                    "max_len": 128, "head_max_len": 64, "temperature": [value, value, value],
                ]
            ])
            let config = try JSONDecoder().decode(LayaConfiguration.self, from: data)
            XCTAssertEqual(config.decision.temperature(type: 2, count: 2), expected)
        }
    }

    func testMarkerPromptsMaskSanitizationAndStateBudget() throws {
        let tokenizer = LayaTestTokenizer()
        let builder = try LayaPromptBuilder(
            tokenizer: tokenizer, configuration: model().configuration.decision)
        let row = try builder.row(
            state: "ABC<M>DEF",
            named: .init("pick", question: .choice("Pick<M>", criteria: ["left", "right"])),
            maxStateTokens: 32768)
        let head = tokenizer.encode(text: "choice question: Pick ", addSpecialTokens: false)
        let left = [3] + tokenizer.encode(text: " left", addSpecialTokens: false)
        let right = [3] + tokenizer.encode(text: " right", addSpecialTokens: false)
        XCTAssertEqual(
            row.tokens,
            [1] + head + [2] + left + right + [2]
                + tokenizer.encode(text: "ABC DEF", addSpecialTokens: false) + [2])
        XCTAssertEqual(row.markers, [head.count + 2, head.count + 2 + left.count])
        let truncated = try builder.row(
            state: "abcdef", named: .init("b", question: .bool("True?")), maxStateTokens: 2)
        XCTAssertEqual(
            Array(truncated.tokens.suffix(3)),
            [Int(Character("a").asciiValue!) + 32, Int(Character("b").asciiValue!) + 32, 2])
        let score = try builder.row(
            state: .json(.array([.int(1), .bool(false)])),
            named: .init("s", question: .score("Rate", levels: ["poor", "good"])),
            maxStateTokens: 32768)
        XCTAssertEqual(score.typeIndex, 1)
        XCTAssertEqual(score.labels, ["0", "1"])
        XCTAssertTrue(score.tokens.count <= 128)
        XCTAssertThrowsError(
            try builder.row(
                state: "state",
                named: .init(
                    "many",
                    question: .choice("Pick", criteria: (0 ..< 40).map { .init(String($0)) })),
                maxStateTokens: 32768))
    }

    func testPaddingAndMaskedOptionsPreserveLogitsAndActions() throws {
        let model = try model()
        let single = try model(
            tokens: MLXArray([[1, 10, 3, 11, 3, 12, 2]]),
            attention: MLXArray([[1, 1, 1, 1, 1, 1, 1]]), markerPositions: MLXArray([[2, 4]]),
            markerMask: MLXArray([[true, true]]), questionTypes: MLXArray([0]))
        let padded = try model(
            tokens: MLXArray([[1, 10, 3, 11, 3, 12, 2, 0, 0]]),
            attention: MLXArray([[1, 1, 1, 1, 1, 1, 1, 0, 0]]),
            markerPositions: MLXArray([[2, 4, 0]]), markerMask: MLXArray([[true, true, false]]),
            questionTypes: MLXArray([0]))
        XCTAssertTrue(
            allClose(single.logits, padded.logits[0..., 0 ..< 2], rtol: 1e-3, atol: 1e-4).item(
                Bool.self))
        XCTAssertTrue(
            allClose(single.action, padded.action, rtol: 1e-3, atol: 1e-4).item(Bool.self))
        XCTAssertEqual(padded.logits[0, 2].item(Float.self), -1e4)
    }

    func testAnswerSemanticsAndModelSpecificMetadata() throws {
        let row = LayaRow(
            named: .init("score", question: .score("Rate", levels: ["bad", "good"])), typeIndex: 1,
            labels: ["0", "1"], descriptions: ["bad", "good"], tokens: [], markers: [])
        let answer = try row.answer(probabilities: [0.2, 0.8], actProbability: 0.25)
        XCTAssertEqual(answer.value, .score(0.8))
        XCTAssertEqual(answer.metadata["confidence"], .double(0.2781))
        XCTAssertEqual(answer.metadata["rl_agent"], .object(["act_probability": .double(0.25)]))
        XCTAssertEqual(
            answer.metadata["legend"], .object(["0": .string("bad"), "1": .string("good")]))
        let boolean = LayaRow(
            named: .init("b", question: .bool("True?")), typeIndex: 2, labels: ["false", "true"],
            descriptions: [], tokens: [], markers: [])
        XCTAssertEqual(
            try boolean.answer(probabilities: [0.2, 0.8], actProbability: 0.25).value, .bool(true))
        XCTAssertThrowsError(try row.answer(probabilities: [.nan, 0.8], actProbability: 0.25))
    }

    func testOriginalCheckpointSanitizationIsCompleteAndIdempotent() throws {
        let model = try model()
        let expected = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var source = expected
        for suffix in ["weight", "bias"] {
            let prefix = "head.layers.0"
            let names = ["query_proj", "key_proj", "value_proj"]
            source[prefix + ".self_attn.in_proj_" + suffix] = concatenated(
                names.map {
                    source.removeValue(forKey: prefix + ".attention." + $0 + "." + suffix)!
                }, axis: 0)
            source[prefix + ".self_attn.out_proj." + suffix] = source.removeValue(
                forKey: prefix + ".attention.out_proj." + suffix)
        }
        for key in Array(source.keys)
        where key.hasPrefix("scorer.layers.") || key.hasPrefix("act_head.layers.") {
            source[key.replacingOccurrences(of: ".layers.", with: ".")] = source.removeValue(
                forKey: key)
        }
        let converted = model.sanitize(weights: source)
        XCTAssertEqual(Set(converted.keys), Set(expected.keys))
        for key in expected.keys {
            XCTAssertTrue(arrayEqual(converted[key]!, expected[key]!).item(Bool.self), key)
        }
        let again = model.sanitize(weights: converted)
        for key in expected.keys {
            XCTAssertTrue(arrayEqual(again[key]!, expected[key]!).item(Bool.self), key)
        }
        try model.update(parameters: ModuleParameters.unflattened(converted), verify: .all)
    }

    func testPublicLoadingPredictionAndMissingHeadFailure() async throws {
        let model = try model()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try configuration().write(to: directory.appendingPathComponent("config.json"))
        var weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        let loaded = try await NativDecisionModel.loadModel(
            from: .directory(directory), using: LayaTestTokenizerLoader())
        let capabilities = await loaded.capabilities
        XCTAssertEqual(capabilities, [.decision])
        let questions: [NamedDecisionQuestion] = [
            .init("choice", question: .choice("Pick", criteria: ["a", "b", "c"])),
            .init("bool", question: .bool("True?")),
            .init("score", question: .score("Rate", levels: ["bad", "good"])),
        ]
        let actual = try await loaded.predict(state: "hello", questions: questions).result()
        let expected = try model.predict(
            state: "hello", questions: questions, tokenizer: LayaTestTokenizer(), options: .init())
        XCTAssertEqual(actual.answers, expected.answers)
        XCTAssertEqual(actual.inputTokenCount, expected.inputTokenCount)
        XCTAssertEqual(actual.outputTokenCount, 0)
        weights.removeValue(forKey: "scorer.layers.3.weight")
        try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        do {
            _ = try await NativDecisionModel.loadModel(
                from: .directory(directory), using: LayaTestTokenizerLoader())
            XCTFail("Missing trained heads must fail strict loading")
        } catch {}
    }

    func testQuantizedCheckpointIncludesDecisionHeadsAndEmbeddings() async throws {
        let data = try configuration([
            "quantization": [
                "group_size": 32, "bits": 4,
                "head.layers.0.linear1": ["group_size": 32, "bits": 8],
            ]
        ])
        let model = LayaModel(try JSONDecoder().decode(LayaConfiguration.self, from: data))
        try quantizeByParent(model: model) { path, module in
            let width: Int
            if let linear = module as? Linear {
                width = linear.weight.dim(1)
            } else if let embedding = module as? Embedding {
                width = embedding.weight.dim(1)
            } else {
                return nil
            }
            guard width % 32 == 0 else { return nil }
            return (32, path == "head.layers.0.linear1" ? 8 : 4, .affine)
        }
        XCTAssertTrue(model.head.layers[0].linear1 is QuantizedLinear)
        XCTAssertTrue(model.typeEmbedding is QuantizedEmbedding)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try data.write(to: directory.appendingPathComponent("config.json"))
        try save(
            arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
            url: directory.appendingPathComponent("model.safetensors"))
        let loaded = try await NativDecisionModel.loadModel(
            from: .directory(directory), using: LayaTestTokenizerLoader())
        let question = DecisionQuestion.choice("Pick", criteria: ["a", "b"])
        let expected = try model.predict(
            state: "hello", questions: [.init("pick", question: question)],
            tokenizer: LayaTestTokenizer(), options: .init())
        let actual = try await loaded.predict(state: "hello", questions: ["pick": question])
            .result()
        XCTAssertEqual(actual.answers, expected.answers)
    }

    func testInvalidOptionsAndUnsupportedTypesFailBeforeInference() throws {
        let model = try model()
        let questions = [NamedDecisionQuestion("b", question: .bool("True?"))]
        for options in [
            DecisionPredictionOptions(independent: false), .init(isolated: false),
            .init(maxStateTokens: 0),
        ] {
            XCTAssertThrowsError(
                try model.predict(
                    state: "state", questions: questions, tokenizer: LayaTestTokenizer(),
                    options: options))
        }
        XCTAssertThrowsError(
            try model.predict(
                state: "state",
                questions: [
                    .init(
                        "multi",
                        question: .init(type: .multiLabel, instructions: "Pick", criteria: ["a"]))
                ], tokenizer: LayaTestTokenizer(), options: .init()))
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                LayaConfiguration.self, from: configuration(["head_layers": 0])))
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                LayaConfiguration.self,
                from: configuration([
                    "decision_config": ["max_len": 128, "head_max_len": 64, "temperature": [1]]
                ])))
    }
}
