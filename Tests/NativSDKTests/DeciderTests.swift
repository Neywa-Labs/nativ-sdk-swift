import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct DecisionTestTokenizer: Tokenizer {
    var duplicateLabels = false
    var bosToken: String? { nil }
    var eosToken: String? { "<pad>" }
    var unknownToken: String? { nil }
    static let labels: [String] = {
        let alphabet = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZ").map(String.init)
        return Array((alphabet + alphabet.flatMap { a in alphabet.map { a + $0 } }).prefix(255))
    }()
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        if let label = Self.labels.firstIndex(of: text) { return [duplicateLabels ? 1 : label + 1] }
        return text.utf8.map { Int($0) + 300 }
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { token == "<pad>" ? 0 : nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { throw TokenizerError.missingChatTemplate }
}
private struct DecisionTestTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer { DecisionTestTokenizer() }
}

final class DeciderTests: XCTestCase {
    private func configuration(_ changes: [String: Any] = [:]) throws -> Data {
        var object: [String: Any] = [
            "model_type": "decider2", "hidden_size": 16, "intermediate_size": 32,
            "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
            "head_dim": 8, "vocab_size": 1024, "full_attention_interval": 2,
            "linear_num_key_heads": 1, "linear_num_value_heads": 2,
            "linear_key_head_dim": 8, "linear_value_head_dim": 8,
            "linear_conv_kernel_dim": 4, "partial_rotary_factor": 0.5,
            "tie_word_embeddings": true, "pad_token_id": 0,
            "decision_config": [
                "temperature": 1.1,
                "temperature_by_type": ["choice": 1.2, "score": 1.3, "noul": 1.4],
                "isolated_levels": true,
            ],
        ]
        object.merge(changes) { _, new in new }
        return try JSONSerialization.data(withJSONObject: object)
    }
    private func model() throws -> DeciderModel {
        let model = try XCTUnwrap(
            try ModelRegistry.default.createDecisionModel(
                configuration: configuration(), modelType: "decider2") as? DeciderModel)
        model.train(false)
        return model
    }
    private func calibration() throws -> DeciderCalibration {
        try JSONDecoder().decode(
            DeciderCalibration.self,
            from: Data(
                #"{"temperature":1.1,"temperature_by_type":{"choice":1.2,"score":1.3,"noul":1.4},"isolated_levels":true}"#
                    .utf8))
    }
    private func named(_ questions: [String: DecisionQuestion]) -> [NamedDecisionQuestion] {
        questions.keys.sorted().map { .init($0, question: questions[$0]!) }
    }

    func testQuestionPlanUsesNativePerTypeTemperaturesAndIsolation() throws {
        let plan = try DeciderPredictionPlan(
            questions: named([
                "a": .choice("Pick", criteria: ["a", "b"]),
                "b": .bool("True?"), "c": .score("Rate", levels: ["0: Poor", "1: Good"]),
            ]), calibration: calibration(), options: .init())
        XCTAssertEqual(plan.rows.map(\.temperature), [1.2, 1.4, 1.3, 1.3])
        XCTAssertEqual(
            plan.rows[2].instruction, "Rate\nProposed answer: Poor\nDoes the proposed answer fit?")
        let answers = try plan.answers(probabilities: [
            [0.1, 0.9], [0.2, 0.8], [0.8, 0.2], [0.2, 0.8],
        ])
        XCTAssertEqual(answers["a"]?.value, .choice("b"))
        XCTAssertEqual(answers["b"]?.value, .bool(true))
        XCTAssertEqual(answers["b"]?.probability, 0.8)
        XCTAssertNil(answers["b"]?.probabilities)
        XCTAssertEqual(answers["c"]?.value, .score(0.8))
        XCTAssertEqual(answers["c"]?.metadata["fit_mass"], .double(1))
        XCTAssertEqual(
            answers["c"]?.metadata["level_fit"], .object(["0": .double(0.2), "1": .double(0.8)]))
        let packed = try DeciderPredictionPlan(
            questions: named(["score": .score("Rate", levels: ["low", "high"])]),
            calibration: calibration(), options: .init(independent: false, isolated: true))
        XCTAssertEqual(packed.rows.count, 1)
        XCTAssertEqual(packed.rows[0].options, ["0: low", "1: high"])
        XCTAssertFalse(packed.lookup[0].isolated)
    }

    func testPromptsHaveNoChatTemplateAndExplicitLongChoiceLabelTokens() throws {
        let tokenizer = DecisionTestTokenizer()
        let builder = try DeciderPromptBuilder(
            tokenizer: tokenizer, state: "refund", maxStateTokens: 32768)
        let row = DeciderRow(
            instruction: "Which team?", options: ["billing", "sales"], temperature: 1)
        let text =
            "Context:\nrefund\n\nQuestion: Which team?\nOptions:\n(A) billing\n(B) sales\nAnswer: ("
        XCTAssertEqual(builder.prompt(row), tokenizer.encode(text: text, addSpecialTokens: false))
        let long = DeciderRow(
            instruction: "Pick", options: (0 ..< 11).map(String.init), temperature: 1)
        let tokens = builder.prompt(long)
        for id in builder.labelTokenIDs.prefix(11) { XCTAssertTrue(tokens.contains(id)) }
        let packed = builder.packed([row, row])
        XCTAssertEqual(packed.positions.count, 2)
        XCTAssertLessThan(packed.positions[0], packed.positions[1])
        XCTAssertEqual(packed.positions.last, packed.tokens.count - 1)
        let truncated = try DeciderPromptBuilder(
            tokenizer: tokenizer, state: "a long state", maxStateTokens: 2)
        XCTAssertEqual(truncated.contextTokens.count, 2)
        XCTAssertTrue(truncated.prompt(row).count > 2)
    }

    func testStructuredStateAnnotationsAndUnicode() throws {
        let state = JSONValue.array((0 ..< 8).map { .object(["name": .string("é/\($0)")]) })
        guard case .array(let items) = DeciderPromptBuilder.annotate(state) else {
            return XCTFail()
        }
        XCTAssertEqual(items[3], .object(["_index": .int(3), "name": .string("é/3")]))
        XCTAssertEqual(DeciderPromptBuilder.annotate(.array([.int(1)])), .array([.int(1)]))
        let text = try DeciderPromptBuilder.renderJSON(
            .object(["a": .string("é/x"), "b": .bool(false)]))
        XCTAssertEqual(text, #"{"a": "é/x", "b": false}"#)
    }

    func testPredictionBackboneMatchesDirectLabelLogitsAndPackedEvaluation() throws {
        let model = try model()
        let tokenizer = DecisionTestTokenizer()
        let questions = named(["pick": .choice("Pick one", criteria: ["left", "right"])])
        let result = try model.predict(
            state: "hello", questions: questions, tokenizer: tokenizer,
            options: .init(isolated: false))
        let builder = try DeciderPromptBuilder(
            tokenizer: tokenizer, state: "hello", maxStateTokens: 32768)
        let row = DeciderRow(instruction: "Pick one", options: ["left", "right"], temperature: 1.2)
        let ids = builder.prompt(row)
        let hidden = model.languageModel.model(MLXArray(ids, [1, ids.count]))
        let logits = matmul(
            hidden[0, ids.count - 1],
            model.languageModel.model.embedTokens(MLXArray(Array(builder.labelTokenIDs.prefix(2))))
                .T)
        let expected = softmax(logits.asType(.float32) / Float(1.2)).asArray(Float.self)
        XCTAssertEqual(
            try XCTUnwrap(result.answers["pick"]?.probabilities?["left"]),
            Double(expected[0]), accuracy: 1e-3)
        XCTAssertEqual(result.inputTokenCount, ids.count)
        XCTAssertEqual(result.outputTokenCount, 0)
        let packed = try model.predict(
            state: "hello", questions: questions, tokenizer: tokenizer,
            options: .init(independent: false))
        XCTAssertEqual(packed.answers, result.answers)
        let multiple = try model.predict(
            state: "hello",
            questions: named([
                "pick": questions[0].question, "bool": .bool("Is this true?"),
            ]), tokenizer: tokenizer, options: .init())
        // Batch shape can select different floating-point accumulation kernels.
        XCTAssertEqual(multiple.answers["pick"]?.value, result.answers["pick"]?.value)
        for label in ["left", "right"] {
            XCTAssertEqual(
                try XCTUnwrap(multiple.answers["pick"]?.probabilities?[label]),
                try XCTUnwrap(result.answers["pick"]?.probabilities?[label]), accuracy: 1e-3)
        }
        let json =
            try JSONSerialization.jsonObject(with: JSONEncoder().encode(result)) as! [String: Any]
        XCTAssertEqual((json["usage"] as? [String: Int])?["output_tokens"], 0)
    }

    func testInvalidQuestionsTypesAndCalibrationFailBeforeInference() throws {
        let model = try model()
        let cases: [[NamedDecisionQuestion]] = [
            [], named(["q": .choice("Pick", criteria: ["a"])]),
            named(["q": .choice("Pick", criteria: ["a", "a"])]),
            named(["q": .choice("", criteria: ["a", "b"])]),
            named(["q": .score("Score", levels: Array(repeating: "level", count: 11))]),
            named(["q": .init(type: .multiLabel, instructions: "Pick", criteria: ["a"])]),
        ]
        for questions in cases {
            XCTAssertThrowsError(
                try model.predict(
                    state: "hello", questions: questions, tokenizer: DecisionTestTokenizer(),
                    options: .init()))
        }
        XCTAssertThrowsError(
            try model.predict(
                state: "hello", questions: named(["q": .bool("True?")]),
                tokenizer: DecisionTestTokenizer(duplicateLabels: true), options: .init()))
        XCTAssertThrowsError(
            try model.predict(
                state: "hello", questions: named(["q": .bool("True?")]),
                tokenizer: DecisionTestTokenizer(), options: .init(maxStateTokens: 0)))
        for temperature in [0, -1] {
            XCTAssertThrowsError(
                try ModelRegistry.default.createDecisionModel(
                    configuration: configuration(["decision_config": ["temperature": temperature]]),
                    modelType: "decider2"))
        }
        XCTAssertThrowsError(
            try ModelRegistry.default.createDecisionModel(
                configuration: configuration(["decision_config": [:]]), modelType: "decider2"))
        XCTAssertThrowsError(
            try ModelRegistry.default.createDecisionModel(
                configuration: configuration(["num_attention_heads": 0]), modelType: "decider2"))
        XCTAssertThrowsError(
            try ModelRegistry.default.createDecisionModel(
                configuration: configuration(), modelType: "qwen3_5_text"))
        XCTAssertThrowsError(
            try ModelRegistry.default.createModel(
                configuration: configuration(), modelType: "decider2"))
    }

    func testStrictCheckpointLoadingDispatchAndUnsupportedOperations() async throws {
        let model = try model()
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try configuration().write(to: directory.appendingPathComponent("config.json"))
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        // Converted names are already normalized; loading must not offset norms twice.
        let published = Dictionary(
            uniqueKeysWithValues: weights.map {
                (String($0.key.dropFirst("language_model.".count)), $0.value)
            })
        try MLX.save(arrays: published, url: directory.appendingPathComponent("model.safetensors"))
        let loaded = try await NativDecisionModel.loadModel(
            from: .directory(directory), using: DecisionTestTokenizerLoader())
        let capabilities = await loaded.capabilities
        XCTAssertEqual(capabilities, [.decision])
        let result = try await loaded.predict(
            state: "hello", questions: ["pick": .choice("Pick one", criteria: ["left", "right"])]
        ).result()
        let expected = try model.predict(
            state: "hello",
            questions: named(["pick": .choice("Pick one", criteria: ["left", "right"])]),
            tokenizer: DecisionTestTokenizer(), options: .init())
        XCTAssertEqual(result.answers, expected.answers)
        let assets = ResolvedModelAssets(
            source: .directory(directory), modelDirectory: directory, tokenizerDirectory: directory)
        let local = try await NativDecisionModel.loadModel(
            assets: assets, tokenizerLoader: DecisionTestTokenizerLoader())
        _ = try await local.predict(state: "hello", questions: ["b": .bool("Is this true?")])
            .result()
        do {
            _ = try await loaded.withModel { _, context in
                _ = try context.languageModel
                return true
            }
            XCTFail("Decision resources must not generate")
        } catch { XCTAssertEqual(error as? ModelExecutionError, .unsupportedOperation("generate")) }
        do {
            _ = try await loaded.withModel { _, context in
                try context.embeddingModel.maximumSequenceLength
            }
            XCTFail("Decision resources must not embed")
        } catch { XCTAssertEqual(error as? ModelExecutionError, .unsupportedOperation("embed")) }
        // Failed requests must release admission for subsequent predictions.
        _ = try await loaded.predict(state: "hello", questions: ["b": .bool("Is this true?")])
            .result()
        var missing = published
        missing.removeValue(forKey: "model.embed_tokens.weight")
        try MLX.save(arrays: missing, url: directory.appendingPathComponent("model.safetensors"))
        do {
            _ = try await NativDecisionModel.loadModel(
                from: .directory(directory), using: DecisionTestTokenizerLoader())
            XCTFail("Missing backbone weights must fail loading")
        } catch {}
    }

    func testQuantizedCheckpointUsesCanonicalPerLayerPathsAndLabelEmbeddings() async throws {
        let data = try configuration([
            "hidden_size": 32, "intermediate_size": 64,
            "quantization": [
                "group_size": 32, "bits": 4,
                "model.embed_tokens": ["group_size": 32, "bits": 8],
            ],
        ])
        let model = DeciderModel(try JSONDecoder().decode(DeciderConfiguration.self, from: data))
        try quantizeByParent(model: model) { path, module in
            if module is Embedding { return (32, 8, .affine) }
            guard let linear = module as? Linear, linear.weight.dim(1) % 32 == 0 else {
                return nil
            }
            return (32, 4, .affine)
        }
        XCTAssertTrue(model.languageModel.model.embedTokens is QuantizedEmbedding)
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try data.write(to: directory.appendingPathComponent("config.json"))
        let published = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map {
                (String($0.0.dropFirst("language_model.".count)), $0.1)
            })
        try MLX.save(arrays: published, url: directory.appendingPathComponent("model.safetensors"))
        let loaded = try await NativDecisionModel.loadModel(
            from: .directory(directory), using: DecisionTestTokenizerLoader())
        let questions: [String: DecisionQuestion] = ["pick": .choice("Pick", criteria: ["a", "b"])]
        let expected = try model.predict(
            state: "hello", questions: named(questions), tokenizer: DecisionTestTokenizer(),
            options: .init())
        let actual = try await loaded.predict(state: "hello", questions: questions).result()
        XCTAssertEqual(actual.answers, expected.answers)
    }

    func testPredictionRejectsLanguageModelsWithoutKeepingAdmission() async throws {
        let config = try JSONDecoder().decode(DeciderConfiguration.self, from: configuration())
        let tokenizer = DecisionTestTokenizer()
        let directory = FileManager.default.temporaryDirectory
        let context = ModelContext(
            metadata: .init(), model: Qwen35TextModel(config.text),
            processor: TextProcessor(.init(), tokenizer: tokenizer), tokenizer: tokenizer)
        let loaded = NativDecisionModel(
            assets: .init(
                source: .directory(directory), modelDirectory: directory,
                tokenizerDirectory: directory),
            metadata: .init(), context: ExecutionTransfer(context), policy: .init())
        do {
            _ = try await loaded.predict(state: "hello", questions: ["q": .bool("True?")]).result()
            XCTFail("Language models must not predict decisions")
        } catch { XCTAssertEqual(error as? ModelExecutionError, .unsupportedOperation("predict")) }
        let available = try await loaded.withModel { _, _ in true }
        XCTAssertTrue(available)
    }
}
