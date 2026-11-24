import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct ResourceTokenizer: Tokenizer {
    var tokens: [String: Int] = [:]
    var eosToken: String? = nil
    var bosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [2, 3] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "fixture" }
    func convertTokenToId(_ token: String) -> Int? { tokens[token] }
    func convertIdToToken(_ id: Int) -> String? { tokens.first { $0.value == id }?.key }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [2, 3] }
}

private actor ResourceTokenizerLoader: TokenizerLoader {
    var directories: [URL] = []
    let tokenizer: ResourceTokenizer
    init(tokenizer: ResourceTokenizer = .init()) { self.tokenizer = tokenizer }
    func load(from directory: URL) async throws -> any Tokenizer {
        directories.append(directory)
        return tokenizer
    }
}

private struct FailingResourceTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer {
        throw CocoaError(.fileReadNoPermission)
    }
}

private struct ResourceMessageGenerator: MessageGenerator {
    func generate(message: Chat.Message) -> Message {
        ["role": message.role.rawValue, "content": "caller"]
    }
}

private final class ResourceModel: Module, LLMModel, SpeculativeDecodingModel {
    @ModuleInfo var embedding = Embedding(embeddingCount: 8, dimensions: 64)
    @ModuleInfo var projection = Linear(64, 8, bias: false)
    var preparations = 0
    var cacheCreations = 0
    var loraLayers: [Module] { [] }
    var toolCallFormat: ToolCallFormat? { .glm4 }
    var reasoningConfig: ReasoningConfig? { .alwaysOnThinking }
    func prepare() { preparations += 1 }
    func newCache(configuration: KVCacheConfiguration?) -> [KVCache] {
        cacheCreations += 1
        return []
    }
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        projection(embedding(inputs))
    }
}

final class LoadedModelTests: XCTestCase {
    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: url) }
        return url
    }

    private func factory() -> ModelLoader {
        ModelLoader(
            registry: ModelRegistry(models: ["resource_fixture": { _ in ResourceModel() }]))
    }

    private func checkpoint(in directory: URL, quantized: Bool = false) throws {
        let model = ResourceModel()
        if quantized { quantize(model: model, groupSize: 32, bits: 4) }
        try MLX.save(
            arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
            url: directory.appendingPathComponent("model.safetensors"))
        try Data(#"{"model_type":"resource_fixture","eos_token_id":1}"#.utf8)
            .write(to: directory.appendingPathComponent("config.json"))
    }

    private static func greedyTokens(_ model: any LanguageModel) throws -> [Int] {
        var tokens = [2, 3]
        var generated: [Int] = []
        for _ in 0 ..< 4 {
            let logits = model(MLXArray(tokens)[.newAxis], cache: nil)
            let token = argMax(logits[0, -1], axis: -1).item(Int.self)
            generated.append(token)
            tokens.append(token)
        }
        return generated
    }

    func testLoadingMergesMetadataAndPreparesModelWithoutAllocatingCaches() async throws {
        let dir = try directory()
        let tokenizerDir = try directory()
        try checkpoint(in: dir)
        try Data(#"{"eos_token_id":[5,6],"stop_strings":["generation-stop"],"stop":"done"}"#.utf8)
            .write(to: dir.appendingPathComponent("generation_config.json"))
        let metadata = ModelMetadata(
            extraEOSTokens: ["extra"], stopStrings: ["caller-stop"],
            eosTokenIds: [9],
            messageGenerator: ResourceMessageGenerator())
        let loader = ResourceTokenizerLoader()
        let loadedModel = try await factory().load(
            assets: ResolvedModelAssets(
                source: .repository("fixture/model", revision: "pinned"),
                tokenizerSource: .directory(tokenizerDir),
                modelDirectory: dir, tokenizerDirectory: tokenizerDir),
            metadata: metadata, tokenizerLoader: loader, policy: .init())

        XCTAssertEqual(loadedModel.metadata.eosTokenIds, [5, 6, 9])
        XCTAssertEqual(
            loadedModel.metadata.stopStrings, ["caller-stop", "generation-stop", "done"])
        let seen = await loader.directories
        XCTAssertEqual(seen, [tokenizerDir])
        try await loadedModel.withModel { _, context in
            XCTAssertEqual(context.metadata.extraEOSTokens, ["extra"])
            XCTAssertEqual(context.metadata.toolCallFormat, .glm4)
            XCTAssertEqual(context.metadata.reasoningConfig, .alwaysOnThinking)
            XCTAssertTrue(context.metadata.messageGenerator is ResourceMessageGenerator)
            XCTAssertTrue(context.processor is TextProcessor)
            let model = try XCTUnwrap(context.model as? ResourceModel)
            XCTAssertFalse(model.training)
            XCTAssertEqual(model.preparations, 1)
            XCTAssertEqual(model.cacheCreations, 0)
        }
    }

    func testExecutorMatchesDirectCheckpointForward() async throws {
        let dir = try directory()
        try checkpoint(in: dir)
        let upstream = ResourceModel()
        try upstream.update(
            parameters: ModuleParameters.unflattened(
                MLX.loadArrays(url: dir.appendingPathComponent("model.safetensors"))), verify: .all)
        upstream.train(false)
        let expected = try Self.greedyTokens(upstream)
        XCTAssertEqual(expected.count, 4)
        let loadedModel = try await factory().load(
            from: .directory(dir), using: ResourceTokenizerLoader())
        let actual = try await loadedModel.withModel { _, context in
            try Self.greedyTokens(context.languageModel)
        }
        XCTAssertEqual(actual, expected)
    }

    func testExplicitMetadataAndMalformedOptionalGenerationConfig()
        async throws
    {
        let dir = try directory()
        try checkpoint(in: dir)
        try Data("invalid json".utf8).write(
            to: dir.appendingPathComponent("generation_config.json"))
        let resources = try await factory().loadResources(
            from: .directory(dir), using: ResourceTokenizerLoader(),
            metadata: .init(
                extraEOSTokens: ["stop"], toolCallFormat: .json))

        XCTAssertEqual(resources.metadata.toolCallFormat, .json)
        XCTAssertEqual(resources.metadata.eosTokenIds, [1])
        XCTAssertEqual(resources.metadata.stopStrings, ["stop"])
    }

    func testResolvedLocalAndRepositoryAssetsHaveIdenticalStoppingPolicies() async throws {
        let dir = try directory()
        try checkpoint(in: dir)
        let config = #"{"model_type":"qwen3_vl","text_config":{"eos_token_id":[3,4]}}"#
        try Data(config.utf8).write(to: dir.appendingPathComponent("config.json"))
        let factory = ModelLoader(
            registry: ModelRegistry(models: [
                "qwen3_vl": { _ in ResourceModel() }
            ]))
        let tokenizer = ResourceTokenizer(
            tokens: ["<eos>": 7, "<|im_end|>": 10], eosToken: "<eos>")
        for generationOverride in [false, true] {
            if generationOverride {
                try Data(#"{"eos_token_id":[5,6],"stop_strings":"<done>"}"#.utf8)
                    .write(to: dir.appendingPathComponent("generation_config.json"))
            }
            for source in [ModelSource.directory(dir), .repository("renamed/quantized-model")] {
                let loaded = try await factory.loadResources(
                    assets: ResolvedModelAssets(
                        source: source, modelDirectory: dir, tokenizerDirectory: dir),
                    metadata: .init(
                        extraEOSTokens: ["multi-token-stop"], eosTokenIds: [9],
                        reasoningConfig: .init(
                            startDelimiter: "<analysis>", endDelimiter: "</analysis>",
                            promptStrategy: .none)),
                    tokenizerLoader: ResourceTokenizerLoader(tokenizer: tokenizer))
                XCTAssertEqual(
                    loaded.metadata.eosTokenIds,
                    generationOverride ? [5, 6, 7, 9, 10] : [3, 4, 7, 9, 10])
                XCTAssertEqual(loaded.metadata.extraEOSTokens, ["<|im_end|>", "multi-token-stop"])
                XCTAssertEqual(
                    loaded.metadata.stopStrings,
                    generationOverride
                        ? ["<|im_end|>", "multi-token-stop", "<done>"]
                        : ["<|im_end|>", "multi-token-stop"])
                XCTAssertEqual(loaded.metadata.reasoningConfig?.startDelimiter, "<analysis>")
            }
        }
    }

    func testMultiTokenFamilyTerminatorSurvivesWithoutVocabularyEntry() async throws {
        let dir = try directory()
        try checkpoint(in: dir)
        try Data(#"{"model_type":"glm4v","eos_token_id":1}"#.utf8)
            .write(to: dir.appendingPathComponent("config.json"))
        let factory = ModelLoader(
            registry: ModelRegistry(models: ["glm4v": { _ in ResourceModel() }]))
        let loaded = try await factory.loadResources(
            from: .directory(dir), using: ResourceTokenizerLoader())
        XCTAssertTrue(loaded.metadata.stopStrings.contains("</answer>"))
        XCTAssertEqual(loaded.metadata.eosTokenIds, [1])
    }

    func testSidecarQuantizationAndEmbeddedPrecedence() async throws {
        for embedded in [false, true] {
            let dir = try directory()
            try checkpoint(in: dir, quantized: true)
            if embedded {
                try Data(
                    #"{"model_type":"resource_fixture","quantization":{"group_size":32,"bits":4}}"#
                        .utf8
                )
                .write(to: dir.appendingPathComponent("config.json"))
            }
            // An embedded configuration must bypass even an invalid sidecar.
            try Data((embedded ? "invalid" : #"{"group_size":32,"bits":4}"#).utf8)
                .write(to: dir.appendingPathComponent("quantization.json"))
            let resources = try await factory().loadResources(
                from: .directory(dir), using: ResourceTokenizerLoader())
            let context = resources.intoModelContext()
            let model = try XCTUnwrap(context.model as? ResourceModel)
            XCTAssertTrue(model.projection is QuantizedLinear)
            XCTAssertEqual(try Self.greedyTokens(model).count, 4)
        }
    }

    func testSpeculativeLoadingFromSourcesAndAssetsMatchesDirectExecutor() async throws {
        let dir = try directory()
        try checkpoint(in: dir)
        try Data(#"{"eos_token_id":[]}"#.utf8)
            .write(to: dir.appendingPathComponent("generation_config.json"))
        let options = ModelGenerationOptions(
            sampling: .init(temperature: 0), output: .init(maxTokens: 4, format: .tokens))
        let direct = try await factory().load(
            from: .directory(dir), using: ResourceTokenizerLoader())
        var expected = [Int]()
        for try await event in direct.generate(
            input: UserInput(prompt: "fixture"), options: options)
        {
            if case .token(let id) = event { expected.append(id) }
        }
        XCTAssertEqual(expected.count, 4)
        let assets = ResolvedModelAssets(
            source: .directory(dir), modelDirectory: dir, tokenizerDirectory: dir)
        for resolved in [false, true] {
            let loader = ResourceTokenizerLoader()
            let loadedModel: NativDraftModel
            if resolved {
                loadedModel = try await factory().loadSpeculativeExecutor(
                    target: assets, draft: assets, tokenizerLoader: loader)
            } else {
                loadedModel = try await factory().loadSpeculativeExecutor(
                    target: .directory(dir), draft: .directory(dir), using: loader)
            }
            let directories = await loader.directories
            XCTAssertEqual(directories, [dir, dir])
            var actual = [Int]()
            let request = loadedModel.generate(
                input: UserInput(prompt: "fixture"), options: options)
            for try await event in request {
                if case .token(let id) = event { actual.append(id) }
            }
            XCTAssertEqual(actual, expected)
            let summary = try await request.metrics()
            XCTAssertNotNil(summary.speculative)
        }
    }

    func testSpeculativeLoadingRejectsMismatchedTokenizersBeforeWeights() async throws {
        let target = try directory()
        let draft = try directory()
        try Data("target".utf8).write(to: target.appendingPathComponent("tokenizer.json"))
        try Data("draft".utf8).write(to: draft.appendingPathComponent("tokenizer.json"))
        let loader = ResourceTokenizerLoader()
        do {
            _ = try await NativDraftModel.loadModel(
                target: .directory(target), draft: .directory(draft), using: loader)
            XCTFail("Mismatched tokenizers must fail before reading missing model configurations")
        } catch ModelExecutionError.invalidOptions(let message) {
            XCTAssertEqual(message, "target and draft tokenizer assets differ")
        }
        let directories = await loader.directories
        XCTAssertTrue(directories.isEmpty)
    }

    func testTokenizerFailurePropagatesWithoutPublishingExecutor() async throws {
        let dir = try directory()
        try checkpoint(in: dir)
        do {
            _ = try await factory().load(
                from: .directory(dir), using: FailingResourceTokenizerLoader())
            XCTFail("Tokenizer failure must propagate")
        } catch let error as CocoaError {
            XCTAssertEqual(error.code, .fileReadNoPermission)
        }
    }

    func testRequiredConfigurationErrorsRetainSourceIdentity() async throws {
        let dir = try directory()
        for malformed in [false, true] {
            if malformed {
                try Data("invalid".utf8).write(to: dir.appendingPathComponent("config.json"))
            }
            do {
                _ = try await factory().load(
                    assets: ResolvedModelAssets(
                        source: .repository("fixture/model"), modelDirectory: dir,
                        tokenizerDirectory: dir),
                    tokenizerLoader: ResourceTokenizerLoader())
                XCTFail("Required configuration must fail")
            } catch ModelFactoryError.configurationFileError(let file, let name, _) {
                XCTAssertFalse(malformed)
                XCTAssertEqual(file, "config.json")
                XCTAssertEqual(name, "fixture/model")
            } catch ModelFactoryError.configurationDecodingError(let file, let name, _) {
                XCTAssertTrue(malformed)
                XCTAssertEqual(file, "config.json")
                XCTAssertEqual(name, "fixture/model")
            }
        }
    }
}
