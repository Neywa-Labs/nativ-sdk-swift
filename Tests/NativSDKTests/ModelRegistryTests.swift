import Foundation
import XCTest

@testable import NativSDK

private struct RegistryTestTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [] }
}

private enum RegistryValidationError: Error, Equatable {
    case invalidConfiguration
}

private struct ValidatedRegistryConfiguration: Decodable, Sendable, ModelConfigurationValidating {
    let valid: Bool

    func validateModelConfiguration() throws {
        if !valid { throw RegistryValidationError.invalidConfiguration }
    }
}

final class ModelRegistryTests: XCTestCase {

    func testCreatorsValidateConfigurationBeforeConstruction() throws {
        let invalid = Data("{valid: false}".utf8)
        let model = ModelRegistry.model(ValidatedRegistryConfiguration.self) {
            _ in
            Mistral4(
                .init(
                    hiddenSize: 8, hiddenLayers: 0, intermediateSize: 16,
                    attentionHeads: 2, vocabularySize: 32))
        }
        let processor = ModelRegistry.processor(
            ValidatedRegistryConfiguration.self
        ) {
            _, tokenizer in TextProcessor(.init(), tokenizer: tokenizer)
        }
        XCTAssertThrowsError(try model(invalid)) {
            XCTAssertEqual($0 as? RegistryValidationError, .invalidConfiguration)
        }
        XCTAssertThrowsError(try processor(invalid, RegistryTestTokenizer())) {
            XCTAssertEqual($0 as? RegistryValidationError, .invalidConfiguration)
        }
        let valid = try processor(Data("{valid: true}".utf8), RegistryTestTokenizer())
        XCTAssertTrue(valid is TextProcessor)
        XCTAssertThrowsError(try processor(Data("{valid: 3}".utf8), RegistryTestTokenizer())) {
            XCTAssertTrue($0 is DecodingError)
        }
    }

    func testProcessorSelectionPreservesFilePriorityOverridesAndFallbacks() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let factory = ModelLoader()
        let modelData = Data("{model_type: 'diffusion_gemma'}".utf8)
        let fallback = try await factory.loadProcessorConfig(
            from: directory, modelType: "diffusion_gemma", configData: modelData)
        XCTAssertEqual(fallback.0, modelData)
        XCTAssertEqual(fallback.1, "TextProcessor")

        for name in ["falcon_perception", "falcon-perception"] {
            let selection = try await factory.loadProcessorConfig(
                from: directory, modelType: name, configData: modelData)
            XCTAssertEqual(selection.0, modelData)
            XCTAssertEqual(selection.1, "FalconPerceptionProcessor")
        }

        let processorData = Data("{processor_class: 'DeclaredProcessor'}".utf8)
        let processorURL = directory.appendingPathComponent("processor_config.json")
        try processorData.write(to: processorURL)
        let declared = try await factory.loadProcessorConfig(
            from: directory, modelType: "qwen2_vl", configData: modelData)
        XCTAssertEqual(declared.0, processorData)
        XCTAssertEqual(declared.1, "DeclaredProcessor")

        let preprocessorData = Data("{processor_class: 'PreferredProcessor'}".utf8)
        try preprocessorData.write(to: directory.appendingPathComponent("preprocessor_config.json"))
        try Data("invalid".utf8).write(to: processorURL)
        let preferred = try await factory.loadProcessorConfig(
            from: directory, modelType: "qwen2_vl", configData: modelData)
        XCTAssertEqual(preferred.0, preprocessorData)
        XCTAssertEqual(preferred.1, "PreferredProcessor")
        let override = try await factory.loadProcessorConfig(
            from: directory, modelType: "diffusion_gemma", configData: modelData)
        XCTAssertEqual(override.0, preprocessorData)
        XCTAssertEqual(override.1, "DiffusionGemma4Processor")
        try Data("invalid".utf8).write(
            to: directory.appendingPathComponent("preprocessor_config.json"))
        do {
            _ = try await factory.loadProcessorConfig(
                from: directory, modelType: "qwen2_vl", configData: modelData)
            XCTFail("Malformed required processor metadata must fail")
        } catch ModelFactoryError.configurationDecodingError(let filename, _, _) {
            XCTAssertEqual(filename, "preprocessor_config.json")
        }
    }

    func testProcessorSelectionDoesNotInventFallbacksOrIgnoreMalformedMetadata() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        for name in ["mistral3", "gemma4_unified", "unknown_model"] {
            do {
                _ = try await ModelLoader().loadProcessorConfig(
                    from: directory, modelType: name, configData: Data("{}".utf8))
                XCTFail("Missing processor metadata should fail for \(name)")
            } catch ModelFactoryError.configurationFileError(let filename, _, _) {
                XCTAssertEqual(filename, "processor_config.json")
            }
        }
        try Data("{".utf8).write(to: directory.appendingPathComponent("processor_config.json"))
        do {
            _ = try await ModelLoader().loadProcessorConfig(
                from: directory, modelType: "falcon_ocr", configData: Data("{}".utf8))
            XCTFail("Malformed metadata must not activate the missing-file fallback")
        } catch ModelFactoryError.configurationDecodingError(let filename, _, _) {
            XCTAssertEqual(filename, "processor_config.json")
        }
    }

    func testQwen35UsesProcessorClassFromCompanionFile() async throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let preprocessor = Data(#"{"patch_size":16}"#.utf8)
        try preprocessor.write(to: directory.appending(path: "preprocessor_config.json"))
        try Data(#"{"processor_class":"Qwen3VLProcessor"}"#.utf8)
            .write(to: directory.appending(path: "processor_config.json"))
        let selected = try await ModelLoader().loadProcessorConfig(
            from: directory, modelType: "qwen3_5", configData: Data("{}".utf8))
        XCTAssertEqual(selected.0, preprocessor)
        XCTAssertEqual(selected.1, "Qwen3VLProcessor")
    }

    func testUnsupportedTypesFailDirectly() throws {
        let registry = ModelRegistry.default
        XCTAssertThrowsError(try registry.createModel(configuration: Data(), modelType: "missing"))
        {
            guard case ModelFactoryError.unsupportedModelType("missing") = $0 else {
                return XCTFail("Unexpected error: \($0)")
            }
        }
        XCTAssertThrowsError(
            try registry.createProcessor(
                configuration: Data(), processorType: "missing", tokenizer: RegistryTestTokenizer())
        ) {
            guard case ModelFactoryError.unsupportedProcessorType("missing") = $0 else {
                return XCTFail("Unexpected error: \($0)")
            }
        }
    }
}
