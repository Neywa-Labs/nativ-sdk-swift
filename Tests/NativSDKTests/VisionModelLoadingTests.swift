import CoreImage
import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct VisionFixtureTokenizer: Tokenizer {
    var compact = false
    var imageMarker = "[IMG]"
    var addBOS = false
    let special = [
        "<imgpad>": 14, "<img>": 15, "</img>": 16,
        "<|image_pad|>": 14, "<|image_start|>": 15, "<|image_end|>": 16,
        "<|vision_start|>": 12, "<|vision_end|>": 13, "<|im_start|>": 17, "<|im_end|>": 18,
        "[IMG]": 10,
    ]
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func convertTokenToId(_ token: String) -> Int? { special[token] }
    func convertIdToToken(_ id: Int) -> String? { special.first { $0.value == id }?.key }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var remaining = text[...]
        var ids = addSpecialTokens && addBOS ? [1] : []
        while !remaining.isEmpty {
            if let token = special.keys.first(where: { remaining.hasPrefix($0) }) {
                ids.append(special[token]!)
                remaining = remaining.dropFirst(token.count)
            } else {
                let value = Int(remaining.removeFirst().unicodeScalars.first!.value)
                ids.append(compact ? 20 + value % 40 : 256 + value)
            }
        }
        return ids
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { id in
            if id >= 256 { return String(UnicodeScalar(id - 256)!) }
            if id == 10 { return "[IMG]" }
            return "x"
        }.joined()
    }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let text = messages.map { message in
            let content: String
            if let value = message["content"] as? String {
                content = value
            } else {
                content = (message["content"] as? [[String: String]] ?? []).map {
                    $0["type"] == "image" ? imageMarker : ($0["text"] ?? "")
                }.joined()
            }
            return "\(message["role"] as? String ?? "user"):\(content)\n"
        }.joined()
        return encode(text: text, addSpecialTokens: false)
    }
}

private struct VisionFixtureTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer {
        let config = try String(
            contentsOf: directory.appendingPathComponent("config.json"), encoding: .utf8)
        return VisionFixtureTokenizer(
            compact: true,
            imageMarker: config.contains("indic_ocr")
                ? "<|vision_start|><|image_pad|><|vision_end|>" : "[IMG]")
    }
}

final class VisionModelLoadingTests: XCTestCase {
    private func indicConfig() throws -> Data {
        try ModelTestConfigurations.configuration("indic_ocr")
    }

    func testIndicOCRRejectsCombinedPipelinesAndPreservesVocabularyConventions() throws {
        var config = try XCTUnwrap(
            JSONSerialization.jsonObject(with: indicConfig()) as? [String: Any])
        for key in [
            "image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id",
            "vocab_size",
        ] {
            config.removeValue(forKey: key)
        }
        let defaults = try IndicOCR.configuration(
            from: JSONSerialization.data(withJSONObject: config))
        // These defaults differ from Qwen3.5 and determine embedding placement / M-RoPE.
        XCTAssertEqual(defaults.imageTokenIndex, 262155)
        XCTAssertEqual(defaults.videoTokenIndex, 262156)
        XCTAssertEqual(defaults.visionStartTokenId, 262153)
        for key in ["layout_config", "ocr_config"] {
            var combined = config
            combined[key] = [:] as [String: Any]
            XCTAssertThrowsError(
                try ModelRegistry.default.createModel(
                    configuration: JSONSerialization.data(withJSONObject: combined),
                    modelType: "indic_ocr"))
        }
        XCTAssertThrowsError(
            try ModelRegistry.default.createModel(
                configuration: Data(
                    #"{"model_type":"indic_ocr","stages":{"ocr":"weights/ocr"}}"#.utf8),
                modelType: "indic_ocr"))
    }

    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: url) }
        return url
    }

    private func processor(_ name: String, directory: URL) async throws -> any UserInputProcessor {
        let data = try ModelTestConfigurations.configuration(name)
        let (config, type) = try await ModelLoader().loadProcessorConfig(
            from: directory, modelType: name, configData: data)
        return try ModelRegistry.default.createProcessor(
            configuration: config, processorType: type,
            tokenizer: VisionFixtureTokenizer(addBOS: name == "mistral_large3"))
    }

    func testCheckpointsLoadAndGenerateWithImages() async throws {
        for name in ["llmjpvl", "mistral_large3", "indic_ocr"] {
            let model: any LanguageModel
            let data: Data
            if name == "indic_ocr" {
                data = try indicConfig()
                model = try ModelRegistry.default.createModel(configuration: data, modelType: name)
            } else {
                data = try ModelTestConfigurations.configuration(name)
                model = try ModelTestConfigurations.model(name)
            }
            let url = try directory()
            try data.write(to: url.appendingPathComponent("config.json"))
            if name == "indic_ocr" {
                try Data(#"{"min_pixels":1024,"max_pixels":1024}"#.utf8).write(
                    to: url.appendingPathComponent("preprocessor_config.json"))
            }
            try MLX.save(
                arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
                url: url.appendingPathComponent("model.safetensors"))
            let loadedModel = try await NativModel.loadModel(
                from: .directory(url), using: VisionFixtureTokenizerLoader())
            let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 32, height: 32))
            let request = loadedModel.generate(
                input: UserInput(prompt: "Read", images: [.ciImage(image)]),
                options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 2)))
            for try await _ in request {}
            let summary = try await request.metrics()
            XCTAssertGreaterThan(summary.generatedTokenCount, 0, name)
        }
    }

    func testLLMjpProcessorPreservesHistoryAndAlignsImageTokens() async throws {
        let url = try directory()
        // Processor and image sidecars must both survive composition.
        try Data(#"{"max_dynamic_patch":2,"use_thumbnail":true}"#.utf8).write(
            to: url.appendingPathComponent("processor_config.json"))
        try Data(#"{"image_mean":[0,0,0],"image_std":[1,1,1]}"#.utf8).write(
            to: url.appendingPathComponent("preprocessor_config.json"))
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 32, height: 16))
        let processor = try await processor("llmjpvl", directory: url)
        let input = try await processor.prepare(
            input: UserInput(chat: [
                .system("Custom instructions"),
                .user("First question", images: [.ciImage(image)]),
                .assistant("Earlier answer"), .user("Follow-up"),
            ]))
        let ids = input.text.tokens.asArray(Int.self)
        let decoded = VisionFixtureTokenizer().decode(tokenIds: ids, skipSpecialTokens: false)
        for text in ["Custom instructions", "First question", "Earlier answer", "Follow-up"] {
            XCTAssertTrue(decoded.contains(text), text)
        }
        let pixels = try XCTUnwrap(input.image?.pixels)
        let expectedTiles = 3
        XCTAssertEqual(pixels.dim(0), expectedTiles)
        XCTAssertEqual(ids.filter { $0 == 14 }.count, 4 * expectedTiles)
        XCTAssertEqual(pixels[0, 0, 0, 0].item(Float.self), 1, accuracy: 1e-3)
        XCTAssertEqual(pixels[0, 1, 0, 0].item(Float.self), 0, accuracy: 1e-3)
    }

    func testMistralResizesToMergedPatchGrid() async throws {
        let url = try directory()
        try Data(#"{"size":{"longest_edge":256},"patch_size":{"height":99,"width":99}}"#.utf8)
            .write(
                to: url.appendingPathComponent("preprocessor_config.json"))
        let processor = try await processor("mistral_large3", directory: url)
        // Exceeds the old hardcoded 24-patch cap; odd dimensions need merge alignment.
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 127, height: 61))
        let input = try await processor.prepare(
            input: UserInput(prompt: "Read", images: [.ciImage(image)]))
        XCTAssertEqual(input.image?.pixels.shape, [1, 3, 64, 128])
        XCTAssertEqual(
            input.text.tokens.asArray(Int.self).filter { $0 != 10 },
            VisionFixtureTokenizer().encode(text: "user:Read\n", addSpecialTokens: false))
        XCTAssertEqual(input.text.tokens.asArray(Int.self).filter { $0 == 10 }.count, 8 * 16)
    }
}
