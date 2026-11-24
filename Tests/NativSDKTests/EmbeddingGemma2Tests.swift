import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class EmbeddingGemma2Tests: XCTestCase {
    private func configurationObject() -> [String: Any] {
        [
            "model_type": "embedding_gemma2",
            "text_config": [
                "model_type": "embedding_gemma2_text", "vocab_size": 64, "hidden_size": 32,
                "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 8, "hidden_size_per_layer_input": 16,
                "embedding_dim": 16, "sliding_window": 2, "max_position_embeddings": 512,
                "layer_types": ["sliding_attention", "full_attention"],
                "per_layer_config": ["01": ["head_dim": 16, "num_key_value_heads": 1]],
            ],
            "vision_config": [
                "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
                "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 4,
                "patch_size": 2, "pooling_kernel_size": 2, "position_embedding_size": 16,
                "default_output_length": 4, "use_clipped_linears": false,
            ],
            "audio_config": [
                "hidden_size": 16, "num_hidden_layers": 1, "num_attention_heads": 2,
                "subsampling_conv_channels": [4, 2], "output_proj_dims": 16,
                "attention_chunk_size": 4, "attention_context_left": 4, "conv_kernel_size": 3,
                "input_feature_size": 128, "use_clipped_linears": false,
            ],
            "image_token_id": 54, "audio_token_id": 55, "video_token_id": 56,
            "boi_token_id": 57, "eoi_token_id": 58, "boa_token_id": 59, "eoa_token_index": 60,
            "vision_soft_tokens_per_image": 4,
        ]
    }
    private func configuration() throws -> EmbeddingGemma2Configuration {
        try JSONDecoder().decode(
            EmbeddingGemma2Configuration.self,
            from: JSONSerialization.data(withJSONObject: configurationObject()))
    }
    private func model() throws -> EmbeddingGemma2Model {
        let model = try EmbeddingGemma2Model(configuration())
        // Deterministic in-memory weights keep sensitivity checks repeatable without a checkpoint.
        let weights = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().sorted { $0.0 < $1.0 }
                .enumerated().map { index, parameter in
                    let (key, value) = parameter
                    var tensor =
                        sin(arange(value.size).asType(.float32) * 0.17 + Float(index) * 0.13)
                        .reshaped(value.shape) * 0.1
                    if key.hasSuffix("layer_scalar")
                        || (key.hasSuffix("weight") && key.contains("norm"))
                    {
                        tensor = tensor + 1
                    }
                    return (key, tensor)
                })
        try model.update(parameters: .unflattened(weights), verify: [.all])
        eval(model)
        return model
    }
    private func vector(_ result: EmbeddingOutput) throws -> MLXArray {
        guard case .pooled(let array) = result else {
            throw ModelExecutionError.invalidOptions("Expected pooled embeddings")
        }
        return array
    }
    private func close(
        _ actual: MLXArray, _ expected: MLXArray, file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertTrue(
            allClose(actual, expected, rtol: 3e-3, atol: 5e-4).item(Bool.self),
            "Maximum error: \(abs(actual - expected).max().item(Float.self))", file: file,
            line: line)
    }

    func testNormalizedEmbeddingsAcrossTextImageAudioVideoAndComposedInputs() throws {
        let model = try model()
        let pixels = arange(96).asType(.float32).reshaped(1, 3, 4, 8) / 96
        let frames = arange(48).asType(.float32).reshaped(1, 3, 4, 4) / 48
        let features = sin(arange(16 * 128).asType(.float32)).reshaped(1, 16, 128)
        let examples: [(String, LMInput)] = [
            ("text", LMInput(tokens: MLXArray([[2, 3, 4, 5]]))),
            (
                "image",
                LMInput(
                    text: .init(tokens: MLXArray([[2, 54, 54, 3]])),
                    image: .init(pixels: pixels))
            ),
            (
                "video",
                LMInput(
                    text: .init(tokens: MLXArray([[2, 56, 3]])),
                    video: .init(pixels: frames))
            ),
            (
                "audio",
                LMInput(
                    text: .init(tokens: MLXArray([[2, 55, 55, 55, 55, 3]])),
                    audio: .init(features: features))
            ),
            (
                "composed",
                LMInput(
                    text: .init(tokens: MLXArray([[2, 54, 54, 3, 55, 55, 55, 55, 56, 4]])),
                    image: .init(pixels: pixels), video: .init(pixels: frames),
                    audio: .init(features: features))
            ),
        ]
        for (name, input) in examples {
            let result = try vector(model.embed(input))
            XCTAssertEqual(result.shape, [1, 16], name)
            XCTAssertTrue(all(isFinite(result)).item(Bool.self), name)
            XCTAssertEqual(sum(square(result)).item(Float.self), 1, accuracy: 1e-5, name)
            close(try vector(model.embed(input)), result)
        }
    }

    func testMediaContentChangesEmbeddings() throws {
        let model = try model()
        for (name, first, second) in [
            (
                "image",
                LMInput(
                    text: .init(tokens: MLXArray([[2, 54, 3]])),
                    image: .init(pixels: MLXArray.zeros([1, 3, 4, 4]))),
                LMInput(
                    text: .init(tokens: MLXArray([[2, 54, 3]])),
                    image: .init(pixels: arange(48).asType(.float32).reshaped(1, 3, 4, 4) / 48))
            ),
            (
                "video",
                LMInput(
                    text: .init(tokens: MLXArray([[2, 56, 3]])),
                    video: .init(pixels: MLXArray.zeros([1, 3, 4, 4]))),
                LMInput(
                    text: .init(tokens: MLXArray([[2, 56, 3]])),
                    video: .init(pixels: arange(48).asType(.float32).reshaped(1, 3, 4, 4) / 48))
            ),
            (
                "audio",
                LMInput(
                    text: .init(tokens: MLXArray([[2, 55, 55, 55, 55, 3]])),
                    audio: .init(features: MLXArray.zeros([1, 16, 128]))),
                LMInput(
                    text: .init(tokens: MLXArray([[2, 55, 55, 55, 55, 3]])),
                    audio: .init(
                        features: sin(arange(16 * 128).asType(.float32)).reshaped(1, 16, 128)))
            ),
        ] {
            let a = try vector(model.embed(first))
            let b = try vector(model.embed(second))
            XCTAssertGreaterThan(abs(a - b).max().item(Float.self), 1e-6, name)
        }
    }

    func testExplicitPositionsAffectTokenEmbeddingsAndRejectInvalidShapes() throws {
        let model = try model()
        let ids = MLXArray([[2, 3, 4, 5]])
        let embeddings = model.languageModel.embedTokens(ids)
        let valid = MLXArray.ones(ids.shape, dtype: .bool)
        let standard = try model.languageModel(embeddings, valid: valid)
        let positioned = try model.languageModel(
            embeddings, valid: valid, positions: MLXArray([[1, 3, 6, 10]]))
        XCTAssertEqual(positioned.shape, [1, 4, 16])
        XCTAssertTrue(all(isFinite(positioned)).item(Bool.self))
        XCTAssertGreaterThan(abs(standard - positioned).max().item(Float.self), 1e-6)
        XCTAssertThrowsError(
            try model.languageModel(embeddings, valid: valid, positions: MLXArray([[1, 3]])))
    }

    func testBidirectionalAttentionPaddingAndInclusiveLocalWindow() throws {
        let model = try model()
        let ids = MLXArray([[2, 3, 4, 5]])
        let plain = try vector(model.embed(LMInput(tokens: ids)))
        let padded = try vector(
            model.embed(
                LMInput(
                    tokens: MLXArray([[0, 2, 3, 4, 5, 0]]), mask: MLXArray([[0, 1, 1, 1, 1, 0]]))))
        close(plain, padded)
        let first = try model.languageModel(
            model.languageModel.embedTokens(ids), valid: MLXArray.ones(ids.shape, dtype: .bool))
        let changed = try model.languageModel(
            model.languageModel.embedTokens(MLXArray([[2, 3, 4, 6]])),
            valid: MLXArray.ones(ids.shape, dtype: .bool))
        XCTAssertGreaterThan(abs(first[0, 0] - changed[0, 0]).max().item(Float.self), 1e-6)
        // Remove the full layer's token mixing so the first layer's exact local boundary is observable.
        let projection = model.languageModel.layers[1].attention.oProj
        try projection.update(
            parameters: .unflattened(["weight": MLXArray.zeros(projection.weight.shape)]),
            verify: [.all])
        func firstToken(_ values: [[Int]]) throws -> MLXArray {
            let ids = MLXArray(values)
            return try model.languageModel(
                model.languageModel.embedTokens(ids), valid: MLXArray.ones(ids.shape, dtype: .bool))[
                    0, 0]
        }
        let base = try firstToken([[2, 3, 4, 5]])
        XCTAssertGreaterThan(
            abs(base - (try firstToken([[2, 3, 6, 5]]))).max().item(Float.self), 1e-6)
        close(base, try firstToken([[2, 3, 4, 6]]))
    }

    func testConfigurationValidationAndPerLayerOverrides() throws {
        let config = try configuration()
        XCTAssertEqual(config.text.perLayerConfiguration["1"]?["head_dim"], 16)
        XCTAssertEqual(try model().languageModel.layers[1].attention.kvHeads, 1)
        let original = configurationObject()
        for (key, value) in [
            ("num_attention_heads", 0 as Any), ("hidden_size_per_layer_input", -1 as Any),
            ("layer_types", ["sliding_attention"] as Any), ("hidden_activation", "relu" as Any),
            ("per_layer_config", ["01": ["head_dim": 3]] as Any),
        ] {
            var object = original
            var text = try XCTUnwrap(object["text_config"] as? [String: Any])
            text[key] = value
            object["text_config"] = text
            XCTAssertThrowsError(
                try JSONDecoder().decode(
                    EmbeddingGemma2Configuration.self,
                    from: JSONSerialization.data(withJSONObject: object)), key)
        }
    }

    func testMissingMediaAndDisabledTowersAreRejected() throws {
        let model = try model()
        XCTAssertThrowsError(try model.embed(LMInput(tokens: MLXArray([[2, 54, 3]]))))
        XCTAssertThrowsError(
            try model.embed(
                LMInput(
                    text: .init(tokens: MLXArray([[2, 3]])),
                    image: .init(pixels: MLXArray.zeros([1, 3, 4, 4])))))
        var object = configurationObject()
        object["vision_config"] = NSNull()
        object["audio_config"] = NSNull()
        let reduced = try EmbeddingGemma2Model(
            JSONDecoder().decode(
                EmbeddingGemma2Configuration.self,
                from: JSONSerialization.data(withJSONObject: object)))
        let weights = reduced.sanitize(
            weights: Dictionary(uniqueKeysWithValues: model.parameters().flattened()))
        XCTAssertFalse(
            weights.keys.contains { $0.hasPrefix("vision_tower.") || $0.hasPrefix("audio_tower.") })
        try reduced.update(parameters: .unflattened(weights), verify: [.all])
        close(
            try vector(reduced.embed(LMInput(tokens: MLXArray([[2, 3]])))),
            try vector(model.embed(LMInput(tokens: MLXArray([[2, 3]])))))
        XCTAssertThrowsError(
            try reduced.embed(
                LMInput(
                    text: .init(tokens: MLXArray([[2, 54, 3]])),
                    image: .init(pixels: MLXArray.zeros([1, 3, 4, 4])))))
    }

    private func processor(maximumLength: Int = 512, timestamps: Bool = false) throws
        -> EmbeddingGemma2Processor
    {
        let settings: [String: JSONValue] = [
            "image_processor": .object(["max_soft_tokens": .int(4)]),
            "video_processor": .object([
                "max_soft_tokens": .int(4), "add_timestamps": .bool(timestamps),
            ]),
            "feature_extractor": .object([
                "feature_size": .int(128), "frame_length": .int(320), "hop_length": .int(160),
                "sampling_rate": .int(16000),
            ]),
        ]
        return try EmbeddingGemma2Processor(
            tokenizer: GemmaEmbeddingTokenizer(), model: configuration(), settings: settings,
            maximumLength: maximumLength)
    }
    private var image: CIImage {
        CIImage(color: CIColor(red: 1, green: 0, blue: 0)).cropped(
            to: CGRect(x: 0, y: 0, width: 8, height: 4))
    }

    func testProcessorExpandsAndInterleavesAllModalities() async throws {
        let processor = try processor(timestamps: true)
        let waveform = sin(arange(3200).asType(.float32) * 0.01)
        let input = UserInput(
            prompt: .text("<|audio|> then <|image|> then <|video|>"),
            images: [.ciImage(image)],
            videos: [.frames([.init(image: .ciImage(image), timeStamp: .zero)])],
            audios: [.array(waveform)])
        let prepared = try await processor.prepare(input: input)
        let ids = prepared.text.tokens.asArray(Int.self)
        let audioStart = try XCTUnwrap(ids.firstIndex(of: 59))
        let imageStart = try XCTUnwrap(ids.firstIndex(of: 57))
        XCTAssertLessThan(audioStart, imageStart)
        XCTAssertTrue(ids.contains(56))
        let encoder = try model()
        let audioFeatures = try encoder.audioFeatures(XCTUnwrap(prepared.audio))
        XCTAssertEqual(ids.filter { $0 == 55 }.count, audioFeatures.dim(0))
        let result = try vector(encoder.embed(prepared))
        XCTAssertEqual(result.shape, [1, 16])
        XCTAssertTrue(all(isFinite(result)).item(Bool.self))
        XCTAssertEqual(sum(square(result)).item(Float.self), 1, accuracy: 1e-5)
        let textOnly = try await processor.prepare(input: UserInput(prompt: .text("text")))
        XCTAssertNil(textOnly.image)
    }

    func testProcessorRejectsMissingInputsEmptyTextAndTokenOverflow() async throws {
        let processor = try processor(maximumLength: 3)
        for input in [
            UserInput(prompt: .text("")), UserInput(prompt: .text("<|image|>")),
            UserInput(prompt: .text("text"), images: [.ciImage(image)]),
        ] {
            do {
                _ = try await processor.prepare(input: input)
                XCTFail("Expected invalid input")
            } catch {}
        }
    }

    func testVideoSamplingUniformAndTruncateStrategies() throws {
        for (fps, strategy, expected) in [
            (Double(2) as Double?, "uniform", [0, 4, 8]),
            (Double(2) as Double?, "truncate", [0, 2, 4]),
            (nil as Double?, "uniform", [0, 5, 10]),
            (nil as Double?, "truncate", [0, 1, 2]),
        ] {
            XCTAssertEqual(
                try EmbeddingGemma2Processor.sampleIndices(
                    totalFrames: 11, sourceFPS: 4, duration: 2.75, targetFPS: fps,
                    maximumFrames: 3, strategy: strategy), expected)
        }
    }

    func testQuantizedCheckpointRoundtripAndMatryoshkaDimensions() async throws {
        for bits in [4, 6, 8] {
            let root = FileManager.default.temporaryDirectory.appendingPathComponent(
                UUID().uuidString)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            defer { try? FileManager.default.removeItem(at: root) }
            let encoder = try model()
            try quantizeByParent(model: encoder) { _, module in
                guard let linear = module as? Linear, linear.weight.dim(1) % 32 == 0 else {
                    return nil
                }
                return (32, bits, .affine)
            }
            var config = configurationObject()
            config["quantization"] = ["group_size": 32, "bits": bits]
            try JSONSerialization.data(withJSONObject: config).write(
                to: root.appendingPathComponent("config.json"))
            try save(
                arrays: Dictionary(uniqueKeysWithValues: encoder.parameters().flattened()),
                url: root.appendingPathComponent("model.safetensors"))
            try Data(
                #"{"feature_extractor":{"feature_size":128},"image_processor":{"max_soft_tokens":4},"video_processor":{"max_soft_tokens":4}}"#
                    .utf8
            ).write(to: root.appendingPathComponent("processor_config.json"))
            let loaded = try await NativEmbeddingModel.loadModel(
                from: .directory(root), using: GemmaEmbeddingTokenizerLoader())
            let expected = try vector(encoder.embed(LMInput(tokens: MLXArray([[2, 3, 4]]))))
            let result = try await loaded.embed(input: .text("text")).result()
            guard case .vector(let values) = result.embeddings else {
                return XCTFail("Expected vector")
            }
            close(MLXArray(values).reshaped(1, -1), expected)
            let prefix = try await loaded.embed(input: .text("text"), dimensions: 8).result()
            guard case .vector(let shortened) = prefix.embeddings else {
                return XCTFail("Expected vector")
            }
            close(MLXArray(shortened).reshaped(1, -1), normalizeEmbeddings(expected[0..., ..<8]))
            let prepared = try await loaded.prepare(
                input: UserInput(
                    prompt: .text("<|image|><|audio|><|video|>"),
                    images: [.ciImage(image)],
                    videos: [.frames([.init(image: .ciImage(image), timeStamp: .zero)])],
                    audios: [.array(sin(arange(3200).asType(.float32) * 0.01))]))
            let media = try await loaded.embed(prepared: prepared).result()
            guard case .vector(let values) = media.embeddings else {
                return XCTFail("Expected composed media vector")
            }
            XCTAssertEqual(values.count, 16)
            XCTAssertTrue(values.allSatisfy(\.isFinite))
            XCTAssertEqual(values.reduce(0) { $0 + $1 * $1 }, 1, accuracy: 1e-5)
            for dimensions in [0, 17] {
                do {
                    _ = try await loaded.embed(input: .text("text"), dimensions: dimensions)
                        .result()
                    XCTFail("Expected invalid dimensions")
                } catch {}
            }
        }
    }

}

private struct GemmaEmbeddingTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer { GemmaEmbeddingTokenizer() }
}
private struct GemmaEmbeddingTokenizer: Tokenizer {
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        let markers = ["<|image|>": 54, "<|audio|>": 55, "<|video|>": 56]
        var remaining = text[...]
        var ids: [Int] = []
        while !remaining.isEmpty {
            if let marker = markers.first(where: { remaining.hasPrefix($0.key) }) {
                ids.append(marker.value)
                remaining = remaining.dropFirst(marker.key.count)
            } else {
                ids.append(3)
                remaining = remaining.dropFirst()
            }
        }
        return markers.keys.contains(where: { text.contains($0) }) ? ids : [2, 3, 4]
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "text" }
    func convertTokenToId(_ token: String) -> Int? { nil }
    func convertIdToToken(_ id: Int) -> String? { nil }
    func applyChatTemplate(
        messages: [[String: any Sendable]], tools: [[String: any Sendable]]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] { [2, 3, 4] }
}
