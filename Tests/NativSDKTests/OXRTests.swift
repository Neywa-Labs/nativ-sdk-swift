import CoreImage
import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct OXRTestTokenizer: Tokenizer {
    let specials = [
        "<|image_pad|>": 60, "<|vision_start|>": 61, "<|vision_end|>": 62,
        "<|im_start|>": 63, "<|im_end|>": 59, "<|endoftext|>": 58,
    ]
    var bosToken: String? { nil }
    var eosToken: String? { "<|endoftext|>" }
    var unknownToken: String? { nil }
    func convertTokenToId(_ token: String) -> Int? { specials[token] }
    func convertIdToToken(_ id: Int) -> String? { specials.first { $0.value == id }?.key }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var text = text[...]
        var result = [Int]()
        while !text.isEmpty {
            if let key = specials.keys.first(where: { text.hasPrefix($0) }) {
                result.append(specials[key]!)
                text = text.dropFirst(key.count)
            } else {
                result += String(text.removeFirst()).utf8.map { Int($0) + 64 }
            }
        }
        return result
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        var bytes = [UInt8]()
        var result = ""
        for id in tokenIds {
            if let token = convertIdToToken(id) {
                result += String(decoding: bytes, as: UTF8.self) + token
                bytes = []
            } else {
                bytes.append(UInt8(id - 64))
            }
        }
        return result + String(decoding: bytes, as: UTF8.self)
    }
    func applyChatTemplate(
        messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let text =
            messages.map {
                "<|im_start|>\($0["role"] as? String ?? "user")\n\($0["content"] as? String ?? "")<|im_end|>\n"
            }.joined()
            + "<|im_start|>assistant\n"
        return encode(text: text, addSpecialTokens: false)
    }
}

private struct OXRTestTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer { OXRTestTokenizer() }
}

final class OXRTests: XCTestCase {
    private func configuration() -> Data {
        Data(
            """
            {
              "model_type":"oxr", "visual_vocab_size":36, "image_token_id":60,
              "vision_start_token_id":61,"vision_end_token_id":62,"eos_token_id":58,
              "llm_config":{"hidden_size":32,"num_hidden_layers":2,"intermediate_size":64,
                "num_attention_heads":4,"num_key_value_heads":2,"head_dim":8,
                "rms_norm_eps":0.000001,"vocab_size":512,"rope_theta":10000,"tie_word_embeddings":true},
              "vit_config":{"hidden_size":32,"intermediate_size":64,"num_hidden_layers":2,
                "num_attention_heads":4,"patch_size":4,"image_size":16,"hidden_stride":2}
            }
            """.utf8)
    }
    private func model() throws -> OXR {
        MLXRandom.seed(47)
        let model = try XCTUnwrap(
            ModelRegistry.default.createModel(configuration: configuration(), modelType: "oxr")
                as? OXR)
        model.train(false)
        return model
    }
    private func processor() throws -> OXRProcessor {
        OXRProcessor(
            try JSONDecoder().decode(
                OXRProcessorConfiguration.self,
                from: Data(
                    """
                    {"patch_size":4,"hidden_stride":2,"min_pixels":256,"max_pixels":1024}
                    """.utf8)), tokenizer: OXRTestTokenizer())
    }
    private func image(width: Int = 16, height: Int = 16) -> CIImage {
        let bytes = (0 ..< width * height).flatMap { index -> [UInt8] in
            [UInt8(index % width * 7 % 256), UInt8(index / width * 11 % 256), 20, 255]
        }
        return CIImage(
            bitmapData: Data(bytes), bytesPerRow: width * 4,
            size: CGSize(width: width, height: height), format: .RGBA8,
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
    }
    private func input(padding: Bool = false) -> LMInput {
        let raw = [1, 61] + Array(repeating: 60, count: 4) + [62, 2]
        let ids = padding ? [0, 0] + raw + [0] : raw
        return LMInput(
            text: .init(
                tokens: MLXArray(ids, [1, ids.count]),
                mask: padding ? MLXArray(ids.map { $0 == 0 ? 0 : 1 }, [1, ids.count]) : nil),
            image: .init(
                pixels: cos(MLXArray(0 ..< 16 * 48).asType(.float32) / 32).reshaped(16, 48),
                frames: [THW(1, 4, 4)]))
    }
    private func logits(_ model: OXR, input: LMInput, step: Int?) throws -> MLXArray {
        let cache = try model.newCache(configuration: nil)
        guard
            case .logits(let output) = try model.prepare(
                input, cache: cache, state: nil, prefill: .init(stepSize: step))
        else { throw VLMError.processing("Expected OXR logits") }
        return concatenated([
            output.logits[0, -1], model(MLXArray([3], [1, 1]), cache: cache)[0, 0],
        ])
    }
    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: url) }
        return url
    }

    func testOriginalAndConvertedConfigurationsAndValidation() throws {
        let config = try JSONDecoder().decode(OXRConfiguration.self, from: configuration())
        XCTAssertEqual(config.vision.patchSize, 4)
        XCTAssertEqual(config.text.hiddenSize, 32)
        var data = try XCTUnwrap(
            JSONSerialization.jsonObject(with: configuration()) as? [String: Any])
        data["text_config"] = data.removeValue(forKey: "llm_config")
        data["vision_config"] = data.removeValue(forKey: "vit_config")
        XCTAssertTrue(
            try ModelRegistry.default.createModel(
                configuration: JSONSerialization.data(withJSONObject: data), modelType: "oxr")
                is OXR)
        data["visual_vocab_size"] = 4
        XCTAssertThrowsError(
            try ModelRegistry.default.createModel(
                configuration: JSONSerialization.data(withJSONObject: data), modelType: "oxr"))
    }

    func testBicubicIdentityAndPartitionOfUnity() {
        XCTAssertLessThan(
            abs(oxrBicubicMatrix(input: 8, output: 8) - eye(8)).max().item(Float.self), 1e-6)
        for output in [2, 13, 45] {
            let matrix = oxrBicubicMatrix(input: 32, output: output)
            XCTAssertEqual(matrix.shape, [output, 32])
            XCTAssertLessThan(abs(matrix.sum(axis: 1) - 1).max().item(Float.self), 1e-6)
        }
    }

    func testPackedImagesAreEncodedIndependentlyAndIndicatorsUseReservedRows() throws {
        let model = try model()
        let first = input()
        let pixels = try XCTUnwrap(first.image?.pixels)
        let together = model.vision(
            concatenated([pixels, pixels]), frames: [THW(1, 4, 4), THW(1, 4, 4)])
        let alone = model.vision(pixels, frames: [THW(1, 4, 4)])
        XCTAssertTrue(arrayEqual(together[..<16], alone).item(Bool.self))
        XCTAssertTrue(arrayEqual(together[16...], alone).item(Bool.self))
        let (_, embeddings) = try model.inputEmbeddings(first)
        XCTAssertTrue(
            arrayEqual(embeddings[0, 1], model.visionEmbedding.weight[-4]).item(Bool.self))
        XCTAssertTrue(
            arrayEqual(embeddings[0, 6], model.visionEmbedding.weight[-3]).item(Bool.self))
        XCTAssertTrue(
            arrayEqual(embeddings[0, 0], model.languageModel.model.embedTokens(MLXArray([1]))[0])
                .item(Bool.self))
        let invalid = LMInput(tokens: first.text.tokens)
        XCTAssertThrowsError(
            try model.inputEmbeddings(
                LMInput(text: invalid.text, image: .init(pixels: pixels, frames: [THW(1, 2, 4)]))))
    }

    func testChunkedPaddingAndCachedDecodeAgree() throws {
        let model = try model()
        let expected = try logits(model, input: input(), step: nil)
        XCTAssertTrue(all(isFinite(expected)).item(Bool.self))
        for step in [3, 5] {
            let chunked = try logits(model, input: input(), step: step)
            // Qwen3's vector and matrix attention kernels differ slightly in FP32.
            XCTAssertLessThan(
                abs(chunked - expected).max().item(Float.self), 2e-3)
            XCTAssertTrue(
                arrayEqual(try logits(model, input: input(padding: true), step: step), chunked)
                    .item(Bool.self))
        }
    }

    func testProcessorPreservesTemplateHistoryAndExpandsImageBlocksInOrder() async throws {
        let processor = try processor()
        let prepared = try await processor.prepare(
            input: UserInput(chat: [
                .system("Read documents."), .user("First <image>", images: [.ciImage(image())]),
                .assistant("ok"), .user("<image> Second", images: [.ciImage(image(width: 32))]),
            ]))
        let text = OXRTestTokenizer().decode(
            tokenIds: prepared.text.tokens.asArray(Int.self), skipSpecialTokens: false)
        XCTAssertTrue(text.hasPrefix("<|im_start|>system\nRead documents.<|im_end|>\n"))
        XCTAssertTrue(text.contains("<|im_start|>assistant\nok<|im_end|>\n"))
        XCTAssertTrue(text.hasSuffix("<|im_start|>assistant\n"))
        let frames = try XCTUnwrap(prepared.image?.frames)
        XCTAssertEqual(frames.map { [$0.t, $0.h, $0.w] }, [[1, 4, 4], [1, 4, 8]])
        XCTAssertEqual(prepared.text.tokens.asArray(Int.self).filter { $0 == 60 }.count, 12)
        XCTAssertFalse(text.contains("<image>"))
        do {
            _ = try await processor.prepare(
                input: UserInput(
                    prompt: .messages([["role": "user", "content": "no image placeholder"]]),
                    images: [.ciImage(image())]))
            XCTFail("Require matching image placeholders")
        } catch {}
        let raw = try await processor.prepare(
            input: UserInput(prompt: .text("Read."), images: [.ciImage(image())]))
        XCTAssertEqual(raw.text.tokens.asArray(Int.self).filter { $0 == 60 }.count, 4)
    }

    func testResizeAndPatchOrderAndRequestPixelBudgets() async throws {
        let processor = try processor()
        let (pixels, frame) = try processor.preprocess(image(), processing: .init())
        XCTAssertEqual([frame.t, frame.h, frame.w], [1, 4, 4])
        XCTAssertEqual(pixels.shape, [16, 48])
        let reds = (0 ..< 4).map { pixels[$0, 0].item(Float.self) }
        let expected: [Float] = [0, 28, 0, 28].map { Float($0) / 255 * 2 - 1 }
        for (actual, reference) in zip(reds, expected) {
            XCTAssertEqual(actual, reference, accuracy: 1e-6)
        }
        let original = [UInt8](repeating: 123, count: 7 * 5 * 3)
        XCTAssertEqual(
            Set(
                PillowImageProcessing.resizeRGB(
                    original, width: 7, height: 5, targetWidth: 3, targetHeight: 13)), [123])
        var small = UserInput.Processing()
        small.maxPixels = 64
        small.minPixels = 64
        let (_, smaller) = try processor.preprocess(image(), processing: small)
        XCTAssertEqual([smaller.h, smaller.w], [2, 2])
        let (_, again) = try processor.preprocess(image(), processing: .init())
        XCTAssertEqual([again.h, again.w], [4, 4])
        XCTAssertThrowsError(
            try OXRProcessor.targetSize(
                height: 0, width: 10, factor: 8, minPixels: 64, maxPixels: 128))
    }

    func testSourceAndNativeCheckpointLayoutsRoundTrip() throws {
        let reference = try model()
        let expected = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
        var source = [String: MLXArray]()
        for (name, original) in expected {
            var name = name
            var value = original
            if name.hasPrefix("vision_tower.") {
                name =
                    "visual_tokenizer.vit.vision_model."
                    + name.dropFirst("vision_tower.".count).replacingOccurrences(
                        of: "layers.", with: "encoder.layers.")
                if name.hasSuffix("patch_embedding.weight") { value = value.reshaped(32, 3, 4, 4) }
            } else if name.hasPrefix("vision_head.layers.") {
                name = "visual_tokenizer.head." + name.dropFirst("vision_head.layers.".count)
            } else if name.hasPrefix("vision_embed.") {
                name = "vte." + name.dropFirst("vision_embed.".count)
            } else if name.hasPrefix("language_model.model.") {
                name = "llm.model." + name.dropFirst("language_model.model.".count)
            }
            source[name] = value
        }
        source["llm.lm_head.weight"] = reference.languageModel.model.embedTokens.weight
        source["visual_tokenizer.vit.vision_model.post_layernorm.weight"] = MLXArray.ones([32])
        let sanitized = reference.sanitize(weights: source)
        XCTAssertEqual(Set(sanitized.keys), Set(expected.keys))
        XCTAssertTrue(
            sanitized.allSatisfy {
                arrayEqual(reference.sanitize(weights: sanitized)[$0.key]!, $0.value).item(
                    Bool.self)
            })
        let url = try directory()
        for arrays in [source, expected] {
            try MLX.save(arrays: arrays, url: url.appendingPathComponent("model.safetensors"))
            let loaded = try model()
            try CheckpointLoader.load(modelDirectory: url, model: loaded)
            XCTAssertLessThan(
                abs(
                    try logits(loaded, input: input(), step: 3)
                        - logits(reference, input: input(), step: 3)
                ).max().item(Float.self), 1e-6)
        }
    }

    func testConversionKeepsVisualTokenizerDenseAndReloads() throws {
        let original = try model()
        let source = try directory()
        let destination = source.appendingPathComponent("quantized")
        try configuration().write(to: source.appendingPathComponent("config.json"))
        let weights = Dictionary(uniqueKeysWithValues: original.parameters().flattened())
        try MLX.save(
            arrays: weights, metadata: ["format": "mlx"],
            url: source.appendingPathComponent("model.safetensors"))
        let result = try ModelConversion.convert(
            source: source, destination: destination, quantization: .init(groupSize: 32, bits: 4))
        XCTAssertGreaterThan(result.quantizedMatrixCount, 0)
        let arrays = try safetensorWeightURLs(in: destination).reduce(into: [String: MLXArray]()) {
            result, url in result.merge(try MLX.loadArrays(url: url)) { _, new in new }
        }
        for (key, value) in weights where key.hasPrefix("vision_") {
            XCTAssertTrue(arrayEqual(arrays[key]!, value).item(Bool.self), key)
        }
        XCTAssertFalse(arrays.keys.contains { $0.hasPrefix("vision_") && $0.hasSuffix(".scales") })
        let base = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: Data(contentsOf: destination.appendingPathComponent("config.json")))
        let loaded = try model()
        try CheckpointLoader.load(
            modelDirectory: destination, model: loaded,
            perLayerQuantization: base.perLayerQuantization)
        XCTAssertTrue(all(isFinite(try logits(loaded, input: input(), step: 3))).item(Bool.self))
    }

    func testNativeLoaderOverridesGenericSiglipProcessorAndUsesModelGeometry() async throws {
        let url = try directory()
        let original = try model()
        try configuration().write(to: url.appendingPathComponent("config.json"))
        try Data(#"{"processor_class":"SiglipProcessor","min_pixels":256,"max_pixels":1024}"#.utf8)
            .write(to: url.appendingPathComponent("preprocessor_config.json"))
        try MLX.save(
            arrays: Dictionary(uniqueKeysWithValues: original.parameters().flattened()),
            metadata: ["format": "mlx"], url: url.appendingPathComponent("model.safetensors"))
        let loaded = try await NativModel.loadModel(
            from: .directory(url), using: OXRTestTokenizerLoader())
        try await loaded.withModel { _, context in
            let processor = try XCTUnwrap(context.processor as? OXRProcessor)
            XCTAssertEqual(processor.config.patchSize, 4)
            XCTAssertEqual(processor.config.hiddenStride, 2)
            XCTAssertTrue(context.metadata.eosTokenIds.contains(59))
        }
    }

    func testRealCheckpointWhenRequested() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["OXR_MODEL"] else {
            throw XCTSkip("Set OXR_MODEL for local full-checkpoint validation")
        }
        let file = URL(fileURLWithPath: try XCTUnwrap(env["OXR_REFERENCE_CASES"]))
        struct Case: Decodable {
            let image: String
            let prompt: String
            let text: String
        }
        let cases = try JSONDecoder().decode([Case].self, from: Data(contentsOf: file))
        let model = try await NativModel.loadModel(from: .directory(URL(fileURLWithPath: path)))
        for row in cases {
            let image = file.deletingLastPathComponent().appendingPathComponent(row.image)
                .appendingPathExtension("png")
            let request = model.generate(
                input: UserInput(chat: [.user(row.prompt, images: [.url(image)])]),
                options: .init(
                    prefill: .init(stepSize: 128), sampling: .init(temperature: 0),
                    output: .init(maxTokens: 512)))
            let text = try await request.response()
            print("OXR \(row.prompt): \(text)")
            XCTAssertEqual(text, row.text)
            let metrics = try await request.metrics()
            XCTAssertEqual(metrics.stopReason, .stop)
        }
    }
}
