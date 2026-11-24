import CoreImage
import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct Parse35Tokenizer: Tokenizer {
    let specials = [
        "</s>": 2, "<s>": 0, "<predict_bbox>": 7, "<predict_classes>": 8, "<output_markdown>": 9,
        "<predict_text_in_pic>": 10,
    ]
    var bosToken: String? { "<s>" }
    var eosToken: String? { "</s>" }
    var unknownToken: String? { nil }
    func convertTokenToId(_ token: String) -> Int? { specials[token] }
    func convertIdToToken(_ id: Int) -> String? { specials.first { $0.value == id }?.key }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        precondition(!addSpecialTokens)
        var text = text[...]
        var ids = [Int]()
        while !text.isEmpty {
            if let key = specials.keys.first(where: { text.hasPrefix($0) }) {
                ids.append(specials[key]!)
                text = text.dropFirst(key.count)
            } else {
                ids.append(Int(text.removeFirst().asciiValue!))
            }
        }
        return ids
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func applyChatTemplate(
        messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw VLMError.processing("Parse control prompts must not use chat templates")
    }
}
private struct Parse35TokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer { Parse35Tokenizer() }
}

final class NemotronParse35Tests: XCTestCase {
    private func configuration(limit: Int = 128) -> Data {
        Data(
            """
            {"model_type":"nemotron_parse","architectures":["RadioMBartPrefixForCausalLM"],"eos_token_id":2,"max_sequence_length":\(limit),
            "vision_config":{"hidden_size":32,"num_heads":4,"mlp_ratio":2,"num_layers":1,"neck_dim":32},
            "decoder":{"d_model":32,"decoder_attention_heads":4,"self_attention_kv_heads":2,"decoder_ffn_dim":64,"decoder_layers":2,"vocab_size":128}}
            """.utf8)
    }
    private func model(limit: Int = 128) throws -> NemotronParse35 {
        MLXRandom.seed(47)
        let model = try XCTUnwrap(
            ModelRegistry.default.createModel(
                configuration: configuration(limit: limit), modelType: "nemotron_parse")
                as? NemotronParse35)
        model.train(false)
        return model
    }
    private func input() -> LMInput {
        LMInput(
            text: .init(tokens: MLXArray([2, 0, 7, 8, 9, 10], [1, 6])),
            image: .init(
                pixels: cos(MLXArray(0 ..< 16 * 768).asType(.float32) / 32).reshaped(16, 768),
                frames: [THW(1, 2, 8)], contentSizes: [CGSize(width: 65, height: 17)]))
    }
    private func image(width: Int = 65, height: Int = 17) -> CIImage {
        let bytes = (0 ..< width * height).flatMap { i -> [UInt8] in
            [UInt8(i % width * 3 % 256), UInt8(i / width * 5 % 256), 23, 255]
        }
        return CIImage(
            bitmapData: Data(bytes), bytesPerRow: width * 4,
            size: CGSize(width: width, height: height), format: .RGBA8,
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
    }
    private func processor() throws -> NemotronParse35Processor {
        NemotronParse35Processor(
            try JSONDecoder().decode(
                NemotronParseProcessorConfiguration.self,
                from: Data(#"{"final_size":[64,128]}"#.utf8)), tokenizer: Parse35Tokenizer())
    }
    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: url) }
        return url
    }
    private func logits(_ model: NemotronParse35, chunk: Int?) throws -> MLXArray {
        let cache = try model.newCache(configuration: nil)
        guard
            case .logits(let output) = try model.prepare(
                input(), cache: cache, state: nil, prefill: .init(stepSize: chunk))
        else { throw VLMError.processing("Expected prefix logits") }
        XCTAssertEqual(cache.first?.offset, 11)
        return concatenated([
            output.logits[0, -1], model(MLXArray([3], [1, 1]), cache: cache)[0, 0],
        ])
    }

    func testSourceArchitectureDispatchAndConvertedAlias() throws {
        let model = try model()
        XCTAssertEqual(model.config.vision.clsTokens, 2)
        XCTAssertEqual(model.config.vision.registerTokens, 8)
        XCTAssertEqual(model.config.vision.summaryIndices, [0, 1])
        XCTAssertEqual(model.config.kvHeads, 2)
        XCTAssertEqual(model.config.text.maxPositionEmbeddings, 16384)
        XCTAssertEqual(model.languageModel.layers[0].attention.key.weight.shape, [16, 32])
        XCTAssertFalse(
            model.parameters().flattened().contains {
                $0.0.contains("embed_positions") || $0.0.contains("encoder_attn")
                    || $0.0.contains("lm_head")
            })
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: configuration()) as? [String: Any])
        object.removeValue(forKey: "architectures")
        XCTAssertTrue(
            try ModelRegistry.default.createModel(
                configuration: JSONSerialization.data(withJSONObject: object),
                modelType: "nemotron_parse") is NemotronParse)
        object["model_type"] = "nemotron_parse_v3_5"
        object["text_config"] = object.removeValue(forKey: "decoder")
        XCTAssertTrue(
            try ModelRegistry.default.createModel(
                configuration: JSONSerialization.data(withJSONObject: object),
                modelType: "nemotron_parse_v3_5") is NemotronParse35)
    }

    func testVisualPrefixAndTextAreCausalAndCachesIncludeBoth() throws {
        let model = try model()
        let (embeddings, prompt) = try model.inputEmbeddings(input())
        XCTAssertEqual(embeddings.shape, [1, 11, 32])
        XCTAssertEqual(prompt, 6)
        let expected = try logits(model, chunk: nil)
        for chunk in [1, 3, 4] {
            XCTAssertLessThan(
                abs(try logits(model, chunk: chunk) - expected).max().item(Float.self), 3e-3)
        }
        let baseline = model.languageModel(embeddings, cache: nil)
        let changed = embeddings + 0
        changed[0, 4] = changed[0, 4] + MLXArray(0 ..< 32).asType(.float32) / 10
        let modified = model.languageModel(changed, cache: nil)
        XCTAssertTrue(arrayEqual(baseline[0, ..<4], modified[0, ..<4]).item(Bool.self))
        XCTAssertGreaterThan(abs(baseline[0, 4...] - modified[0, 4...]).max().item(Float.self), 0)
    }

    func testContentPositionsSupportPartialAndSinglePatchAxes() throws {
        let model = try model()
        for (height, width) in [(17, 65), (1, 49), (33, 127)] {
            let rows = (height + 15) / 16
            let columns = (width + 63) / 64 * 4
            let positions = model.visionTower.positions(
                rows: rows, columns: columns, contentHeight: height, contentWidth: width,
                dtype: .bfloat16)
            XCTAssertEqual(positions.shape, [1, rows * columns, 32])
            XCTAssertEqual(positions.dtype, .bfloat16)
            XCTAssertTrue(all(isFinite(positions)).item(Bool.self))
        }
        let weights = NemotronParse35Vision.resizeWeights(source: 128, target: 7, area: false)
        XCTAssertLessThan(abs(weights.sum(axis: 1) - 1).max().item(Float.self), 1e-6)
    }

    func testPackedPageGeometryAndRawControlTokens() async throws {
        let processor = try processor()
        for prompt in ["", NemotronParse35Processor.defaultPrompt] {
            let result = try await processor.prepare(
                input: UserInput(chat: [.user(prompt, images: [.ciImage(image())])]))
            XCTAssertEqual(result.text.tokens.asArray(Int.self), [2, 0, 7, 8, 9, 10])
            XCTAssertEqual(result.image?.pixels.shape, [16, 768])
            XCTAssertEqual(result.image?.frames?.map { [$0.t, $0.h, $0.w] }, [[1, 2, 8]])
            XCTAssertEqual(result.image?.contentSizes, [CGSize(width: 65, height: 17)])
            let white = (Float(1) - processor.config.imageMean[0]) / processor.config.imageStd[0]
            XCTAssertEqual(result.image!.pixels[7, 0].item(Float.self), white, accuracy: 1e-6)
        }
        let shrunk = try processor.preprocess(image(width: 289, height: 137), processing: .init())
        XCTAssertEqual(shrunk.contentSizes, [CGSize(width: 128, height: 60)])
        XCTAssertEqual(shrunk.frames?.map { [$0.h, $0.w] }, [[4, 8]])
    }

    func testGeometryAndContextLimitFailuresDoNotAdvanceTheCache() throws {
        let model = try model(limit: 10)
        let cache = try model.newCache(configuration: nil)
        XCTAssertThrowsError(try model.prepare(input(), cache: cache, state: nil, prefill: .init()))
        XCTAssertEqual(cache[0].offset, 0)
        XCTAssertThrowsError(
            try model.visionTower(.init(pixels: input().image!.pixels, frames: [THW(1, 2, 8)])))
        let short = try self.model(limit: 12)
        let shortCache = try short.newCache(configuration: nil)
        _ = try short.prepare(input(), cache: shortCache, state: nil, prefill: .init())
        _ = short(MLXArray([3], [1, 1]), cache: shortCache)
        try short.checkInferenceError()
        _ = short(MLXArray([4], [1, 1]), cache: shortCache)
        XCTAssertThrowsError(try short.checkInferenceError())
        XCTAssertEqual(shortCache[0].offset, 12)
    }

    func testSourceAndNativeWeightsRoundTripAndSkipMTP() throws {
        let reference = try model()
        let url = try directory()
        let expected = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
        var source = [String: MLXArray]()
        for (key, value) in expected {
            var name = key
            var array = value
            if key.hasPrefix("vision_tower.blocks.") {
                name =
                    "encoder.model_encoder.radio_model.model.blocks."
                    + key.dropFirst("vision_tower.blocks.".count)
            } else if key.hasPrefix("vision_tower.neck.") {
                name = "encoder." + key.dropFirst("vision_tower.neck.".count)
                if key.hasSuffix("conv1.weight") { array = value.transposed(0, 2, 1) }
                if key.hasSuffix("conv2.weight") { array = value.transposed(0, 3, 1, 2) }
            } else if key == "vision_tower.patch_embed.weight" {
                name = "encoder.model_encoder.radio_model.model.patch_generator.embedder.weight"
            } else if key == "vision_tower.pos_embed" {
                name = "encoder.model_encoder.radio_model.model.patch_generator.pos_embed"
            } else if key == "vision_tower.cls_token" {
                name = "encoder.model_encoder.radio_model.model.patch_generator.cls_token.token"
            } else if key.hasPrefix("language_model.") {
                name = "decoder." + key.dropFirst(15)
            }
            source[name] = array
        }
        source["decoder.extra_heads.0.weight"] = MLXArray.zeros([1])
        source["decoder.extra_proj.weight"] = MLXArray.zeros([1])
        source["lm_head.weight"] = reference.languageModel.embedTokens.weight
        let first = reference.sanitize(weights: source)
        let second = reference.sanitize(weights: first)
        XCTAssertEqual(Set(first.keys), Set(expected.keys))
        for (key, value) in expected {
            XCTAssertTrue(arrayEqual(first[key]!, value).item(Bool.self), key)
            XCTAssertTrue(arrayEqual(second[key]!, value).item(Bool.self), key)
        }
        for weights in [source, expected] {
            try MLX.save(arrays: weights, url: url.appendingPathComponent("model.safetensors"))
            let reloaded = try model()
            try CheckpointLoader.load(modelDirectory: url, model: reloaded)
            XCTAssertTrue(
                arrayEqual(try logits(reference, chunk: 3), try logits(reloaded, chunk: 3)).item(
                    Bool.self))
        }
    }

    func testQuantizationKeepsVisionDenseAndReloadsDecoder() throws {
        let reference = try model()
        let source = try directory()
        let destination = source.appendingPathComponent("quantized")
        try configuration().write(to: source.appendingPathComponent("config.json"))
        let weights = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
        try MLX.save(
            arrays: weights, metadata: ["format": "mlx"],
            url: source.appendingPathComponent("model.safetensors"))
        let result = try ModelConversion.convert(
            source: source, destination: destination, quantization: .init(groupSize: 32, bits: 4))
        XCTAssertGreaterThan(result.quantizedMatrixCount, 0)
        let arrays = try safetensorWeightURLs(in: destination).reduce(into: [String: MLXArray]()) {
            values, file in values.merge(try MLX.loadArrays(url: file)) { _, new in new }
        }
        for (key, value) in weights where key.hasPrefix("vision_tower.") {
            XCTAssertTrue(arrayEqual(arrays[key]!, value).item(Bool.self), key)
        }
        let base = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: Data(contentsOf: destination.appendingPathComponent("config.json")))
        let loaded = try model()
        try CheckpointLoader.load(
            modelDirectory: destination, model: loaded,
            perLayerQuantization: base.perLayerQuantization)
        XCTAssertTrue(all(isFinite(try logits(loaded, chunk: 3))).item(Bool.self))
    }

    func testLoaderChoosesPrefixProcessorFromOriginalArchitecture() async throws {
        let url = try directory()
        let reference = try model()
        try configuration().write(to: url.appendingPathComponent("config.json"))
        try Data(#"{"processor_class":"NemotronParseProcessor","final_size":[64,128]}"#.utf8).write(
            to: url.appendingPathComponent("preprocessor_config.json"))
        try MLX.save(
            arrays: Dictionary(uniqueKeysWithValues: reference.parameters().flattened()),
            metadata: ["format": "mlx"], url: url.appendingPathComponent("model.safetensors"))
        let loaded = try await NativModel.loadModel(
            from: .directory(url), using: Parse35TokenizerLoader())
        try await loaded.withModel { _, context in
            XCTAssertTrue(context.model is NemotronParse35)
            XCTAssertTrue(context.processor is NemotronParse35Processor)
        }
    }

    func testRealCheckpointWhenRequested() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["NEMOTRON_PARSE35_MODEL"] else {
            throw XCTSkip("Set NEMOTRON_PARSE35_MODEL for full-checkpoint validation")
        }
        let image = URL(fileURLWithPath: try XCTUnwrap(env["NEMOTRON_PARSE35_IMAGE"]))
        let expected = try String(
            contentsOfFile: XCTUnwrap(env["NEMOTRON_PARSE35_REFERENCE_TEXT"]), encoding: .utf8)
        let loaded = try await NativModel.loadModel(from: .directory(URL(fileURLWithPath: path)))
        let request = loaded.generate(
            input: UserInput(chat: [
                .user(NemotronParse35Processor.defaultPrompt, images: [.url(image)])
            ]),
            options: .init(
                prefill: .init(stepSize: 128),
                sampling: .init(temperature: 0, repetition: .init(1.1, contextSize: 0)),
                output: .init(maxTokens: 512)))
        let output = try await request.response()
        print("Parse 3.5: \(output)")
        XCTAssertEqual(output, expected)
        let metrics = try await request.metrics()
        XCTAssertEqual(metrics.stopReason, .stop)
    }
}
