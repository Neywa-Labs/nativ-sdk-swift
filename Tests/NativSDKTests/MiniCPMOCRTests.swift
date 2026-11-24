import CoreImage
import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct OCRTestTokenizer: Tokenizer {
    let specials = [
        "<|image_pad|>": 48, "<|video_pad|>": 49, "<image>": 50, "</image>": 51,
        "<slice>": 52, "</slice>": 53, "<|im_end|>": 54, "<|im_start|>": 55,
        "<unk>": 56, "<image_id>": 57, "</image_id>": 58,
    ]
    var bosToken: String? { "<|im_start|>" }
    var eosToken: String? { "<|im_end|>" }
    var unknownToken: String? { "<unk>" }
    func convertTokenToId(_ token: String) -> Int? { specials[token] }
    func convertIdToToken(_ id: Int) -> String? { specials.first { $0.value == id }?.key }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        var remaining = text[...]
        var result = [Int]()
        while !remaining.isEmpty {
            if let key = specials.keys.first(where: { remaining.hasPrefix($0) }) {
                result.append(specials[key]!)
                remaining = remaining.dropFirst(key.count)
            } else {
                result.append(1000 + Int(remaining.removeFirst().asciiValue!))
            }
        }
        return result
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String {
        tokenIds.map { convertIdToToken($0) ?? String(UnicodeScalar($0 - 1000)!) }.joined()
    }
    func applyChatTemplate(
        messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        let prompt =
            messages.map {
                "<|im_start|>\($0["role"] as? String ?? "user")\n\($0["content"] as? String ?? "")<|im_end|>\n"
            }.joined() + "<|im_start|>assistant\n"
            + ((additionalContext?["enable_thinking"] as? Bool ?? true)
                ? "thinking" : "not thinking")
        return encode(text: prompt, addSpecialTokens: false)
    }
}

final class MiniCPMOCRTests: XCTestCase {
    private func configuration() -> Data {
        Data(
            """
            {
              "model_type": "minicpm_ocr", "insert_layer_id": 0,
              "image_token_id": 48, "video_token_id": 49,
              "image_start_id": 50, "image_end_id": 51,
              "slice_start_id": 52, "slice_end_id": 53,
              "text_config": {
                "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
                "full_attention_interval": 2, "num_attention_heads": 2,
                "num_key_value_heads": 2, "head_dim": 8, "vocab_size": 64,
                "linear_num_value_heads": 2, "linear_num_key_heads": 2,
                "linear_key_head_dim": 32, "linear_value_head_dim": 8,
                "linear_conv_kernel_dim": 3, "rms_norm_eps": 0.00001,
                "tie_word_embeddings": true,
                "rope_parameters": {"mrope_section": [1, 0, 0], "partial_rotary_factor": 0.25}
              },
              "vision_config": {
                "hidden_size": 16, "intermediate_size": 32, "num_hidden_layers": 2,
                "num_attention_heads": 2, "image_size": 8, "patch_size": 2
              }
            }
            """.utf8)
    }

    private func model() throws -> MiniCPMV {
        let model = try XCTUnwrap(
            ModelRegistry.default.createModel(
                configuration: configuration(), modelType: "minicpm_ocr") as? MiniCPMV)
        model.train(false)
        return model
    }

    private func input(mode: String = "4x", padding: Bool = false) -> LMInput {
        let count = mode == "4x" ? 16 : 4
        let unpadded = [1, 50] + Array(repeating: 48, count: count) + [51, 2]
        let tokens = padding ? [0, 0] + unpadded + [0, 0] : unpadded
        let start = padding ? 4 : 2
        let pixels = cos(MLXArray(0 ..< 768).asType(.float32) / 32)
        return LMInput(
            text: .init(
                tokens: MLXArray(tokens, [1, tokens.count]),
                mask: padding
                    ? MLXArray(
                        [0, 0] + Array(repeating: 1, count: unpadded.count) + [0, 0],
                        [1, tokens.count]) : nil),
            image: .init(pixels: pixels, frames: [THW(1, 8, 8)]),
            video: .init(pixels: MLXArray([start, start + count], [1, 2])))
    }

    func testPaddingDoesNotChangeRecurrentPrefillOrDecode() throws {
        let model = try model()
        for mode in ["4x", "16x"] {
            var outputs = [MLXArray]()
            for padding in [false, true] {
                let cache = try model.newCache(configuration: nil)
                _ = try model.prepare(
                    input(mode: mode, padding: padding), cache: cache, state: nil,
                    prefill: .init(stepSize: 3))
                outputs.append(model(MLXArray([3], [1, 1]), cache: cache))
            }
            XCTAssertLessThan(abs(outputs[0] - outputs[1]).max().item(Float.self), 2e-4, mode)
        }
    }

    func testNormSanitizationPreservesPrecisionAndIsIdempotent() throws {
        let model = try model()
        let source = MLXArray.full([16], values: MLXArray(Float(0.001))).asType(.bfloat16)
        let key = "language_model.model.layers.0.input_layernorm.weight"
        let first = model.sanitize(weights: [
            "model.language_model.layers.0.input_layernorm.weight": source,
            "model.language_model.layers.0.linear_attn.conv1d.weight": MLXArray.zeros([144, 1, 3]),
        ])
        let shifted = try XCTUnwrap(first[key])
        XCTAssertEqual(shifted.dtype, .float32)
        XCTAssertTrue(arrayEqual(shifted, source.asType(.float32) + 1).item(Bool.self))
        XCTAssertTrue(
            arrayEqual(try XCTUnwrap(model.sanitize(weights: first)[key]), shifted).item(Bool.self))
        XCTAssertTrue(
            arrayEqual(
                try XCTUnwrap(model.sanitize(weights: first, metadata: ["format": "mlx"])[key]),
                shifted
            ).item(Bool.self))
        let norms = model.leafModules().flattened().compactMap { $0.1 as? RMSNorm }
        XCTAssertFalse(norms.isEmpty)
        XCTAssertTrue(norms.allSatisfy { $0 is MiniCPMOCRRMSNorm })
        for norm in norms {
            XCTAssertEqual(
                norm(MLXArray.ones([1, norm.weight.size], dtype: .bfloat16)).dtype, .bfloat16)
        }
    }

    func testBF16ActivationsKeepFloat32RecurrentState() throws {
        let model = try model()
        let parameters = model.parameters().flattened().map { key, value in
            (
                key,
                value.asType(
                    MiniCPMV.normWeightSuffixes.contains(where: key.hasSuffix)
                        ? .float32 : .bfloat16)
            )
        }
        try model.update(parameters: ModuleParameters.unflattened(parameters), verify: .all)
        let cache = try model.newCache(configuration: nil)
        guard
            case .logits(let output) = try model.prepare(
                input(), cache: cache, state: nil,
                prefill: .init(stepSize: 3))
        else { return XCTFail("Expected prefill logits") }
        XCTAssertEqual(output.logits.dtype, .bfloat16)
        let recurrent = try XCTUnwrap(cache.first as? MambaCache)
        XCTAssertEqual(recurrent[1]?.dtype, .float32)
        let decoded = model(MLXArray([3], [1, 1]), cache: cache)
        XCTAssertEqual(decoded.dtype, .bfloat16)
        XCTAssertEqual(recurrent[1]?.dtype, .float32)
        XCTAssertTrue(all(isFinite(decoded)).item(Bool.self))
    }

    func testQuantizedMLXCheckpointReloadPreservesShiftedNormsAndImageLogits() throws {
        let source = try model()
        let paths = Set(
            source.leafModules().flattened().compactMap { path, module -> String? in
                guard let linear = module as? Linear, linear.weight.dim(-1).isMultiple(of: 32)
                else { return nil }
                return path
            })
        XCTAssertFalse(paths.isEmpty)
        quantize(model: source) { path, _ in paths.contains(path) ? (32, 4, .affine) : nil }
        let weights = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try MLX.save(
            arrays: weights, metadata: ["format": "mlx"],
            url: directory.appendingPathComponent("model.safetensors"))
        let loaded = MiniCPMV(
            try JSONDecoder().decode(MiniCPMVConfiguration.self, from: configuration()))
        let settings = Dictionary(
            uniqueKeysWithValues: paths.map {
                ($0, BaseConfiguration.QuantizationOption.quantize(.init(groupSize: 32, bits: 4)))
            })
        try CheckpointLoader.load(
            modelDirectory: directory, model: loaded,
            perLayerQuantization: .init(perLayerQuantization: settings))
        let actual = Dictionary(uniqueKeysWithValues: loaded.parameters().flattened())
        XCTAssertEqual(Set(weights.keys), Set(actual.keys))
        for (key, value) in weights {
            XCTAssertTrue(arrayEqual(try XCTUnwrap(actual[key]), value).item(Bool.self), key)
        }
        var logits = [MLXArray]()
        for model in [source, loaded] {
            guard
                case .logits(let output) = try model.prepare(
                    input(), cache: model.newCache(configuration: nil),
                    state: nil, prefill: .init(stepSize: 3))
            else { return XCTFail("Expected image logits") }
            logits.append(output.logits)
        }
        XCTAssertLessThan(abs(logits[0] - logits[1]).max().item(Float.self), 2e-4)
    }

    func testPositionBucketsMatchFloat32SearchBoundaries() {
        for count in [28, 38, 70] {
            let expected = (0 ..< count).map { index in
                (1 ..< 70).filter { Float(Double($0) / 70) <= Float(index) / Float(count) }.count
            }
            XCTAssertEqual(miniCPMVPositionBuckets(count: count, gridSide: 70), expected)
        }
    }

    private func image(width: Int, height: Int, dense: Bool = false) -> CIImage {
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        if dense {
            for y in stride(from: 10, to: 390, by: 10) {
                for x in stride(from: 10, to: 340, by: 8) {
                    for row in y ..< y + 5 {
                        for column in x ..< x + 3 {
                            let offset = (row * width + column) * 4
                            rgba[offset] = 0
                            rgba[offset + 1] = 0
                            rgba[offset + 2] = 0
                        }
                    }
                }
            }
        }
        return CIImage(
            bitmapData: Data(rgba), bytesPerRow: width * 4,
            size: CGSize(width: width, height: height), format: .RGBA8,
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
    }

    private func processor() throws -> MiniCPMVProcessor {
        let config = try JSONDecoder().decode(
            MiniCPMVProcessorConfiguration.self,
            from: Data(#"{"processor_class":"MiniCPMOcrProcessor"}"#.utf8))
        return MiniCPMVProcessor(config, tokenizer: OCRTestTokenizer())
    }

    func testProcessorAlignsCropGeometryAndPlaceholderSpans() async throws {
        let processor = try processor()
        let cases = [
            (mode: "4x", width: 301, height: 173, adaptive: false, dense: false),
            (mode: "16x", width: 301, height: 173, adaptive: false, dense: false),
            (mode: "4x", width: 40, height: 20, adaptive: true, dense: false),
            (mode: "16x", width: 40, height: 20, adaptive: true, dense: false),
            (mode: "4x", width: 700, height: 400, adaptive: true, dense: true),
            (mode: "16x", width: 700, height: 400, adaptive: true, dense: true),
        ]
        for item in cases {
            let context: [String: any Sendable] = [
                "downsample_mode": item.mode, "adaptive_small_image_upsize": item.adaptive,
                "adaptive_tile_upsize": item.adaptive, "disable_small_image_upsize": item.adaptive,
            ]
            let prepared = try await processor.prepare(
                input: UserInput(
                    chat: [
                        .user(
                            "Read",
                            images: [
                                .ciImage(
                                    image(
                                        width: item.width, height: item.height, dense: item.dense
                                    ))
                            ])
                    ],
                    additionalContext: context))
            let frames = try XCTUnwrap(prepared.image?.frames)
            XCTAssertFalse(frames.isEmpty)
            let divisor = item.mode == "4x" ? 4 : 16
            let counts = frames.map { $0.h * $0.w / divisor }
            let tokens = prepared.text.tokens.asArray(Int.self)
            XCTAssertEqual(tokens.filter { $0 == 48 }.count, counts.reduce(0, +))
            let bounds = try XCTUnwrap(prepared.video?.pixels).asArray(Int.self)
            XCTAssertEqual(
                stride(from: 0, to: bounds.count, by: 2).map { bounds[$0 + 1] - bounds[$0] }, counts
            )
            XCTAssertEqual(
                try miniCPMOCRDownsampleMode(
                    frames: frames, bounds: prepared.video?.pixels), item.mode)
        }
    }

    func testTemplateHistoryDefaultsAndRequestOptionsDoNotLeak() async throws {
        let processor = try processor()
        let picture = image(width: 40, height: 20)
        let first = try await processor.prepare(
            input: UserInput(
                chat: [
                    .system("Instructions"), .user("First"), .assistant("Answer"),
                    .user("Read", images: [.ciImage(picture)]),
                ],
                additionalContext: ["adaptive_small_image_upsize": true, "downsample_mode": "16x"]))
        let decoded = OCRTestTokenizer().decode(
            tokenIds: first.text.tokens.asArray(Int.self), skipSpecialTokens: false)
        for text in ["Instructions", "First", "Answer", "Read", "not thinking"] {
            XCTAssertTrue(decoded.contains(text))
        }
        let again = try await processor.prepare(
            input: UserInput(chat: [.user("Read", images: [.ciImage(picture)])]))
        XCTAssertEqual(
            try miniCPMOCRDownsampleMode(frames: again.image!.frames!, bounds: again.video?.pixels),
            "4x")
        XCTAssertGreaterThan(again.image!.pixels.size, first.image!.pixels.size)
    }

    func testAdaptiveDetectorUsesEightConnectedComponentsAndInvertsDarkPages() {
        let options = MiniCPMOCRImageOptions()
        var gray = [UInt8](repeating: 255, count: 10000)
        for y in stride(from: 5, to: 95, by: 10) {
            for x in stride(from: 5, to: 95, by: 10) {
                for delta in 0 ..< 3 { gray[(y + delta) * 100 + x + delta] = 0 }
            }
        }
        XCTAssertEqual(options.tier(gray: gray, width: 100, height: 100), .full)
        XCTAssertEqual(options.tier(gray: gray.map { 255 - $0 }, width: 100, height: 100), .full)
        XCTAssertEqual(
            options.tier(gray: [UInt8](repeating: 255, count: 800), width: 40, height: 20), .tiny)
        XCTAssertEqual(
            options.tier(gray: [UInt8](repeating: 255, count: 10000), width: 100, height: 100),
            .native)
    }

    func testComposedProcessorSettingsAndInvalidCompression() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try Data(#"{"processor_class":"MiniCPMOcrProcessor"}"#.utf8).write(
            to: directory.appendingPathComponent("processor_config.json"))
        try Data(
            #"{"adaptive_small_image_upsize":true,"scale_resolution":112,"downsample_mode":"16x"}"#
                .utf8
        ).write(to: directory.appendingPathComponent("preprocessor_config.json"))
        let (data, type) = try await ModelLoader().loadProcessorConfig(
            from: directory,
            modelType: "minicpm_ocr", configData: Data(#"{"vision_config":{"patch_size":14}}"#.utf8)
        )
        let config = try JSONDecoder().decode(MiniCPMVProcessorConfiguration.self, from: data)
        XCTAssertEqual(type, "MiniCPMOcrProcessor")
        XCTAssertEqual(config.downsampleMode, "4x")
        XCTAssertEqual(config.scaleResolution, 112)
        XCTAssertTrue(config.ocrImageOptions.adaptiveSmallImageUpsize)
        XCTAssertThrowsError(try config.applyingOCROptions(["downsample_mode": "8x"]))
        XCTAssertThrowsError(try config.applyingOCROptions(["adaptive_upsize_font_px": Double.nan]))
        XCTAssertThrowsError(try config.applyingOCROptions(["adaptive_small_image_upsize": "yes"]))
        XCTAssertThrowsError(
            try miniCPMOCRDownsampleMode(frames: [THW(1, 8, 8)], bounds: MLXArray([2, 5], [1, 2])))
    }

    func testRealCheckpointWhenRequested() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["MINICPM_OCR_MODEL"] else {
            throw XCTSkip("Set MINICPM_OCR_MODEL for a local full-checkpoint OCR smoke test")
        }
        let imagePath = try XCTUnwrap(environment["MINICPM_OCR_IMAGE"])
        let expected = try XCTUnwrap(environment["MINICPM_OCR_EXPECTED_TEXT"])
        let model = try await NativModel.loadModel(from: .directory(URL(fileURLWithPath: path)))
        for mode in ["4x", "16x"] {
            let request = model.generate(
                input: UserInput(
                    chat: [
                        .user(
                            "Convert this page to markdown.",
                            images: [.url(URL(fileURLWithPath: imagePath))])
                    ],
                    additionalContext: ["downsample_mode": mode]),
                options: .init(
                    prefill: .init(stepSize: 128), sampling: .init(temperature: 0),
                    output: .init(maxTokens: 256)))
            let text = try await request.response()
            XCTAssertTrue(text.contains(expected), "\(mode): \(text)")
            let summary = try await request.metrics()
            XCTAssertEqual(summary.stopReason, .stop, text)
        }
    }
}
