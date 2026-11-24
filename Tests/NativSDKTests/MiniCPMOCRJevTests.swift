import CoreImage
import CryptoKit
import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct JevTestTokenizer: Tokenizer {
    let specials = [
        "<|image_pad|>": 48, "<|video_pad|>": 49, "<|vision_start|>": 50,
        "<|vision_end|>": 51, "<|im_start|>": 52, "<|im_end|>": 53,
        "<think>": 54, "</think>": 55,
    ]
    var bosToken: String? { nil }
    var eosToken: String? { nil }
    var unknownToken: String? { nil }
    func convertTokenToId(_ token: String) -> Int? { specials[token] }
    func convertIdToToken(_ id: Int) -> String? { specials.first { $0.value == id }?.key }
    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        precondition(!addSpecialTokens)
        var text = text[...]
        var tokens = [Int]()
        while !text.isEmpty {
            if let special = specials.keys.first(where: { text.hasPrefix($0) }) {
                tokens.append(specials[special]!)
                text = text.dropFirst(special.count)
            } else {
                tokens += String(text.removeFirst()).utf8.map { Int($0) + 64 }
            }
        }
        return tokens
    }
    func decode(tokenIds: [Int], skipSpecialTokens: Bool) -> String { "" }
    func applyChatTemplate(
        messages: [Message], tools: [ToolSpec]?, additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        throw VLMError.processing("Jev must not apply a chat template")
    }
}

private struct JevTestTokenizerLoader: TokenizerLoader {
    func load(from directory: URL) async throws -> any Tokenizer { JevTestTokenizer() }
}

final class MiniCPMOCRJevTests: XCTestCase {
    private func configuration() -> Data {
        Data(
            """
            {
              "model_type": "qwen3_5", "problem_type": "single_label_classification",
              "architectures": ["Qwen3_5ForConditionalGeneration"], "pad_token_id": 0,
              "image_token_id": 48, "video_token_id": 49, "vision_start_token_id": 50,
              "vision_end_token_id": 51,
              "text_config": {
                "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2,
                "full_attention_interval": 2, "num_attention_heads": 2, "num_key_value_heads": 2,
                "head_dim": 16, "vocab_size": 512, "linear_num_value_heads": 2,
                "linear_num_key_heads": 2, "linear_key_head_dim": 32, "linear_value_head_dim": 16,
                "linear_conv_kernel_dim": 3, "rms_norm_eps": 0.00001,
                "rope_parameters": {"mrope_section": [1, 0, 1], "partial_rotary_factor": 0.25}
              },
              "vision_config": {
                "model_type": "qwen3_5_vision", "hidden_size": 32, "intermediate_size": 64,
                "depth": 1, "num_heads": 2, "out_hidden_size": 32, "patch_size": 2,
                "spatial_merge_size": 2, "temporal_patch_size": 2, "num_position_embeddings": 16
              }
            }
            """.utf8)
    }

    private func model() throws -> MiniCPMOCRJev {
        let model = try MiniCPMOCRJev(
            JSONDecoder().decode(MiniCPMOCRJevConfiguration.self, from: configuration()))
        model.train(false)
        return model
    }

    private func processor() throws -> MiniCPMOCRJevProcessor {
        let config = try JSONDecoder().decode(
            Qwen3VLProcessorConfiguration.self,
            from: Data(
                """
                {"image_mean":[0.5,0.5,0.5],"image_std":[0.5,0.5,0.5],"min_pixels":64,
                 "max_pixels":1024,"patch_size":2,"temporal_patch_size":2,"merge_size":2,
                 "image_processor_type":"Qwen2VLImageProcessor"}
                """.utf8))
        return MiniCPMOCRJevProcessor(config: config, tokenizer: JevTestTokenizer())
    }

    private func image(_ color: CIColor, width: Int = 8, height: Int = 8) -> CIImage {
        CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
    }

    private func input(padding: Int = 0) -> LMInput {
        let ids = [1, 50, 48, 48, 48, 48, 51, 2, 50, 48, 48, 48, 48, 51, 3]
        let tokens = padding < 0 ? [0, 0] + ids : padding > 0 ? ids + [0, 0] : ids
        return LMInput(
            text: .init(
                tokens: MLXArray(tokens, [1, tokens.count]),
                mask: MLXArray(tokens.map { $0 == 0 ? 0 : 1 }, [1, tokens.count])),
            image: .init(
                pixels: cos(MLXArray(0 ..< 768).asType(.float32) / 32).reshaped(32, 24),
                frames: [THW(1, 4, 4), THW(1, 4, 4)]))
    }

    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: url) }
        return url
    }

    func testClassificationDispatchAndHead() throws {
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: configuration()) as? [String: Any])
        XCTAssertTrue(MiniCPMOCRJevConfiguration.matches(configuration()))
        XCTAssertThrowsError(
            try ModelRegistry.default.createModel(
                configuration: configuration(), modelType: "qwen3_5",
                weightNames: ["model.language_model.norm.weight"]))
        object.removeValue(forKey: "problem_type")
        XCTAssertFalse(
            MiniCPMOCRJevConfiguration.matches(try JSONSerialization.data(withJSONObject: object)))
        object["architectures"] = ["Qwen3_5ForSequenceClassification"]
        XCTAssertTrue(
            MiniCPMOCRJevConfiguration.matches(try JSONSerialization.data(withJSONObject: object)))
        object["num_labels"] = 3
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                MiniCPMOCRJevConfiguration.self,
                from: JSONSerialization.data(withJSONObject: object)))
        let model = try model()
        XCTAssertEqual(model.scoreHead.weight.shape, [2, 32])
        XCTAssertNil(model.backbone.languageModel.lmHead)
        XCTAssertFalse(model.parameters().flattened().contains { $0.0.contains("lm_head") })
        guard case .skip = model.quantizationPredicate(path: "score", module: model.scoreHead)
        else { return XCTFail("Keep score head unquantized") }
    }

    func testImageAndTextPaddingPoolTheLastValidToken() throws {
        let model = try model()
        let expected = try model.logits(input())
        XCTAssertEqual(expected.shape, [1, 2])
        XCTAssertTrue(all(isFinite(expected)).item(Bool.self))
        for padding in [-1, 1] {
            XCTAssertLessThan(
                abs(try model.logits(input(padding: padding)) - expected).max().item(Float.self),
                2e-4)
        }
        let ids = MLXArray([1, 2, 3, 0, 0, 0, 0, 1, 2, 3], [2, 5])
        let actual = try model.logits(LMInput(tokens: ids, mask: ids .!= 0))
        XCTAssertLessThan(abs(actual[0] - actual[1]).max().item(Float.self), 2e-4)
        XCTAssertThrowsError(
            try model.logits(LMInput(tokens: MLXArray.zeros([1, 3], dtype: .int32))))
        XCTAssertThrowsError(
            try model.logits(
                LMInput(tokens: MLXArray([1, 2, 3], [1, 3]), mask: MLXArray([1, 0, 1], [1, 3]))))
        XCTAssertThrowsError(try model.logits(LMInput(tokens: ids, mask: MLXArray.ones([1, 5]))))
    }

    func testNormOffsetsStayInFloat32AndSanitizationIsIdempotent() throws {
        let model = try model()
        let source = MLXArray.full([32], values: MLXArray(Float(0.001))).asType(.bfloat16)
        let result = model.sanitize(weights: [
            "model.language_model.norm.weight": source,
            "model.language_model.layers.0.linear_attn.conv1d.weight": MLXArray.zeros([160, 1, 3]),
            "score.weight": model.scoreHead.weight,
        ])
        let key = "backbone.language_model.model.norm.weight"
        let shifted = try XCTUnwrap(result[key])
        XCTAssertEqual(shifted.dtype, .float32)
        XCTAssertTrue(arrayEqual(shifted, source.asType(.float32) + 1).item(Bool.self))
        for metadata in [[:], ["format": "mlx"]] {
            let sanitized = model.sanitize(weights: result, metadata: metadata)
            XCTAssertTrue(arrayEqual(try XCTUnwrap(sanitized[key]), shifted).item(Bool.self))
            XCTAssertTrue(
                arrayEqual(try XCTUnwrap(sanitized["score.weight"]), model.scoreHead.weight).item(
                    Bool.self))
        }
        let norms = model.backbone.languageModel.namedModules().compactMap { $0.1 as? RMSNorm }
        XCTAssertFalse(norms.isEmpty)
        XCTAssertTrue(norms.allSatisfy { $0 is MiniCPMOCRRMSNorm })
        XCTAssertTrue(
            norms.allSatisfy {
                $0(MLXArray.ones([1, $0.weight.size], dtype: .bfloat16)).dtype == .bfloat16
            })
    }

    func testPromptMatchesTrainedBytesAndPreservesCandidateWhitespace() {
        let prompt = MiniCPMOCRJevProcessor.prompt(text: "{text}")
        XCTAssertEqual(
            Insecure.MD5.hash(data: Data(prompt.utf8)).map { String(format: "%02x", $0) }.joined(),
            "9aab68e8ba8ab4da17ba9752bddb199c")
        for text in ["", "{text}", "\n A\n  B \t\n"] {
            XCTAssertTrue(
                MiniCPMOCRJevProcessor.prompt(text: text).contains(
                    "OCR文本：\n\(text)\n<|vision_start|>"))
        }
        XCTAssertTrue(prompt.hasSuffix("<|im_start|>assistant\n<think>\n\n</think>\n\n"))
    }

    func testProcessorPreservesImageOrderAndPatchSpans() async throws {
        let processor = try processor()
        let input = try await processor.prepare(
            input: UserInput(
                prompt: .text("A\n  B "),
                images: [.ciImage(image(.red)), .ciImage(image(.blue, width: 16))]))
        let frames = try XCTUnwrap(input.image?.frames)
        XCTAssertEqual(frames.map { [$0.t, $0.h, $0.w] }, [[1, 4, 4], [1, 4, 8]])
        XCTAssertEqual(input.text.tokens.asArray(Int.self).filter { $0 == 48 }.count, 12)
        let pixels = try XCTUnwrap(input.image?.pixels)
        XCTAssertGreaterThan(pixels[0, 0].item(Float.self), pixels[16, 0].item(Float.self))
        XCTAssertThrowsError(try MiniCPMOCRJevProcessor.padExtremeAspectRatio(CIImage.empty()))
        let padded = try MiniCPMOCRJevProcessor.padExtremeAspectRatio(
            image(.black, width: 400, height: 1))
        XCTAssertEqual(padded.extent.size, CGSize(width: 400, height: 3))
        do {
            _ = try await processor.prepare(
                input: UserInput(prompt: .text("A"), images: [.ciImage(image(.red))]))
            XCTFail("Require the rendering")
        } catch {}
    }

    func testNativeAndQuantizedCheckpointsReloadWithIdenticalScores() throws {
        for packed in [false, true] {
            let original = try model()
            if packed {
                quantize(model: original) { path, module in
                    guard
                        case .useDefault = original.quantizationPredicate(
                            path: path, module: module),
                        let linear = module as? Linear, linear.weight.dim(-1).isMultiple(of: 32)
                    else { return nil }
                    return (32, 4, .affine)
                }
            }
            let weights = Dictionary(uniqueKeysWithValues: original.parameters().flattened())
            let url = try directory()
            for wrapped in [false, true] {
                let arrays = Dictionary(
                    uniqueKeysWithValues: weights.map { key, value in
                        (
                            wrapped || !key.hasPrefix("backbone.") ? key : String(key.dropFirst(9)),
                            value
                        )
                    })
                try MLX.save(
                    arrays: arrays, metadata: ["format": "mlx"],
                    url: url.appendingPathComponent("model.safetensors"))
                let loaded = try model()
                try CheckpointLoader.load(
                    modelDirectory: url, model: loaded,
                    quantization: packed ? .init(groupSize: 32, bits: 4) : nil)
                XCTAssertTrue(
                    arrayEqual(try original.logits(input()), try loaded.logits(input())).item(
                        Bool.self))
            }
        }
    }

    func testConversionKeepsClassificationHeadAndReloadsQuantizedBackbone() throws {
        let original = try model()
        let source = try directory()
        let destination = source.appendingPathComponent("quantized")
        try configuration().write(to: source.appendingPathComponent("config.json"))
        try MLX.save(
            arrays: Dictionary(uniqueKeysWithValues: original.parameters().flattened()),
            metadata: ["format": "mlx"], url: source.appendingPathComponent("model.safetensors"))
        let result = try ModelConversion.convert(
            source: source, destination: destination, quantization: .init(groupSize: 32, bits: 4))
        XCTAssertGreaterThan(result.quantizedMatrixCount, 0)
        let files = try safetensorWeightURLs(in: destination)
        let weights = try files.reduce(into: [String: MLXArray]()) { values, file in
            values.merge(try MLX.loadArrays(url: file)) { _, new in new }
        }
        XCTAssertNil(weights["score.scales"])
        XCTAssertTrue(
            arrayEqual(try XCTUnwrap(weights["score.weight"]), original.scoreHead.weight).item(
                Bool.self))
        let config = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: Data(contentsOf: destination.appendingPathComponent("config.json")))
        let loaded = try model()
        try CheckpointLoader.load(
            modelDirectory: destination, model: loaded,
            perLayerQuantization: config.perLayerQuantization)
        XCTAssertTrue(all(isFinite(try loaded.logits(input()))).item(Bool.self))
    }

    func testMissingClassificationWeightsAreRejected() throws {
        let original = try model()
        let url = try directory()
        let weights = Dictionary(
            uniqueKeysWithValues: original.parameters().flattened().filter {
                $0.0 != "score.weight"
            })
        try MLX.save(
            arrays: weights, metadata: ["format": "mlx"],
            url: url.appendingPathComponent("model.safetensors"))
        XCTAssertThrowsError(try CheckpointLoader.load(modelDirectory: url, model: model()))
    }

    func testPublicLoaderAndScoringOperation() async throws {
        let original = try model()
        let url = try directory()
        try configuration().write(to: url.appendingPathComponent("config.json"))
        try JSONEncoder().encode(processor().config).write(
            to: url.appendingPathComponent("preprocessor_config.json"))
        try MLX.save(
            arrays: Dictionary(uniqueKeysWithValues: original.parameters().flattened()),
            metadata: ["format": "mlx"], url: url.appendingPathComponent("model.safetensors"))
        let scorer = try await NativOCRScorer.loadModel(
            from: .directory(url), using: JevTestTokenizerLoader())
        let result = try await scorer.score(
            original: .ciImage(image(.red)), text: "A", rendering: .ciImage(image(.blue))
        ).result()
        XCTAssertEqual(
            result.consistentProbability + result.inconsistentProbability, 1, accuracy: 1e-6)
        XCTAssertGreaterThan(result.tokenCount, 8)
    }

    func testRealCheckpointWhenRequested() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["MINICPM_OCR_JEV_MODEL"] else {
            throw XCTSkip("Set MINICPM_OCR_JEV_MODEL for a full-checkpoint scoring test")
        }
        let original = URL(fileURLWithPath: try XCTUnwrap(env["MINICPM_OCR_JEV_ORIGINAL"]))
        let rendering = URL(fileURLWithPath: try XCTUnwrap(env["MINICPM_OCR_JEV_RENDERING"]))
        let text = try XCTUnwrap(env["MINICPM_OCR_JEV_TEXT"])
        let scorer = try await NativOCRScorer.loadModel(
            from: .directory(URL(fileURLWithPath: path)))
        let result = try await scorer.score(original: original, text: text, rendering: rendering)
            .result()
        XCTAssertEqual(
            result.consistentProbability + result.inconsistentProbability, 1, accuracy: 1e-6)
        XCTAssertTrue(result.consistentLogit.isFinite && result.inconsistentLogit.isFinite)
        print(
            "Jev: logits=[\(result.inconsistentLogit),\(result.consistentLogit)] prob_1=\(result.consistentProbability) tokens=\(result.tokenCount)"
        )
        if let expected = env["MINICPM_OCR_JEV_EXPECTED_LOGITS"]?.split(separator: ",").compactMap({
            Float($0)
        }), expected.count == 2 {
            XCTAssertEqual(result.inconsistentLogit, expected[0], accuracy: 0.2)
            XCTAssertEqual(result.consistentLogit, expected[1], accuracy: 0.2)
        }
    }
}
