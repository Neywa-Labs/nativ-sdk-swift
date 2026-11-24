import CoreImage
import MLX
import MLXNN
import XCTest

@testable import NativSDK

private struct MageVLTestTokenizer: Tokenizer {
    var imagePadID = 151_655

    var bosToken: String? { "<|im_start|>" }
    var eosToken: String? { "<|im_end|>" }
    var unknownToken: String? { nil }

    func encode(text: String, addSpecialTokens _: Bool) -> [Int] {
        var output: [Int] = []
        var index = text.startIndex
        while index < text.endIndex {
            if text[index...].hasPrefix("<|image_pad|>") {
                output.append(imagePadID)
                index = text.index(index, offsetBy: "<|image_pad|>".count)
            } else {
                output.append(Int(text[index].unicodeScalars.first!.value))
                index = text.index(after: index)
            }
        }
        return output
    }

    func decode(tokenIds tokenIDs: [Int], skipSpecialTokens _: Bool) -> String {
        tokenIDs.map { $0 == imagePadID ? "<|image_pad|>" : String(UnicodeScalar($0)!) }
            .joined()
    }

    func convertTokenToId(_ token: String) -> Int? {
        token == "<|image_pad|>" ? imagePadID : nil
    }

    func convertIdToToken(_ id: Int) -> String? {
        id == imagePadID ? "<|image_pad|>" : nil
    }

    func applyChatTemplate(
        messages: [[String: any Sendable]],
        tools _: [[String: any Sendable]]?,
        additionalContext _: [String: any Sendable]?,
    )
        throws -> [Int]
    {
        let images = messages.reduce(into: 0) { total, message in
            guard let content = message["content"] as? [[String: any Sendable]] else { return }
            total += content.count { ($0["type"] as? String) == "image" }
        }
        return encode(
            text: String(repeating: "<|image_pad|>", count: images) + "ok",
            addSpecialTokens: false)
    }
}

private func makeMageProcessor() throws -> MageVLProcessor {
    let json = """
        {
          "processor_class": "MageVLProcessor",
          "image_mean": [0.48145466, 0.4578275, 0.40821073],
          "image_std": [0.26862954, 0.26130258, 0.27577711],
          "min_pixels": 3136,
          "max_pixels": 4000000,
          "patch_size": 16,
          "temporal_patch_size": 1,
          "merge_size": 2
        }
        """.data(using: .utf8)!
    let config = try JSONDecoder.json5().decode(
        MageVLProcessorConfiguration.self, from: json)
    return MageVLProcessor(config, tokenizer: MageVLTestTokenizer())
}

final class MageVLTests: XCTestCase {
    private func tinyConfiguration(modelType: String) -> Data {
        Data(
            """
            {
              "model_type": "\(modelType)", "image_token_id": 250, "video_token_id": 251,
              "text_config": {
                "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
                "num_key_value_heads": 1, "head_dim": 16, "intermediate_size": 64,
                "vocab_size": 256
              },
              "vision_config": {
                "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2,
                "intermediate_size": 64, "patch_size": 16, "spatial_merge_size": 2,
                "temporal_patch_size": 1, "out_hidden_size": 32
              }
            }
            """.utf8)
    }

    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: url) }
        return url
    }

    func testCoworkAndMageSourceCheckpointsRoundTripParametersAndImageLogits() throws {
        let input = LMInput(
            text: .init(tokens: MLXArray([1, 250, 250, 250, 250, 2], [1, 6])),
            image: .init(pixels: MLXArray.ones([16, 768]), frames: [THW(1, 4, 4)]))
        func logits(_ model: MageVL) throws -> MLXArray {
            let result = try model.prepare(
                input, cache: model.newCache(configuration: nil), state: nil,
                prefill: .init(chunking: .unchunked))
            guard case .logits(let output) = result else {
                throw VLMError.processing("Expected image-prefill logits")
            }
            return output.logits
        }

        for modelType in ["mage_vl", "llava_onevision2"] {
            for quantized in [false, true] {
                let data = tinyConfiguration(modelType: modelType)
                let reference = try XCTUnwrap(
                    ModelRegistry.default.createModel(configuration: data, modelType: modelType)
                        as? MageVL)
                reference.train(false)
                if quantized { quantize(model: reference, groupSize: 32, bits: 4) }
                let expected = Dictionary(uniqueKeysWithValues: reference.parameters().flattened())
                var source = [String: MLXArray]()
                for (key, value) in expected {
                    if key.hasPrefix("vision_tower.") {
                        let prefix = modelType == "llava_onevision2" ? "visual." : "model.visual."
                        source[prefix + key.dropFirst("vision_tower.".count)] =
                            key.hasSuffix("patch_embedding.weight")
                            ? value.transposed(0, 3, 1, 2) : value
                    } else if key.hasPrefix("language_model.model.") {
                        let prefix =
                            modelType == "llava_onevision2" ? "model." : "model.language_model."
                        source[prefix + key.dropFirst("language_model.model.".count)] = value
                    } else {
                        source[String(key.dropFirst("language_model.".count))] = value
                    }
                }
                let expectedLogits = try logits(reference)
                XCTAssertTrue(all(isFinite(expectedLogits)).item(Bool.self))
                let url = try directory()
                try data.write(to: url.appendingPathComponent("config.json"))
                // Both the original HF namespaces and saved MLX namespaces must load.
                for weights in [source, expected] {
                    try MLX.save(
                        arrays: weights, url: url.appendingPathComponent("model.safetensors"))
                    let model = try XCTUnwrap(
                        ModelRegistry.default.createModel(configuration: data, modelType: modelType)
                            as? MageVL)
                    try CheckpointLoader.load(
                        modelDirectory: url, model: model,
                        quantization: quantized ? .init(groupSize: 32, bits: 4) : nil)
                    let loaded = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
                    XCTAssertEqual(Set(loaded.keys), Set(expected.keys))
                    for (key, value) in expected {
                        XCTAssertTrue(
                            arrayEqual(try XCTUnwrap(loaded[key]), value).item(Bool.self), key)
                    }
                    XCTAssertTrue(allClose(try logits(model), expectedLogits).item(Bool.self))
                }
            }
        }
    }

    func testCoworkAndMageComposeNativeProcessorAndImageGeometry() async throws {
        let url = try directory()
        // The released Cowork sidecar names a generic processor. Image settings may
        // also be nested; model geometry must take precedence over sidecar defaults.
        try Data(
            #"{"processor_class":"Qwen2_5_VLProcessor","image_processor":{"min_pixels":2048,"max_pixels":2048,"patch_size":14,"merge_size":4,"temporal_patch_size":2}}"#
                .utf8
        )
        .write(to: url.appendingPathComponent("preprocessor_config.json"))
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 64, height: 32))
        for modelType in ["mage_vl", "llava_onevision2"] {
            let data = tinyConfiguration(modelType: modelType)
            let (settings, processorType) = try await ModelLoader().loadProcessorConfig(
                from: url, modelType: modelType, configData: data)
            XCTAssertEqual(processorType, "MageVLProcessor")
            XCTAssertTrue(
                ModelRegistry.processorPolicies[modelType]?.supportsMultimodalInput == true)
            let processor = try ModelRegistry.default.createProcessor(
                configuration: settings, processorType: processorType,
                tokenizer: MageVLTestTokenizer(imagePadID: 250))
            XCTAssertTrue(processor is MageVLProcessor)
            let prepared = try await processor.prepare(
                input: UserInput(chat: [.user("Click Save.", images: [.ciImage(image)])]))
            XCTAssertEqual(prepared.image?.pixels.shape, [8, 768])
            XCTAssertEqual(prepared.image?.frames?.first?.t, 1)
            XCTAssertEqual(prepared.image?.frames?.first?.h, 2)
            XCTAssertEqual(prepared.image?.frames?.first?.w, 4)
            XCTAssertEqual(prepared.text.tokens.asArray(Int.self).filter { $0 == 250 }.count, 2)
        }
    }

    /// Opt in with a local checkpoint, screenshot containing Save, and the upstream system prompt.
    /// Generates a proposed action without executing it; never downloads weights.
    func testCoworkRealCheckpointWhenRequested() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["MAGE_VL_COWORK_MODEL"] else {
            throw XCTSkip("Set MAGE_VL_COWORK_MODEL to run the full-checkpoint image smoke test")
        }
        let imagePath = try XCTUnwrap(environment["MAGE_VL_COWORK_IMAGE"])
        let systemPath = try XCTUnwrap(environment["MAGE_VL_COWORK_SYSTEM_PROMPT"])
        let systemPrompt = try String(contentsOfFile: systemPath, encoding: .utf8)
        let model = try await NativModel.loadModel(
            from: .directory(URL(fileURLWithPath: modelPath)))
        let request = model.generate(
            input: UserInput(
                chat: [
                    .system(systemPrompt),
                    .user(
                        "Click the Save button.",
                        images: [.url(URL(fileURLWithPath: imagePath))]),
                ],
                additionalContext: ["enable_thinking": true]),
            options: .init(sampling: .init(temperature: 0), output: .init(maxTokens: 512)))
        var text = ""
        var calls = [ToolCall]()
        for try await event in request {
            switch event {
            case .chunk(let value), .reasoning(let value): text += value
            case .toolCall(let call): calls.append(call)
            default: break
            }
        }
        XCTAssertTrue(text.contains("Save"), text)
        XCTAssertTrue(calls.contains { $0.function.name == "computer_use" }, "\(calls)")
        let summary = try await request.metrics()
        XCTAssertEqual(summary.stopReason, .stop)
    }

    /// The checkpoint nests rope under `rope_parameters`; a missed rope_theta would
    /// parity-pass at short lengths and diverge at long ones.
    func testTextConfigurationFlattensNestedRopeParameters() throws {
        let data = """
            {
              "model_type": "mage_vl",
              "image_token_id": 151655,
              "video_token_id": 151656,
              "text_config": {
                "model_type": "qwen3",
                "hidden_size": 2560,
                "num_hidden_layers": 36,
                "num_attention_heads": 32,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 9728,
                "vocab_size": 151936,
                "rope_parameters": {"rope_theta": 5000000, "rope_type": "default"},
                "tie_word_embeddings": false
              },
              "vision_config": {
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
                "patch_size": 16,
                "spatial_merge_size": 2,
                "temporal_patch_size": 1,
                "out_hidden_size": 2560,
                "frame_windows_size": 4,
                "use_head": false
              }
            }
            """.data(using: .utf8)!

        let config = try JSONDecoder.json5().decode(MageVLConfiguration.self, from: data)

        XCTAssertEqual(config.textConfiguration.ropeTheta, 5_000_000)
        XCTAssertEqual(config.textConfiguration.headDim, 128)
        XCTAssertEqual(config.imageTokenID, 151_655)
        XCTAssertFalse(config.visionConfiguration.useHead)
        XCTAssertEqual(config.visionConfiguration.frameWindowsSize, 4)
    }

    /// Shapes captured from mlx-vlm on mlx-community/Mage-VL-8bit for a 448x448 input.
    func testProcessorMatchesReferenceGridAndPatchCount() async throws {
        let processor = try makeMageProcessor()
        let image = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 448, height: 448))
        let input = UserInput(chat: [.user("Describe this image.", images: [.ciImage(image)])])

        let prepared = try await processor.prepare(input: input)

        XCTAssertEqual(prepared.image?.pixels.shape, [784, 768])
        XCTAssertEqual(prepared.image?.frames?.first?.t, 1)
        XCTAssertEqual(prepared.image?.frames?.first?.h, 28)
        XCTAssertEqual(prepared.image?.frames?.first?.w, 28)

        let tokens = prepared.text.tokens.asArray(Int.self)
        XCTAssertEqual(tokens.count(where: { $0 == 151_655 }), 196)
    }

    /// Interleaved, `(x1,x2,x3,x4) -> (-x2,x1,-x4,x3)`, not the usual split-half.
    func testRotateHalfIsInterleaved() {
        let x = MLXArray([1, 2, 3, 4] as [Float], [1, 4])
        let rotated = MageVLVision.rotateHalfInterleaved(x)
        XCTAssertEqual(rotated.asArray(Float.self), [-2, 1, -4, 3])
    }

    /// A single image is one window, so attention stays unmasked; only multi-frame video
    /// past `frame_windows_size` splits into blocks.
    func testWindowBoundariesAndMask() {
        let single = MageVLVision.windowBoundaries(
            grids: [THW(1, 28, 28)], totalPatches: 784, frameWindow: 4)
        XCTAssertEqual(single, [0, 784])
        XCTAssertNil(MageVLVision.blockDiagonalMask(boundaries: single, dtype: .float32))

        let video = MageVLVision.windowBoundaries(
            grids: [THW(9, 2, 2)], totalPatches: 36, frameWindow: 4)
        XCTAssertEqual(video, [0, 16, 32, 36])
        let mask = MageVLVision.blockDiagonalMask(boundaries: video, dtype: .float32)
        XCTAssertEqual(mask?.shape, [1, 1, 36, 36])
        XCTAssertEqual(mask?[0, 0, 0, 0].item(Float.self), 0)
        XCTAssertEqual(mask?[0, 0, 0, 20].item(Float.self), -.infinity)
    }

    /// The processor groups patches so each run of merge^2 is one merge block; a row-major
    /// t/h/w walk would put the right count of patches in the wrong places.
    func testPatchPositionsFollowMergeBlockOrder() {
        let positions = MageVLVision.patchPositions(grids: [THW(1, 4, 4)], mergeSize: 2)
        XCTAssertEqual(positions.shape, [16, 3])
        let rows = positions.asArray(Int32.self)
        XCTAssertEqual(Array(rows[0 ..< 12]), [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1])
    }

    func testSanitizeRemapsHubNamespace() throws {
        let config = try JSONDecoder.json5().decode(
            MageVLConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "mage_vl",
                  "text_config": {
                    "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
                    "num_key_value_heads": 1, "head_dim": 4, "intermediate_size": 16,
                    "vocab_size": 32
                  },
                  "vision_config": {
                    "hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2,
                    "intermediate_size": 64, "patch_size": 16, "spatial_merge_size": 2,
                    "out_hidden_size": 8
                  }
                }
                """.data(using: .utf8)),
        )
        let model = MageVL(config)

        let sanitized = model.sanitize(weights: [
            "model.language_model.norm.weight": MLXArray.zeros([8]),
            "model.visual.embeddings.patch_embedding.weight": MLXArray.zeros([64, 3, 16, 16]),
            "lm_head.weight": MLXArray.zeros([32, 8]),
        ])

        XCTAssertNotNil(sanitized["language_model.model.norm.weight"])
        XCTAssertNotNil(sanitized["language_model.lm_head.weight"])
        XCTAssertEqual(
            sanitized["vision_tower.embeddings.patch_embedding.weight"]?.shape,
            [64, 16, 16, 3],
        )
    }
}
