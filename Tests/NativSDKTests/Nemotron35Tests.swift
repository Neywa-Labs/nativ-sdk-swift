import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class Nemotron35Tests: XCTestCase {
    /// Opt-in public API check against the original checkpoint. Never downloads weights.
    /// The image fixture must contain the word NATIV and the number 42.
    func testLocalPretrainedTextAndImage() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let path = environment["NATIV_NEMOTRON_CHECKPOINT"],
            let image = environment["NATIV_NEMOTRON_IMAGE"]
        else { throw XCTSkip("Set NATIV_NEMOTRON_CHECKPOINT and NATIV_NEMOTRON_IMAGE") }
        let previousLimit = Memory.cacheLimit
        Memory.cacheLimit = 64 * 1024 * 1024
        defer { Memory.cacheLimit = previousLimit }
        let model = try await NativModel.loadModel(
            from: .directory(URL(fileURLWithPath: path)),
            expertStreaming: .init(cacheSizeBytes: 64 * 1024 * 1024))
        let options = ModelGenerationOptions(
            prefill: .init(stepSize: 256), sampling: .init(temperature: 0),
            output: .init(maxTokens: 32))
        for useImage in [false, true] {
            let request = model.generate(
                input: Self.validationInput(image: useImage ? image : nil), options: options)
            let response = try await request.response()
            let metrics = try await request.metrics()
            print(
                "NEMOTRON_CHECK image=\(useImage) response=\(response.debugDescription) prompt=\(metrics.promptTokenCount) generated=\(metrics.generatedTokenCount) prefillSeconds=\(metrics.promptSeconds) decodeSeconds=\(metrics.generationSeconds) peakMLX=\(Memory.peakMemory)"
            )
            XCTAssertEqual(metrics.stopReason, .stop)
            XCTAssertFalse(response.contains("</think>"))
            if useImage {
                XCTAssertTrue(response.uppercased().contains("NATIV"))
                XCTAssertTrue(response.contains("42"))
            } else {
                XCTAssertTrue(response.contains("Paris"))
            }
        }
    }

    private static func validationInput(image: String?) -> sending UserInput {
        UserInput(
            prompt: image != nil
                ? "Read the word and number in this image. Give only the text you see."
                : "What is the capital of France? Answer in one sentence.",
            images: image.map { [.url(URL(fileURLWithPath: $0))] } ?? [],
            additionalContext: ["enable_thinking": false])
    }

    private func configuration(modelOpt: Bool = false) throws -> Data {
        var config: [String: Any] = [
            "model_type": "nemotron_h_omni", "img_context_token_id": 18,
            "projector_hidden_size": 32,
            "llm_config": [
                "model_type": "nemotron_h_puzzle", "vocab_size": 128, "hidden_size": 128,
                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
                "mamba_num_heads": 2, "mamba_head_dim": 32, "ssm_state_size": 128,
                "conv_kernel": 4, "n_groups": 1, "intermediate_size": 128,
                "moe_latent_size": 64, "moe_shared_expert_intermediate_size": 64,
                "n_shared_experts": 1, "routed_scaling_factor": 5.0,
                "num_nextn_predict_layers": modelOpt ? 1 : 0,
                "block_configs": [
                    ["block_type": "mamba"],
                    [
                        "block_type": "moe", "n_routed_experts": 2,
                        "moe_intermediate_size": 64, "num_experts_per_tok": 1,
                    ],
                    ["block_type": "attention"],
                    [
                        "block_type": "moe", "n_routed_experts": 3,
                        "moe_intermediate_size": 128, "num_experts_per_tok": 2,
                    ],
                ],
            ],
            "vision_config": [
                "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 2,
                "intermediate_size": 16, "image_size": 4, "patch_size": 2,
                "max_resolution": 4, "num_cls_tokens": 3, "num_registers": 7,
                "min_num_patches": 4, "max_num_patches": 4,
            ],
        ]
        if modelOpt { config["quantization_config"] = sourceQuantization }
        return try JSONSerialization.data(withJSONObject: config)
    }

    private var sourceQuantization: [String: Any] {
        [
            "quant_method": "modelopt_mixed", "quant_algo": "MIXED_PRECISION",
            "config_groups": [
                "fp4": ["weights": ["num_bits": 4, "type": "float", "group_size": 16]],
                "fp8": ["weights": ["num_bits": 8, "type": "float"]],
            ],
        ]
    }

    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: url) }
        return url
    }

    func testNemotronHMambaPreservesBF16DuringPrefillAndDecode() throws {
        let config = NemotronHConfiguration(
            vocabSize: 128, hiddenSize: 128, numHiddenLayers: 2,
            numAttentionHeads: 4, numKeyValueHeads: 2,
            mambaNumHeads: 2, mambaHeadDim: 32, ssmStateSize: 128,
            convKernel: 4, nGroups: 1, intermediateSize: 128,
            moeIntermediateSize: 64, moeSharedExpertIntermediateSize: 64,
            nRoutedExperts: 2, numExpertsPerTok: 1, hybridOverridePattern: "M*")
        let model = NemotronHModel(config)
        model.update(parameters: model.parameters().mapValues { $0.asType(.bfloat16) })
        let cache = try model.newCache(configuration: nil)
        for tokens in [[1, 2, 3, 4], [5], [6]] {
            let logits = model(MLXArray(tokens, [1, tokens.count]), cache: cache)
            eval(logits, cache)
            XCTAssertEqual(logits.dtype, .bfloat16)
            XCTAssertTrue(isFinite(logits).all().item(Bool.self))
            XCTAssertEqual(cache[0].state[0].dtype, .bfloat16)
        }
    }

    func testPuzzleAndNamedBlockConfiguration() throws {
        let data = try configuration()
        let config = try JSONDecoder().decode(NemotronOmniConfiguration.self, from: data)
        let language = try XCTUnwrap(config.languageConfiguration)
        XCTAssertEqual(language.numHiddenLayers, 4)
        XCTAssertEqual(language.hybridOverridePattern, "ME*E")
        XCTAssertEqual(language.configuration(forLayer: 1).nRoutedExperts, 2)
        XCTAssertEqual(language.configuration(forLayer: 3).nRoutedExperts, 3)
        XCTAssertEqual(language.configuration(forLayer: 3).moeIntermediateSize, 128)
        XCTAssertEqual(language.moeLatentSize, 64)
        let model = try ModelRegistry.default.createModel(
            configuration: data, modelType: "nemotron_h_omni")
        XCTAssertTrue(model is NemotronOmni)
        XCTAssertEqual(model.reasoningConfig, .thinkTagsWithEnableThinking)
        XCTAssertEqual(model.toolCallFormat, .xmlFunction)
        let weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        XCTAssertEqual(
            weights["vision_model.radio_model.model.patch_generator.cls_token.token"]?.shape,
            [10, 8])
        XCTAssertEqual(
            weights["language_model.backbone.layers.3.mixer.switch_mlp.fc1.weight"]?.shape,
            [3, 128, 64])

        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        var llm = try XCTUnwrap(object["llm_config"] as? [String: Any])
        llm.removeValue(forKey: "block_configs")
        llm["layers_block_type"] = ["mamba", "moe", "attention"]
        llm["n_routed_experts"] = 4
        llm["moe_intermediate_size"] = 128
        llm["num_experts_per_tok"] = 2
        object["llm_config"] = llm
        let uniform = try JSONDecoder().decode(
            NemotronOmniConfiguration.self,
            from: JSONSerialization.data(withJSONObject: object))
        XCTAssertEqual(uniform.languageConfiguration?.hybridOverridePattern, "ME*")
        XCTAssertEqual(uniform.languageConfiguration?.numHiddenLayers, 3)
        // Legacy configs retain the original block list alongside an explicit override.
        for override in ["MEE" as Any, ["M", "E", "E"] as Any] {
            llm["hybrid_override_pattern"] = override
            let legacy = try JSONDecoder().decode(
                NemotronHConfiguration.self, from: JSONSerialization.data(withJSONObject: llm))
            XCTAssertEqual(legacy.hybridOverridePattern, "MEE")
        }
        llm["num_hidden_layers"] = 99
        XCTAssertThrowsError(
            try JSONDecoder().decode(
                NemotronHConfiguration.self,
                from: JSONSerialization.data(withJSONObject: llm)))
    }

    func testSeparateBF16AndModelOptExpertsMatchResidentPrefillAndDecode() throws {
        for modelOpt in [false, true] {
            let root = try directory()
            let data = try configuration(modelOpt: modelOpt)
            try data.write(to: root.appendingPathComponent("config.json"))
            let config = try JSONDecoder().decode(NemotronOmniConfiguration.self, from: data)
            MLXRandom.seed(47)
            let initial = NemotronOmni(config)
            var weights: [String: MLXArray] = [:]
            for (name, value) in initial.parameters().flattened() {
                if name.contains(".switch_mlp.") {
                    for expert in 0 ..< value.dim(0) {
                        let source = try XCTUnwrap(
                            initial.separateExpertWeightNames(for: name, expert: expert).first)
                        let matrix = value[expert].asType(.bfloat16)
                        if modelOpt {
                            let packed = quantized(matrix, groupSize: 16, bits: 4, mode: .nvfp4)
                            let prefix = String(source.dropLast(".weight".count))
                            weights[source] = packed.wq.view(dtype: .uint8)
                            weights[prefix + ".weight_scale"] = packed.scales
                            weights[prefix + ".weight_scale_2"] = MLXArray(
                                Float(expert + 1) * 0.03125)
                            weights[prefix + ".input_scale"] = MLXArray(Float(0.5))
                        } else {
                            weights[source] = matrix
                        }
                    }
                } else if modelOpt && name.contains(".shared_experts.") && name.hasSuffix(".weight")
                {
                    let packed = quantized(
                        value.asType(.bfloat16), groupSize: 16, bits: 4, mode: .nvfp4)
                    let prefix = String(name.dropLast(".weight".count))
                    weights[name] = packed.wq.view(dtype: .uint8)
                    weights[prefix + ".weight_scale"] = packed.scales
                    weights[prefix + ".weight_scale_2"] = MLXArray(Float(0.03125))
                    weights[prefix + ".input_scale"] = MLXArray(Float(0.5))
                } else if modelOpt && name == "language_model.backbone.layers.2.mixer.q_proj.weight"
                {
                    weights[name] = MLXRandom.randInt(low: 32, high: 64, value.shape).asType(.uint8)
                    weights[String(name.dropLast(7)) + ".weight_scale"] = MLXArray(Float(0.0625))
                    weights[String(name.dropLast(7)) + ".input_scale"] = MLXArray(Float(0.25))
                } else {
                    let key =
                        name.hasPrefix("vision_final_layernorm.")
                        ? "vision_projector." + name : name
                    weights[key] = value.asType(.bfloat16)
                }
            }
            weights["vision_model.summary_idxs"] = MLXArray([0, 1])
            // Shard components separately to exercise actual header/offset resolution.
            let keys = weights.keys.sorted()
            for part in 0 ..< 2 {
                let selected = keys.enumerated().filter { $0.offset % 2 == part }.map(\.element)
                try save(
                    arrays: Dictionary(uniqueKeysWithValues: selected.map { ($0, weights[$0]!) }),
                    url: root.appendingPathComponent("model-\(part).safetensors"))
            }
            let resident = NemotronOmni(config)
            let streamed = NemotronOmni(config)
            try CheckpointLoader.load(modelDirectory: root, model: resident)
            try CheckpointLoader.load(
                modelDirectory: root, model: streamed,
                expertStreaming: .init(cacheSizeBytes: 64 * 1024, prefetch: false))
            let check = InferenceErrorCheck(streamed)
            let left = resident.newCache(configuration: nil)
            let right = streamed.newCache(configuration: nil)
            for tokens in [Array(1 ... 40), [5], [6]] {
                let input = LMInput.Text(tokens: MLXArray(tokens, [1, tokens.count]))
                let a = resident(input, cache: left, state: nil).logits
                let b = streamed(input, cache: right, state: nil).logits
                eval(a, b)
                // The unweighted Mamba normalization must not promote the BF16 model.
                XCTAssertEqual(a.dtype, .bfloat16)
                XCTAssertEqual(b.dtype, .bfloat16)
                try check.check()
                XCTAssertTrue(
                    allClose(a, b, rtol: 1e-5, atol: 1e-5).item(Bool.self), "ModelOpt=\(modelOpt)")
            }
            XCTAssertEqual(left.map(\.offset), right.map(\.offset))
            XCTAssertEqual(left.map(\.offset), [0, 42])
            for (a, b) in zip(left[0].state, right[0].state) {
                XCTAssertTrue(allClose(a, b, rtol: 1e-5, atol: 1e-5).item(Bool.self))
            }
            // KV storage reserves capacity beyond its valid offset.
            for (a, b) in zip(left[1].state, right[1].state) {
                XCTAssertTrue(
                    allClose(
                        a[0..., 0..., ..<left[1].offset, 0...],
                        b[0..., 0..., ..<right[1].offset, 0...], rtol: 1e-5, atol: 1e-5
                    ).item(Bool.self))
            }
            let input = LMInput(tokens: MLXArray([1, 2, 3, 4], [1, 4]))
            let options = ModelGenerationOptions(
                prefill: .init(stepSize: 2), sampling: .init(temperature: 0),
                output: .init(maxTokens: 2))
            let a = try AutoregressiveIterator(input: input, model: resident, options: options)
            let b = try AutoregressiveIterator(input: input, model: streamed, options: options)
            XCTAssertEqual(Array(a), Array(b))
            XCTAssertNil(a.failure)
            XCTAssertNil(b.failure)
            // Three class tokens plus seven registers must be removed before the
            // four patch features become one projected image token.
            let imageInput = LMInput(
                text: .init(tokens: MLXArray([1, 18, 2], [1, 3])),
                image: .init(pixels: MLXArray.zeros([1, 3, 4, 4]), frames: [THW(1, 4, 4)]))
            let imageA = try AutoregressiveIterator(
                input: imageInput, model: resident, options: options)
            let imageB = try AutoregressiveIterator(
                input: imageInput, model: streamed, options: options)
            XCTAssertEqual(Array(imageA), Array(imageB))
            XCTAssertNil(imageA.failure)
            XCTAssertNil(imageB.failure)
        }
    }

    func testNativeMixedConversionPreservesModelOptGlobalScalesAndExpertStreaming() throws {
        let root = try directory()
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: configuration(modelOpt: true)) as? [String: Any])
        let config = try JSONDecoder().decode(
            NemotronOmniConfiguration.self, from: JSONSerialization.data(withJSONObject: object))
        let original = NemotronOmni(config)
        var weights: [String: MLXArray] = [:]
        var quantization: [String: Any] = ["group_size": 64, "bits": 4, "mode": "affine"]
        for (name, value) in original.parameters().flattened() {
            if name.contains(".switch_mlp.") && name.hasSuffix(".weight") {
                let packed = quantized(value.asType(.bfloat16), groupSize: 16, bits: 4, mode: .nvfp4)
                let path = String(name.dropLast(7))
                weights[name] = packed.wq
                weights[path + ".scales"] = packed.scales
                weights[path + ".weight_scale_2"] = MLXArray.ones([value.dim(0)]) * 0.3125
                quantization[path] = ["group_size": 16, "bits": 4, "mode": "nvfp4"]
            } else {
                weights[name] = value.asType(.bfloat16)
            }
        }
        object["quantization"] = quantization
        object["quantization_config"] = quantization
        try JSONSerialization.data(withJSONObject: object)
            .write(to: root.appendingPathComponent("config.json"))
        try save(
            arrays: weights, metadata: ["format": "mlx"],
            url: root.appendingPathComponent("model.safetensors"))
        let converted = root.appendingPathComponent("mixed")
        _ = try ModelConversion.convert(source: root, destination: converted, recipe: .mixed4_8)
        let metadata = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: Data(contentsOf: converted.appendingPathComponent("config.json")))
        let resident = NemotronOmni(config)
        let streamed = NemotronOmni(config)
        try CheckpointLoader.load(
            modelDirectory: converted, model: resident,
            perLayerQuantization: metadata.perLayerQuantization)
        try CheckpointLoader.load(
            modelDirectory: converted, model: streamed,
            perLayerQuantization: metadata.perLayerQuantization,
            expertStreaming: .init(cacheSizeBytes: 64 * 1024, prefetch: false))
        let installed = Dictionary(uniqueKeysWithValues: resident.parameters().flattened())
        for (name, source) in weights where name.contains(".switch_mlp.") {
            XCTAssertTrue(arrayEqual(try XCTUnwrap(installed[name]), source).item(Bool.self), name)
        }
        XCTAssertEqual(
            metadata.perLayerQuantization?.quantization(layer: "language_model.lm_head")?.bits, 8)
        let left = resident.newCache(configuration: nil), right = streamed.newCache(configuration: nil)
        let errors = InferenceErrorCheck(streamed)
        for tokens in [Array(1 ... 40), [5], [6]] {
            let input = LMInput.Text(tokens: MLXArray(tokens, [1, tokens.count]))
            let a = resident(input, cache: left, state: nil).logits
            let b = streamed(input, cache: right, state: nil).logits
            eval(a, b)
            try errors.check()
            XCTAssertTrue(allClose(a, b, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        }
    }

    func testModelOptGlobalScaleMultipliesAndFP8Decodes() throws {
        final class Projections: Module {
            @ModuleInfo var fp4 = Linear(64, 64, bias: false)
            @ModuleInfo var fp8 = Linear(64, 64, bias: false)
        }
        let packed = quantized(MLXRandom.normal([64, 64]), groupSize: 16, bits: 4, mode: .nvfp4)
        let bytes = MLXArray(
            (0 ..< 4096).map { UInt8([0x00, 0x01, 0x38, 0xB8, 0x7E][$0 % 5]) }, [64, 64])
        let source: [String: MLXArray] = [
            "fp4.weight": packed.wq.view(dtype: .uint8), "fp4.weight_scale": packed.scales,
            "fp4.weight_scale_2": MLXArray(Float(0.3125)), "fp4.input_scale": MLXArray(Float(9)),
            "fp8.weight": bytes, "fp8.weight_scale": MLXArray(Float(0.5)),
            "fp8.input_scale": MLXArray(Float(3)), "fp8.k_scale": MLXArray(Float(2)),
        ]
        let normalized = try ModelOptCheckpoint.prepare(source)
        XCTAssertNil(normalized["fp4.input_scale"])
        XCTAssertNil(normalized["fp8.k_scale"])
        let expectedFP8 = MLXArray(
            (0 ..< 4096).map { [Float(0), 1 / 512, 1, -1, 448][$0 % 5] * 0.5 }, [64, 64]
        ).asType(.bfloat16)
        XCTAssertTrue(arrayEqual(normalized["fp8.weight"]!, expectedFP8).item(Bool.self))
        let model = Projections()
        try CompressedTensorsNVFP4.install(in: model, weights: normalized)
        try model.update(parameters: .unflattened(normalized), verify: [.all])
        let x = MLXRandom.normal([3, 64]).asType(.bfloat16)
        let unpacked = dequantized(
            packed.wq, scales: packed.scales, biases: nil, groupSize: 16, bits: 4, mode: .nvfp4)
        let expected = (matmul(x.asType(.float32), unpacked.T) * 0.3125).asType(x.dtype)
        XCTAssertTrue(allClose(model.fp4(x), expected, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        var invalid = source
        invalid["fp4.weight_scale_2"] = MLXArray(Float.nan)
        XCTAssertThrowsError(try ModelOptCheckpoint.prepare(invalid))
        invalid = source
        invalid.removeValue(forKey: "fp4.weight_scale")
        XCTAssertThrowsError(try ModelOptCheckpoint.prepare(invalid))
    }
}
