import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

/// No source-format protocol or model-specific format configuration is needed.
private final class SourceSanitizingProjection: Module, BaseLanguageModel {
    @ModuleInfo var projection = Linear(64, 64, bias: false)
    var sanitized = false
    var sawNormalizedWeights = false
    var metadataSeen: [String: String] = [:]

    func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray] {
        sanitized = true
        metadataSeen = metadata
        sawNormalizedWeights =
            weights["source.projection.weight"]?.dtype == .uint32
            && weights["source.projection.scales"]?.dtype == .uint8
            && weights["source.projection.weight_global_scale"]?.dtype == .float32
            && weights["source.projection.weight_packed"] == nil
            && weights["source.projection.weight_scale"] == nil
        return Dictionary(
            uniqueKeysWithValues: weights.map {
                ($0.key.replacingOccurrences(of: "source.", with: ""), $0.value)
            })
    }
}

final class NorthCompressedTensorsTests: XCTestCase {
    private var sourceQuantization: [String: Any] {
        [
            "quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
            "config_groups": [
                "group_0": [
                    "weights": [
                        "num_bits": 4, "type": "float", "symmetric": true, "group_size": 16,
                        "strategy": "tensor_group", "dynamic": false,
                        "scale_dtype": "torch.float8_e4m3fn",
                    ],
                    "input_activations": NSNull(), "output_activations": NSNull(),
                ]
            ],
        ]
    }

    private func configuration() throws -> Data {
        try ModelTestConfigurations.configuration(
            "_text",
            overrides: [
                // Use the FP4 matrix kernel's 64-wide tile, as the official model does.
                "model_type": "cohere2_moe", "hidden_size": 64, "intermediate_size": 64,
                "head_dim": 16, "num_attention_heads": 4, "num_key_value_heads": 2,
                "num_hidden_layers": 2, "num_experts": 4, "num_experts_per_tok": 2,
                "num_shared_experts": 0, "first_k_dense_replace": 1,
                "prefix_dense_intermediate_size": 64, "rms_norm_eps": 1e-6,
                "norm_topk_prob": false, "logit_scale": 1.0,
                "quantization_config": sourceQuantization,
            ])
    }

    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: url) }
        return url
    }

    private func saveSource(
        _ weights: [String: MLXArray], at root: URL, metadata: [String: String] = [:]
    ) throws {
        // Separate components of the same expert across files, as in the official snapshot.
        var parts = Array(repeating: [String: MLXArray](), count: 2)
        for (index, entry) in weights.sorted(by: { $0.key < $1.key }).enumerated() {
            parts[index % 2][entry.key] = entry.value
        }
        for (index, part) in parts.enumerated() {
            let url = root.appendingPathComponent("model-\(index).safetensors")
            try save(arrays: part, metadata: metadata, url: url)
            // MLX stores E4M3 bytes as uint8 arrays; the source header identifies them as FP8.
            let data = try Data(contentsOf: url)
            let count = data.prefix(8).withUnsafeBytes { Int($0.loadUnaligned(as: UInt64.self)) }
            var header = try XCTUnwrap(
                JSONSerialization.jsonObject(with: data[8 ..< 8 + count]) as? [String: Any])
            for key in part.keys where key.hasSuffix(".weight_scale") {
                var value = try XCTUnwrap(header[key] as? [String: Any])
                value["dtype"] = "F8_E4M3"
                header[key] = value
            }
            var encoded = try JSONSerialization.data(withJSONObject: header)
            encoded.append(Data(repeating: 32, count: (8 - encoded.count % 8) % 8))
            var length = UInt64(encoded.count).littleEndian
            var result = withUnsafeBytes(of: &length) { Data($0) }
            result.append(encoded)
            result.append(data[(8 + count)...])
            try result.write(to: url)
        }
    }

    private func fixture(_ root: URL, dtype: DType) throws -> [String: MLXArray] {
        let data = try configuration()
        try data.write(to: root.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(Cohere2MoEConfiguration.self, from: data)
        MLXRandom.seed(47)
        let model = Cohere2MoE(config)
        var source: [String: MLXArray] = [:]
        for (name, array) in model.parameters().flattened() {
            let key = String(name.dropFirst("language_model.".count))
            let value = array.asType(dtype)
            if name.contains(".switch_mlp.") {
                for expert in 0 ..< config.numExperts {
                    let path = String(key.dropLast(".weight".count))
                        .replacingOccurrences(of: ".switch_mlp.", with: ".experts.\(expert).")
                    addPacked(
                        value[expert], path: path, global: Float(expert + 1) + 0.3125, to: &source)
                    source[path + ".bias"] = MLXArray.zeros([value.dim(1)], dtype: dtype)
                }
            } else if key == "model.layers.1.self_attn.o_proj.weight" {
                addPacked(value, path: String(key.dropLast(7)), global: 1.3125, to: &source)
            } else {
                source[key] = value
            }
        }
        try saveSource(source, at: root)
        return source
    }

    private func addPacked(
        _ weight: MLXArray, path: String, global: Float, to source: inout [String: MLXArray]
    ) {
        let packed = quantized(weight, groupSize: 16, bits: 4, mode: .nvfp4)
        source[path + ".weight_packed"] = packed.wq.view(dtype: .uint8)
        source[path + ".weight_scale"] = packed.scales
        source[path + ".weight_global_scale"] = MLXArray(global).asType(.bfloat16)
    }

    func testPackedValuesMatchIndependentFP4ReferenceBeforeRounding() throws {
        final class Projection: Module {
            @ModuleInfo var linear = Linear(32, 16, bias: false)
        }
        let bytes = (0 ..< 256).map(UInt8.init)
        let scales = (0 ..< 32).map { UInt8([0x30, 0x38, 0x40][$0 % 3]) }
        let raw: [String: MLXArray] = [
            "linear.weight_packed": MLXArray(bytes, [16, 16]),
            "linear.weight_scale": MLXArray(scales, [16, 2]),
            "linear.weight_global_scale": MLXArray(Float(1.3125)).asType(.bfloat16),
        ]
        let prepared = try CompressedTensorsNVFP4.prepare(raw)
        XCTAssertEqual(prepared["linear.weight"]?.view(dtype: .uint8).asArray(UInt8.self), bytes)
        XCTAssertEqual(prepared["linear.scales"]?.asArray(UInt8.self), scales)
        let model = Projection()
        try CompressedTensorsNVFP4.install(in: model, weights: prepared)
        let lut: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
        var reference: [Float] = []
        for row in 0 ..< 16 {
            for column in 0 ..< 32 {
                let byte = bytes[row * 16 + column / 2]
                let code = Int(column % 2 == 0 ? byte & 15 : byte >> 4)
                let scale: Float = [0x30: 0.5, 0x38: 1, 0x40: 2][
                    Int(scales[row * 2 + column / 16])]!
                reference.append(lut[code] * scale)
            }
        }
        let weight = MLXArray(reference, [16, 32])
        for dtype: DType in [.float32, .float16, .bfloat16] {
            let input = ((MLXArray(0 ..< 96).asType(.float32) - 48) / 32)
                .reshaped(3, 32).asType(dtype)
            let expected = (matmul(input.asType(.float32), weight.T) / 1.3125).asType(dtype)
            XCTAssertTrue(
                allClose(model.linear(input), expected, rtol: 1e-5, atol: 1e-5).item(Bool.self))
        }
    }

    func testSharedTransformationRunsBeforeSanitizationOnAnotherModel() throws {
        let root = try directory()
        try JSONSerialization.data(withJSONObject: ["quantization_config": sourceQuantization])
            .write(to: root.appendingPathComponent("config.json"))
        let weight = MLXRandom.normal([64, 64]).asType(.bfloat16)
        var source: [String: MLXArray] = [:]
        addPacked(weight, path: "source.projection", global: 1.3125, to: &source)
        let metadata = ["origin": "shared-transformation-test"]
        try saveSource(source, at: root, metadata: metadata)

        let model = SourceSanitizingProjection()
        try CheckpointLoader.load(modelDirectory: root, model: model)
        XCTAssertTrue(model.sawNormalizedWeights)
        XCTAssertEqual(model.metadataSeen, metadata)
        XCTAssertTrue(model.projection is QuantizedLinear)
        XCTAssertEqual(
            model.projection.weight.view(dtype: .uint8).asArray(UInt8.self),
            source["source.projection.weight_packed"]!.asArray(UInt8.self))

        let input = MLXRandom.normal([3, 64]).asType(.bfloat16)
        let packed = quantized(weight, groupSize: 16, bits: 4, mode: .nvfp4)
        let expected =
            (quantizedMM(
                input.asType(.float32), packed.wq, scales: packed.scales, biases: nil,
                transpose: true, groupSize: 16, bits: 4, mode: .nvfp4) / 1.3125).asType(input.dtype)
        XCTAssertTrue(arrayEqual(model.projection(input), expected).item(Bool.self))

        // An unsupported source format is rejected centrally, before any sanitizer runs.
        var invalid = sourceQuantization
        invalid["format"] = "pack-quantized"
        try JSONSerialization.data(withJSONObject: ["quantization_config": invalid])
            .write(to: root.appendingPathComponent("config.json"))
        let rejected = SourceSanitizingProjection()
        XCTAssertThrowsError(try CheckpointLoader.load(modelDirectory: root, model: rejected))
        XCTAssertFalse(rejected.sanitized)
    }

    func testUnrelatedQuantizationMetadataDoesNotEnableSourceTransformation() throws {
        let root = try directory()
        try JSONSerialization.data(withJSONObject: [
            "quantization_config": ["quant_method": "mlx", "config_groups": "unrelated"]
        ]).write(to: root.appendingPathComponent("config.json"))
        let source = SourceSanitizingProjection()
        try saveSource(["source.projection.weight": source.projection.weight], at: root)
        let model = SourceSanitizingProjection()
        try CheckpointLoader.load(modelDirectory: root, model: model)
        XCTAssertTrue(model.sanitized)
        XCTAssertFalse(model.sawNormalizedWeights)
        XCTAssertFalse(model.projection is QuantizedLinear)
        XCTAssertTrue(arrayEqual(model.projection.weight, source.projection.weight).item(Bool.self))
    }

    func testNativeQuantizationTakesPrecedenceOverRetainedSourceMetadata() throws {
        let root = try directory()
        let data = try configuration()
        let config = try JSONDecoder().decode(Cohere2MoEConfiguration.self, from: data)
        let reference = Cohere2MoE(config)
        let native = BaseConfiguration.Quantization(groupSize: 32, bits: 5)
        try quantizeByParent(model: reference, groupSize: native.groupSize, bits: native.bits) {
            _, module in module is SwitchLinear
        }
        try save(
            arrays: Dictionary(uniqueKeysWithValues: reference.parameters().flattened()),
            url: root.appendingPathComponent("model.safetensors"))
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        object["quantization"] = ["group_size": native.groupSize, "bits": native.bits]

        // Source metadata may describe either a supported format or a format that
        // the native conversion no longer uses. Neither changes the saved MLX weights.
        var unsupported = sourceQuantization
        unsupported["format"] = "pack-quantized"
        for retained in [sourceQuantization, unsupported] {
            object["quantization_config"] = retained
            try JSONSerialization.data(withJSONObject: object)
                .write(to: root.appendingPathComponent("config.json"))
            for streaming: ExpertStreamingOptions? in [nil, .init(cacheSizeBytes: 65_536)] {
                let loaded = Cohere2MoE(config)
                try CheckpointLoader.load(
                    modelDirectory: root, model: loaded, quantization: native,
                    expertStreaming: streaming)
                let left = try reference.newCache(configuration: nil)
                let right = try loaded.newCache(configuration: nil)
                for tokens in [Array(0 ..< 32), [9], [11]] {
                    let input = MLXArray(tokens, [1, tokens.count])
                    let expected = reference(input, cache: left)
                    let actual = loaded(input, cache: right)
                    eval(expected, actual)
                    try InferenceErrorCheck(loaded).check()
                    XCTAssertTrue(
                        allClose(expected, actual, rtol: 1e-5, atol: 1e-5).item(Bool.self))
                }
            }
        }
    }

    func testOfficialLayoutResidentAndStreamedPrefillDecodeParity() throws {
        for dtype: DType in [.float32, .bfloat16] {
            let root = try directory()
            _ = try fixture(root, dtype: dtype)
            let config = try JSONDecoder().decode(
                Cohere2MoEConfiguration.self, from: configuration())
            let resident = Cohere2MoE(config)
            try CheckpointLoader.load(modelDirectory: root, model: resident)
            for budget in [4096, 65_536] {
                let streamed = Cohere2MoE(config)
                try CheckpointLoader.load(
                    modelDirectory: root, model: streamed,
                    expertStreaming: .init(
                        cacheSizeBytes: budget, prefetch: budget > 4096, expertBatchSize: 4))
                let left = try resident.newCache(configuration: nil)
                let right = try streamed.newCache(configuration: nil)
                let projections = streamed.modules().compactMap { $0 as? StreamingSwitchLinear }
                XCTAssertEqual(projections.count, 3)
                for tokens in [Array(0 ..< 32), [9], [11], [2, 3, 4], [12]] {
                    let input = MLXArray(tokens, [1, tokens.count])
                    let a = resident(input, cache: left)
                    let b = streamed(input, cache: right)
                    eval(a, b)
                    try InferenceErrorCheck(streamed).check()
                    XCTAssertTrue(
                        allClose(a, b, rtol: 1e-5, atol: 1e-5).item(Bool.self),
                        "dtype=\(dtype) budget=\(budget) tokens=\(tokens.count) maxError=\(abs(a - b).max().item(Float.self))"
                    )
                    XCTAssertEqual(
                        argMax(a, axis: -1).asArray(Int.self), argMax(b, axis: -1).asArray(Int.self)
                    )
                    for projection in projections {
                        XCTAssertLessThanOrEqual(projection.schedule.store.residentBytes, budget)
                    }
                }
                XCTAssertGreaterThan(projections[0].schedule.evaluation.evaluationCount, 0)
            }
        }
    }

    func testMissingGlobalScaleAndInvalidFormatFailBeforeLoading() throws {
        let root = try directory()
        var source = try fixture(root, dtype: .bfloat16)
        let key = "model.layers.1.mlp.experts.0.gate_proj.weight_global_scale"
        source.removeValue(forKey: key)
        try saveSource(source, at: root)
        let config = try JSONDecoder().decode(Cohere2MoEConfiguration.self, from: configuration())
        for streaming: ExpertStreamingOptions? in [nil, .init(cacheSizeBytes: 65_536)] {
            XCTAssertThrowsError(
                try CheckpointLoader.load(
                    modelDirectory: root, model: Cohere2MoE(config), expertStreaming: streaming)
            )
        }
        var data = try XCTUnwrap(
            JSONSerialization.jsonObject(with: configuration()) as? [String: Any])
        var quantization = sourceQuantization
        quantization["format"] = "pack-quantized"
        data["quantization_config"] = quantization
        try JSONSerialization.data(withJSONObject: data)
            .write(to: root.appendingPathComponent("config.json"))
        XCTAssertThrowsError(
            try CheckpointLoader.load(modelDirectory: root, model: Cohere2MoE(config)))
    }

    func testNonpositiveGlobalScaleTerminatesStreamedInference() throws {
        let root = try directory()
        var source = try fixture(root, dtype: .bfloat16)
        for key in source.keys
        where key.contains(".experts.") && key.hasSuffix(".weight_global_scale") {
            source[key] = MLXArray(Float(0)).asType(.bfloat16)
        }
        try saveSource(source, at: root)
        let config = try JSONDecoder().decode(Cohere2MoEConfiguration.self, from: configuration())
        XCTAssertThrowsError(
            try CheckpointLoader.load(modelDirectory: root, model: Cohere2MoE(config)))
        let model = Cohere2MoE(config)
        try CheckpointLoader.load(
            modelDirectory: root, model: model,
            expertStreaming: .init(cacheSizeBytes: 65_536, prefetch: false))
        let logits = model(MLXArray([1, 2], [1, 2]), cache: nil)
        eval(logits)
        XCTAssertThrowsError(try InferenceErrorCheck(model).check())
    }

    func testIncompleteExpertGroupFailsWithoutStackingInvalidArrays() throws {
        let root = try directory()
        var source = try fixture(root, dtype: .bfloat16)
        let prefix = "model.layers.1.mlp.experts.2.gate_proj."
        for key in source.keys where key.hasPrefix(prefix) {
            source.removeValue(forKey: key)
        }
        try saveSource(source, at: root)
        let config = try JSONDecoder().decode(Cohere2MoEConfiguration.self, from: configuration())
        for streaming: ExpertStreamingOptions? in [nil, .init(cacheSizeBytes: 65_536)] {
            XCTAssertThrowsError(
                try CheckpointLoader.load(
                    modelDirectory: root, model: Cohere2MoE(config), expertStreaming: streaming)
            )
        }
    }
}
