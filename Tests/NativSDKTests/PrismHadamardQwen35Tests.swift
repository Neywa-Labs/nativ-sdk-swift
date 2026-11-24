import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class PrismHadamardQwen35Tests: XCTestCase {
    private func signs(_ width: Int) -> MLXArray {
        let draw = MLXRandom.randInt(0 ..< 2, [width], key: MLXRandom.key(7))
        return draw.asType(.float32) * 2 - 1
    }

    func testSignedRotationAndInverseAcrossBlocks() {
        let block = 512
        let x = MLXRandom.normal([2, 3, block * 2], key: MLXRandom.key(11))
        let s = signs(block * 2)
        let rotated = hadamardRotate(x, block: block, signs: s)
        let expected = hadamardTransform((x * s).reshaped([-1, block])).reshaped(x.shape)
        XCTAssertLessThan(abs(rotated - expected).max().item(Float.self), 1e-5)
        let restored = hadamardRotate(rotated, block: block, signs: s, inverse: true)
        XCTAssertLessThan(abs(restored - x).max().item(Float.self), 1e-5)
        let twice = hadamardRotate(rotated, block: block, signs: s)
        XCTAssertGreaterThan(abs(twice - x).mean().item(Float.self), 0.1)
    }

    private func configuration() -> [String: Any] {
        let paths = [
            "lm_head", "model.embed_tokens", "model.layers.0.linear_attn.in_proj_qkv",
            "model.layers.0.linear_attn.in_proj_z", "model.layers.0.linear_attn.in_proj_b",
            "model.layers.0.linear_attn.in_proj_a",
        ]
        return [
            "model_type": "prism_hadamard_qwen35", "schema_version": 2,
            "base_model_type": "qwen3_5", "tensor_namespace": "mlx-vlm-qwen3_5",
            "gdn_activation_layout": "grouped",
            "quantization": ["bits": 2, "group_size": 128, "mode": "affine"],
            "modules": paths.map {
                [
                    "path": $0, "block": $0.hasSuffix("in_proj_a") ? 0 : 512,
                    "embedding": $0 == "model.embed_tokens", "dtype": "float16",
                ] as [String: Any]
            },
            "image_token_id": 28, "video_token_id": 29,
            "vision_start_token_id": 30, "vision_end_token_id": 31,
            "text_config": [
                "model_type": "qwen3_5_text", "hidden_size": 512, "intermediate_size": 512,
                "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 32, "vocab_size": 32, "full_attention_interval": 2,
                "linear_num_value_heads": 4, "linear_num_key_heads": 2,
                "linear_key_head_dim": 32, "linear_value_head_dim": 32,
                "linear_conv_kernel_dim": 4, "tie_word_embeddings": false,
                "rope_parameters": [
                    "type": "default", "mrope_section": [8, 4, 4],
                    "rope_theta": 100000.0, "partial_rotary_factor": 1.0,
                ],
            ],
            "vision_config": [
                "model_type": "qwen3_vl", "depth": 1, "hidden_size": 32,
                "intermediate_size": 64, "out_hidden_size": 512, "num_heads": 2,
                "patch_size": 16, "spatial_merge_size": 2, "temporal_patch_size": 2,
                "num_position_embeddings": 64,
            ],
        ]
    }

    private func model(_ values: [String: Any]? = nil) throws -> PrismHadamardQwen35 {
        let data = try JSONSerialization.data(withJSONObject: values ?? configuration())
        let result = try ModelRegistry.default.createModel(
            configuration: data, modelType: "prism_hadamard_qwen35")
        return try XCTUnwrap(result as? PrismHadamardQwen35)
    }

    private func weights(_ model: PrismHadamardQwen35) -> [String: MLXArray] {
        var weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        for (key, value) in weights where key.hasSuffix(".scales") {
            let prefix = String(key.dropLast("scales".count))
            let rows = value.dim(0)
            let width = value.dim(1) * 128
            let dense = MLXRandom.normal([rows, width], key: MLXRandom.key(12)) / 20
            let (weight, scales, biases) = quantized(
                dense.asType(.float16), groupSize: 128, bits: 2)
            weights[prefix + "weight"] = weight
            weights[key] = scales
            weights[prefix + "biases"] = biases
            if weights[prefix + "signs"] != nil { weights[prefix + "signs"] = signs(width) }
        }
        return weights
    }

    private func checkpoint(_ weights: [String: MLXArray], metadata: [String: String] = [:]) throws
        -> URL
    {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: directory) }
        try save(
            arrays: weights, metadata: metadata, url: directory.appending(path: "model.safetensors")
        )
        return directory
    }

    func testPackedCheckpointLoadsWithoutRequantizingOrFusingRotations() throws {
        let loaded = try model()
        let source = weights(loaded)
        let directory = try checkpoint(source)
        let report = try CheckpointLoader.load(
            modelDirectory: directory, model: loaded, quantization: .init(groupSize: 128, bits: 2))
        XCTAssertNil(report.optimizationError)
        let modules = Dictionary(
            loaded.languageModel.namedModules(), uniquingKeysWith: { first, _ in first })
        let attention = try XCTUnwrap(
            modules["model.layers.0.linear_attn"] as? Qwen35Language.GatedDeltaNet)
        XCTAssertFalse(attention.hasFusedInputProjection)
        let head = try XCTUnwrap(modules["lm_head"] as? Linear)
        let embedding = try XCTUnwrap(modules["model.embed_tokens"] as? Embedding)
        XCTAssertTrue(head is any Quantized)
        XCTAssertTrue(embedding is any Quantized)
        let input = MLXRandom.normal([1, 2, 512], key: MLXRandom.key(19)).asType(.float16)
        for path in ["lm_head", "model.layers.0.linear_attn.in_proj_a"] {
            let linear = try XCTUnwrap(modules[path] as? Linear)
            let prefix = "language_model.\(path)."
            var rotated = input
            if let signs = source[prefix + "signs"] {
                rotated = hadamardRotate(input, block: 512, signs: signs)
            }
            let expected = quantizedMM(
                rotated, source[prefix + "weight"]!, scales: source[prefix + "scales"]!,
                biases: source[prefix + "biases"]!, transpose: true, groupSize: 128, bits: 2)
            XCTAssertLessThan(abs(linear(input) - expected).max().item(Float.self), 1e-3)
        }
        let ids = MLXArray([1, 3]).reshaped(1, 2)
        let rows = dequantized(
            source["language_model.model.embed_tokens.weight"]!,
            scales: source["language_model.model.embed_tokens.scales"]!,
            biases: source["language_model.model.embed_tokens.biases"]!, groupSize: 128, bits: 2)
        let expected = hadamardRotate(
            rows[ids].asType(.float16), block: 512,
            signs: source["language_model.model.embed_tokens.signs"]!, inverse: true)
        XCTAssertLessThan(abs(embedding(ids) - expected).max().item(Float.self), 1e-3)

        let cache = try loaded.newCache(configuration: nil)
        let prepared = try loaded.prepare(
            LMInput(tokens: ids), cache: cache, state: nil, prefill: .init())
        guard case .logits(let output) = prepared else { return XCTFail("Expected logits") }
        let next = loaded(
            LMInput.Text(tokens: MLXArray([4]).reshaped(1, 1)), cache: cache, state: output.state)
        XCTAssertEqual(next.logits.shape, [1, 1, 32])
        XCTAssertTrue(isFinite(next.logits).all().item(Bool.self))
        XCTAssertEqual(cache[1].offset, 3)
    }

    func testInvalidOrMissingSignsFailCheckpointLoading() throws {
        let source = weights(try model())
        for path in ["lm_head", "model.embed_tokens"] {
            var invalid = source
            invalid["language_model.\(path).signs"] = MLXArray.zeros([512])
            let directory = try checkpoint(invalid, metadata: ["format": "mlx"])
            XCTAssertThrowsError(
                try CheckpointLoader.load(modelDirectory: directory, model: model())
            ) {
                guard case PrismHadamardError.invalidSigns(path) = $0 else {
                    return XCTFail("Unexpected error: \($0)")
                }
            }
        }
        var missing = source
        missing.removeValue(forKey: "language_model.model.embed_tokens.signs")
        XCTAssertThrowsError(
            try CheckpointLoader.load(modelDirectory: checkpoint(missing), model: model()))
    }

    func testSanitizationPreservesConvertedLayoutAndWidensOnlyPackedScales() throws {
        let model = try model()
        let weights: [String: MLXArray] = [
            "language_model.lm_head.scales": MLXArray([Float16(0.5)]),
            "language_model.lm_head.biases": MLXArray([Float16(-1)]),
            "language_model.model.norm.weight": MLXArray([Float16(2)]),
            "language_model.model.layers.0.linear_attn.conv1d.weight": MLXArray.ones([4, 1, 4]),
        ]
        for metadata in [[:], ["format": "mlx"]] {
            let sanitized = model.sanitize(weights: weights, metadata: metadata)
            XCTAssertEqual(Set(sanitized.keys), Set(weights.keys))
            XCTAssertEqual(sanitized["language_model.lm_head.scales"]!.dtype, .float32)
            XCTAssertEqual(sanitized["language_model.lm_head.biases"]!.dtype, .float32)
            XCTAssertTrue(
                arrayEqual(
                    sanitized["language_model.model.norm.weight"]!,
                    weights["language_model.model.norm.weight"]!
                ).item(Bool.self))
            XCTAssertEqual(
                sanitized["language_model.model.layers.0.linear_attn.conv1d.weight"]!.shape,
                [4, 1, 4])
        }
    }

    func testManifestValidationAndConfigurationEncoding() throws {
        var config = configuration()
        config["schema_version"] = 1
        XCTAssertThrowsError(try model(config))
        config = configuration()
        var modules = config["modules"] as! [[String: Any]]
        modules.append(modules[0])
        config["modules"] = modules
        XCTAssertThrowsError(try model(config)) {
            guard case PrismHadamardError.duplicateModule("lm_head") = $0 else {
                return XCTFail("Unexpected error: \($0)")
            }
        }
        config = configuration()
        modules = config["modules"] as! [[String: Any]]
        modules[0]["block"] = 1024
        config["modules"] = modules
        XCTAssertThrowsError(try model(config))

        let decoded = try JSONDecoder().decode(
            PrismHadamardQwen35Configuration.self,
            from: JSONSerialization.data(withJSONObject: configuration()))
        let encoded = try JSONEncoder().encode(decoded)
        XCTAssertNoThrow(
            try ModelRegistry.default.createModel(
                configuration: encoded, modelType: "prism_hadamard_qwen35"))
    }
}
