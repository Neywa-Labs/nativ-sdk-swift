import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class ModelConversionTests: XCTestCase {
    private let expert = "language_model.model.layers.0.mlp.switch_mlp.gate_proj.weight"
    private let embedding = "language_model.model.embed_tokens"
    private let vision = "vision_tower.encoder.layers.0.self_attn.q_proj"
    private let router = "language_model.model.layers.0.mlp.gate"
    private let packed = "language_model.packed"

    private var tinyConfig: Data {
        Data(
            """
            {"model_type":"minicpmv4_7", "text_config": {
              "hidden_size":64,"intermediate_size":128,"num_hidden_layers":2,
              "num_attention_heads":2,"num_key_value_heads":1,"head_dim":32,
              "vocab_size":128,"full_attention_interval":2,"tie_word_embeddings":false,
              "linear_num_value_heads":2,"linear_num_key_heads":1,
              "linear_key_head_dim":32,"linear_value_head_dim":32,
              "num_experts":4,"num_experts_per_tok":2,
              "shared_expert_intermediate_size":64,"moe_intermediate_size":64,
              "rope_parameters":{"partial_rotary_factor":1.0,"mrope_section":[4,6,6]}},
             "vision_config":{"hidden_size":8,"intermediate_size":16,"num_hidden_layers":2,
              "num_attention_heads":2,"patch_size":2,"image_size":8}}
            """.utf8)
    }

    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: url) }
        return url
    }

    private func writeJSON(_ object: [String: Any], _ url: URL) throws {
        try JSONSerialization.data(withJSONObject: object).write(to: url)
    }

    private func fixture(_ root: URL, dtype: DType = .float32) throws -> [String: MLXArray] {
        let matrix = (MLXArray(0 ..< 512).reshaped(8, 64).asType(.float32) / 512 - 0.5)
            .asType(dtype)
        let q = quantized(matrix, groupSize: 32, bits: 4)
        let weights: [String: MLXArray] = [
            embedding + ".weight": matrix, vision + ".weight": matrix,
            router + ".weight": matrix,
            expert: stacked([matrix, -matrix]),
            packed + ".weight": q.wq, packed + ".scales": q.scales,
            packed + ".biases": q.biases!,
            "norm.weight": MLXArray.ones([64], dtype: dtype),
            "unaligned.weight": MLXArray.ones([2, 17], dtype: dtype),
        ]
        // Scales and packed weights can be in different source shards.
        let a = weights.filter { !$0.key.hasSuffix(".scales") }
        let b = weights.filter { $0.key.hasSuffix(".scales") }
        for (file, arrays) in [("model-a.safetensors", a), ("model-b.safetensors", b)] {
            try save(
                arrays: arrays, metadata: ["format": "mlx"], url: root.appendingPathComponent(file))
        }
        var config = try XCTUnwrap(JSONSerialization.jsonObject(with: tinyConfig) as? [String: Any])
        config["custom_field"] = ["keep": 47]
        config["quantization"] = [
            "group_size": 32, "bits": 4, "mode": "affine", "norm": false,
        ]
        try writeJSON(config, root.appendingPathComponent("config.json"))
        try writeJSON(
            [
                "weight_map": Dictionary(
                    uniqueKeysWithValues: weights.keys.map {
                        (
                            $0,
                            $0.hasSuffix(".scales") ? "model-b.safetensors" : "model-a.safetensors"
                        )
                    })
            ],
            root.appendingPathComponent("model.safetensors.index.json"))
        return weights
    }

    private func readWeights(_ directory: URL) throws -> [String: MLXArray] {
        try loadWeightArrays(urls: safetensorWeightURLs(in: directory)).weights
    }

    private func assertUnchanged(
        _ actual: MLXArray?, _ expected: MLXArray, file: StaticString = #filePath,
        line: UInt = #line
    ) throws {
        let actual = try XCTUnwrap(actual, file: file, line: line)
        XCTAssertEqual(actual.dtype, expected.dtype, file: file, line: line)
        XCTAssertEqual(actual.shape, expected.shape, file: file, line: line)
        XCTAssertTrue(arrayEqual(actual, expected).item(Bool.self), file: file, line: line)
    }

    func testMixedRecipePreservesExpertsAndPackedWeightsAcrossPrecisions() throws {
        for dtype: DType in [.float32, .float16, .bfloat16] {
            for denseBits in [4, 8] {
                let root = try directory()
                let original = try fixture(root, dtype: dtype)
                let output = root.appendingPathComponent("prepared")
                let result = try ModelConversion.convert(
                    source: root, destination: output,
                    quantization: .init(groupSize: 64, bits: denseBits),
                    recipe: .mixed,
                    shardSizeBytes: 1024)
                XCTAssertEqual(result.quantizedMatrixCount, 3)
                XCTAssertGreaterThan(result.shardCount, 1)
                let arrays = try readWeights(output)
                for name in [
                    expert, packed + ".weight", packed + ".scales", packed + ".biases",
                    "norm.weight", "unaligned.weight",
                ] {
                    try assertUnchanged(arrays[name], original[name]!)
                }
                let config = try JSONDecoder().decode(
                    BaseConfiguration.self,
                    from: Data(contentsOf: output.appendingPathComponent("config.json")))
                for (prefix, bits) in [(embedding, denseBits), (vision, 8), (router, 8)] {
                    let q = try XCTUnwrap(config.perLayerQuantization?.quantization(layer: prefix))
                    XCTAssertEqual(q.bits, bits)
                    XCTAssertEqual(q.groupSize, 64)
                    XCTAssertEqual(q.mode, .affine)
                    let weight = try XCTUnwrap(arrays[prefix + ".weight"])
                    XCTAssertEqual(weight.dtype, .uint32)
                    XCTAssertEqual(weight.shape, [8, 64 * bits / 32])
                    let restored = dequantized(
                        weight, scales: arrays[prefix + ".scales"]!,
                        biases: arrays[prefix + ".biases"],
                        groupSize: 64, bits: bits)
                    XCTAssertLessThan(
                        abs(restored - original[prefix + ".weight"]!).max().item(Float.self), 0.02)
                }
                XCTAssertEqual(result.tensorBytes, arrays.values.reduce(0) { $0 + $1.nbytes })
                let source = try readWeights(root)
                for (name, value) in original { try assertUnchanged(source[name], value) }
            }
        }
    }

    func testDefaultMiniCPMHonorsModelParametersSkipsAndRequestedDefaults() throws {
        for settings: BaseConfiguration.Quantization in [
            .init(groupSize: 64, bits: 4), .init(groupSize: 64, bits: 8),
            .init(groupSize: 32, bits: 4, mode: .mxfp4),
        ] {
            let root = try directory()
            let original = try fixture(root)
            let output = root.appendingPathComponent("converted")
            let result = try ModelConversion.convert(
                source: root, destination: output, quantization: settings)
            XCTAssertEqual(result.quantizedMatrixCount, 2)
            let arrays = try readWeights(output)
            let config = try JSONDecoder().decode(
                BaseConfiguration.self,
                from: Data(contentsOf: output.appendingPathComponent("config.json")))
            for (path, expected): (String, BaseConfiguration.Quantization) in [
                (router, .init(groupSize: 64, bits: 8)),
                (String(expert.dropLast(7)), settings),
            ] {
                let q = try XCTUnwrap(config.perLayerQuantization?.quantization(layer: path))
                XCTAssertEqual(q.bits, expected.bits, path)
                XCTAssertEqual(q.mode, expected.mode, path)
                XCTAssertEqual(q.groupSize, expected.groupSize, path)
                let direct = quantized(
                    original[path + ".weight"]!, groupSize: expected.groupSize,
                    bits: expected.bits, mode: expected.mode)
                try assertUnchanged(arrays[path + ".weight"], direct.wq)
                try assertUnchanged(arrays[path + ".scales"], direct.scales)
            }
            for name in [embedding + ".weight", vision + ".weight", packed + ".weight"] {
                try assertUnchanged(arrays[name], original[name]!)
            }
        }
    }

    func testNamedRecipeOverridesMiniCPMModelParametersAndSkips() throws {
        let root = try directory()
        let original = try fixture(root)
        let output = root.appendingPathComponent("mixed-3-4")
        let result = try ModelConversion.convert(source: root, destination: output, recipe: .mixed3_4)
        XCTAssertEqual(result.quantizedMatrixCount, 3)
        let arrays = try readWeights(output)
        let config = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: Data(contentsOf: output.appendingPathComponent("config.json")))
        // The named recipe replaces both the 8-bit router rule and the embedding exclusion.
        for (path, bits) in [(router, 3), (embedding, 4), (String(expert.dropLast(7)), 3)] {
            let q = try XCTUnwrap(config.perLayerQuantization?.quantization(layer: path))
            XCTAssertEqual(q.bits, bits, path)
            XCTAssertEqual(q.groupSize, 64, path)
            XCTAssertEqual(q.mode, .affine, path)
            let direct = quantized(original[path + ".weight"]!, groupSize: 64, bits: bits)
            try assertUnchanged(arrays[path + ".weight"], direct.wq)
            try assertUnchanged(arrays[path + ".scales"], direct.scales)
        }
        for name in [vision + ".weight", packed + ".weight", packed + ".scales", packed + ".biases"] {
            try assertUnchanged(arrays[name], original[name]!)
        }
    }

    func testDefaultLFMHonorsModelRouterParametersAcrossModesAndReloads() throws {
        let data = try ModelTestConfigurations.configuration(
            "_text", overrides: [
                "model_type": "lfm2_moe", "moe_intermediate_size": 64,
                "num_dense_layers": 1, "use_expert_bias": false, "conv_bias": false,
            ])
        let config = try JSONDecoder().decode(LFM2MoEConfiguration.self, from: data)
        let root = try directory()
        try data.write(to: root.appendingPathComponent("config.json"))
        let original = LFM2MoEModel(config)
        try save(
            arrays: Dictionary(uniqueKeysWithValues: original.parameters().flattened()),
            metadata: ["format": "mlx"], url: root.appendingPathComponent("model.safetensors"))
        for settings: BaseConfiguration.Quantization in [
            .init(groupSize: 64, bits: 4), .init(groupSize: 64, bits: 8),
            .init(groupSize: 32, bits: 4, mode: .mxfp4),
            .init(groupSize: 32, bits: 8, mode: .mxfp8),
        ] {
            let output = root.appendingPathComponent("\(settings.mode.rawValue)-\(settings.bits)")
            _ = try ModelConversion.convert(source: root, destination: output, quantization: settings)
            let metadata = try JSONDecoder().decode(
                BaseConfiguration.self,
                from: Data(contentsOf: output.appendingPathComponent("config.json")))
            let arrays = try readWeights(output)
            var experts = 0
            var routers = 0
            for (path, module) in original.namedModules() where module is Quantizable {
                let weight = try XCTUnwrap(
                    Dictionary(uniqueKeysWithValues: module.parameters().flattened())["weight"])
                let expectedSettings: BaseConfiguration.Quantization =
                    path.hasSuffix("feed_forward.gate") ? .init(groupSize: 64, bits: 8) : settings
                guard weight.dim(-1).isMultiple(of: expectedSettings.groupSize) else { continue }
                let q = try XCTUnwrap(metadata.perLayerQuantization?.quantization(layer: path), path)
                XCTAssertEqual(q.bits, expectedSettings.bits, path)
                XCTAssertEqual(q.mode, expectedSettings.mode, path)
                XCTAssertEqual(q.groupSize, expectedSettings.groupSize, path)
                let expected = quantized(
                    weight, groupSize: expectedSettings.groupSize, bits: expectedSettings.bits,
                    mode: expectedSettings.mode)
                try assertUnchanged(arrays[path + ".weight"], expected.wq)
                try assertUnchanged(arrays[path + ".scales"], expected.scales)
                if path.contains("switch_mlp") { experts += 1 }
                if path.hasSuffix("feed_forward.gate") { routers += 1 }
            }
            XCTAssertGreaterThan(experts, 0)
            XCTAssertGreaterThan(routers, 0)
            let loaded = LFM2MoEModel(config)
            try CheckpointLoader.load(
                modelDirectory: output, model: loaded,
                perLayerQuantization: metadata.perLayerQuantization)
            let direct = LFM2MoEModel(config)
            try CheckpointLoader.load(modelDirectory: root, model: direct)
            try quantizeByParent(model: direct) { path, _ in
                path.hasSuffix("feed_forward.gate")
                    ? (64, 8, .affine) : (settings.groupSize, settings.bits, settings.mode)
            }
            let input = MLXArray([1, 2, 3], [1, 3])
            XCTAssertTrue(allClose(direct(input, cache: nil), loaded(input, cache: nil)).item(Bool.self))
        }
        let mixed = root.appendingPathComponent("mixed-4-8")
        _ = try ModelConversion.convert(source: root, destination: mixed, recipe: .mixed4_8)
        let metadata = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: Data(contentsOf: mixed.appendingPathComponent("config.json")))
        for (path, bits) in [
            ("model.layers.0.feed_forward.w2", 4),
            ("model.layers.1.feed_forward.switch_mlp.down_proj", 8),
            ("model.layers.1.feed_forward.switch_mlp.gate_proj", 4),
            ("model.layers.1.feed_forward.gate", 4), ("model.embed_tokens", 8),
        ] {
            let q = try XCTUnwrap(metadata.perLayerQuantization?.quantization(layer: path))
            XCTAssertEqual(q.bits, bits, path)
            XCTAssertEqual(q.mode, .affine, path)
        }
    }

    func testPythonMixedRecipesUseExpectedLayerScheduleAndReload() throws {
        let data = try ModelTestConfigurations.configuration(
            "_text", overrides: [
                "model_type": "qwen2", "num_hidden_layers": 16, "intermediate_size": 64,
            ])
        let config = try JSONDecoder().decode(Qwen2Configuration.self, from: data)
        let root = try directory()
        try data.write(to: root.appendingPathComponent("config.json"))
        let original = Qwen2Model(config)
        try save(
            arrays: Dictionary(uniqueKeysWithValues: original.parameters().flattened()),
            metadata: ["format": "mlx"], url: root.appendingPathComponent("model.safetensors"))
        let highLayers: Set<Int> = [0, 1, 4, 7, 10, 13, 14, 15]
        for (recipe, low, high): (QuantizationRecipe, Int, Int) in [
            (.mixed2_6, 2, 6), (.mixed3_4, 3, 4), (.mixed3_5, 3, 5),
            (.mixed3_6, 3, 6), (.mixed3_8, 3, 8), (.mixed4_6, 4, 6), (.mixed4_8, 4, 8),
        ] {
            let output = root.appendingPathComponent(recipe.rawValue)
            _ = try ModelConversion.convert(source: root, destination: output, recipe: recipe)
            let metadata = try JSONDecoder().decode(
                BaseConfiguration.self,
                from: Data(contentsOf: output.appendingPathComponent("config.json")))
            XCTAssertEqual(metadata.perLayerQuantization?.quantization?.bits, low)
            let arrays = try readWeights(output)
            for (path, module) in original.namedModules() where module is Quantizable {
                let layer = path.split(separator: ".").compactMap { Int($0) }.first
                let moreBits = path == "model.embed_tokens" || path == "lm_head"
                    || (layer.map { highLayers.contains($0) } == true
                        && (path.hasSuffix("v_proj") || path.hasSuffix("down_proj")))
                let bits = moreBits ? high : low
                let q = try XCTUnwrap(metadata.perLayerQuantization?.quantization(layer: path), path)
                XCTAssertEqual(q.bits, bits, path)
                XCTAssertEqual(q.groupSize, 64)
                XCTAssertEqual(q.mode, .affine)
                let weight = try XCTUnwrap(
                    Dictionary(uniqueKeysWithValues: module.parameters().flattened())["weight"])
                let expected = quantized(weight, groupSize: 64, bits: bits)
                try assertUnchanged(arrays[path + ".weight"], expected.wq)
                try assertUnchanged(arrays[path + ".scales"], expected.scales)
            }
            let loaded = Qwen2Model(config)
            try CheckpointLoader.load(
                modelDirectory: output, model: loaded,
                perLayerQuantization: metadata.perLayerQuantization)
            let direct = Qwen2Model(config)
            try CheckpointLoader.load(modelDirectory: root, model: direct)
            try quantizeByParent(model: direct) { path, _ in
                metadata.perLayerQuantization?.quantization(layer: path).map {
                    ($0.groupSize, $0.bits, $0.mode)
                }
            }
            let input = MLXArray([1, 2, 3], [1, 3])
            XCTAssertTrue(allClose(direct(input, cache: nil), loaded(input, cache: nil)).item(Bool.self))
        }
        XCTAssertThrowsError(try ModelConversion.convert(
            source: root, destination: root.appendingPathComponent("unsupported"), recipe: .mixed))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("unsupported").path))
        XCTAssertThrowsError(try ModelConversion.convert(
            source: root, destination: root.appendingPathComponent("conflict"), recipe: .mixed4_8,
            predicate: { _, _ in .useDefault }))
    }

    func testSidecarsMetadataIndexAndRepeatedPreparation() throws {
        let root = try directory()
        _ = try fixture(root)
        let sidecars = [
            "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
            "chat_template.jinja", "processor_config.json", "preprocessor_config.json",
            "generation_config.json",
        ]
        let blobs = try directory()
        for (index, name) in sidecars.enumerated() {
            if index % 2 == 0 {
                let blob = blobs.appendingPathComponent(name)
                try Data(name.utf8).write(to: blob)
                try FileManager.default.createSymbolicLink(
                    at: root.appendingPathComponent(name), withDestinationURL: blob)
            } else {
                try Data(name.utf8).write(to: root.appendingPathComponent(name))
            }
        }
        let output = root.appendingPathComponent("first")
        _ = try ModelConversion.convert(
            source: root, destination: output, shardSizeBytes: 1024)
        for name in sidecars {
            XCTAssertEqual(
                try Data(contentsOf: output.appendingPathComponent(name)), Data(name.utf8))
            XCTAssertEqual(
                try output.appendingPathComponent(name).resourceValues(forKeys: [.isSymbolicLinkKey]
                )
                .isSymbolicLink, false)
        }
        let configData = try Data(contentsOf: output.appendingPathComponent("config.json"))
        let config = try XCTUnwrap(JSONSerialization.jsonObject(with: configData) as? [String: Any])
        XCTAssertEqual(config["custom_field"] as? [String: Int], ["keep": 47])
        let q = try XCTUnwrap(config["quantization"] as? [String: Any])
        XCTAssertEqual(q["bits"] as? Int, 4)
        XCTAssertEqual(q["group_size"] as? Int, 32)
        XCTAssertEqual(q["norm"] as? Bool, false)
        let index = try XCTUnwrap(
            JSONSerialization.jsonObject(
                with: Data(
                    contentsOf: output.appendingPathComponent("model.safetensors.index.json")))
                as? [String: Any])
        let map = try XCTUnwrap(index["weight_map"] as? [String: String])
        let arrays = try readWeights(output)
        XCTAssertEqual(Set(map.keys), Set(arrays.keys))
        XCTAssertEqual(
            (index["metadata"] as? [String: Int])?["total_size"],
            arrays.values.reduce(0) { $0 + $1.nbytes })
        for file in Set(map.values) {
            let (weights, metadata) = try loadArraysAndMetadata(
                url: output.appendingPathComponent(file))
            XCTAssertEqual(metadata["format"], "mlx")
            XCTAssertTrue(weights.keys.allSatisfy { map[$0] == file })
        }
        let repeated = root.appendingPathComponent("second")
        let result = try ModelConversion.convert(source: output, destination: repeated)
        XCTAssertEqual(result.quantizedMatrixCount, 0)
        let second = try readWeights(repeated)
        for (name, value) in arrays { try assertUnchanged(second[name], value) }
        XCTAssertEqual(
            try Data(contentsOf: repeated.appendingPathComponent("config.json")), configData)
    }

    func testInvalidInputsAreRejectedWithoutPublishingOutput() throws {
        for scenario in [
            "model", "format", "missingShard", "missingTensor", "wrongShard", "duplicate",
            "emptyIndex", "bits", "missingQuantization",
        ] {
            let root = try directory()
            let original = try fixture(root)
            switch scenario {
            case "model":
                try writeJSON(["model_type": "other"], root.appendingPathComponent("config.json"))
            case "missingQuantization":
                try tinyConfig.write(to: root.appendingPathComponent("config.json"))
            case "format":
                try save(
                    arrays: original, metadata: ["format": "pt"],
                    url: root.appendingPathComponent("model-a.safetensors"))
            case "missingShard", "missingTensor", "wrongShard", "emptyIndex":
                var map = Dictionary(
                    uniqueKeysWithValues: original.keys.map {
                        (
                            $0,
                            $0.hasSuffix(".scales") ? "model-b.safetensors" : "model-a.safetensors"
                        )
                    })
                if scenario == "missingShard" { map[expert] = "absent.safetensors" }
                if scenario == "missingTensor" { map["absent.weight"] = "model-a.safetensors" }
                if scenario == "wrongShard" { map[expert] = "model-b.safetensors" }
                if scenario == "emptyIndex" { map = [:] }
                try writeJSON(
                    ["weight_map": map], root.appendingPathComponent("model.safetensors.index.json")
                )
            case "duplicate":
                try FileManager.default.removeItem(
                    at: root.appendingPathComponent("model.safetensors.index.json"))
                try save(
                    arrays: [expert: original[expert]!], metadata: ["format": "mlx"],
                    url: root.appendingPathComponent("model-c.safetensors"))
            default: break
            }
            let output = root.appendingPathComponent("output")
            XCTAssertThrowsError(
                try ModelConversion.convert(
                    source: root, destination: output,
                    quantization: .init(groupSize: 64, bits: scenario == "bits" ? 7 : 4)),
                scenario)
            XCTAssertFalse(FileManager.default.fileExists(atPath: output.path), scenario)
            XCTAssertFalse(
                try FileManager.default.contentsOfDirectory(atPath: root.path).contains {
                    $0.hasPrefix(".convert-")
                })
        }
    }

    func testExistingDestinationAndMidConversionFailurePreserveFiles() throws {
        let root = try directory()
        let original = try fixture(root)
        let sentinel = root.appendingPathComponent("output")
        try Data("keep".utf8).write(to: sentinel)
        XCTAssertThrowsError(
            try ModelConversion.convert(source: root, destination: sentinel))
        XCTAssertEqual(try Data(contentsOf: sentinel), Data("keep".utf8))
        try FileManager.default.removeItem(
            at: root.appendingPathComponent("model.safetensors.index.json"))
        try save(
            arrays: [
                "language_model.lm_head.weight": MLXArray.ones([2, 64]),
                "language_model.lm_head.biases": MLXArray.ones([2, 1]),
            ],
            metadata: ["format": "mlx"], url: root.appendingPathComponent("model-z.safetensors"))
        let output = root.appendingPathComponent("failed")
        XCTAssertThrowsError(
            try ModelConversion.convert(
                source: root, destination: output, recipe: .mixed, shardSizeBytes: 1024))
        XCTAssertFalse(FileManager.default.fileExists(atPath: output.path))
        XCTAssertFalse(
            try FileManager.default.contentsOfDirectory(atPath: root.path).contains {
                $0.hasPrefix(".convert-")
            })
        let retained = try readWeights(root)
        for (name, value) in original { try assertUnchanged(retained[name], value) }
    }

    func testConvertedModelLoadsAndStreamedLogitsMatchResident() throws {
        let root = try directory()
        let data = tinyConfig
        let config = try JSONDecoder().decode(MiniCPMVConfiguration.self, from: data)
        MLXRandom.seed(47)
        let model = MiniCPMV(config)
        // Unquantized input also needs a valid default quantization entry in its output config.
        try data.write(to: root.appendingPathComponent("config.json"))
        try save(
            arrays: Dictionary(uniqueKeysWithValues: model.parameters().flattened()),
            metadata: ["format": "mlx"], url: root.appendingPathComponent("model.safetensors"))
        let output = root.appendingPathComponent("prepared")
        let result = try ModelConversion.convert(
            source: root, destination: output, shardSizeBytes: 32_768)
        XCTAssertGreaterThan(result.quantizedMatrixCount, 0)
        let base = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: Data(contentsOf: output.appendingPathComponent("config.json")))
        let resident = MiniCPMV(config)
        let streamed = MiniCPMV(config)
        for (model, options): (MiniCPMV, ExpertStreamingOptions?) in [
            (resident, nil), (streamed, .init(cacheSizeBytes: 131_072)),
        ] {
            try CheckpointLoader.load(
                modelDirectory: output, model: model,
                quantization: base.perLayerQuantization?.quantization,
                perLayerQuantization: base.perLayerQuantization,
                expertStreaming: options)
        }
        let a = try resident.newCache(configuration: nil)
        let b = try streamed.newCache(configuration: nil)
        for tokens in [[1, 2, 3], [4], [5]] {
            let input = MLXArray(tokens, [1, tokens.count])
            let lhs = resident(input, cache: a)
            let rhs = streamed(input, cache: b)
            eval(lhs, rhs)
            try streamed.checkInferenceError()
            XCTAssertLessThan(abs(lhs - rhs).max().item(Float.self), 2e-4)
        }
    }

    func testMiniCPM46KeepsSensitiveAndMultimodalWeights() throws {
        let root = try directory()
        let original = try fixture(root)
        let url = root.appendingPathComponent("config.json")
        var config = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        config["model_type"] = "minicpmv4_6"
        try writeJSON(config, url)
        let output = root.appendingPathComponent("converted")
        _ = try ModelConversion.convert(source: root, destination: output)
        let arrays = try readWeights(output)
        for path in [embedding, vision] {
            try assertUnchanged(arrays[path + ".weight"], original[path + ".weight"]!)
            XCTAssertNil(arrays[path + ".scales"])
        }
        XCTAssertNotNil(arrays[router + ".scales"])
        // The 4.6 predicate does not apply the 4.7 expert-preservation policy.
        XCTAssertNotNil(arrays[String(expert.dropLast(7)) + ".scales"])
    }

    func testCustomPredicateAndShapeCompatibilityUseResolvedSettings() throws {
        let root = try directory()
        let original = try fixture(root)
        let output = root.appendingPathComponent("converted")
        let result = try ModelConversion.convert(
            source: root, destination: output,
            quantization: .init(groupSize: 128, bits: 4),
            predicate: { path, _ in
                if path == self.embedding { return .skip }
                if path == self.vision { return .parameters(.init(groupSize: 32, bits: 8)) }
                return .useDefault
            })
        XCTAssertEqual(result.quantizedMatrixCount, 1)
        let arrays = try readWeights(output)
        for name in [embedding + ".weight", router + ".weight", expert] {
            try assertUnchanged(arrays[name], original[name]!)
        }
        let base = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: Data(contentsOf: output.appendingPathComponent("config.json")))
        XCTAssertNil(base.perLayerQuantization?.quantization(layer: embedding))
        XCTAssertNil(base.perLayerQuantization?.quantization(layer: router))
        XCTAssertEqual(base.perLayerQuantization?.quantization(layer: vision)?.groupSize, 32)
        XCTAssertEqual(base.perLayerQuantization?.quantization(layer: vision)?.bits, 8)
        XCTAssertThrowsError(
            try ModelConversion.convert(
                source: root, destination: root.appendingPathComponent("invalid"),
                predicate: { _, _ in .parameters(.init(groupSize: 17, bits: 4)) }))
    }

    func testGenericDefaultsReloadAndMatchDirectQuantizationAcrossModes() throws {
        let data = Data(
            """
            {"model_type":"qwen2","hidden_size":64,"intermediate_size":128,
             "num_hidden_layers":1,"num_attention_heads":2,"num_key_value_heads":1,
             "vocab_size":128,"rms_norm_eps":0.00001,"tie_word_embeddings":false}
            """.utf8)
        let config = try JSONDecoder().decode(Qwen2Configuration.self, from: data)
        for settings: BaseConfiguration.Quantization in [
            .init(groupSize: 64, bits: 4), .init(groupSize: 32, bits: 3),
            .init(groupSize: 32, bits: 4, mode: .mxfp4),
            .init(groupSize: 32, bits: 8, mode: .mxfp8),
            .init(groupSize: 16, bits: 4, mode: .nvfp4),
        ] {
            let root = try directory()
            let original = Qwen2Model(config)
            try data.write(to: root.appendingPathComponent("config.json"))
            try save(
                arrays: Dictionary(uniqueKeysWithValues: original.parameters().flattened()),
                metadata: ["format": "mlx", "custom": "preserved"],
                url: root.appendingPathComponent("model.safetensors"))
            let output = root.appendingPathComponent("converted")
            let result = try ModelConversion.convert(
                source: root, destination: output, quantization: settings, shardSizeBytes: 2048)
            XCTAssertEqual(result.quantizedMatrixCount, 9)
            try quantizeByParent(
                model: original, groupSize: settings.groupSize, bits: settings.bits,
                mode: settings.mode)
            let loaded = Qwen2Model(config)
            let base = try JSONDecoder().decode(
                BaseConfiguration.self,
                from: Data(contentsOf: output.appendingPathComponent("config.json")))
            try CheckpointLoader.load(
                modelDirectory: output, model: loaded,
                perLayerQuantization: base.perLayerQuantization)
            let arrays = Dictionary(uniqueKeysWithValues: loaded.parameters().flattened())
            for (name, value) in original.parameters().flattened() {
                try assertUnchanged(arrays[name], value)
            }
            let input = MLXArray([1, 2, 3], [1, 3])
            XCTAssertTrue(
                allClose(original(input, cache: nil), loaded(input, cache: nil)).item(Bool.self))
            let metadata = try loadArraysAndMetadata(
                url: output.appendingPathComponent("model-00001.safetensors")
            ).1
            XCTAssertEqual(metadata["custom"], "preserved")
        }
    }
}
