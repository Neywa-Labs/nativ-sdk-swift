import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class LFM2ExpertStreamingTests: XCTestCase {
    func testMixedConversionAndGroupedDecodePreserveResidentResults() throws {
        MLXRandom.seed(47)
        let data = try ModelTestConfigurations.configuration(
            "_text", overrides: [
                "model_type": "lfm2_moe", "num_hidden_layers": 3, "num_dense_layers": 1,
                "full_attn_idxs": [2], "use_expert_bias": true, "conv_bias": false,
                "moe_intermediate_size": 64,
            ])
        let configuration = try JSONDecoder().decode(LFM2MoEConfiguration.self, from: data)
        let source = try directory()
        try data.write(to: source.appendingPathComponent("config.json"))
        let original = LFM2MoEModel(configuration)
        original.update(parameters: original.parameters().mapValues { key, value in
            key.hasSuffix("expert_bias") ? value : (value.ndim > 1 ? value * 0.1 : value)
        })
        try save(
            arrays: Dictionary(uniqueKeysWithValues: original.parameters().flattened()),
            metadata: ["format": "mlx"], url: source.appendingPathComponent("model.safetensors"))
        let converted = source.appendingPathComponent("mixed")
        _ = try ModelConversion.convert(
            source: source, destination: converted,
            quantization: .init(groupSize: 64, bits: 4), recipe: .mixed)
        let base = try JSONDecoder().decode(
            BaseConfiguration.self,
            from: Data(contentsOf: converted.appendingPathComponent("config.json")))
        let metadata = try XCTUnwrap(base.perLayerQuantization)
        for (path, bits, group, mode) in [
            ("model.embed_tokens", 4, 64, QuantizationMode.affine),
            ("model.layers.1.feed_forward.gate", 8, 64, .affine),
            ("model.layers.1.feed_forward.switch_mlp.gate_proj", 4, 32, .mxfp4),
        ] {
            let q = try XCTUnwrap(metadata.quantization(layer: path))
            XCTAssertEqual(q.bits, bits)
            XCTAssertEqual(q.groupSize, group)
            XCTAssertEqual(q.mode, mode)
        }
        let resident = LFM2MoEModel(configuration)
        try CheckpointLoader.load(
            modelDirectory: converted, model: resident,
            quantization: metadata.quantization, perLayerQuantization: metadata)
        // Exercise whole-expert batching, one-expert batches and the small-cache fallback.
        for (budget, batch, prefetch) in [(65_536, 4, true), (65_536, 1, false), (3_000, 4, false)] {
            let streamed = LFM2MoEModel(configuration)
            try CheckpointLoader.load(
                modelDirectory: converted, model: streamed,
                quantization: metadata.quantization, perLayerQuantization: metadata,
                expertStreaming: .init(
                    cacheSizeBytes: budget, prefetch: prefetch, expertBatchSize: batch))
            let left = try resident.newCache(configuration: nil)
            let right = try streamed.newCache(configuration: nil)
            let schedule = try XCTUnwrap(
                streamed.modules().compactMap { $0 as? StreamingSwitchLinear }.first?.schedule)
            for tokens in [[1, 3, 5, 7], [9], [11], [13, 15, 17], [19]] {
                let before = schedule.evaluation.evaluationCount
                let input = MLXArray(tokens, [1, tokens.count])
                let a = resident(input, cache: left)
                let b = streamed(input, cache: right)
                eval(a, b)
                try InferenceErrorCheck(streamed).check()
                if tokens.count == 1 {
                    let expected = budget > 3_000 ? 2 : 2 * 3 * configuration.numExpertsPerToken
                    XCTAssertEqual(schedule.evaluation.evaluationCount - before, expected)
                }
                XCTAssertEqual(schedule.evaluation.pendingWeightBytes, 0)
                XCTAssertLessThanOrEqual(schedule.evaluation.peakPendingWeightBytes, budget)
                XCTAssertTrue(allClose(a, b, rtol: 0.01, atol: 0.002).item(Bool.self))
                XCTAssertEqual(argMax(a, axis: -1).asArray(Int.self), argMax(b, axis: -1).asArray(Int.self))
                for projection in streamed.modules().compactMap({ $0 as? StreamingSwitchLinear }) {
                    XCTAssertLessThanOrEqual(projection.schedule.store.residentBytes, budget)
                }
            }
        }
        // A second conversion preserves the packed experts and resolved recipe.
        let repeated = source.appendingPathComponent("repeated")
        let result = try ModelConversion.convert(source: converted, destination: repeated)
        XCTAssertEqual(result.quantizedMatrixCount, 0)
        let first = try loadWeightArrays(urls: safetensorWeightURLs(in: converted)).weights
        let second = try loadWeightArrays(urls: safetensorWeightURLs(in: repeated)).weights
        for (name, value) in first {
            XCTAssertTrue(arrayEqual(value, try XCTUnwrap(second[name])).item(Bool.self), name)
        }
    }

    private func model() throws -> LFM2MoEModel {
        let data = try ModelTestConfigurations.configuration(
            "_text",
            overrides: [
                "model_type": "lfm2_moe", "num_hidden_layers": 3, "num_dense_layers": 1,
                "full_attn_idxs": [2], "use_expert_bias": true, "conv_bias": false,
            ])
        return try LFM2MoEModel(JSONDecoder().decode(LFM2MoEConfiguration.self, from: data))
    }

    private func directory() throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: url) }
        return url
    }

    private func rawWeights(_ model: LFM2MoEModel, originalNames: Bool = true) -> [String: MLXArray]
    {
        var weights: [String: MLXArray] = [:]
        for (name, value) in model.parameters().flattened() {
            if name.contains(".switch_mlp.") {
                for expert in 0 ..< model.configuration.numExperts {
                    let aliases = model.separateExpertWeightNames(for: name, expert: expert)
                    weights[originalNames ? aliases[0] : aliases[1]] = value[expert]
                }
            } else {
                var key = name
                if originalNames {
                    for (native, raw) in ["gate_proj": "w1", "down_proj": "w2", "up_proj": "w3"] {
                        key = key.replacingOccurrences(
                            of: ".\(native).weight", with: ".\(raw).weight")
                    }
                }
                weights[key] =
                    name.hasSuffix("conv.conv.weight") ? value.transposed(0, 2, 1) : value
            }
        }
        return weights
    }

    private func saveShards(_ weights: [String: MLXArray], at directory: URL) throws {
        var shards = Array(repeating: [String: MLXArray](), count: 3)
        for (index, entry) in weights.sorted(by: { $0.key < $1.key }).enumerated() {
            shards[index % shards.count][entry.key] = entry.value
        }
        for (index, shard) in shards.enumerated() {
            try save(
                arrays: shard, url: directory.appendingPathComponent("model-\(index).safetensors"))
        }
    }

    func testSeparateExpertsAcrossShardsPreserveBF16PrefillDecodeAndCacheBudget() throws {
        MLXRandom.seed(47)
        for dtype in [DType.float32, .bfloat16] {
            for originalNames in [true, false] {
                let original = try model()
                original.update(
                    parameters: original.parameters().mapValues { key, value in
                        key.hasSuffix("expert_bias")
                            ? value : (value.ndim > 1 ? value * 0.1 : value).asType(dtype)
                    })
                let url = try directory()
                try saveShards(rawWeights(original, originalNames: originalNames), at: url)
                let resident = try model()
                try CheckpointLoader.load(modelDirectory: url, model: resident)
                let streamed = try model()
                let budget = 16_384
                try CheckpointLoader.load(
                    modelDirectory: url, model: streamed,
                    expertStreaming: .init(
                        cacheSizeBytes: budget, prefetch: originalNames, expertBatchSize: 2))
                XCTAssertFalse(
                    streamed.parameters().flattened().contains { $0.0.contains("switch_mlp") })
                let projections = streamed.modules().compactMap { $0 as? StreamingSwitchLinear }
                XCTAssertEqual(projections.count, 6)
                let store = try XCTUnwrap(projections.first?.schedule.store)
                XCTAssertEqual(store.excludedWeightNames.count, 24)
                let leftCache = try resident.newCache(configuration: nil)
                let rightCache = try streamed.newCache(configuration: nil)
                for tokens in [[1, 3, 5, 7, 9], [11], [13, 15, 17], [19]] {
                    let input = MLXArray(tokens, [1, tokens.count])
                    let lhs = resident(input, cache: leftCache)
                    let rhs = streamed(input, cache: rightCache)
                    eval(lhs, rhs)
                    try InferenceErrorCheck(streamed).check()
                    XCTAssertTrue(allClose(lhs, rhs, rtol: 0.01, atol: 0.002).item(Bool.self))
                    XCTAssertEqual(
                        argMax(lhs, axis: -1).asArray(Int.self),
                        argMax(rhs, axis: -1).asArray(Int.self))
                    for (left, right) in zip(leftCache, rightCache) {
                        XCTAssertEqual(left.offset, right.offset)
                        for (a, b) in zip(left.state, right.state) {
                            XCTAssertTrue(allClose(a, b, rtol: 0.01, atol: 0.002).item(Bool.self))
                        }
                    }
                    XCTAssertLessThanOrEqual(store.residentBytes, budget)
                }
                for stepSize in [256, 512] {
                    let input = LMInput(tokens: MLXArray((0 ..< 520).map { $0 % 96 }))
                    let options = ModelGenerationOptions(
                        prefill: .init(stepSize: stepSize), sampling: .init(temperature: 0),
                        output: .init(maxTokens: 8))
                    let lhs = try AutoregressiveIterator(
                        input: input, model: resident, options: options)
                    let rhs = try AutoregressiveIterator(
                        input: input, model: streamed, options: options)
                    XCTAssertEqual(Array(lhs), Array(rhs))
                    XCTAssertNil(rhs.failure)
                }
            }
        }
    }

    func testInvalidSeparateExpertsFailBeforeInstallingProjections() throws {
        let original = try model()
        let weights = rawWeights(original)
        let prefix = "model.layers.1.feed_forward"
        let first = prefix + ".experts.0.w1.weight"
        let last = prefix + ".experts.3.w1.weight"
        let mutations: [(String, (inout [String: MLXArray]) -> Void)] = [
            ("missing first", { $0.removeValue(forKey: first) }),
            ("missing last", { $0.removeValue(forKey: last) }),
            ("wrong shape", { $0[last] = MLXArray.zeros([31, 64]) }),
            ("mixed precision", { $0[last] = $0[last]!.asType(.bfloat16) }),
            ("duplicate alias", { $0[prefix + ".experts.0.gate_proj.weight"] = $0[first] }),
            (
                "stacked and separate",
                {
                    $0[prefix + ".switch_mlp.gate_proj.weight"] = MLXArray.zeros([4, 32, 64])
                }
            ),
        ]
        for (name, mutate) in mutations {
            let url = try directory()
            var invalid = weights
            mutate(&invalid)
            try saveShards(invalid, at: url)
            let streamed = try model()
            XCTAssertThrowsError(
                try CheckpointLoader.load(
                    modelDirectory: url, model: streamed, expertStreaming: .init()), name)
            XCTAssertFalse(streamed.modules().contains { $0 is StreamingSwitchLinear }, name)
        }
    }

    func testTruncatedSeparateExpertShardStopsGeneration() throws {
        let url = try directory()
        try saveShards(rawWeights(model()), at: url)
        let streamed = try model()
        try CheckpointLoader.load(
            modelDirectory: url, model: streamed,
            expertStreaming: .init(prefetch: false))
        for shard in 0 ..< 3 {
            let handle = try FileHandle(
                forWritingTo: url.appendingPathComponent("model-\(shard).safetensors"))
            try handle.truncate(atOffset: 0)
            try handle.close()
        }
        XCTAssertThrowsError(
            try AutoregressiveIterator(
                input: LMInput(tokens: MLXArray([1, 2, 3])), model: streamed,
                options: .init(prefill: .init(stepSize: 2))))
    }
}
