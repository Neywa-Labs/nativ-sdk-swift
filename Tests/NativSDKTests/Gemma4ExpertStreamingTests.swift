import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class Gemma4ExpertStreamingTests: XCTestCase {
    private func configuration(moe: Bool = true) throws -> Gemma4Configuration {
        try JSONDecoder().decode(
            Gemma4Configuration.self,
            from: JSONSerialization.data(withJSONObject: [
                "model_type": "gemma4", "image_token_id": 63,
                "text_config": [
                    "hidden_size": 64, "intermediate_size": 64, "num_hidden_layers": 2,
                    "num_attention_heads": 4, "num_key_value_heads": 1,
                    "num_global_key_value_heads": 1, "head_dim": 16, "global_head_dim": 16,
                    "attention_k_eq_v": true, "vocab_size": 64,
                    "num_kv_shared_layers": 0, "hidden_size_per_layer_input": 0,
                    "sliding_window": 16, "use_double_wide_mlp": false,
                    "tie_word_embeddings": true, "enable_moe_block": moe,
                    "num_experts": 4, "top_k_experts": 2, "moe_intermediate_size": 64,
                    "layer_types": ["sliding_attention", "full_attention"],
                ],
                "vision_config": [
                    "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
                    "num_attention_heads": 2, "head_dim": 4, "patch_size": 2,
                    "pooling_kernel_size": 1, "position_embedding_size": 16,
                    "use_clipped_linears": false,
                ],
            ]))
    }

    private func pair(
        mode: QuantizationMode? = nil, fused: Bool = false, prefetch: Bool = true,
        batchSize: Int = 3, cacheSizeBytes: Int = 200_000, bits: Int = 4,
        sharedBuffers: Bool = false, separateExperts: Bool = false
    ) throws -> (Gemma4, Gemma4, URL) {
        let config = try configuration()
        let resident = withRandomState(MLXRandom.RandomState(seed: 47)) { Gemma4(config) }
        for layer in resident.languageModel.model.layers {
            // Nonuniform learned scales catch an accidental plain top-k/softmax replacement.
            try layer.router!.update(
                parameters: .unflattened([
                    "proj.weight": layer.router!.proj.weight,
                    "scale": MLXArray.linspace(0.5, 1.5, count: 64),
                    "per_expert_scale": MLXArray([Float(0.4), 0.8, 1.2, 1.6]),
                ]), verify: .all)
        }
        var q: BaseConfiguration.Quantization?
        if let mode {
            quantize(model: resident) { path, _ in
                path.contains(".switch_glu.") ? (32, bits, mode) : nil
            }
            q = .init(groupSize: 32, bits: bits, mode: mode)
        }
        let original = Dictionary(uniqueKeysWithValues: resident.parameters().flattened())
        var weights = original
        if separateExperts {
            for (name, value) in original where name.contains(".experts.switch_glu.") {
                weights[name] = nil
                for expert in 0 ..< 4 {
                    weights[
                        name.replacingOccurrences(
                            of: ".experts.switch_glu.", with: ".experts.\(expert).")] =
                        value[expert]
                }
            }
        }
        if fused {
            for layer in 0 ..< config.textConfiguration.hiddenLayers {
                let prefix = "language_model.model.layers.\(layer).experts."
                let gate = weights.removeValue(forKey: prefix + "switch_glu.gate_proj.weight")!
                let up = weights.removeValue(forKey: prefix + "switch_glu.up_proj.weight")!
                let down = weights.removeValue(forKey: prefix + "switch_glu.down_proj.weight")!
                let raw = "model.language_model.layers.\(layer).experts."
                weights[raw + "gate_up_proj"] = concatenated([gate, up], axis: -2)
                weights[raw + "down_proj"] = down
            }
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        addTeardownBlock { try FileManager.default.removeItem(at: directory) }
        try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
        let streamed = Gemma4(config)
        try CheckpointLoader.load(
            modelDirectory: directory, model: streamed, quantization: q,
            expertStreaming: .init(
                cacheSizeBytes: cacheSizeBytes, prefetch: prefetch, expertBatchSize: batchSize,
                sharedExpertBuffers: sharedBuffers, deferVisionWeights: sharedBuffers))
        XCTAssertFalse(streamed.parameters().flattened().contains { $0.0.contains(".experts.") })
        for (name, value) in streamed.parameters().flattened() {
            // Leave deferred vision lazy; text inference must not require materializing it.
            if sharedBuffers && (name.hasPrefix("vision_tower.") || name.hasPrefix("embed_vision."))
            {
                continue
            }
            XCTAssertTrue(arrayEqual(value, original[name]!).item(Bool.self), name)
        }
        return (resident, streamed, directory)
    }

    func testStreamedGELUAndRoutingMatchResidentAcrossEvictionAndDecode() throws {
        for mode: QuantizationMode? in [nil, .affine, .mxfp4] {
            for prefetch in [false, true] {
                // Float weights fit exactly one expert; quantized weights fit several.
                let (resident, streamed, _) = try pair(
                    mode: mode, prefetch: prefetch, cacheSizeBytes: 49_152)
                let lhsCache = try resident.newCache(configuration: nil)
                let rhsCache = try streamed.newCache(configuration: nil)
                for tokens in [Array(1 ... 20), [21], [22], [23]] {
                    let input = MLXArray(tokens, [1, tokens.count])
                    let lhs = resident(input, cache: lhsCache)
                    let rhs = streamed(input, cache: rhsCache)
                    eval(lhs, rhs)
                    try streamed.checkInferenceError()
                    XCTAssertTrue(allClose(lhs, rhs, rtol: 2e-4, atol: 2e-4).item(Bool.self))
                    XCTAssertEqual(lhsCache.map(\.offset), rhsCache.map(\.offset))
                }
            }
        }
    }

    func testOfficialFusedExpertLayoutMatchesResident() throws {
        for batch in [1, 3] {
            let (resident, streamed, _) = try pair(fused: true, batchSize: batch)
            let tokens = MLXArray([1, 3, 5, 7], [1, 4])
            let lhs = resident(tokens, cache: try resident.newCache(configuration: nil))
            let rhs = streamed(tokens, cache: try streamed.newCache(configuration: nil))
            // The single-expert fused path uses GEMM; resident experts use gather
            // kernels. Bound their float32 accumulation drift and require equal tokens.
            XCTAssertLessThan(abs(lhs - rhs).max().item(Float.self), batch == 1 ? 2e-3 : 2e-4)
            XCTAssertTrue(arrayEqual(argMax(lhs, axis: -1), argMax(rhs, axis: -1)).item(Bool.self))
            try streamed.checkInferenceError()
        }
    }

    func testSharedBuffersAndJointRoutersMatchResidentAcrossDecodeAndEviction() throws {
        for prefetch in [false, true] {
            for batchSize in [1, 2] {
                let (resident, streamed, _) = try pair(
                    mode: .affine, prefetch: prefetch, batchSize: batchSize,
                    cacheSizeBytes: 327_680, bits: 3, sharedBuffers: true, separateExperts: true)
                let lhsCache = try resident.newCache(configuration: nil)
                let rhsCache = try streamed.newCache(configuration: nil)
                for tokens in [Array(1 ... 20)] + (21 ... 40).map({ [$0] }) {
                    let input = MLXArray(tokens, [1, tokens.count])
                    let lhs = resident(input, cache: lhsCache)
                    let rhs = streamed(input, cache: rhsCache)
                    eval(lhs, rhs)
                    try streamed.checkInferenceError()
                    XCTAssertLessThan(abs(lhs - rhs).max().item(Float.self), 2e-3)
                    XCTAssertEqual(
                        argMax(lhs, axis: -1).asArray(Int.self),
                        argMax(rhs, axis: -1).asArray(Int.self))
                    XCTAssertEqual(lhsCache.map(\.offset), rhsCache.map(\.offset))
                }
            }
        }
    }

    func testPrefill256And512PreserveCacheContinuation() throws {
        let (resident, streamed, _) = try pair(mode: .affine)
        let tokens = MLXArray((0 ..< 513).map { $0 % 60 + 1 })
        let input = LMInput(tokens: tokens)
        for step in [256, 512] {
            let lhsCache = try resident.newCache(configuration: nil)
            let rhsCache = try streamed.newCache(configuration: nil)
            guard
                case .logits(let lhs) = try resident.prepare(
                    input, cache: lhsCache, state: nil, prefill: .init(stepSize: step)),
                case .logits(let rhs) = try streamed.prepare(
                    input, cache: rhsCache, state: nil, prefill: .init(stepSize: step))
            else { return XCTFail("Expected prefilled logits") }
            // Long resident prefills select sorted quantized gather kernels; bounded
            // batches use a different reduction order. Test the numerical bound and
            // greedy decision separately, rather than requiring bitwise equality.
            XCTAssertLessThan(abs(lhs.logits - rhs.logits).max().item(Float.self), 2e-3)
            XCTAssertEqual(
                argMax(lhs.logits, axis: -1).asArray(Int.self),
                argMax(rhs.logits, axis: -1).asArray(Int.self))
            let next = MLXArray([9], [1, 1])
            let lhsNext = resident(next, cache: lhsCache)
            let rhsNext = streamed(next, cache: rhsCache)
            XCTAssertLessThan(abs(lhsNext - rhsNext).max().item(Float.self), 2e-3)
            XCTAssertEqual(
                argMax(lhsNext, axis: -1).asArray(Int.self),
                argMax(rhsNext, axis: -1).asArray(Int.self))
            XCTAssertEqual(lhsCache.map(\.offset), rhsCache.map(\.offset))
            try streamed.checkInferenceError()
        }
    }

    func testStorageFailurePropagatesFromPrefill() throws {
        let (_, streamed, directory) = try pair(prefetch: false)
        let handle = try FileHandle(
            forWritingTo: directory.appendingPathComponent("model.safetensors"))
        try handle.truncate(atOffset: 0)
        try handle.close()
        XCTAssertThrowsError(
            try streamed.prepare(
                LMInput(tokens: MLXArray([1, 2, 3])),
                cache: streamed.newCache(configuration: nil), state: nil,
                prefill: .init(stepSize: 2)))
        XCTAssertThrowsError(try streamed.checkInferenceError())
    }

    func testDenseGemmaRejectsExpertStreaming() throws {
        let model = Gemma4(try configuration(moe: false))
        XCTAssertThrowsError(
            try model.configureExpertStreaming(
                urls: [], options: .init(), quantization: nil, perLayerQuantization: nil))
    }
}
