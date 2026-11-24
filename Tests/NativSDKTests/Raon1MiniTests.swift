import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class Raon1MiniTests: XCTestCase {
    private let configuration = Data(
        """
        {"model_type":"raon1_mini","hidden_size":64,"vocab_size":128,"num_hidden_layers":2,
         "num_attention_heads":2,"head_dim":32,"intermediate_size":128,"num_experts":4,
         "num_experts_per_token":2,"moe_intermediate_size":64,"q_lora_rank":32,
         "kv_lora_rank":64,"qk_nope_head_dim":32,"qk_rope_head_dim":16,"v_head_dim":32,
         "rda_decay_rank":32,"rda_config":{"rda_layers":[1],"mla_layers":[2],"conv_kernel_size":4}}
        """.utf8)

    func testHybridCacheAndGeneration() async throws {
        let model = try ModelRegistry.default.createModel(
            configuration: configuration, modelType: "raon1_mini")
        let cache = try model.newCache(configuration: nil)
        let recurrent = try XCTUnwrap(cache[0] as? ArraysCache)
        XCTAssertTrue(cache[1] is KVCacheSimple)
        eval(model(MLXArray([1, 2, 3], [1, 3]), cache: cache), cache)
        XCTAssertEqual(recurrent[0]?.shape, [1, 3, 192])
        XCTAssertEqual(recurrent[1]?.shape, [1, 2, 32, 32])
        XCTAssertEqual(recurrent[1]?.dtype, .float32)
        XCTAssertEqual(cache.map(\.offset), [3, 3])
        try await assertTextCheckpoint(configuration: configuration, modelType: "raon1_mini")
    }

    func testChannelDecayMetalKernelMatchesArrayRecurrence() throws {
        guard Device.defaultDevice().deviceType == .gpu else { throw XCTSkip("Requires Metal") }
        MLXRandom.seed(34)
        let q = MLXRandom.normal([2, 5, 2, 32]) * 0.1
        let k = MLXRandom.normal(q.shape) * 0.1
        let v = MLXRandom.normal(q.shape) * 0.1
        let g = sigmoid(MLXRandom.normal(q.shape))
        let beta = sigmoid(MLXRandom.normal([2, 5, 2]))
        let state = MLXRandom.normal([2, 2, 32, 32]) * 0.1
        let mask = MLXArray(
            [true, true, true, true, true, false, true, true, false, false], [2, 5])
        let expected = gatedDeltaOps(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
        let actual = gatedDeltaKernel(q: q, k: k, v: v, g: g, beta: beta, state: state, mask: mask)
        let active = mask[.ellipsis, .newAxis, .newAxis]
        XCTAssertLessThan(
            abs(actual.0 - MLX.where(active, expected.0, 0)).max().item(Float.self), 1e-6)
        XCTAssertLessThan(abs(actual.1 - expected.1).max().item(Float.self), 1e-6)
    }

    func testSourceExpertBytesRoundTripAsNativeMXFP8() async throws {
        let model = try XCTUnwrap(
            ModelRegistry.default.createModel(configuration: configuration, modelType: "raon1_mini")
                as? Raon1Mini)
        let prefix = "model.layers.1.moe."
        try quantizeByParent(model: model) { path, _ in
            path.hasPrefix(prefix + "switch_mlp.") ? (32, 8, .mxfp8) : nil
        }
        model.train(false)
        let native = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        var source = native
        for projection in ["up_proj", "gate_proj", "down_proj"] {
            let name = prefix + "switch_mlp." + projection
            let packed = try XCTUnwrap(source.removeValue(forKey: name + ".weight"))
            let scales = try XCTUnwrap(source.removeValue(forKey: name + ".scales"))
            for expert in 0 ..< 4 {
                let path = prefix + "experts.\(expert).\(projection)."
                source[path + "weight"] = contiguous(packed[expert]).view(dtype: .uint8)
                source[path + "weight_scale_inv"] = scales[expert]
            }
        }
        let sanitized = model.sanitize(weights: source)
        XCTAssertEqual(Set(sanitized.keys), Set(native.keys))
        for (name, array) in sanitized where name.hasPrefix(prefix + "switch_mlp.") {
            XCTAssertTrue(arrayEqual(array, native[name]!).item(Bool.self), name)
        }
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try MLX.save(arrays: source, url: directory.appendingPathComponent("model.safetensors"))
        var config = try XCTUnwrap(
            JSONSerialization.jsonObject(with: configuration) as? [String: Any])
        config["quantization_config"] = ["quant_method": "mxfp8", "weight_block_size": [1, 32]]
        try JSONSerialization.data(withJSONObject: config).write(
            to: directory.appendingPathComponent("config.json"))
        let loaded = try await NativModel.loadModel(
            from: .directory(directory), using: TextFixtureTokenizerLoader())
        let result = try await loaded.withModel { _, context in
            let model = try XCTUnwrap(context.model as? Raon1Mini)
            let layers = Dictionary(uniqueKeysWithValues: model.namedModules())
            XCTAssertTrue(layers["lm_head"] is Linear && !(layers["lm_head"] is Quantized))
            return model(MLXArray([1, 5, 2], [1, 3]), cache: nil).asArray(Float.self)
        }
        XCTAssertEqual(result, model(MLXArray([1, 5, 2], [1, 3]), cache: nil).asArray(Float.self))
    }

    func testSourceCheckpointWhenRequested() async throws {
        guard let path = ProcessInfo.processInfo.environment["RAON1_MINI_MODEL"] else {
            throw XCTSkip("Set RAON1_MINI_MODEL to the local KRAFTON checkpoint")
        }
        let model = try await NativModel.loadModel(from: .directory(URL(fileURLWithPath: path)))
        let request = model.generate(
            input: UserInput(
                chat: [.user("What is 17 + 25? Reply with only the number.")],
                additionalContext: ["enable_thinking": false]),
            options: .init(
                prefill: .init(stepSize: 8), sampling: .init(temperature: 0),
                output: .init(maxTokens: 128)))
        let answer = try await request.response()
        print("Raon-1 Mini answer: \(answer)")
        XCTAssertEqual(answer.trimmingCharacters(in: .whitespacesAndNewlines), "42")
        let metrics = try await request.metrics()
        XCTAssertEqual(metrics.stopReason, .stop)
    }
}
