import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class BailingHybridTests: XCTestCase {
    private let configuration = Data(
        """
        {"model_type":"bailing_hybrid","hidden_size":64,"intermediate_size":128,"vocab_size":128,
         "num_hidden_layers":3,"num_attention_heads":2,"head_dim":32,"layer_group_size":2,
         "q_lora_rank":32,"kv_lora_rank":64,"qk_nope_head_dim":32,"qk_rope_head_dim":16,
         "v_head_dim":32,"rms_norm_eps":1e-6,"rope_theta":10000,"num_experts":8,
         "moe_intermediate_size":64,"num_shared_experts":1,"num_experts_per_tok":2,
         "norm_topk_prob":true,"n_group":2,"topk_group":1,"first_k_dense_replace":1,
         "expert_swiglu_limit_list":[0,0.5,1],"share_expert_swiglu_limit_list":[0,1,0.5],
         "num_nextn_predict_layers":1}
        """.utf8)

    private func model() throws -> BailingHybrid {
        try XCTUnwrap(
            ModelRegistry.default.createModel(
                configuration: configuration, modelType: "bailing_hybrid") as? BailingHybrid)
    }

    func testHybridScheduleAndGeneration() async throws {
        let model = try model()
        let cache = try model.newCache(configuration: nil)
        XCTAssertEqual(cache.map { $0 is ArraysCache }, [true, false, false])
        eval(model(MLXArray([1, 2, 3], [1, 3]), cache: cache), cache)
        let recurrent = try XCTUnwrap(cache[0] as? ArraysCache)
        XCTAssertEqual(recurrent[0]?.shape, [1, 3, 64])
        XCTAssertEqual(recurrent[3]?.shape, [1, 2, 32, 32])
        XCTAssertEqual(recurrent[3]?.dtype, .float32)
        XCTAssertEqual(cache.map(\.offset), [3, 3, 3])
        try await assertTextCheckpoint(configuration: configuration, modelType: "bailing_hybrid")
    }

    func testBlockFP8DecodingAndNextNWeightRemoval() throws {
        let model = try model()
        let weights = model.sanitize(weights: [
            "model.layers.0.attention.q_proj.weight": toFP8(MLXArray.ones([129, 131])),
            "model.layers.0.attention.q_proj.weight_scale_inv": MLXArray(
                [1, 2, 3, 4] as [Float], [2, 2]),
            "model.layers.3.eh_proj.weight": MLXArray.zeros([2, 2]),
            "model.layers.0.attention.q_conv1d.weight": MLXArray.ones([64, 1, 4]),
        ])
        XCTAssertEqual(
            Set(weights.keys),
            [
                "model.layers.0.attention.q_proj.weight",
                "model.layers.0.attention.q_conv1d.conv.weight",
            ])
        let matrix = try XCTUnwrap(weights["model.layers.0.attention.q_proj.weight"])
        XCTAssertEqual(matrix.dtype, .bfloat16)
        XCTAssertEqual(matrix.shape, [129, 131])
        XCTAssertEqual(matrix[0, 0].item(Float.self), 1)
        XCTAssertEqual(matrix[0, 130].item(Float.self), 2)
        XCTAssertEqual(matrix[128, 0].item(Float.self), 3)
        XCTAssertEqual(matrix[128, 130].item(Float.self), 4)
        XCTAssertEqual(weights["model.layers.0.attention.q_conv1d.conv.weight"]?.shape, [64, 4, 1])
    }

    func testFusedKVSplitKeepsPerLayerQuantization() async throws {
        let model = try model()
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let prefix = "model.layers.1.attention."
        var weights = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let key = try XCTUnwrap(weights.removeValue(forKey: prefix + "embed_q.weight")).swappedAxes(
            -1, -2)
        let value = try XCTUnwrap(weights.removeValue(forKey: prefix + "unembed_out.weight"))
        let combined = quantized(
            concatenated([key, value], axis: 1).reshaped(-1, 64), groupSize: 32, bits: 8)
        weights[prefix + "kv_b_proj.weight"] = combined.wq
        weights[prefix + "kv_b_proj.scales"] = combined.scales
        weights[prefix + "kv_b_proj.biases"] = combined.biases
        try MLX.save(
            arrays: Dictionary(
                uniqueKeysWithValues: weights.map { ("language_model." + $0.key, $0.value) }),
            metadata: ["format": "mlx"], url: dir.appendingPathComponent("model.safetensors"))
        var config = try XCTUnwrap(
            JSONSerialization.jsonObject(with: configuration) as? [String: Any])
        config["quantization"] = [
            "group_size": 32, "bits": 4,
            "language_model." + prefix + "kv_b_proj": ["group_size": 32, "bits": 8],
        ]
        try JSONSerialization.data(withJSONObject: config).write(
            to: dir.appendingPathComponent("config.json"))
        let loaded = try await NativModel.loadModel(
            from: .directory(dir), using: TextFixtureTokenizerLoader())
        try await loaded.withModel { _, context in
            let modules = Dictionary(uniqueKeysWithValues: context.model.namedModules())
            for name in ["embed_q", "unembed_out"] {
                XCTAssertEqual((modules[prefix + name] as? QuantizedMultiLinear)?.bits, 8)
            }
            let input = MLXArray([1, 2, 3], [1, 3])
            let language = try context.languageModel
            XCTAssertTrue(all(isFinite(language(input, cache: nil))).item(Bool.self))
        }
    }

    func testSwiGLUClipsAfterActivationAndOneTapConvolutionKeepsNoHistory() {
        let activate = BailingHybridLayers.activation(limit: 0.1)
        let actual = activate(MLXArray([3, -5] as [Float]), MLXArray([2, -3] as [Float]))
        XCTAssertEqual(actual[0].item(Float.self), 0.01, accuracy: 1e-6)
        XCTAssertEqual(
            actual[1].item(Float.self), silu(MLXArray(Float(-5))).item(Float.self) * -0.1,
            accuracy: 1e-6)
        let convolution = KimiK3Text.ShortConv1d(channels: 4, kernelSize: 1)
        let first = convolution(.ones([1, 2, 4]), state: nil, mask: nil, lengths: nil)
        let next = convolution(.ones([1, 1, 4]), state: first.1, mask: nil, lengths: nil)
        XCTAssertEqual(first.1.shape, [1, 0, 4])
        XCTAssertEqual(next.0.shape, [1, 1, 4])
    }

    func testLingToolCallsWithAndWithoutArguments() throws {
        let model = try model()
        let parser = try XCTUnwrap(model.toolCallFormat).createParser()
        let lookup = parser.parse(
            content:
                "<tool_call>lookup<arg_key>filter</arg_key>\n<arg_value>{\"id\":17}</arg_value>\n</tool_call>",
            tools: nil)
        XCTAssertEqual(lookup?.function.name, "lookup")
        XCTAssertEqual(lookup?.function.arguments["filter"], .object(["id": .int(17)]))
        let empty = parser.parse(content: "<tool_call>list_items\n</tool_call>", tools: nil)
        XCTAssertEqual(empty?.function.name, "list_items")
        XCTAssertEqual(empty?.function.arguments, [:])
        XCTAssertNil(parser.parse(content: "<tool_call>invalid function</tool_call>", tools: nil))
    }
}
