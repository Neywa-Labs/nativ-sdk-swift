import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class Qwen2MoETests: XCTestCase {
    func testQuantizedCheckpointPreservesLogits() async throws {
        try await assertTextCheckpoint(
            configuration: Data(Self.json().utf8), modelType: "qwen2_moe", quantized: true)
    }

    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(
            configuration: Data(Self.json().utf8), modelType: "qwen2_moe")
    }

    private static func json(extra: String = "") -> String {
        """
        {
          "model_type": "qwen2_moe",
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "intermediate_size": 128,
          "num_attention_heads": 4,
          "num_experts_per_tok": 2,
          "num_experts": 4,
          "moe_intermediate_size": 48,
          "shared_expert_intermediate_size": 64,
          "rms_norm_eps": 0.000001,
          "vocab_size": 48,
          "num_key_value_heads": 2\(extra)
        }
        """
    }

    private static func config(extra: String = "") throws -> Qwen2MoEConfiguration {
        try JSONDecoder.json5().decode(
            Qwen2MoEConfiguration.self, from: XCTUnwrap(json(extra: extra).data(using: .utf8)))
    }

    func testSanitizerStacksSeparatelyShippedExperts() throws {
        let config = try Self.config()
        let model = Qwen2MoEModel(config)
        var weights: [String: MLXArray] = ["model.embed_tokens.weight": MLXArray.zeros([48, 32])]
        for expert in 0 ..< config.experts {
            weights["model.layers.0.mlp.experts.\(expert).up_proj.weight"] =
                MLXArray.zeros([48, 32]) + Float(expert)
            weights["model.layers.0.mlp.experts.\(expert).gate_proj.weight"] =
                MLXArray.zeros([48, 32])
            weights["model.layers.0.mlp.experts.\(expert).down_proj.weight"] =
                MLXArray.zeros([32, 48])
        }

        let sanitized = model.sanitize(weights: weights)
        let stacked = try XCTUnwrap(sanitized["model.layers.0.mlp.switch_mlp.up_proj.weight"])
        XCTAssertEqual(stacked.shape, [config.experts, 48, 32])
        eval(stacked)
        XCTAssertEqual(stacked[3, 0, 0].item(Float.self), 3.0)
        XCTAssertNil(sanitized["model.layers.0.mlp.experts.0.up_proj.weight"])
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
    }

    func testRopeScalingRequiresTypeNotRopeType() throws {
        XCTAssertNoThrow(
            try Self.config(extra: ",\n  \"rope_scaling\": {\"factor\": 2.0, \"type\": \"linear\"}")
        )
        // Upstream requires `type` here specifically, unlike nemotron which takes either.
        XCTAssertThrowsError(
            try Self.config(
                extra: ",\n  \"rope_scaling\": {\"factor\": 2.0, \"rope_type\": \"linear\"}"))
    }

}
