import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class DeepseekTests: XCTestCase {
    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(configuration: Data(Self.json().utf8), modelType: "deepseek")
    }

    private static func json(extra: String = "") -> String {
        """
        {
          "model_type": "deepseek",
          "vocab_size": 48,
          "hidden_size": 32,
          "intermediate_size": 64,
          "moe_intermediate_size": 16,
          "num_hidden_layers": 4,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "n_routed_experts": 4,
          "num_experts_per_tok": 2,
          "first_k_dense_replace": 1,
          "moe_layer_freq": 1,
          "max_position_embeddings": 256,
          "rms_norm_eps": 0.000001\(extra)
        }
        """
    }

    private static func config(extra: String = "") throws -> DeepseekConfiguration {
        try JSONDecoder.json5().decode(
            DeepseekConfiguration.self, from: XCTUnwrap(json(extra: extra).data(using: .utf8)))
    }

    /// Three conditions decide each layer's MLP, and a checkpoint's keys differ per layer
    /// accordingly -- so a wrong predicate leaves layers unbound rather than merely slower.
    func testEachLayerGetsTheKindItsIndexAsksFor() throws {
        let config = try Self.config()
        // first_k_dense_replace 1, so layer 0 is dense and the rest are mixtures.
        XCTAssertFalse(config.isMixtureLayer(0))
        XCTAssertTrue(config.isMixtureLayer(1))
        XCTAssertTrue(config.isMixtureLayer(3))

        let model = DeepseekModel(config)
        XCTAssertFalse(model.model.layers[0].isMixture)
        XCTAssertTrue(model.model.layers[1].isMixture)
        XCTAssertTrue(model.model.layers[0].mlp is SwiGLUMLP)
        XCTAssertTrue(model.model.layers[1].mlp is DeepseekMoE)
    }

    /// `moe_layer_freq` thins the mixtures further, and with no routed experts at all every
    /// layer is dense however the other two read.
    func testFrequencyAndAbsentExpertsBothThinTheMixtures() throws {
        // Replaced rather than appended: a duplicate key leaves the decoder holding the
        // first one, which is how this test first passed against an unchanged frequency.
        let everyOther = try JSONDecoder.json5().decode(
            DeepseekConfiguration.self,
            from: XCTUnwrap(
                Self.json()
                    .replacingOccurrences(
                        of: "\"moe_layer_freq\": 1", with: "\"moe_layer_freq\": 2"
                    )
                    .data(using: .utf8)))
        XCTAssertEqual(everyOther.moeLayerFrequency, 2)
        XCTAssertFalse(everyOther.isMixtureLayer(1))
        XCTAssertTrue(everyOther.isMixtureLayer(2))
        XCTAssertFalse(everyOther.isMixtureLayer(3))

        // n_routed_experts absent: dense throughout.
        let dense = try JSONDecoder.json5().decode(
            DeepseekConfiguration.self,
            from: XCTUnwrap(
                Self.json()
                    .split(separator: "\n")
                    .filter { !$0.contains("n_routed_experts") }
                    .joined(separator: "\n").data(using: .utf8)))
        XCTAssertFalse(dense.isMixtureLayer(0))
        XCTAssertFalse(dense.isMixtureLayer(3))
        XCTAssertTrue(DeepseekModel(dense).model.layers.allSatisfy { !$0.isMixture })
    }

    /// Softmax over every logit, then select -- qwen2_moe's order, so the kept weights sum
    /// to less than one. phixtral and granitemoe do the reverse.
    func testRouterSoftmaxesBeforeSelecting() throws {
        let config = try Self.config()
        let moe = try XCTUnwrap(DeepseekModel(config).model.layers[1].mlp as? DeepseekMoE)
        // A non-degenerate router: the zero-initialized weight would make every expert tie.
        moe.gate.update(
            parameters: ModuleParameters.unflattened([
                ("weight", MLXArray(0 ..< (4 * 32)).reshaped(4, 32).asType(.float32) / 128)
            ]))

        let x = (MLXArray(0 ..< 32).reshaped(1, 1, 32).asType(.float32) - 16) / 16
        let (_, scores) = moe.gate(x)
        eval(scores)

        XCTAssertEqual(scores.shape.last, config.expertsPerToken)
        XCTAssertLessThan(
            scores.sum().item(Float.self), 1.0 - 1e-4,
            "two of four experts kept the whole mass, so this cannot distinguish the orders")
    }

    /// Only the mixture layers have experts to stack; a dense layer has no such keys, so
    /// the sanitizer asks each layer rather than assuming a uniform stack.
    func testSanitizerStacksOnlyTheMixtureLayers() throws {
        let config = try Self.config()
        let model = DeepseekModel(config)
        var weights: [String: MLXArray] = [
            "model.layers.0.mlp.gate_proj.weight": MLXArray.zeros([64, 32])
        ]
        for expert in 0 ..< 4 {
            for projection in ["gate_proj", "up_proj"] {
                weights["model.layers.1.mlp.experts.\(expert).\(projection).weight"] =
                    MLXArray.zeros([16, 32]) + Float(expert)
            }
            weights["model.layers.1.mlp.experts.\(expert).down_proj.weight"] =
                MLXArray.zeros([32, 16])
        }

        let sanitized = model.sanitize(weights: weights)
        let stacked = try XCTUnwrap(
            sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"])
        eval(stacked)
        XCTAssertEqual(stacked.shape, [4, 16, 32])
        XCTAssertEqual(stacked[2, 0, 0].item(Float.self), 2.0)

        // The dense layer's own MLP is untouched.
        XCTAssertNotNil(sanitized["model.layers.0.mlp.gate_proj.weight"])
        XCTAssertNil(sanitized["model.layers.1.mlp.experts.0.gate_proj.weight"])
    }

}
