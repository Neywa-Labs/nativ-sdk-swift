import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class KlearTests: XCTestCase {
    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(configuration: Data(Self.json().utf8), modelType: "klear")
    }

    private static func json(normalize: Bool = true, step: Int = 1, mlpOnly: String = "[]")
        -> String
    {
        """
        {
          "model_type": "klear",
          "hidden_size": 32,
          "num_hidden_layers": 4,
          "intermediate_size": 64,
          "num_attention_heads": 4,
          "attention_bias": false,
          "mlp_only_layers": \(mlpOnly),
          "num_experts": 4,
          "num_experts_per_tok": 2,
          "decoder_sparse_step": \(step),
          "n_shared_experts": 2,
          "moe_intermediate_size": 16,
          "rms_norm_eps": 0.000001,
          "vocab_size": 48,
          "num_key_value_heads": 2,
          "rope_theta": 10000.0,
          "max_position_embeddings": 256,
          "norm_topk_prob": \(normalize)
        }
        """
    }

    private static func config(normalize: Bool = true, step: Int = 1, mlpOnly: String = "[]")
        throws -> KlearConfiguration
    {
        try JSONDecoder.json5().decode(
            KlearConfiguration.self,
            from: XCTUnwrap(
                json(normalize: normalize, step: step, mlpOnly: mlpOnly).data(using: .utf8)))
    }

    /// The sparse step is one-based here: layers whose index *plus one* divides. deepseek's
    /// same-looking field uses a plain modulo, so the two are off by one from each other.
    func testTheSparseStepIsOneBased() throws {
        let everyOther = try Self.config(step: 2)
        XCTAssertTrue(everyOther.isMixtureLayer(1))
        XCTAssertFalse(everyOther.isMixtureLayer(2))
        XCTAssertTrue(everyOther.isMixtureLayer(3))

        let model = KlearModel(everyOther)
        XCTAssertFalse(model.model.layers[0].isMixture)
        XCTAssertTrue(model.model.layers[1].isMixture)
    }

    func testMLPOnlyLayersOverrideTheStep() throws {
        let config = try Self.config(mlpOnly: "[2]")
        XCTAssertTrue(config.isMixtureLayer(1))
        XCTAssertFalse(config.isMixtureLayer(2))
        XCTAssertTrue(
            KlearModel(config).model.layers[2].mlp is SwiGLUMLP)
    }

    func testSanitizerStacksUnderTheExpertsNameAndSkipsDenseLayers() throws {
        let config = try Self.config(step: 2)
        let model = KlearModel(config)
        var weights: [String: MLXArray] = ["model.embed_tokens.weight": MLXArray.zeros([48, 32])]
        for layer in [1, 3] {
            for expert in 0 ..< config.experts {
                for projection in ["gate_proj", "up_proj"] {
                    weights["model.layers.\(layer).mlp.experts.\(expert).\(projection).weight"] =
                        MLXArray.zeros([16, 32]) + Float(expert)
                }
                weights["model.layers.\(layer).mlp.experts.\(expert).down_proj.weight"] =
                    MLXArray.zeros([32, 16])
            }
        }

        let sanitized = model.sanitize(weights: weights)
        let stacked = try XCTUnwrap(sanitized["model.layers.1.mlp.experts.gate_proj.weight"])
        eval(stacked)
        XCTAssertEqual(stacked.shape, [config.experts, 16, 32])
        XCTAssertEqual(stacked[3, 0, 0].item(Float.self), 3.0)
        XCTAssertNil(sanitized["model.layers.1.mlp.experts.0.gate_proj.weight"])
        // Dense layers can precede the first mixture layer.
        XCTAssertNil(sanitized["model.layers.0.mlp.experts.gate_proj.weight"])
        XCTAssertNil(sanitized["model.layers.2.mlp.experts.gate_proj.weight"])
    }

}
