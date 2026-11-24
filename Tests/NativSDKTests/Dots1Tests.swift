import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class Dots1Tests: XCTestCase {
    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(configuration: Data(Self.json().utf8), modelType: "dots1")
    }

    private static func json(scoring: String = "noaux_tc", dense: Int = 1) -> String {
        """
        {
          "model_type": "dots1",
          "hidden_size": 32,
          "num_hidden_layers": 4,
          "intermediate_size": 64,
          "num_attention_heads": 4,
          "rms_norm_eps": 0.000001,
          "vocab_size": 48,
          "max_position_embeddings": 256,
          "num_key_value_heads": 2,
          "first_k_dense_replace": \(dense),
          "moe_intermediate_size": 16,
          "n_routed_experts": 6,
          "n_shared_experts": 1,
          "norm_topk_prob": true,
          "num_experts_per_tok": 2,
          "rope_theta": 10000.0,
          "routed_scaling_factor": 2.5,
          "scoring_func": "\(scoring)",
          "n_group": 2,
          "topk_group": 1
        }
        """
    }

    private static func config(scoring: String = "noaux_tc", dense: Int = 1) throws
        -> Dots1Configuration
    {
        try JSONDecoder.json5().decode(
            Dots1Configuration.self,
            from: XCTUnwrap(json(scoring: scoring, dense: dense).data(using: .utf8)))
    }

    /// The two-stage router is the only scoring this family implements, so another one is
    /// refused rather than quietly routed the one way.
    func testAnUnsupportedScoringFunctionIsRefused() throws {
        XCTAssertNoThrow(try Self.config(scoring: "noaux_tc"))
        XCTAssertThrowsError(try Self.config(scoring: "sigmoid"))
        XCTAssertThrowsError(try Self.config(scoring: "softmax"))
    }

    /// A single threshold with no stride: dense below `first_k_dense_replace`, mixture at
    /// and above. deepseek and klear each add a stride to the same-looking field.
    func testTheStackTurnsAtASingleThreshold() throws {
        let model = Dots1Model(try Self.config(dense: 2))
        XCTAssertFalse(model.model.layers[0].isMixture)
        XCTAssertFalse(model.model.layers[1].isMixture)
        XCTAssertTrue(model.model.layers[2].isMixture)
        XCTAssertTrue(model.model.layers[3].isMixture)
        XCTAssertTrue(model.model.layers[1].mlp is SwiGLUMLP)
        XCTAssertTrue(model.model.layers[2].mlp is Dots1MoE)
    }

    /// The router goes through the shared grouped selection, so its scores carry the
    /// routed scaling factor. With normalization on, the kept weights sum to the factor
    /// rather than to one.
    func testRoutedScalingReachesTheScores() throws {
        let config = try Self.config()
        let moe = try XCTUnwrap(Dots1Model(config).model.layers[1].mlp as? Dots1MoE)
        moe.gate.update(
            parameters: ModuleParameters.unflattened([
                ("weight", MLXArray(0 ..< (6 * 32)).reshaped(6, 32).asType(.float32) / 96),
                ("e_score_correction_bias", MLXArray.zeros([6])),
            ]))

        let x = (MLXArray(0 ..< 32).reshaped(1, 1, 32).asType(.float32) - 16) / 16
        let (indices, scores) = moe.gate(x)
        eval(indices, scores)

        XCTAssertEqual(scores.shape.last, config.expertsPerToken)
        XCTAssertEqual(
            scores.sum().item(Float.self), config.routedScalingFactor, accuracy: 1e-3)
    }

    /// Grouping constrains the choice: with two groups of three and one group kept, both
    /// selected experts must come from the winning half, even with negative bias.
    func testGroupingConstrainsTheSelectionToOneGroup() throws {
        let config = try Self.config()
        let moe = try XCTUnwrap(Dots1Model(config).model.layers[1].mlp as? Dots1MoE)
        moe.gate.update(
            parameters: ModuleParameters.unflattened([
                ("weight", MLXArray(0 ..< (6 * 32)).reshaped(6, 32).asType(.float32) / 96),
                ("e_score_correction_bias", MLXArray([Float(-4), -4, -4, -3, -3, -3])),
            ]))

        let x = (MLXArray(0 ..< 32).reshaped(1, 1, 32).asType(.float32) - 16) / 16
        let (indices, _) = moe.gate(x)
        eval(indices)

        let picked = (0 ..< config.expertsPerToken).map { indices[0, 0, $0].item(Int.self) }
        let halves = Set(picked.map { $0 / (config.routedExperts / config.groups) })
        XCTAssertEqual(
            halves, Set([1]),
            "Negative selection scores must not allow experts from the discarded group")
    }

    func testSanitizerStacksExpertsAndDropsPrecomputedFrequencies() throws {
        let config = try Self.config()
        let model = Dots1Model(config)
        var weights: [String: MLXArray] = [
            "model.embed_tokens.weight": MLXArray.zeros([48, 32]),
            "model.layers.1.self_attn.rotary_emb.inv_freq": MLXArray.ones([8]),
        ]
        for layer in 1 ..< config.hiddenLayers {
            for expert in 0 ..< config.routedExperts {
                for projection in ["gate_proj", "up_proj"] {
                    weights["model.layers.\(layer).mlp.experts.\(expert).\(projection).weight"] =
                        MLXArray.zeros([16, 32]) + Float(expert)
                }
                weights["model.layers.\(layer).mlp.experts.\(expert).down_proj.weight"] =
                    MLXArray.zeros([32, 16])
            }
        }

        let sanitized = model.sanitize(weights: weights)
        let stacked = try XCTUnwrap(sanitized["model.layers.2.mlp.experts.gate_proj.weight"])
        eval(stacked)
        XCTAssertEqual(stacked.shape, [config.routedExperts, 16, 32])
        XCTAssertEqual(stacked[4, 0, 0].item(Float.self), 4.0)
        XCTAssertNil(sanitized["model.layers.1.self_attn.rotary_emb.inv_freq"])
    }

}
