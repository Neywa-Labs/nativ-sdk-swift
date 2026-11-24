import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class VisionModelWeightTests: XCTestCase {
    private func mistralConfig() throws -> MistralLarge3Configuration {
        try JSONDecoder().decode(
            MistralLarge3Configuration.self,
            from: ModelTestConfigurations.configuration("mistral_large3"))
    }

    func testMistralNativeFP8ExpertsAreDecodedBeforeStacking() throws {
        let config = try mistralConfig()
        let model = MistralLarge3(config)
        var weights = [String: MLXArray]()
        for expert in 0 ..< 4 {
            weights["layers.1.experts.\(expert).w1.weight"] = toFP8(
                MLXArray.ones([16, 32]) * Float(expert + 1))
            weights["layers.1.experts.\(expert).w1.weight_scale"] = MLXArray([[Float(expert + 2)]])
        }
        weights["pre_mm_projector_norm.weight"] = MLXArray.ones([16])
        weights["vision_language_adapter.w_in.weight"] = MLXArray.ones([32, 16])
        weights["unknown.weight"] = MLXArray([7])
        let sanitized = model.sanitize(weights: weights)
        let packed = try XCTUnwrap(
            sanitized["language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight"])
        for expert in 0 ..< 4 {
            XCTAssertTrue(
                allClose(packed[expert], MLXArray(Float((expert + 1) * (expert + 2)))).item(
                    Bool.self))
        }
        XCTAssertFalse(
            sanitized.keys.contains { $0.contains(".experts.") || $0.hasSuffix("weight_scale_inv") }
        )
        XCTAssertNotNil(sanitized["multi_modal_projector.norm.weight"])
        XCTAssertNotNil(sanitized["multi_modal_projector.linear_1.weight"])
        // Unknown tensors must reach the strict checkpoint verifier rather than disappear.
        XCTAssertNotNil(sanitized["unknown.weight"])
    }

    func testMistralReconstructsQuantizedSplitAttentionProjections() throws {
        let original = try mistralConfig()
        var object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: JSONEncoder().encode(original)) as? [String: Any])
        var text = try XCTUnwrap(object["text_config"] as? [String: Any])
        text["kv_lora_rank"] = 32
        text["qk_nope_head_dim"] = 32
        text["v_head_dim"] = 32
        object["text_config"] = text
        let config = try JSONDecoder().decode(
            MistralLarge3Configuration.self, from: JSONSerialization.data(withJSONObject: object))
        // Use dimensions supported by affine group quantization.
        let model = MistralLarge3(config)
        let key = MLXArray(0 ..< 4096).asType(.float32).reshaped(4, 32, 32) / 4096
        let value = sin(key)
        let (kw, ks, kb) = quantized(key, groupSize: 32, bits: 4)
        let (vw, vs, vb) = quantized(value, groupSize: 32, bits: 4)
        let prefix = "language_model.model.layers.0.self_attn."
        let weights = model.sanitize(weights: [
            prefix + "embed_q.weight": kw, prefix + "embed_q.scales": ks,
            prefix + "embed_q.biases": try XCTUnwrap(kb),
            prefix + "unembed_out.weight": vw, prefix + "unembed_out.scales": vs,
            prefix + "unembed_out.biases": try XCTUnwrap(vb),
        ])
        let expected = concatenated(
            [
                dequantized(kw, scales: ks, biases: kb, groupSize: 32, bits: 4).transposed(0, 2, 1),
                dequantized(vw, scales: vs, biases: vb, groupSize: 32, bits: 4),
            ], axis: 1
        ).reshaped(-1, 32)
        XCTAssertEqual(weights.keys.sorted(), [prefix + "kv_b_proj.weight"])
        XCTAssertTrue(
            allClose(try XCTUnwrap(weights[prefix + "kv_b_proj.weight"]), expected).item(Bool.self))
    }

    func testRouterCorrectionBiasAffectsSelectionOnly() throws {
        var config = try mistralConfig().textConfiguration
        config.numExpertsPerTok = 2
        let gate = MoEGate(config: config)
        gate.weight = MLXArray.zeros([4, 32])
        gate.e_score_correction_bias = MLXArray([Float(0), 10, 2, 0])
        let (indices, weights) = gate(MLXArray.ones([1, 1, 32]))
        XCTAssertEqual(Set(indices.asArray(Int.self)), [1, 2])
        // Equal sigmoid scores must retain equal weights despite unequal selection biases.
        XCTAssertEqual(weights.asArray(Float.self), [0.5, 0.5])
    }
}
