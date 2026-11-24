import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

/// Expectations captured from upstream mlx-vlm's language.py. Its Metal
/// fast path for the FP8 round-trip and the array path it falls back to were checked to
/// agree bit for bit, so matching the array path matches both.
final class DeepSeekV41MoETests: XCTestCase {
    private func configuration() throws -> DeepSeekV41Configuration {
        try JSONDecoder().decode(
            DeepSeekV41Configuration.self,
            from: Data(
                #"""
                {
                  "hidden_size": 32, "moe_intermediate_size": 32, "n_routed_experts": 4,
                  "num_experts_per_tok": 2, "routed_scaling_factor": 1.5,
                  "swiglu_limit": 10.0, "num_hidden_layers": 4
                }
                """#.utf8))
    }

    /// `(arange % 13 - 6) / 11`, the same deterministic fill the reference run used.
    private func ramp(_ shape: [Int], scale: Float = 1) -> MLXArray {
        let count = shape.reduce(1, *)
        return (MLXArray(0 ..< count).asType(.float32) % 13 - 6).reshaped(shape) / 11
            * scale
    }

    private func moe() throws -> DeepSeekV41Language.MoE {
        let moe = DeepSeekV41Language.MoE(try configuration())
        moe.update(
            parameters: ModuleParameters.unflattened([
                ("gate.weight", ramp([4, 32])),
                ("gate.bias", MLXArray([Float(0), 0, 0, 0])),
                ("gate.bias_vl", MLXArray([Float(0), -10, 0, 10])),
                ("switch_mlp.gate_proj.weight", ramp([4, 32, 32])),
                ("switch_mlp.up_proj.weight", ramp([4, 32, 32], scale: -1)),
                ("switch_mlp.down_proj.weight", ramp([4, 32, 32], scale: 0.5)),
                ("shared_experts.gate_proj.weight", ramp([32, 32])),
                ("shared_experts.up_proj.weight", ramp([32, 32], scale: -1)),
                ("shared_experts.down_proj.weight", ramp([32, 32], scale: 0.5)),
            ]))
        return moe
    }

    private var input: MLXArray { ramp([1, 3, 32]) }

    /// The limit clamps the gate from above and the up-projection on both sides, before
    /// the activation, and both are kept in FP32.
    func testLimitedSwiGLUMatchesUpstream() {
        let out = DeepSeekV41Language.swiGLU(
            gate: MLXArray([Float(-2), 0.5, 20, 3]).reshaped(1, 4),
            up: MLXArray([Float(1), -30, 2, 0.25]).reshaped(1, 4), limit: 10)
        assertClose(out[0], [-0.238_406, -3.112_297, 19.999_092, 0.714_431])
    }

    /// The image bias steers selection only: the weights come from the unbiased scores,
    /// so a flipped selection brings its own scores with it.
    func testGateBiasSteersSelectionOnly() throws {
        let gate = try moe().gate
        let (indices, weights) = gate(input)
        XCTAssertEqual(indices[0].asArray(Int32.self), [0, 2, 1, 3, 2, 0])
        assertClose(
            weights[0].flattened(),
            [0.857_275, 0.642_725, 0.838_827, 0.661_173, 0.870_515, 0.629_485])

        let mask = MLXArray([false, true, false]).reshaped(1, 3)
        let (biased, biasedWeights) = gate(input, imageMask: mask)
        XCTAssertEqual(biased[0].asArray(Int32.self), [0, 2, 3, 0, 2, 0])
        assertClose(
            biasedWeights[0].flattened(),
            [0.857_275, 0.642_725, 1.184_392, 0.315_608, 0.870_515, 0.629_485])
    }

    func testMoEForwardMatchesUpstream() throws {
        let moe = try moe()
        let out = moe(input)
        XCTAssertEqual(out.shape, [1, 3, 32])
        assertClose(
            out[0, 0, 0 ..< 6],
            [18.141_376, 5.509_319, -23.179_165, 13.435_184, -16.337_923, 5.110_92],
            accuracy: 2e-4)
        assertClose(
            out[0, 1, 0 ..< 6],
            [13.318_977, -7.419_511, 9.219_239, -4.790_832, -6.599_305, 0.966_613],
            accuracy: 2e-4)

        let mask = MLXArray([false, true, false]).reshaped(1, 3)
        assertClose(
            moe(input, imageMask: mask)[0, 1, 0 ..< 6],
            [16.883_564, -4.364_557, 4.899_859, 2.331_349, -17.306_915, 8.567_576],
            accuracy: 2e-4)
    }

    /// At 64 or more index entries the switch layers sort the rows by expert before
    /// gathering, and the routing weights have to travel with them. Thirty-two tokens at
    /// top-2 is exactly that threshold.
    func testSortedExpertPathMatchesUpstream() throws {
        let wide = ramp([1, 32, 32])
        let out = try moe()(wide)
        assertClose(
            out[0, 0, 0 ..< 6],
            [18.142_015, 5.506_681, -23.178_318, 13.434_128, -16.334_202, 5.109_534],
            accuracy: 2e-4)
        assertClose(
            out[0, 31, 0 ..< 6],
            [3.321_47, -9.994_89, 7.849_303, -11.220_259, 12.081_213, -6.774_601],
            accuracy: 2e-4)
    }

    /// The expert outputs are summed in FP32 before returning to the input dtype, which
    /// only shows on a narrower one.
    func testBFloat16ForwardMatchesUpstream() throws {
        let moe = try moe()
        var weights = Dictionary(uniqueKeysWithValues: moe.parameters().flattened())
        for (key, value) in weights where !key.hasPrefix("gate.bias") {
            weights[key] = value.asType(.bfloat16)
        }
        moe.update(parameters: ModuleParameters.unflattened(weights.map { ($0, $1) }))

        let out = moe(ramp([1, 32, 32]).asType(.bfloat16))
        XCTAssertEqual(out.dtype, .bfloat16)
        assertClose(out[0, 0, 0 ..< 6], [18, 6.1875, -23.375, 14.25, -16.375, 4.25], accuracy: 0)
        assertClose(
            out[0, 2, 0 ..< 6], [10.5625, -3.6875, 9, 4.9375, 17.125, 8.875], accuracy: 0)
    }

    /// The shared expert every token goes through, with its two FP8 round-trips.
    func testSharedExpertMatchesUpstream() throws {
        assertClose(
            try moe().sharedExperts(input)[0, 0, 0 ..< 6],
            [11.884_951, -0.006_351, -11.307_061, 3.390_518, -8.736_438, 6.343_931],
            accuracy: 2e-4)
    }

    // MARK: Weights

    /// Shared experts are renamed and routed experts stack over the expert axis.
    func testMoEWeightsAreStackedAndRenamed() throws {
        var weights: [String: MLXArray] = [
            "m.shared_experts.w1.weight": MLXArray.ones([4, 2]),
            "m.shared_experts.w2.weight": MLXArray.ones([2, 4]),
            "m.shared_experts.w3.weight": MLXArray.ones([4, 2]),
            "m.other.weight": MLXArray.ones([2]),
        ]
        for expert in 0 ..< 3 {
            weights["m.experts.\(expert).w1.weight"] = MLXArray.full(
                [4, 2], values: MLXArray(Float(expert)))
            weights["m.experts.\(expert).w2.weight"] = MLXArray.ones([2, 4])
            weights["m.experts.\(expert).w3.weight"] = MLXArray.ones([4, 2])
        }
        let out = try DeepSeekV41Language.sanitizeMoEWeights(
            weights, ffnPrefix: "m", experts: 3)

        XCTAssertNotNil(out["m.shared_experts.gate_proj.weight"])
        XCTAssertNotNil(out["m.shared_experts.down_proj.weight"])
        XCTAssertNotNil(out["m.shared_experts.up_proj.weight"])
        XCTAssertNil(out["m.shared_experts.w1.weight"])
        XCTAssertNotNil(out["m.other.weight"])

        let gate = try XCTUnwrap(out["m.switch_mlp.gate_proj.weight"])
        XCTAssertEqual(gate.shape, [3, 4, 2])
        XCTAssertEqual(gate[0, 0, 0].item(Float.self), 0)
        XCTAssertEqual(gate[1, 0, 0].item(Float.self), 1)
        XCTAssertEqual(gate[2, 0, 0].item(Float.self), 2)
        XCTAssertTrue(out.keys.allSatisfy { !$0.contains(".experts.0.") })
    }

    /// A checkpoint converted by the reference port already stacks them, and only needs
    /// the move onto switch_mlp.
    func testAlreadyStackedWeightsAreMoved() throws {
        let out = try DeepSeekV41Language.sanitizeMoEWeights(
            [
                "m.experts.gate_proj.weight": MLXArray.ones([3, 4, 2]),
                "m.experts.gate_proj.scales": MLXArray.ones([3, 4, 1]),
            ], ffnPrefix: "m", experts: 3)
        XCTAssertEqual(out["m.switch_mlp.gate_proj.weight"]?.shape, [3, 4, 2])
        XCTAssertEqual(out["m.switch_mlp.gate_proj.scales"]?.shape, [3, 4, 1])
        XCTAssertNil(out["m.experts.gate_proj.weight"])
    }

    /// A checkpoint missing an expert is refused, naming the ones it lacks, rather than
    /// stacking a short list into a wrong-width tensor.
    func testMissingExpertIsRefused() throws {
        var weights: [String: MLXArray] = [:]
        for expert in [0, 1, 3] {
            weights["m.experts.\(expert).w1.weight"] = MLXArray.ones([4, 2])
        }
        XCTAssertThrowsError(
            try DeepSeekV41Language.sanitizeMoEWeights(weights, ffnPrefix: "m", experts: 4)
        ) { error in
            XCTAssertTrue(
                error.localizedDescription.contains("missing 2"),
                "the message should name the absent expert: \(error.localizedDescription)")
        }
    }

    private func assertClose(
        _ actual: MLXArray, _ expected: [Float], accuracy: Float = 2e-5,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let values = actual.asType(.float32).asArray(Float.self)
        XCTAssertEqual(values.count, expected.count, file: file, line: line)
        for (index, pair) in zip(values, expected).enumerated() {
            XCTAssertEqual(
                pair.0, pair.1, accuracy: accuracy, "index \(index)", file: file, line: line)
        }
    }
}
