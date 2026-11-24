import Foundation
import MLX
import MLXNN
import Testing

@testable import NativSDK

private func inklingConfig(_ json: String) throws -> InklingTextConfiguration {
    try JSONDecoder.json5().decode(InklingTextConfiguration.self, from: Data(json.utf8))
}

private let moeConfig = """
    {
      "hidden_size": 8, "intermediate_size": 3,
      "n_routed_experts": 4, "n_shared_experts": 2,
      "num_experts_per_tok": 2, "route_scale": 8.0
    }
    """

@Suite("Inkling mixture of experts")
struct InklingMoETests {

    /// Selection and weighting read different quantities, and the shared experts compete
    /// with the routed ones for the same weight mass. Values captured from mlx-vlm.
    @Test("Routing matches mlx-vlm")
    func routing() throws {
        let moe = InklingSparseMoE(try inklingConfig(moeConfig))
        moe.update(
            parameters: .unflattened([
                "e_score_correction_bias": MLXArray([Float(0), 0.5, 0, 0])
            ]))

        let logits = MLXArray(
            [Float(1.0), 0.2, 2.0, -1.0, 0.5, -0.5, -2.0, 0.1, 0.0, 3.0, 1.0, 0.0], [2, 6])
        let (indices, weights, gamma) = moe.route(logits)
        eval(indices, weights, gamma)

        // Expert 1 wins on row 0 only because the correction bias lifts it.
        #expect(indices.asType(.int32).reshaped(-1).asArray(Int32.self) == [1, 2, 1, 3])

        let expected: [Float] = [1.80968, 2.89899, 1.55055, 2.81347]
        for (got, want) in zip(weights.reshaped(-1).asArray(Float.self), expected) {
            #expect(abs(got - want) < 1e-4, "\(got) != \(want)")
        }

        // The shared experts' weights are broadcast over the expert-major intermediate.
        #expect(gamma.shape == [2, 2 * 3])
        let gammaRow = gamma[0].asArray(Float.self)
        for value in gammaRow[0 ..< 3] { #expect(abs(value - 2.04872) < 1e-4) }
        for value in gammaRow[3 ..< 6] { #expect(abs(value - 1.24261) < 1e-4) }
    }

    /// Routed and shared weights are one softmax, so together they carry exactly the route
    /// scale. Adding the shared experts on top at full strength would break this.
    @Test("Routed and shared weights sum to the route scale")
    func weightMass() throws {
        let moe = InklingSparseMoE(try inklingConfig(moeConfig))
        let logits = MLXArray(
            [Float(1.0), 0.2, 2.0, -1.0, 0.5, -0.5, -2.0, 0.1, 0.0, 3.0, 1.0, 0.0], [2, 6])
        let (_, weights, gamma) = moe.route(logits)

        // gamma repeats each shared weight across the intermediate, so one column each.
        let sharedPerRow = gamma[0..., MLXArray([Int32(0), 3])]
        let total = weights.sum(axis: -1) + sharedPerRow.sum(axis: -1)
        eval(total)
        for value in total.asArray(Float.self) {
            #expect(abs(value - 8.0) < 1e-3, "\(value) != route scale")
        }
    }

    /// The correction bias steers which experts are chosen and never enters the weight
    /// computation, which reads the raw logits. A bias that leaves the selection alone
    /// therefore leaves the weights untouched, while one that changes it changes which
    /// experts run -- and, since the softmax is over the selected set, their weights too.
    @Test("The correction bias steers selection and never enters the weights")
    func biasSteersSelectionOnly() throws {
        let config = try inklingConfig(moeConfig)
        let logits = MLXArray([Float(1.0), 0.2, 2.0, -1.0, 0.5, -0.5], [1, 6])

        let unbiased = InklingSparseMoE(config)
        let (plainIndices, plainWeights, _) = unbiased.route(logits)
        eval(plainIndices, plainWeights)
        #expect(plainIndices.asType(.int32).reshaped(-1).asArray(Int32.self) == [2, 0])

        // Small enough not to reorder the top two.
        let harmless = InklingSparseMoE(config)
        harmless.update(
            parameters: .unflattened([
                "e_score_correction_bias": MLXArray([Float(0.05), 0, 0, 0])
            ]))
        let (sameIndices, sameWeights, _) = harmless.route(logits)
        eval(sameIndices, sameWeights)
        #expect(
            sameIndices.asType(.int32).reshaped(-1).asArray(Int32.self)
                == plainIndices.asType(.int32).reshaped(-1).asArray(Int32.self))
        for (a, b) in zip(
            plainWeights.reshaped(-1).asArray(Float.self),
            sameWeights.reshaped(-1).asArray(Float.self))
        {
            #expect(abs(a - b) < 1e-6, "the bias reached the weights: \(a) != \(b)")
        }

        // Large enough to lift expert 1 over expert 0.
        let steering = InklingSparseMoE(config)
        steering.update(
            parameters: .unflattened([
                "e_score_correction_bias": MLXArray([Float(0), 0.5, 0, 0])
            ]))
        let (steeredIndices, _, _) = steering.route(logits)
        eval(steeredIndices)
        #expect(steeredIndices.asType(.int32).reshaped(-1).asArray(Int32.self) == [1, 2])
    }

    /// All-ones scales take a fast path that skips the per-expert gathers.
    @Test("All-ones scales take the fast path")
    func trivialScales() {
        let switched = InklingSwitchGLU(inputDimensions: 8, hiddenDimensions: 3, experts: 4)
        #expect(switched.trivialScales())

        let scaled = InklingSwitchGLU(inputDimensions: 8, hiddenDimensions: 3, experts: 4)
        scaled.update(
            parameters: .unflattened([
                "gate_scale": MLXArray([Float(1), 2, 1, 1])
            ]))
        #expect(scaled.trivialScales() == false)
    }

    /// The scaled path applies the gate scale between the projections and the output scale
    /// after the down projection, both gathered per selected expert.
    @Test("The scaled path applies both per-expert scales")
    func scaledPath() {
        let experts = 4
        let switched = InklingSwitchGLU(
            inputDimensions: 8, hiddenDimensions: 3, experts: experts)
        let gateScale = MLXArray([Float(1), 2, 0.5, 3])
        let outScale = MLXArray([Float(2), 1, 1, 0.25])
        switched.update(
            parameters: .unflattened(["gate_scale": gateScale, "out_scale": outScale]))
        #expect(switched.trivialScales() == false)

        let x = (MLXArray(0 ..< 16).asType(.float32) / 9).reshaped(2, 8)
        let indices = MLXArray([Int32(0), 2, 1, 3], [2, 2])
        let actual = switched(x, indices)

        // The same computation, written out.
        let expanded = MLX.expandedDimensions(x, axes: [-2, -3])
        let up = switched.upProj(expanded, indices)
        var gate = switched.gateProj(expanded, indices)
        var perGate = gateScale[indices]
        while perGate.ndim < gate.ndim { perGate = perGate[.ellipsis, .newAxis] }
        gate = gate * perGate
        var expected = switched.downProj(compiledSiluProduct(gate, up), indices)
        var perOut = outScale[indices]
        while perOut.ndim < expected.ndim { perOut = perOut[.ellipsis, .newAxis] }
        expected = MLX.squeezed(expected * perOut, axis: -2)

        eval(actual, expected)
        #expect(actual.shape == expected.shape)
        for (a, b) in zip(
            actual.reshaped(-1).asArray(Float.self),
            expected.reshaped(-1).asArray(Float.self))
        {
            #expect(abs(a - b) < 1e-5, "\(a) != \(b)")
        }
    }

    /// The shared experts are stored switched-shaped and loaded dense: gate and up stack
    /// experts along rows, down along input columns.
    @Test("Shared experts are remapped to the dense layout")
    func sharedExpertsRemap() {
        let remapped = inklingSharedExpertsToDense(weights: [
            "mlp.shared_experts.gate_proj.weight": MLXArray.zeros([2, 3, 8]),
            "mlp.shared_experts.up_proj.weight": MLXArray.zeros([2, 3, 8]),
            "mlp.shared_experts.down_proj.weight": MLXArray.zeros([2, 8, 3]),
            "mlp.switch_mlp.gate_proj.weight": MLXArray.zeros([4, 3, 8]),
        ])
        #expect(remapped["mlp.shared_experts.gate_proj.weight"]?.shape == [6, 8])
        #expect(remapped["mlp.shared_experts.up_proj.weight"]?.shape == [6, 8])
        #expect(remapped["mlp.shared_experts.down_proj.weight"]?.shape == [8, 6])
        // Routed experts stay switched-shaped.
        #expect(remapped["mlp.switch_mlp.gate_proj.weight"]?.shape == [4, 3, 8])
    }

    /// The four attention projections are stacked at load so decode does one matmul.
    @Test("q/k/v/r stack into one projection")
    func fuseProjections() throws {
        var weights: [String: MLXArray] = [:]
        for (name, rows) in [("q", 8), ("k", 4), ("v", 4), ("r", 2)] {
            weights["model.layers.0.self_attn.\(name)_proj.weight"] = MLXArray.zeros([
                rows, 16,
            ])
        }
        let fused = try fuseInklingQKVR(weights: weights)
        #expect(fused["model.layers.0.self_attn.qkvr_proj.weight"]?.shape == [18, 16])
        #expect(!fused.keys.contains { $0.hasSuffix("q_proj.weight") })

        // A partial set is a malformed checkpoint, not something to stack anyway.
        var partial = weights
        partial.removeValue(forKey: "model.layers.0.self_attn.r_proj.weight")
        #expect(throws: InklingWeightError.self) { _ = try fuseInklingQKVR(weights: partial) }
    }

}
