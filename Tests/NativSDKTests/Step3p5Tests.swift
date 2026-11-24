import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class Step3p5Tests: XCTestCase {

    private static func config(overrides: [String: Any] = [:]) throws -> Step3p5Configuration {
        try JSONDecoder.json5().decode(
            Step3p5Configuration.self,
            from: ModelTestConfigurations.configuration("step3p5", overrides: overrides))
    }

    func testExplicitLayerKindsAreHonored() throws {
        let config = try Self.config(
            overrides: [
                "layer_types": [
                    "full_attention", "sliding_attention", "full_attention", "sliding_attention",
                ],
                "moe_layers_enum": "0,3",
            ])
        let model = Step3p5Model(config)
        XCTAssertEqual(model.model.layers.map(\.isSliding), [false, true, false, true])
        XCTAssertEqual(model.model.layers.map(\.isMoELayer), [true, false, false, true])
    }

    /// A sliding layer takes its head counts from `attention_other_setting`, so two layers
    /// in one model can have different numbers of heads.
    func testSlidingLayersCanHaveTheirOwnHeadCounts() throws {
        let config = try Self.config(
            overrides: [
                "attention_other_setting": ["num_attention_heads": 2, "num_attention_groups": 1]
            ])
        let model = Step3p5Model(config)
        XCTAssertEqual(model.model.layers[0].attention.heads, 2)
        XCTAssertEqual(model.model.layers[0].attention.kvHeads, 1)
        XCTAssertEqual(model.model.layers[1].attention.heads, 4)
        XCTAssertEqual(model.model.layers[1].attention.kvHeads, 2)
        XCTAssertEqual(model.kvHeads, [1, 2, 1, 2])
    }

    /// Attention is gated per head. Zeroing the gate projection sends every gate to
    /// `sigmoid(0) = 0.5`, halving the attention output before `o_proj`.
    func testAttentionIsGatedPerHead() throws {
        let config = try Self.config()
        let gated = Step3p5Attention(config, layerIndex: 1)
        XCTAssertNotNil(gated.wg)
        XCTAssertEqual(gated.wg?.weight.dim(0), 4, "one gate per head, not per feature")

        let ungatedConfig = try Self.config(overrides: ["use_head_wise_attn_gate": false])
        let ungated = Step3p5Attention(ungatedConfig, layerIndex: 1)
        XCTAssertNil(ungated.wg)

        ungated.update(
            parameters: ModuleParameters.unflattened(
                gated.parameters().flattened().filter { !$0.0.hasPrefix("g_proj") }))
        gated.update(
            parameters: ModuleParameters.unflattened([
                ("g_proj.weight", MLXArray.zeros([4, 32]))
            ]))

        let x = MLXRandom.normal([1, 5, 32], key: MLXRandom.key(7))
        let withGate = gated(x, mask: .causal, cache: nil)
        let withoutGate = ungated(x, mask: .causal, cache: nil)
        eval(withGate, withoutGate)

        XCTAssertLessThan(
            abs(withGate - withoutGate * 0.5).max().item(Float.self), 1e-5,
            "a zeroed gate should halve the attention output exactly")
    }

    /// The gate is bounded above only; the up branch on both sides.
    func testClampedSwiGLUBoundsTheBranchesDifferently() {
        let big = MLXArray(converting: [40.0])
        let small = MLXArray(converting: [-40.0])
        let one = MLXArray(converting: [1.0])

        let gateHigh = step3p5ClampedSwiGLU(gate: big, up: one, limit: 5)
        eval(gateHigh)
        XCTAssertEqual(gateHigh.item(Float.self), 5, accuracy: 1e-4)

        let upHigh = step3p5ClampedSwiGLU(gate: one, up: big, limit: 5)
        let upLow = step3p5ClampedSwiGLU(gate: one, up: small, limit: 5)
        eval(upHigh, upLow)
        XCTAssertEqual(
            upHigh.item(Float.self), -upLow.item(Float.self), accuracy: 1e-5,
            "the up branch is clipped asymmetrically")

        let gateLow = step3p5ClampedSwiGLU(gate: small, up: one, limit: 5)
        eval(gateLow)
        XCTAssertEqual(
            gateLow.item(Float.self), 0, accuracy: 1e-5,
            "the gate was clipped from below, which the reference does not do")
    }

    /// The routed experts and the shared expert read two different per-layer lists, and a
    /// dense layer reads the *shared* one.
    func testTheTwoLimitListsGoToDifferentPlaces() throws {
        let config = try Self.config(
            overrides: [
                "swiglu_limits": [0.0, 7.0, 0.0, 0.0],
                "swiglu_limits_shared": [11.0, 13.0, 0.0, 0.0],
            ])
        XCTAssertEqual(config.swiGLULimit(at: 1), 7)
        XCTAssertEqual(config.sharedSwiGLULimit(at: 1), 13)
        XCTAssertEqual(config.swiGLULimit(at: 3), 0)

        let model = Step3p5Model(config)
        let dense = try XCTUnwrap(model.model.layers[0].mlp as? Step3p5MLP)
        XCTAssertEqual(dense.limit, 11, "a dense layer must read the shared list")
        let mixture = try XCTUnwrap(model.model.layers[1].mlp as? Step3p5MoE)
        XCTAssertEqual(mixture.shareExpert.limit, 13)
    }

    /// `rope_theta` may be one value per layer.
    func testPerLayerRopeTheta() throws {
        let scalar = try Self.config()
        XCTAssertEqual(scalar.ropeTheta(at: 0), 10_000)
        XCTAssertEqual(scalar.ropeTheta(at: 3), 10_000)

        let perLayer = try Self.config(overrides: ["rope_theta": [1000.0, 2000.0, 3000.0, 4000.0]])
        XCTAssertEqual(perLayer.ropeTheta(at: 0), 1000)
        XCTAssertEqual(perLayer.ropeTheta(at: 3), 4000)

        let model = Step3p5Model(perLayer)
        let heads = MLXArray(1 ..< 33).reshaped(1, 4, 1, 8).asType(.float32)
        let first = applyRotaryPosition(
            model.model.layers[0].attention.rope, to: heads, offset: .scalar(5))
        let last = applyRotaryPosition(
            model.model.layers[3].attention.rope, to: heads, offset: .scalar(5))
        eval(first, last)
        XCTAssertGreaterThan(
            abs(first - last).max().item(Float.self), 1e-3,
            "the per-layer bases reached the rotary identically, so they were not read")
    }

    /// Rotary scaling reaches only the layer types `yarn_only_types` names.
    func testYarnOnlyTypesGateTheScaling() throws {
        let config = try Self.config(
            overrides: [
                "layer_types": [
                    "sliding_attention", "full_attention", "sliding_attention", "full_attention",
                ],
                "yarn_only_types": ["full_attention"],
                "rope_scaling": ["rope_type": "yarn", "factor": 4.0],
            ])
        XCTAssertNil(config.ropeScaling(at: 0), "a sliding layer must be left unscaled")
        XCTAssertNotNil(config.ropeScaling(at: 1))

        let unlisted = try Self.config(
            overrides: ["rope_scaling": ["rope_type": "yarn", "factor": 4.0]])
        XCTAssertNotNil(
            unlisted.ropeScaling(at: 0),
            "with no yarn_only_types the scaling applies everywhere")
    }

    /// `partial_rotary_factors` shortens the rotary per layer, so a rotated head has an
    /// untouched tail.
    func testPartialRotaryFactorPerLayer() throws {
        let config = try Self.config(
            overrides: ["partial_rotary_factors": [0.5, 1.0, 1.0, 1.0]])
        XCTAssertEqual(config.partialRotaryFactor(at: 0), 0.5)

        let model = Step3p5Model(config)
        let heads = MLXArray(1 ..< 33).reshaped(1, 4, 1, 8).asType(.float32)
        let partial = applyRotaryPosition(
            model.model.layers[0].attention.rope, to: heads, offset: .scalar(3))
        eval(partial, heads)
        XCTAssertEqual(
            abs(partial[0..., 0..., 0..., 4...] - heads[0..., 0..., 0..., 4...]).max()
                .item(Float.self), 0, accuracy: 1e-7)
        XCTAssertGreaterThan(
            abs(partial[0..., 0..., 0..., ..<4] - heads[0..., 0..., 0..., ..<4]).max()
                .item(Float.self), 1e-3)
    }

    /// A checkpoint that still needs `.moe.` remapping is zero-centered, so every norm
    /// weight gains one; an already-converted checkpoint must be left alone.
    func testNormWeightsGainOneOnlyForAVanillaCheckpoint() throws {
        let model = Step3p5Model(try Self.config())
        let norm = MLXArray.ones([32]) * 0.25

        let vanilla = model.sanitize(weights: [
            "model.layers.1.moe.gate.weight": MLXArray.zeros([6, 32]),
            "model.layers.1.input_layernorm.weight": norm,
        ])
        XCTAssertNotNil(vanilla["model.layers.1.mlp.gate.gate.weight"])
        XCTAssertNil(vanilla["model.layers.1.moe.gate.weight"])
        eval(vanilla["model.layers.1.input_layernorm.weight"]!)
        XCTAssertEqual(
            vanilla["model.layers.1.input_layernorm.weight"]!.max().item(Float.self), 1.25,
            accuracy: 1e-6)

        let converted = model.sanitize(weights: [
            "model.layers.1.mlp.gate.gate.weight": MLXArray.zeros([6, 32]),
            "model.layers.1.input_layernorm.weight": norm,
        ])
        eval(converted["model.layers.1.input_layernorm.weight"]!)
        XCTAssertEqual(
            converted["model.layers.1.input_layernorm.weight"]!.max().item(Float.self),
            0.25, accuracy: 1e-6,
            "a converted checkpoint's norms were shifted a second time")
    }

    func testSanitizeDropsExtraHeadsAndSurplusLayers() throws {
        let model = Step3p5Model(try Self.config())
        let sanitized = model.sanitize(weights: [
            "model.mtp.layers.0.weight": MLXArray.zeros([2, 2]),
            "model.layers.9.input_layernorm.weight": MLXArray.zeros([32]),
            "model.layers.2.input_layernorm.weight": MLXArray.zeros([32]),
        ])
        XCTAssertNil(sanitized["model.mtp.layers.0.weight"])
        XCTAssertNil(sanitized["model.layers.9.input_layernorm.weight"])
        XCTAssertNotNil(sanitized["model.layers.2.input_layernorm.weight"])
    }

}
