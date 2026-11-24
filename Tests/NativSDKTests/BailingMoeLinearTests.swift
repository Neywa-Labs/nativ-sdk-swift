import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class BailingMoeLinearTests: XCTestCase {

    private static func config(groupSize: Int = 2, layers: Int = 4, normHead: Bool = false)
        throws -> BailingMoeLinearConfiguration
    {
        try JSONDecoder.json5().decode(
            BailingMoeLinearConfiguration.self,
            from: ModelTestConfigurations.configuration(
                "bailing_moe_linear",
                overrides: [
                    "layer_group_size": groupSize, "num_hidden_layers": layers,
                    "norm_head": normHead,
                ]))
    }

    /// The last layer of each complete group attends normally, and so does anything
    /// trailing past the last complete group.
    func testGlobalLayerPlacement() throws {
        XCTAssertEqual(
            try (0 ..< 4).map { try Self.config(groupSize: 2, layers: 4).isGlobal(at: $0) },
            [false, true, false, true])
        let trailing = try Self.config(groupSize: 2, layers: 5)
        XCTAssertEqual(
            (0 ..< 5).map { trailing.isGlobal(at: $0) },
            [false, true, false, true, true])
    }

    /// `h_t = h_{t-1} * exp(decay) + k_t^T v_t`, `y_t = (q_t * scale) h_t`, written out.
    func testRecurrenceMatchesTheWrittenForm() {
        let q = MLXArray(converting: [1.0, 1.0]).reshaped(1, 1, 2, 1)
        let k = MLXArray(converting: [1.0, 2.0]).reshaped(1, 1, 2, 1)
        let v = MLXArray(converting: [3.0, 4.0]).reshaped(1, 1, 2, 1)
        let decay = MLXArray(converting: [log(0.5)])

        let (y, state) = bailingRecurrentGLA(
            q: q, k: k, v: v, decay: decay, scale: 2, state: nil)
        eval(y, state)

        XCTAssertEqual(y.reshaped(2).asArray(Float.self), [6, 19])
        XCTAssertEqual(state.reshaped(1).asArray(Float.self), [9.5])
    }

    /// A fresh sequence's first position takes the outer product alone. Passing a zero
    /// state instead is the same thing here, but passing a *nonzero* one must differ --
    /// which is what makes the special case observable at all.
    func testFirstPositionSkipsTheDecay() {
        let q = MLXArray.ones([1, 1, 1, 1])
        let k = MLXArray.ones([1, 1, 1, 1])
        let v = MLXArray.ones([1, 1, 1, 1]) * 5
        let decay = MLXArray(converting: [log(0.5)])

        let (fresh, _) = bailingRecurrentGLA(
            q: q, k: k, v: v, decay: decay, scale: 1, state: nil)
        let (resumed, _) = bailingRecurrentGLA(
            q: q, k: k, v: v, decay: decay, scale: 1,
            state: MLXArray.ones([1, 1, 1, 1]) * 4)
        eval(fresh, resumed)

        XCTAssertEqual(fresh.reshaped(1).asArray(Float.self), [5])
        XCTAssertEqual(resumed.reshaped(1).asArray(Float.self), [7])
    }

    /// The slopes are ALiBi's, negated, and scaled down as the layer index rises -- with
    /// layers 0 and 1 sharing a factor.
    func testSlopesAreNegatedAndScaledPerLayer() {
        let zero = BailingLinearAttention.slopes(heads: 4, layers: 8, layerIndex: 0)
        let one = BailingLinearAttention.slopes(heads: 4, layers: 8, layerIndex: 1)
        let last = BailingLinearAttention.slopes(heads: 4, layers: 8, layerIndex: 7)

        XCTAssertEqual(zero.count, 4)
        XCTAssertTrue(zero.allSatisfy { $0 < 0 }, "the slopes must be negative")
        XCTAssertEqual(zero, one, "layers 0 and 1 share a factor")
        XCTAssertEqual(zero[0], -0.25 * (1 + 1e-5), accuracy: 1e-6)
        XCTAssertEqual(zero[3], -pow(2.0, -8) * (1 + 1e-5), accuracy: 1e-8)
        XCTAssertEqual(last[0], -0.25 * (1 - 6.0 / 7 + 1e-5), accuracy: 1e-6)
        XCTAssertGreaterThan(abs(last[0]), 0)

        XCTAssertEqual(
            BailingLinearAttention.slopes(heads: 6, layers: 4, layerIndex: 2).count, 6)
    }

    /// A linear layer's rotary offset comes from the attention layer's cache. Without it a
    /// decode step would rotate as though it were position zero.
    func testLinearLayersBorrowTheAttentionOffset() throws {
        let model = BailingMoeLinearModel(try Self.config(groupSize: 2, layers: 4))
        XCTAssertEqual(model.model.attentionIndex, 1)

        let cache = try model.newCache(configuration: nil)
        XCTAssertTrue(cache[0] is ArraysCache)
        XCTAssertTrue(cache[1] is KVCacheSimple)

        _ = model(MLXArray(0 ..< 5).reshaped(1, 5), cache: cache)
        eval(cache[1].innerState())
        XCTAssertEqual(cache[1].offset, 5)
        XCTAssertNotNil((cache[0] as? ArraysCache)?[0], "the recurrent state was not kept")

        let step = model(MLXArray([Int32(9)]).reshaped(1, 1), cache: cache)
        eval(step)
        XCTAssertEqual(cache[1].offset, 6)
    }

    /// `GroupRMSNorm` normalizes within groups and then applies one full-width weight.
    /// Scaling one group's input must leave the other groups untouched.
    func testGroupNormNormalizesWithinGroups() {
        let norm = BailingGroupRMSNorm(dimensions: 8, eps: 1e-5, groups: 2)
        let x = MLXArray(converting: [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0])
            .reshaped(1, 8)
        let scaledFirst = MLXArray(
            converting: [10.0, 20.0, 30.0, 40.0, 1.0, 2.0, 3.0, 4.0]).reshaped(1, 8)

        let a = norm(x)
        let b = norm(scaledFirst)
        eval(a, b)

        XCTAssertLessThan(
            abs(a[0, 4...] - b[0, 4...]).max().item(Float.self), 1e-5,
            "scaling the first group changed the second, so the groups are not separate")
        XCTAssertLessThan(
            abs(a[0, ..<4] - b[0, ..<4]).max().item(Float.self), 1e-4,
            "RMS normalization is scale invariant, so the first group should match too")
        XCTAssertEqual(norm.weight.shape, [8], "one weight across the whole axis")
    }

    /// `norm_head` normalizes the head's columns at load, and only when the head is untied.
    func testNormalizeHeadScalesColumnsToUnitLength() throws {
        let model = BailingMoeLinearModel(try Self.config(normHead: true))
        let head = MLXArray.ones([48, 32]) * 3
        let sanitized = model.sanitize(weights: ["lm_head.weight": head])
        let normalized = try XCTUnwrap(sanitized["lm_head.weight"])
        eval(normalized)

        let columnNorms = sqrt(normalized.square().sum(axis: 0))
        eval(columnNorms)
        XCTAssertEqual(columnNorms.max().item(Float.self), 1, accuracy: 1e-4)

        let plain = BailingMoeLinearModel(try Self.config())
        let untouched = try XCTUnwrap(
            plain.sanitize(weights: ["lm_head.weight": head])["lm_head.weight"])
        eval(untouched)
        XCTAssertEqual(untouched.max().item(Float.self), 3, accuracy: 1e-6)
    }

    func testSanitizeStacksExpertsAndMovesTheRouterWeight() throws {
        let config = try Self.config()
        let model = BailingMoeLinearModel(config)

        var weights: [String: MLXArray] = [
            "model.layers.1.mlp.gate.weight": MLXArray.zeros([6, 32])
        ]
        for expert in 0 ..< 6 {
            weights["model.layers.1.mlp.experts.\(expert).gate_proj.weight"] =
                MLXArray.zeros([16, 32])
        }
        let sanitized = model.sanitize(weights: weights)

        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape,
            [6, 16, 32])
        XCTAssertNotNil(sanitized["model.layers.1.mlp.gate.gate_proj.weight"])
        XCTAssertNil(sanitized["model.layers.1.mlp.gate.weight"])
    }

}
