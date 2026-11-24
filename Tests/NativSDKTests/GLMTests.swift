import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class GLMTests: XCTestCase {
    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(configuration: Data(Self.json().utf8), modelType: "glm")
    }

    private static func json(extra: String = "") -> String {
        """
        {
          "model_type": "glm",
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "intermediate_size": 64,
          "num_attention_heads": 4,
          "rms_norm_eps": 0.00001,
          "vocab_size": 48,
          "head_dim": 8,
          "num_key_value_heads": 2\(extra)
        }
        """
    }

    private static func config(extra: String = "") throws -> GLMConfiguration {
        try JSONDecoder.json5().decode(
            GLMConfiguration.self, from: XCTUnwrap(json(extra: extra).data(using: .utf8)))
    }

    /// The fused MLP's gate is the *first* half, where qwen and olmo in this same package
    /// put it second. All three bind one projection of the same shape, so only arithmetic
    /// separates them.
    func testGateIsTheFirstHalfOfTheFusedProjection() throws {
        let mlp = GLMModel(try Self.config()).model.layers[0].mlp
        let x = (MLXArray(0 ..< 32).reshaped(1, 1, 32).asType(.float32) - 16) / 16

        let halves = split(mlp.gateUp(x), parts: 2, axis: -1)
        let gateFirst = mlp.down(silu(halves[0]) * halves[1])
        let gateSecond = mlp.down(silu(halves[1]) * halves[0])
        let actual = mlp(x)
        eval(actual, gateFirst, gateSecond)

        XCTAssertLessThan(abs(actual - gateFirst).max().item(Float.self), 1e-5)
        XCTAssertGreaterThan(
            abs(actual - gateSecond).max().item(Float.self), 1e-4,
            "the two orderings agree here, so this test cannot tell them apart")
    }

    /// Traditional rope, unlike most of the zoo's half-rotation form. The two produce
    /// different positions from the same weights.
    func testRopeIsTraditional() throws {
        let attention = GLMModel(try Self.config()).model.layers[0].attention
        let x = MLXArray(1 ..< 9).reshaped(1, 1, 1, 8).asType(.float32)

        let mine = applyRotaryPosition(attention.rope, to: x, offset: .scalar(3))
        let traditional = MLXFast.RoPE(
            x, dimensions: 8, traditional: true, base: 10000, scale: 1, offset: 3)
        let halved = MLXFast.RoPE(
            x, dimensions: 8, traditional: false, base: 10000, scale: 1, offset: 3)
        eval(mine, traditional, halved)

        XCTAssertLessThan(abs(mine - traditional).max().item(Float.self), 1e-5)
        XCTAssertGreaterThan(abs(traditional - halved).max().item(Float.self), 1e-4)
    }
}
