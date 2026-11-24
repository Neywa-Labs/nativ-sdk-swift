import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class NemotronTests: XCTestCase {
    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(configuration: Data(Self.json().utf8), modelType: "nemotron")
    }

    private static func json(extra: String = "") -> String {
        """
        {
          "model_type": "nemotron",
          "hidden_size": 32,
          "hidden_act": "relu2",
          "num_hidden_layers": 2,
          "intermediate_size": 64,
          "num_attention_heads": 4,
          "norm_eps": 0.00001,
          "vocab_size": 48,
          "num_key_value_heads": 2\(extra)
        }
        """
    }

    private static func config(extra: String = "") throws -> NemotronConfiguration {
        try JSONDecoder.json5().decode(
            NemotronConfiguration.self,
            from: XCTUnwrap(json(extra: extra).data(using: .utf8)))
    }

    /// The checkpoint stores the norm scale *minus one*. Applying the stored weight as-is
    /// scales by roughly one instead of by what was trained, which is a quiet accuracy loss
    /// rather than a failure, so the arithmetic is asserted against a plain LayerNorm.
    func testNormAddsOneToItsStoredWeight() throws {
        let dimensions = 8
        let onePlus = NemotronLayerNorm1P(dimensions: dimensions, eps: 1e-5)
        let plain = LayerNorm(dimensions: dimensions, eps: 1e-5)

        let stored = MLXArray([Float](repeating: 0.25, count: dimensions))
        onePlus.update(parameters: ModuleParameters.unflattened([("weight", stored)]))
        plain.update(parameters: ModuleParameters.unflattened([("weight", stored + 1)]))

        let x = (MLXArray(0 ..< dimensions).asType(.float32) - 4).reshaped(1, 1, dimensions)
        let mine = onePlus(x)
        let expected = plain(x)
        eval(mine, expected)
        XCTAssertLessThan(abs(mine - expected).max().item(Float.self), 1e-5)

        // And it is not the same as applying the stored weight directly.
        let naive = LayerNorm(dimensions: dimensions, eps: 1e-5)
        naive.update(parameters: ModuleParameters.unflattened([("weight", stored)]))
        let wrong = naive(x)
        eval(wrong)
        XCTAssertGreaterThan(abs(mine - wrong).max().item(Float.self), 1e-3)
    }

    /// The MLP is ungated: up, squared ReLU, down. There is no gate_proj to bind, so a
    /// SwiGLU shape would leave a missing weight rather than a wrong answer -- but the
    /// squaring itself is easy to drop, and that would answer.
    func testMLPSquaresItsActivation() throws {
        let mlp = NemotronModel(try Self.config()).model.layers[0].mlp
        let keys = mlp.parameters().flattened().map { $0.0 }
        XCTAssertFalse(keys.contains { $0.contains("gate_proj") })

        let x = (MLXArray(0 ..< 32).reshaped(1, 1, 32).asType(.float32) - 16) / 16
        let activated = relu(mlp.up(x))
        let squared = mlp.down(activated * activated)
        let unsquared = mlp.down(activated)
        let actual = mlp(x)
        eval(actual, squared, unsquared)

        XCTAssertLessThan(abs(actual - squared).max().item(Float.self), 1e-5)
        XCTAssertGreaterThan(abs(actual - unsquared).max().item(Float.self), 1e-4)
    }

    /// Upstream raises rather than falling back to default positions, and takes the kind
    /// under either `type` or `rope_type`.
    func testRopeScalingIsValidatedUnderEitherKey() throws {
        XCTAssertNoThrow(try Self.config())
        XCTAssertNoThrow(
            try Self.config(extra: ",\n  \"rope_scaling\": {\"factor\": 2.0, \"type\": \"linear\"}")
        )
        XCTAssertNoThrow(
            try Self.config(
                extra: ",\n  \"rope_scaling\": {\"factor\": 2.0, \"rope_type\": \"linear\"}"))
        XCTAssertThrowsError(
            try Self.config(extra: ",\n  \"rope_scaling\": {\"type\": \"linear\"}"))
        XCTAssertThrowsError(
            try Self.config(extra: ",\n  \"rope_scaling\": {\"factor\": 2.0}"))
        XCTAssertThrowsError(
            try Self.config(
                extra: ",\n  \"rope_scaling\": {\"factor\": 2.0, \"type\": \"dynamic\"}"))
    }
}
