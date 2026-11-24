import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class PhixtralTests: XCTestCase {
    func testQuantizedCheckpointPreservesLogits() async throws {
        try await assertTextCheckpoint(
            configuration: Data(Self.json.utf8), modelType: "phixtral", quantized: true)
    }

    func testCheckpointLoadingAndCachedContinuation() async throws {
        for modelType in ["phixtral", "phi-msft"] {
            try await assertTextCheckpoint(
                configuration: Data(Self.json.utf8), modelType: modelType)
        }
    }

    private static let json = """
        {
          "model_type": "phixtral",
          "num_vocab": 48,
          "model_dim": 32,
          "num_heads": 4,
          "num_layers": 2,
          "rotary_dim": 4,
          "num_experts_per_tok": 2,
          "num_local_experts": 4
        }
        """

    private static func config() throws -> PhixtralConfiguration {
        try JSONDecoder.json5().decode(
            PhixtralConfiguration.self, from: XCTUnwrap(json.data(using: .utf8)))
    }

    /// Checkpoints ship each expert separately; the runtime wants them stacked on a leading
    /// axis. A quantized checkpoint carries scales and biases that stack the same way.
    func testSanitizerStacksSeparatelyShippedExperts() throws {
        let model = PhixtralModel(try Self.config())
        var weights: [String: MLXArray] = [:]
        for expert in 0 ..< 4 {
            weights["transformer.h.0.moe.mlp.\(expert).fc1.weight"] =
                MLXArray.zeros([128, 32]) + Float(expert)
            weights["transformer.h.0.moe.mlp.\(expert).fc1.bias"] = MLXArray.zeros([128])
            weights["transformer.h.0.moe.mlp.\(expert).fc2.weight"] =
                MLXArray.zeros([32, 128])
        }
        weights["transformer.embd.wte.weight"] = MLXArray.zeros([48, 32])

        let sanitized = model.sanitize(weights: weights)

        let stacked = try XCTUnwrap(
            sanitized["transformer.h.0.moe.switch_mlp.fc1.weight"])
        XCTAssertEqual(stacked.shape, [4, 128, 32])
        // Stacked in expert order, not some other order.
        eval(stacked)
        XCTAssertEqual(stacked[2, 0, 0].item(Float.self), 2.0)

        XCTAssertEqual(
            sanitized["transformer.h.0.moe.switch_mlp.fc1.bias"]?.shape, [4, 128])
        XCTAssertEqual(
            sanitized["transformer.h.0.moe.switch_mlp.fc2.weight"]?.shape, [4, 32, 128])
        // The per-expert keys are consumed, not left behind to bind to nothing.
        XCTAssertNil(sanitized["transformer.h.0.moe.mlp.0.fc1.weight"])
        XCTAssertNotNil(sanitized["transformer.embd.wte.weight"])
    }

}
