import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class QwenTests: XCTestCase {
    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(configuration: Data(Self.json.utf8), modelType: "qwen")
    }

    private static let json = """
        {
          "model_type": "qwen",
          "hidden_size": 32,
          "num_attention_heads": 4,
          "num_hidden_layers": 2,
          "kv_channels": 8,
          "max_position_embeddings": 512,
          "layer_norm_epsilon": 0.000001,
          "intermediate_size": 128,
          "vocab_size": 48
        }
        """

    private static func config() throws -> QwenConfiguration {
        try JSONDecoder.json5().decode(
            QwenConfiguration.self, from: XCTUnwrap(json.data(using: .utf8)))
    }

    /// `c_attn` is biased whatever `no_bias` says; everything else follows it. A checkpoint
    /// ships that bias, so building `c_attn` without one leaves it unbound.
    func testOnlyTheFusedProjectionIsAlwaysBiased() throws {
        let model = QwenModel(try Self.config())
        let attention = model.languageModel.transformer.h[0].attention
        let mlp = model.languageModel.transformer.h[0].mlp

        XCTAssertNotNil(attention.cAttn.bias)
        XCTAssertNil(attention.cProj.bias)
        XCTAssertNil(mlp.w1.bias)
        XCTAssertNil(mlp.cProj.bias)
        XCTAssertNil(model.languageModel.lmHead.bias)
    }

    /// Upstream calls `swiglu(a2, a1)`, so the gate is `w2` and the linear branch is `w1` --
    /// the reverse of how these names usually read. Both orderings bind the same weights,
    /// so only arithmetic can tell them apart.
    func testGateIsW2NotW1() throws {
        let config = try Self.config()
        let mlp = QwenModel(config).languageModel.transformer.h[0].mlp

        let x = (MLXArray(0 ..< 32).reshaped(1, 1, 32).asType(.float32) - 16) / 16
        let actual = mlp(x)
        let gateIsW2 = mlp.cProj(silu(mlp.w2(x)) * mlp.w1(x))
        let gateIsW1 = mlp.cProj(silu(mlp.w1(x)) * mlp.w2(x))
        eval(actual, gateIsW2, gateIsW1)

        XCTAssertLessThan(abs(actual - gateIsW2).max().item(Float.self), 1e-5)
        XCTAssertGreaterThan(
            abs(actual - gateIsW1).max().item(Float.self), 1e-4,
            "the two orderings agree here, so this test cannot tell them apart")
    }

    func testWeightsBindThroughThePublishedSpelling() throws {
        let model = QwenModel(try Self.config())
        var published: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() {
            published[key.replacingOccurrences(of: "language_model.", with: "")] = value
        }
        let sanitized = model.sanitize(weights: published)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)

        let logits = model(MLXArray([Int32(5), 6]).reshaped(1, 2), cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 2, 48])
    }
}
