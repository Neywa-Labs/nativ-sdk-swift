import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class PlamoTests: XCTestCase {
    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(configuration: Data(Self.json().utf8), modelType: "plamo")
    }

    private static func json(sharedHeads: Int = 4) -> String {
        """
        {
          "model_type": "plamo",
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "intermediate_size": 64,
          "num_attention_heads": 8,
          "rms_norm_eps": 0.000001,
          "vocab_size": 48,
          "n_shared_head": \(sharedHeads),
          "rope_theta": 10000.0
        }
        """
    }

    private static func config(sharedHeads: Int = 4) throws -> PlamoConfiguration {
        try JSONDecoder.json5().decode(
            PlamoConfiguration.self,
            from: XCTUnwrap(json(sharedHeads: sharedHeads).data(using: .utf8)))
    }

    /// One norm per layer, feeding both branches, and both join the residual.
    func testBlockIsParallelOffASingleNorm() throws {
        let block = PlamoModel(try Self.config()).model.layers.layers[0]
        let keys = block.parameters().flattened().map { $0.0 }
        XCTAssertFalse(keys.contains { $0.contains("post_attention_layernorm") })

        let none = MLXFast.ScaledDotProductAttentionMaskMode.none
        let x = (MLXArray(0 ..< 64).reshaped(1, 2, 32).asType(.float32) - 32) / 32
        let h = block.norm(x)
        let expected = x + block.attention(h, mask: none, cache: nil) + block.mlp(h)
        let sequential = {
            let a = x + block.attention(h, mask: none, cache: nil)
            return a + block.mlp(block.norm(a))
        }()
        let actual = block(x, mask: none, cache: nil)
        eval(actual, expected, sequential)

        XCTAssertLessThan(abs(actual - expected).max().item(Float.self), 1e-5)
        XCTAssertGreaterThan(abs(actual - sequential).max().item(Float.self), 1e-4)
    }

    /// The cache stores the untiled heads, so its width is the stored count rather than the
    /// query count -- storing the tiled copies would multiply its size by the sharing factor.
    func testCacheHoldsUntiledHeads() throws {
        let config = try Self.config(sharedHeads: 4)
        let model = PlamoModel(config)
        XCTAssertEqual(model.kvHeads, [config.resolvedKVHeads, config.resolvedKVHeads])

        let cache = try model.newCache(configuration: nil)
        _ = model(MLXArray([Int32(3), 9]).reshaped(1, 2), cache: cache)
        let stored = try XCTUnwrap((cache[0] as? KVCacheSimple)?.state.first)
        eval(stored)
        XCTAssertEqual(stored.dim(1), config.resolvedKVHeads)
    }
}
