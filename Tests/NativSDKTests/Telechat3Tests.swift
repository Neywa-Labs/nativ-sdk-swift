import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class Telechat3Tests: XCTestCase {
    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(
            configuration: Data(Self.json().utf8), modelType: "telechat3")
    }

    private static func json(extra: String = "") -> String {
        """
        {
          "model_type": "telechat3",
          "hidden_size": 32,
          "intermediate_size": 64,
          "max_position_embeddings": 128,
          "num_attention_heads": 4,
          "num_hidden_layers": 2,
          "num_key_value_heads": 2,
          "rms_norm_eps": 0.00001,
          "vocab_size": 48,
          "rope_theta": 10000.0\(extra)
        }
        """
    }

    private static func config(extra: String = "") throws -> Telechat3Configuration {
        try JSONDecoder.json5().decode(
            Telechat3Configuration.self,
            from: XCTUnwrap(json(extra: extra).data(using: .utf8)))
    }

    /// One flag here for all four attention projections, where seed_oss beside it splits
    /// the output projection onto its own. Reusing seed_oss's two-flag reading would leave
    /// the output projection's bias unbound on a checkpoint that ships it.
    func testOneFlagGovernsAllFourAttentionProjections() throws {
        let plain = Telechat3Model(try Self.config())
        XCTAssertNil(plain.model.layers[0].attention.wq.bias)
        XCTAssertNil(plain.model.layers[0].attention.wo.bias)

        let biased = Telechat3Model(try Self.config(extra: ",\n  \"attention_bias\": true"))
        XCTAssertNotNil(biased.model.layers[0].attention.wq.bias)
        XCTAssertNotNil(biased.model.layers[0].attention.wo.bias)
        // The MLP keeps its own flag.
        XCTAssertNil(biased.model.layers[0].mlp.gateProj.bias)
    }
}
