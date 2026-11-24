import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class SeedOSSTests: XCTestCase {
    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(configuration: Data(Self.json().utf8), modelType: "seed_oss")
    }

    private static func json(extra: String = "") -> String {
        """
        {
          "model_type": "seed_oss",
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "intermediate_size": 64,
          "num_attention_heads": 4,
          "rms_norm_eps": 0.00001,
          "vocab_size": 48,
          "num_key_value_heads": 2,
          "head_dim": 8\(extra)
        }
        """
    }

    private static func config(extra: String = "") throws -> SeedOSSConfiguration {
        try JSONDecoder.json5().decode(
            SeedOSSConfiguration.self, from: XCTUnwrap(json(extra: extra).data(using: .utf8)))
    }

    /// Three separate bias flags govern different projections, so a checkpoint can ship a
    /// bias on the output projection alone. Collapsing them into one flag leaves either an
    /// unbound weight or a randomly initialized one, depending on which way it collapses.
    func testEachBiasFlagGovernsItsOwnProjections() throws {
        let plain = SeedOSSModel(try Self.config())
        XCTAssertNil(plain.model.layers[0].attention.wq.bias)
        XCTAssertNil(plain.model.layers[0].attention.wo.bias)
        XCTAssertNil(plain.model.layers[0].mlp.gateProj.bias)

        let outOnly = SeedOSSModel(try Self.config(extra: ",\n  \"attention_out_bias\": true"))
        XCTAssertNil(outOnly.model.layers[0].attention.wq.bias)
        XCTAssertNotNil(outOnly.model.layers[0].attention.wo.bias)
        XCTAssertNil(outOnly.model.layers[0].mlp.gateProj.bias)

        let inOnly = SeedOSSModel(try Self.config(extra: ",\n  \"attention_bias\": true"))
        XCTAssertNotNil(inOnly.model.layers[0].attention.wq.bias)
        XCTAssertNil(inOnly.model.layers[0].attention.wo.bias)

        let mlpOnly = SeedOSSModel(try Self.config(extra: ",\n  \"mlp_bias\": true"))
        XCTAssertNotNil(mlpOnly.model.layers[0].mlp.gateProj.bias)
        XCTAssertNil(mlpOnly.model.layers[0].attention.wq.bias)
    }

    func testATiedCheckpointDropsAHeadItShipped() throws {
        let model = SeedOSSModel(try Self.config())
        let sanitized = model.sanitize(weights: [
            "model.embed_tokens.weight": MLXArray.zeros([48, 32]),
            "lm_head.weight": MLXArray.zeros([48, 32]),
        ])
        XCTAssertNotNil(sanitized["model.embed_tokens.weight"])
        XCTAssertNil(sanitized["lm_head.weight"])
    }
}
