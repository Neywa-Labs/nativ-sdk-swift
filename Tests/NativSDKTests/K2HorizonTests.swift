import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class K2HorizonTests: XCTestCase {
    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(
            configuration: Data(Self.json().utf8), modelType: "k2_horizon")
    }

    private static func json(rope: String = "") -> String {
        """
        {
          "model_type": "k2_horizon",
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "intermediate_size": 64,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "head_dim": 8,
          "vocab_size": 48,
          "rms_norm_eps": 0.000001,
          "max_position_embeddings": 256\(rope)
        }
        """
    }

    private static func config(rope: String = "") throws -> K2HorizonConfiguration {
        try JSONDecoder.json5().decode(
            K2HorizonConfiguration.self, from: XCTUnwrap(json(rope: rope).data(using: .utf8)))
    }

    /// `rope_parameters` holds the base *inside* the scaling dictionary, and a "default"
    /// type means no scaling at all rather than scaling by one. Reading it as an ordinary
    /// rope_scaling would hand the rope a dictionary it would try to honour.
    func testRopeParametersTranslateToThetaAndScaling() throws {
        // Default type: the base is taken, the scaling is dropped entirely.
        let plain = try Self.config(
            rope: ",\n  \"rope_parameters\": {\"rope_type\": \"default\", \"rope_theta\": 50000.0}")
        XCTAssertEqual(plain.ropeTheta, 50000)
        XCTAssertNil(plain.ropeScaling)

        // yarn: the base is taken and the rest is kept for the rope to honour.
        let yarn = try Self.config(
            rope: """
                ,
                  "rope_parameters": {"rope_type": "yarn", "rope_theta": 12345.0, \
                "factor": 4.0, "original_max_position_embeddings": 128}
                """)
        XCTAssertEqual(yarn.ropeTheta, 12345)
        let scaling = try XCTUnwrap(yarn.ropeScaling)
        XCTAssertEqual(scaling["rope_type"], .string("yarn"))
        // Compared numerically: StringOrNumber reports an integral JSON number as `.int`,
        // so `4.0` arrives as `.int(4)` and a case-wise comparison would be about the
        // decoder rather than about this translation.
        switch scaling["factor"] {
        case .int(let value): XCTAssertEqual(value, 4)
        case .float(let value): XCTAssertEqual(value, 4.0)
        default: XCTFail("factor missing from the kept scaling: \(scaling)")
        }
        // The base must not be left behind in the scaling dictionary it came from.
        XCTAssertNil(scaling["rope_theta"])
    }

    /// Checkpoints converted before the rename carry `rope_scaling` instead, and must read
    /// the same way.
    func testFallsBackToRopeScalingWhenParametersAreAbsent() throws {
        let older = try Self.config(
            rope: ",\n  \"rope_scaling\": {\"rope_type\": \"yarn\", \"factor\": 2.0}")
        XCTAssertNotNil(older.ropeScaling)
        switch older.ropeScaling?["factor"] {
        case .int(let value): XCTAssertEqual(value, 2)
        case .float(let value): XCTAssertEqual(value, 2.0)
        default: XCTFail("factor missing from the fallback scaling")
        }

        // And with neither key, the documented base default applies.
        XCTAssertEqual(try Self.config().ropeTheta, 1e7)
        XCTAssertNil(try Self.config().ropeScaling)
    }
}
