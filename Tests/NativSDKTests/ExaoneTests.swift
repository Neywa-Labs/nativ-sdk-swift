import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class ExaoneTests: XCTestCase {
    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(configuration: Data(Self.json().utf8), modelType: "exaone")
    }

    private static func json(tied: String = "") -> String {
        """
        {
          "model_type": "exaone",
          "hidden_size": 32,
          "num_layers": 2,
          "intermediate_size": 64,
          "num_attention_heads": 4,
          "vocab_size": 48,
          "rope_theta": 10000.0,
          "layer_norm_epsilon": 0.00001,
          "num_key_value_heads": 2\(tied)
        }
        """
    }

    private static func config(tied: Bool? = nil) throws -> ExaoneConfiguration {
        let extra = tied.map { ",\n  \"tie_word_embeddings\": \($0)" } ?? ""
        return try JSONDecoder.json5().decode(
            ExaoneConfiguration.self, from: XCTUnwrap(json(tied: extra).data(using: .utf8)))
    }

    func testATiedCheckpointDropsAHeadItShipped() throws {
        let model = ExaoneModel(try Self.config())
        let sanitized = model.sanitize(weights: [
            "transformer.wte.weight": MLXArray.zeros([48, 32]),
            "lm_head.weight": MLXArray.zeros([48, 32]),
        ])
        XCTAssertNotNil(sanitized["language_model.transformer.wte.weight"])
        XCTAssertNil(sanitized["language_model.lm_head.weight"])
    }

    func testWeightsBindThroughThePublishedSpelling() throws {
        for tied in [true, false] {
            let model = ExaoneModel(try Self.config(tied: tied))
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

}
