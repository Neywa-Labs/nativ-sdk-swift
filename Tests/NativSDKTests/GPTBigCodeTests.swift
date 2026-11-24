import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class GPTBigCodeTests: XCTestCase {
    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(
            configuration: Data(Self.json().utf8), modelType: "gpt_bigcode")
    }

    private static func json(multiQuery: Bool = true) -> String {
        """
        {
          "model_type": "gpt_bigcode",
          "n_embd": 32,
          "n_layer": 2,
          "n_inner": 64,
          "n_head": 4,
          "n_positions": 64,
          "layer_norm_epsilon": 0.00001,
          "vocab_size": 48,
          "multi_query": \(multiQuery)
        }
        """
    }

    private static func config(multiQuery: Bool = true) throws -> GPTBigCodeConfiguration {
        try JSONDecoder.json5().decode(
            GPTBigCodeConfiguration.self,
            from: XCTUnwrap(json(multiQuery: multiQuery).data(using: .utf8)))
    }

    /// The fused projection is `dim + 2 * kv_dim`, and under multi_query the key and value
    /// halves are one head each -- so the three pieces are 32, 8, 8 here, nothing like
    /// equal. An even three-way split would take the queries' tail as the keys, which
    /// binds the same weight and produces attention over the wrong vectors.
    func testFusedProjectionIsSplitAsymmetrically() throws {
        let config = try Self.config()
        let attention = GPTBigCodeModel(config).languageModel.transformer.h[0].attention

        XCTAssertEqual(attention.dimensions, 32)
        XCTAssertEqual(attention.kvDimensions, 8)
        XCTAssertEqual(attention.cAttn.weight.dim(0), 32 + 2 * 8)

        // The even split this is not: 48 / 3 = 16, so a third would run into the queries.
        XCTAssertNotEqual(attention.cAttn.weight.dim(0) / 3, attention.dimensions)
    }

    /// Positions are looked up, not rotated, so the cache offset has to reach wpe. One
    /// model, one changed position row, identical cache contents -- the same isolation the
    /// gpt2 test needs, and for the same reason.
    func testPositionEmbeddingFollowsTheCacheOffset() throws {
        let config = try Self.config()
        let model = GPTBigCodeModel(config)
        let transformer = model.languageModel.transformer

        func output(positionRowThree: Float) throws -> MLXArray {
            let wpe = MLXArray.zeros([config.positions, config.embeddingDimensions])
            // A ramp, not a constant: LayerNorm subtracts the mean, so a uniform row is
            // removed before it reaches attention.
            let ramp = MLXArray(0 ..< config.embeddingDimensions).asType(.float32)
            wpe[3] = (ramp - Float(config.embeddingDimensions) / 2) * positionRowThree
            transformer.wpe.update(
                parameters: ModuleParameters.unflattened([("weight", wpe)]))

            let cache = try model.newCache(configuration: nil)
            _ = transformer(MLXArray([Int32(1), 2, 3]).reshaped(1, 3), cache: cache)
            XCTAssertEqual(cache[0].offset, 3)

            let out = transformer(MLXArray([Int32(6)]).reshaped(1, 1), cache: cache)
            eval(out)
            return out
        }

        let quiet = try output(positionRowThree: 0)
        let loud = try output(positionRowThree: 8)
        XCTAssertGreaterThan(
            abs(quiet - loud).max().item(Float.self), 1e-3,
            "the two runs differ only in position row 3, so an identical answer means the "
                + "cache offset never reached the position embedding")
    }

    func testConfigurationRejectsHeadsThatDoNotDivide() throws {
        let bad = Self.json().replacingOccurrences(of: "\"n_head\": 4", with: "\"n_head\": 5")
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                GPTBigCodeConfiguration.self, from: XCTUnwrap(bad.data(using: .utf8))))
    }

    func testWeightsBindThroughThePublishedSpelling() throws {
        let model = GPTBigCodeModel(try Self.config())
        var published: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() {
            published[key.replacingOccurrences(of: "language_model.", with: "")] = value
        }
        published["lm_head.weight"] = MLXArray.zeros([48, 32])

        let sanitized = model.sanitize(weights: published)
        XCTAssertNil(sanitized["language_model.lm_head.weight"])
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)

        let logits = model(MLXArray([Int32(5), 6]).reshaped(1, 2), cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 2, 48])
    }

}
