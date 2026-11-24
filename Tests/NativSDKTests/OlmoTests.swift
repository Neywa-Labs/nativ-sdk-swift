import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class OlmoTests: XCTestCase {
    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(configuration: Data(Self.json().utf8), modelType: "olmo")
    }

    private static func json(mlpHidden: String = "", tying: Bool = false) -> String {
        """
        {
          "model_type": "olmo",
          "d_model": 32,
          "n_layers": 2,
          "n_heads": 4,
          "vocab_size": 40,
          "embedding_size": 48,
          "rope_theta": 10000.0,
          "weight_tying": \(tying)\(mlpHidden)
        }
        """
    }

    private static func config(mlpHidden: Int? = nil, tying: Bool = false) throws
        -> OlmoConfiguration
    {
        let extra = mlpHidden.map { ",\n  \"mlp_hidden_size\": \($0)" } ?? ""
        return try JSONDecoder.json5().decode(
            OlmoConfiguration.self,
            from: XCTUnwrap(json(mlpHidden: extra, tying: tying).data(using: .utf8)))
    }

    /// The embedding is sized by `embedding_size`, not `vocab_size`; published configs give
    /// both and the embedding is the larger. Sizing it by vocab_size builds a head too
    /// narrow for the checkpoint and truncates the vocabulary.
    func testEmbeddingIsSizedByEmbeddingSizeNotVocabSize() throws {
        let config = try Self.config()
        XCTAssertEqual(config.vocabularySize, 40)
        XCTAssertEqual(config.embeddingSize, 48)

        let model = OlmoModel(config)
        XCTAssertEqual(model.vocabularySize, 48)
        XCTAssertEqual(model.model.transformer.wte.weight.dim(0), 48)

        let logits = model(MLXArray([Int32(3), 9]).reshaped(1, 2), cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 2, 48])
    }

    /// `swiglu(second, first)` upstream: the gate is the *second* half of the projection.
    /// Both readings bind the same weights, so only arithmetic separates them.
    func testGateIsTheSecondHalfOfTheProjection() throws {
        let block = OlmoModel(try Self.config()).model.transformer.blocks[0]
        let none = MLXFast.ScaledDotProductAttentionMaskMode.none
        let x = (MLXArray(0 ..< 32).reshaped(1, 1, 32).asType(.float32) - 16) / 16

        let h = x + block.attend(block.attNorm(x), mask: none, cache: nil)
        let halves = split(block.ffProj(block.ffNorm(h)), parts: 2, axis: -1)
        let gateSecond = h + block.ffOut(silu(halves[1]) * halves[0])
        let gateFirst = h + block.ffOut(silu(halves[0]) * halves[1])

        let actual = block(x, mask: none, cache: nil)
        eval(actual, gateSecond, gateFirst)

        XCTAssertLessThan(abs(actual - gateSecond).max().item(Float.self), 1e-5)
        XCTAssertGreaterThan(
            abs(actual - gateFirst).max().item(Float.self), 1e-4,
            "the two orderings agree here, so this test cannot tell them apart")
    }

    func testWeightTyingDropsTheSeparateHead() throws {
        XCTAssertNotNil(OlmoModel(try Self.config(tying: false)).model.transformer.ffOut)
        XCTAssertNil(OlmoModel(try Self.config(tying: true)).model.transformer.ffOut)
    }

    func testWeightsBindThroughThePublishedSpelling() throws {
        let model = OlmoModel(try Self.config())
        var published: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() {
            published[String(key.dropFirst("model.".count))] = value
        }
        let sanitized = model.sanitize(weights: published)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)

        let logits = model(MLXArray([Int32(5), 6]).reshaped(1, 2), cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 2, 48])
    }

}
