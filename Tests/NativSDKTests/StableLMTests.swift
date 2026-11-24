import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class StableLMTests: XCTestCase {
    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(
            configuration: Data(Self.json(parallel: false, qkLayerNorm: true).utf8),
            modelType: "stablelm")
    }

    private static func json(parallel: Bool, qkLayerNorm: Bool, rotary: Float = 0.25) -> String {
        """
        {
          "model_type": "stablelm",
          "vocab_size": 48,
          "hidden_size": 32,
          "num_attention_heads": 4,
          "num_hidden_layers": 2,
          "num_key_value_heads": 2,
          "intermediate_size": 64,
          "rope_theta": 10000.0,
          "use_qkv_bias": true,
          "partial_rotary_factor": \(rotary),
          "layer_norm_eps": 0.00001,
          "use_parallel_residual": \(parallel),
          "qk_layernorm": \(qkLayerNorm)
        }
        """
    }

    private static func config(
        parallel: Bool = false, qkLayerNorm: Bool = false, rotary: Float = 0.25
    )
        throws -> StableLMConfiguration
    {
        try JSONDecoder.json5().decode(
            StableLMConfiguration.self,
            from: XCTUnwrap(
                json(parallel: parallel, qkLayerNorm: qkLayerNorm, rotary: rotary)
                    .data(using: .utf8)))
    }

    func testConfigurationRejectsHeadsThatDoNotDivide() throws {
        let bad = Self.json(parallel: false, qkLayerNorm: false)
            .replacingOccurrences(
                of: "\"num_attention_heads\": 4", with: "\"num_attention_heads\": 5")
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                StableLMConfiguration.self, from: XCTUnwrap(bad.data(using: .utf8))))
    }

    /// Only `partial_rotary_factor` of each head rotates. Rotating the whole head loads
    /// identically and misplaces every position past the fraction, so the width is asserted
    /// against the config rather than trusted.
    func testRotaryCoversOnlyItsFractionOfTheHead() throws {
        // head_dim is 32 / 4 = 8.
        let quarter = StableLMModel(try Self.config(rotary: 0.25))
        let whole = StableLMModel(try Self.config(rotary: 1.0))
        let narrow = quarter.languageModel.model.layers[0].attention
        let wide = whole.languageModel.model.layers[0].attention

        XCTAssertEqual(narrow.rotaryDimensions, 2)
        XCTAssertEqual(wide.rotaryDimensions, 8)

        // And the width is real, not just recorded: position must leave the dimensions
        // past the fraction untouched. Rotating the whole head loads identically and
        // misplaces everything after it.
        let x = MLXArray(1 ..< 9).reshaped(1, 1, 1, 8).asType(.float32)
        let near = applyRotaryPosition(narrow.rope, to: x, offset: .scalar(0))
        let far = applyRotaryPosition(narrow.rope, to: x, offset: .scalar(5))
        eval(near, far)

        let moved = abs(near - far)[0, 0, 0]
        eval(moved)
        XCTAssertGreaterThan(moved[0].item(Float.self), 1e-4)
        for dimension in 2 ..< 8 {
            XCTAssertLessThan(
                moved[dimension].item(Float.self), 1e-6,
                "dimension \(dimension) is past the rotary fraction and must ignore position")
        }
    }

    /// The parallel and sequential blocks share every weight except the second norm, so the
    /// wrong arrangement loads cleanly and answers plausibly.
    func testParallelResidualDiffersFromSequential() throws {
        let sequential = StableLMModel(try Self.config(parallel: false))
        let parallel = StableLMModel(try Self.config(parallel: true))

        XCTAssertNotNil(sequential.languageModel.model.layers[0].postAttentionLayerNorm)
        // The parallel arrangement has no second norm to load, which is the only weight
        // difference between the two.
        XCTAssertNil(parallel.languageModel.model.layers[0].postAttentionLayerNorm)

        let block = sequential.languageModel.model.layers[0]
        let x = (MLXArray(0 ..< 64).reshaped(1, 2, 32).asType(.float32) - 32) / 32
        let mask = createAttentionMask(h: x, cache: nil)

        let normalized = block.inputLayerNorm(x)
        let attended = block.attention(normalized, mask: mask, cache: nil)
        let actual = block(x, mask: mask, cache: nil)
        let asParallel = x + attended + block.mlp(normalized)
        eval(actual, asParallel)

        XCTAssertGreaterThan(
            abs(actual - asParallel).max().item(Float.self), 1e-4,
            "use_parallel_residual is false here, so the block must not be the parallel form")
    }

    /// qk normalization is per head: one LayerNorm weight per head, applied to an unaffine
    /// normalization over each head's own dimensions. A single norm over the flattened
    /// projection would bind a differently shaped weight and normalize across heads.
    func testPerHeadNormHasOneWeightPerHeadAndNormalizesWithinAHead() throws {
        let config = try Self.config(qkLayerNorm: true)
        let model = StableLMModel(config)
        let attention = model.languageModel.model.layers[0].attention

        let qNorm = try XCTUnwrap(attention.qNorm)
        let kNorm = try XCTUnwrap(attention.kNorm)
        XCTAssertEqual(qNorm.norms.count, config.attentionHeads)
        XCTAssertEqual(kNorm.norms.count, config.kvHeads)
        XCTAssertEqual(qNorm.norms[0].weight?.shape, [8])

        // Give head 0 a weight of one and head 1 a weight of two, then check the output
        // scales per head rather than across the projection.
        for (index, norm) in qNorm.norms.enumerated() {
            norm.update(
                parameters: ModuleParameters.unflattened([
                    ("weight", MLXArray.ones([8]) * Float(index + 1))
                ]))
        }
        let x = MLXArray(0 ..< 32).reshaped(1, 1, 4, 8).asType(.float32)
        let out = qNorm(x)
        eval(out)

        let head0 = out[0, 0, 0]
        let head1 = out[0, 0, 1]
        let bare = MLXFast.layerNorm(x, weight: nil, bias: nil, eps: config.layerNormEps)
        eval(bare)

        XCTAssertLessThan(abs(head0 - bare[0, 0, 0]).max().item(Float.self), 1e-5)
        XCTAssertLessThan(abs(head1 - bare[0, 0, 1] * 2).max().item(Float.self), 1e-5)
    }

    func testWeightsBindThroughThePublishedSpelling() throws {
        let model = StableLMModel(try Self.config(qkLayerNorm: true))
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
