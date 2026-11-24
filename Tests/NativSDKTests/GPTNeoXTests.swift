import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class GPTNeoXTests: XCTestCase {
    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(configuration: Data(Self.json().utf8), modelType: "gpt_neox")
    }

    private static func json(parallel: Bool = true, rotaryPct: Float = 0.25) -> String {
        """
        {
          "model_type": "gpt_neox",
          "max_position_embeddings": 2048,
          "hidden_size": 32,
          "num_attention_heads": 4,
          "num_hidden_layers": 2,
          "layer_norm_eps": 0.00001,
          "vocab_size": 48,
          "rotary_emb_base": 10000,
          "rotary_pct": \(rotaryPct),
          "use_parallel_residual": \(parallel)
        }
        """
    }

    private static func config(parallel: Bool = true, rotaryPct: Float = 0.25) throws
        -> GPTNeoXConfiguration
    {
        try JSONDecoder.json5().decode(
            GPTNeoXConfiguration.self,
            from: XCTUnwrap(json(parallel: parallel, rotaryPct: rotaryPct).data(using: .utf8)))
    }

    /// Both arrangements keep both norms -- unlike stablelm, where the sequential form owns
    /// an extra one. Parallel feeds the second norm the block's input; sequential feeds it
    /// the attention's output. Same weights either way.
    func testParallelResidualFeedsTheSecondNormTheBlockInput() throws {
        let config = try Self.config(parallel: true)
        let model = GPTNeoXModel(config)
        let block = model.languageModel.model.h[0]

        let x = (MLXArray(0 ..< 64).reshaped(1, 2, 32).asType(.float32) - 32) / 32
        let mask = createAttentionMask(h: x, cache: nil)

        let attended = block.attention(block.inputLayerNorm(x), mask: mask, cache: nil)
        let actual = block(x, mask: mask, cache: nil)
        let asParallel = attended + block.mlp(block.postAttentionLayerNorm(x)) + x
        let asSequential =
            (x + attended)
            + block.mlp(
                block.postAttentionLayerNorm(x + attended))
        eval(actual, asParallel, asSequential)

        XCTAssertLessThan(abs(actual - asParallel).max().item(Float.self), 1e-5)
        XCTAssertGreaterThan(abs(actual - asSequential).max().item(Float.self), 1e-4)
    }

    func testRotaryCoversOnlyItsPercentageOfTheHead() throws {
        let model = GPTNeoXModel(try Self.config(rotaryPct: 0.25))
        let attention = model.languageModel.model.h[0].attention
        XCTAssertEqual(attention.rotaryDimensions, 2)

        let x = MLXArray(1 ..< 9).reshaped(1, 1, 1, 8).asType(.float32)
        let moved = abs(
            applyRotaryPosition(attention.rope, to: x, offset: .scalar(0))
                - applyRotaryPosition(attention.rope, to: x, offset: .scalar(5)))[0, 0, 0]
        eval(moved)

        XCTAssertGreaterThan(moved[0].item(Float.self), 1e-4)
        for dimension in 2 ..< 8 {
            XCTAssertLessThan(moved[dimension].item(Float.self), 1e-6)
        }
    }

    func testSanitizerFlattensTheNestingAndDropsBuffers() throws {
        let model = GPTNeoXModel(try Self.config())
        let sanitized = model.sanitize(weights: [
            "gpt_neox.embed_in.weight": MLXArray.zeros([48, 32]),
            "gpt_neox.layers.0.attention.query_key_value.weight": MLXArray.zeros([96, 32]),
            "gpt_neox.final_layer_norm.weight": MLXArray.zeros([32]),
            "embed_out.weight": MLXArray.zeros([48, 32]),
            "gpt_neox.layers.0.attention.bias": MLXArray.ones([1, 1, 8, 8]),
            "gpt_neox.layers.0.attention.masked_bias": MLXArray.ones([1]),
            "gpt_neox.layers.0.attention.rotary_emb.inv_freq": MLXArray.ones([1]),
        ])

        XCTAssertNotNil(sanitized["language_model.model.embed_in.weight"])
        XCTAssertNotNil(
            sanitized["language_model.model.h.0.attention.query_key_value.weight"])
        XCTAssertNotNil(sanitized["language_model.model.final_layer_norm.weight"])
        XCTAssertNotNil(sanitized["language_model.model.embed_out.weight"])

        // Buffers, not parameters: nothing in the tree binds them.
        XCTAssertNil(sanitized["language_model.model.h.0.attention.bias"])
        XCTAssertNil(sanitized["language_model.model.h.0.attention.masked_bias"])
        XCTAssertNil(sanitized["language_model.model.h.0.attention.rotary_emb.inv_freq"])
    }

    func testWeightsBindThroughThePublishedSpelling() throws {
        let model = GPTNeoXModel(try Self.config())
        var published: [String: MLXArray] = [:]
        for (key, value) in model.parameters().flattened() {
            var name = key.replacingOccurrences(of: "language_model.model.", with: "")
            // Only a leading `h.` is the layer list. Replacing it everywhere also rewrites
            // the `h.` inside `dense_4h_to_h.bias`, which is how this test first failed.
            if name.hasPrefix("h.") {
                name = "gpt_neox.layers." + name.dropFirst("h.".count)
            } else if name.hasPrefix("embed_in") || name.hasPrefix("final_layer_norm") {
                name = "gpt_neox." + name
            }
            published[name] = value
        }
        published["gpt_neox.layers.0.attention.bias"] = MLXArray.ones([1, 1, 8, 8])

        let sanitized = model.sanitize(weights: published)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)

        let logits = model(MLXArray([Int32(5), 6]).reshaped(1, 2), cache: nil)
        eval(logits)
        XCTAssertEqual(logits.shape, [1, 2, 48])
    }
}
