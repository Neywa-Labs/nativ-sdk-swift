import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class Llama4TextTests: XCTestCase {
    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(
            configuration: try JSONEncoder().encode(Self.config(noRoPELayers: [1, 1, 0, 1])),
            modelType: "llama4_text")
    }

    private static func config(noRoPELayers: [Int], tied: Bool = false) throws
        -> Llama4TextConfiguration
    {
        let json = """
            {
              "model_type": "llama4_text",
              "hidden_size": 32,
              "num_attention_heads": 4,
              "num_hidden_layers": 4,
              "vocab_size": 48,
              "intermediate_size": 16,
              "intermediate_size_mlp": 64,
              "num_key_value_heads": 2,
              "rms_norm_eps": 0.00001,
              "rope_theta": 500000.0,
              "head_dim": 8,
              "tie_word_embeddings": \(tied),
              "no_rope_layers": \(noRoPELayers),
              "use_qk_norm": true
            }
            """
        return try JSONDecoder.json5().decode(
            Llama4TextConfiguration.self, from: XCTUnwrap(json.data(using: .utf8)))
    }

    /// A NoPE layer has to actually drop the rotation.
    ///
    /// Isolating that means holding everything except the flag fixed, so layer 0's weights
    /// are copied into layer 1 and both blocks then see the same tokens through the same
    /// fresh cache. The only remaining difference is that layer 0 ropes. Comparing a layer
    /// against itself at two cache offsets does not work: a longer cache gives attention
    /// more keys to read, and the output moves whether or not any rotation was applied.
    func testANoPELayerIgnoresPosition() throws {
        let config = try Self.config(noRoPELayers: [1, 0])
        let model = Llama4TextModel(config)
        let layers = model.languageModel.model.layers

        XCTAssertNotNil(layers[0].attention.rope, "layer 0 is flagged to rope")
        XCTAssertNil(layers[1].attention.rope, "layer 1 is flagged not to rope")

        // Flattening drops the rope entry, which carries no weights: the two blocks'
        // parameter trees differ in shape by exactly that module, and updating across them
        // directly is rejected for it.
        layers[1].update(
            parameters: ModuleParameters.unflattened(layers[0].parameters().flattened()))

        let x = (MLXArray(0 ..< 64).reshaped(1, 2, 32).asType(.float32) / 64)
        func output(_ block: Llama4TextBlock) -> MLXArray {
            let cache = KVCacheSimple()
            let mask = createAttentionMask(h: x, cache: cache)
            let out = block(x, mask: mask, cache: cache)
            eval(out)
            return out
        }

        let roped = output(layers[0])
        let unroped = output(layers[1])
        let difference = abs(roped - unroped).max().item(Float.self)

        XCTAssertGreaterThan(
            difference, 1e-4,
            "the two blocks hold identical weights and differ only in the rope flag, so an "
                + "identical answer means the flag was not honoured")
    }

    func testSanitizerNestsPublishedKeysAndRenamesTheHead() throws {
        let model = Llama4TextModel(try Self.config(noRoPELayers: [1, 1, 1, 1]))
        let sanitized = model.sanitize(weights: [
            "model.embed_tokens.weight": zeros([48, 32]),
            "model.layers.0.self_attn.q_proj.weight": zeros([32, 32]),
            "lm_head.weight": zeros([48, 32]),
            "language_model.model.norm.weight": zeros([32]),
        ])

        XCTAssertNotNil(sanitized["language_model.model.embed_tokens.weight"])
        XCTAssertNotNil(sanitized["language_model.model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(sanitized["language_model.output.weight"])
        XCTAssertNil(sanitized["language_model.lm_head.weight"])
        // Already-nested keys are left where they are rather than nested twice.
        XCTAssertNotNil(sanitized["language_model.model.norm.weight"])
        XCTAssertNil(sanitized["language_model.language_model.model.norm.weight"])
    }

    func testATiedCheckpointDropsTheSeparateHead() throws {
        let model = Llama4TextModel(try Self.config(noRoPELayers: [1, 1, 1, 1], tied: true))
        let sanitized = model.sanitize(weights: [
            "model.embed_tokens.weight": zeros([48, 32]),
            "lm_head.weight": zeros([48, 32]),
        ])
        XCTAssertNotNil(sanitized["language_model.model.embed_tokens.weight"])
        XCTAssertNil(sanitized["language_model.output.weight"])
    }

}
