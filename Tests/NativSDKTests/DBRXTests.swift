import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class DBRXTests: XCTestCase {
    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(configuration: Data(Self.json().utf8), modelType: "dbrx")
    }

    private static func json(clipQKV: String = "8.0") -> String {
        """
        {
          "model_type": "dbrx",
          "vocab_size": 32,
          "d_model": 16,
          "n_layers": 2,
          "n_heads": 4,
          "attn_config": {
            "kv_n_heads": 2,
            "clip_qkv": \(clipQKV),
            "rope_theta": 500000.0
          },
          "ffn_config": {
            "ffn_hidden_size": 8,
            "moe_num_experts": 4,
            "moe_top_k": 2
          }
        }
        """
    }

    private static func config(clipQKV: String = "8.0") throws -> DBRXConfiguration {
        try JSONDecoder.json5().decode(
            DBRXConfiguration.self,
            from: XCTUnwrap(json(clipQKV: clipQKV).data(using: .utf8)))
    }

    /// None of the shapes live at the top level: `num_key_value_heads`, `rope_theta`,
    /// `intermediate_size`, `num_experts` and `num_experts_per_tok` are all absent, and
    /// their values sit inside `attn_config` and `ffn_config`.
    func testShapesComeFromTheNestedDictionaries() throws {
        let config = try Self.config()
        XCTAssertEqual(config.attention.kvHeads, 2)
        XCTAssertEqual(config.attention.ropeTheta, 500_000)
        XCTAssertEqual(config.attention.clipQKV, 8)
        XCTAssertEqual(config.feedForward.hiddenSize, 8)
        XCTAssertEqual(config.feedForward.experts, 4)
        XCTAssertEqual(config.feedForward.topK, 2)

        let model = DBRXModel(config)
        XCTAssertEqual(model.kvHeads, [2, 2])
        let probe = MLXArray(1 ..< 5).reshaped(1, 1, 1, 4).asType(.float32)
        let rotated = model.transformer.blocks[0].normAttentionNorm.attention.rope(
            probe, offset: 7)
        let nested = RoPE(dimensions: 4, traditional: false, base: 500_000)(probe, offset: 7)
        let flatDefault = RoPE(dimensions: 4, traditional: false, base: 10_000)(
            probe, offset: 7)
        eval(rotated, nested, flatDefault)
        XCTAssertLessThan(abs(rotated - nested).max().item(Float.self), 1e-6)
        XCTAssertGreaterThan(
            abs(rotated - flatDefault).max().item(Float.self), 1e-3,
            "rope fell back to the default base, so attn_config.rope_theta was not read")
    }

    /// The clip lands on the fused `Wqkv` output, before it is split, so it bounds values as
    /// well as queries and keys. Clipping to zero therefore zeroes the whole attention
    /// output; a clip applied to only part of the split would leave values alive.
    func testClipAppliesToTheFusedProjectionBeforeSplitting() throws {
        let attention = DBRXAttention(try Self.config(clipQKV: "0.0"))
        let x = MLXRandom.normal([1, 6, 16], key: MLXRandom.key(3)) * 10
        let out = attention(x, mask: .causal, cache: nil)
        eval(out)
        XCTAssertEqual(
            abs(out).max().item(Float.self), 0, accuracy: 1e-7,
            "values survived a zero clip, so the clip is not covering the fused tensor")

        let unclipped = DBRXAttention(try Self.config(clipQKV: "null"))
        unclipped.update(
            parameters: ModuleParameters.unflattened(attention.parameters().flattened()))
        let free = unclipped(x, mask: .causal, cache: nil)
        eval(free)
        XCTAssertGreaterThan(abs(free).max().item(Float.self), 1e-3)
    }

    /// A realistic clip has to bite too: at `clip_qkv = 8` a projection driven hard enough
    /// saturates, and a port that ignored the key would answer differently.
    func testRealisticClipChangesTheOutput() throws {
        let clipped = DBRXAttention(try Self.config(clipQKV: "8.0"))
        let unclipped = DBRXAttention(try Self.config(clipQKV: "null"))
        unclipped.update(
            parameters: ModuleParameters.unflattened(clipped.parameters().flattened()))

        let x = MLXRandom.normal([1, 6, 16], key: MLXRandom.key(5)) * 60
        let a = clipped(x, mask: .causal, cache: nil)
        let b = unclipped(x, mask: .causal, cache: nil)
        eval(a, b)
        XCTAssertGreaterThan(
            abs(a - b).max().item(Float.self), 1e-2,
            "the clip made no difference even at 60x normal activations")
    }

    /// `norm_attn_norm` hands the block two values. The feed-forward reads the normalized
    /// one; the residual added back is the un-normalized post-attention sum. Adding the
    /// normalized value instead is the natural mistake and still generates text.
    func testBlockAddsTheUnnormalizedResidual() throws {
        let block = DBRXBlock(try Self.config())
        let x = MLXRandom.normal([1, 5, 16], key: MLXRandom.key(9))

        let out = block(x, mask: .causal, cache: nil)

        let inner = block.normAttentionNorm
        let residual = inner.attention(inner.norm1(x), mask: .causal, cache: nil) + x
        let normalized = inner.norm2(residual)
        let correct = block.ffn(normalized) + residual
        let doubleNormed = block.ffn(normalized) + normalized
        eval(out, correct, doubleNormed)

        XCTAssertLessThan(abs(out - correct).max().item(Float.self), 1e-5)
        XCTAssertGreaterThan(
            abs(out - doubleNormed).max().item(Float.self), 1e-3,
            "the two wirings agree, so this probe cannot tell them apart")
    }

    /// A checkpoint packs every expert of a projection into one `experts.mlp.<name>` tensor
    /// with no `.weight` suffix, and stores `w2` transposed relative to `w1` and `v1`.
    func testSanitizeUnpacksStackedExpertsAndTransposesW2() throws {
        let config = try Self.config()
        let model = DBRXModel(config)

        let ramp = MLXArray(0 ..< 512).reshaped(32, 16).asType(.float32)
        let sanitized = model.sanitize(weights: [
            "transformer.blocks.0.ffn.experts.mlp.w1": ramp,
            "transformer.blocks.0.ffn.experts.mlp.v1": ramp,
            "transformer.blocks.0.ffn.experts.mlp.w2": ramp,
            "transformer.blocks.0.ffn.router.layer.weight": MLXArray.zeros([4, 16]),
        ])

        XCTAssertEqual(
            sanitized["transformer.blocks.0.ffn.switch_mlp.gate_proj.weight"]?.shape,
            [4, 8, 16])
        XCTAssertEqual(
            sanitized["transformer.blocks.0.ffn.switch_mlp.up_proj.weight"]?.shape,
            [4, 8, 16])
        let down = try XCTUnwrap(
            sanitized["transformer.blocks.0.ffn.switch_mlp.down_proj.weight"])
        XCTAssertEqual(down.shape, [4, 16, 8])
        eval(down)
        XCTAssertEqual(
            down[0, 3, 2].item(Float.self), 35,
            "w2 was reshaped but not transposed (an untransposed read gives 50)")

        XCTAssertNotNil(sanitized["transformer.blocks.0.ffn.router.layer.weight"])
        XCTAssertNil(sanitized["transformer.blocks.0.ffn.experts.mlp.w1"])
    }

}
