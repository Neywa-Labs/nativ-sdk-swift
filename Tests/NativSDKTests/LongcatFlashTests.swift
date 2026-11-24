import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class LongcatFlashTests: XCTestCase {

    private static func config(
        zeroExperts: Int = 2, qLoraRank: Int? = 16
    ) throws -> LongcatFlashConfiguration {
        try JSONDecoder.json5().decode(
            LongcatFlashConfiguration.self,
            from: ModelTestConfigurations.configuration(
                "longcat_flash",
                overrides: [
                    "zero_expert_num": zeroExperts, "q_lora_rank": qLoraRank as Any? ?? NSNull(),
                ]))
    }

    /// The mixture reads the *first* sub-layer's normalized state. Reading the second's is
    /// the natural order and shifts its input by a whole block, so the block's output has
    /// to match the first reading and not the second.
    func testMixtureReadsTheFirstSubLayer() throws {
        let config = try Self.config()
        let block = LongcatFlashBlock(config)
        let x = MLXRandom.normal([1, 4, 32], key: MLXRandom.key(11))

        let out = block(x, mask: nil, cache: nil)

        func stage(_ input: MLXArray, _ i: Int) -> (residual: MLXArray, normed: MLXArray) {
            let afterAttention =
                input
                + block.attentions[i](
                    block.inputLayerNorms[i](input), mask: nil, cache: nil)
            return (afterAttention, block.postAttentionLayerNorms[i](afterAttention))
        }

        let first = stage(x, 0)
        let fromFirst = block.mixture(first.normed)
        let afterFirst = first.residual + block.feedForwards[0](first.normed)
        let second = stage(afterFirst, 1)
        let fromSecond = block.mixture(second.normed)
        let tail = second.residual + block.feedForwards[1](second.normed)

        let correct = tail + fromFirst
        let shifted = tail + fromSecond
        eval(out, correct, shifted)

        XCTAssertLessThan(abs(out - correct).max().item(Float.self), 1e-4)
        XCTAssertGreaterThan(
            abs(out - shifted).max().item(Float.self), 1e-4,
            "the two readings agree here, so this probe cannot tell them apart")
    }

    /// The router has `n_routed_experts + zero_expert_num` outputs; the extra ones are
    /// identity experts.
    func testRouterCoversTheIdentityExperts() throws {
        let model = LongcatFlashModel(try Self.config(zeroExperts: 2))
        let router = model.model.layers[0].mixture.router
        XCTAssertEqual(router.classifier.weight.dim(0), 6)
        XCTAssertEqual(router.correctionBias.shape, [6])
        XCTAssertEqual(model.model.layers[0].mixture.switchMLP.numExperts, 4)
    }

    /// An identity expert contributes the hidden state itself, weighted. Biasing the
    /// router entirely onto the identity experts must make the mixture reproduce its input
    /// scaled by the summed weights -- and nothing from the switched feed-forward.
    func testIdentityExpertsReturnTheirInput() throws {
        let config = try Self.config(zeroExperts: 4)
        let mixture = LongcatFlashMoE(config)

        var bias = [Float](repeating: 0, count: 8)
        for i in 4 ..< 8 { bias[i] = 1 }
        mixture.update(
            parameters: ModuleParameters.unflattened([
                ("router.e_score_correction_bias", MLXArray(bias)),
                ("router.classifier.weight", MLXArray.zeros([8, 32])),
            ]))

        let x = MLXRandom.normal([1, 3, 32], key: MLXRandom.key(13))
        let out = mixture(x)
        eval(out, x)

        let expected = x * 0.75
        eval(expected)
        XCTAssertLessThan(
            abs(out - expected).max().item(Float.self), 1e-4,
            "the identity experts did not pass the hidden state through unchanged")
    }

    /// The low-rank rescalings are `sqrt(hidden / rank)` and are switched independently.
    func testLowRankRescalings() throws {
        let scaled = LongcatFlashMLA(try Self.config())
        XCTAssertEqual(scaled.queryLoraScale ?? 0, sqrt(Float(32) / 16), accuracy: 1e-6)
        XCTAssertEqual(
            scaled.keyValueLoraScale ?? 0, sqrt(Float(32) / 16), accuracy: 1e-6)

        let dense = LongcatFlashMLA(try Self.config(qLoraRank: nil))
        XCTAssertNil(dense.queryLoraScale)
        XCTAssertNil(dense.qAProj)
        XCTAssertNotNil(dense.qProj)
        XCTAssertNotNil(scaled.qAProj)
        XCTAssertNil(scaled.qProj)
    }

    /// `kv_b_proj` is fused in the checkpoint and has to be split per sub-layer into the
    /// absorbed projections, with the query half transposed.
    func testSanitizeSplitsTheFusedKeyValueProjection() throws {
        let config = try Self.config()
        let model = LongcatFlashModel(config)
        let fused = MLXArray(0 ..< 1024).reshaped(64, 16).asType(.float32)

        let sanitized = model.sanitize(weights: [
            "model.layers.0.self_attn.0.kv_b_proj.weight": fused,
            "model.layers.0.self_attn.1.kv_b_proj.weight": fused,
            "model.mtp.layers.0.weight": MLXArray.zeros([2, 2]),
        ])

        XCTAssertNil(sanitized["model.layers.0.self_attn.0.kv_b_proj.weight"])
        XCTAssertNil(sanitized["model.mtp.layers.0.weight"])
        XCTAssertEqual(
            sanitized["model.layers.0.self_attn.0.embed_q.weight"]?.shape, [4, 16, 8],
            "embed_q must be transposed to [heads, kv_lora_rank, qk_nope]")
        XCTAssertEqual(
            sanitized["model.layers.0.self_attn.0.unembed_out.weight"]?.shape, [4, 8, 16])
        XCTAssertNotNil(sanitized["model.layers.0.self_attn.1.embed_q.weight"])
    }

    /// Both absorption branches must agree: a short prompt takes the absorbed path and a
    /// long one materializes K and V, and they answer the same.
    func testAbsorbedAndMaterializedPathsAgree() throws {
        let config = try Self.config()
        let attention = LongcatFlashMLA(config)
        let x = MLXRandom.normal([1, 3, 32], key: MLXRandom.key(17))

        let stepwise = MLXArray.zeros([1, 3, 48])
        _ = stepwise
        var perStep: [MLXArray] = []
        let cache = CacheList(KVCacheSimple(), KVCacheSimple())
        for t in 0 ..< 3 {
            perStep.append(attention(x[0..., t ..< (t + 1)], mask: nil, cache: cache[0]))
        }
        let decoded = concatenated(perStep, axis: 1)

        let block = attention(x, mask: createCausalMask(n: 3, offset: 0), cache: nil)
        eval(decoded, block)

        XCTAssertLessThan(
            abs(decoded[0, 2] - block[0, 2]).max().item(Float.self), 5e-3,
            "the absorbed and materialized paths disagree at the last position")
    }

}
