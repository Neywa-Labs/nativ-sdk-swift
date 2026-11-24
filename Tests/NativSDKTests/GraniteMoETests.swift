import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class GraniteMoETests: XCTestCase {
    func testCheckpointLoadingAndCachedContinuation() async throws {
        try await assertTextCheckpoint(
            configuration: Data(Self.json().utf8), modelType: "granitemoe")
    }

    private static func json(extra: String = "") -> String {
        """
        {
          "model_type": "granitemoe",
          "hidden_size": 32,
          "num_hidden_layers": 2,
          "intermediate_size": 48,
          "num_attention_heads": 4,
          "rms_norm_eps": 0.00001,
          "vocab_size": 48,
          "logits_scaling": 4.0,
          "attention_multiplier": 0.0625,
          "embedding_multiplier": 12.0,
          "residual_multiplier": 0.25,
          "max_position_embeddings": 256,
          "num_key_value_heads": 2,
          "attention_bias": false,
          "rope_theta": 10000.0,
          "num_local_experts": 4,
          "num_experts_per_tok": 2\(extra)
        }
        """
    }

    private static func config(extra: String = "") throws -> GraniteMoEConfiguration {
        try JSONDecoder.json5().decode(
            GraniteMoEConfiguration.self,
            from: XCTUnwrap(json(extra: extra).data(using: .utf8)))
    }

    func testTokenEmbeddingsAreScaledBeforeTheDecoder() throws {
        let config = try Self.config()
        let model = GraniteMoEModel(config).model
        let tokens = MLXArray([Int32(3), 9]).reshaped(1, 2)
        let embedding = model.embedTokens(tokens)
        let mask = createAttentionMask(h: embedding, cache: nil)
        func decode(_ input: MLXArray) -> MLXArray {
            var hidden = input
            for layer in model.layers {
                hidden = layer(hidden, mask: mask, cache: nil)
            }
            return model.norm(hidden)
        }
        let expected = decode(embedding * config.embeddingMultiplier)
        let unscaled = decode(embedding)
        let actual = model(tokens, cache: nil)
        eval(actual, expected, unscaled)
        XCTAssertLessThan(abs(actual - expected).max().item(Float.self), 1e-5)
        XCTAssertGreaterThan(abs(actual - unscaled).max().item(Float.self), 1e-3)
    }

    /// `attention_multiplier` *replaces* the 1/sqrt(head_dim) scale rather than adjusting
    /// it. Head dim is 8 here, so the usual scale would be 0.3536 against Granite's
    /// 0.0625 -- a factor of five and a half, and the model still runs either way.
    func testAttentionScaleIsTheMultiplierNotOneOverRootD() throws {
        let config = try Self.config()
        let attention = GraniteMoEModel(config).model.layers[0].attention

        XCTAssertEqual(attention.scale, config.attentionMultiplier)
        XCTAssertNotEqual(
            attention.scale, pow(Float(attention.headDimensions), -0.5), accuracy: 1e-6)
    }

    /// Both residual branches are scaled. At 0.25 the model still answers fluently without
    /// it, which is why this is asserted rather than eyeballed.
    func testBothResidualBranchesAreScaled() throws {
        let config = try Self.config()
        let block = GraniteMoEModel(config).model.layers[0]
        XCTAssertEqual(block.residualMultiplier, 0.25)

        let none = MLXFast.ScaledDotProductAttentionMaskMode.none
        let x = (MLXArray(0 ..< 64).reshaped(1, 2, 32).asType(.float32) - 32) / 32

        let attended = block.attention(block.inputLayerNorm(x), mask: none, cache: nil)
        let h = x + attended * block.residualMultiplier
        let expected = h + block.moe(block.postAttentionLayerNorm(h)) * block.residualMultiplier
        let unscaled = (x + attended) + block.moe(block.postAttentionLayerNorm(x + attended))
        let actual = block(x, mask: none, cache: nil)
        eval(actual, expected, unscaled)

        XCTAssertLessThan(abs(actual - expected).max().item(Float.self), 1e-5)
        XCTAssertGreaterThan(abs(actual - unscaled).max().item(Float.self), 1e-4)
    }

    /// `logits_scaling` divides. Multiplying instead would sharpen the distribution by the
    /// square of the factor rather than flattening it.
    func testLogitsAreDividedByTheScaling() throws {
        let config = try Self.config()
        let model = GraniteMoEModel(config)
        let tokens = MLXArray([Int32(3), 9]).reshaped(1, 2)

        let hidden = model.model(tokens, cache: nil)
        let unscaled = model.model.embedTokens.asLinear(hidden)
        let actual = model(tokens, cache: nil)
        eval(actual, unscaled)

        XCTAssertLessThan(
            abs(actual - unscaled / config.logitsScaling).max().item(Float.self), 1e-4)
        XCTAssertGreaterThan(
            abs(actual - unscaled * config.logitsScaling).max().item(Float.self), 1e-3)
    }

    /// Granite ships gate and up fused on the output axis of an already-per-expert tensor,
    /// so this is a split rather than the stack the other mixtures need. Gate is the first
    /// half; the other order binds the same shapes and swaps every expert's branches.
    func testSanitizerSplitsTheFusedExpertProjection() throws {
        let config = try Self.config()
        let model = GraniteMoEModel(config)
        let experts = config.localExperts
        let hidden = config.intermediateSize

        // Gate half is zeros, up half is ones, so which half went where is unambiguous.
        let gateHalf = MLXArray.zeros([experts, hidden, config.hiddenSize])
        let upHalf = MLXArray.ones([experts, hidden, config.hiddenSize])
        let fused = concatenated([gateHalf, upHalf], axis: 1)

        let sanitized = model.sanitize(weights: [
            "model.layers.0.block_sparse_moe.input_linear.weight": fused,
            "model.layers.0.block_sparse_moe.output_linear.weight":
                MLXArray.zeros([experts, config.hiddenSize, hidden]),
            "model.embed_tokens.weight": MLXArray.zeros([48, 32]),
        ])

        let gate = try XCTUnwrap(
            sanitized["model.layers.0.block_sparse_moe.switch_mlp.gate_proj.weight"])
        let up = try XCTUnwrap(
            sanitized["model.layers.0.block_sparse_moe.switch_mlp.up_proj.weight"])
        eval(gate, up)

        XCTAssertEqual(gate.shape, [experts, hidden, config.hiddenSize])
        XCTAssertEqual(gate.max().item(Float.self), 0, "gate should be the first half")
        XCTAssertEqual(up.min().item(Float.self), 1, "up should be the second half")

        XCTAssertNotNil(
            sanitized["model.layers.0.block_sparse_moe.switch_mlp.down_proj.weight"])
        XCTAssertNil(sanitized["model.layers.0.block_sparse_moe.input_linear.weight"])
    }

    /// The router keeps the largest logits and softmaxes those, so the kept weights sum to
    /// one -- phixtral's order, not qwen2_moe's.
    func testRouterKeepsTheLargestAndNormalizesThem() throws {
        let config = try Self.config()
        let router = GraniteMoEModel(config).model.layers[0].moe.router
        let x = (MLXArray(0 ..< 32).reshaped(1, 1, 32).asType(.float32) - 16) / 16

        let (indices, gates) = router(x)
        eval(indices, gates)

        XCTAssertEqual(gates.shape.last, config.expertsPerToken)
        XCTAssertEqual(gates.sum().item(Float.self), 1.0, accuracy: 1e-5)

        // The selected logits really are the largest ones.
        let logits = router.layer(x)
        eval(logits)
        let keptMin = takeAlong(logits, indices, axis: -1).min().item(Float.self)
        let all = (0 ..< config.localExperts).map { logits[0, 0, $0].item(Float.self) }
        let dropped = all.sorted().prefix(config.localExperts - config.expertsPerToken)
        XCTAssertGreaterThanOrEqual(keptMin, dropped.max() ?? -.infinity)
    }

}
