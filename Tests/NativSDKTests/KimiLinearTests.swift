import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class KimiLinearTests: XCTestCase {

    private static func config(
        kdaLayers: [Int] = [1, 2, 4], experts: Int = 4, activation: String = "sigmoid"
    ) throws -> KimiLinearConfiguration {
        try JSONDecoder.json5().decode(
            KimiLinearConfiguration.self,
            from: ModelTestConfigurations.configuration(
                "kimi_linear",
                overrides: [
                    "num_experts": experts, "moe_router_activation_func": activation,
                    "linear_attn_config": [
                        "num_heads": 4, "head_dim": 32, "kda_layers": kdaLayers,
                        "short_conv_kernel_size": 4,
                    ],
                ]))
    }

    /// `kda_layers` is one-based, so `[1, 2, 4]` means layers 0, 1 and 3.
    func testKdaLayersAreOneBased() throws {
        let model = KimiLinearModel(try Self.config(kdaLayers: [1, 2, 4]))
        XCTAssertEqual(model.model.layers.map(\.isLinear), [true, true, false, true])
        XCTAssertEqual(
            model.model.attentionIndex, 2, "the first latent layer is index two")
        XCTAssertEqual(model.model.recurrentIndex, 0)

        let shifted = KimiLinearModel(try Self.config(kdaLayers: [2, 3]))
        XCTAssertEqual(shifted.model.layers.map(\.isLinear), [false, true, true, false])
        XCTAssertEqual(shifted.model.recurrentIndex, 1)
        XCTAssertEqual(shifted.model.attentionIndex, 0)
    }

    /// The two attention kinds share the key `self_attn`, so they share one property. This
    /// checks each kind's own weights are published -- the delta layers' convolutions and
    /// the latent layers' compressed projection.
    func testBothAttentionKindsPublishTheirWeights() throws {
        let model = KimiLinearModel(try Self.config())
        let keys = Set(model.parameters().flattened().map(\.0))

        XCTAssertTrue(keys.contains("model.layers.0.self_attn.q_conv.conv.weight"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.A_log"))
        XCTAssertTrue(keys.contains("model.layers.0.self_attn.dt_bias"))
        XCTAssertFalse(keys.contains("model.layers.0.self_attn.kv_a_proj_with_mqa.weight"))

        XCTAssertTrue(keys.contains("model.layers.2.self_attn.kv_a_proj_with_mqa.weight"))
        XCTAssertTrue(keys.contains("model.layers.2.self_attn.embed_q.weight"))
        XCTAssertFalse(keys.contains("model.layers.2.self_attn.q_conv.conv.weight"))
    }

    /// There is no rotary in this model at all: the latent layers attend to their
    /// positional halves unrotated, and every position comes from the delta layers.
    /// Permuting two tokens through a latent layer permutes its output and nothing else.
    func testLatentAttentionHasNoRotary() throws {
        let model = KimiLinearModel(try Self.config())
        let latent = try XCTUnwrap(
            model.model.layers[2].attention as? KimiMLAAttention)

        let x = MLXRandom.normal([1, 3, 32], key: MLXRandom.key(21))
        let swapped = concatenated([x[0..., 1 ..< 2], x[0..., 0 ..< 1], x[0..., 2...]], axis: 1)

        let out = latent(x, mask: nil, cache: nil)
        let swappedOut = latent(swapped, mask: nil, cache: nil)
        let reordered = concatenated(
            [out[0..., 1 ..< 2], out[0..., 0 ..< 1], out[0..., 2...]], axis: 1)
        eval(swappedOut, reordered)

        XCTAssertLessThan(
            abs(swappedOut - reordered).max().item(Float.self), 1e-4,
            "the latent layer noticed the swap, so a rotary crept in")
    }

    /// Four array slots for a delta layer, an ordinary cache for a latent one, and a
    /// forward writes all of them.

    /// The short convolution carries its own tail, so two chunks must reproduce one pass.
    func testShortConvolutionIsResumable() {
        let conv = KimiShortConv1d(channels: 8, kernelSize: 4)
        conv.update(
            parameters: ModuleParameters.unflattened([
                ("conv.weight", MLXRandom.normal([8, 4, 1], key: MLXRandom.key(31)))
            ]))
        let x = MLXRandom.normal([1, 6, 8], key: MLXRandom.key(32))

        let (whole, _) = conv(x, state: nil, mask: nil)
        let (head, tail) = conv(x[0..., ..<4], state: nil, mask: nil)
        let (rest, _) = conv(x[0..., 4...], state: tail, mask: nil)
        let joined = concatenated([head, rest], axis: 1)
        eval(whole, joined)

        XCTAssertEqual(tail.shape, [1, 3, 8], "the tail is kernel_size - 1 positions")
        XCTAssertLessThan(abs(whole - joined).max().item(Float.self), 1e-5)
    }

    /// A masked position is zeroed *before* the convolution, so it cannot leak into its
    /// neighbours' receptive field.
    func testMaskedPositionsAreZeroedBeforeConvolving() {
        let conv = KimiShortConv1d(channels: 4, kernelSize: 2)
        conv.update(
            parameters: ModuleParameters.unflattened([
                ("conv.weight", MLXArray.ones([4, 2, 1]))
            ]))
        let x = MLXArray.ones([1, 3, 4]) * 5
        let maskValues = MLXArray([true, false, true]).reshaped(1, 3)

        let (masked, _) = conv(x, state: nil, mask: maskValues)
        let (plain, _) = conv(x, state: nil, mask: nil)
        eval(masked, plain)

        XCTAssertEqual(
            masked[0, 2, 0].item(Float.self), silu(MLXArray(Float(5))).item(Float.self),
            accuracy: 1e-4)
        XCTAssertEqual(
            plain[0, 2, 0].item(Float.self), silu(MLXArray(Float(10))).item(Float.self),
            accuracy: 1e-4)
    }

    /// The router scores in the logits' own dtype, where every other family here casts to
    /// float32 first.
    ///
    /// What this pins is that the dtype carries through to the returned scores, which then
    /// multiply the expert outputs -- so a bf16 checkpoint's mixture is combined in bf16.
    /// A near-tie could also resolve the other way, but that depends on the checkpoint and
    /// is not constructed here.
    func testRouterScoresInTheLogitsDtype() {
        let logits = MLXArray(converting: [1.0, 1.0 + 1.0 / 512, 0.0, 0.0])
            .reshaped(1, 4).asType(.bfloat16)
        let bias = MLXArray.zeros([4])

        func select(inFloat32: Bool) -> (MLXArray, MLXArray) {
            groupedExpertSelection(
                gates: logits, correctionBias: bias, topK: 2, groups: 1, topGroups: 1,
                routedScalingFactor: 1, normalizeTopK: false, scoring: .sigmoid,
                inFloat32: inFloat32)
        }

        let (narrowIndices, narrowScores) = select(inFloat32: false)
        let (wideIndices, wideScores) = select(inFloat32: true)
        eval(narrowIndices, narrowScores, wideIndices, wideScores)

        XCTAssertEqual(narrowScores.dtype, .bfloat16, "kimi_linear must stay in bf16")
        XCTAssertEqual(wideScores.dtype, .float32, "every other family widens to float32")
        XCTAssertEqual(
            Set(narrowIndices.asArray(Int32.self)), Set(wideIndices.asArray(Int32.self)),
            "this probe is not a selection flip, only a dtype difference")
    }

    /// The correction bias sits on the mixture itself, not under its gate, and the
    /// checkpoint spells the whole block `block_sparse_moe` with `w1`/`w2`/`w3`.
    func testSanitizeRemapsTheMixture() throws {
        let config = try Self.config()
        let model = KimiLinearModel(config)

        var weights: [String: MLXArray] = [
            "model.layers.1.block_sparse_moe.gate.weight": MLXArray.zeros([4, 32]),
            "model.layers.1.block_sparse_moe.gate.e_score_correction_bias":
                MLXArray.zeros([4]),
            "model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight":
                MLXArray.zeros([16, 32]),
            "model.mtp.layers.0.weight": MLXArray.zeros([2, 2]),
        ]
        for expert in 0 ..< 4 {
            weights["model.layers.1.block_sparse_moe.experts.\(expert).w1.weight"] =
                MLXArray.zeros([16, 32])
        }

        let sanitized = model.sanitize(weights: weights)
        XCTAssertEqual(
            sanitized["model.layers.1.mlp.switch_mlp.gate_proj.weight"]?.shape,
            [4, 16, 32], "w1 must land on gate_proj")
        XCTAssertNotNil(sanitized["model.layers.1.mlp.gate.weight"])
        XCTAssertNotNil(sanitized["model.layers.1.mlp.e_score_correction_bias"])
        XCTAssertNil(sanitized["model.layers.1.mlp.gate.e_score_correction_bias"])
        XCTAssertNotNil(sanitized["model.layers.1.mlp.shared_experts.gate_proj.weight"])
        XCTAssertNil(sanitized["model.mtp.layers.0.weight"])
    }

    /// The delta layers' convolution weights arrive transposed, and only they are remapped
    /// -- a latent layer's fused `kv_b_proj` takes the other branch.
    func testSanitizeHandlesBothLayerKinds() throws {
        let model = KimiLinearModel(try Self.config())
        let sanitized = model.sanitize(weights: [
            "model.layers.0.self_attn.q_conv1d.weight": MLXArray.zeros([32, 1, 4]),
            "model.layers.0.self_attn.dt_bias": MLXArray.zeros([4, 32]),
            "model.layers.2.self_attn.kv_b_proj.weight": MLXArray.zeros([64, 16]),
        ])

        XCTAssertEqual(
            sanitized["model.layers.0.self_attn.q_conv.conv.weight"]?.shape, [32, 4, 1],
            "the convolution axis was not moved")
        XCTAssertEqual(sanitized["model.layers.0.self_attn.dt_bias"]?.shape, [128])
        XCTAssertNil(sanitized["model.layers.2.self_attn.kv_b_proj.weight"])
        XCTAssertEqual(
            sanitized["model.layers.2.self_attn.embed_q.weight"]?.shape, [4, 16, 8])
    }

}
