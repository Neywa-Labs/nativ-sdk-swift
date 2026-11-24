import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class KimiK3Tests: XCTestCase {

    private static func json(
        layers: Int = 4,
        kdaLayers: String = "[1, 3]",
        residualBlockSize: String = "null",
        latentSize: String = "null",
        linearBeta: String = "null",
        experts: Int = 4,
        sharedExperts: Int = 1,
        fullRankGate: Bool = false
    ) -> String {
        """
        {
          "model_type": "kimi_k3",
          "media_placeholder_token_id": 90,
          "text_config": {
            "model_type": "kimi_linear",
            "vocab_size": 128,
            "hidden_size": 32,
            "num_hidden_layers": \(layers),
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "intermediate_size": 48,
            "rms_norm_eps": 1e-05,
            "hidden_act": "situ",
            "activation_situ_beta": 2.0,
            "activation_situ_linear_beta": \(linearBeta),
            "attn_res_block_size": \(residualBlockSize),
            "q_lora_rank": 24,
            "kv_lora_rank": 16,
            "qk_nope_head_dim": 16,
            "qk_rope_head_dim": 8,
            "v_head_dim": 16,
            "mla_use_nope": true,
            "mla_use_output_gate": true,
            "num_experts": \(experts),
            "num_experts_per_token": 2,
            "num_shared_experts": \(sharedExperts),
            "moe_intermediate_size": 24,
            "moe_router_activation_func": "sigmoid",
            "moe_renormalize": true,
            "first_k_dense_replace": 1,
            "num_expert_group": 2,
            "topk_group": 1,
            "routed_expert_hidden_size": \(latentSize),
            "tie_word_embeddings": false,
            "linear_attn_config": {
              "num_heads": 4, "head_dim": 32, "short_conv_kernel_size": 4,
              "gate_lower_bound": -5.0, "use_full_rank_gate": \(fullRankGate),
              "kda_layers": \(kdaLayers)
            }
          },
          "vision_config": {
            "model_type": "moonvit3d",
            "patch_size": 4,
            "init_pos_emb_height": 8,
            "init_pos_emb_width": 8,
            "init_pos_emb_time": 2,
            "vt_num_attention_heads": 4,
            "vt_num_hidden_layers": 2,
            "vt_hidden_size": 24,
            "vt_intermediate_size": 32,
            "qkv_hidden_size": 32,
            "merge_kernel_size": [2, 2],
            "mm_hidden_size": 24,
            "text_hidden_size": 32
          }
        }
        """
    }

    private static func config(
        layers: Int = 4,
        kdaLayers: String = "[1, 3]",
        residualBlockSize: String = "null",
        latentSize: String = "null",
        linearBeta: String = "null",
        experts: Int = 4,
        sharedExperts: Int = 1,
        fullRankGate: Bool = false
    ) throws -> KimiK3Configuration {
        try JSONDecoder.json5().decode(
            KimiK3Configuration.self,
            from: XCTUnwrap(
                json(
                    layers: layers, kdaLayers: kdaLayers,
                    residualBlockSize: residualBlockSize, latentSize: latentSize,
                    linearBeta: linearBeta, experts: experts,
                    sharedExperts: sharedExperts, fullRankGate: fullRankGate
                ).data(using: .utf8)))
    }

    private static func alive(_ model: Module, seed: UInt64 = 83) -> Module {
        var updates: [(String, MLXArray)] = []
        for (index, (key, value)) in model.parameters().flattened().enumerated() {
            eval(value)
            if abs(value.asType(.float32)).max().item(Float.self) == 0 {
                updates.append(
                    (
                        key,
                        MLXRandom.normal(value.shape, key: MLXRandom.key(seed + UInt64(index)))
                            * 0.2
                    ))
            }
        }
        model.update(parameters: ModuleParameters.unflattened(updates))
        return model
    }

    /// `kda_layers` is one-based, so the schedule `[1, 3]` makes layers 0 and 2 recurrent.
    /// Read as zero-based it would pick the wrong two layers and still run.
    func testKdaLayersAreOneBased() throws {
        let text = try Self.config().textConfiguration
        XCTAssertEqual(
            (0 ..< 4).map { text.isLinear(layerIndex: $0) }, [true, false, true, false])

        let model = KimiK3(try Self.config())
        let layers = model.languageModel.model.layers
        XCTAssertTrue(layers[0].attention is KimiK3Text.DeltaAttention)
        XCTAssertTrue(layers[1].attention is KimiK3Text.MLAAttention)
    }

    /// The activation refuses anything but `situ` and sigmoid routing, loudly, rather than
    /// quietly running the wrong nonlinearity.
    func testUnsupportedActivationsAreRejected() {
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                KimiK3Configuration.TextConfiguration.self,
                from: XCTUnwrap("{\"hidden_act\": \"silu\"}".data(using: .utf8))))
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                KimiK3Configuration.TextConfiguration.self,
                from: XCTUnwrap(
                    "{\"moe_router_activation_func\": \"softmax\"}".data(using: .utf8))))
    }

    /// `situ` bounds both of its arguments. The gate goes through
    /// `beta * tanh(gate / beta) * sigmoid(gate)`, so a large gate saturates at beta
    /// rather than growing; the value's own squash is optional.
    func testSituIsBoundedInBothArguments() {
        let large = MLXArray(converting: [40.0]).asType(.float32)
        let one = MLXArray(converting: [1.0]).asType(.float32)

        let saturated = KimiK3Text.situ(value: one, gate: large, beta: 2, linearBeta: nil)
        eval(saturated)
        // tanh(20) is 1 and sigmoid(40) is 1, so the gate contributes exactly beta.
        XCTAssertEqual(saturated.item(Float.self), 2, accuracy: 1e-4)

        // A negative gate is squashed toward zero from the other side.
        let negative = KimiK3Text.situ(
            value: one, gate: -large, beta: 2, linearBeta: nil)
        eval(negative)
        XCTAssertEqual(negative.item(Float.self), 0, accuracy: 1e-4)

        // The value's squash bounds it at linear_beta.
        let bounded = KimiK3Text.situ(
            value: MLXArray(converting: [100.0]).asType(.float32), gate: large,
            beta: 2, linearBeta: 3)
        eval(bounded)
        XCTAssertEqual(bounded.item(Float.self), 2 * 3, accuracy: 1e-3)
    }

    /// The value and the gate are not interchangeable, so the argument order matters --
    /// and `SwitchGLU` hands its combiner the two the other way round from mlx-vlm.
    func testSituIsNotSymmetric() {
        let a = MLXArray(converting: [1.5]).asType(.float32)
        let b = MLXArray(converting: [-0.5]).asType(.float32)
        let forward = KimiK3Text.situ(value: a, gate: b, beta: 2, linearBeta: 3)
        let reversed = KimiK3Text.situ(value: b, gate: a, beta: 2, linearBeta: 3)
        eval(forward, reversed)
        XCTAssertGreaterThan(
            abs(forward - reversed).item(Float.self), 1e-3,
            "situ is symmetric, so this test cannot catch a swapped pair")
    }

    /// The correction bias steers the selection but never the returned weights.
    func testCorrectionBiasSteersSelectionOnly() {
        let gates = MLXArray(converting: [0.0, 0.5, 1.0, 1.5]).reshaped(1, 4)
        let noBias = KimiK3Text.groupExpertSelect(
            gates: gates, bias: nil, topK: 2, groups: 1, topkGroups: 1,
            routedScalingFactor: 1, renormalize: false)
        eval(noBias.indices, noBias.scores)
        XCTAssertEqual(Set(noBias.indices.reshaped(-1).asArray(Int32.self)), [2, 3])

        // A bias that lifts expert 0 above the rest changes which experts win.
        let bias = MLXArray(converting: [10.0, 0, 0, 0])
        let biased = KimiK3Text.groupExpertSelect(
            gates: gates, bias: bias, topK: 2, groups: 1, topkGroups: 1,
            routedScalingFactor: 1, renormalize: false)
        eval(biased.indices, biased.scores)
        XCTAssertTrue(biased.indices.reshaped(-1).asArray(Int32.self).contains(0))

        // Expert 0's returned weight is still its unbiased sigmoid, not the biased score.
        let position = biased.indices.reshaped(-1).asArray(Int32.self).firstIndex(of: 0)!
        XCTAssertEqual(
            biased.scores.reshaped(-1).asArray(Float.self)[position], 0.5, accuracy: 1e-5)
    }

    /// Group routing drops whole groups by their two best experts' sum, so a group with
    /// one strong expert can lose to a group with two middling ones.
    func testGroupRoutingScoresByTheTopTwo() {
        // Group 0 has one very strong expert; group 1 has two moderately strong ones.
        let gates = MLXArray(converting: [8.0, -8, 2, 2]).reshaped(1, 4)
        let (indices, _) = KimiK3Text.groupExpertSelect(
            gates: gates, bias: nil, topK: 2, groups: 2, topkGroups: 1,
            routedScalingFactor: 1, renormalize: false)
        eval(indices)
        // sigmoid(8) + sigmoid(-8) is about 1.0; sigmoid(2) + sigmoid(2) is about 1.76.
        XCTAssertEqual(Set(indices.reshaped(-1).asArray(Int32.self)), [2, 3])
    }

    /// A full-rank gate replaces the two-step one, and the checkpoint keys follow.
    func testFullRankGateReplacesTheTwoStepOne() throws {
        let keys = Set(
            KimiK3(try Self.config(fullRankGate: true)).parameters().flattened()
                .map(\.0))
        XCTAssertTrue(keys.contains("language_model.model.layers.0.self_attn.g_proj.weight"))
        XCTAssertFalse(
            keys.contains("language_model.model.layers.0.self_attn.g_a_proj.weight"))
    }

    /// A latent routed size wraps the switched experts in a narrower space, with an
    /// optional norm inside it.
    func testLatentRoutingAddsItsProjections() throws {
        let plain = Set(KimiK3(try Self.config()).parameters().flattened().map(\.0))
        XCTAssertFalse(
            plain.contains("language_model.model.layers.1.mlp.routed_expert_down_proj.weight"))

        let latent = Set(
            KimiK3(try Self.config(latentSize: "20")).parameters().flattened().map(\.0))
        XCTAssertTrue(
            latent.contains("language_model.model.layers.1.mlp.routed_expert_down_proj.weight"))
        XCTAssertTrue(
            latent.contains("language_model.model.layers.1.mlp.routed_expert_up_proj.weight"))
        XCTAssertFalse(
            latent.contains("language_model.model.layers.1.mlp.routed_expert_norm.weight"))
    }

    /// The block-residual scheme is off unless a checkpoint asks for it, and asking for it
    /// adds one projection and one norm per layer plus a final pair.
    func testResidualBlocksAreOptional() throws {
        let off = Set(KimiK3(try Self.config()).parameters().flattened().map(\.0))
        XCTAssertFalse(off.contains { $0.contains("res_proj") })

        let on = Set(
            KimiK3(try Self.config(residualBlockSize: "2")).parameters().flattened()
                .map(\.0))
        XCTAssertTrue(
            on.contains("language_model.model.layers.0.self_attention_res_proj.weight"))
        XCTAssertTrue(on.contains("language_model.model.layers.0.mlp_res_norm.weight"))
        XCTAssertTrue(on.contains("language_model.model.output_attn_res_proj.weight"))
    }

    /// With no saved residual the mixture is the identity, and with one it is a convex
    /// combination -- never an unbounded sum.
    func testResidualMixtureIsConvex() {
        let blocks = KimiK3Text.ResidualBlocks(eps: 1e-5)
        let partial = MLXArray.ones([1, 2, 4])
        let weight = MLXArray.ones([4]).asType(.float32)

        let untouched = KimiK3Text.mixResiduals(
            blocks: blocks, partialSum: partial, weight: weight, eps: 1e-5)
        eval(untouched)
        XCTAssertEqual(
            abs(untouched - partial).max().item(Float.self), 0, accuracy: 1e-7,
            "an empty block list has to leave the partial sum alone")

        blocks.append(MLXArray.ones([1, 2, 4]) * 3)
        let mixed = KimiK3Text.mixResiduals(
            blocks: blocks, partialSum: partial, weight: weight, eps: 1e-5)
        eval(mixed)
        // Every element is between the two candidates.
        XCTAssertGreaterThan(mixed.min().item(Float.self), 0.99)
        XCTAssertLessThan(mixed.max().item(Float.self), 3.01)
    }

    /// The routed experts arrive one per key under `block_sparse_moe` and are stacked for
    /// the switched projection; the latent attention's fused `kv_b_proj` is split in two.
    func testSanitizeStacksExpertsAndSplitsTheLatentProjection() throws {
        let config = try Self.config()
        let model = KimiK3(config)
        let text = config.textConfiguration
        var weights: [String: MLXArray] = [:]

        for expert in 0 ..< 4 {
            weights["language_model.model.layers.1.block_sparse_moe.experts.\(expert).w1.weight"] =
                MLXArray.ones([24, 32]) * Float(expert + 1)
        }
        weights["language_model.model.layers.1.block_sparse_moe.gate.weight"] =
            MLXArray.zeros([4, 32])
        weights["language_model.model.layers.1.block_sparse_moe.e_score_correction_bias"] =
            MLXArray.zeros([4])

        let headDim = (text.qkNopeHeadDim ?? 0) + (text.vHeadDim ?? 0)
        weights["language_model.model.layers.1.self_attn.kv_b_proj.weight"] =
            MLXArray.zeros([text.attentionHeads * headDim, text.kvLoraRank ?? 0])

        weights["language_model.model.layers.0.self_attn.q_proj.weight"] =
            MLXArray.zeros([128, 32])
        weights["language_model.model.layers.0.self_attn.k_proj.weight"] =
            MLXArray.zeros([128, 32])
        weights["language_model.model.layers.0.self_attn.v_proj.weight"] =
            MLXArray.zeros([128, 32])
        weights["language_model.model.layers.0.self_attn.q_conv1d.weight"] =
            MLXArray.zeros([128, 1, 4])
        weights["language_model.model.layers.0.self_attn.k_conv1d.weight"] =
            MLXArray.zeros([128, 1, 4])
        weights["language_model.model.layers.0.self_attn.v_conv1d.weight"] =
            MLXArray.zeros([128, 1, 4])
        weights["language_model.model.mtp.layers.0.weight"] = MLXArray.zeros([4, 4])

        let out = model.sanitize(weights: weights)

        let stacked = try XCTUnwrap(
            out["language_model.model.layers.1.mlp.switch_mlp.gate_proj.weight"])
        eval(stacked)
        XCTAssertEqual(stacked.shape, [4, 24, 32])
        XCTAssertEqual(stacked[3, 0, 0].item(Float.self), 4)
        XCTAssertNotNil(out["language_model.model.layers.1.mlp.gate.weight"])
        XCTAssertNotNil(out["language_model.model.layers.1.mlp.e_score_correction_bias"])

        // The fused latent projection becomes two, transposed the way each side wants.
        XCTAssertNil(out["language_model.model.layers.1.self_attn.kv_b_proj.weight"])
        XCTAssertEqual(
            out["language_model.model.layers.1.self_attn.embed_q.weight"]?.shape,
            [text.attentionHeads, text.kvLoraRank ?? 0, text.qkNopeHeadDim ?? 0])
        XCTAssertEqual(
            out["language_model.model.layers.1.self_attn.unembed_out.weight"]?.shape,
            [text.attentionHeads, text.vHeadDim ?? 0, text.kvLoraRank ?? 0])

        // The three linear projections and the three convolutions fuse.
        XCTAssertEqual(
            out["language_model.model.layers.0.self_attn.qkv_proj.weight"]?.shape,
            [384, 32])
        XCTAssertEqual(
            out["language_model.model.layers.0.self_attn.qkv_conv.conv.weight"]?.shape,
            [384, 4, 1])

        // The draft head is not part of this model.
        XCTAssertNil(out["language_model.model.mtp.layers.0.weight"])
    }

    /// The residual projection and norm are stored as bare parameters under one name.
    func testSanitizeRenamesTheResidualParameters() throws {
        let model = KimiK3(try Self.config(residualBlockSize: "2"))
        let out = model.sanitize(weights: [
            "language_model.model.layers.0.self_attention_res.proj_weight":
                MLXArray.zeros([1, 32]),
            "language_model.model.layers.0.mlp_res.norm_weight": MLXArray.zeros([32]),
            "language_model.model.output_attn_res.proj_weight": MLXArray.zeros([1, 32]),
        ])
        XCTAssertNotNil(
            out["language_model.model.layers.0.self_attention_res_proj.weight"])
        XCTAssertNotNil(out["language_model.model.layers.0.mlp_res_norm.weight"])
        XCTAssertNotNil(out["language_model.model.output_attn_res_proj.weight"])
    }

    /// The tower averages the frame axis away and groups each merge block, and the
    /// projector flattens the group.
    func testVisionTowerAndProjectorShapes() throws {
        let config = try Self.config()
        let vision = config.visionConfiguration
        let model = KimiK3(config)
        _ = Self.alive(model)

        // A 4x4 patch grid, one frame: 16 patches merge to 4 groups of 4.
        let patches = MLXRandom.normal(
            [16, vision.patchSize, vision.patchSize, 3], key: MLXRandom.key(5))
        let features = model.visionTower(patches, grid: [[1, 4, 4]])
        let projected = model.projector(features)
        eval(features, projected)

        XCTAssertEqual(features.shape, [4, vision.mergeUnit, vision.hiddenSize])
        XCTAssertEqual(projected.shape, [4, config.textConfiguration.hiddenSize])
    }

    /// Two frames average into one grid rather than producing twice the rows.
    func testFramesArePooledAway() throws {
        let config = try Self.config()
        let vision = config.visionConfiguration
        let model = KimiK3(config)
        _ = Self.alive(model)

        let patches = MLXRandom.normal(
            [32, vision.patchSize, vision.patchSize, 3], key: MLXRandom.key(6))
        let features = model.visionTower(patches, grid: [[2, 4, 4]])
        eval(features)
        XCTAssertEqual(features.shape, [4, vision.mergeUnit, vision.hiddenSize])
    }

    /// A chunked prefill has to reproduce the flat forward through both mixer kinds.
    func testChunkedPrefillMatchesTheFlatForward() throws {
        let model = KimiK3(try Self.config())
        _ = Self.alive(model)
        let tokens = MLXArray([3, 9, 15, 21, 27, 33, 39, 45, 51]).reshaped(1, 9)

        let flat = model(tokens, cache: nil)
        let cache = try model.newCache(configuration: nil)
        _ = model(tokens[0..., ..<5], cache: cache)
        let tail = model(tokens[0..., 5...], cache: cache)
        eval(flat, tail)

        let scale = abs(flat).max().item(Float.self)
        XCTAssertGreaterThan(scale, 1e-3)
        XCTAssertLessThan(
            abs(flat[0, 8] - tail[0, 3]).max().item(Float.self), 0.02 * scale,
            "the chunk boundary does not reproduce the flat forward")
    }

    /// The same, with the block-residual mixture switched on.
    func testChunkedPrefillMatchesWithResidualBlocks() throws {
        let model = KimiK3(try Self.config(residualBlockSize: "2"))
        _ = Self.alive(model)
        let tokens = MLXArray([3, 9, 15, 21, 27, 33, 39, 45, 51]).reshaped(1, 9)

        let flat = model(tokens, cache: nil)
        let cache = try model.newCache(configuration: nil)
        _ = model(tokens[0..., ..<5], cache: cache)
        let tail = model(tokens[0..., 5...], cache: cache)
        eval(flat, tail)

        let scale = abs(flat).max().item(Float.self)
        XCTAssertGreaterThan(scale, 1e-3)
        XCTAssertLessThan(
            abs(flat[0, 8] - tail[0, 3]).max().item(Float.self), 0.02 * scale,
            "the residual mixture does not survive a chunk boundary")
    }
}
