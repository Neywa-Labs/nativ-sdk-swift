import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class Glm5NextTests: XCTestCase {

    private static func json(
        layers: Int = 4,
        layerTypes: String =
            "[\"linear_attention\", \"full_attention\", \"linear_attention\", \"full_attention\"]",
        mlpLayerTypes: String = "[\"dense\", \"sparse\", \"sparse\", \"sparse\"]",
        indexTopK: Int = 8,
        linearAttention: String =
            "{\"num_heads\": 4, \"head_dim\": 32, \"short_conv_kernel_size\": 4}"
    ) -> String {
        """
        {
          "model_type": "glm5_next",
          "image_token_id": 90,
          "video_token_id": 91,
          "text_config": {
            "model_type": "glm5_next",
            "vocab_size": 96,
            "hidden_size": 32,
            "intermediate_size": 48,
            "moe_intermediate_size": 24,
            "num_hidden_layers": \(layers),
            "num_attention_heads": 4,
            "num_key_value_heads": 4,
            "n_shared_experts": 1,
            "n_routed_experts": 4,
            "num_experts_per_tok": 2,
            "routed_scaling_factor": 1.0,
            "first_k_dense_replace": 1,
            "kv_lora_rank": 16,
            "q_lora_rank": 24,
            "qk_rope_head_dim": 0,
            "v_head_dim": 16,
            "qk_nope_head_dim": 16,
            "rms_norm_eps": 1e-06,
            "index_topk": \(indexTopK),
            "index_head_dim": 32,
            "index_n_heads": 2,
            "index_kpool": 4,
            "layer_types": \(layerTypes),
            "mlp_layer_types": \(mlpLayerTypes),
            "linear_attn_config": \(linearAttention),
            "hc_mult": 2,
            "hc_sinkhorn_iters": 3,
            "mla_use_nope": true,
            "tie_word_embeddings": false
          },
          "vision_config": {
            "model_type": "glm5_next_vision",
            "depth": 2,
            "hidden_size": 32,
            "intermediate_size": 40,
            "num_heads": 4,
            "patch_size": 4,
            "out_hidden_size": 32,
            "projection_intermediate_size": 48,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "in_channels": 3,
            "rms_norm_eps": 1e-05
          }
        }
        """
    }

    private static func config(
        layers: Int = 4,
        layerTypes: String =
            "[\"linear_attention\", \"full_attention\", \"linear_attention\", \"full_attention\"]",
        mlpLayerTypes: String = "[\"dense\", \"sparse\", \"sparse\", \"sparse\"]",
        indexTopK: Int = 8,
        linearAttention: String =
            "{\"num_heads\": 4, \"head_dim\": 32, \"short_conv_kernel_size\": 4}"
    ) throws -> Glm5NextConfiguration {
        try JSONDecoder.json5().decode(
            Glm5NextConfiguration.self,
            from: XCTUnwrap(
                json(
                    layers: layers, layerTypes: layerTypes, mlpLayerTypes: mlpLayerTypes,
                    indexTopK: indexTopK, linearAttention: linearAttention
                ).data(using: .utf8)))
    }

    private static func alive(_ model: Module, seed: UInt64 = 43) -> Module {
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

    /// `layer_types` chooses the mixer and `mlp_layer_types` the feed-forward, and the
    /// first dense layers are dense whatever the mlp schedule says.
    func testSchedulesSelectMixersAndFeedForwards() throws {
        let text = try Self.config().textConfiguration
        XCTAssertEqual(
            (0 ..< 4).map { text.isLinear(layerIndex: $0) }, [true, false, true, false])
        XCTAssertEqual(
            (0 ..< 4).map { text.isSparseMLP(layerIndex: $0) },
            [false, true, true, true])

        let model = Glm5NextLanguageModel(text)
        XCTAssertTrue(model.model.layers[0].attention is Glm5NextLinearAttention)
        XCTAssertTrue(model.model.layers[1].attention is Glm5NextSparseAttention)
        XCTAssertTrue(model.model.layers[0].mlp is DeepseekV32MLP)
        XCTAssertTrue(model.model.layers[1].mlp is DeepseekV32MoE)
    }

    /// The residual stream carries `hc_mult` copies through every layer and is averaged
    /// once at the end. A stream count of one would make the hyper-connections a no-op and
    /// hide any mistake in them.
    func testResidualStreamIsExpandedAndAveraged() throws {
        let config = try Self.config()
        let model = Glm5NextLanguageModel(config.textConfiguration)
        _ = Self.alive(model)

        XCTAssertEqual(model.model.hcMultiplier, 2)
        let out = model(MLXArray([1, 2, 3, 4, 5]).reshaped(1, 5), cache: nil)
        eval(out)
        XCTAssertEqual(out.shape, [1, 5, 96])
        XCTAssertGreaterThan(abs(out).max().item(Float.self), 1e-4)
    }

    /// The checkpoint names the hyper-connection blocks by role, stores the linear
    /// attention's three convolutions separately, and keeps the forget gate's parameters
    /// directly on `self_attn`.
    func testSanitizeFusesConvolutionsAndRenamesBlocks() throws {
        let config = try Self.config()
        let model = Glm5NextLanguageModel(config.textConfiguration)
        let width =
            config.textConfiguration.linearHeads
            * config.textConfiguration.linearHeadDim

        let out = model.sanitize(weights: [
            "model.layers.0.hc_attn_fn": MLXArray.zeros([8, 64]),
            "model.layers.0.hc_ffn_base": MLXArray.zeros([8]),
            "model.layers.0.self_attn.q_conv1d.weight": MLXArray.ones([width, 4, 1]),
            "model.layers.0.self_attn.k_conv1d.weight": MLXArray.ones([width, 4, 1]) * 2,
            "model.layers.0.self_attn.v_conv1d.weight": MLXArray.ones([width, 4, 1]) * 3,
            "model.layers.0.self_attn.A_log": MLXArray.zeros([4]),
            "model.layers.0.self_attn.dt_bias": MLXArray.zeros([width]),
            "model.layers.0.self_attn.f_b_proj.weight": MLXArray.zeros([width, 32]),
            "model.layers.9.self_attn.q_proj.weight": MLXArray.zeros([4, 4]),
            "mtp.layers.0.weight": MLXArray.zeros([4, 4]),
        ])

        XCTAssertNotNil(out["model.layers.0.attn_hc.fn"])
        XCTAssertNotNil(out["model.layers.0.ffn_hc.base"])
        XCTAssertNil(out["model.layers.0.hc_attn_fn"])

        // The three convolutions fuse in q, k, v order.
        let fused = try XCTUnwrap(out["model.layers.0.self_attn.conv1d.weight"])
        eval(fused)
        XCTAssertEqual(fused.shape, [3 * width, 4, 1])
        XCTAssertEqual(fused[0, 0, 0].item(Float.self), 1)
        XCTAssertEqual(fused[width, 0, 0].item(Float.self), 2)
        XCTAssertEqual(fused[2 * width, 0, 0].item(Float.self), 3)

        XCTAssertNotNil(out["model.layers.0.self_attn.forget_gate.A_log"])
        XCTAssertNotNil(out["model.layers.0.self_attn.forget_gate.dt_bias"])
        XCTAssertNotNil(out["model.layers.0.self_attn.forget_gate.f_b_proj.weight"])
        XCTAssertNil(out["model.layers.0.self_attn.A_log"])

        // A layer past the end of the stack, and the draft head, belong to the drafter.
        XCTAssertNil(out["model.layers.9.self_attn.q_proj.weight"])
        XCTAssertNil(out["mtp.layers.0.weight"])
    }

    /// A checkpoint whose experts are stored one per key gets them stacked for the
    /// switched projection, which is what DeepSeek V3.2's own sanitize does.
    func testSanitizeStacksExperts() throws {
        let config = try Self.config()
        let model = Glm5NextLanguageModel(config.textConfiguration)
        var weights: [String: MLXArray] = [:]
        for expert in 0 ..< 4 {
            weights["model.layers.1.mlp.experts.\(expert).gate_proj.weight"] =
                MLXArray.ones([24, 32]) * Float(expert + 1)
        }
        let out = model.sanitize(weights: weights)

        let stacked = try XCTUnwrap(out["model.layers.1.mlp.switch_mlp.gate_proj.weight"])
        eval(stacked)
        XCTAssertEqual(stacked.shape, [4, 24, 32])
        XCTAssertEqual(stacked[3, 0, 0].item(Float.self), 4)
        XCTAssertNil(out["model.layers.1.mlp.experts.0.gate_proj.weight"])
    }

    /// Nothing is selected while the cache still fits inside the budget: the attention has
    /// to stay dense rather than mask everything away.
    func testIndexerStaysQuietBelowItsBudget() throws {
        let config = try Self.config(indexTopK: 64)
        let model = Glm5NextLanguageModel(config.textConfiguration)
        _ = Self.alive(model)

        let indexer = try XCTUnwrap(
            (model.model.layers[1].attention as? Glm5NextSparseAttention)?.indexer)
        let x = MLXRandom.normal([1, 6, 32], key: MLXRandom.key(3))
        let queryRank = MLXRandom.normal([1, 6, 24], key: MLXRandom.key(4))

        XCTAssertNil(
            indexer(x, queryRank: queryRank, mask: nil, cache: KVCacheSimple()),
            "the indexer selected while everything already fit")
    }

    /// Past the budget it selects whole pools, plus the partial pool at the end of the
    /// visible range, and marks everything else with -1.
    func testIndexerSelectsWholePoolsAndTheTail() throws {
        let config = try Self.config(indexTopK: 8)
        let model = Glm5NextLanguageModel(config.textConfiguration)
        _ = Self.alive(model)

        let indexer = try XCTUnwrap(
            (model.model.layers[1].attention as? Glm5NextSparseAttention)?.indexer)
        let length = 16
        let x = MLXRandom.normal([1, length, 32], key: MLXRandom.key(5))
        let queryRank = MLXRandom.normal([1, length, 24], key: MLXRandom.key(6))

        let selection = try XCTUnwrap(
            indexer(x, queryRank: queryRank, mask: nil, cache: KVCacheSimple()))
        eval(selection)

        // index_topk plus the kpool - 1 tail slots.
        XCTAssertEqual(selection.shape, [1, 1, length, 8 + 3])
        XCTAssertEqual(selection.dtype, .int32)

        // Position 0 can only see itself, so every slot but its own is empty.
        let first = selection[0, 0, 0].asArray(Int32.self)
        XCTAssertTrue(first.allSatisfy { $0 == -1 || $0 == 0 }, "\(first)")

        // A later position selects only positions it can see.
        let last = selection[0, 0, length - 1].asArray(Int32.self)
        XCTAssertTrue(last.contains { $0 >= 0 }, "nothing was selected")
        XCTAssertTrue(last.allSatisfy { $0 < Int32(length) }, "\(last)")
    }

    /// A chunked prefill has to reproduce the flat forward through both mixer kinds, the
    /// hyper-connections and the indexer's cache.
    func testChunkedPrefillMatchesTheFlatForward() throws {
        let model = Glm5NextLanguageModel(try Self.config().textConfiguration)
        _ = Self.alive(model)
        let tokens = MLXArray([3, 9, 15, 21, 27, 33, 39, 45, 51]).reshaped(1, 9)

        let flat = model(tokens, cache: nil)

        let cache = try model.newCache(configuration: nil)
        _ = model(tokens[0..., ..<5], cache: cache)
        let tail = model(tokens[0..., 5...], cache: cache)
        eval(flat, tail)

        let scale = abs(flat).max().item(Float.self)
        XCTAssertGreaterThan(scale, 1e-2)
        XCTAssertLessThan(
            abs(flat[0, 8] - tail[0, 3]).max().item(Float.self), 1e-5 * scale,
            "the chunk boundary does not reproduce the flat forward")
    }

    /// The same, in the regime where the indexer selects on every call rather than
    /// bypassing the short first chunk.
    func testChunkedPrefillMatchesWhenTheIndexerAlwaysSelects() throws {
        let model = Glm5NextLanguageModel(
            try Self.config(indexTopK: 4).textConfiguration)
        _ = Self.alive(model)
        let tokens = MLXArray([3, 9, 15, 21, 27, 33, 39, 45, 51]).reshaped(1, 9)

        let flat = model(tokens, cache: nil)
        let cache = try model.newCache(configuration: nil)
        _ = model(tokens[0..., ..<5], cache: cache)
        let tail = model(tokens[0..., 5...], cache: cache)
        eval(flat, tail)

        let scale = abs(flat).max().item(Float.self)
        XCTAssertGreaterThan(scale, 1e-2)
        XCTAssertLessThan(
            abs(flat[0, 8] - tail[0, 3]).max().item(Float.self), 1e-5 * scale,
            "the selection differs between a chunked and a one-shot prefill")
    }

    /// The partial pool at the end of a query's visible range is appended separately,
    /// because it can never be a complete pool and so would never be selected -- and it
    /// is the most recent context.
    ///
    /// Its width is `visible_count mod kpool`, which needs integer division: on a float
    /// division every tail comes out empty and the model silently loses its own recent
    /// tokens while still looking like it selects.
    func testVisibleTailIsTheIncompletePool() throws {
        let indexer = Glm5NextIndexer(try Self.config().textConfiguration)
        let positions = MLXArray(0 ..< 9)
        let visible = positions[.newAxis, .newAxis, 0...] .<= positions[.newAxis, 0..., .newAxis]
        let tail = indexer.visibleTail(
            visible: visible, valid: MLXArray.ones([1, 9], dtype: .bool))
        eval(tail)

        XCTAssertEqual(tail.shape, [1, 9, 3])
        XCTAssertEqual(tail[0, 0].asArray(Int32.self), [0, -1, -1])
        XCTAssertEqual(tail[0, 1].asArray(Int32.self), [0, 1, -1])
        XCTAssertEqual(tail[0, 2].asArray(Int32.self), [0, 1, 2])
        // A complete pool needs no tail.
        XCTAssertEqual(tail[0, 3].asArray(Int32.self), [-1, -1, -1])
        XCTAssertEqual(tail[0, 4].asArray(Int32.self), [4, -1, -1])
        XCTAssertEqual(tail[0, 6].asArray(Int32.self), [4, 5, 6])
        XCTAssertEqual(tail[0, 7].asArray(Int32.self), [-1, -1, -1])
        XCTAssertEqual(tail[0, 8].asArray(Int32.self), [8, -1, -1])
    }

    /// The vision tower's rotary ids are laid out by position *within* each merge block,
    /// not in raster order, which is what the merged downsample expects.
    func testVisionRotaryIdsFollowTheMergeBlocks() throws {
        let config = try Self.config()
        let tower = Glm5NextVisionModel(config.visionConfiguration)
        _ = Self.alive(tower)

        let (cosine, sine) = tower.rotaryPositionEmbeddings(grid: [[1, 4, 4]])
        eval(cosine, sine)
        XCTAssertEqual(cosine.shape[0], 16)
        XCTAssertEqual(sine.shape[0], 16)

        // A 2x2 merge over a 4x4 grid gives four blocks of four; the first block holds
        // rows 0 and 1 against columns 0 and 1, so its ids repeat before advancing.
        XCTAssertEqual(
            abs(cosine[0] - cosine[1]).max().item(Float.self) > 0, true,
            "the first two ids are identical, so the layout is not per-block")
    }

    /// Patch features land at the image token positions, and video tokens stand in when no
    /// image token is present.
    func testFeaturesLandAtImageOrVideoTokens() throws {
        let config = try Self.config()
        let model = Glm5Next(config)
        _ = Self.alive(model)

        let tokens = MLXArray([
            Int32(1), Int32(config.imageTokenID), Int32(config.imageTokenID), Int32(2),
        ]).reshaped(1, 4)
        let features = MLXRandom.normal([2, 32], key: MLXRandom.key(9))
        let embedded = MLXRandom.normal([1, 4, 32], key: MLXRandom.key(10))

        let merged = model.merged(
            embedded: embedded, features: features, inputIDs: tokens)
        eval(merged)

        XCTAssertEqual(
            abs(merged[0, 0] - embedded[0, 0]).max().item(Float.self), 0, accuracy: 1e-6)
        XCTAssertEqual(
            abs(merged[0, 1] - features[0]).max().item(Float.self), 0, accuracy: 1e-6)
        XCTAssertEqual(
            abs(merged[0, 2] - features[1]).max().item(Float.self), 0, accuracy: 1e-6)
        XCTAssertEqual(
            abs(merged[0, 3] - embedded[0, 3]).max().item(Float.self), 0, accuracy: 1e-6)

        let videoTokens = MLXArray([
            Int32(1), Int32(config.videoTokenID), Int32(config.videoTokenID), Int32(2),
        ]).reshaped(1, 4)
        let fromVideo = model.merged(
            embedded: embedded, features: features, inputIDs: videoTokens)
        eval(fromVideo)
        XCTAssertEqual(
            abs(fromVideo[0, 1] - features[0]).max().item(Float.self), 0, accuracy: 1e-6)
    }

    /// The checkpoint's container prefixes split into the two halves, and only the
    /// language half's keys go through the language sanitize.
    func testSanitizeSplitsTheContainer() throws {
        let model = Glm5Next(try Self.config())
        let out = model.sanitize(weights: [
            "model.visual.blocks.0.norm1.weight": MLXArray.zeros([32]),
            "model.visual.patch_embed.proj.weight": MLXArray.zeros([32, 3, 2, 4, 4]),
            "model.language_model.layers.0.hc_attn_fn": MLXArray.zeros([8, 64]),
            "lm_head.weight": MLXArray.zeros([96, 32]),
        ])

        XCTAssertNotNil(out["vision_model.blocks.0.norm1.weight"])
        XCTAssertNotNil(out["language_model.model.layers.0.attn_hc.fn"])
        XCTAssertNotNil(out["language_model.lm_head.weight"])
        XCTAssertEqual(
            out["vision_model.patch_embed.proj.weight"]?.shape, [32, 2, 4, 4, 3])
    }
}
