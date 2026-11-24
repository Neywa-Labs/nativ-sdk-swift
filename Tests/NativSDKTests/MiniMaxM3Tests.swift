import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class MiniMaxM3Tests: XCTestCase {

    private static func json(
        layers: Int = 4,
        layerTypes: String = "null",
        mlpLayerTypes: String = "null",
        sharedIntermediate: Int = 24,
        packShared: String = "null",
        useRoutingBias: Bool = true,
        featureLayer: String = "-1",
        selectStrategy: String = "full"
    ) -> String {
        """
        {
          "model_type": "minimax_m3_vl",
          "image_token_index": 90,
          "video_token_index": 91,
          "projector_hidden_size": 40,
          "projector_hidden_act": "gelu",
          "vision_feature_layer": \(featureLayer),
          "vision_feature_select_strategy": "\(selectStrategy)",
          "text_config": {
            "model_type": "minimax_m3",
            "hidden_size": 32,
            "intermediate_size": 24,
            "dense_intermediate_size": 48,
            "shared_intermediate_size": \(sharedIntermediate),
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "num_hidden_layers": \(layers),
            "rms_norm_eps": 1e-06,
            "rope_theta": 5000000.0,
            "partial_rotary_factor": 0.5,
            "vocab_size": 128,
            "hidden_act": "swigluoai",
            "swiglu_alpha": 1.702,
            "swiglu_beta": 1.0,
            "swiglu_limit": 7.0,
            "use_qk_norm": true,
            "use_gemma_norm": true,
            "num_local_experts": 4,
            "num_experts_per_tok": 2,
            "n_shared_experts": 1,
            "use_routing_bias": \(useRoutingBias),
            "routed_scaling_factor": 2.0,
            "mlp_layer_types": \(mlpLayerTypes),
            "layer_types": \(layerTypes),
            "index_n_heads": 2,
            "index_head_dim": 16,
            "index_block_size": 2,
            "index_topk_blocks": 2,
            "index_local_blocks": 1,
            "pack_shared_expert": \(packShared),
            "tie_word_embeddings": false
          },
          "vision_config": {
            "model_type": "clip_vision_model",
            "hidden_size": 24,
            "intermediate_size": 32,
            "num_attention_heads": 4,
            "num_hidden_layers": 2,
            "patch_size": 2,
            "num_channels": 3,
            "layer_norm_eps": 1e-05,
            "vision_segment_max_frames": 2,
            "img_token_compression_config": {
              "spatial_merge_size": 2, "temporal_patch_size": 2
            }
          }
        }
        """
    }

    private static func config(
        layers: Int = 4,
        layerTypes: String = "null",
        mlpLayerTypes: String = "null",
        sharedIntermediate: Int = 24,
        packShared: String = "null",
        useRoutingBias: Bool = true,
        featureLayer: String = "-1",
        selectStrategy: String = "full"
    ) throws -> MiniMaxM3Configuration {
        try JSONDecoder.json5().decode(
            MiniMaxM3Configuration.self,
            from: XCTUnwrap(
                json(
                    layers: layers, layerTypes: layerTypes,
                    mlpLayerTypes: mlpLayerTypes,
                    sharedIntermediate: sharedIntermediate, packShared: packShared,
                    useRoutingBias: useRoutingBias, featureLayer: featureLayer,
                    selectStrategy: selectStrategy
                ).data(using: .utf8)))
    }

    private static func alive(_ model: Module, seed: UInt64 = 97) -> Module {
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

    /// The flat `index_*` fields fill the sparse settings when there is no nested object.
    func testFlatIndexFieldsFillTheSparseSettings() throws {
        let text = try Self.config().textConfiguration
        XCTAssertEqual(text.indexHeads, 2)
        XCTAssertEqual(text.indexHeadDim, 16)
        XCTAssertEqual(text.indexBlockSize, 2)
        XCTAssertEqual(text.indexTopKBlocks, 2)
        XCTAssertEqual(text.indexLocalBlocks, 1)
        XCTAssertEqual(text.indexInitialBlocks, 0)

        // Absent everywhere, the documented defaults apply.
        let bare = try JSONDecoder.json5().decode(
            MiniMaxM3TextConfiguration.self,
            from: XCTUnwrap("{\"hidden_size\": 64}".data(using: .utf8)))
        XCTAssertEqual(bare.indexHeadDim, 128)
        XCTAssertEqual(bare.indexHeads, 4)
        XCTAssertEqual(bare.indexTopKBlocks, 16)
        XCTAssertEqual(bare.indexBlockSize, 128)
        XCTAssertEqual(bare.indexLocalBlocks, 1)
    }

    /// The activation refuses anything else, loudly.
    func testUnsupportedActivationsAreRejected() {
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                MiniMaxM3TextConfiguration.self,
                from: XCTUnwrap("{\"hidden_act\": \"silu\"}".data(using: .utf8))))
        XCTAssertThrowsError(
            try JSONDecoder.json5().decode(
                MiniMaxM3TextConfiguration.self,
                from: XCTUnwrap("{\"scoring_func\": \"softmax\"}".data(using: .utf8))))
    }

    /// The clamped SwiGLU bounds the gate above only, the linear half on both sides, and
    /// carries a `+ beta` offset the ordinary form has no room for.
    func testClampedSwiGLUBoundsAndOffset() {
        let big = MLXArray(converting: [100.0]).asType(.float32)
        let small = MLXArray(converting: [-100.0]).asType(.float32)

        // A large gate saturates at the limit, so the factor is limit * sigmoid(alpha *
        // limit), which is essentially the limit itself.
        let saturated = MiniMaxM3Text.swiGLUOAI(
            linear: MLXArray(converting: [0.0]).asType(.float32), gate: big,
            alpha: 1.702, limit: 7, beta: 1)
        eval(saturated)
        XCTAssertEqual(saturated.item(Float.self), 7 * 1.0, accuracy: 1e-3)

        // A very negative linear half clamps at -limit, and the +beta offset moves it.
        let clamped = MiniMaxM3Text.swiGLUOAI(
            linear: small, gate: big, alpha: 1.702, limit: 7, beta: 1)
        eval(clamped)
        XCTAssertEqual(clamped.item(Float.self), 7 * (-7 + 1), accuracy: 1e-2)

        // Beta is not a no-op: a zero linear half still contributes.
        let withoutBeta = MiniMaxM3Text.swiGLUOAI(
            linear: MLXArray(converting: [0.0]).asType(.float32), gate: big,
            alpha: 1.702, limit: 7, beta: 0)
        eval(withoutBeta)
        XCTAssertEqual(withoutBeta.item(Float.self), 0, accuracy: 1e-5)
    }

    /// The Gemma convention stores the norm's weight as an offset from one and starts it
    /// at zero; the other convention stores the scale and starts at one.
    func testNormConventionChangesWeightAndInitialValue() {
        let gemma = MiniMaxM3Text.RMSNorm(dimensions: 4, eps: 1e-6, gemma: true)
        let plain = MiniMaxM3Text.RMSNorm(dimensions: 4, eps: 1e-6, gemma: false)
        eval(gemma.weight, plain.weight)
        XCTAssertEqual(abs(gemma.weight).max().item(Float.self), 0)
        XCTAssertEqual(plain.weight.min().item(Float.self), 1)

        // Both start as the identity scale, so a fresh norm is the same function.
        let x = MLXArray(converting: [1.0, 2.0, 3.0, 4.0]).reshaped(1, 4)
        let a = gemma(x)
        let b = plain(x)
        eval(a, b)
        XCTAssertEqual(abs(a - b).max().item(Float.self), 0, accuracy: 1e-6)
    }

    /// Blocks are forced in by locality and by position, and the packed selection sorts
    /// with invalid slots as -1 so a consumer can stop at the first negative.
    func testBlockSelectionForcesTheLocalBlockAndPacksInvalidSlots() {
        let selector = MiniMaxM3Text.BlockSelector(
            blockSize: 2, topKBlocks: 2, initialBlocks: 1, localBlocks: 1, scale: 1)

        // Six keys, three blocks. Query 1 can only see block 0.
        let queries = MLXArray.zeros([1, 1, 2, 4])
        let keys = MLXArray.zeros([1, 1, 6, 4])
        let positions = MLXArray([Int32(1), 5]).reshaped(1, 2)

        let selection = selector.select(
            queries: queries, keys: keys, queryPositions: positions)
        eval(selection)
        XCTAssertEqual(selection.shape, [1, 2, 2])

        // Position 1 lives in block 0, which is both the initial and the local block, so
        // only block 0 is valid; the packing sorts the valid slot first and leaves -1.
        XCTAssertEqual(selection[0, 0].asArray(Int32.self), [0, -1])
        // Position 5 lives in block 2; the initial block 0 and the local block 2 win.
        XCTAssertEqual(Set(selection[0, 1].asArray(Int32.self)), [0, 2])
    }

    /// The selected blocks expand back into a causal per-key mask.
    func testBlockMaskExpandsToKeys() {
        let selector = MiniMaxM3Text.BlockSelector(
            blockSize: 2, topKBlocks: 2, initialBlocks: 0, localBlocks: 0, scale: 1)
        let indices = MLXArray([Int32(0), 2]).reshaped(1, 1, 2)
        let positions = MLXArray([Int32(5)]).reshaped(1, 1)

        let mask = selector.blockMask(
            indices: indices, keyLength: 6, queryPositions: positions)
        eval(mask)
        XCTAssertEqual(mask.shape, [1, 1, 1, 6])
        // Blocks 0 and 2 are keys 0-1 and 4-5.
        XCTAssertEqual(
            mask.reshaped(-1).asType(.int32).asArray(Int32.self),
            [1, 1, 0, 0, 1, 1])
    }

    /// The index cache keeps the indexer's keys on their own timeline, beside the
    /// ordinary ones, and both grow as the sequence does.
    func testIndexCacheTracksItsOwnKeys() {
        let cache = MiniMaxM3Text.IndexKVCache()
        let first = cache.updateIndexAndFetch(MLXArray.ones([1, 1, 3, 4]))
        eval(first)
        XCTAssertEqual(first.shape, [1, 1, 3, 4])
        XCTAssertEqual(cache.indexOffset, 3)

        let second = cache.updateIndexAndFetch(MLXArray.ones([1, 1, 2, 4]) * 2)
        eval(second)
        XCTAssertEqual(second.shape, [1, 1, 5, 4])
        XCTAssertEqual(second[0, 0, 4, 0].item(Float.self), 2)
        XCTAssertEqual(second[0, 0, 0, 0].item(Float.self), 1)

        // The ordinary keys are separate and start empty.
        XCTAssertEqual(cache.offset, 0)
    }

    /// When the shared expert's width matches the bank's it rides along as one more
    /// expert; when it does not, it stays a separate module.
    func testSharedExpertPackingFollowsTheWidths() throws {
        let packed = try Self.config().textConfiguration
        XCTAssertTrue(packed.packsSharedExpert)
        let packedKeys = Set(
            MiniMaxM3(try Self.config()).parameters().flattened().map(\.0))
        XCTAssertTrue(
            packedKeys.contains(
                "language_model.model.layers.3.block_sparse_moe.switch_mlp.gate_up_proj.weight"))
        XCTAssertFalse(packedKeys.contains { $0.contains("block_sparse_moe.shared_experts") })

        let split = try Self.config(sharedIntermediate: 32).textConfiguration
        XCTAssertFalse(split.packsSharedExpert)
        let splitKeys = Set(
            MiniMaxM3(try Self.config(sharedIntermediate: 32)).parameters().flattened()
                .map(\.0))
        XCTAssertTrue(
            splitKeys.contains(
                "language_model.model.layers.3.block_sparse_moe.switch_mlp.gate_proj.weight"))
        XCTAssertTrue(
            splitKeys.contains(
                "language_model.model.layers.3.block_sparse_moe.shared_experts.gate_proj.weight"))

        // And the flag can force the split even when the widths agree.
        let forced = try Self.config(packShared: "false").textConfiguration
        XCTAssertFalse(forced.packsSharedExpert)
    }

    /// The container nests both halves one level deeper than the modules, and a
    /// `weight_scale_inv` marks its sibling as a packed uint8 payload.
    func testSanitizeMovesTheContainerAndPacksScaledWeights() throws {
        let model = MiniMaxM3(try Self.config(layers: 6))
        let payload = MLXArray(
            (0 ..< 8).map { UInt8($0) }, [2, 4])
        let out = model.sanitize(weights: [
            "model.language_model.layers.0.self_attn.q_proj.weight":
                MLXArray.zeros([64, 32]),
            "model.vision_tower.vision_model.pre_layrnorm.weight": MLXArray.zeros([24]),
            "model.multi_modal_projector.linear_1.weight": MLXArray.zeros([40, 24]),
            "model.patch_merge_mlp.linear_1.weight": MLXArray.zeros([32, 128]),
            "language_model.model.layers.1.mlp.gate_proj.weight": payload,
            "language_model.model.layers.1.mlp.gate_proj.weight_scale_inv":
                MLXArray.ones([2, 1]),
        ])

        XCTAssertNotNil(out["language_model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(out["vision_tower.vision_model.pre_layrnorm.weight"])
        XCTAssertNotNil(out["multi_modal_projector.linear_1.weight"])
        XCTAssertNotNil(out["patch_merge_mlp.linear_1.weight"])

        // Four uint8 values fold into one uint32 word, and the scale is renamed.
        let packed = try XCTUnwrap(out["language_model.model.layers.1.mlp.gate_proj.weight"])
        eval(packed)
        XCTAssertEqual(packed.shape, [2, 1])
        XCTAssertEqual(packed.dtype, .uint32)
        XCTAssertEqual(packed[0, 0].item(UInt32.self), 0 | (1 << 8) | (2 << 16) | (3 << 24))
        XCTAssertNotNil(out["language_model.model.layers.1.mlp.gate_proj.scales"])
        XCTAssertNil(out["language_model.model.layers.1.mlp.gate_proj.weight_scale_inv"])
    }

    /// The routed experts arrive one per key and fuse with the shared expert into one
    /// bank of `experts + 1`.
    func testSanitizeFusesExpertsWithTheSharedOne() throws {
        let config = try Self.config(layers: 6)
        let model = MiniMaxM3(config)
        let prefix = "language_model.model.layers.3.block_sparse_moe"
        var weights: [String: MLXArray] = [:]
        for expert in 0 ..< 4 {
            weights["\(prefix).experts.\(expert).w1.weight"] =
                MLXArray.ones([24, 32]) * Float(expert + 1)
            weights["\(prefix).experts.\(expert).w3.weight"] =
                MLXArray.ones([24, 32]) * Float(10 + expert)
            weights["\(prefix).experts.\(expert).w2.weight"] =
                MLXArray.ones([32, 24]) * Float(expert + 1)
        }
        weights["\(prefix).shared_experts.gate_proj.weight"] = MLXArray.ones([24, 32]) * 100
        weights["\(prefix).shared_experts.up_proj.weight"] = MLXArray.ones([24, 32]) * 200
        weights["\(prefix).shared_experts.down_proj.weight"] = MLXArray.ones([32, 24]) * 300

        let out = model.sanitize(weights: weights)
        let gateUp = try XCTUnwrap(out["\(prefix).switch_mlp.gate_up_proj.weight"])
        let down = try XCTUnwrap(out["\(prefix).switch_mlp.down_proj.weight"])
        eval(gateUp, down)

        // Five experts, each with the gate stacked above the up projection.
        XCTAssertEqual(gateUp.shape, [5, 48, 32])
        XCTAssertEqual(down.shape, [5, 32, 24])
        XCTAssertEqual(gateUp[0, 0, 0].item(Float.self), 1)
        XCTAssertEqual(gateUp[0, 24, 0].item(Float.self), 10)
        XCTAssertEqual(gateUp[4, 0, 0].item(Float.self), 100)
        XCTAssertEqual(gateUp[4, 24, 0].item(Float.self), 200)
        XCTAssertEqual(down[4, 0, 0].item(Float.self), 300)
        XCTAssertNil(out["\(prefix).experts.0.w1.weight"])
        XCTAssertNil(out["\(prefix).shared_experts.gate_proj.weight"])
    }

    /// The tower's rotary table covers three axes and is doubled, and the passthrough
    /// channels above it are left unrotated.
    func testVisionRotaryCoversThreeAxesAndDoubles() throws {
        let config = try Self.config()
        let tower = MiniMaxM3Vision.Transformer(config.visionConfiguration)
        let angles = tower.rotaryAngles([[1, 2, 2]])
        eval(angles)

        // head_dim 6, rope_dims 6, axis_dim 2 * ((6/3)/2) = 2 -> one frequency per axis,
        // three axes, doubled.
        let axisDim = 2 * ((2 * (config.visionConfiguration.headDim / 2) / 3) / 2)
        XCTAssertEqual(angles.dim(-1), 3 * (axisDim / 2) * 2)
        XCTAssertEqual(angles.dim(0), 4)
    }

    /// A video longer than the segment limit is cut, and each segment attends alone.
    func testVideoIsCutIntoSegments() throws {
        let config = try Self.config()
        let tower = MiniMaxM3Vision.Transformer(config.visionConfiguration)
        XCTAssertEqual(config.visionConfiguration.segmentMaxFrames, 2)
        XCTAssertEqual(tower.segments([[5, 2, 2]]), [[2, 2, 2], [2, 2, 2], [1, 2, 2]])
        XCTAssertEqual(tower.segments([[2, 2, 2]]), [[2, 2, 2]])
    }

    /// The spatial merge folds each block's rows into one, so an image's token count
    /// drops by the square of the merge size.
    func testSpatialMergeReducesTheTokenCount() throws {
        let config = try Self.config()
        let model = MiniMaxM3(config)
        _ = Self.alive(model)

        let vision = config.visionConfiguration
        // A 4x4 patch grid, one frame: 16 patches merge to 4 tokens.
        let patches = MLXRandom.normal([16, vision.patchDim], key: MLXRandom.key(4))
        let features = model.visualFeatures(patches, grid: [[1, 4, 4]])
        eval(features)
        XCTAssertEqual(features.shape, [4, config.textConfiguration.hiddenSize])
    }

    /// A chunked prefill has to reproduce the flat forward through the dense layers, the
    /// routed ones and the block-sparse index.
    func testChunkedPrefillMatchesTheFlatForward() throws {
        let model = MiniMaxM3(try Self.config(layers: 6))
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
            abs(flat[0, 8] - tail[0, 3]).max().item(Float.self), 1e-3 * scale,
            "the chunk boundary does not reproduce the flat forward")
    }

}
