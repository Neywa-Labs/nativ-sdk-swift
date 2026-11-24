import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

/// MiMo-V2.6 reuses MiMo-V2-Flash's decoder stack and differs in how its checkpoint is
/// stored: the text hyperparameters sit at the top level, attention projections are fused
/// into one tensor-parallel `qkv_proj`, and two audio fields ship as strings.
final class MiMoV2Tests: XCTestCase {
    private static func json(
        layers: Int = 4, valueScale: Float? = nil, speechVocabulary: String = "\"16\"",
        projectionLayers: Int = 2, skipVision: Bool = false, depth: Int = 4,
        windowTypes: String = "[0, 1, 1, 0]", nested: Bool = false
    ) -> String {
        let textFields = """
            "vocab_size": 128, "hidden_size": 64, "intermediate_size": 128,
            "moe_intermediate_size": 32, "num_hidden_layers": \(layers),
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "v_head_dim": 16,
            "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2,
            "swa_head_dim": 16, "swa_v_head_dim": 16,
            "hybrid_layer_pattern": [0, 1, 0, 1], "moe_layer_freq": [0, 1, 0, 1],
            "n_routed_experts": 4, "num_experts_per_tok": 2,
            "routed_scaling_factor": 2.5, "sliding_window_size": 8,
            "max_position_embeddings": 256, "layernorm_epsilon": 1e-6,
            "rope_theta": 1000000.0, "swa_rope_theta": 10000.0,
            "partial_rotary_factor": 0.5
            \(valueScale.map { ", \"attention_value_scale\": \($0)" } ?? "")
            """
        return """
            {
              "model_type": "mimo_v2", "image_token_id": 7, "audio_token_id": 9,
              "skip_vision": \(skipVision),
              \(nested ? "\"text_config\": {\(textFields)}," : "\(textFields),")
              "vision_config": {
                "depth": \(depth), "hidden_size": 32, "intermediate_size": 64,
                "out_hidden_size": 64, "num_heads": 4, "num_key_value_heads": 2,
                "qk_channels": 8, "patch_size": 2, "temporal_patch_size": 2,
                "in_chans": 3, "spatial_merge_size": 2,
                "fullatt_block_indexes": [0, 2], "visual_token_window_size": 4,
                "vit_window_attn_types": \(windowTypes)
              },
              "audio_config": {
                "out_hidden_size": 64, "audio_channels": 2, "group_size": 2,
                "input_local_dim": 32, "input_local_attn_heads": 4,
                "input_local_intermediate_size": 64, "input_local_layers": 2,
                "speech_vocab_size": \(speechVocabulary), "speech_zeroemb_idx": "12",
                "projection_layers": \(projectionLayers)
              }
            }
            """
    }

    private func configuration(_ json: String = MiMoV2Tests.json()) throws -> MiMoV2Configuration {
        let configuration = try JSONDecoder().decode(
            MiMoV2Configuration.self, from: Data(json.utf8))
        try configuration.validateModelConfiguration()
        return configuration
    }

    // MARK: Configuration

    /// The text hyperparameters sit at the top level, and a nested text_config still wins.
    func testTextConfigurationIsReadFlatAndNested() throws {
        let flat = try configuration()
        XCTAssertEqual(flat.textConfiguration.hiddenSize, 64)
        XCTAssertEqual(flat.textConfiguration.hiddenLayers, 4)
        XCTAssertEqual(flat.textConfiguration.hybridLayerPattern, [0, 1, 0, 1])

        let nested = try configuration(MiMoV2Tests.json(nested: true))
        XCTAssertEqual(nested.textConfiguration.hiddenSize, 64)
        XCTAssertEqual(nested.textConfiguration.hybridLayerPattern, [0, 1, 0, 1])
    }

    /// Released configs ship these as strings. Read as anything else the defaults stand,
    /// and an undersized speech table indexes out of bounds, which MLX answers with zeros.
    func testStringValuedAudioFieldsAreRead() throws {
        let audio = try configuration().audioConfiguration
        XCTAssertEqual(audio.speechVocabularySize, 16)
        XCTAssertEqual(audio.speechZeroEmbeddingIndex, 12)

        let numeric = try configuration(
            MiMoV2Tests.json(speechVocabulary: "20"))
        XCTAssertEqual(numeric.audioConfiguration.speechVocabularySize, 20)
    }

    /// `in_chans` is the released spelling; the window-attention pattern defaults to one
    /// entry per block so the forward can index it.
    func testVisionAliasAndDerivedWindowTypes() throws {
        let vision = try configuration().visionConfiguration
        XCTAssertEqual(vision.inChannels, 3)
        XCTAssertEqual(vision.windowAttentionTypes, [0, 1, 1, 0])

        let bare = try JSONDecoder().decode(
            MiMoV2VisionConfiguration.self, from: Data("{\"depth\": 3}".utf8))
        XCTAssertEqual(bare.windowAttentionTypes, [-1, -1, -1])
        XCTAssertEqual(bare.inChannels, 3)
        let aliased = try JSONDecoder().decode(
            MiMoV2VisionConfiguration.self, from: Data("{\"in_chans\": 5}".utf8))
        XCTAssertEqual(aliased.inChannels, 5)
        let named = try JSONDecoder().decode(
            MiMoV2VisionConfiguration.self, from: Data("{\"in_channels\": 6}".utf8))
        XCTAssertEqual(named.inChannels, 6)
    }

    /// The decoder stack indexes both per-layer patterns, and V2.6's text config defaults
    /// them to empty rather than deriving them.
    func testInvalidConfigurationsAreRefused() throws {
        XCTAssertThrowsError(
            try configuration(MiMoV2Tests.json(layers: 5)))
        XCTAssertThrowsError(
            try configuration(MiMoV2Tests.json(projectionLayers: 3)))
    }

    // MARK: Vision

    func testVisionTowerShapeAndSinkPlacement() throws {
        let config = try configuration()
        let tower = MiMoV2Vision.VisionModel(config.visionConfiguration)
        let keys = Set(tower.parameters().flattened().map(\.0))
        for block in [0, 2] {
            XCTAssertFalse(keys.contains("blocks.\(block).attn.sinks"), "block \(block)")
        }
        for block in [1, 3] {
            XCTAssertTrue(keys.contains("blocks.\(block).attn.sinks"), "block \(block)")
        }

        let patches = MLXArray.ones([16, 3 * 2 * 2 * 2])
        let out = tower(patches, gridTHW: [(1, 4, 4)])
        XCTAssertEqual(out.shape, [4, 64])
    }

    /// The patch convolution is a linear map, so a 5D checkpoint weight is folded flat.
    func testPatchEmbedWeightIsFlattened() throws {
        let tower = MiMoV2Vision.VisionModel(try configuration().visionConfiguration)
        let out = tower.sanitize(weights: [
            "patch_embed.proj.weight": MLXArray.ones([32, 3, 2, 2, 2])
        ])
        XCTAssertEqual(out["patch_embed.proj.weight"]?.shape, [32, 24])
    }

    /// The column ordering and its inverse must compose to the identity, or the blocks
    /// that reorder would not hand back what the next row-major block expects.
    func testWindowOrderingRoundTrips() throws {
        let tower = MiMoV2Vision.VisionModel(try configuration().visionConfiguration)
        let index = tower.windowIndex([(1, 4, 4)])
        let reverse = argSort(index, axis: -1)
        let x = MLXArray(0 ..< 16).reshaped(16, 1).asType(.float32)
        let restored = tower.applyIndex(tower.applyIndex(x, index), reverse)
        XCTAssertEqual(restored.flattened().asArray(Float.self), (0 ..< 16).map(Float.init))
    }

    // MARK: Audio

    /// A frame count that is not a whole number of groups is padded by repeating the last
    /// frame, so the group axis always divides.
    func testAudioEncoderGroupsAndPads() throws {
        let config = try configuration()
        let encoder = MiMoV2Audio.Encoder(config.audioConfiguration)
        let embeddings = MiMoV2Audio.speechEmbeddings(config.audioConfiguration)
        XCTAssertEqual(embeddings.count, 2)

        XCTAssertEqual(
            encoder(
                MLXArray([Int32(1), 2, 3, 4, 5, 6, 7, 8]).reshaped(4, 2),
                speechEmbeddings: embeddings
            ).shape, [2, 64])
        XCTAssertEqual(
            encoder(
                MLXArray([Int32(1), 2, 3, 4, 5, 6]).reshaped(3, 2),
                speechEmbeddings: embeddings
            ).shape, [2, 64])
    }

    // MARK: Weights

    /// The fused tensor is the tensor-parallel shards concatenated, each holding its own
    /// q heads then k then v. Walking it as one tensor misaligns every shard after the
    /// first, so the split has to know the degree.
    func testFusedQKVIsSplitPerShard() throws {
        let model = MiMoV2(try configuration())
        let heads = model.qkvHeads(layer: 0)
        XCTAssertEqual(heads.heads, 4)

        let qRows = heads.heads * heads.headDim
        let kRows = heads.kvHeads * heads.headDim
        let vRows = heads.kvHeads * heads.vHeadDim
        let fused = MLXArray(0 ..< Int32(qRows + kRows + vRows))
            .reshaped(qRows + kRows + vRows, 1)
            .asType(.float32)
        let out = try model.unfuseQKV([
            "model.layers.0.self_attn.qkv_proj.weight": fused,
            "model.layers.0.mlp.gate_proj.weight": MLXArray.ones([4, 4]),
        ])
        XCTAssertNil(out["model.layers.0.self_attn.qkv_proj.weight"])
        XCTAssertNotNil(out["model.layers.0.mlp.gate_proj.weight"])
        XCTAssertEqual(out["model.layers.0.self_attn.q_proj.weight"]?.shape, [qRows, 1])
        XCTAssertEqual(out["model.layers.0.self_attn.k_proj.weight"]?.shape, [kRows, 1])
        XCTAssertEqual(out["model.layers.0.self_attn.v_proj.weight"]?.shape, [vRows, 1])
        XCTAssertEqual(
            out["model.layers.0.self_attn.k_proj.weight"]?.flattened().asArray(Float.self),
            (qRows ..< (qRows + kRows)).map(Float.init))
    }

    /// A layer whose fused rows are a whole number of 128-row blocks admits several
    /// degrees; a full attention layer, where the per-shard padding shows up in the grid,
    /// is what pins it down.
    func testTensorParallelDegreeIsIntersected() throws {
        let ambiguous = MiMoV2.candidateDegrees(
            rows: 13568, heads: 64, kvHeads: 4, headDim: 192, vHeadDim: 128, gridRows: 106)
        XCTAssertTrue(ambiguous.contains(1))
        let pinned = MiMoV2.candidateDegrees(
            rows: 13568, heads: 64, kvHeads: 4, headDim: 192, vHeadDim: 128, gridRows: 108)
        XCTAssertEqual(pinned, [4])
        XCTAssertTrue(
            MiMoV2.candidateDegrees(
                rows: 13568, heads: 64, kvHeads: 4, headDim: 192, vHeadDim: 128,
                gridRows: 999
            ).isEmpty)
    }

    func testUnexplainedFusedShapeIsRefused() throws {
        let model = MiMoV2(try configuration())
        XCTAssertThrowsError(
            try model.unfuseQKV([
                "model.layers.0.self_attn.qkv_proj.weight": MLXArray.ones([7, 1]),
                "model.layers.0.self_attn.qkv_proj.weight_scale_inv": MLXArray.ones([3, 1]),
            ]))
    }

    /// A released checkpoint keeps the tower under `visual.` and the audio encoder at the
    /// top level; a converted one is already prefixed and is passed through.
    func testSanitizeRoutesEachHalf() throws {
        let model = MiMoV2(try configuration())
        let out = model.sanitize(weights: [
            "visual.merger.ln_q.weight": MLXArray.ones([32]),
            "audio_encoder.projection.mlp.0.weight": MLXArray.ones([4, 4]),
            "speech_embeddings.0.weight": MLXArray.ones([16, 32]),
            "model.norm.weight": MLXArray.ones([64]),
        ])
        XCTAssertNotNil(out["vision_tower.merger.ln_q.weight"])
        XCTAssertNotNil(out["audio_encoder.projection.mlp.0.weight"])
        XCTAssertNotNil(out["speech_embeddings.0.weight"])
        XCTAssertNotNil(out["language_model.model.norm.weight"])

        let converted = model.sanitize(weights: [
            "language_model.model.norm.weight": MLXArray.ones([64])
        ])
        XCTAssertEqual(converted.count, 1)
    }

    /// The value scale multiplies the values after `v_proj` and before the cache, which
    /// is exactly equivalent to scaling that projection's weight. Dropped, or applied
    /// anywhere else, the two models disagree.
    func testValueScaleIsEquivalentToScalingVProj() throws {
        let scale: Float = 0.5
        let scaled = MiMoV2FlashModel(
            try configuration(MiMoV2Tests.json(valueScale: scale)).textConfiguration
                .flashConfiguration)
        var plainConfiguration = try configuration().textConfiguration.flashConfiguration
        plainConfiguration.attentionValueScale = nil
        let plain = MiMoV2FlashModel(plainConfiguration)

        var weights = Dictionary(uniqueKeysWithValues: scaled.parameters().flattened())
        for (key, value) in weights where key.hasSuffix("v_proj.weight") {
            weights[key] = value * scale
        }
        plain.update(parameters: ModuleParameters.unflattened(weights.map { ($0, $1) }))

        let tokens = MLXArray([Int32(5), 1, 9, 2]).reshaped(1, 4)
        XCTAssertTrue(
            allClose(plain(tokens, cache: nil), scaled(tokens, cache: nil), atol: 1e-4).item(),
            "the value scale is not applied to the values")
    }

    /// The column ordering walks each frame's merge-unit blocks column-major, so a 4x4
    /// grid of 2x2 blocks visits them in this order and not in the identity order.
    func testWindowIndexIsColumnMajor() throws {
        let tower = MiMoV2Vision.VisionModel(try configuration().visionConfiguration)
        XCTAssertEqual(
            tower.windowIndex([(1, 4, 4)]).asArray(Int32.self), [0, 2, 1, 3])
        XCTAssertEqual(
            tower.windowIndex([(2, 4, 4)]).asArray(Int32.self), [0, 2, 1, 3, 4, 6, 5, 7])
    }

    /// The sink is an additive bias on the first key only, and is constant across
    /// queries, so it lands in column 0 and nowhere else.
    func testSinkBiasOccupiesTheFirstKeyOnly() throws {
        let config = try configuration()
        let attention = MiMoV2Vision.Attention(
            config.visionConfiguration, useSinks: true, windowSize: 0)
        attention.update(
            parameters: ModuleParameters.unflattened([
                ("sinks", MLXArray([Float(1), 2, 3, 4]))
            ]))
        let mask = try XCTUnwrap(attention.mask(3, fullAttention: true, dtype: .float32))
        XCTAssertEqual(mask.shape, [1, 4, 1, 3])
        XCTAssertEqual(mask[0, 0..., 0, 0].asArray(Float.self), [1, 2, 3, 4])
        XCTAssertEqual(mask[0, 0..., 0, 1].asArray(Float.self), [0, 0, 0, 0])

        let without = MiMoV2Vision.Attention(
            config.visionConfiguration, useSinks: false, windowSize: 0)
        XCTAssertNil(without.mask(3, fullAttention: true, dtype: .float32))
    }

    /// A short final group is filled by repeating the last frame, not with zeros or the
    /// zero-embedding index.
    func testShortGroupRepeatsTheLastFrame() throws {
        let config = try configuration()
        let encoder = MiMoV2Audio.Encoder(config.audioConfiguration)
        let embeddings = MiMoV2Audio.speechEmbeddings(config.audioConfiguration)

        let short = MLXArray([Int32(1), 2, 3, 4, 5, 6]).reshaped(3, 2)
        let explicit = MLXArray([Int32(1), 2, 3, 4, 5, 6, 5, 6]).reshaped(4, 2)
        XCTAssertTrue(
            allClose(
                encoder(short, speechEmbeddings: embeddings),
                encoder(explicit, speechEmbeddings: embeddings), atol: 1e-5
            ).item(),
            "the short group was not padded by repeating the last frame")
    }

    /// Each tensor-parallel shard carries its own scale grid. Read as one grid the
    /// second shard is scaled by the first shard's factors.
    func testShardedFusedQKVUsesEachShardsScales() throws {
        let model = MiMoV2(try configuration())
        let ones = MLXArray([UInt8](repeating: 0x38, count: 128 * 128)).reshaped(128, 128)
        let out = try model.unfuseQKV([
            "model.layers.0.self_attn.qkv_proj.weight": ones,
            "model.layers.0.self_attn.qkv_proj.weight_scale_inv":
                MLXArray([Float(1), 4]).reshaped(2, 1),
        ])
        XCTAssertEqual(
            try model.tensorParallelDegree([
                "model.layers.0.self_attn.qkv_proj.weight": ones,
                "model.layers.0.self_attn.qkv_proj.weight_scale_inv":
                    MLXArray([Float(1), 4]).reshaped(2, 1),
            ]), 2)

        let q = try XCTUnwrap(out["model.layers.0.self_attn.q_proj.weight"])
        XCTAssertEqual(q.shape, [64, 128])
        XCTAssertEqual(q[0, 0].item(Float.self), 1, accuracy: 1e-3)
        XCTAssertEqual(q[32, 0].item(Float.self), 4, accuracy: 1e-3)
        let k = try XCTUnwrap(out["model.layers.0.self_attn.k_proj.weight"])
        XCTAssertEqual(k.shape, [32, 128])
        XCTAssertEqual(k[0, 0].item(Float.self), 1, accuracy: 1e-3)
        XCTAssertEqual(k[16, 0].item(Float.self), 4, accuracy: 1e-3)
    }

    /// The wrapper merges vision features into the embeddings and then runs the backbone
    /// over them, so the backbone has to read them instead of the token table.
    func testBackboneHonoursSuppliedEmbeddings() throws {
        let model = MiMoV2FlashModel(
            try configuration().textConfiguration.flashConfiguration)
        let tokens = MLXArray([Int32(5), 1, 9, 2]).reshaped(1, 4)
        let zeros = MLXArray.zeros([1, 4, 64])
        XCTAssertFalse(
            allClose(
                model(tokens, cache: nil, inputEmbedding: zeros), model(tokens, cache: nil),
                atol: 1e-3
            ).item(),
            "the supplied embeddings were ignored")
    }

    // MARK: Forward

    /// The image placeholders are replaced one per merged vision token, and a mismatch is
    /// refused rather than written past the end.
    func testImageEmbeddingsReplacePlaceholders() throws {
        let model = MiMoV2(try configuration())
        let tokens = MLXArray([Int32(1), 7, 7, 7, 7, 2]).reshaped(1, 6)
        let pixels = MLXArray.ones([16, 24])
        let embeddings = try model.inputEmbeddings(
            inputIds: tokens, pixelValues: pixels, gridTHW: [(1, 4, 4)])
        XCTAssertEqual(embeddings.shape, [1, 6, 64])

        let tooFew = MLXArray([Int32(1), 7, 2]).reshaped(1, 3)
        XCTAssertThrowsError(
            try model.inputEmbeddings(
                inputIds: tooFew, pixelValues: pixels, gridTHW: [(1, 4, 4)]))
        XCTAssertThrowsError(
            try model.inputEmbeddings(inputIds: tokens, pixelValues: pixels, gridTHW: nil))
    }

    func testVisionCanBeDisabled() throws {
        let model = MiMoV2(try configuration(MiMoV2Tests.json(skipVision: true)))
        XCTAssertFalse(model.parameters().flattened().contains { $0.0.hasPrefix("vision_tower") })
        XCTAssertThrowsError(
            try model.inputEmbeddings(
                inputIds: MLXArray([Int32(1)]).reshaped(1, 1),
                pixelValues: MLXArray.ones([16, 24]), gridTHW: [(1, 4, 4)]))
    }
}
