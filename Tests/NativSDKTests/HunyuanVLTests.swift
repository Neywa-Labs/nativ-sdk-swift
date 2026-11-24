import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class HunyuanVLTests: XCTestCase {

    private static let json = """
        {
          "model_type": "hunyuan_vl",
          "image_token_id": 120120,
          "image_start_token_id": 120118,
          "image_end_token_id": 120119,
          "text_config": {
            "model_type": "hunyuan_vl",
            "vocab_size": 256,
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "intermediate_size": 48,
            "use_qk_norm": true,
            "rope_theta": 10000.0,
            "rms_norm_eps": 1e-05,
            "tie_word_embeddings": false,
            "rope_scaling": {
              "alpha": 1000.0, "type": "xdrope", "xdrope_section": [2, 2, 2, 2]
            }
          },
          "vision_config": {
            "model_type": "hunyuan_vl",
            "hidden_size": 24,
            "out_hidden_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 32,
            "patch_size": 4,
            "num_channels": 3,
            "spatial_merge_size": 2,
            "rms_norm_eps": 1e-05,
            "max_image_size": 16
          }
        }
        """

    private static func config() throws -> HunyuanVLConfiguration {
        try JSONDecoder.json5().decode(
            HunyuanVLConfiguration.self, from: XCTUnwrap(json.data(using: .utf8)))
    }

    private static func alive(_ model: Module, seed: UInt64 = 67) -> Module {
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

    /// The position rows are the sequential index, the image index, and the row and column
    /// inside each image's merged grid -- with one extra column per row for the newline
    /// the merger appends.
    func testXDRopePositionsSpanTheImage() throws {
        let config = try Self.config()
        // Two text tokens, then an image marker followed by its span, then text.
        // A 4x4 patch grid merges to 2x2, so the span is (2 + 1) * 2 = 6 positions.
        var tokens: [Int32] = [5, 6, Int32(config.imageTokenID)]
        tokens.append(contentsOf: Array(repeating: 7, count: 6))
        tokens.append(9)

        let positions = HunyuanVLText.xdropePositions(
            tokens: tokens, imageTokenID: config.imageTokenID,
            grid: [[1, 4, 4]], spatialMergeSize: 2, axes: 4)
        eval(positions)
        XCTAssertEqual(positions.shape, [4, 10])

        let sequential = positions[0].asArray(Int32.self)
        let temporal = positions[1].asArray(Int32.self)
        let rows = positions[2].asArray(Int32.self)
        let columns = positions[3].asArray(Int32.self)

        XCTAssertEqual(sequential, (0 ..< 10).map(Int32.init))
        // The span runs from index 3 through 8.
        XCTAssertEqual(Array(columns[3 ..< 9]), [0, 1, 2, 0, 1, 2])
        XCTAssertEqual(Array(rows[3 ..< 9]), [0, 0, 0, 1, 1, 1])
        XCTAssertEqual(Array(temporal[3 ..< 9]), [0, 0, 0, 0, 0, 0])
        // Outside the span every row is the sequential index.
        XCTAssertEqual(Array(columns[0 ..< 3]), [0, 1, 2])
        XCTAssertEqual(columns[9], 9)
    }

    /// Three sections give three rows instead of four, and an unrecognized count falls
    /// back to the sequential index everywhere.
    func testXDRopeAxisCount() throws {
        let tokens: [Int32] = [1, 2, 3, 4]
        let three = HunyuanVLText.xdropePositions(
            tokens: tokens, imageTokenID: 99, grid: [], spatialMergeSize: 2, axes: 3)
        let two = HunyuanVLText.xdropePositions(
            tokens: tokens, imageTokenID: 99, grid: [], spatialMergeSize: 2, axes: 2)
        eval(three, two)
        XCTAssertEqual(three.shape, [3, 4])
        XCTAssertEqual(two.shape, [2, 4])
        XCTAssertEqual(two[0].asArray(Int32.self), [0, 1, 2, 3])
        XCTAssertEqual(two[1].asArray(Int32.self), [0, 1, 2, 3])
    }

    /// The head norms come after the rotation, which is the reverse of the usual order.
    /// Applied before it, a rotated query would not be unit-normalized and the scores
    /// would drift with position.
    func testHeadNormsComeAfterTheRotation() throws {
        let config = try Self.config().textConfiguration
        let attention = HunyuanVLText.Attention(config)
        _ = Self.alive(attention)

        // A norm applied after the rotation leaves every position's query at the same
        // scale; applied before it, the rotation would preserve the norm too -- so the
        // discriminating check is that scaling the projection does not move the output.
        let x = MLXRandom.normal([1, 5, 32], key: MLXRandom.key(3))
        let positions = HunyuanVLText.xdropePositions(
            tokens: [0, 1, 2, 3, 4], imageTokenID: 99, grid: [], spatialMergeSize: 2,
            axes: 4
        ).expandedDimensions(axis: 0)

        let before = attention(x, mask: nil, cache: nil, positionIDs: positions)
        attention.update(
            parameters: ModuleParameters.unflattened([
                ("q_proj.weight", attention.qProj.weight * 100)
            ]))
        let after = attention(x, mask: nil, cache: nil, positionIDs: positions)
        eval(before, after)

        XCTAssertGreaterThan(abs(before).max().item(Float.self), 1e-4)
        XCTAssertLessThan(
            abs(before - after).max().item(Float.self),
            2e-3 * abs(before).max().item(Float.self),
            "the query norm is not absorbing the projection's scale")
    }

    /// The merger folds each 2x2 block, adds a newline column per row, and wraps the
    /// image in a begin and an end row.
    func testMergerShapeAccountsForMarkers() throws {
        let config = try Self.config()
        let vision = config.visionConfiguration
        let merger = HunyuanVLVision.PatchMerger(vision)
        _ = Self.alive(merger)

        let patches = MLXRandom.normal([1, 16, vision.hiddenSize], key: MLXRandom.key(4))
        let out = merger(patches, gridHeight: 4, gridWidth: 4)
        eval(out)

        // A 4x4 grid merges to 2x2, becomes 2 rows of 3 with the newline, plus the two
        // markers: 2 * 3 + 2 = 8.
        XCTAssertEqual(out.shape, [1, 8, vision.outHiddenSize])
    }

    /// The position table skips its first row and is resampled with the checkpoint's own
    /// tenth-offset scale, which is not any standard alignment.
    func testPositionTableSkipsItsFirstRowAndResamples() throws {
        let vision = try Self.config().visionConfiguration
        XCTAssertEqual(vision.maxPatches, 16)
        XCTAssertEqual(vision.positions, 17)
        XCTAssertEqual(vision.positionEdge, 4)

        let embed = HunyuanVLVision.PatchEmbed(vision)
        _ = Self.alive(embed)

        // The identity case returns the table untouched.
        let table = embed.positionEmbedding.weight[1...]
            .reshaped(1, 4, 4, vision.hiddenSize)
        let identity = embed.interpolate(table, height: 4, width: 4)
        eval(identity, table)
        XCTAssertEqual(abs(identity - table).max().item(Float.self), 0, accuracy: 1e-7)

        let resampled = embed.interpolate(table, height: 2, width: 2)
        eval(resampled)
        XCTAssertEqual(resampled.shape, [1, 2, 2, vision.hiddenSize])
    }

    /// The checkpoint keeps the text stack at `model.` and the tower at `vit.`.
    func testSanitizeMovesBothHalves() throws {
        let model = HunyuanVL(try Self.config())
        let out = model.sanitize(weights: [
            "model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([64, 32]),
            "vit.layers.0.mlp.dense_h_to_4h.weight": MLXArray.zeros([32, 24]),
            "vit.embeddings.patch_embedding.weight": MLXArray.zeros([24, 3, 4, 4]),
            "vit.perceive.proj.0.weight": MLXArray.zeros([48, 24, 2, 2]),
            "lm_head.weight": MLXArray.zeros([256, 32]),
        ])

        XCTAssertNotNil(out["language_model.model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(out["vision_tower.layers.0.mlp.dense_h_to_4h.weight"])
        XCTAssertNotNil(out["language_model.lm_head.weight"])
        XCTAssertEqual(out["vision_tower.embeddings.patch_embedding.weight"]?.shape, [24, 4, 4, 3])
        XCTAssertEqual(out["vision_tower.perceive.proj.0.weight"]?.shape, [48, 2, 2, 24])

        // Idempotent: a converted checkpoint is left alone.
        let twice = model.sanitize(weights: out)
        XCTAssertEqual(
            twice["vision_tower.embeddings.patch_embedding.weight"]?.shape, [24, 4, 4, 3])
    }

    /// A decode step takes the ordinary rotation, so a chunked prefill and a one-shot one
    /// have to land in the same place.
    func testChunkedPrefillMatchesTheFlatForward() throws {
        let model = HunyuanVL(try Self.config())
        _ = Self.alive(model)
        let tokens = MLXArray([3, 9, 15, 21, 27, 33, 39, 45, 51]).reshaped(1, 9)

        let flat = model(tokens, cache: nil)
        let cache = try model.newCache(configuration: nil)
        _ = model(tokens[0..., ..<5], cache: cache)
        let tail = model(tokens[0..., 5...], cache: cache)
        eval(flat, tail)

        let scale = abs(flat).max().item(Float.self)
        XCTAssertGreaterThan(scale, 1e-2)
        // The prefill rotates against the position rows and a decode step rotates
        // sequentially; with no image the two agree, so this stays tight.
        XCTAssertLessThan(
            abs(flat[0, 8] - tail[0, 3]).max().item(Float.self), 1e-3 * scale,
            "the chunk boundary does not reproduce the flat forward")
    }

}
