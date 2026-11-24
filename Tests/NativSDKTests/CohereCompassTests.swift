import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class CohereCompassTests: XCTestCase {

    private static func json(
        normType: String = "layer_norm",
        blockType: String = "parallel",
        layerTypes: String = "null",
        slidingWindow: String = "null",
        ropeStyle: String = "split",
        ropeParameters: String = "null",
        logitScale: String = "null",
        vision: Bool = true
    ) -> String {
        """
        {
          "model_type": "cohere_compass",
          "image_token_id": 90,
          "vision_start_token_id": 88,
          "text_config": {
            "model_type": "cohere_compass_text",
            "vocab_size": 128,
            "hidden_size": 32,
            "intermediate_size": 48,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "norm_type": "\(normType)",
            "transformer_block_type": "\(blockType)",
            "layer_norm_eps": 1e-05,
            "rope_theta": 10000.0,
            "swa_rope_theta": 5000.0,
            "rope_style": "\(ropeStyle)",
            "rope_parameters": \(ropeParameters),
            "layer_types": \(layerTypes),
            "sliding_window": \(slidingWindow),
            "logit_scale": \(logitScale),
            "tie_word_embeddings": false
          }\(vision ? """
          ,
          "vision_config": {
            "model_type": "cohere_compass_vision",
            "depth": 4,
            "hidden_size": 32,
            "intermediate_size": 40,
            "num_heads": 4,
            "in_channels": 3,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [1, 2]
          }
          """ : "")
        }
        """
    }

    private static func config(
        normType: String = "layer_norm",
        blockType: String = "parallel",
        layerTypes: String = "null",
        slidingWindow: String = "null",
        ropeStyle: String = "split",
        ropeParameters: String = "null",
        logitScale: String = "null",
        vision: Bool = true
    ) throws -> CohereCompassConfiguration {
        try JSONDecoder.json5().decode(
            CohereCompassConfiguration.self,
            from: XCTUnwrap(
                json(
                    normType: normType, blockType: blockType, layerTypes: layerTypes,
                    slidingWindow: slidingWindow, ropeStyle: ropeStyle,
                    ropeParameters: ropeParameters, logitScale: logitScale,
                    vision: vision
                ).data(using: .utf8)))
    }

    private static func alive(_ model: Module, seed: UInt64 = 53) -> Module {
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

    /// `rope_style` decides which halves rotate into each other. Getting it wrong is
    /// silent: both forms consume the same table and produce the same shapes.
    func testRopeStyleSelectsTheRotation() {
        let x = MLXArray(converting: [1.0, 2.0, 3.0, 4.0]).reshaped(1, 1, 1, 4)
        let split = CohereCompassText.rotateHalfSplit(x)
        let interleaved = CohereCompassText.rotateHalfInterleaved(x)
        eval(split, interleaved)
        XCTAssertEqual(split.reshaped(4).asArray(Float.self), [-3, -4, 1, 2])
        XCTAssertEqual(interleaved.reshaped(4).asArray(Float.self), [-2, 1, -4, 3])
    }

    /// The sliding layers rotate at their own base, so a checkpoint with two thetas has
    /// to build two tables.
    func testSlidingLayersUseTheirOwnRopeBase() throws {
        let config = try Self.config(
            layerTypes:
                "[\"sliding_attention\", \"full_attention\", \"sliding_attention\", \"full_attention\"]",
            slidingWindow: "4")
        let backbone = CohereCompassText.Backbone(config.textConfiguration)
        XCTAssertEqual(backbone.rotaryByLayerType.count, 2)

        let positions = MLXArray(0 ..< 6).expandedDimensions(axis: 0)
        let x = MLXArray.zeros([1, 6, 32])
        let sliding = try XCTUnwrap(
            backbone.rotaryByLayerType.first { $0.type == "sliding_attention" }?.rotary)
        let global = try XCTUnwrap(
            backbone.rotaryByLayerType.first { $0.type == "full_attention" }?.rotary)
        let (slidingCos, _) = sliding(x, positionIDs: positions)
        let (globalCos, _) = global(x, positionIDs: positions)
        eval(slidingCos, globalCos)
        XCTAssertGreaterThan(
            abs(slidingCos - globalCos).max().item(Float.self), 1e-4,
            "both layer types rotate identically, so swa_rope_theta is not being read")
    }

    /// `rope_on_all_layers` false leaves the global layers unrotated, which is what a
    /// NoPE-global checkpoint asks for.
    func testRopeCanBeRestrictedToSlidingLayers() throws {
        var text = try Self.config(
            layerTypes: "[\"sliding_attention\", \"full_attention\"]",
            slidingWindow: "4"
        ).textConfiguration
        XCTAssertTrue(text.ropeOnAllLayers)
        XCTAssertTrue(
            CohereCompassText.RotaryEmbedding(text, layerType: "full_attention").enabled)

        text.ropeOnAllLayers = false
        XCTAssertFalse(
            CohereCompassText.RotaryEmbedding(text, layerType: "full_attention").enabled)
        XCTAssertTrue(
            CohereCompassText.RotaryEmbedding(text, layerType: "sliding_attention").enabled)
    }

    /// Three M-RoPE sections that sum to the frequency count select the temporal, height
    /// and width position rows every third frequency.
    func testMRopeSelectorInterleavesTheAxes() throws {
        let config = try Self.config(
            ropeParameters: "{\"mrope_section\": [2, 1, 1], \"rope_theta\": 10000.0}")
        let rotary = CohereCompassText.RotaryEmbedding(
            config.textConfiguration, layerType: "full_attention")
        let selector = try XCTUnwrap(rotary.positionSelector)
        eval(selector)
        // head_dim 8 gives four frequencies; index 1 takes height and index 2 width.
        XCTAssertEqual(selector.asArray(Int32.self), [0, 1, 2, 0])

        // No section, no selector.
        XCTAssertNil(
            CohereCompassText.RotaryEmbedding(
                try Self.config().textConfiguration, layerType: "full_attention"
            ).positionSelector)
    }

    /// Sliding layers get a rotating cache at the architecture's window.
    func testCacheFollowsTheLayerSchedule() throws {
        let model = CohereCompass(
            try Self.config(
                layerTypes:
                    "[\"sliding_attention\", \"full_attention\", \"sliding_attention\", \"full_attention\"]",
                slidingWindow: "4"))
        let cache = try model.newCache(configuration: nil)
        XCTAssertEqual(cache.count, 4)
        for index in [0, 2] {
            let rotating = try XCTUnwrap(cache[index] as? RotatingKVCache)
            XCTAssertEqual(rotating.maxSize, 4)
        }
        for index in [1, 3] {
            XCTAssertFalse(cache[index] is RotatingKVCache)
        }
    }

    /// `logit_scale` multiplies the head's output, which a checkpoint that sets it needs
    /// and one that does not must not get.
    func testLogitScaleMultipliesTheHead() throws {
        let plain = CohereCompass(try Self.config())
        _ = Self.alive(plain)
        let scaled = CohereCompass(try Self.config(logitScale: "0.25"))
        scaled.update(parameters: ModuleParameters.unflattened(plain.parameters().flattened()))

        let tokens = MLXArray([1, 2, 3, 4]).reshaped(1, 4)
        let a = plain(tokens, cache: nil)
        let b = scaled(tokens, cache: nil)
        eval(a, b)
        XCTAssertLessThan(
            abs(a * 0.25 - b).max().item(Float.self),
            1e-4 * abs(b).max().item(Float.self))
    }

    /// The plain merger normalizes the unfolded rows and the deep-stack mergers the folded
    /// ones, so their norms are different widths.
    func testDeepstackMergersNormalizeTheFoldedWidth() throws {
        let config = try Self.config()
        let model = CohereCompass(config)
        let shapes = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
        let vision = try XCTUnwrap(config.visionConfiguration)
        XCTAssertEqual(shapes["vision_tower.merger.norm.weight"], [vision.hiddenSize])
        XCTAssertEqual(
            shapes["vision_tower.deepstack_merger_list.0.norm.weight"],
            [vision.hiddenSize * vision.mergeUnit])
    }

    /// A checkpoint with no vision section builds no tower, and the text path still runs.
    func testTextOnlyCheckpointHasNoTower() throws {
        let model = CohereCompass(try Self.config(vision: false))
        let keys = Set(model.parameters().flattened().map(\.0))
        XCTAssertFalse(keys.contains { $0.hasPrefix("vision_tower.") })

        let out = model(MLXArray([1, 2, 3]).reshaped(1, 3), cache: nil)
        eval(out)
        XCTAssertEqual(out.shape, [1, 3, 128])
    }

    /// The container's prefixes split into the two halves and the tied head is dropped.
    func testSanitizeSplitsTheContainer() throws {
        let model = CohereCompass(try Self.config())
        let out = model.sanitize(weights: [
            "model.language_model.layers.0.self_attn.q_proj.weight": MLXArray.zeros([32, 32]),
            "model.visual.blocks.0.norm1.weight": MLXArray.zeros([32]),
            "model.visual.patch_embed.proj.weight": MLXArray.zeros([32, 3, 2, 4, 4]),
            "lm_head.weight": MLXArray.zeros([128, 32]),
        ])
        XCTAssertNotNil(out["language_model.model.layers.0.self_attn.q_proj.weight"])
        XCTAssertNotNil(out["vision_tower.blocks.0.norm1.weight"])
        XCTAssertNotNil(out["language_model.lm_head.weight"])
        XCTAssertEqual(out["vision_tower.patch_embed.proj.weight"]?.shape, [32, 2, 4, 4, 3])

        let tied = CohereCompass(
            try JSONDecoder.json5().decode(
                CohereCompassConfiguration.self,
                from: XCTUnwrap(
                    Self.json().replacingOccurrences(
                        of: "\"tie_word_embeddings\": false",
                        with: "\"tie_word_embeddings\": true"
                    ).data(using: .utf8))))
        XCTAssertNil(
            tied.sanitize(weights: ["lm_head.weight": MLXArray.zeros([128, 32])])["lm_head.weight"])
    }

    /// Every projection runs one row at a time above batch one, so a row's logits do not
    /// depend on how many rows shared the call.
    func testRowwiseBatchKeepsRowsIndependent() throws {
        let model = CohereCompass(try Self.config())
        _ = Self.alive(model)

        let first = MLXArray([1, 2, 3, 4]).reshaped(1, 4)
        let second = MLXArray([9, 8, 7, 6]).reshaped(1, 4)
        let together = concatenated([first, second], axis: 0)

        let alone = model(first, cache: nil)
        let batched = model(together, cache: nil)
        eval(alone, batched)

        XCTAssertLessThan(
            abs(alone[0] - batched[0]).max().item(Float.self),
            1e-5 * abs(alone).max().item(Float.self),
            "a row's logits changed when another row joined the call")
    }

    /// A chunked prefill has to reproduce the flat forward through the parallel blocks and
    /// the rotating caches.
    func testChunkedPrefillMatchesTheFlatForward() throws {
        let model = CohereCompass(
            try Self.config(
                layerTypes:
                    "[\"sliding_attention\", \"full_attention\", \"sliding_attention\", \"full_attention\"]",
                slidingWindow: "4"))
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
            abs(flat[0, 8] - tail[0, 3]).max().item(Float.self), 1e-3 * scale,
            "the chunk boundary does not reproduce the flat forward")
    }

    /// The deep-stack features are added at the image positions, once per layer, and
    /// nowhere else.
    func testDeepstackAddsAtVisualPositionsOnly() {
        let hidden = MLXArray.zeros([1, 5, 4])
        let mask = MLXArray([false, true, true, false, false]).reshaped(1, 5)
        let embeddings = concatenated(
            [MLXArray.ones([1, 4]), MLXArray.ones([1, 4]) * 2], axis: 0)

        let out = CohereCompassText.Backbone.addDeepstack(
            hidden, visualPositionMask: mask, embeddings: embeddings)
        eval(out)
        XCTAssertEqual(out[0, 0].asArray(Float.self), [0, 0, 0, 0])
        XCTAssertEqual(out[0, 1].asArray(Float.self), [1, 1, 1, 1])
        XCTAssertEqual(out[0, 2].asArray(Float.self), [2, 2, 2, 2])
        XCTAssertEqual(out[0, 4].asArray(Float.self), [0, 0, 0, 0])

        // No mask, no change.
        let untouched = CohereCompassText.Backbone.addDeepstack(
            hidden, visualPositionMask: nil, embeddings: embeddings)
        eval(untouched)
        XCTAssertEqual(abs(untouched).max().item(Float.self), 0, accuracy: 1e-7)
    }
}
