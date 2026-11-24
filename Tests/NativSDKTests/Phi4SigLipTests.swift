import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class Phi4SigLipTests: XCTestCase {

    private static let json = """
        {
          "model_type": "phi4-siglip",
          "hidden_size": 32,
          "num_hidden_layers": 4,
          "num_attention_heads": 4,
          "num_key_value_heads": 2,
          "intermediate_size": 48,
          "vocab_size": 128,
          "rms_norm_eps": 1e-05,
          "rope_theta": 500000.0,
          "partial_rotary_factor": 0.5,
          "mm_hidden_size": 24,
          "mm_vision_select_layer": -2,
          "text_config": {
            "hidden_size": 999,
            "num_hidden_layers": 99,
            "partial_rotary_factor": 1.0
          },
          "vision_config": {
            "model_type": "siglip2_vision_model",
            "hidden_size": 24,
            "intermediate_size": 32,
            "num_hidden_layers": 3,
            "num_attention_heads": 4,
            "num_channels": 3,
            "image_size": 32,
            "patch_size": 8,
            "num_patches": 16,
            "layer_norm_eps": 1e-06
          }
        }
        """

    private static func config() throws -> Phi4SigLipConfiguration {
        try JSONDecoder.json5().decode(
            Phi4SigLipConfiguration.self, from: XCTUnwrap(json.data(using: .utf8)))
    }

    private static func alive(_ model: Module, seed: UInt64 = 23) -> Module {
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

    /// The checkpoint repeats the text fields at the top level, and those win over
    /// anything nested under `text_config`. Reading only the nested object builds a
    /// model of the wrong size against a checkpoint that looks fine.
    func testTopLevelTextFieldsOverrideTheNestedOnes() throws {
        let config = try Self.config()
        XCTAssertEqual(config.textConfiguration.hiddenSize, 32)
        XCTAssertEqual(config.textConfiguration.hiddenLayers, 4)
        XCTAssertEqual(config.textConfiguration.attentionHeads, 4)
        XCTAssertEqual(config.textConfiguration.kvHeads, 2)
        XCTAssertEqual(config.textConfiguration.vocabularySize, 128)
        XCTAssertEqual(config.textConfiguration.partialRotaryFactor, 0.5)

        // A field only `text_config` declares still comes through.
        let nestedOnly = try JSONDecoder.json5().decode(
            Phi4SigLipConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "phi4-siglip",
                  "text_config": { "hidden_size": 64, "num_key_value_heads": 8 },
                  "vision_config": { "model_type": "siglip2_vision_model" }
                }
                """.data(using: .utf8)))
        XCTAssertEqual(nestedOnly.textConfiguration.hiddenSize, 64)
        XCTAssertEqual(nestedOnly.textConfiguration.kvHeads, 8)
    }

    /// Only `linear` scaling is read, which is all upstream honors here.
    func testOnlyLinearRopeScalingScales() throws {
        func scale(_ scaling: String) throws -> Float {
            try JSONDecoder.json5().decode(
                Phi4SigLipConfiguration.self,
                from: XCTUnwrap(
                    """
                    {
                      "model_type": "phi4-siglip", "rope_scaling": \(scaling),
                      "vision_config": { "model_type": "siglip2_vision_model" }
                    }
                    """.data(using: .utf8))
            ).textConfiguration.ropeScale
        }

        XCTAssertEqual(try scale("null"), 1)
        XCTAssertEqual(try scale("{\"type\": \"linear\", \"factor\": 4.0}"), 0.25)
        XCTAssertEqual(try scale("{\"rope_type\": \"linear\", \"factor\": 2.0}"), 0.5)
        XCTAssertEqual(try scale("{\"type\": \"longrope\", \"factor\": 4.0}"), 1)
    }

    /// The patch embedding is a linear over flattened patches, not a convolution, so the
    /// pixel input arrives already patchified.
    func testPatchEmbeddingIsLinearOverFlattenedPatches() throws {
        let config = try Self.config()
        let model = Phi4SigLip(try Self.config())
        let shapes = Dictionary(
            uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
        let patchWidth =
            config.visionConfiguration.channels * config.visionConfiguration.patchSize
            * config.visionConfiguration.patchSize
        XCTAssertEqual(
            shapes["vision_tower.vision_tower.vision_model.embeddings.patch_embedding.weight"],
            [config.visionConfiguration.hiddenSize, patchWidth])
    }

    /// The hidden-state list is seeded with the patch embedding, so `-2` names the
    /// second-to-last *layer*, not the last one.
    func testHiddenStatesAreSeededWithTheEmbedding() throws {
        let config = try Self.config()
        let model = Phi4SigLip(try Self.config())
        _ = Self.alive(model)

        let patchWidth =
            config.visionConfiguration.channels * config.visionConfiguration.patchSize
            * config.visionConfiguration.patchSize
        let pixels = MLXRandom.normal([1, 16, patchWidth], key: MLXRandom.key(9))

        let (states, embedded, pooled) = model.visionTower(
            pixels, spatialShapes: nil)
        eval(embedded, pooled)

        XCTAssertEqual(states.count, config.visionConfiguration.hiddenLayers + 1)
        XCTAssertEqual(
            abs(states[0] - embedded).max().item(Float.self), 0, accuracy: 1e-7,
            "the list does not start at the patch embedding")
        XCTAssertEqual(
            Phi4SigLip.index(-2, count: states.count),
            config.visionConfiguration.hiddenLayers - 1)
        XCTAssertEqual(
            Phi4SigLip.index(-1, count: states.count),
            config.visionConfiguration.hiddenLayers)
    }

    /// Positional embeddings are resampled to each image's own patch grid, and the
    /// positions past that grid repeat the first row.
    func testPositionalEmbeddingsResampleToTheSpatialShape() throws {
        let config = try Self.config()
        let model = Phi4SigLip(try Self.config())
        _ = Self.alive(model)

        let patchWidth =
            config.visionConfiguration.channels * config.visionConfiguration.patchSize
            * config.visionConfiguration.patchSize
        let pixels = MLXArray.zeros([1, 16, patchWidth])

        // A 2x3 grid means six real positions; the remaining ten repeat the first.
        let shapes = MLXArray([Int32(2), 3], [1, 2])
        let (_, embedded, _) = model.visionTower(pixels, spatialShapes: shapes)
        eval(embedded)

        XCTAssertEqual(embedded.shape, [1, 16, config.visionConfiguration.hiddenSize])
        for position in 6 ..< 16 {
            XCTAssertEqual(
                abs(embedded[0, position] - embedded[0, 0]).max().item(Float.self), 0,
                accuracy: 1e-5, "position \(position) is not the repeated first row")
        }
        XCTAssertGreaterThan(
            abs(embedded[0, 1] - embedded[0, 0]).max().item(Float.self), 1e-5,
            "the resampled grid is constant")
    }

    /// The checkpoint stores the projector as a numbered sequence and keeps the tower and
    /// projector under `model.`, which the language model must not absorb.
    func testSanitizeNamesTheProjectorAndPrefixesOnlyTheLanguageModel() throws {
        let model = Phi4SigLip(try Self.config())
        let out = model.sanitize(weights: [
            "model.mm_projector.0.weight": MLXArray.zeros([32, 24]),
            "model.mm_projector.2.bias": MLXArray.zeros([32]),
            "model.vision_tower.vision_tower.vision_model.post_layernorm.weight":
                MLXArray.zeros([24]),
            "model.layers.0.mlp.down_proj.weight": MLXArray.zeros([32, 48]),
            "lm_head.weight": MLXArray.zeros([128, 32]),
            "model.vision_tower.vision_tower.vision_model.head.probe": MLXArray.zeros([1]),
            "model.embed_tokens.position_ids": MLXArray.zeros([4]),
        ])

        XCTAssertNotNil(out["mm_projector.linear_1.weight"])
        XCTAssertNotNil(out["mm_projector.linear_2.bias"])
        XCTAssertNotNil(
            out["vision_tower.vision_tower.vision_model.post_layernorm.weight"])
        XCTAssertNotNil(out["language_model.model.layers.0.mlp.down_proj.weight"])
        XCTAssertNotNil(out["language_model.lm_head.weight"])

        XCTAssertNil(out["language_model.model.mm_projector.0.weight"])
        XCTAssertNil(
            out["language_model.model.vision_tower.vision_tower.vision_model.post_layernorm.weight"]
        )
        XCTAssertNil(out["vision_tower.vision_tower.vision_model.head.probe"])
        XCTAssertNil(out["language_model.model.embed_tokens.position_ids"])
    }

    /// Text generation with the fused projection and partial rotary: a chunked prefill has
    /// to reproduce the flat forward.
    func testChunkedPrefillMatchesTheFlatForward() throws {
        let model = Phi4SigLip(try Self.config())
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
        XCTAssertEqual(cache.count, 4)
        XCTAssertEqual(cache[0].offset, 9)
    }

}
