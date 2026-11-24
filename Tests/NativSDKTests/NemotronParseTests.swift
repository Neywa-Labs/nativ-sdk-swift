import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class NemotronParseTests: XCTestCase {

    private static let json = """
        {
          "model_type": "nemotron_parse",
          "decoder_start_token_id": 2,
          "image_size": [64, 96],
          "encoder": {
            "hidden_size": 32,
            "num_heads": 4,
            "mlp_ratio": 2.0,
            "num_layers": 2,
            "patch_size": 16,
            "num_cls_tokens": 4,
            "num_register_tokens": 4,
            "summary_idxs": [0, 1, 2],
            "neck_dim": 24
          },
          "decoder": {
            "d_model": 24,
            "decoder_attention_heads": 4,
            "decoder_ffn_dim": 32,
            "decoder_layers": 3,
            "vocab_size": 96,
            "decoder_start_token_id": null,
            "scale_embedding": true
          }
        }
        """

    private static func config() throws -> NemotronParseConfiguration {
        try JSONDecoder.json5().decode(
            NemotronParseConfiguration.self, from: XCTUnwrap(json.data(using: .utf8)))
    }

    private static func alive(_ model: Module, seed: UInt64 = 31) -> Module {
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

    /// The two halves live under `encoder` and `decoder`, and the decoder sub-config
    /// writes a null start token that the top level has to supply.
    func testHalvesComeFromEncoderAndDecoderKeys() throws {
        let config = try Self.config()
        XCTAssertEqual(config.visionConfiguration.hiddenSize, 32)
        XCTAssertEqual(config.visionConfiguration.layers, 2)
        XCTAssertEqual(config.visionConfiguration.neckDimensions, 24)
        XCTAssertEqual(config.textConfiguration.modelDimension, 24)
        XCTAssertEqual(config.textConfiguration.layers, 3)
        XCTAssertEqual(config.textConfiguration.vocabularySize, 96)
        XCTAssertEqual(config.textConfiguration.decoderStartTokenID, 2)

        // The top-level image size promotes into the vision half.
        XCTAssertEqual(config.visionConfiguration.imageSize, [64, 96])
    }

    /// A checkpoint that keeps its halves under the conventional names still loads.
    func testConventionalConfigKeysAlsoWork() throws {
        let config = try JSONDecoder.json5().decode(
            NemotronParseConfiguration.self,
            from: XCTUnwrap(
                """
                {
                  "model_type": "nemotron_parse",
                  "vision_config": { "hidden_size": 40, "num_layers": 1 },
                  "text_config": { "d_model": 16, "decoder_layers": 2 }
                }
                """.data(using: .utf8)))
        XCTAssertEqual(config.visionConfiguration.hiddenSize, 40)
        XCTAssertEqual(config.textConfiguration.modelDimension, 16)
    }

    /// Pixels arrive channel-first and become one row per patch, each row the patch's
    /// channels and rows and columns in that order.
    func testPatchifyOrdersEachPatchByChannelThenRowThenColumn() throws {
        let model = NemotronParse(try Self.config())
        let patch = 16
        let pixels = MLXArray(0 ..< (1 * 3 * 32 * 32)).reshaped(1, 3, 32, 32)
            .asType(.float32)

        let (patches, grid) = model.visionTower.patchify(pixels)
        eval(patches)
        XCTAssertEqual(grid.0, 2)
        XCTAssertEqual(grid.1, 2)
        XCTAssertEqual(patches.shape, [1, 4, 3 * patch * patch])

        // Patch (0, 0), channel 0, row 0 is the image's first sixteen pixels.
        let first = patches[0, 0, ..<patch].asArray(Float.self)
        XCTAssertEqual(first, (0 ..< patch).map(Float.init))

        // The same patch's channel 1 starts a full plane later.
        let plane = 32 * 32
        XCTAssertEqual(
            patches[0, 0, (patch * patch) ..< (patch * patch + 1)].item(Float.self),
            Float(plane))

        // Patch (0, 1) starts sixteen columns into the first row.
        XCTAssertEqual(patches[0, 1, 0].item(Float.self), Float(patch))
    }

    /// The learned grid is 128 by 128. A grid that fits is cropped, not resampled; a
    /// smaller one is resampled to the square of its larger side and then cropped.
    func testPositionGridIsCroppedWhenItFitsAndResampledOtherwise() throws {
        let model = NemotronParse(try Self.config())
        _ = Self.alive(model)
        let hiddenSize = try Self.config().visionConfiguration.hiddenSize

        let full = model.visionTower.positions(rows: 128, columns: 128)
        eval(full)
        XCTAssertEqual(full.shape, [128 * 128, hiddenSize])

        // 128 by 104 is a pure crop: the square resample to 128 is the identity, so
        // row r column c has to equal the full grid's own row r column c.
        let cropped = model.visionTower.positions(rows: 128, columns: 104)
        eval(cropped)
        XCTAssertEqual(cropped.shape, [128 * 104, hiddenSize])
        XCTAssertEqual(
            abs(cropped[104] - full[128]).max().item(Float.self), 0, accuracy: 1e-6,
            "the crop is not taking the first columns of each row")

        let resampled = model.visionTower.positions(rows: 4, columns: 3)
        eval(resampled)
        XCTAssertEqual(resampled.shape, [12, hiddenSize])
    }

    /// Four class tokens then four register tokens sit in front of the patches. The
    /// summary reads three of the class tokens; the features start after the registers.
    func testSummaryAndFeatureSplitSkipsTheRegisterTokens() throws {
        let config = try Self.config()
        let model = NemotronParse(config)
        _ = Self.alive(model)

        let pixels = MLXRandom.normal([1, 3, 32, 32], key: MLXRandom.key(2))
        let encoded = model.visionTower(pixels)
        eval(encoded)

        // A 2x2 patch grid compresses to 2x1 horizontally, plus one summary row.
        XCTAssertEqual(encoded.shape, [1, 2 * 1 + 1, config.visionConfiguration.neckDimensions])
        XCTAssertEqual(config.visionConfiguration.skippedTokens, 8)
    }

    /// The horizontal 4x compression is what sets the encoder's sequence length, and the
    /// summary is appended at the end rather than the front.
    func testNeckCompressesHorizontallyByFour() throws {
        let config = try Self.config()
        let model = NemotronParse(config)
        _ = Self.alive(model)

        // A 64x128 image is a 4x8 patch grid, compressing to 4x2 plus the summary.
        let pixels = MLXRandom.normal([1, 3, 64, 128], key: MLXRandom.key(3))
        let encoded = model.visionTower(pixels)
        eval(encoded)
        XCTAssertEqual(encoded.shape, [1, 4 * 2 + 1, config.visionConfiguration.neckDimensions])
    }

    /// The encoder's keys and values are projected once and then read from the cache.
    /// Recomputing them every step would run two projections over the whole page per
    /// decoded token, so this is a throughput property as much as a correctness one.
    func testCrossAttentionKeysAreProjectedOnlyOnce() throws {
        let config = try Self.config()
        let model = NemotronParse(config)
        _ = Self.alive(model)

        let pixels = MLXRandom.normal([1, 3, 32, 32], key: MLXRandom.key(4))
        let encoderStates = model.visionTower(pixels)
        var state = LMOutput.State()
        state[nemotronParseEncoderStatesKey] = encoderStates

        let start = MLXArray([Int32(config.textConfiguration.decoderStartTokenID)])
        let next = MLXArray([Int32(7)])

        let warm = try model.newCache(configuration: nil)
        _ = model(.init(tokens: start, mask: nil), cache: warm, state: state)
        eval(warm.flatMap { $0.innerState() })

        let slots = try XCTUnwrap(
            (warm[0] as? CacheList)?.children[1] as? ArraysCache)
        XCTAssertNotNil(slots[0], "the encoder keys were not cached")
        XCTAssertNotNil(slots[1], "the encoder values were not cached")

        // Two copies of the same warm state, so the only difference between the runs is
        // the perturbation -- the self-attention cache advances on every call.
        let branchA = warm.map { $0.copy() }
        let branchB = warm.map { $0.copy() }

        let before = model(.init(tokens: next, mask: nil), cache: branchA, state: state)
        eval(before.logits)

        let path = "language_model.model.decoder.layers.0.encoder_attn.k_proj.weight"
        let weight = try XCTUnwrap(model.parameters().flattened().first { $0.0 == path }?.1)
        try model.update(
            parameters: ModuleParameters.unflattened([(path, weight * 7)]), verify: .noUnusedKeys)
        let after = model(.init(tokens: next, mask: nil), cache: branchB, state: state)
        eval(after.logits)

        XCTAssertLessThan(
            abs(before.logits - after.logits).max().item(Float.self), 1e-4,
            "the cross-attention keys are being recomputed every step")

        // On a cold cache the same perturbation does change the output, so the assertion
        // above is not passing because the projection is unused.
        let cold = try model.newCache(configuration: nil)
        let perturbed = model(.init(tokens: start, mask: nil), cache: cold, state: state)
        let unperturbed = NemotronParse(config)
        unperturbed.update(
            parameters: ModuleParameters.unflattened(model.parameters().flattened()))
        unperturbed.update(
            parameters: ModuleParameters.unflattened([
                (
                    "language_model.model.decoder.layers.0.encoder_attn.k_proj.weight",
                    model.parameters().flattened().first {
                        $0.0
                            == "language_model.model.decoder.layers.0.encoder_attn.k_proj.weight"
                    }!.1 / 7
                )
            ]))
        let baseline = unperturbed(
            .init(tokens: start, mask: nil),
            cache: try unperturbed.newCache(configuration: nil), state: state)
        eval(perturbed.logits, baseline.logits)
        XCTAssertGreaterThan(
            abs(perturbed.logits - baseline.logits).max().item(Float.self), 1e-4,
            "the cross-attention key projection does not reach the logits at all")
    }

    /// The decoder carries the encoder states forward in the output state, so a step
    /// after the first does not need the image again.
    func testEncoderStatesAreCarriedInTheOutputState() throws {
        let config = try Self.config()
        let model = NemotronParse(config)
        _ = Self.alive(model)

        let pixels = MLXRandom.normal([1, 3, 32, 32], key: MLXRandom.key(5))
        let encoderStates = model.visionTower(pixels)
        var state = LMOutput.State()
        state[nemotronParseEncoderStatesKey] = encoderStates

        let cache = try model.newCache(configuration: nil)
        let out = model(
            .init(tokens: MLXArray([Int32(2)]), mask: nil), cache: cache, state: state)
        eval(out.logits)

        XCTAssertEqual(out.logits.shape, [1, 1, config.textConfiguration.vocabularySize])
        let carried = try XCTUnwrap(out.state?[nemotronParseEncoderStatesKey])
        eval(carried)
        XCTAssertEqual(
            abs(carried - encoderStates).max().item(Float.self), 0, accuracy: 1e-7)
    }

    func testInterleavedDocumentsKeepIndependentEncoderState() throws {
        let model = NemotronParse(try Self.config())
        _ = Self.alive(model)
        func prepare(_ seed: UInt64) throws -> ([KVCache], LMOutput) {
            let cache = try model.newCache(configuration: nil)
            let image = MLXRandom.normal([1, 3, 32, 32], key: MLXRandom.key(seed))
            let input = LMInput(text: .init(tokens: MLXArray([2])), image: .init(pixels: image))
            guard
                case .logits(let output) = try model.prepare(
                    input, cache: cache, state: nil, prefill: .init())
            else {
                throw VLMError.processing("Expected initial decoder logits")
            }
            eval(output.logits, cache)
            return (cache, output)
        }
        let (firstCache, first) = try prepare(9)
        let savedCache = firstCache.map { $0.copy() }
        let savedState = try model.copyContinuationState(XCTUnwrap(first.state))
        let expected = model(.init(tokens: MLXArray([3])), cache: savedCache, state: savedState)
        eval(expected.logits)
        let (_, second) = try prepare(27)
        XCTAssertGreaterThan(abs(first.logits - second.logits).max().item(Float.self), 1e-5)
        let actual = model(.init(tokens: MLXArray([3])), cache: firstCache, state: first.state)
        XCTAssertLessThan(abs(actual.logits - expected.logits).max().item(Float.self), 1e-6)
    }

    /// The checkpoint's names have to land on the module tree: the ViT under a
    /// patch-generator prefix, the neck under everything else named `encoder`, and the
    /// tied embedding moved off the decoder onto the shared table.
    func testSanitizeMapsTheCheckpointNames() throws {
        let model = NemotronParse(try Self.config())
        let out = model.sanitize(weights: [
            "encoder.model_encoder.radio_model.model.patch_generator.embedder.weight":
                MLXArray.zeros([32, 768]),
            "encoder.model_encoder.radio_model.model.patch_generator.pos_embed":
                MLXArray.zeros([1, 128 * 128, 32]),
            "encoder.model_encoder.radio_model.model.patch_generator.cls_token.token":
                MLXArray.zeros([1, 8, 32]),
            "encoder.model_encoder.radio_model.model.blocks.0.norm1.weight":
                MLXArray.zeros([32]),
            "encoder.conv1.weight": MLXArray.zeros([24, 32, 1]),
            "encoder.sum_proj.weight": MLXArray.zeros([24, 96]),
            "decoder.embed_tokens.weight": MLXArray.zeros([96, 24]),
            "decoder.layers.0.self_attn.q_proj.weight": MLXArray.zeros([24, 24]),
            "encoder.model_encoder.summary_idxs": MLXArray.zeros([3]),
            "encoder.model_encoder.input_conditioner.norm_mean": MLXArray.zeros([3]),
        ])

        XCTAssertNotNil(out["vision_tower.patch_embed.weight"])
        XCTAssertNotNil(out["vision_tower.pos_embed"])
        XCTAssertNotNil(out["vision_tower.cls_token"])
        XCTAssertNotNil(out["vision_tower.blocks.0.norm1.weight"])
        XCTAssertNotNil(out["vision_tower.neck.conv1.weight"])
        XCTAssertNotNil(out["vision_tower.neck.sum_proj.weight"])
        XCTAssertNotNil(out["language_model.model.shared.weight"])
        XCTAssertNotNil(out["language_model.model.decoder.layers.0.self_attn.q_proj.weight"])

        XCTAssertNil(out["language_model.model.decoder.embed_tokens.weight"])
        XCTAssertNil(out["encoder.model_encoder.summary_idxs"])
        XCTAssertNil(out["encoder.model_encoder.input_conditioner.norm_mean"])

        // Tied, so the head is reconstructed from the shared table.
        let head = try XCTUnwrap(out["language_model.lm_head.weight"])
        XCTAssertEqual(head.shape, [96, 24])

        // Torch stores the width-one convolution as [out, in, 1].
        XCTAssertEqual(out["vision_tower.neck.conv1.weight"]?.shape, [24, 1, 32])
    }

    /// The layout moves are idempotent, because the loader runs sanitize over
    /// already-converted checkpoints too.
    func testSanitizeIsIdempotent() throws {
        let model = NemotronParse(try Self.config())
        let once = model.sanitize(weights: [
            "encoder.conv1.weight": MLXArray.zeros([24, 32, 1]),
            "encoder.conv2.weight": MLXArray.zeros([24, 24, 1, 4]),
        ])
        let twice = model.sanitize(weights: once)

        XCTAssertEqual(once["vision_tower.neck.conv1.weight"]?.shape, [24, 1, 32])
        XCTAssertEqual(twice["vision_tower.neck.conv1.weight"]?.shape, [24, 1, 32])
        XCTAssertEqual(once["vision_tower.neck.conv2.weight"]?.shape, [24, 1, 4, 24])
        XCTAssertEqual(twice["vision_tower.neck.conv2.weight"]?.shape, [24, 1, 4, 24])
    }

}
