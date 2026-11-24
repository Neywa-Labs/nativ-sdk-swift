import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class RecurrentGemmaTests: XCTestCase {
    func testCheckpointWithoutOutputHeadUsesEmbeddingWeightsForLogits() throws {
        let config = try Self.config()
        let model = RecurrentGemmaModel(config)
        let original = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        let checkpoint = original.filter { !$0.key.hasPrefix("lm_head.") }
        let sanitized = model.sanitize(weights: checkpoint)
        try model.update(parameters: ModuleParameters.unflattened(sanitized), verify: .all)
        XCTAssertNil(model.lmHead)
        let input = MLXArray([Int32(1), 2]).reshaped(1, 2)
        let raw = model.model.embedTokens.asLinear(model.model(input, cache: nil))
        let expected = tanh(raw / config.logitsSoftCap) * config.logitsSoftCap
        let actual = model(input, cache: nil)
        eval(expected, actual)
        XCTAssertLessThan(abs(expected - actual).max().item(Float.self), 1e-6)
        let untied = RecurrentGemmaModel(config)
        _ = untied.sanitize(weights: original)
        XCTAssertNotNil(untied.lmHead)
    }

    private static func config(
        blockTypes: [String] = ["recurrent", "recurrent", "attention"]
    ) throws -> RecurrentGemmaConfiguration {
        try JSONDecoder.json5().decode(
            RecurrentGemmaConfiguration.self,
            from: ModelTestConfigurations.configuration(
                "recurrent_gemma",
                overrides: [
                    "block_types": blockTypes
                ]))
    }

    /// `h_t = a_t * h_{t-1} + x_t`, checked against the recurrence written out by hand,
    /// and the returned state must be the last `h`.
    func testScanMatchesTheRecurrenceAndReturnsTheLastState() {
        let x = MLXArray(converting: [1.0, 2.0, 3.0]).reshaped(1, 3, 1)
        let a = MLXArray(converting: [0.5, 0.5, 0.5]).reshaped(1, 3, 1)
        let (y, state) = recurrentGemmaScan(x: x, a: a, state: nil)
        eval(y, state)

        XCTAssertEqual(y.reshaped(3).asArray(Float.self), [1, 2.5, 4.25])
        XCTAssertEqual(state.reshaped(1).asArray(Float.self), [4.25])
    }

    /// Splitting a sequence and carrying the state must give the same trajectory as one
    /// pass, which is the property the decode path depends on.
    func testScanIsResumable() {
        let x = MLXArray(converting: [1.0, 2.0, 3.0, 4.0]).reshaped(1, 4, 1)
        let a = MLXArray(converting: [0.9, 0.8, 0.7, 0.6]).reshaped(1, 4, 1)

        let (whole, _) = recurrentGemmaScan(x: x, a: a, state: nil)
        let (head, carried) = recurrentGemmaScan(
            x: x[0..., ..<2], a: a[0..., ..<2], state: nil)
        let (tail, _) = recurrentGemmaScan(
            x: x[0..., 2...], a: a[0..., 2...], state: carried)
        let joined = concatenated([head, tail], axis: 1)
        eval(whole, joined)

        XCTAssertLessThan(abs(whole - joined).max().item(Float.self), 1e-6)
    }

    /// A single position with no incoming state takes its input unattenuated, where every
    /// other position -- and position zero of a resumed sequence -- is scaled by
    /// `sqrt(1 - a^2)`.
    ///
    /// With zero gate weights both gates sit at `sigmoid(0) = 0.5`, so a fresh position zero
    /// must come out at exactly half its input. `recurrent_param` has to be well negative
    /// for the contrast to exist at all: `a = exp(-8 * gate * softplus(param))`, and a
    /// positive param drives `a` to about 2e-4, where `sqrt(1 - a^2)` rounds to one and the
    /// override changes nothing.
    func testFirstPositionOfAFreshSequenceIsUnattenuated() throws {
        let unit = RecurrentGemmaRGLRU(width: 4, heads: 1)
        unit.update(
            parameters: ModuleParameters.unflattened([
                ("recurrent_param", MLXArray.ones([4]) * -10)
            ]))

        let x = MLXArray.ones([1, 3, 4])
        let (fresh, _) = unit(x, state: nil)
        let (resumed, _) = unit(x, state: MLXArray.zeros([1, 4]))
        eval(fresh, resumed)

        XCTAssertEqual(
            fresh[0, 0].max().item(Float.self), 0.5, accuracy: 1e-4,
            "a fresh position zero was attenuated")
        XCTAssertLessThan(
            resumed[0, 0].max().item(Float.self), 0.05,
            "a resumed position zero was not attenuated, so the override is unconditional")
    }

    /// The rotary covers `head_dim / 2` dimensions, so the upper half of every head passes
    /// through untouched. A full-width rotary would move all of it.
    func testRotaryCoversHalfTheHead() throws {
        let block = RecurrentGemmaLocalAttentionBlock(width: 32, heads: 4)
        let heads = MLXArray(1 ..< 33).reshaped(1, 4, 1, 8).asType(.float32)
        let rotated = block.rope(heads, offset: 3)
        eval(rotated, heads)

        let lower = abs(rotated[0..., 0..., 0..., ..<4] - heads[0..., 0..., 0..., ..<4]).max()
        let upper = abs(rotated[0..., 0..., 0..., 4...] - heads[0..., 0..., 0..., 4...]).max()
        eval(lower, upper)
        XCTAssertGreaterThan(lower.item(Float.self), 1e-3)
        XCTAssertEqual(
            upper.item(Float.self), 0, accuracy: 1e-7,
            "the rotary reached the upper half of the head")
    }

    /// Keys and values project to exactly `head_dim` -- one head -- whatever
    /// `num_key_value_heads` says.
    func testKeysAndValuesAreASingleHead() throws {
        let block = RecurrentGemmaLocalAttentionBlock(width: 32, heads: 4)
        XCTAssertEqual(block.wq.weight.dim(0), 32)
        XCTAssertEqual(block.wk.weight.dim(0), 8)
        XCTAssertEqual(block.wv.weight.dim(0), 8)
        XCTAssertNotNil(block.wo.bias, "o_proj carries a bias where q/k/v do not")
        XCTAssertNil(block.wq.bias)
    }

    /// `intermediate_size` is the total expanded width; each branch is half.
    func testFeedForwardHalvesTheExpandedWidth() {
        let mlp = RecurrentGemmaMLPBlock(width: 32, expandedWidth: 64)
        XCTAssertEqual(mlp.gateProj.weight.dim(0), 32)
        XCTAssertEqual(mlp.upProj.weight.dim(0), 32)
        XCTAssertEqual(mlp.downProj.weight.dim(1), 32)
        XCTAssertNotNil(mlp.gateProj.bias)
    }

    /// The whole state handoff at once: convolution tail, recurrent state and windowed
    /// key/value cache. Two chunks through one cache must reproduce a single forward.
    func testChunkedPrefillMatchesOnePass() throws {
        let config = try Self.config()
        let tokens = MLXArray(0 ..< 12).reshaped(1, 12)

        let whole = RecurrentGemmaModel(config)
        let onePass = whole(tokens, cache: try whole.newCache(configuration: nil))

        let split = RecurrentGemmaModel(config)
        split.update(
            parameters: ModuleParameters.unflattened(whole.parameters().flattened()))
        let cache = try split.newCache(configuration: nil)
        _ = split(tokens[0..., ..<8], cache: cache)
        let tail = split(tokens[0..., 8...], cache: cache)
        eval(onePass, tail)

        XCTAssertLessThan(
            abs(onePass[0, 11] - tail[0, 3]).max().item(Float.self), 2e-3,
            "chunking changed the answer, so some state is not carried across forwards")
    }

    /// A checkpoint stores the depthwise kernel as `[channels, 1, kernel]`.
    func testSanitizeMovesTheConvolutionAxis() throws {
        let model = RecurrentGemmaModel(try Self.config())
        let stored = MLXArray(0 ..< 24).reshaped(6, 1, 4).asType(.float32)
        let sanitized = model.sanitize(weights: [
            "model.layers.0.temporal_block.conv_1d.weight": stored,
            "model.layers.0.temporal_block.conv_1d.bias": MLXArray.zeros([6]),
        ])

        let moved = try XCTUnwrap(
            sanitized["model.layers.0.temporal_block.conv_1d.weight"])
        XCTAssertEqual(moved.shape, [6, 4, 1])
        eval(moved)
        XCTAssertEqual(moved[1, 2, 0].item(Float.self), 6)

        let alreadyRight = MLXArray.zeros([6, 4, 1])
        XCTAssertEqual(
            model.sanitize(weights: ["a.conv_1d.weight": alreadyRight])["a.conv_1d.weight"]?
                .shape,
            [6, 4, 1])
    }

    func testLogitsAreSoftcappedAndShapeIsRight() throws {
        let model = RecurrentGemmaModel(try Self.config())
        model.update(
            parameters: ModuleParameters.unflattened([
                ("lm_head.weight", MLXRandom.normal([48, 32], key: MLXRandom.key(3)) * 40)
            ]))
        let logits = model(
            MLXArray(0 ..< 8).reshaped(1, 8), cache: try model.newCache(configuration: nil))
        eval(logits)

        XCTAssertEqual(logits.shape, [1, 8, 48])
        XCTAssertLessThan(abs(logits).max().item(Float.self), 30.0001)
        XCTAssertFalse(isNaN(logits.asType(.float32)).any().item(Bool.self))
    }

    func testUnderscoredBlockTypesAreRead() throws {
        let config = try JSONDecoder.json5().decode(
            RecurrentGemmaConfiguration.self,
            from: ModelTestConfigurations.configuration(
                "recurrent_gemma",
                overrides: [
                    "_block_types": ["recurrent", "recurrent", "attention"]
                ], removing: ["block_types"]))
        XCTAssertEqual(config.blockTypes, ["recurrent", "recurrent", "attention"])
    }

}
