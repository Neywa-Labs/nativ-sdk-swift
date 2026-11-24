import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

/// Expectations captured from upstream mlx-vlm's `engram.py`, for a
/// deliberately tiny layout, and written out rather than recomputed.
final class DeepSeekV41EngramTests: XCTestCase {
    private static let tokenMap = [0, 1, 2, 3, 4, 5, 6, 7, 3, 1]

    private func layout() throws -> DeepSeekV41EngramLayout {
        try XCTUnwrap(
            DeepSeekV41EngramLayout(
                layerIds: [1, 14], maxNgramSize: 3, heads: 2, headDim: 4,
                vocabularySize: 100, numEmbeddings: [1000, 1000]))
    }

    private func hashState() throws -> DeepSeekV41NgramHashState {
        try DeepSeekV41NgramHashState(
            layout: try layout(), tokenMap: Self.tokenMap, padTokenId: 2,
            compressedVocabularySize: 8)
    }

    /// Every (n-gram size, head) pair owns a disjoint prime-sized range, drawn in order
    /// and never reused — including across layers.
    func testPrimeRangesMatchUpstream() throws {
        XCTAssertEqual(
            try layout().primes, [[[101, 103], [107, 109]], [[113, 127], [131, 137]]])
    }

    /// The offsets are the running start of each bucket range within its layer's table.
    func testOffsetsAreTheRunningRangeStarts() throws {
        XCTAssertEqual(
            try hashState().offsets.asType(.int32).asArray(Int32.self),
            [0, 101, 204, 311, 0, 113, 240, 371])
    }

    /// One generator per layer, seeded 10007 * layer_id, so the two layers hash
    /// differently. These are NumPy's draws.
    func testMultipliersMatchUpstream() throws {
        XCTAssertEqual(
            try hashState().multipliers.asArray(Int64.self),
            [
                949_203_457_676_829_153, 59_949_125_229_827_907, 445_414_481_183_620_465,
                838_774_276_221_860_879, 638_038_608_439_546_039, 383_007_267_126_509_345,
            ])
        XCTAssertEqual(
            DeepSeekV41NgramHashState.hashMultipliers(
                layerIds: [1], maxNgramSize: 3, vocabularySize: 8),
            [[949_203_457_676_829_153, 59_949_125_229_827_907, 445_414_481_183_620_465]])
    }

    func testHashIdsMatchUpstream() throws {
        let state = try hashState()
        var history: MLXArray?
        let out = state(
            MLXArray([Int32(5), 1, 9, 2, 0]).reshaped(1, 5), startPosition: 0,
            history: &history)
        XCTAssertEqual(out.shape, [1, 5, 2, 4])
        XCTAssertEqual(
            out.asType(.int32).asArray(Int32.self),
            [
                27, 135, 280, 325, 35, 184, 248, 385,
                0, 139, 206, 406, 65, 171, 294, 371,
                20, 200, 205, 413, 44, 125, 347, 398,
                32, 124, 260, 400, 100, 153, 265, 413,
                53, 131, 236, 344, 82, 220, 277, 381,
            ])
    }

    /// The history carries across the prefill/decode split, so a token decoded at
    /// position 5 still looks back into the prompt.
    func testDecodeStepContinuesTheHistory() throws {
        let state = try hashState()
        var history: MLXArray?
        _ = state(
            MLXArray([Int32(5), 1, 9, 2, 0]).reshaped(1, 5), startPosition: 0,
            history: &history)
        XCTAssertNotNil(history)
        let out = state(
            MLXArray([Int32(7)]).reshaped(1, 1), startPosition: 5, history: &history)
        XCTAssertEqual(
            out.asType(.int32).asArray(Int32.self), [28, 152, 306, 365, 38, 130, 339, 503])
    }

    /// Look-back stops at a dead token, so no n-gram spans an image placeholder.
    func testDeadTokensBlockTheLookback() throws {
        let state = try hashState()
        var history: MLXArray?
        let out = state(
            MLXArray([Int32(5), 1, 9, 2, 0]).reshaped(1, 5), startPosition: 0,
            history: &history,
            tokenMask: MLXArray([true, true, false, true, true]).reshaped(1, 5))
        XCTAssertEqual(
            out.asType(.int32).asArray(Int32.self),
            [
                27, 135, 280, 325, 35, 184, 248, 385,
                0, 139, 206, 406, 65, 171, 294, 371,
                40, 196, 214, 384, 88, 137, 303, 490,
                40, 196, 214, 384, 88, 137, 303, 490,
                53, 131, 231, 382, 82, 220, 310, 461,
            ])
    }

    /// The padding token is the *compressed* id of `engram_pad_token_id`, not the id
    /// itself, and it is what a blocked look-back reads.
    func testPaddingUsesTheCompressedId() throws {
        let shuffled = [3, 1, 7, 0, 4, 5, 6, 2, 3, 1]
        let state = try DeepSeekV41NgramHashState(
            layout: try layout(), tokenMap: shuffled, padTokenId: 2,
            compressedVocabularySize: 8)
        XCTAssertEqual(state.padId, 7)

        var history: MLXArray?
        let out = state(
            MLXArray([Int32(5), 1, 9, 2, 0]).reshaped(1, 5), startPosition: 0,
            history: &history)
        XCTAssertEqual(
            out.asType(.int32).asArray(Int32.self),
            [
                26, 154, 239, 411, 105, 120, 367, 486,
                0, 139, 247, 340, 65, 171, 320, 390,
                20, 200, 205, 413, 44, 125, 347, 398,
                28, 182, 302, 390, 36, 221, 244, 474,
                18, 107, 295, 353, 67, 168, 321, 483,
            ])
    }

    /// No engram layers means no layout at all, and the compressed size the config
    /// declares has to be the size the map actually compresses to.
    func testLayoutAndMapAreValidated() throws {
        XCTAssertNil(
            DeepSeekV41EngramLayout(
                layerIds: [], maxNgramSize: 3, heads: 2, headDim: 4, vocabularySize: 100,
                numEmbeddings: []))
        XCTAssertThrowsError(
            try DeepSeekV41NgramHashState(
                layout: try layout(), tokenMap: Self.tokenMap, padTokenId: 2,
                compressedVocabularySize: 9))
        XCTAssertThrowsError(
            try DeepSeekV41NgramHashState(
                layout: try layout(), tokenMap: Self.tokenMap, padTokenId: 99,
                compressedVocabularySize: 8))
    }

    /// The real config's first and last primes, which is where a prime search that
    /// restarted per layer or reused a prime would show.
    func testReleasedLayoutPrimeEndpoints() throws {
        let layout = try XCTUnwrap(
            DeepSeekV41EngramLayout(
                layerIds: [1, 14], maxNgramSize: 4, heads: 8, headDim: 256,
                vocabularySize: 16_000_000, numEmbeddings: [384_006_168, 384_016_682]))
        XCTAssertEqual(layout.primes[0][0][0], 16_000_057)
        XCTAssertEqual(layout.primes.count, 2)
        XCTAssertEqual(layout.hashColumns, 24)
        let flat = layout.primes.flatMap { $0.flatMap { $0 } }
        XCTAssertEqual(flat.count, 48)
        XCTAssertEqual(Set(flat).count, 48, "no prime is handed out twice")
        XCTAssertEqual(flat, flat.sorted(), "the primes are drawn in increasing order")
    }

    // MARK: Gate

    private func gate() throws -> DeepSeekV41Engram {
        let layout = try XCTUnwrap(
            DeepSeekV41EngramLayout(
                layerIds: [1, 14], maxNgramSize: 3, heads: 2, headDim: 8,
                vocabularySize: 100, numEmbeddings: [64, 64]))
        let engram = DeepSeekV41Engram(
            layout: layout, layerId: 1, hiddenSize: 8, hcMultiplier: 2, rmsNormEps: 1e-6)
        engram.update(
            parameters: ModuleParameters.unflattened([
                (
                    "embed.weight",
                    (MLXArray(0 ..< 64 * 8).asType(.float32) % 13 - 6)
                        .reshaped(64, 8) / 13
                ),
                (
                    "wkv.weight",
                    (MLXArray(0 ..< 24 * 32).asType(.float32) % 5 - 2)
                        .reshaped(24, 32) / 8
                ),
                ("q_weight", MLXArray.full([2, 8], values: MLXArray(Float(0.75)))),
                ("k_weight", MLXArray.full([2, 8], values: MLXArray(Float(1.25)))),
            ]))
        return engram
    }

    private var streamInput: MLXArray {
        (MLXArray(0 ..< 1 * 3 * 2 * 8).asType(.float32) % 11 - 5).reshaped(1, 3, 2, 8) / 6
    }

    private var hashIds: MLXArray {
        MLXArray([Int32(1), 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15]).reshaped(1, 3, 4)
    }

    /// Upstream's output for the same fixed weights, including the FP8 round-trip the
    /// embeddings pass through before `wkv`.
    func testGateOutputMatchesUpstream() throws {
        let out = try gate()(streamInput, hashIds: hashIds)
        XCTAssertEqual(out.shape, [1, 3, 2, 8])
        assertClose(
            out[0, 0, 0],
            [
                -0.877_525, -0.679_601, -0.557_126, -0.251_417, -0.134_331, -0.044_192,
                0.153_732, 0.276_207,
            ])
        assertClose(
            out[0, 2, 1],
            [
                0.374_851, 0.412_351, 0.738_169, 0.971_726, -0.997_098, -0.625_149,
                -0.587_649, -0.261_831,
            ])
    }

    /// A masked position passes through untouched, which is what lets an image span skip
    /// the n-gram path entirely.
    func testMaskedPositionsPassThrough() throws {
        let x = streamInput
        let out = try gate()(
            x, hashIds: hashIds,
            tokenMask: MLXArray([true, false, true]).reshaped(1, 3))
        assertClose(out[0, 1, 0], x[0, 1, 0].asArray(Float.self))
        XCTAssertFalse(
            allClose(out[0, 0, 0], x[0, 0, 0], atol: 1e-4).item(),
            "an unmasked position should still be written to")
    }

    private func assertClose(
        _ actual: MLXArray, _ expected: [Float], file: StaticString = #filePath,
        line: UInt = #line
    ) {
        let values = actual.asType(.float32).asArray(Float.self)
        XCTAssertEqual(values.count, expected.count, file: file, line: line)
        for (index, pair) in zip(values, expected).enumerated() {
            XCTAssertEqual(pair.0, pair.1, accuracy: 2e-5, "index \(index)", file: file, line: line)
        }
    }
}
