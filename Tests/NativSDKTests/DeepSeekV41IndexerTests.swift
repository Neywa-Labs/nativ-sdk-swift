import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

/// Expectations captured from upstream mlx-vlm's language.py helpers.
final class DeepSeekV41IndexerTests: XCTestCase {
    private static let row: [Float] = [3, 1, 9, 2, 5, 0.5, -Float.infinity, -Float.infinity, 4]

    private func rope(originalLength: Int = 64, factor: Float = 4)
        -> DeepSeekV41IndexRope
    {
        DeepSeekV41IndexRope(
            ropeDimensions: 8, theta: 160_000,
            yarn: .init(
                factor: factor, betaFast: 32, betaSlow: 1, originalLength: originalLength))
    }

    /// The YaRN ramp interpolates the low frequencies and leaves the high ones, and the
    /// phases are evaluated in double precision before rounding back.
    func testRopeTableMatchesUpstream() {
        let (cos, sin) = rope().table(length: 6)
        XCTAssertEqual(cos.shape, [6, 4])
        assertClose(
            cos[5],
            [0.283_662_199_974_060_06, 0.998_047_530_651_092_5, 0.999_995_112_419_128_4, 1],
            accuracy: 0)
        assertClose(
            sin[5],
            [
                -0.958_924_293_518_066_4, 0.062_459_316_104_650_5,
                0.003_124_994_691_461_324_7, 0.000_156_250_011_059_455_57,
            ], accuracy: 0)
    }

    /// An original length of zero disables the correction, as upstream's guard does.
    func testRopeCorrectionCanBeDisabled() {
        let (cos, _) = rope(originalLength: 0, factor: 4).table(length: 4)
        assertClose(
            cos[3],
            [
                -0.989_992_499_351_501_5, 0.988_771_080_970_764_2, 0.999_971_866_607_666,
                0.999_999_940_395_355_2,
            ], accuracy: 0)
    }

    /// The table is built to a multiple of 256, so a decode step that advances by one
    /// position reuses it instead of rebuilding.
    func testTableIsBuiltInBlocksOf256() {
        let table = rope()
        let (first, _) = table.table(length: 6)
        XCTAssertEqual(first.dim(0), 6)
        XCTAssertEqual(table.builtCapacity, 256, "six positions still build a block of 256")
        let (second, _) = table.table(length: 7)
        XCTAssertEqual(second.dim(0), 7)
        XCTAssertEqual(table.builtCapacity, 256, "and the next position reuses it")
        assertClose(second[5], first[5].asArray(Float.self), accuracy: 0)
        let (third, _) = table.table(length: 300)
        XCTAssertEqual(third.dim(0), 300)
        XCTAssertEqual(table.builtCapacity, 512)
    }

    /// Only the trailing `ropeDimensions` entries rotate; the rest pass through.
    func testIndexRotaryMatchesUpstream() {
        let (cos, sin) = rope().table(length: 6)
        let x = (MLXArray(0 ..< 12).asType(.float32) - 5).reshaped(1, 1, 12) / 4
        let out = DeepSeekV41IndexScores.applyRotary(
            x, cos: cos[2 ..< 3][.newAxis], sin: sin[2 ..< 3][.newAxis],
            ropeDimensions: 8)
        assertClose(
            out[0, 0],
            [
                -1.25, -1, -0.75, -0.5, 0.104_037, -0.227_324, 0.237_423, 0.506_093,
                0.748_749, 1.000_937, 1.249_906, 1.500_078,
            ])
    }

    /// Per-head scores are rectified, weighted and summed away; unreachable positions go
    /// to -inf.
    func testIndexScoresMatchUpstream() {
        let queries = ((MLXArray(0 ..< 24).asType(.float32) % 7) - 3).reshaped(1, 2, 3, 4) / 5
        let keys = ((MLXArray(0 ..< 20).asType(.float32) % 5) - 2).reshaped(1, 5, 4) / 3
        let weights = ((MLXArray(0 ..< 6).asType(.float32) % 4) - 1).reshaped(1, 2, 3) / 2

        let plain = DeepSeekV41IndexScores.scores(
            queries: queries, indexKeys: keys, weights: weights)
        assertClose(plain[0, 0], [-0.066_553, 0, 0, 0, 0.066_675])
        assertClose(plain[0, 1], [-0.133_268, 0.066_715, 1.065_82, 0.066_715, -0.199_902])

        let lengths = MLXArray([Int32(1), 2]).reshaped(2, 1)
        let masked = DeepSeekV41IndexScores.scores(
            queries: queries, indexKeys: keys, weights: weights, compressLengths: lengths)
        let first = masked[0, 0].asArray(Float.self)
        XCTAssertEqual(first[0], -0.066_553, accuracy: 1e-5)
        XCTAssertTrue(first[1...].allSatisfy { $0 == -.infinity })
        let second = masked[0, 1].asArray(Float.self)
        XCTAssertEqual(second[1], 0.066_715, accuracy: 1e-5)
        XCTAssertTrue(second[2...].allSatisfy { $0 == -.infinity })
    }

    /// The block holding the last reachable position is pinned, so it is kept even when
    /// a higher-scoring block exists — which is what keeps the newest keys visible.
    func testCandidateBlocksMatchUpstream() {
        let logits = MLXArray(Self.row).reshaped(1, 9)

        XCTAssertEqual(
            DeepSeekV41IndexScores.selectCandidateBlocks(
                logits: logits, compressLengths: .scalar(9), topKBlocks: 2, blockSize: 2
            )[0].asArray(Bool.self),
            [false, false, true, true, false, false, false, false, true])
        XCTAssertEqual(
            DeepSeekV41IndexScores.selectCandidateBlocks(
                logits: logits, compressLengths: .scalar(5), topKBlocks: 2, blockSize: 2
            )[0].asArray(Bool.self),
            [false, false, true, true, true, true, false, false, false])
        XCTAssertEqual(
            DeepSeekV41IndexScores.selectCandidateBlocks(
                logits: logits, compressLengths: .scalar(9), topKBlocks: 1, blockSize: 3
            )[0].asArray(Bool.self),
            [false, false, false, false, false, false, true, true, true])
        XCTAssertEqual(
            DeepSeekV41IndexScores.selectCandidateBlocks(
                logits: logits, compressLengths: .scalar(9), topKBlocks: 5, blockSize: 2
            )[0].asArray(Bool.self),
            [true, true, true, true, true, true, false, false, true])
    }

    /// The width divides the key axis as evenly as it can, and only once both thresholds
    /// are crossed. These are upstream's widths for the same counts.
    func testTileWidthMatchesUpstream() {
        let held = 1 << 27
        XCTAssertEqual(
            [9000, 8192, 4097, 12289, 100_000].map {
                DeepSeekV41IndexScores.tileWidth(keyCount: $0, heldElements: held)
            }, [3000, 4096, 2049, 3073, 4000])
        XCTAssertEqual(
            DeepSeekV41IndexScores.tileWidth(keyCount: 4096, heldElements: held), 4096)
        XCTAssertEqual(
            DeepSeekV41IndexScores.tileWidth(keyCount: 9000, heldElements: 1000), 9000)
    }

    /// During prefill each query row has its own reachable length.
    func testCandidateBlocksPerQuery() {
        let logits = MLXArray(Self.row + Self.row.reversed()).reshaped(2, 9)
        let out = DeepSeekV41IndexScores.selectCandidateBlocks(
            logits: logits, compressLengths: .perQuery(MLXArray([Int32(4), 9]).reshaped(2, 1)),
            topKBlocks: 2, blockSize: 2)
        XCTAssertEqual(
            out[0].asArray(Bool.self),
            [false, false, true, true, true, true, false, false, false])
        XCTAssertEqual(
            out[1].asArray(Bool.self),
            [false, false, false, false, false, false, true, true, true])
    }

    /// A key axis wide enough to tile must give the same scores as the untiled form —
    /// bit for bit, which is why the tile width is kept even. Both thresholds have to be
    /// crossed for tiling to engage: more than 4096 keys and more than 2^26 held scores,
    /// so the shape here is deliberately large enough to reach it.
    func testTiledScoresAgreeWithUntiled() {
        let keyCount = 9000
        let queries = MLXRandom.normal([1, 1024, 8, 8], key: MLXRandom.key(3))
        let keys = MLXRandom.normal([1, keyCount, 8], key: MLXRandom.key(4))
        let weights = MLXRandom.normal([1, 1024, 8], key: MLXRandom.key(5))
        XCTAssertGreaterThan(
            queries.dim(1) * queries.dim(2) * keyCount,
            DeepSeekV41IndexScores.tileMinimumElements,
            "this shape must cross the tiling threshold or the test proves nothing")

        let tiled = DeepSeekV41IndexScores.scores(
            queries: queries, indexKeys: keys, weights: weights)
        XCTAssertEqual(tiled.shape, [1, 1024, keyCount])

        let flat = matmul(queries.reshaped(1, -1, 8), keys.transposed(0, 2, 1))
        let untiled =
            (maximum(flat.reshaped(1, 1024, 8, keyCount), 0)
            * weights[.ellipsis, .newAxis]).sum(axis: 2)
        XCTAssertTrue(allClose(tiled, untiled, rtol: 0, atol: 0).item(), "tiling moved the scores")
    }

    private func assertClose(
        _ actual: MLXArray, _ expected: [Float], accuracy: Float = 2e-6,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let values = actual.asType(.float32).asArray(Float.self)
        XCTAssertEqual(values.count, expected.count, file: file, line: line)
        for (index, pair) in zip(values, expected).enumerated() {
            XCTAssertEqual(
                pair.0, pair.1, accuracy: accuracy, "index \(index)", file: file, line: line)
        }
    }
}
