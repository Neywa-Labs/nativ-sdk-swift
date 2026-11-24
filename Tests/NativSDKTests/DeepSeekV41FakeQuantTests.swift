import Foundation
import MLX
import XCTest

@testable import NativSDK

/// Every expectation here is the value upstream mlx-vlm's `fakequant.py` produces for the
/// same input, captured from its array path. They are written out rather
/// than recomputed so the test does not agree with itself.
final class DeepSeekV41FakeQuantTests: XCTestCase {
    /// A block whose amax is exactly 6 gets scale 1 under both FP4 paths, which makes the
    /// e2m1 grid itself visible in the output.
    private static let values: [Float] = [
        0, 0.3, 0.7, 1.2, 2.5, 5, 6, -0.4, -1.75, -3.5, -6, 1e-6, 0.25, 1.25, 0.75, 1.75,
    ]

    private func input(repeated count: Int = 1) -> MLXArray {
        MLXArray(Array(repeatElement(Self.values, count: count).joined()))
            .reshaped(1, 16 * count)
    }

    func testFP8RoundTripMatchesUpstream() {
        let out = DeepSeekV41FakeQuant.fp8UE8M0(input(repeated: 2), block: 32)
        XCTAssertEqual(
            out[0, 0 ..< 16].asArray(Float.self),
            [
                0, 0.3125, 0.6875, 1.25, 2.5, 5, 6, -0.40625, -1.75, -3.5, -6, 0, 0.25,
                1.25, 0.75, 1.75,
            ])
    }

    func testFP4RoundTripsMatchUpstream() {
        let grid: [Float] = [
            0, 0.5, 0.5, 1, 2, 4, 6, -0.5, -2, -4, -6, 0, 0, 1, 1, 2,
        ]
        XCTAssertEqual(
            DeepSeekV41FakeQuant.fp4UE8M0(input(repeated: 2), block: 32)[0, 0 ..< 16]
                .asArray(Float.self), grid)
        XCTAssertEqual(
            DeepSeekV41FakeQuant.fp4E4M3(input(), block: 16)[0].asArray(Float.self), grid)
    }

    /// The scale is per block, so an outlier in one block does not reach the other. Read
    /// as a single block the first half would come back quantized to zero.
    func testScalesAreIndependentPerBlock() {
        var values = Self.values
        values[8] = 600
        let mixed = MLXArray(Self.values + values).reshaped(1, 32)
        let out = DeepSeekV41FakeQuant.fp4UE8M0(mixed, block: 16)
        XCTAssertEqual(
            out[0, 0 ..< 8].asArray(Float.self), [0, 0.5, 0.5, 1, 2, 4, 6, -0.5])
        XCTAssertEqual(out[0, 16 ..< 24].asArray(Float.self), [Float](repeating: 0, count: 8))
    }

    /// `2**ceil(log2(amax / max))` by IEEE bit manipulation, which is how the reference's
    /// fast_round_scale computes it, so the boundary cases agree.
    func testUE8M0ScaleMatchesUpstream() {
        let amax = MLXArray([Float(1e-4), 0.5, 1, 6, 448, 600]).reshaped(6, 1)
        XCTAssertEqual(
            DeepSeekV41FakeQuant.ue8m0Scale(amax, maxInverse: 1 / 448).flattened()
                .asArray(Float.self),
            [2.384_185_8e-07, 0.001_953_125, 0.003_906_25, 0.015_625, 1, 2])
        XCTAssertEqual(
            DeepSeekV41FakeQuant.ue8m0Scale(amax, maxInverse: 1 / 6).flattened()
                .asArray(Float.self),
            [3.051_757_8e-05, 0.125, 0.25, 1, 128, 128])
    }

    /// The scale reads the exponent field of `amax * (1/max)`, so it is exact only if
    /// that product is. `compile` is not bit-exact — it evaluates `6 * (1/6)` as
    /// 1.0000002 rather than 1.0 — and two ulp there doubles the scale and shifts every
    /// value in the block by a factor of two. This pins the boundary so the scale path
    /// is not compiled by a later change.
    func testScaleIsExactAtAPowerOfTwoBoundary() {
        let amax = MLXArray([Float(6)]).reshaped(1, 1)
        XCTAssertEqual(
            DeepSeekV41FakeQuant.ue8m0Scale(amax, maxInverse: 1 / 6)[0, 0]
                .item(Float.self), 1, accuracy: 0)
        XCTAssertEqual(
            (amax * Float(1.0 / 6.0)).view(dtype: .uint32)[0, 0].item(UInt32.self),
            0x3F80_0000, "6 * (1/6) must be exactly 1.0")
    }

    /// Ties at .25/1.25/2.5/5 round down and ties at .75/1.75/3.5 round up, which is
    /// round-to-nearest-even on the e2m1 grid.
    func testE2M1TiesRoundToEven() {
        let ties = MLXArray([
            Float(0.25), 0.75, 1.25, 1.75, 2.5, 3.5, 5, 6,
            -0.25, -0.75, -1.25, -1.75, -2.5, -3.5, -5, 6,
        ]).reshaped(1, 16)
        XCTAssertEqual(
            DeepSeekV41FakeQuant.roundE2M1(ties)[0].asArray(Float.self),
            [0, 1, 1, 2, 2, 4, 4, 6, 0, -1, -1, -2, -2, -4, -4, 6])
    }

    /// A tiny block is floored before the scale is taken, so an all-but-zero block does
    /// not divide by zero, and the dtype the caller passed comes back.
    func testFloorsAndDtypeArePreserved() {
        let small = MLXArray([Float](repeating: 0.001, count: 32)).reshaped(1, 32)
        XCTAssertEqual(
            DeepSeekV41FakeQuant.fp4E4M3(small[0..., 0 ..< 16], block: 16)[0, 0]
                .item(Float.self), 0.000_976_562_5, accuracy: 1e-12)
        XCTAssertEqual(
            DeepSeekV41FakeQuant.fp4UE8M0(small, block: 32)[0, 0].item(Float.self),
            0.000_976_562_5, accuracy: 1e-12)

        for values in [[Float](repeating: 0, count: 32), [Float](repeating: 1e-30, count: 32)] {
            let block = MLXArray(values).reshaped(1, 32)
            for out in [
                DeepSeekV41FakeQuant.fp8UE8M0(block),
                DeepSeekV41FakeQuant.fp4UE8M0(block),
                DeepSeekV41FakeQuant.fp4E4M3(block, block: 16),
            ] {
                XCTAssertTrue(
                    out.asArray(Float.self).allSatisfy { $0.isFinite },
                    "a non-finite value came back for \(values[0])")
            }
        }
        let half = MLXArray(Self.values).reshaped(1, 16).asType(.bfloat16)
        XCTAssertEqual(DeepSeekV41FakeQuant.fp4E4M3(half, block: 16).dtype, .bfloat16)
        XCTAssertEqual(DeepSeekV41FakeQuant.fp8UE8M0(half, block: 16).dtype, .bfloat16)
    }
}
