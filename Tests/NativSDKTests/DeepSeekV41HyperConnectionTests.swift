import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

/// Expectations captured from upstream mlx-vlm's language.py.
final class DeepSeekV41HyperConnectionTests: XCTestCase {
    private func ramp(_ shape: [Int], scale: Float = 1, offset: Float = 0) -> MLXArray {
        let count = shape.reduce(1, *)
        return
            ((MLXArray(0 ..< count).asType(.float32) % 13 - 6).reshaped(shape) / 11
            + offset) * scale
    }

    private var residual: MLXArray { ramp([1, 2, 4, 6]) }

    /// The first copy passes through and the rest start at zero.
    func testIdentityPreMix() {
        let mix = DeepSeekV41Language.identityPreMix(batch: 1, length: 2, multiplier: 4)
        XCTAssertEqual(mix.shape, [1, 2, 4])
        XCTAssertEqual(
            mix[0].flattened().asArray(Float.self), [1, 0, 0, 0, 1, 0, 0, 0])
    }

    func testCollapseMatchesUpstream() {
        let out = DeepSeekV41Language.hyperConnectionPre(
            residual, preMix: ramp([1, 2, 4], offset: 1))
        XCTAssertEqual(out.shape, [1, 2, 6])
        assertClose(
            out[0, 0],
            [0.033_058, -0.504_132, -0.289_256, -0.074_38, 0.140_496, 0.355_372])
    }

    /// The residual mix is a multiply-then-sum, which upstream chose over a matmul
    /// because a matmul fuses the products into its accumulation.
    func testExpandMatchesUpstream() {
        let out = DeepSeekV41Language.hyperConnectionExpand(
            ramp([1, 2, 6], scale: 0.5), residual: residual,
            post: ramp([1, 2, 4], offset: 1.5), comb: ramp([1, 2, 4, 4], offset: 2) / 8)
        XCTAssertEqual(out.shape, [1, 2, 4, 6])
        assertClose(
            out[0, 0, 0],
            [-0.239_669, -0.427_686, -0.293_388, -0.159_091, -0.024_793, 0.109_504])
        assertClose(
            out[0, 1, 3],
            [0.065_083, 0.226_24, 0.091_942, 0.024_793, 0.185_95, 0.347_107])
    }

    /// `comb` is doubly normalized by Sinkhorn iteration, so its rows and columns both
    /// approach one; `pre` is a shifted sigmoid and `post` is twice one.
    func testMixCoefficientsMatchUpstream() {
        let (pre, post, comb) = DeepSeekV41Language.hyperConnectionMixCoefficients(
            residual, fn: ramp([24, 24]), scale: MLXArray([Float(1), 0.75, 1.25]),
            base: ramp([24], scale: 0.5), multiplier: 4, sinkhornIterations: 3,
            eps: 1e-6, normEps: 1e-6)
        assertClose(pre[0, 0], [0.999_434, 0.683_465, 0.036_412, 0.009_364])
        assertClose(post[0, 0], [0.142_926, 1.195_506, 1.967_49, 1.911_633])
        assertClose(comb[0, 0, 0], [0.137_313, 0.010_035, 0.760_734, 0.162_423])
        assertClose(comb[0, 1, 3], [0.126_047, 0.974_712, 0.000_531, 0.000_025])
    }

    /// `pre` carries an eps floor so a saturated sigmoid still leaves a strictly
    /// positive weight on every copy; without it a copy can be switched off entirely.
    func testPreCoefficientIsFloored() {
        let (pre, _, comb) = DeepSeekV41Language.hyperConnectionMixCoefficients(
            residual, fn: MLXArray.zeros([24, 24]),
            scale: MLXArray([Float(1), 1, 1]),
            base: MLXArray.full([24], values: MLXArray(Float(-60))), multiplier: 4,
            sinkhornIterations: 3, eps: 1e-6, normEps: 1e-6)
        XCTAssertTrue(
            pre.asArray(Float.self).allSatisfy { $0 > 0 },
            "a saturated sigmoid must still leave a positive weight")
        XCTAssertEqual(pre[0, 0, 0].item(Float.self), 1e-6, accuracy: 1e-12)
        XCTAssertTrue(comb.asArray(Float.self).allSatisfy { $0 > 0 })

        var peaked = [Float](repeating: 0, count: 8)
        for row in 0 ..< 4 {
            peaked += [0] + [Float](repeating: -200, count: 3)
            _ = row
        }
        let (_, _, peakedComb) = DeepSeekV41Language.hyperConnectionMixCoefficients(
            residual, fn: MLXArray.zeros([24, 24]),
            scale: MLXArray([Float(1), 1, 1]), base: MLXArray(peaked), multiplier: 4,
            sinkhornIterations: 3, eps: 1e-6, normEps: 1e-6)
        XCTAssertTrue(
            peakedComb.asArray(Float.self).allSatisfy { $0 > 0 },
            "a peaked softmax must still leave every entry positive")
    }

    /// The head keeps its weight and its output in FP32 whatever the stream's dtype.
    func testParallelHeadIsFP32() throws {
        let config = try JSONDecoder().decode(
            DeepSeekV41Configuration.self,
            from: Data(#"{"vocab_size": 5, "hidden_size": 4}"#.utf8))
        let head = DeepSeekV41Language.ParallelHead(config)
        XCTAssertEqual(head.weight.dtype, .float32)
        head.update(
            parameters: ModuleParameters.unflattened([("weight", ramp([5, 4]))]))
        let out = head(ramp([1, 2, 4]).asType(.bfloat16))
        XCTAssertEqual(out.dtype, .float32)
        XCTAssertEqual(out.shape, [1, 2, 5])
    }

    private func assertClose(
        _ actual: MLXArray, _ expected: [Float], accuracy: Float = 2e-5,
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
