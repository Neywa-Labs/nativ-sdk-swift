import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

/// Expectations captured from upstream mlx-vlm's ViT and Aligner.
final class DeepSeekV41VisionTests: XCTestCase {
    private func configuration() throws -> DeepSeekV41Configuration {
        try JSONDecoder().decode(
            DeepSeekV41Configuration.self,
            from: Data(
                #"""
                {
                  "hidden_size": 16, "num_hidden_layers": 2,
                  "compress_ratios": [0, 1],
                  "vision_config": {
                    "hidden_size": 8, "num_heads": 2, "intermediate_size": 8,
                    "patch_size": 2, "num_layers": 2, "rope_theta": 10000.0,
                    "downsample_ratio": 2
                  }
                }
                """#.utf8))
    }

    private func ramp(_ shape: [Int], scale: Float = 1) -> MLXArray {
        let count = shape.reduce(1, *)
        return (MLXArray(0 ..< count).asType(.float32) % 13 - 6).reshaped(shape) / 11
            * scale
    }

    /// One row per grid position, its height and width coordinates interleaved across
    /// the frequency axis — so position 5 of a 2x3 grid is (h=1, w=2).
    func testRopeTableMatchesUpstream() {
        let (cos, sin) = DeepSeekV41Vision.cosSin(
            height: 2, width: 3, dimensions: 2, theta: 10000)
        XCTAssertEqual(cos.shape, [6, 1, 2])
        assertClose(cos[5, 0], [0.540_302, -0.416_147], label: "cos")
        assertClose(sin[5, 0], [0.841_471, 0.909_297], label: "sin")
    }

    private func tower() throws -> DeepSeekV41Vision.ViT {
        let vit = DeepSeekV41Vision.ViT(try configuration())
        var updates: [(String, MLXArray)] = [
            ("patch_embed.proj.weight", ramp([8, 12], scale: 0.4)),
            ("patch_embed.proj.bias", ramp([8], scale: 0.1)),
            ("norm.weight", ramp([8]) + 1),
        ]
        for index in 0 ..< 2 {
            let p = "blocks.\(index)"
            updates += [
                ("\(p).norm1.weight", ramp([8]) + 1),
                ("\(p).norm2.weight", ramp([8], scale: 0.5) + 1),
                ("\(p).attn.wqkv.weight", ramp([24, 8], scale: 0.3)),
                ("\(p).attn.wqkv.bias", ramp([24], scale: 0.05)),
                ("\(p).attn.wo.weight", ramp([8, 8], scale: 0.25)),
                ("\(p).attn.wo.bias", ramp([8], scale: 0.02)),
                ("\(p).mlp.w1.weight", ramp([16, 8], scale: 0.2)),
                ("\(p).mlp.w2.weight", ramp([8, 8], scale: 0.15)),
            ]
        }
        vit.update(parameters: ModuleParameters.unflattened(updates))
        return vit
    }

    private func aligner() throws -> DeepSeekV41Vision.Aligner {
        let aligner = DeepSeekV41Vision.Aligner(try configuration())
        aligner.update(
            parameters: ModuleParameters.unflattened([
                ("w1.weight", ramp([16, 32], scale: 0.3)),
                ("w1.bias", ramp([16], scale: 0.05)),
                ("w2.weight", ramp([16, 16], scale: 0.2)),
                ("w2.bias", ramp([16], scale: 0.03)),
            ]))
        return aligner
    }

    /// Six patches of a 2x3 grid through two bidirectional blocks.
    func testTowerMatchesUpstream() throws {
        let features = try tower()(ramp([6, 3, 2, 2], scale: 0.8), height: 2, width: 3)
        XCTAssertEqual(features.shape, [6, 8])
        assertClose(
            features[0, 0 ..< 6],
            [0.796_777, 0.379_261, 0.056_552, -0.437_9, -0.454_444, -1.108_507],
            label: "token 0")
        assertClose(
            features[5, 0 ..< 6],
            [-0.479_196, -0.423_246, -0.129_749, 0.161_49, 0.989_897, 1.665_834],
            label: "token 5")
    }

    /// A 2x3 grid at ratio 2 rounds up to 1x2 blocks, so the trailing column is
    /// zero-filled rather than dropped and two embeddings come out.
    func testAlignerMatchesUpstream() throws {
        let features = try tower()(ramp([6, 3, 2, 2], scale: 0.8), height: 2, width: 3)
        let out = try aligner()(features, height: 2, width: 3)
        XCTAssertEqual(out.shape, [2, 16])
        assertClose(
            out[0, 0 ..< 6],
            [-0.044_347, 0.020_504, 0.018_617, -0.012_267, -0.123_743, -0.025_718],
            label: "embedding 0")
        assertClose(
            out[1, 0 ..< 6],
            [-0.016_659, -0.000_519, -0.048_324, -0.010_61, 0.005_023, -0.003_147],
            label: "embedding 1")
    }

    /// An exact grid needs no padding, and the output count is the rounded-up block
    /// count either way.
    func testAlignerGridRounding() throws {
        let aligner = try aligner()
        let exact = try tower()(ramp([4, 3, 2, 2], scale: 0.8), height: 2, width: 2)
        XCTAssertEqual(aligner(exact, height: 2, width: 2).shape, [1, 16])
        let wide = try tower()(ramp([12, 3, 2, 2], scale: 0.8), height: 3, width: 4)
        XCTAssertEqual(aligner(wide, height: 3, width: 4).shape, [4, 16])
    }

    private func assertClose(
        _ actual: MLXArray, _ expected: [Float], label: String, accuracy: Float = 2e-5,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        let values = actual.asType(.float32).asArray(Float.self)
        XCTAssertEqual(values.count, expected.count, label, file: file, line: line)
        for (index, pair) in zip(values, expected).enumerated() {
            XCTAssertEqual(
                pair.0, pair.1, accuracy: accuracy, "\(label) index \(index)", file: file,
                line: line)
        }
    }
}
