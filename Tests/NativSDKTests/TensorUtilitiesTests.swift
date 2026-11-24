import MLX
import XCTest

@testable import NativSDK

final class TensorUtilitiesTests: XCTestCase {
    func testRotaryLayoutsPreserveShapeDtypeAndPairOrdering() {
        for dtype: DType in [.float32, .float16, .bfloat16] {
            // Transposing the leading dimensions also exercises noncontiguous inputs.
            let input = MLXArray(1 ... 24, [2, 3, 4]).asType(dtype).transposed(1, 0, 2)
            let values = input.asArray(Float.self)
            var splitExpected = [Float]()
            var interleavedExpected = [Float]()
            for offset in stride(from: 0, to: values.count, by: 4) {
                let a = values[offset]
                let b = values[offset + 1]
                let c = values[offset + 2]
                let d = values[offset + 3]
                splitExpected += [-c, -d, a, b]
                interleavedExpected += [-b, a, -d, c]
            }
            let split = RotaryEmbeddingOps.rotateHalf(input)
            let interleaved = RotaryEmbeddingOps.rotateInterleaved(input)
            for result in [split, interleaved] {
                XCTAssertEqual(result.shape, input.shape)
                XCTAssertEqual(result.dtype, dtype)
            }
            XCTAssertEqual(split.asArray(Float.self), splitExpected)
            XCTAssertEqual(interleaved.asArray(Float.self), interleavedExpected)
            XCTAssertEqual(
                RotaryEmbeddingOps.rotateHalf(split).asArray(Float.self), values.map { -$0 })
            XCTAssertEqual(
                RotaryEmbeddingOps.rotateInterleaved(interleaved).asArray(Float.self),
                values.map { -$0 })
        }
    }

    func testVisualFeaturesReplaceOnlyPlaceholdersForBothSupportedRanks() {
        for embeddingRank in [2, 3] {
            for featureRank in [2, 3] {
                let embeddings = MLXArray(0 ..< 10).asType(.float16)
                    .reshaped(embeddingRank == 2 ? [5, 2] : [1, 5, 2])
                let features = MLXArray([20, 21, 30, 31, 40, 41]).asType(.float16)
                    .reshaped(featureRank == 2 ? [3, 2] : [1, 3, 2])
                let result = MultimodalEmbedding.mergeInputIDsWithImageFeatures(
                    inputIDs: MLXArray([7, 100, 101, 8, 100], [1, 5]),
                    inputEmbeds: embeddings, imageFeatures: features,
                    imageTokenID: 100, videoTokenID: 101)
                XCTAssertEqual(result.shape, [1, 5, 2])
                XCTAssertEqual(result.dtype, .float16)
                XCTAssertEqual(
                    result.asArray(Float.self), [0, 1, 20, 21, 30, 31, 6, 7, 40, 41])
            }
        }
    }
}
