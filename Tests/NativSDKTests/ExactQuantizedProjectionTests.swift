import MLX
import MLXNN
import XCTest

@testable import NativSDK

final class ExactQuantizedProjectionTests: XCTestCase {
    private final class AdaptedQuantizedLinear: QuantizedLinear {
        override func callAsFunction(_ x: MLXArray) -> MLXArray { super.callAsFunction(x) + 1 }
    }

    private func singleton(_ layer: Linear, _ x: MLXArray) -> MLXArray {
        concatenated(
            (0 ..< x.dim(0)).map { batch in
                concatenated(
                    (0 ..< x.dim(1)).map { token in
                        layer(x[batch ..< (batch + 1), token ..< (token + 1)])
                    }, axis: 1)
            }, axis: 0)
    }

    func testAllKernelLayoutsMatchSingletonQuantizedDecodeExactly() throws {
        for dtype: DType in [.bfloat16, .float16] {
            for bits in [4, 5, 8] {
                for group in [32, 64, 128] {
                    let values = withRandomState(MLXRandom.RandomState(seed: 117)) {
                        (
                            MLXRandom.normal([256, 512]).asType(dtype),
                            MLXRandom.normal([2, 18, 512]).asType(dtype)
                        )
                    }
                    let layer = QuantizedLinear(
                        weight: values.0, bias: nil, groupSize: group, bits: bits)
                    for length in [1, 2, 5, 6, 7, 8, 9, 15, 16] {
                        let x = values.1[0..., 1 ..< (length + 1)]
                        let actual = try XCTUnwrap(exactQuantizedProjection(layer, x))
                        XCTAssertEqual(
                            actual.asArray(Float.self), singleton(layer, x).asArray(Float.self),
                            "dtype=\(dtype), bits=\(bits), group=\(group), length=\(length)")
                    }
                }
            }
        }
    }

    func testFusedProjectionsMatchSeparateSingletons() throws {
        for bits in [4, 5, 8] {
            for dtype: DType in [.bfloat16, .float16] {
                let x = MLXRandom.normal([1, 8, 2560]).asType(dtype)
                let layers = [256, 128, 32, 32].map { size in
                    QuantizedLinear(
                        weight: MLXRandom.normal([size, 2560]).asType(dtype),
                        bias: nil, groupSize: 64, bits: bits)
                }
                for count in [2, 3, 4] {
                    for length in [2, 5, 8] {
                        let selected = Array(layers.prefix(count))
                        let input = x[0..., ..<length]
                        let outputs = try XCTUnwrap(exactQuantizedProjections(selected, input))
                        for (actual, layer) in zip(outputs, selected) {
                            XCTAssertEqual(
                                actual.asArray(Float.self),
                                singleton(layer, input).asArray(Float.self),
                                "bits=\(bits), count=\(count), length=\(length)")
                        }
                    }
                }
            }
        }
    }

    func testBiasAndUnsupportedLayoutsUseSafeFallbacks() throws {
        let x = MLXRandom.normal([1, 8, 512]).asType(.bfloat16)
        let layer = QuantizedLinear(
            weight: MLXRandom.normal([256, 512]).asType(.bfloat16),
            bias: MLXArray.ones([256], dtype: .bfloat16), groupSize: 64, bits: 4)
        XCTAssertEqual(
            try XCTUnwrap(exactQuantizedProjection(layer, x)).asArray(Float.self),
            singleton(layer, x).asArray(Float.self))
        XCTAssertNil(exactQuantizedProjections([layer, layer], x))
        XCTAssertNil(exactQuantizedProjection(layer, x.asType(.float32)))
        XCTAssertNil(
            exactQuantizedProjection(layer, MLXArray.zeros([1, 17, 512], dtype: .bfloat16)))
        for bits in [2, 3, 6] {
            let unsupported = QuantizedLinear(
                weight: MLXRandom.normal([256, 512]).asType(.bfloat16),
                bias: nil, groupSize: 64, bits: bits)
            XCTAssertNil(exactQuantizedProjection(unsupported, x))
            XCTAssertEqual(
                exactSpeculativeLinear(unsupported, x).asArray(Float.self),
                singleton(unsupported, x).asArray(Float.self))
        }
        let narrow = QuantizedLinear(
            weight: MLXArray.ones([24, 256], dtype: .bfloat16),
            bias: nil, groupSize: 64, bits: 4)
        XCTAssertNil(exactQuantizedProjection(narrow, x[0..., 0..., ..<256]))
        let adapted = AdaptedQuantizedLinear(
            weight: MLXArray.ones([256, 512], dtype: .bfloat16),
            bias: nil, groupSize: 64, bits: 4)
        XCTAssertNil(exactQuantizedProjection(adapted, x))
        XCTAssertNil(exactQuantizedProjections([adapted, adapted], x))
        XCTAssertEqual(
            exactSpeculativeLinear(adapted, x).asArray(Float.self),
            singleton(adapted, x).asArray(Float.self))
    }
}
