import Foundation
import MLX
import MLXNN
import XCTest

@testable import NativSDK

/// Expectations captured from upstream mlx-vlm's Compressor.
final class DeepSeekV41CompressorTests: XCTestCase {
    private func configuration() throws -> DeepSeekV41Configuration {
        try JSONDecoder().decode(
            DeepSeekV41Configuration.self,
            from: Data(
                #"""
                {
                  "hidden_size": 8, "head_dim": 4, "num_hidden_layers": 4,
                  "compress_ratios": [0, 1, 3, 3], "rms_norm_eps": 1e-6
                }
                """#.utf8))
    }

    private func ramp(_ shape: [Int], scale: Float = 1) -> MLXArray {
        let count = shape.reduce(1, *)
        return (MLXArray(0 ..< count).asType(.float32) % 13 - 6).reshaped(shape) / 11
            * scale
    }

    private func compressor(layer: Int) throws -> DeepSeekV41Language.Compressor {
        let compressor = DeepSeekV41Language.Compressor(
            try configuration(), layerIndex: layer)
        var updates: [(String, MLXArray)] = [
            ("wkv.weight", ramp([4, 8])),
            ("norm.weight", MLXArray([Float(1), 0.5, 1.5, 2])),
        ]
        if compressor.wgate != nil {
            updates.append(("wgate.weight", ramp([4, 8], scale: -0.5)))
        }
        compressor.update(parameters: ModuleParameters.unflattened(updates))
        return compressor
    }

    /// Ratio 1 is a plain projection: no gate, no promotion, one latent per token.
    func testRatioOneIsAPlainProjection() throws {
        let compressor = try compressor(layer: 1)
        XCTAssertNil(compressor.wgate)
        let state = DeepSeekV41CompressorState(layers: 4)
        let out = try XCTUnwrap(
            compressor(ramp([1, 7, 8]), startPosition: 0, state: state))
        XCTAssertEqual(out.shape, [1, 7, 4])
        assertClose(out[0, 0], [1.542_32, -0.569_961, 0.804_74, -0.368_694])
        XCTAssertNil(state.kvState[1], "ratio 1 holds nothing back")
    }

    /// Seven tokens at ratio 3 give two latents, and the odd token is held for the next
    /// call rather than pooled into a short group.
    func testPrefillPoolsAndHoldsTheRemainder() throws {
        let compressor = try compressor(layer: 2)
        let state = DeepSeekV41CompressorState(layers: 4)
        let out = try XCTUnwrap(
            compressor(ramp([1, 7, 8]), startPosition: 0, state: state))
        XCTAssertEqual(out.shape, [1, 2, 4])
        assertClose(out[0, 0], [0.019_419, -0.723_27, -0.038_627, -2.761_349])
        assertClose(out[0, 1], [-0.170_644, -0.959_487, -0.518_266, -0.821_454])
        assertClose(
            try XCTUnwrap(state.kvState[2])[0, 0],
            [-0.512_162, 0.776_308, -0.512_156, 0.239_595])
    }

    /// A prefill that divides evenly holds nothing, and one shorter than the ratio yields
    /// no latent at all.
    func testExactAndShortPrefills() throws {
        let compressor = try compressor(layer: 2)
        let exact = DeepSeekV41CompressorState(layers: 4)
        let out = try XCTUnwrap(
            compressor(ramp([1, 7, 8])[0..., 0 ..< 6], startPosition: 0, state: exact))
        XCTAssertEqual(out.shape, [1, 2, 4])
        XCTAssertNil(exact.kvState[2])

        let short = DeepSeekV41CompressorState(layers: 4)
        XCTAssertNil(
            compressor(ramp([1, 7, 8])[0..., 0 ..< 2], startPosition: 0, state: short))
    }

    /// Decode yields a latent only on the step that completes a group, and the group it
    /// completes spans the tokens held from the prefill.
    func testDecodeYieldsOnlyOnGroupBoundaries() throws {
        let compressor = try compressor(layer: 2)
        let state = DeepSeekV41CompressorState(layers: 4)
        _ = compressor(ramp([1, 7, 8]), startPosition: 0, state: state)

        XCTAssertNil(
            compressor(ramp([1, 1, 8]), startPosition: 7, state: state),
            "position 7 does not complete the group")
        let completed = try XCTUnwrap(
            compressor(ramp([1, 1, 8], scale: -1), startPosition: 8, state: state))
        XCTAssertEqual(completed.shape, [1, 1, 4])
        assertClose(completed[0, 0], [-1.662_341, 0.170_733, -1.511_993, 0.644_62])
    }

    /// A decode that begins mid-group with no state pools a group whose first slot was
    /// never written. The -inf initialization is what keeps that slot out of the gate's
    /// softmax; a zero-filled score would give it weight and dilute the rest.
    func testUnwrittenSlotsStayOutOfTheGate() throws {
        let compressor = try compressor(layer: 2)
        let state = DeepSeekV41CompressorState(layers: 4)
        XCTAssertNil(compressor(ramp([1, 1, 8]), startPosition: 7, state: state))
        let completed = try XCTUnwrap(
            compressor(ramp([1, 1, 8], scale: -1), startPosition: 8, state: state))
        assertClose(completed[0, 0], [-1.735_999, -0.484_152, -0.328_162, -0.051_968])
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
