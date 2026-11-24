import Foundation
import MLX
import MLXNN
import Testing

@testable import NativSDK

/// Inkling's short convolutions carry state between steps, which is what makes the model
/// hybrid: a rejected speculative block has to put that state back, not just trim a KV
/// cache. These pin the state arithmetic and the fused decode kernel against the path it
/// replaces.
@Suite("Inkling short convolution")
struct InklingShortConvolutionTests {
    /// Values captured from mlx-vlm's `_next_conv_state`.
    @Test("The state advance matches mlx-vlm")
    func stateAdvance() {
        let state = MLXArray(
            [Float(-1), -2, -3, -4, -5, -6], [2, 3, 1])
        let inputs = MLXArray([Float(1), 2, 3, 4, 5, 6, 7, 8], [2, 4, 1])

        let unmasked = nextInklingConvolutionState(
            state: state, inputs: inputs, mask: nil)
        eval(unmasked)
        #expect(unmasked.reshaped(-1).asArray(Float.self) == [2, 3, 4, 6, 7, 8])

        // Row 1 has only two valid positions, so its window reaches one back into the
        // state rather than folding padding in.
        let mask = MLXArray([true, true, true, true, false, false, true, true], [2, 4])
        let masked = nextInklingConvolutionState(
            state: state, inputs: inputs, mask: mask)
        eval(masked)
        #expect(masked.reshaped(-1).asArray(Float.self) == [2, 3, 4, -6, 7, 8])
    }

    /// An empty window is a no-op rather than an error.
    @Test("A zero-width state is returned unchanged")
    func emptyState() {
        let state = MLXArray.zeros([1, 0, 4])
        let advanced = nextInklingConvolutionState(
            state: state, inputs: MLXArray.zeros([1, 3, 4]), mask: nil)
        #expect(advanced.shape == [1, 0, 4])
    }

    /// The fused decode kernel replaces the array path for short unmasked blocks, and has
    /// to produce the same values and leave the same state behind. Its rounding is what
    /// makes that non-trivial: the convolution rounds to the layer dtype before the fp32
    /// residual add, and the layer residual is a second add on top.
    @Test("The fused kernel matches the path it replaces")
    func fusedMatchesUnfused() throws {
        let channels = 6
        let convolution = InklingShortConvolution(
            channels: channels, kernelSize: 4, convolutionIndex: 0)
        // A known weight, so the comparison is not against an all-zero convolution.
        convolution.update(
            parameters: .unflattened([
                "conv.weight": (MLXArray(0 ..< (channels * 4)).asType(.float32) / 20)
                    .reshaped(channels, 4, 1)
            ]))

        let x = (MLXArray(0 ..< (2 * 3 * channels)).asType(.float32) / 7)
            .reshaped(2, 3, channels)
        let residual = (MLXArray(0 ..< (2 * 3 * channels)).asType(.float32) / 11)
            .reshaped(2, 3, channels)

        for withResidual in [false, true] {
            let fusedCache = ArraysCache(size: 1)
            let arrayCache = ArraysCache(size: 1)
            let seed = (MLXArray(0 ..< (2 * 3 * channels)).asType(.float32) / 13)
                .reshaped(2, 3, channels)
            fusedCache[0] = seed
            arrayCache[0] = seed

            let fused = convolution(
                x, cache: fusedCache, mask: nil,
                residual: withResidual ? residual : nil)
            // An all-true mask is the same window, and forces the array path.
            let unfused = convolution(
                x, cache: arrayCache,
                mask: MLXArray(Array(repeating: true, count: 6), [2, 3]),
                residual: withResidual ? residual : nil)
            eval(fused, unfused)

            let a = fused.reshaped(-1).asArray(Float.self)
            let b = unfused.reshaped(-1).asArray(Float.self)
            #expect(a.count == b.count)
            for (index, (lhs, rhs)) in zip(a, b).enumerated() {
                #expect(
                    abs(lhs - rhs) < 1e-4,
                    "residual \(withResidual)[\(index)]: \(lhs) != \(rhs)")
            }

            // Both leave the same window behind, so the next step continues identically.
            let fusedState = try #require(fusedCache[0])
            let arrayState = try #require(arrayCache[0])
            eval(fusedState, arrayState)
            #expect(fusedState.shape == arrayState.shape)
            for (lhs, rhs) in zip(
                fusedState.reshaped(-1).asArray(Float.self),
                arrayState.reshaped(-1).asArray(Float.self))
            {
                #expect(abs(lhs - rhs) < 1e-4)
            }
        }
    }

    /// The slot index is what lets one cache serve a layer's several convolutions.
    @Test("Each convolution writes its own slot")
    func slots() throws {
        let cache = ArraysCache(size: 2)
        let first = InklingShortConvolution(channels: 4, kernelSize: 4, convolutionIndex: 0)
        let second = InklingShortConvolution(channels: 4, kernelSize: 4, convolutionIndex: 1)

        _ = first(MLXArray.zeros([1, 2, 4]), cache: cache)
        #expect(cache[0] != nil)
        #expect(cache[1] == nil)

        _ = second(MLXArray.zeros([1, 2, 4]), cache: cache)
        #expect(cache[1] != nil)
        #expect(try #require(cache[1]).shape == [1, 3, 4])
    }
}
