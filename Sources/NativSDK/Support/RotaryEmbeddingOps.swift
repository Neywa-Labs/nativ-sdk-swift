import MLX

/// Rotation layouts used when applying precomputed rotary-position frequencies.
enum RotaryEmbeddingOps {
    /// Split the final dimension into halves: [a, b, c, d] -> [-c, -d, a, b].
    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    }

    /// Rotate adjacent pairs: [a, b, c, d] -> [-b, a, -d, c].
    static func rotateInterleaved(_ x: MLXArray) -> MLXArray {
        let lastDim = x.dim(-1)
        let even = x[.ellipsis, stride(from: 0, to: lastDim, by: 2)]
        let odd = x[.ellipsis, stride(from: 1, to: lastDim, by: 2)]
        return stacked([-odd, even], axis: -1).reshaped(x.shape)
    }
}
