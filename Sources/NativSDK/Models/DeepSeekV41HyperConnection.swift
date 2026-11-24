import Foundation
import MLX
import MLXNN

extension DeepSeekV41Language {
    /// Collapse the hc copies into one, weighted by `preMix`.
    static func hyperConnectionPre(_ x: MLXArray, preMix: MLXArray) -> MLXArray {
        (preMix[.ellipsis, .newAxis].asType(.float32) * x.asType(.float32))
            .sum(axis: 2)
            .asType(x.dtype)
    }

    /// Fold a sublayer's output back into the multi-copy residual.
    ///
    /// The residual mix is a multiply-then-sum rather than a matmul, which is upstream's
    /// choice and not an oversight: a matmul fuses the products into its accumulation and
    /// moves some BF16 residuals. DeepSeek-V4 does use the matmul form, so the shared
    /// helper is right there and wrong here.
    private static let compiledExpand:
        @Sendable (MLXArray, MLXArray, MLXArray, MLXArray) -> MLXArray =
            compile { x, residual, post, comb in
                let mixed =
                    (comb[.ellipsis, .newAxis]
                    * residual.asType(.float32)[.ellipsis, .newAxis, 0...])
                    .sum(axis: 2)
                return
                    (post[.ellipsis, .newAxis]
                    * x.asType(.float32)[0..., 0..., .newAxis, 0...] + mixed)
                    .asType(x.dtype)
            }

    static func hyperConnectionExpand(
        _ x: MLXArray, residual: MLXArray, post: MLXArray, comb: MLXArray
    ) -> MLXArray {
        compiledExpand(x, residual, post, comb)
    }

    /// The initial one-hot mix: the first copy passes through untouched.
    static func identityPreMix(batch: Int, length: Int, multiplier: Int) -> MLXArray {
        concatenated(
            [
                MLXArray.ones([batch, length, 1], dtype: .float32),
                MLXArray.zeros([batch, length, multiplier - 1], dtype: .float32),
            ], axis: -1)
    }

    /// The collapse coefficients for the *next* sublayer: pre, post and comb.
    static func hyperConnectionMixCoefficients(
        _ x: MLXArray, fn: MLXArray, scale: MLXArray, base: MLXArray, multiplier: Int,
        sinkhornIterations: Int, eps: Float, normEps: Float
    ) -> (pre: MLXArray, post: MLXArray, comb: MLXArray) {
        hyperConnectionSplitSinkhorn(
            mixes: hyperConnectionMixes(x, fn: fn, rmsNormEps: normEps),
            scale: scale, base: base, multiplier: multiplier,
            sinkhornIterations: sinkhornIterations, eps: eps)
    }

    /// Vocabulary projection kept in FP32 so the logits come out FP32 directly.
    final class ParallelHead: Module {
        @ParameterInfo(key: "weight") var weight: MLXArray

        init(_ config: DeepSeekV41Configuration) {
            _weight.wrappedValue = MLXArray.zeros(
                [config.vocabularySize, config.hiddenSize], dtype: .float32)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            matmul(x.asType(.float32), weight.T)
        }
    }
}
