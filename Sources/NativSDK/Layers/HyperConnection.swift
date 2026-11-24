import Foundation
import MLX
import MLXNN

/// A hyper-connection: the residual stream carries `multiplier` parallel copies, and each
/// block reads a learned mixture of them and writes back a learned redistribution.
///
/// `callAsFunction` returns the collapsed input the block should see, plus the two
/// coefficient sets ``hyperConnectionExpand(_:residual:post:comb:)`` needs to fold the
/// block's output back into the stream. The mixing matrix is Sinkhorn-normalized, so the
/// iteration count is part of the architecture rather than a tolerance.
open class HyperConnection: Module {

    public let multiplier: Int
    public let rmsNormEps: Float
    public let eps: Float
    public let sinkhornIterations: Int

    @ModuleInfo public var fn: MLXArray
    @ModuleInfo public var base: MLXArray
    @ModuleInfo public var scale: MLXArray

    public init(
        hiddenSize: Int,
        multiplier: Int,
        rmsNormEps: Float,
        eps: Float,
        sinkhornIterations: Int
    ) {
        self.multiplier = multiplier
        self.rmsNormEps = rmsNormEps
        self.eps = eps
        self.sinkhornIterations = sinkhornIterations
        let mixDimensions = (2 + multiplier) * multiplier
        _fn.wrappedValue = MLXArray.zeros([mixDimensions, multiplier * hiddenSize])
        _base.wrappedValue = MLXArray.zeros([mixDimensions])
        _scale.wrappedValue = MLXArray.ones([3])
        super.init()
    }

    open func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let y = x.asType(.float32)
        let (pre, post, comb) = hyperConnectionSplitSinkhorn(
            mixes: hyperConnectionMixes(y, fn: fn, rmsNormEps: rmsNormEps),
            scale: scale, base: base, multiplier: multiplier,
            sinkhornIterations: sinkhornIterations, eps: eps)
        let collapsed = (pre[.ellipsis, .newAxis] * y).sum(axis: 2).asType(x.dtype)
        return (collapsed, post, comb)
    }
}

/// The normalized residual projected through `fn`, which is what the coefficients are
/// split out of.
public func hyperConnectionMixes(_ x: MLXArray, fn: MLXArray, rmsNormEps: Float)
    -> MLXArray
{
    MLXFast.rmsNorm(
        x.asType(.float32).flattened(start: -2, end: -1),
        weight: MLXArray.mlxNone,
        eps: rmsNormEps
    ).matmul(fn.T)
}

/// The `pre` / `post` / `comb` coefficients of single-pass mHC.
///
/// `comb` is doubly normalized by Sinkhorn iteration; the first half-step is folded into
/// the softmax that precedes it, which is why the loop runs one fewer time.
/// One compiled split per distinct shape of the computation.
///
/// The multiplier and the iteration count decide the graph, and the eps is folded into
/// it, so each combination needs its own trace: `compile` freezes a captured Swift
/// scalar at the first call and would otherwise keep it for every later one. A model
/// holds these fixed, so a process sees one or two entries.
private struct SinkhornShape: Hashable {
    let multiplier: Int
    let iterations: Int
    let eps: Float
}

nonisolated(unsafe) private var sinkhornCache:
    [SinkhornShape:
        @Sendable (MLXArray, MLXArray, MLXArray) -> (
            MLXArray, MLXArray, MLXArray
        )] = [:]
private let sinkhornCacheLock = NSLock()

private func sinkhornSplit(_ shape: SinkhornShape)
    -> @Sendable (MLXArray, MLXArray, MLXArray) -> (MLXArray, MLXArray, MLXArray)
{
    sinkhornCacheLock.withLock {
        if let cached = sinkhornCache[shape] { return cached }
        let compiled:
            @Sendable (MLXArray, MLXArray, MLXArray) -> (
                MLXArray, MLXArray, MLXArray
            ) = compile { mixes, scale, base in
                splitSinkhornOps(
                    mixes: mixes, scale: scale, base: base, multiplier: shape.multiplier,
                    sinkhornIterations: shape.iterations, eps: shape.eps)
            }
        sinkhornCache[shape] = compiled
        return compiled
    }
}

public func hyperConnectionSplitSinkhorn(
    mixes: MLXArray, scale: MLXArray, base: MLXArray, multiplier: Int,
    sinkhornIterations: Int, eps: Float
) -> (pre: MLXArray, post: MLXArray, comb: MLXArray) {
    let split = sinkhornSplit(
        .init(multiplier: multiplier, iterations: sinkhornIterations, eps: eps))
    let (pre, post, comb) = split(mixes, scale, base)
    return (pre, post, comb)
}

private func splitSinkhornOps(
    mixes: MLXArray, scale: MLXArray, base: MLXArray, multiplier: Int,
    sinkhornIterations: Int, eps: Float
) -> (pre: MLXArray, post: MLXArray, comb: MLXArray) {
    let mixes = mixes.asType(.float32)
    let scale = scale.asType(.float32)
    let base = base.asType(.float32)
    let h = multiplier
    let pre = sigmoid(mixes[.ellipsis, ..<h] * scale[0] + base[..<h]) + eps
    let post =
        2 * sigmoid(mixes[.ellipsis, h ..< (2 * h)] * scale[1] + base[h ..< (2 * h)])
    var comb = mixes[.ellipsis, (2 * h)...].reshaped(mixes.shape.dropLast() + [h, h])
    comb = comb * scale[2] + base[(2 * h)...].reshaped(h, h)
    comb = softmax(comb, axis: -1, precise: true) + eps
    comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
    for _ in 0 ..< max(sinkhornIterations - 1, 0) {
        comb = comb / (comb.sum(axis: -1, keepDims: true) + eps)
        comb = comb / (comb.sum(axis: -2, keepDims: true) + eps)
    }
    return (pre, post, comb)
}

/// Folds a block's output back into the multi-stream residual.
public func hyperConnectionExpand(
    _ x: MLXArray,
    residual: MLXArray,
    post: MLXArray,
    comb: MLXArray,
)
    -> MLXArray
{
    let y =
        post[.ellipsis, .newAxis] * x[0..., 0..., .newAxis, 0...].asType(.float32)
        + comb.swappedAxes(-1, -2).matmul(residual.asType(.float32))
    return y.asType(x.dtype)
}
