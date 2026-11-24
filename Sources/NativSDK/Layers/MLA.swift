import Foundation
import MLX
import MLXNN

/// Query count below which folding into the query beats materializing K/V.
///
/// Per head, with `r = kvLoraRank` and `d = qkNopeHeadDim + vHeadDim`, the absorbed path
/// costs `L*r*d + 2*L*T*r` and materializing costs `T*r*d + L*T*d`, where `T` is the
/// post-update cache length. The absorbed path wins while
/// `L < r*d / (r*d/T + 2*r - d)`.
///
/// With `cacheLength` omitted this is the `T -> inf` limit, `r*d / (2*r - d)`, which is the
/// right answer once the cache is much larger than the latent. Pass `cacheLength` for the
/// exact bound: on a cold cache, where `T == L`, it correctly falls below `L`, because
/// materializing K and V is cheaper there for every current model.
public func maxAbsorbedQueries(
    kvLoraRank: Int,
    qkNopeHeadDim: Int,
    vHeadDim: Int,
    cacheLength: Int? = nil
) -> Int {
    let d = qkNopeHeadDim + vHeadDim
    var denominator = Double(2 * kvLoraRank - d)
    if let cacheLength, cacheLength > 0 {
        denominator += Double(kvLoraRank * d) / Double(cacheLength)
    }
    guard denominator > 0 else { return 1 }
    return max(1, Int(Double(kvLoraRank * d) / denominator))
}

/// Length of a cached latent, tolerating the quantized form where it arrives as a tuple.
public func latentLength(_ latent: MLXArray) -> Int {
    latent.dim(-2)
}

public func latentLength(_ latent: (MLXArray, MLXArray, MLXArray?)) -> Int {
    latent.0.dim(-2)
}

/// One weight per head, applied as a batched matmul.
///
/// Not `final`, and its quantized form below is a subclass rather than a sibling: a
/// `@ModuleInfo` slot declared `MultiLinear` can only be filled with something that *is* a
/// `MultiLinear`, which is why mlx-swift makes `QuantizedLinear` a subclass of `Linear`.
/// As siblings, a model that declares the concrete type -- kimi_k3 does -- cannot load a
/// quantized checkpoint at all: the update fails with `needModuleInfo(... maybe type
/// mismatch: QuantizedMultiLinear, unableToCast("MultiLinear"))`. The sibling shape is
/// also why the callers that type the slot `Module` have to dispatch by hand; with the
/// override in place those paths keep working through dynamic dispatch.
public class MultiLinear: Module, Quantizable {
    @ModuleInfo public var weight: MLXArray

    public init(inputDimensions: Int, outputDimensions: Int, numHeads: Int) {
        let scale = sqrt(1.0 / Float(inputDimensions))
        self._weight.wrappedValue = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numHeads, outputDimensions, inputDimensions]
        )
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, transpose: Bool = true) -> MLXArray {
        if transpose {
            x.matmul(weight.swappedAxes(-1, -2))
        } else {
            x.matmul(weight)
        }
    }

    public func toQuantized(
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode = .affine
    ) -> Module {
        QuantizedMultiLinear(
            weight: weight,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }

    /// For a subclass supplying weights it has already prepared.
    fileprivate init(preparedWeight: MLXArray) {
        self._weight.wrappedValue = preparedWeight
        super.init()
    }
}

public final class QuantizedMultiLinear: MultiLinear, Quantized {
    public let groupSize: Int
    public let bits: Int
    public let mode: QuantizationMode

    @ModuleInfo public var scales: MLXArray
    @ModuleInfo public var biases: MLXArray?

    public convenience init(
        inputDimensions: Int,
        outputDimensions: Int,
        numHeads: Int,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode = .affine
    ) {
        let scale = sqrt(1.0 / Float(inputDimensions))
        let weight = MLXRandom.uniform(
            low: -scale,
            high: scale,
            [numHeads, outputDimensions, inputDimensions]
        )
        self.init(
            weight: weight,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }

    public init(
        weight: MLXArray,
        groupSize: Int,
        bits: Int,
        mode: QuantizationMode = .affine
    ) {
        let values = MLX.quantized(
            weight,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
        self.groupSize = groupSize
        self.bits = bits
        self.mode = mode
        self._scales.wrappedValue = values.scales
        self._biases.wrappedValue = values.biases
        super.init(preparedWeight: values.wq)
        freeze()
    }

    public override func callAsFunction(_ x: MLXArray, transpose: Bool = true) -> MLXArray {
        quantizedMM(
            x,
            weight,
            scales: scales,
            biases: biases,
            transpose: transpose,
            groupSize: groupSize,
            bits: bits,
            mode: mode
        )
    }
}
