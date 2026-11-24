import Foundation
import MLX

/// Activation fake-quantization: the inference-time half of V4.1's QAT.
///
/// The reference *simulates* its training-time quantization on several activations —
/// they are rounded through FP8 or FP4 and written back before use. Skipping them
/// changes the numbers the model was trained to see. V4.1 applies it to the native
/// FP8/FP4 linear inputs, the window KV (over the whole head vector, rope tail
/// included), the compressed KV latents, and the indexer queries and keys.
///
/// Scale semantics follow the reference's `kernel.py`:
/// - ue8m0: `2**ceil(log2(amax * (1/max)))`, with `amax` clamped to a format-specific
///   floor first. The ceil is computed by IEEE bit manipulation, exactly as the
///   reference's `fast_round_scale` does, so boundary cases match.
/// - e4m3 scale: `amax` clamped to `6*2^-9`, then `amax/6` rounded to the nearest
///   e4m3 value.
///
/// FP4 rounding is round-to-nearest-even on the e2m1 grid
/// {0, .5, 1, 1.5, 2, 3, 4, 6}: ties at .25/1.25/2.5/5 round down, ties at
/// .75/1.75/3.5 round up.
public enum DeepSeekV41FakeQuant {
    public static let fp8Max: Float = 448
    public static let fp4Max: Float = 6

    static let e2m1Values: [Float] = [0, 0.5, 1, 1.5, 2, 3, 4, 6]

    /// `ceil(log2(x))` for positive normal fp32 `x`, via IEEE 754 bits.
    static func log2CeilBits(_ x: MLXArray) -> MLXArray {
        let bits = x.asType(.float32).view(dtype: .uint32)
        let exponent = ((bits >> 23) & 0xFF).asType(.int32) - 127
        let mantissa = (bits & 0x7F_FFFF).asType(.int32)
        return exponent + (mantissa .!= 0).asType(.int32)
    }

    /// `2**e` for integer `e`, via IEEE 754 bits.
    static func powerOfTwo(_ e: MLXArray) -> MLXArray {
        ((e + 127) << 23).asType(.uint32).view(dtype: .float32)
    }

    static func ue8m0Scale(_ amax: MLXArray, maxInverse: Float) -> MLXArray {
        powerOfTwo(log2CeilBits(amax * maxInverse))
    }

    /// Round `|v| <= 6` to the nearest e2m1 value, ties to even, keeping the sign.
    ///
    /// Compiled, which fuses its seven comparisons and the gather. Safe to trace only
    /// because it rounds a value it is handed: `compile` is not bit-exact — it turns
    /// `6 * (1/6)` into 1.0000002 rather than 1.0 — and the scale computations upstream
    /// of this read the exponent field of exactly that product, so they are left alone.
    static let roundE2M1: @Sendable (MLXArray) -> MLXArray = compile { v in
        let magnitude = abs(v)
        var index = MLXArray.zeros(magnitude.shape, dtype: .int32)
        for threshold in [Float(0.25), 1.25, 2.5, 5] {
            index = index + (magnitude .> threshold).asType(.int32)
        }
        for threshold in [Float(0.75), 1.75, 3.5] {
            index = index + (magnitude .>= threshold).asType(.int32)
        }
        let table = MLXArray(DeepSeekV41FakeQuant.e2m1Values)
        return sign(v) * take(table, index, axis: 0)
    }

    static func blockify(_ x: MLXArray, block: Int) -> MLXArray {
        let n = x.dim(-1)
        precondition(n % block == 0, "last dim \(n) not divisible by block \(block)")
        return x.reshaped(x.shape.dropLast() + [n / block, block])
    }

    /// FP8 round-trip with pow2 scales.
    public static func fp8UE8M0(_ x: MLXArray, block: Int = 32) -> MLXArray {
        let shape = x.shape
        let blocked = blockify(x.asType(.float32), block: block)
        let amax = maximum(abs(blocked).max(axis: -1, keepDims: true), 1e-4)
        let scale = ue8m0Scale(amax, maxInverse: 1 / fp8Max)
        let quantized =
            fromFP8(toFP8(clip(blocked / scale, min: -fp8Max, max: fp8Max)), dtype: .float32)
            * scale
        return quantized.reshaped(shape).asType(x.dtype)
    }

    /// FP4 round-trip with ue8m0 scales: the indexer q/k path.
    public static func fp4UE8M0(_ x: MLXArray, block: Int = 32) -> MLXArray {
        let shape = x.shape
        let blocked = blockify(x.asType(.float32), block: block)
        let amax = maximum(
            abs(blocked).max(axis: -1, keepDims: true), fp4Max * pow(2, -126))
        let scale = ue8m0Scale(amax, maxInverse: 1 / fp4Max)
        let quantized = roundE2M1(clip(blocked / scale, min: -fp4Max, max: fp4Max)) * scale
        return quantized.reshaped(shape).asType(x.dtype)
    }

    /// FP4 round-trip with e4m3 scales: the compressed-KV path.
    public static func fp4E4M3(_ x: MLXArray, block: Int = 16) -> MLXArray {
        let shape = x.shape
        let blocked = blockify(x.asType(.float32), block: block)
        let amax = maximum(abs(blocked).max(axis: -1, keepDims: true), fp4Max * pow(2, -9))
        let scale = fromFP8(toFP8(amax / fp4Max), dtype: .float32)
        let quantized = roundE2M1(clip(blocked / scale, min: -fp4Max, max: fp4Max)) * scale
        return quantized.reshaped(shape).asType(x.dtype)
    }
}
