import Foundation
import MLX
import MLXNN

/// DeepSeek-YaRN rope tables for the indexer, one row per position.
///
/// The powers and phases are built on the CPU rather than through MLX, because MLX's
/// power approximation drifts into phase errors at long positions, and a small table
/// error is enough to change the BF16/FP4 rounding downstream, which reorders positions
/// whose scores are close together. The trig itself is evaluated in Double as upstream
/// does, though at the released dims and theta the single-precision result is identical
/// even at position 100000, so that part is parity rather than a fix.
public final class DeepSeekV41IndexRope {
    public struct YarnParameters: Equatable, Sendable {
        public let factor: Float
        public let betaFast: Float
        public let betaSlow: Float
        public let originalLength: Int

        public init(factor: Float, betaFast: Float, betaSlow: Float, originalLength: Int) {
            self.factor = factor
            self.betaFast = betaFast
            self.betaSlow = betaSlow
            self.originalLength = originalLength
        }
    }

    let ropeDimensions: Int
    let theta: Float
    let yarn: YarnParameters

    /// The number of positions actually built, which is rounded up to 256.
    var builtCapacity = 0
    private var cachedCos: MLXArray?
    private var cachedSin: MLXArray?

    public init(ropeDimensions: Int, theta: Float, yarn: YarnParameters) {
        self.ropeDimensions = ropeDimensions
        self.theta = theta
        self.yarn = yarn
    }

    /// The inverse frequencies after YarN interpolation, in Float as the source keeps
    /// them. `originalLength` of zero disables the correction, as upstream's guard does.
    public func inverseFrequencies() -> [Float] {
        let dimensions = ropeDimensions
        var inverse = stride(from: 0, to: dimensions, by: 2).map { index -> Float in
            1 / pow(theta, Float(index) / Float(dimensions))
        }
        guard yarn.originalLength > 0 else { return inverse }

        func correctionDimension(_ rotations: Float) -> Double {
            Double(dimensions)
                * log(Double(yarn.originalLength) / (Double(rotations) * 2 * .pi))
                / (2 * log(Double(theta)))
        }
        let low = max(correctionDimension(yarn.betaFast).rounded(.down), 0)
        let high = min(correctionDimension(yarn.betaSlow).rounded(.up), Double(dimensions - 1))
        for index in inverse.indices {
            let ramp = (Float(index) - Float(low)) / max(Float(high - low), 1e-3)
            let smooth = 1 - min(max(ramp, 0), 1)
            inverse[index] = inverse[index] / yarn.factor * (1 - smooth) + inverse[index] * smooth
        }
        return inverse
    }

    /// `cos` and `sin` for positions `0 ..< length`, each `[length, ropeDimensions / 2]`.
    /// The table is built to a multiple of 256 so successive decode positions reuse it.
    public func table(length: Int) -> (cos: MLXArray, sin: MLXArray) {
        let capacity = ((length + 255) / 256) * 256
        if capacity > builtCapacity || cachedCos == nil {
            let inverse = inverseFrequencies().map(Double.init)
            let half = inverse.count
            var cosValues = [Float](repeating: 0, count: capacity * half)
            var sinValues = [Float](repeating: 0, count: capacity * half)
            for position in 0 ..< capacity {
                let scalar = Double(Float(position))
                for index in 0 ..< half {
                    let phase = Double(Float(scalar * inverse[index]))
                    cosValues[position * half + index] = Float(Foundation.cos(phase))
                    sinValues[position * half + index] = Float(Foundation.sin(phase))
                }
            }
            cachedCos = MLXArray(cosValues).reshaped(capacity, half)
            cachedSin = MLXArray(sinValues).reshaped(capacity, half)
            builtCapacity = capacity
        }
        return (cachedCos![0 ..< length], cachedSin![0 ..< length])
    }
}

public enum DeepSeekV41IndexScores {
    /// Scores above this key count are computed a tile at a time.
    static let tile = 4096
    /// And only once the held scores are worth not holding.
    static let tileMinimumElements = 1 << 26

    /// The key width one tile covers, or the whole axis when tiling would not pay.
    ///
    /// Tiling only pays once the scores are big enough to be worth not holding: one
    /// query's worth is a few megabytes, and splitting that just buys kernel launches.
    /// The width divides the axis as evenly as it can, because a tile one key wide
    /// selects a different matmul kernel and moves the scores by about 1e-4 relative.
    static func tileWidth(keyCount: Int, heldElements: Int) -> Int {
        guard keyCount > tile, heldElements > tileMinimumElements else { return keyCount }
        let tiles = (keyCount + tile - 1) / tile
        return (keyCount + tiles - 1) / tiles
    }

    /// Rotate adjacent pairs in the trailing `ropeDimensions` entries of each head.
    public static func applyRotary(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, ropeDimensions: Int
    ) -> MLXArray {
        let dtype = x.dtype
        let passive = x[.ellipsis, ..<(x.dim(-1) - ropeDimensions)]
        let rotating = x[.ellipsis, (x.dim(-1) - ropeDimensions)...].asType(.float32)
        let pairs = rotating.reshaped(rotating.shape.dropLast() + [ropeDimensions / 2, 2])
        let even = pairs[.ellipsis, 0]
        let odd = pairs[.ellipsis, 1]
        let rotated = stacked(
            [even * cos - odd * sin, even * sin + odd * cos], axis: -1)
        return concatenated(
            [
                passive,
                rotated.reshaped(rotating.shape).asType(dtype),
            ], axis: -1)
    }

    /// Head-weighted rectified scores of every query against every index key.
    ///
    /// The per-head scores are summed away immediately, so they are computed a tile of
    /// keys at a time: materializing them for the whole key axis costs
    /// `queries * heads * keys` floats, which at prefill widths is tens of gigabytes per
    /// layer. No reduction crosses a tile, and tiles are kept even in width because a
    /// tile one key wide selects a different matmul kernel and moves the scores by about
    /// 1e-4 relative, enough to reorder positions that close together.
    public static func scores(
        queries: MLXArray, indexKeys: MLXArray, weights: MLXArray,
        compressLengths: MLXArray? = nil, candidates: MLXArray? = nil
    ) -> MLXArray {
        let keyCount = indexKeys.dim(1)
        let width = tileWidth(
            keyCount: keyCount, heldElements: queries.dim(1) * queries.dim(2) * keyCount)
        let whole = width >= keyCount

        func one(_ start: Int, _ stop: Int) -> MLXArray {
            let keys = whole ? indexKeys : indexKeys[0..., start ..< stop]
            let flat = matmul(
                queries.reshaped(queries.dim(0), -1, queries.dim(3)),
                keys.transposed(0, 2, 1))
            var scores = flat.reshaped(
                queries.dim(0), queries.dim(1), queries.dim(2), keys.dim(1))
            scores = (maximum(scores, 0) * weights[.ellipsis, .newAxis]).sum(axis: 2)
            if let compressLengths {
                let positions = MLXArray(start ..< stop)
                scores = MLX.where(
                    positions[.newAxis, 0...] .< compressLengths, scores,
                    MLXArray(-Float.infinity))
            }
            if let candidates {
                let window = whole ? candidates : candidates[.ellipsis, start ..< stop]
                scores = MLX.where(window, scores, MLXArray(-Float.infinity))
            }
            return scores
        }

        if whole { return one(0, keyCount) }
        return concatenated(
            stride(from: 0, to: keyCount, by: width).map { one($0, min($0 + width, keyCount)) },
            axis: -1)
    }

    /// The reachable length per query, either one value for a decode step or one per
    /// query position during prefill.
    public enum CompressLengths {
        case scalar(Int)
        case perQuery(MLXArray)
    }

    /// Level one of the two-level top-k: keep the `topKBlocks` highest-scoring blocks per
    /// query.
    ///
    /// `logits` is `[..., positions]` with unreachable positions already at -inf, which
    /// is what makes a block score of -inf mean "not reachable yet". Returns a bool mask
    /// shaped like `logits`, so consuming layers just mask and never think about blocks.
    public static func selectCandidateBlocks(
        logits: MLXArray, compressLengths: CompressLengths, topKBlocks: Int, blockSize: Int
    ) -> MLXArray {
        let width = logits.dim(-1)
        var logits = logits
        let padding = ((-width) % blockSize + blockSize) % blockSize
        if padding > 0 {
            logits = concatenated(
                [
                    logits,
                    MLXArray.full(
                        logits.shape.dropLast() + [padding],
                        values: MLXArray(-Float.infinity)),
                ], axis: -1)
        }
        let scores =
            logits
            .reshaped(logits.shape.dropLast() + [-1, blockSize])
            .max(axis: -1)
        let blocks = scores.dim(-1)

        let match: MLXArray
        switch compressLengths {
        case .scalar(let length):
            match = MLXArray(0 ..< blocks) .== ((length - 1) / blockSize)
        case .perQuery(let lengths):
            let last = (lengths - 1).floorDivide(blockSize)
            let flat = last.reshaped(-1)
            let rows = MLXArray(0 ..< blocks).reshaped(1, -1) .== flat[0..., .newAxis]
            match = rows.reshaped(lengths.shape.dropLast() + [blocks])
        }

        let pinned = MLX.where(match, MLXArray(Float.infinity), scores)
        let order = argSort(-pinned, axis: -1)
        let rank = argSort(order, axis: -1)
        let keep = (rank .< min(topKBlocks, blocks)) .&& (scores .> -Float.infinity)
        return repeated(keep, count: blockSize, axis: -1)[.ellipsis, ..<width]
    }
}
