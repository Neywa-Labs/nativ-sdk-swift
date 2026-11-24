import Foundation
import MLX
import MLXNN

enum MuseSparkVision {
    static func positionEmbedding(
        _ table: MLXArray, side: Int, height: Int, width: Int, ij: Bool
    ) -> MLXArray {
        // The reference rounds the linspace endpoints and samples to the table dtype.
        func axis(_ count: Int) -> MLXArray {
            let start = MLXArray(-1 + 1 / Float(count), dtype: table.dtype).asType(.float32)
            if count == 1 { return start.reshaped(1) }
            let end = MLXArray(1 - 1 / Float(count), dtype: table.dtype).asType(.float32)
            let step = (end - start) / Float(count - 1)
            let i = MLXArray(0 ..< count).asType(.float32)
            return MLX.where(
                i .< count / 2, start + step * i, end - step * (Float(count - 1) - i)
            )
            .asType(table.dtype).asType(.float32)
        }

        let h = axis(height)
        let w = axis(width)
        let index = MLXArray(0 ..< (height * width))
        // Older checkpoint tables enumerate the height axis first on non-square grids.
        let x = ij ? w.take(index % width, axis: 0) : h.take(index % height, axis: 0)
        let y =
            ij
            ? h.take(floorDivide(index, width), axis: 0)
            : w.take(floorDivide(index, height), axis: 0)
        let ix = ((x + 1) * side - 1) / 2
        let iy = ((y + 1) * side - 1) / 2
        let x0 = floor(ix)
        let y0 = floor(iy)
        let dx = ix - x0
        let dy = iy - y0
        let corners = [
            (x0, y0, (x0 + 1 - ix) * (y0 + 1 - iy)),
            (x0 + 1, y0, dx * (y0 + 1 - iy)),
            (x0, y0 + 1, (x0 + 1 - ix) * dy),
            (x0 + 1, y0 + 1, dx * dy),
        ]
        let table32 = table.asType(.float32)
        var result = MLXArray.zeros([height * width, table.dim(1)], dtype: .float32)
        for (cx, cy, weight) in corners {
            let inside = (cx .>= 0) & (cx .< side) & (cy .>= 0) & (cy .< side)
            let row =
                clip(cy, min: 0, max: Float(side - 1)) * side
                + clip(cx, min: 0, max: Float(side - 1))
            result =
                result
                + MLX.where(inside, weight, 0).expandedDimensions(axis: 1)
                * table32.take(row.asType(.int32), axis: 0)
        }
        return result.asType(table.dtype)
    }

    static func gelu(_ x: MLXArray) -> MLXArray { MLXNN.gelu(x.asType(.float32)).asType(x.dtype) }

    final class PreciseLayerNorm: LayerNorm {
        override func callAsFunction(_ x: MLXArray) -> MLXArray {
            MLXFast.layerNorm(
                x.asType(.float32), weight: weight?.asType(.float32),
                bias: bias?.asType(.float32), eps: eps
            ).asType(x.dtype)
        }
    }

    /// Batches equally sized windows or temporal groups without crossing their boundaries.
    struct AttentionPlan {
        let groups: [MLXArray]
        let restore: MLXArray?

        init(boundaries: [Int], permutation: MLXArray? = nil) {
            guard boundaries.count > 2 || permutation != nil else {
                groups = []
                restore = nil
                return
            }
            var order: [Int] = []
            var byLength: [Int: [[Int32]]] = [:]
            for (start, end) in zip(boundaries, boundaries.dropFirst()) {
                let length = end - start
                if byLength[length] == nil { order.append(length) }
                byLength[length, default: []].append((start ..< end).map(Int32.init))
            }
            groups = order.map { length in
                let rows = byLength[length]!
                let indices = MLXArray(rows.flatMap { $0 }, [rows.count, length])
                return permutation?.take(indices, axis: 0) ?? indices
            }
            restore = argSort(concatenated(groups.map { $0.reshaped(-1) }, axis: 0), axis: 0)
        }
    }

    final class Attention: Module {
        @ModuleInfo(key: "qkv_proj") var input: Linear
        @ModuleInfo(key: "proj") var output: Linear
        let heads: Int
        let dimension: Int
        let scale: Float

        init(_ config: MuseGlimmerVisionConfiguration) {
            heads = config.attentionHeads
            dimension = config.hiddenSize / heads
            scale = pow(Float(dimension), -0.5)
            _input.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * 3)
            _output.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        }

        static func rotate(_ x: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
            let pairs = x.reshaped(x.dim(0), x.dim(1), -1, 2).asType(.float32)
            let a = pairs[0..., 0..., 0..., 0]
            let b = pairs[0..., 0..., 0..., 1]
            return stacked([a * cosine - b * sine, b * cosine + a * sine], axis: -1)
                .reshaped(x.shape).asType(x.dtype)
        }

        func callAsFunction(
            _ x: MLXArray, cosine: MLXArray, sine: MLXArray, plan: AttentionPlan
        ) -> MLXArray {
            let parts = split(input(x), parts: 3, axis: -1)
            let queries = Self.rotate(
                parts[0].reshaped(-1, heads, dimension), cosine: cosine, sine: sine)
            let keys = Self.rotate(
                parts[1].reshaped(-1, heads, dimension), cosine: cosine, sine: sine)
            let values = parts[2].reshaped(-1, heads, dimension)
            func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray) -> MLXArray {
                MLXFast.scaledDotProductAttention(
                    queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3),
                    values: v.transposed(0, 2, 1, 3), scale: scale, mask: .none
                )
                .transposed(0, 2, 1, 3).reshaped(-1, heads * dimension)
            }
            if let restore = plan.restore {
                let parts = plan.groups.map { indices in
                    attend(
                        queries.take(indices, axis: 0), keys.take(indices, axis: 0),
                        values.take(indices, axis: 0))
                }
                return output(concatenated(parts, axis: 0).take(restore, axis: 0))
            }
            return output(
                attend(
                    queries.expandedDimensions(axis: 0), keys.expandedDimensions(axis: 0),
                    values.expandedDimensions(axis: 0)))
        }
    }

    final class MLP: Module {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        init(input: Int, width: Int, output: Int, bias: Bool) {
            // Quantized checkpoint loading replaces these projections in place.
            _fc1.wrappedValue = Linear(input, width, bias: bias)
            _fc2.wrappedValue = Linear(width, output, bias: bias)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
    }

    final class Block: Module {
        let norm1: PreciseLayerNorm
        let norm2: PreciseLayerNorm
        let attn: Attention
        let mlp: MLP

        init(_ config: MuseGlimmerVisionConfiguration) {
            norm1 = PreciseLayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            norm2 = PreciseLayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            attn = Attention(config)
            mlp = MLP(
                input: config.hiddenSize, width: config.intermediateSize,
                output: config.hiddenSize, bias: true)
        }

        func callAsFunction(
            _ x: MLXArray, cosine: MLXArray, sine: MLXArray, plan: AttentionPlan
        ) -> MLXArray {
            let h = x + attn(norm1(x), cosine: cosine, sine: sine, plan: plan)
            return h + mlp(norm2(h))
        }
    }

    final class PatchEmbedder: Module {
        @ModuleInfo(key: "patch_embedding") var linear: Linear
        @ModuleInfo(key: "position_embedding_table") var table: Embedding

        init(_ g: MuseGlimmerVisionConfiguration) {
            let patchWidth = 3 * g.patchTemporal * g.patchSize * g.patchSize
            _linear.wrappedValue = Linear(patchWidth, g.hiddenSize, bias: false)
            _table.wrappedValue = Embedding(
                embeddingCount: g.positionEmbeddingHeight * g.positionEmbeddingWidth,
                dimensions: g.hiddenSize)
        }
    }

    final class Tower: Module {
        @ModuleInfo(key: "patch_embedder") var patches: PatchEmbedder
        @ModuleInfo(key: "ln_pre") var pre: PreciseLayerNorm
        @ModuleInfo(key: "ln_post") var post: PreciseLayerNorm
        let layers: [Block]
        let config: MuseSparkConfiguration.Vision
        let inverseFrequencies: [Float]

        init(_ config: MuseSparkConfiguration.Vision) {
            self.config = config
            let g = config.geometry
            _patches.wrappedValue = PatchEmbedder(g)
            _pre.wrappedValue = PreciseLayerNorm(dimensions: g.hiddenSize, eps: g.layerNormEps)
            _post.wrappedValue = PreciseLayerNorm(dimensions: g.hiddenSize, eps: g.layerNormEps)
            layers = (0 ..< g.hiddenLayers).map { _ in Block(g) }
            let count = g.hiddenSize / g.attentionHeads / 4
            inverseFrequencies = (0 ..< count).map {
                1 / pow(g.ropeTheta, Float($0) / Float(count))
            }
        }

        func callAsFunction(_ pixels: MLXArray, grid: [THW]) throws -> MLXArray {
            let g = config.geometry
            let patchWidth = 3 * g.patchTemporal * g.patchSize * g.patchSize
            guard pixels.ndim == 2, pixels.dim(-1) == patchWidth, !grid.isEmpty,
                grid.allSatisfy({
                    $0.t > 0 && $0.h > 0 && $0.w > 0 && $0.h.isMultiple(of: g.mergeSize)
                        && $0.w.isMultiple(of: g.mergeSize)
                }),
                grid.reduce(0, { $0 + $1.product }) == pixels.dim(0)
            else {
                throw VLMError.processing(
                    "Muse Spark vision patches do not match the supplied grids.")
            }

            let side = g.positionEmbeddingHeight
            var positionTables: [[Int]: MLXArray] = [:]
            let positions = grid.map { frame in
                let shape = [frame.h, frame.w]
                let table =
                    positionTables[shape]
                    ?? positionEmbedding(
                        patches.table.weight, side: side, height: frame.h, width: frame.w,
                        ij: config.useIJPositions)
                positionTables[shape] = table
                return broadcast(table, to: [frame.t, frame.h * frame.w, g.hiddenSize])
                    .reshaped(-1, g.hiddenSize)
            }
            let positionIDs = musePositionIds(grid).asType(.float32)
            let frequencies = MLXArray(inverseFrequencies)
            let w = positionIDs[0..., 0, .newAxis] * frequencies
            let h = positionIDs[0..., 1, .newAxis] * frequencies
            let angles = concatenated([w, h], axis: -1).expandedDimensions(axis: 1)
            let cosine = cos(angles)
            let sine = sin(angles)

            let full = AttentionPlan(boundaries: museCuSeqlens(grid))
            let (permutation, boundaries) = museWindowIndex(grid, windowSize: side)
            let window = AttentionPlan(boundaries: boundaries, permutation: permutation)
            let dtype = patches.linear.weight.dtype
            let typed = dtype.isFloatingPoint ? pixels.asType(dtype) : pixels
            var x = pre(
                patches.linear(typed)
                    + concatenated(positions, axis: 0))
            for (i, layer) in layers.enumerated() {
                x = layer(
                    x, cosine: cosine, sine: sine,
                    plan: g.layerTypes[i] == "full_attention" ? full : window)
            }
            return musePixelShuffle(post(x), grid: grid, mergeSize: g.mergeSize)
        }
    }

    final class Adapter: Module {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        init(input: Int, width: Int) {
            _fc1.wrappedValue = Linear(input, width, bias: false)
            _fc2.wrappedValue = Linear(width, width, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray { gelu(fc2(gelu(fc1(x)))) }
    }
}
