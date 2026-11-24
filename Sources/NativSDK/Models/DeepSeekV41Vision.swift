import Foundation
import MLX
import MLXNN

public enum DeepSeekV41Vision {
    static let normEps: Float = 1e-6

    /// 2D rope tables: one row per grid position, its height and width coordinates
    /// interleaved across the frequency axis.
    static func cosSin(height: Int, width: Int, dimensions: Int, theta: Float)
        -> (cos: MLXArray, sin: MLXArray)
    {
        let inverse =
            1
            / pow(
                theta,
                MLXArray(stride(from: 0, to: dimensions, by: 2).map(Float.init))
                    / Float(dimensions))
        let hpos = broadcast(
            MLXArray(0 ..< height).expandedDimensions(axis: 1), to: [height, width])
        let wpos = broadcast(
            MLXArray(0 ..< width).expandedDimensions(axis: 0), to: [height, width])
        let positions = stacked([hpos, wpos], axis: -1).reshaped(-1, 2, 1)
        let freqs = (positions.asType(.float32) * inverse).reshaped(height * width, -1)
        return (
            cos(freqs).expandedDimensions(axis: 1),
            sin(freqs).expandedDimensions(axis: 1)
        )
    }

    /// Split-half rotation, as the vision tower pairs its channels.
    static func applyRotary(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let promoted = x.asType(.float32)
        let half = promoted.dim(-1) / 2
        let first = promoted[.ellipsis, ..<half]
        let second = promoted[.ellipsis, half...]
        return concatenated(
            [first * cos - second * sin, second * cos + first * sin], axis: -1
        ).asType(dtype)
    }

    /// Flatten each patch and project to the vision dimension.
    final class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo(key: "proj") var proj: Linear

        init(_ config: DeepSeekV41Configuration) {
            let vision = config.visionConfiguration
            _proj.wrappedValue = Linear(
                3 * vision.patchSize * vision.patchSize, vision.hiddenSize)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            proj(x.reshaped(x.dim(0), -1))
        }
    }

    /// Full bidirectional self-attention with 2D rope.
    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "wqkv") var wqkv: Linear
        @ModuleInfo(key: "wo") var wo: Linear

        init(_ config: DeepSeekV41Configuration) {
            let vision = config.visionConfiguration
            self.heads = vision.numHeads
            self.headDim = vision.hiddenSize / vision.numHeads
            self.scale = pow(Float(headDim), -0.5)
            _wqkv.wrappedValue = Linear(vision.hiddenSize, 3 * vision.hiddenSize)
            _wo.wrappedValue = Linear(vision.hiddenSize, vision.hiddenSize)
            super.init()
        }

        func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
            let n = x.dim(0)
            let fused = wqkv(x)
            let parts = split(fused, parts: 3, axis: -1).map {
                $0.reshaped(n, heads, headDim)
            }
            let q = DeepSeekV41Vision.applyRotary(parts[0], cos: cos, sin: sin)
            let k = DeepSeekV41Vision.applyRotary(parts[1], cos: cos, sin: sin)
            let v = parts[2]

            let scores = matmul(
                q.transposed(1, 0, 2), k.transposed(1, 0, 2).transposed(0, 2, 1))
            let out = matmul(
                softmax(scores * scale, axis: -1), v.transposed(1, 0, 2))
            return wo(out.transposed(1, 0, 2).reshaped(n, -1))
        }
    }

    /// SwiGLU over the vision dimension, with the gate and the value fused into `w1`.
    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "w1") var w1: Linear
        @ModuleInfo(key: "w2") var w2: Linear

        init(_ config: DeepSeekV41Configuration) {
            let vision = config.visionConfiguration
            _w1.wrappedValue = Linear(
                vision.hiddenSize, 2 * vision.intermediateSize, bias: false)
            _w2.wrappedValue = Linear(
                vision.intermediateSize, vision.hiddenSize, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let parts = split(w1(x), parts: 2, axis: -1)
            return w2(silu(parts[0]) * parts[1])
        }
    }

    final class Block: Module {
        @ModuleInfo(key: "norm1") var norm1: RMSNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo(key: "norm2") var norm2: RMSNorm
        @ModuleInfo(key: "mlp") var mlp: MLP

        init(_ config: DeepSeekV41Configuration) {
            let size = config.visionConfiguration.hiddenSize
            _norm1.wrappedValue = RMSNorm(dimensions: size, eps: normEps)
            _attention.wrappedValue = Attention(config)
            _norm2.wrappedValue = RMSNorm(dimensions: size, eps: normEps)
            _mlp.wrappedValue = MLP(config)
            super.init()
        }

        func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
            let attended = x + attention(norm1(x), cos: cos, sin: sin)
            return attended + mlp(norm2(attended))
        }
    }

    /// DeepSeek's ViT: full bidirectional attention over one image with 2D rope.
    public final class ViT: Module {
        let ropeDimensions: Int
        let ropeTheta: Float

        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        let blocks: [Block]
        @ModuleInfo(key: "norm") var norm: RMSNorm

        public init(_ config: DeepSeekV41Configuration) {
            let vision = config.visionConfiguration
            self.ropeDimensions = vision.hiddenSize / vision.numHeads / 2
            self.ropeTheta = vision.ropeTheta
            _patchEmbed.wrappedValue = PatchEmbed(config)
            self.blocks = (0 ..< vision.numLayers).map { _ in Block(config) }
            _norm.wrappedValue = RMSNorm(dimensions: vision.hiddenSize, eps: normEps)
            super.init()
        }

        public func callAsFunction(_ patches: MLXArray, height: Int, width: Int)
            -> MLXArray
        {
            var x = patchEmbed(patches)
            let (cos, sin) = DeepSeekV41Vision.cosSin(
                height: height, width: width, dimensions: ropeDimensions,
                theta: ropeTheta)
            for block in blocks {
                x = block(x, cos: cos, sin: sin)
            }
            return norm(x)
        }
    }

    /// Space-to-depth downsample of the ViT grid into language-dimension embeddings.
    public final class Aligner: Module {
        let downsampleRatio: Int

        @ModuleInfo(key: "w1") var w1: Linear
        @ModuleInfo(key: "w2") var w2: Linear
        let activation = GELU(approximation: .precise)

        public init(_ config: DeepSeekV41Configuration) {
            let vision = config.visionConfiguration
            self.downsampleRatio = vision.downsampleRatio
            let inputDimensions =
                vision.hiddenSize * vision.downsampleRatio * vision.downsampleRatio
            _w1.wrappedValue = Linear(inputDimensions, config.hiddenSize)
            _w2.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
            super.init()
        }

        /// Group each `r x r` block of ViT tokens into one embedding.
        ///
        /// A partial trailing block is zero-filled rather than dropped, so the grid
        /// rounds up and matches the span the prompt reserved. Each output vector runs
        /// channel-major: all `r*r` neighbours of one channel before the next.
        public func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
            let r = downsampleRatio
            let d = x.dim(-1)
            let outHeight = (height + r - 1) / r
            let outWidth = (width + r - 1) / r
            var x = x.reshaped(height, width, d)
            let padHeight = outHeight * r - height
            let padWidth = outWidth * r - width
            if padHeight > 0 || padWidth > 0 {
                x = padded(
                    x,
                    widths: [
                        .init((0, padHeight)), .init((0, padWidth)), .init((0, 0)),
                    ])
            }
            x = x.reshaped(outHeight, r, outWidth, r, d)
                .transposed(0, 2, 4, 1, 3)
                .reshaped(outHeight * outWidth, d * r * r)
            return w2(activation(w1(x)))
        }
    }
}
