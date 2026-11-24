import Foundation
import MLX
import MLXNN

enum Inkling2Vision {
    typealias Configuration = Inkling2Configuration.Vision

    final class Attention: Module {
        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear
        let heads: Int
        let headDimension: Int

        init(_ config: Configuration) {
            heads = config.heads
            headDimension = config.width / config.heads
            _qProj.wrappedValue = Linear(config.width, config.width)
            _kProj.wrappedValue = Linear(config.width, config.width)
            _vProj.wrappedValue = Linear(config.width, config.width)
            _outProj.wrappedValue = Linear(config.width, config.width)
        }

        func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
            func rotate(_ y: MLXArray) -> MLXArray {
                let y = y.reshaped(1, x.dim(0), heads, headDimension)
                let yf = y.asType(.float32)
                let pairs = yf.reshaped(1, x.dim(0), heads, headDimension / 2, 2)
                let rotated = stacked([-pairs[.ellipsis, 1], pairs[.ellipsis, 0]], axis: -1)
                    .reshaped(y.shape)
                return (yf * cos + rotated * sin).asType(y.dtype).transposed(0, 2, 1, 3)
            }

            let v = vProj(x).reshaped(1, x.dim(0), heads, headDimension).transposed(0, 2, 1, 3)
            let out = MLXFast.scaledDotProductAttention(
                queries: rotate(qProj(x)), keys: rotate(kProj(x)), values: v,
                scale: 1 / sqrt(Float(headDimension)), mask: .none)
            return outProj(out.transposed(0, 2, 1, 3).reshaped(x.shape))
        }
    }

    final class MLP: Module {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        init(_ config: Configuration) {
            let inner = Int(Float(config.width) * config.mlpRatio)
            _fc1.wrappedValue = Linear(config.width, inner)
            _fc2.wrappedValue = Linear(inner, config.width)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
    }

    final class Layer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "layernorm_before") var before: LayerNorm
        @ModuleInfo(key: "layernorm_after") var after: LayerNorm
        @ParameterInfo(key: "lambda_1") var attentionScale: MLXArray
        @ParameterInfo(key: "lambda_2") var mlpScale: MLXArray

        init(_ config: Configuration) {
            _attention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(config)
            _before.wrappedValue = LayerNorm(dimensions: config.width, eps: config.epsilon)
            _after.wrappedValue = LayerNorm(dimensions: config.width, eps: config.epsilon)
            _attentionScale.wrappedValue = .ones([config.width])
            _mlpScale.wrappedValue = .ones([config.width])
        }

        func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
            let hidden = x + attentionScale * attention(before(x), cos: cos, sin: sin)
            return hidden + mlpScale * mlp(after(hidden))
        }
    }

    final class Embeddings: Module {
        @ModuleInfo(key: "patch_embedding") var patch: Linear
        @ParameterInfo(key: "position_embedding") var position: MLXArray
        @ParameterInfo(key: "class_embedding") var cls: MLXArray
        @ParameterInfo(key: "cls_position_embedding") var clsPosition: MLXArray

        init(_ config: Configuration) {
            _patch.wrappedValue = Linear(config.patch * config.patch * 3, config.width, bias: false)
            _position.wrappedValue = .zeros([
                config.positionGrid * config.positionGrid, config.width,
            ])
            _cls.wrappedValue = .zeros([config.width])
            _clsPosition.wrappedValue = .zeros([config.width])
        }
    }

    final class Tower: Module {
        @ModuleInfo var embeddings: Embeddings
        @ModuleInfo(key: "pre_layernorm") var preNorm: LayerNorm
        @ModuleInfo var layers: [Layer]
        @ModuleInfo(key: "proj_in") var projIn: Linear
        @ModuleInfo(key: "proj_out") var projOut: Linear
        let config: Configuration

        init(_ config: Configuration, textWidth: Int) {
            self.config = config
            _embeddings.wrappedValue = Embeddings(config)
            _preNorm.wrappedValue = LayerNorm(dimensions: config.width, eps: config.epsilon)
            _layers.wrappedValue = (0 ..< config.layers).map { _ in Layer(config) }
            _projIn.wrappedValue = Linear(
                config.merge * config.merge * config.width, config.projectionWidth,
                bias: config.projectorBias)
            _projOut.wrappedValue = Linear(
                config.projectionWidth, textWidth, bias: config.projectorBias)
        }

        func callAsFunction(_ pixels: MLXArray, grid: [THW]) throws -> MLXArray {
            guard !grid.isEmpty, grid.allSatisfy({ $0.t == 1 && $0.h > 0 && $0.w > 0 }),
                pixels.shape == [grid.reduce(0) { $0 + $1.product }, config.tile, config.tile, 3]
            else {
                throw VLMError.processing("Inkling 2 image tiles do not match the image grid.")
            }
            let pixels = pixels.asType(preNorm.weight!.dtype)
            let merge = config.merge
            let patch = config.patch
            let width = config.width
            let spatialDimension = width / config.heads / 2
            let inverse = exp(
                MLXArray(stride(from: 0, to: spatialDimension, by: 2)).asType(.float32)
                    * (-log(config.theta) / Float(spatialDimension)))
            let learned = embeddings.position.reshaped(
                config.positionGrid, config.positionGrid, width
            )
            .asType(.float32)

            var start = 0
            let outputs = grid.map { item in
                let (rows, cols) = (item.h, item.w)
                let count = rows * cols
                let patches = pixels[start ..< (start + count)]
                    .reshaped(count, merge, patch, merge, patch, 3)
                    .transposed(0, 1, 3, 2, 4, 5).reshaped(count * merge * merge, -1)
                start += count

                let projected = embeddings.patch(patches)
                func tileOrder(_ x: MLXArray, channels: Int) -> MLXArray {
                    x.reshaped(rows, merge, cols, merge, channels)
                        .transposed(0, 2, 1, 3, 4).reshaped(-1, channels)
                }
                let resized = bilinearInterpolate(
                    learned, newHeight: rows * merge, newWidth: cols * merge)
                let positioned =
                    projected + tileOrder(resized, channels: width).asType(projected.dtype)
                let cls = (embeddings.cls + embeddings.clsPosition).asType(projected.dtype)
                var hidden = preNorm(concatenated([cls.reshaped(1, width), positioned]))

                let height = rows * merge
                let columns = cols * merge
                let x = broadcast(
                    MLXArray(1 ... columns).reshaped(1, columns), to: [height, columns])
                let y = broadcast(
                    MLXArray(1 ... height).reshaped(height, 1), to: [height, columns])
                let coords = tileOrder(stacked([x, y], axis: -1), channels: 2).asType(.float32)
                let positions = concatenated([MLXArray.zeros([1, 2]), coords])
                let angles = repeated(
                    (positions[.ellipsis, .newAxis] * inverse).reshaped(hidden.dim(0), -1),
                    count: 2, axis: -1
                ).reshaped(1, hidden.dim(0), 1, width / config.heads)
                let cosine = cos(angles)
                let sine = sin(angles)
                for layer in layers {
                    hidden = layer(hidden, cos: cosine, sin: sine)
                }
                return projOut(gelu(projIn(hidden[1...].reshaped(count, -1))))
            }
            return concatenated(outputs, axis: 0)
        }
    }
}
