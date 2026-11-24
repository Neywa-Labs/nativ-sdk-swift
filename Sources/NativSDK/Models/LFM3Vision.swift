// Copyright 2026 The HuggingFace Team. All rights reserved.
// Adapted from the Apache-2.0 mlx-vlm LFM3-VL implementation.
import Foundation
import MLX
import MLXNN

final class LFM3VisionAttention: Module {
    let heads: Int
    let kvHeads: Int
    let dim: Int
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    @ModuleInfo(key: "q_layernorm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_layernorm") var kNorm: RMSNorm
    init(_ c: LFM3VisionConfiguration) {
        heads = c.attentionHeads
        kvHeads = c.kvHeads
        dim = c.hiddenSize / heads
        _q.wrappedValue = Linear(c.hiddenSize, heads * dim, bias: false)
        _k.wrappedValue = Linear(c.hiddenSize, kvHeads * dim, bias: false)
        _v.wrappedValue = Linear(c.hiddenSize, kvHeads * dim, bias: false)
        _output.wrappedValue = Linear(heads * dim, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: dim, eps: c.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: dim, eps: c.normEps)
    }
    func callAsFunction(_ x: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
        func rotate(_ x: MLXArray) -> MLXArray {
            let f = x.asType(.float32)
            let halves = f.split(parts: 2, axis: -1)
            return
                (f * cosine.expandedDimensions(axis: 1)
                + concatenated([-halves[1], halves[0]], axis: -1) * sine.expandedDimensions(axis: 1))
                .asType(x.dtype)
        }
        let n = x.dim(0)
        let queries = rotate(qNorm(q(x).reshaped(n, heads, dim))).transposed(1, 0, 2)
            .expandedDimensions(axis: 0)
        let keys = rotate(kNorm(k(x).reshaped(n, kvHeads, dim))).transposed(1, 0, 2)
            .expandedDimensions(axis: 0)
        let values = v(x).reshaped(n, kvHeads, dim).transposed(1, 0, 2).expandedDimensions(axis: 0)
        let h = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: pow(Float(dim), -0.5), mask: .none)
        return output(h[0].transposed(1, 0, 2).reshaped(n, -1))
    }
}

final class LFM3VisionConv: Module {
    let merge: Int
    @ModuleInfo(key: "in_proj") var input: Linear
    @ModuleInfo(key: "spatial_conv") var conv: Conv2d
    @ModuleInfo(key: "out_proj") var output: Linear
    init(_ c: LFM3VisionConfiguration) {
        merge = c.mergeSize
        _input.wrappedValue = Linear(c.hiddenSize, 3 * c.hiddenSize, bias: false)
        _conv.wrappedValue = Conv2d(
            inputChannels: c.hiddenSize, outputChannels: c.hiddenSize,
            kernelSize: IntOrPair(c.kernelSize), padding: IntOrPair(c.kernelSize / 2),
            groups: c.hiddenSize, bias: false)
        _output.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: false)
    }
    func callAsFunction(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let parts = input(x).split(parts: 3, axis: -1)
        var grid = (parts[0] * parts[2]).reshaped(height / merge, width / merge, merge, merge, -1)
            .transposed(0, 2, 1, 3, 4).reshaped(1, height, width, -1)
        grid = conv(grid).reshaped(height / merge, merge, width / merge, merge, -1)
            .transposed(0, 2, 1, 3, 4).reshaped(height * width, -1)
        return output(parts[1] * grid)
    }
}

final class LFM3VisionLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: LFM3VisionAttention?
    let conv: LFM3VisionConv?
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "feed_forward") var mlp: LFM3MLP
    init(_ c: LFM3VisionConfiguration, index: Int) {
        if c.layerTypes[index] == "full_attention" {
            conv = nil
            _attention.wrappedValue = LFM3VisionAttention(c)
        } else {
            conv = LFM3VisionConv(c)
        }
        _operatorNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        _mlp.wrappedValue = LFM3MLP(hidden: c.hiddenSize, intermediate: c.intermediateSize)
    }
    func callAsFunction(_ x: MLXArray, height: Int, width: Int, cosine: MLXArray, sine: MLXArray)
        -> MLXArray
    {
        let h = operatorNorm(x)
        let r =
            x
            + (attention?(h, cosine: cosine, sine: sine) ?? conv!(h, height: height, width: width))
        return r + mlp(ffnNorm(r))
    }
}

final class LFM3Vision: Module {
    let config: LFM3VisionConfiguration
    @ModuleInfo var patchifier: Linear
    @ModuleInfo(key: "embedding_norm") var norm: RMSNorm
    let layers: [LFM3VisionLayer]
    init(_ c: LFM3VisionConfiguration) {
        config = c
        _patchifier.wrappedValue = Linear(
            c.channels * c.temporalPatchSize * c.patchSize * c.patchSize, c.hiddenSize, bias: false)
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        layers = (0 ..< c.hiddenLayers).map { LFM3VisionLayer(c, index: $0) }
    }
    func callAsFunction(_ pixels: MLXArray, grids: [THW]) throws -> MLXArray {
        let c = config
        let m = c.mergeSize
        guard pixels.ndim == 2,
            pixels.dim(1) == c.channels * c.temporalPatchSize * c.patchSize * c.patchSize,
            !grids.isEmpty,
            grids.allSatisfy({ $0.t > 0 && $0.h > 0 && $0.w > 0 && $0.h % m == 0 && $0.w % m == 0 }
            ),
            grids.reduce(0, { $0 + $1.t * $1.h * $1.w }) == pixels.dim(0)
        else {
            throw ModelExecutionError.invalidOptions("LFM3 pixel patches do not match vision grids")
        }
        let spatial = c.hiddenSize / c.attentionHeads / 2
        let inv = pow(
            c.rotary?.theta ?? 1_000_000,
            -MLXArray(stride(from: 0, to: spatial, by: 2)).asType(.float32) / Float(spatial))
        var frames = [MLXArray]()
        var offset = 0
        for grid in grids {
            var positions = [Int]()
            for row in stride(from: 0, to: grid.h, by: m) {
                for col in stride(from: 0, to: grid.w, by: m) {
                    for r in 0 ..< m { for s in 0 ..< m { positions += [row + r, col + s] } }
                }
            }
            var angles = (MLXArray(positions, [grid.h * grid.w, 2, 1]) * inv).reshaped(
                grid.h * grid.w, -1)
            angles = concatenated([angles, angles], axis: -1)
            let cosine = cos(angles)
            let sine = sin(angles)
            for _ in 0 ..< grid.t {
                try Task.checkCancellation()
                var x = norm(
                    patchifier(
                        pixels[offset ..< (offset + grid.h * grid.w)].asType(norm.weight.dtype)))
                for layer in layers {
                    x = layer(x, height: grid.h, width: grid.w, cosine: cosine, sine: sine)
                }
                frames.append(x)
                offset += grid.h * grid.w
            }
        }
        return concatenated(frames)
    }
}
