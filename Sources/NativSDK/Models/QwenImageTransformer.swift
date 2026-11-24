import Foundation
import MLX
import MLXNN

final class QwenImageZeroCenterNorm: Module, UnaryLayer {
    @ParameterInfo var weight: MLXArray
    let eps: Float
    init(_ dim: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.zeros([dim])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x.asType(.float32)
        return
            (h * rsqrt(mean(square(h), axis: -1, keepDims: true) + eps)
            * (weight.asType(.float32) + 1)).asType(x.dtype)
    }
}

final class QwenImageTextProjection: Module, UnaryLayer {
    @ModuleInfo(key: "text_norm") var norm: QwenImageZeroCenterNorm
    @ModuleInfo(key: "in_layer") var input: Linear
    @ModuleInfo(key: "out_layer") var output: Linear
    init(_ input: Int, _ hidden: Int, eps: Float) {
        _norm.wrappedValue = QwenImageZeroCenterNorm(input, eps: eps)
        _input.wrappedValue = Linear(input, hidden, bias: false)
        _output.wrappedValue = Linear(hidden, hidden, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { output(geluApproximate(input(norm(x)))) }
}

final class QwenImageMLP: Module, UnaryLayer {
    @ModuleInfo var proj: Linear
    @ModuleInfo var out: Linear
    @ModuleInfo(key: "gate_layer") var gate: Linear
    init(_ dim: Int, ratio: Int) {
        _proj.wrappedValue = Linear(dim, dim * ratio, bias: false)
        _out.wrappedValue = Linear(dim * ratio, dim, bias: false)
        _gate.wrappedValue = Linear(dim, dim * ratio, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { out(silu(gate(x)) * proj(x)) }
}

final class QwenImageReferenceCache {
    var keys: [MLXArray] = [], values: [MLXArray] = []
    var angles: MLXArray?
    var shape: [Int] = []
    var dtype: DType?
    var prefixCount = 0
}

final class QwenImageAttention: Module {
    @ModuleInfo(key: "to_q") var q: Linear
    @ModuleInfo(key: "to_k") var k: Linear
    @ModuleInfo(key: "to_v") var v: Linear
    @ModuleInfo(key: "to_out") var out: [Linear]
    @ModuleInfo(key: "norm_q") var qNorm: RMSNorm
    @ModuleInfo(key: "norm_k") var kNorm: RMSNorm
    let heads: Int, headDim: Int
    init(dim: Int, heads: Int, headDim: Int, eps: Float) {
        self.heads = heads
        self.headDim = headDim
        _q.wrappedValue = Linear(dim, dim, bias: false)
        _k.wrappedValue = Linear(dim, dim, bias: false)
        _v.wrappedValue = Linear(dim, dim, bias: false)
        _out.wrappedValue = [Linear(dim, dim, bias: false)]
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        super.init()
    }
    func callAsFunction(
        _ x: MLXArray, angles: MLXArray, mask: MLXArray?, cache: QwenImageReferenceCache?,
        index: Int, extracting: Bool
    ) -> MLXArray {
        let shape = [x.dim(0), x.dim(1), heads, headDim]
        let queries = diffusionRotatePairs(qNorm(q(x).reshaped(shape)), angles: angles).asType(
            x.dtype
        ).transposed(0, 2, 1, 3)
        var keys = diffusionRotatePairs(kNorm(k(x).reshaped(shape)), angles: angles).asType(x.dtype)
            .transposed(0, 2, 1, 3)
        var values = v(x).reshaped(shape).transposed(0, 2, 1, 3)
        if let cache {
            if extracting {
                cache.keys.append(contiguous(keys[0..., 0..., ..<cache.prefixCount, 0...]))
                cache.values.append(contiguous(values[0..., 0..., ..<cache.prefixCount, 0...]))
            } else {
                keys = concatenated([cache.keys[index], keys], axis: 2)
                values = concatenated([cache.values[index], values], axis: 2)
            }
        }
        let h = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: pow(Float(headDim), -0.5),
            mask: mask)
        return out[0](h.transposed(0, 2, 1, 3).reshaped(x.shape))
    }
}

private func qwenImageSelect(_ params: MLXArray, target: MLXArray?) -> MLXArray {
    guard let target else { return params.expandedDimensions(axis: 1) }
    return MLX.where(
        target.reshaped([1, -1, 1]), params[..<1].expandedDimensions(axis: 1),
        params[1...].expandedDimensions(axis: 1))
}

final class QwenImageBlock: Module {
    @ModuleInfo(key: "img_norm1") var norm1: LayerNorm
    @ModuleInfo var attn: QwenImageAttention
    @ModuleInfo(key: "img_norm2") var norm2: LayerNorm
    @ModuleInfo(key: "img_mlp") var mlp: QwenImageMLP
    init(dim: Int, heads: Int, headDim: Int, ratio: Int, eps: Float) {
        _norm1.wrappedValue = LayerNorm(dimensions: dim, eps: eps, affine: false)
        _norm2.wrappedValue = LayerNorm(dimensions: dim, eps: eps, affine: false)
        _attn.wrappedValue = QwenImageAttention(dim: dim, heads: heads, headDim: headDim, eps: eps)
        _mlp.wrappedValue = QwenImageMLP(dim, ratio: ratio)
        super.init()
    }
    func callAsFunction(
        _ x: MLXArray, modulation: MLXArray, angles: MLXArray, mask: MLXArray?, target: MLXArray?,
        cache: QwenImageReferenceCache?, index: Int, extracting: Bool
    ) -> MLXArray {
        let m = split(modulation, parts: 4, axis: -1).map { qwenImageSelect($0, target: target) }
        let h =
            x + tanh(m[1])
            * attn(
                norm1(x) * (1 + m[0]), angles: angles, mask: mask, cache: cache, index: index,
                extracting: extracting)
        return h + tanh(m[3]) * mlp(norm2(h) * (1 + m[2]))
    }
}

final class QwenImageOutputNorm: Module {
    @ModuleInfo var linear: Linear
    @ModuleInfo var norm: LayerNorm
    init(_ dim: Int, eps: Float) {
        _linear.wrappedValue = Linear(dim, dim, bias: false)
        _norm.wrappedValue = LayerNorm(dimensions: dim, eps: eps, affine: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, time: MLXArray, target: MLXArray?) -> MLXArray {
        norm(x) * (1 + qwenImageSelect(linear(silu(time).asType(x.dtype)), target: target))
    }
}

final class QwenImageTimeEmbedding: Module {
    @ModuleInfo(key: "linear_1") var first: Linear
    @ModuleInfo(key: "linear_2") var second: Linear
    init(_ dim: Int) {
        _first.wrappedValue = Linear(256, dim, bias: false)
        _second.wrappedValue = Linear(dim, dim, bias: false)
        super.init()
    }
    func callAsFunction(_ t: MLXArray, dtype: DType) -> MLXArray {
        second(silu(first(diffusionTimeEmbedding(t * 1000, dimensions: 256).asType(dtype))))
    }
}

final class QwenImageTransformer: Module {
    @ModuleInfo(key: "time_text_embed") var time: QwenImageTimeEmbedding
    @ModuleInfo(key: "txt_in") var text: QwenImageTextProjection
    @ModuleInfo(key: "img_in") var image: Linear
    @ModuleInfo var modulation: [Linear]
    @ModuleInfo(key: "transformer_blocks") var blocks: [QwenImageBlock]
    @ModuleInfo(key: "norm_out") var norm: QwenImageOutputNorm
    @ModuleInfo(key: "proj_out") var output: Linear
    let causal: Bool, sections: [Int]
    init(_ c: DiffusionConfiguration) throws {
        let heads = c.int("num_attention_heads", 32)
        let headDim = c.int("attention_head_dim", 128)
        let dim = heads * headDim
        let eps = c.float("eps", 1e-6)
        sections = c.ints("axes_dims_rope", [16, 56, 56])
        causal = c.bool("causal_condition", true)
        guard heads > 0, sections.count == 3, sections.reduce(0, +) == headDim,
            sections.allSatisfy({ $0 > 0 && $0 % 2 == 0 })
        else { throw ModelExecutionError.invalidOptions("Invalid Qwen-Image rotary dimensions") }
        _time.wrappedValue = QwenImageTimeEmbedding(dim)
        _text.wrappedValue = QwenImageTextProjection(c.int("context_in_dim", 4096), dim, eps: eps)
        _image.wrappedValue = Linear(c.int("in_channels", 64), dim, bias: false)
        _modulation.wrappedValue = [Linear(dim, 4 * dim, bias: false)]
        _blocks.wrappedValue = (0 ..< c.int("num_layers", 32)).map { _ in
            QwenImageBlock(
                dim: dim, heads: heads, headDim: headDim, ratio: c.int("mlp_ratio", 3), eps: eps)
        }
        _norm.wrappedValue = QwenImageOutputNorm(dim, eps: eps)
        _output.wrappedValue = Linear(dim, c.int("out_channels", 64), bias: false)
        super.init()
    }
    func callAsFunction(
        _ latents: MLXArray, text features: MLXArray, timestep: MLXArray, height: Int, width: Int,
        referenceShapes: [(Int, Int)] = [], imageSlots: [Bool]? = nil,
        cache: QwenImageReferenceCache? = nil
    ) throws -> MLXArray {
        guard latents.dim(0) == 1, features.dim(0) == 1 else {
            throw ModelExecutionError.invalidOptions("Qwen-Image generates one image at a time")
        }
        if cache != nil && !causal {
            throw ModelExecutionError.invalidOptions(
                "Reference KV caching requires causal conditioning")
        }
        let targetCount = height * width
        let extracting = cache?.angles == nil
        var hidden: MLXArray
        var angles: MLXArray
        var target: MLXArray?
        var mask: MLXArray?
        if let cache, let cachedAngles = cache.angles {
            guard cache.shape == latents.shape, cache.dtype == latents.dtype,
                cache.keys.count == blocks.count
            else {
                throw ModelExecutionError.invalidOptions(
                    "Reference cache target shape or dtype changed")
            }
            hidden = image(latents)
            angles = cachedAngles
            target = MLXArray.ones([targetCount], dtype: .bool)
            mask = nil
        } else {
            let textCount = features.dim(1)
            let slots = imageSlots ?? Array(repeating: false, count: features.dim(1))
            guard slots.count == textCount,
                latents.dim(1) == targetCount + referenceShapes.reduce(0, { $0 + $1.0 * $1.1 })
            else {
                throw ModelExecutionError.invalidOptions(
                    "Qwen-Image prompt slots and latent shapes disagree")
            }
            let txt = text(features)
            let img = image(latents)
            let shapes = referenceShapes + [(height, width)]
            var pieces = [MLXArray]()
            var ids = [Int]()
            var positions = [Float]()
            var cursor = 0
            var imageCursor = 0
            var position = 0
            for (id, shape) in shapes.enumerated() {
                let count = shape.0 * shape.1
                let isTarget = id == referenceShapes.count
                let start: Int
                let end: Int
                if isTarget {
                    guard !slots[cursor...].contains(true) else {
                        throw ModelExecutionError.invalidOptions(
                            "Extra reference slots without image latents")
                    }
                    start = textCount
                    end = start
                } else {
                    guard count % 4 == 0, let found = slots[cursor...].firstIndex(of: true),
                        found + count / 4 <= textCount,
                        slots[found ..< (found + count / 4)].allSatisfy({ $0 })
                    else {
                        throw ModelExecutionError.invalidOptions(
                            "Reference image slots do not match VAE dimensions")
                    }
                    start = found
                    end = found + count / 4
                }
                if start > cursor {
                    pieces.append(txt[0..., cursor ..< start])
                    ids += Array(repeating: -1, count: start - cursor)
                    for p in position ..< (position + start - cursor) {
                        positions += [Float(p), Float(p), Float(p)]
                    }
                    position += start - cursor
                }
                pieces.append(img[0..., imageCursor ..< (imageCursor + count)])
                ids += Array(repeating: id, count: count)
                for y in 0 ..< shape.0 {
                    for x in 0 ..< shape.1 {
                        positions += [
                            Float(position), Float(y - (shape.0 - shape.0 / 2)),
                            Float(x - (shape.1 - shape.1 / 2)),
                        ]
                    }
                }
                position += max(shape.0, shape.1)
                cursor = end
                imageCursor += count
            }
            hidden = concatenated(pieces, axis: 1)
            angles = diffusionRotaryAngles(
                MLXArray(positions, [1, ids.count, 3]), sections: sections, theta: 10000)
            let imageIDs = MLXArray(ids)
            let index = arange(ids.count)
            let causalMask = index.expandedDimensions(axis: 1) .>= index.expandedDimensions(axis: 0)
            let sameImage =
                (imageIDs.expandedDimensions(axis: 1) .== imageIDs.expandedDimensions(axis: 0))
                .&& (imageIDs.expandedDimensions(axis: 1) .>= 0)
            mask = (causalMask .|| sameImage).reshaped([1, 1, ids.count, ids.count])
            target = causal ? imageIDs .== referenceShapes.count : nil
            if let cache {
                cache.prefixCount = ids.count - targetCount
                cache.keys = []
                cache.values = []
                cache.angles = contiguous(angles[0..., (ids.count - targetCount)...])
                cache.shape = [1, targetCount, latents.dim(-1)]
                cache.dtype = latents.dtype
            }
        }
        var t = timestep.asType(hidden.dtype)
        if causal { t = concatenated([t, MLXArray.zeros([1], dtype: t.dtype)]) }
        let temb = time(t, dtype: hidden.dtype)
        let mod = modulation[0](silu(temb))
        for (index, block) in blocks.enumerated() {
            hidden = block(
                hidden, modulation: mod, angles: angles, mask: mask, target: target, cache: cache,
                index: index, extracting: extracting)
        }
        hidden = hidden[0..., (hidden.dim(1) - targetCount)...]
        return output(
            norm(
                hidden, time: temb,
                target: causal ? MLXArray.ones([targetCount], dtype: .bool) : nil))
    }
}
