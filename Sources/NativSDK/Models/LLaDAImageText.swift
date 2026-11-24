import Foundation
import MLX
import MLXNN

final class LLaDAImageTextNorm: RMSNorm {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.ones([x.dim(-1)], dtype: x.dtype), eps: eps) * weight
    }
}
final class LLaDAImageTextAttention: LLaDA2Language.Attention {
    override init(_ c: LLaDA2Configuration) {
        super.init(c)
        if c.useQKNorm {
            update(
                modules: .unflattened([
                    "query_layernorm": LLaDAImageTextNorm(dimensions: c.headDim, eps: c.rmsNormEps),
                    "key_layernorm": LLaDAImageTextNorm(dimensions: c.headDim, eps: c.rmsNormEps),
                ]))
        }
    }
    override func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        positionIDs: MLXArray? = nil
    ) -> MLXArray {
        let c = config
        let b = x.dim(0)
        let n = x.dim(1)
        let parts = split(
            queryKeyValue(x).reshaped([b, n, c.attentionHeads + 2 * c.kvHeads, c.headDim]),
            indices: [c.attentionHeads, c.attentionHeads + c.kvHeads], axis: -2)
        var q = (queryLayerNorm?(parts[0]) ?? parts[0]).transposed(0, 2, 1, 3)
        var k = (keyLayerNorm?(parts[1]) ?? parts[1]).transposed(0, 2, 1, 3)
        let v = parts[2].transposed(0, 2, 1, 3)
        if c.rotaryDim > 0 {
            let positions =
                positionIDs
                ?? (arange(n).asType(.float32) + Float(cache?.offset ?? 0)).reshaped([1, n])
            let freq = pow(
                MLXArray(c.ropeTheta),
                -arange(0, c.rotaryDim, step: 2).asType(.float32) / Float(c.rotaryDim))
            let a = positions.asType(.float32).expandedDimensions(axis: -1) * freq
            let angles = concatenated([a, a], axis: -1).expandedDimensions(axis: 1)
            let cosine = cos(angles).asType(q.dtype)
            let sine = sin(angles).asType(q.dtype)
            func rotate(_ x: MLXArray) -> MLXArray {
                let r = x[.ellipsis, ..<c.rotaryDim]
                let halves = split(r, parts: 2, axis: -1)
                return concatenated(
                    [
                        r * cosine + concatenated([-halves[1], halves[0]], axis: -1) * sine,
                        x[.ellipsis, c.rotaryDim...],
                    ], axis: -1)
            }
            q = rotate(q)
            k = rotate(k)
        }
        let output = attentionWithCacheUpdate(
            queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask)
        return dense(output.transposed(0, 2, 1, 3).reshaped([b, n, -1]))
    }
}
final class LLaDAImageTextMoE: LLaDA2Language.SparseMoEBlock {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, _) = gate(x)
        let logits = matmul(x.asType(.float32), gate.weight.T.asType(.float32))
        var scores = takeAlong(sigmoid(logits), indices, axis: -1)
        if config.numExpertsPerToken > 1 {
            scores = scores / (sum(scores, axis: -1, keepDims: true) + 1e-20)
        }
        scores = scores * config.routedScalingFactor
        let routed = x.expandedDimensions(axes: [-2, -3])
        let activated =
            silu(switchMLP.gateProj(routed, indices)) * switchMLP.upProj(routed, indices)
        let weighted = (activated * scores.expandedDimensions(axes: [-1, -2])).asType(x.dtype)
        let output = switchMLP.downProj(weighted, indices).squeezed(axis: -2)
        var h = output.asType(.float32).sum(axis: -2).asType(x.dtype)
        if let sharedExperts { h = h + sharedExperts(x) }
        return h
    }
}
final class LLaDAImageTextEncoder: LLaDA2Language.Model {
    override init(_ c: LLaDA2Configuration) {
        super.init(c)
        for (i, layer) in layers.enumerated() {
            var replacements: [String: Module] = [
                "attention": LLaDAImageTextAttention(c),
                "input_layernorm": LLaDAImageTextNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps),
                "post_attention_layernorm": LLaDAImageTextNorm(
                    dimensions: c.hiddenSize, eps: c.rmsNormEps),
            ]
            if c.usesSparseMLP(i) { replacements["mlp"] = LLaDAImageTextMoE(c) }
            layer.update(modules: .unflattened(replacements))
        }
        update(
            modules: .unflattened([
                "norm": LLaDAImageTextNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
            ]))
    }
    func encode(
        _ ids: MLXArray, embeddings: MLXArray? = nil, mask: MLXArray, positions: MLXArray? = nil
    ) -> MLXArray {
        var h = embeddings ?? wordEmbeddings(ids)
        for layer in layers {
            h =
                h
                + layer.attention(
                    layer.inputLayerNorm(h), mask: .array(mask), cache: nil, positionIDs: positions)
            h = h + layer.mlp(layer.postAttentionLayerNorm(h))
        }
        return norm(h)
    }
}
