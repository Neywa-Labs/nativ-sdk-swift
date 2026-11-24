// Copyright 2026 The HuggingFace Team. All rights reserved.
// Adapted from the Apache-2.0 mlx-vlm LFM3 implementation.
import Foundation
import MLX
import MLXNN

/// Partial RoPE keeps unrotated coordinates in both head halves. Rotating a
/// contiguous prefix with ordinary partial RoPE would pair the wrong coordinates.
struct LFM3Rotary {
    let dimensions: Int
    let angles: Int
    let theta: Float

    init(dimensions: Int, configuration: LFM3RotaryConfiguration?, sliding: Bool) {
        self.dimensions = dimensions
        angles = Int(Float(dimensions) * (configuration?.partialFactor ?? (sliding ? 1 : 0.5))) / 2
        theta = configuration?.theta ?? (sliding ? 10_000 : 1_000_000)
    }

    func callAsFunction(_ x: MLXArray, offset: RoPEOffset?) -> MLXArray {
        var positions = MLXArray(0 ..< x.dim(-2))
        switch offset {
        case .scalar(let n): positions = positions + n
        case .batch(let n): positions = positions + n.expandedDimensions(axis: -1)
        case nil: break
        }
        let inv = pow(theta, -MLXArray(0 ..< angles).asType(.float32) * (2 / Float(dimensions)))
        let frequencies = concatenated([inv, MLXArray.zeros([dimensions / 2 - angles])])
        var phase = positions.asType(.float32).expandedDimensions(axis: -1) * frequencies
        phase = concatenated([phase, phase], axis: -1)
        if phase.ndim == 3 { phase = phase.expandedDimensions(axis: 1) }
        let halves = x.split(parts: 2, axis: -1)
        return x * cos(phase).asType(x.dtype)
            + concatenated([-halves[1], halves[0]], axis: -1) * sin(phase).asType(x.dtype)
    }
}

final class LFM3Attention: Module {
    let config: LFM3TextConfiguration
    let rope: LFM3Rotary
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ c: LFM3TextConfiguration, type: String) {
        config = c
        rope = LFM3Rotary(
            dimensions: c.headDim, configuration: c.rotary[type],
            sliding: type == "sliding_attention")
        _q.wrappedValue = Linear(c.hiddenSize, c.attentionHeads * c.headDim, bias: false)
        _k.wrappedValue = Linear(c.hiddenSize, c.kvHeads * c.headDim, bias: false)
        _v.wrappedValue = Linear(c.hiddenSize, c.kvHeads * c.headDim, bias: false)
        _output.wrappedValue = Linear(c.attentionHeads * c.headDim, c.hiddenSize, bias: false)
        _gate.wrappedValue = Linear(c.hiddenSize, c.attentionHeads, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.normEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (b, n) = (x.dim(0), x.dim(1))
        let queries = rope(
            qNorm(q(x).reshaped(b, n, config.attentionHeads, config.headDim)).transposed(
                0, 2, 1, 3), offset: cache?.ropeOffset)
        let keys = rope(
            kNorm(k(x).reshaped(b, n, config.kvHeads, config.headDim)).transposed(0, 2, 1, 3),
            offset: cache?.ropeOffset)
        let values = v(x).reshaped(b, n, config.kvHeads, config.headDim).transposed(0, 2, 1, 3)
        let attended = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: pow(Float(config.headDim), -0.5), mask: mask
        ).transposed(0, 2, 1, 3)
        return output((attended * sigmoid(gate(x)).expandedDimensions(axis: -1)).reshaped(b, n, -1))
    }
}

final class LFM3ShortConv: Module {
    let length: Int
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "in_proj") var input: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    init(_ c: LFM3TextConfiguration) {
        length = c.convLength
        _conv.wrappedValue = Conv1d(
            inputChannels: c.hiddenSize, outputChannels: c.hiddenSize,
            kernelSize: length, groups: c.hiddenSize, bias: c.convBias)
        _input.wrappedValue = Linear(c.hiddenSize, 3 * c.hiddenSize, bias: c.convBias)
        _output.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: c.convBias)
    }
    func callAsFunction(_ x: MLXArray, cache: MambaCache?) -> MLXArray {
        let parts = input(x).split(parts: 3, axis: -1)
        var bx = parts[0] * parts[2]
        if let mask = createSSMMask(h: x, cache: cache) {
            bx = which(mask.expandedDimensions(axis: -1), bx, MLXArray(0).asType(bx.dtype))
        }
        let state = cache?[0] ?? MLXArray.zeros([x.dim(0), length - 1, x.dim(2)], dtype: x.dtype)
        bx = concatenated([state, bx], axis: 1)
        if let cache {
            cache[0] = contiguous(bx[0..., (bx.dim(1) - length + 1)..., 0...])
            cache.advance(x.dim(1))
        }
        return output(parts[1] * conv(bx))
    }
}

final class LFM3MLP: Module, UnaryLayer {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear
    init(hidden: Int, intermediate: Int) {
        _w1.wrappedValue = Linear(hidden, intermediate, bias: false)
        _w3.wrappedValue = Linear(hidden, intermediate, bias: false)
        _w2.wrappedValue = Linear(intermediate, hidden, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { w2(fusedSwiGLU(w1(x), w3(x))) }
}

final class LFM3DecoderLayer: Module {
    let type: String
    @ModuleInfo(key: "self_attn") var attention: LFM3Attention?
    let conv: LFM3ShortConv?
    @ModuleInfo(key: "feed_forward") var feedForward: LFM3MLP
    @ModuleInfo(key: "operator_norm") var operatorNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "post_operator_norm") var postOperatorNorm: RMSNorm?
    @ModuleInfo(key: "post_ffn_norm") var postFFNNorm: RMSNorm?
    init(_ c: LFM3TextConfiguration, index: Int) {
        type = c.layerTypes[index]
        if type == "conv" {
            conv = LFM3ShortConv(c)
        } else {
            conv = nil
            _attention.wrappedValue = LFM3Attention(c, type: type)
        }
        _feedForward.wrappedValue = LFM3MLP(hidden: c.hiddenSize, intermediate: c.intermediateSize)
        _operatorNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        if c.sandwichNorm {
            _postOperatorNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
            _postFFNNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
        }
    }
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil
    ) -> MLXArray {
        let normalized = operatorNorm(x)
        let h =
            attention?(normalized, mask: mask, cache: cache)
            ?? conv!(normalized, cache: cache as? MambaCache)
        let residual = x + (postOperatorNorm?(h) ?? h)
        let f = feedForward(ffnNorm(residual))
        return residual + (postFFNNorm?(f) ?? f)
    }
}

final class LFM3Backbone: Module {
    let config: LFM3TextConfiguration
    let layers: [LFM3DecoderLayer]
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?
    @ModuleInfo(key: "embedding_norm") var norm: RMSNorm
    init(_ c: LFM3TextConfiguration, embeddings: Bool = true) {
        config = c
        if embeddings {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: c.vocabularySize, dimensions: c.hiddenSize)
        }
        layers = (0 ..< c.hiddenLayers).map { LFM3DecoderLayer(c, index: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
    }
    func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]? = nil, causal: Bool = true)
        -> MLXArray
    {
        var h = embeddings
        for (i, layer) in layers.enumerated() {
            let c = cache?[i]
            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                causal
                ? createAttentionMask(
                    h: h, cache: c,
                    windowSize: layer.type == "sliding_attention" ? config.slidingWindow : nil)
                : .none
            h = layer(h, mask: mask, cache: c)
        }
        return norm(h)
    }
    func newCache(configuration: KVCacheConfiguration? = nil) throws -> [KVCache] {
        try config.layerTypes.map { type in
            if type == "conv" { return MambaCache() }
            if type == "sliding_attention" { return RotatingKVCache(maxSize: config.slidingWindow) }
            return try makeAttentionKVCache(configuration: configuration)
        }
    }
}

final class LFM3Language: Module {
    let model: LFM3Backbone
    @ModuleInfo(key: "lm_head") var head: Linear?
    init(_ c: LFM3TextConfiguration) {
        model = LFM3Backbone(c)
        if !c.tieEmbeddings {
            _head.wrappedValue = Linear(c.hiddenSize, c.vocabularySize, bias: false)
        }
    }
    func logits(_ hidden: MLXArray) -> MLXArray {
        head?(hidden) ?? model.embedTokens!.asLinear(hidden)
    }
}
