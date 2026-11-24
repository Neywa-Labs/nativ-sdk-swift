import Foundation
import MLX
import MLXNN

struct ZImageTransformerConfiguration {
    let dim, heads, mlp, channels, textDim, depth, refiners, contextRefiners, patch, framePatch,
        adaln: Int
    let sections: [Int]
    let theta, eps, timeScale: Float
    init(_ c: DiffusionConfiguration) throws {
        dim = c.int("dim", 3840)
        heads = c.int("n_heads", 30)
        mlp = c.int("intermediate_size", dim * 8 / 3)
        channels = c.int("in_channels", 16)
        textDim = c.int("cap_feat_dim", 2560)
        depth = c.int("n_layers", 30)
        refiners = c.int("n_refiner_layers", 2)
        contextRefiners = c.int("n_context_refiner_layers", refiners)
        patch = c.ints("all_patch_size", [2]).first ?? 2
        framePatch = c.ints("all_f_patch_size", [1]).first ?? 1
        adaln = min(dim, c.int("adaln_embed_dim", 256))
        sections = c.ints("axes_dims", [32, 48, 48])
        theta = c.float("rope_theta", 256)
        eps = c.float("norm_eps", 1e-5)
        timeScale = c.float("t_scale", 1000)
        guard heads > 0, dim % heads == 0, sections.count == 3,
            sections.allSatisfy({ $0 > 0 && $0 % 2 == 0 }),
            sections.reduce(0, +) == dim / heads, c.int("n_kv_heads", heads) == heads, patch > 0,
            framePatch > 0
        else {
            throw ModelExecutionError.invalidOptions(
                "Invalid Z-Image attention or patch configuration")
        }
    }
}

final class ZImageTimestepEmbedder: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear
    init(_ dimensions: Int) {
        _linear1.wrappedValue = Linear(256, 1024)
        _linear2.wrappedValue = Linear(1024, dimensions)
        super.init()
    }
    func callAsFunction(_ t: MLXArray) -> MLXArray {
        linear2(silu(linear1(diffusionTimeEmbedding(t, dimensions: 256).asType(t.dtype))))
    }
}

final class ZImageFeedForward: Module, UnaryLayer {
    @ModuleInfo var w1: Linear
    @ModuleInfo var w2: Linear
    @ModuleInfo var w3: Linear
    init(_ dim: Int, _ hidden: Int) {
        _w1.wrappedValue = Linear(dim, hidden, bias: false)
        _w2.wrappedValue = Linear(hidden, dim, bias: false)
        _w3.wrappedValue = Linear(dim, hidden, bias: false)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x)) * w3(x)) }
}

final class ZImageAttention: Module {
    @ModuleInfo(key: "to_q") var q: Linear
    @ModuleInfo(key: "to_k") var k: Linear
    @ModuleInfo(key: "to_v") var v: Linear
    @ModuleInfo(key: "to_out") var out: [Linear]
    @ModuleInfo(key: "norm_q") var qNorm: Module & UnaryLayer
    @ModuleInfo(key: "norm_k") var kNorm: Module & UnaryLayer
    let heads: Int, headDim: Int
    let parameterFree: Bool
    init(_ c: ZImageTransformerConfiguration, parameterFree: Bool = false, qkNorm: Bool = true) {
        heads = c.heads
        headDim = c.dim / c.heads
        self.parameterFree = parameterFree
        _q.wrappedValue = Linear(c.dim, c.dim, bias: false)
        _k.wrappedValue = Linear(c.dim, c.dim, bias: false)
        _v.wrappedValue = Linear(c.dim, c.dim, bias: false)
        _out.wrappedValue = [Linear(c.dim, c.dim, bias: false)]
        _qNorm.wrappedValue =
            parameterFree
            ? (qkNorm ? DiffusionRMSNorm(c.eps) : Identity()) : RMSNorm(dimensions: headDim)
        _kNorm.wrappedValue =
            parameterFree
            ? (qkNorm ? DiffusionRMSNorm(c.eps) : Identity()) : RMSNorm(dimensions: headDim)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, angles: MLXArray, mask: MLXArray?) -> MLXArray {
        let shape = [x.dim(0), x.dim(1), heads, headDim]
        var queries = diffusionRotatePairs(qNorm(q(x).reshaped(shape)), angles: angles).transposed(
            0, 2, 1, 3)
        var keys = diffusionRotatePairs(kNorm(k(x).reshaped(shape)), angles: angles).transposed(
            0, 2, 1, 3)
        if parameterFree {
            queries = queries.asType(x.dtype)
            keys = keys.asType(x.dtype)
        }
        let values = v(x).reshaped(shape).transposed(0, 2, 1, 3)
        let h = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values,
            scale: pow(Float(headDim), -0.5), mask: mask)
        return out[0](h.transposed(0, 2, 1, 3).reshaped(x.shape))
    }
}

final class ZImageBlock: Module {
    @ModuleInfo var attention: ZImageAttention
    @ModuleInfo(key: "feed_forward") var feedForward: ZImageFeedForward
    @ModuleInfo(key: "attention_norm1") var attentionNorm1: Module & UnaryLayer
    @ModuleInfo(key: "attention_norm2") var attentionNorm2: Module & UnaryLayer
    @ModuleInfo(key: "ffn_norm1") var ffnNorm1: Module & UnaryLayer
    @ModuleInfo(key: "ffn_norm2") var ffnNorm2: Module & UnaryLayer
    @ModuleInfo(key: "adaLN_modulation") var modulation: [Linear]?
    init(
        _ c: ZImageTransformerConfiguration, modulated: Bool = true, parameterFree: Bool = false,
        qkNorm: Bool = true
    ) {
        _attention.wrappedValue = ZImageAttention(c, parameterFree: parameterFree, qkNorm: qkNorm)
        _feedForward.wrappedValue = ZImageFeedForward(c.dim, c.mlp)
        _attentionNorm1.wrappedValue =
            parameterFree ? DiffusionRMSNorm(c.eps) : RMSNorm(dimensions: c.dim)
        _attentionNorm2.wrappedValue =
            parameterFree ? DiffusionRMSNorm(c.eps) : RMSNorm(dimensions: c.dim)
        _ffnNorm1.wrappedValue =
            parameterFree ? DiffusionRMSNorm(c.eps) : RMSNorm(dimensions: c.dim)
        _ffnNorm2.wrappedValue =
            parameterFree ? DiffusionRMSNorm(c.eps) : RMSNorm(dimensions: c.dim)
        _modulation.wrappedValue = modulated ? [Linear(c.adaln, 4 * c.dim)] : nil
        super.init()
    }
    func callAsFunction(
        _ x: MLXArray, angles: MLXArray, time: MLXArray? = nil, mask: MLXArray? = nil,
        noiseMask: MLXArray? = nil, cleanTime: MLXArray? = nil
    ) -> MLXArray {
        if let modulation, let time {
            var m = split(modulation[0](time).expandedDimensions(axis: 1), parts: 4, axis: -1)
            if let noiseMask, let cleanTime {
                let clean = split(
                    modulation[0](cleanTime).expandedDimensions(axis: 1), parts: 4, axis: -1)
                m = zip(m, clean).map {
                    MLX.where(noiseMask.expandedDimensions(axis: -1), $0.0, $0.1)
                }
            }
            let h =
                x + tanh(m[1])
                * attentionNorm2(
                    attention(attentionNorm1(x) * (1 + m[0]), angles: angles, mask: mask))
            return h + tanh(m[3]) * ffnNorm2(feedForward(ffnNorm1(h) * (1 + m[2])))
        }
        let h = x + attentionNorm2(attention(attentionNorm1(x), angles: angles, mask: mask))
        return h + ffnNorm2(feedForward(ffnNorm1(h)))
    }
}

final class ZImageFinalLayer: Module {
    @ModuleInfo(key: "norm_final") var norm: LayerNorm
    @ModuleInfo var linear: Linear
    @ModuleInfo(key: "adaLN_modulation") var modulation: [Linear]
    init(_ c: ZImageTransformerConfiguration) {
        _norm.wrappedValue = LayerNorm(dimensions: c.dim, eps: 1e-6, affine: false)
        _linear.wrappedValue = Linear(c.dim, c.patch * c.patch * c.framePatch * c.channels)
        _modulation.wrappedValue = [Linear(c.adaln, c.dim)]
        super.init()
    }
    func callAsFunction(_ x: MLXArray, time: MLXArray) -> MLXArray {
        linear(norm(x) * (1 + modulation[0](silu(time)).expandedDimensions(axis: 1)))
    }
}

/// Z-Image and Ming share the diffusion blocks but use different caption padding semantics.
final class ZImageTransformer: Module {
    let config: ZImageTransformerConfiguration
    @ModuleInfo(key: "t_embedder") var time: ZImageTimestepEmbedder
    @ModuleInfo(key: "cap_embedder") var caption: [Module]
    @ModuleInfo(key: "noise_refiner") var noiseRefiners: [ZImageBlock]
    @ModuleInfo(key: "context_refiner") var contextRefiners: [ZImageBlock]
    @ModuleInfo var layers: [ZImageBlock]
    @ModuleInfo(key: "final_layer") var final: ZImageFinalLayer
    @ModuleInfo(key: "x_embedder") var image: Linear
    @ParameterInfo(key: "x_pad_token") var imagePad: MLXArray?
    @ParameterInfo(key: "cap_pad_token") var captionPad: MLXArray?
    init(_ config: ZImageTransformerConfiguration, learnedPadding: Bool = true) {
        self.config = config
        _time.wrappedValue = ZImageTimestepEmbedder(config.adaln)
        _caption.wrappedValue = [
            RMSNorm(dimensions: config.textDim, eps: config.eps),
            Linear(config.textDim, config.dim),
        ]
        _noiseRefiners.wrappedValue = (0 ..< config.refiners).map { _ in ZImageBlock(config) }
        _contextRefiners.wrappedValue = (0 ..< config.contextRefiners).map { _ in
            ZImageBlock(config, modulated: false)
        }
        _layers.wrappedValue = (0 ..< config.depth).map { _ in ZImageBlock(config) }
        _final.wrappedValue = ZImageFinalLayer(config)
        _image.wrappedValue = Linear(
            config.framePatch * config.patch * config.patch * config.channels, config.dim)
        _imagePad.wrappedValue = learnedPadding ? MLXArray.zeros([1, config.dim]) : nil
        _captionPad.wrappedValue = learnedPadding ? MLXArray.zeros([1, config.dim]) : nil
        super.init()
    }
    func callAsFunction(
        _ x: MLXArray, timestep: MLXArray, caption features: MLXArray,
        directFeatures: MLXArray? = nil
    ) -> MLXArray {
        let c = config
        let batch = x.dim(0)
        let f = x.dim(2) / c.framePatch
        let h = x.dim(3) / c.patch
        let w = x.dim(4) / c.patch
        let imageLength = f * h * w
        let patches = x.reshaped([batch, c.channels, f, c.framePatch, h, c.patch, w, c.patch])
            .transposed(0, 2, 4, 6, 3, 5, 7, 1).reshaped([batch, imageLength, -1])
        var imageTokens = image(patches)
        var captionTokens = (caption[1] as! Linear)((caption[0] as! RMSNorm)(features))
        if let directFeatures {
            captionTokens = concatenated([captionTokens, directFeatures], axis: 1)
        }
        let capLength = captionTokens.dim(1)
        let imgPadding = (32 - imageLength % 32) % 32
        let capPadding = (32 - capLength % 32) % 32
        func pad(_ x: MLXArray, count: Int, token: MLXArray?) -> MLXArray {
            count == 0
                ? x
                : concatenated(
                    [
                        x,
                        broadcast(
                            token ?? MLXArray.zeros([1, c.dim], dtype: x.dtype),
                            to: [batch, count, c.dim]),
                    ], axis: 1)
        }
        imageTokens = pad(imageTokens, count: imgPadding, token: imagePad)
        captionTokens = pad(captionTokens, count: capPadding, token: captionPad)
        let paddedCap = capLength + capPadding
        let capPositions = stacked(
            [
                arange(1, paddedCap + 1).asType(.float32), MLXArray.zeros([paddedCap]),
                MLXArray.zeros([paddedCap]),
            ], axis: -1
        ).expandedDimensions(axis: 0)
        var imagePositions = diffusionPositions(
            frames: f, height: h, width: w, offset: paddedCap + 1)
        if imgPadding > 0 {
            imagePositions = concatenated(
                [imagePositions, MLXArray.zeros([1, imgPadding, 3])], axis: 1)
        }
        let imgAngles = diffusionRotaryAngles(imagePositions, sections: c.sections, theta: c.theta)
        let capAngles = diffusionRotaryAngles(capPositions, sections: c.sections, theta: c.theta)
        let t = time(timestep * c.timeScale)
        let imgValid = arange(imageLength + imgPadding) .< imageLength
        let capValid = arange(paddedCap) .< capLength
        func mask(_ valid: MLXArray) -> MLXArray? {
            imagePad == nil ? valid.reshaped([1, 1, 1, -1]) : nil
        }
        for block in noiseRefiners {
            imageTokens = block(imageTokens, angles: imgAngles, time: t, mask: mask(imgValid))
        }
        for block in contextRefiners {
            captionTokens = block(captionTokens, angles: capAngles, mask: mask(capValid))
        }
        var hidden = concatenated([imageTokens, captionTokens], axis: 1)
        let angles = concatenated([imgAngles, capAngles], axis: 1)
        let combinedMask = mask(concatenated([imgValid, capValid]))
        for block in layers { hidden = block(hidden, angles: angles, time: t, mask: combinedMask) }
        return final(hidden[0..., ..<imageLength], time: t)
            .reshaped([batch, f, h, w, c.framePatch, c.patch, c.patch, c.channels])
            .transposed(0, 7, 1, 4, 2, 5, 3, 6).reshaped(x.shape)
    }
    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        Dictionary(
            uniqueKeysWithValues: weights.map { key, value in
                var key = key
                if key.hasPrefix("all_final_layer.2-1.") {
                    key = key.replacingOccurrences(of: "all_final_layer.2-1.", with: "final_layer.")
                        .replacingOccurrences(
                            of: "adaLN_modulation.1.", with: "adaLN_modulation.0.")
                }
                key = key.replacingOccurrences(of: "all_x_embedder.2-1.", with: "x_embedder.")
                    .replacingOccurrences(of: "t_embedder.mlp.0.", with: "t_embedder.linear1.")
                    .replacingOccurrences(of: "t_embedder.mlp.2.", with: "t_embedder.linear2.")
                return (key, value)
            })
    }
}
