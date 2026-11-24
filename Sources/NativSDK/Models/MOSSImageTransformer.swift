import Foundation
import MLX
import MLXNN

struct MOSSImageTransformerConfiguration {
    let width: Int
    let feedForwardWidth: Int
    let heads: Int
    let layers: Int
    let textWidth: Int
    let frequencyWidth: Int
    let latentChannels: Int
    let ropeBase: Float

    var headWidth: Int { width / heads }

    init(_ configuration: DiffusionConfiguration) throws {
        width = configuration.int("dim", 3072)
        feedForwardWidth = configuration.int("ffn_dim", 8192)
        heads = configuration.int("num_heads", 24)
        layers = configuration.int("num_layers", 42)
        textWidth = configuration.int("text_dim", 4096)
        frequencyWidth = configuration.int("freq_dim", 256)
        latentChannels = configuration.int("latent_channels", 128)
        ropeBase = configuration.float("rope_base", 10_000)
        guard width > 0, heads > 0, width.isMultiple(of: heads), headWidth.isMultiple(of: 8),
            feedForwardWidth > 0, layers > 0, textWidth > 0, latentChannels > 0,
            frequencyWidth >= 2, frequencyWidth.isMultiple(of: 2),
            ropeBase.isFinite, ropeBase > 0
        else {
            throw ModelExecutionError.invalidOptions("Invalid MOSS-Image transformer configuration")
        }
    }
}

struct MOSSImageRotary {
    let sine: MLXArray
    let cosine: MLXArray

    init(
        height: Int, width: Int, textLength: Int,
        headWidth: Int, base: Float, dtype: DType
    ) {
        let imageCount = height * width
        let indices = MLXArray(0 ..< imageCount)
        let y = indices.floorDivide(width).asType(.float32) - Float(height - 1) / 2
        let x = (indices % width).asType(.float32) - Float(width - 1) / 2
        let image = stacked(
            [
                MLXArray.zeros([imageCount]),
                MLXArray.full([imageCount], values: MLXArray(Float(textLength))), y, x,
            ],
            axis: -1)
        let textIndex = MLXArray(0 ..< textLength).asType(.float32)
        let zeros = MLXArray.zeros(like: textIndex)
        let text = stacked([zeros, textIndex, zeros, zeros], axis: -1)
        let positions = concatenated([image, text], axis: 0)
        let bands = headWidth / 8
        let frequency =
            1 / pow(MLXArray(base), MLXArray(0 ..< bands).asType(.float32) / Float(bands))
        let phase = (positions.expandedDimensions(axis: -1) * frequency)
            .reshaped(1, 1, imageCount + textLength, headWidth / 2)
        let sinHalf = sin(phase).asType(dtype)
        let cosHalf = cos(phase).asType(dtype)
        sine = concatenated([sinHalf, sinHalf], axis: -1)
        cosine = concatenated([cosHalf, cosHalf], axis: -1)
    }

    private init(sine: MLXArray, cosine: MLXArray) {
        self.sine = sine
        self.cosine = cosine
    }

    func textSuffix(startingAt index: Int) -> Self {
        Self(sine: sine[0..., 0..., index..., 0...], cosine: cosine[0..., 0..., index..., 0...])
    }

    func apply(_ x: MLXArray) -> MLXArray {
        x * cosine + RotaryEmbeddingOps.rotateHalf(x) * sine
    }
}

private func mossImageAttention(
    _ q: MLXArray, _ k: MLXArray, _ v: MLXArray,
    queryNorm: RMSNorm, keyNorm: RMSNorm, rotary: MOSSImageRotary, heads: Int
) -> MLXArray {
    let (batch, length, width) = (v.dim(0), v.dim(1), v.dim(2))
    let headWidth = width / heads
    let shape = [batch, length, heads, headWidth]
    let q = rotary.apply(queryNorm(q.reshaped(shape)).transposed(0, 2, 1, 3)).asType(v.dtype)
    let k = rotary.apply(keyNorm(k.reshaped(shape)).transposed(0, 2, 1, 3)).asType(v.dtype)
    let values = v.reshaped(shape).transposed(0, 2, 1, 3)
    let attended = MLXFast.scaledDotProductAttention(
        queries: q, keys: k, values: values, scale: 1 / sqrt(Float(headWidth)), mask: .none)
    return attended.transposed(0, 2, 1, 3).reshaped(batch, length, width)
}

private func mossImageSiTU(_ hidden: MLXArray, gate: MLXArray) -> MLXArray {
    (4 * tanh(gate / 4) * sigmoid(gate)) * (25 * tanh(hidden / 25))
}

private final class MOSSImageTextBlock: Module {
    let norm1: RMSNorm
    let norm2: RMSNorm
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    @ModuleInfo var qkv: Linear
    @ModuleInfo var proj: Linear
    @ModuleInfo(key: "ffn_gate") var gate: Linear
    @ModuleInfo(key: "ffn_up") var up: Linear
    @ModuleInfo(key: "ffn_down") var down: Linear
    let heads: Int

    init(_ config: MOSSImageTransformerConfiguration) {
        heads = config.heads
        let width = config.width
        norm1 = RMSNorm(dimensions: width, eps: 1e-6)
        norm2 = RMSNorm(dimensions: width, eps: 1e-6)
        _queryNorm.wrappedValue = RMSNorm(dimensions: config.headWidth, eps: 1e-6)
        _keyNorm.wrappedValue = RMSNorm(dimensions: config.headWidth, eps: 1e-6)
        _qkv.wrappedValue = Linear(width, 3 * width)
        _proj.wrappedValue = Linear(width, width)
        _gate.wrappedValue = Linear(width, config.feedForwardWidth)
        _up.wrappedValue = Linear(width, config.feedForwardWidth)
        _down.wrappedValue = Linear(config.feedForwardWidth, width)
    }

    func callAsFunction(_ x: MLXArray, rotary: MOSSImageRotary) -> MLXArray {
        let qkv = split(self.qkv(norm1(x)), parts: 3, axis: -1)
        let attention = mossImageAttention(
            qkv[0], qkv[1], qkv[2], queryNorm: queryNorm, keyNorm: keyNorm, rotary: rotary,
            heads: heads)
        let residual = x + proj(attention)
        let normalized = norm2(residual)
        return residual + down(mossImageSiTU(up(normalized), gate: gate(normalized)))
    }
}

private final class MOSSImageTextProjection: Module {
    @ModuleInfo var proj: Linear
    let blocks: [MOSSImageTextBlock]

    init(_ config: MOSSImageTransformerConfiguration) {
        _proj.wrappedValue = Linear(config.textWidth, config.width)
        blocks = (0 ..< 2).map { _ in MOSSImageTextBlock(config) }
    }

    func callAsFunction(_ text: MLXArray, rotary: MOSSImageRotary, dtype: DType) -> MLXArray {
        var hidden = proj(text.asType(dtype))
        for block in blocks {
            hidden = block(hidden, rotary: rotary)
        }
        return hidden
    }
}

private final class MOSSImageProjections: Module {
    @ModuleInfo var q: Linear
    @ModuleInfo var k: Linear
    @ModuleInfo var v: Linear
    @ModuleInfo var o: Linear
    @ModuleInfo(key: "ffn_up_gate") var upGate: Linear
    @ModuleInfo(key: "ffn_down") var down: Linear

    init(_ config: MOSSImageTransformerConfiguration) {
        let width = config.width
        _q.wrappedValue = Linear(width, width, bias: false)
        _k.wrappedValue = Linear(width, width, bias: false)
        _v.wrappedValue = Linear(width, width, bias: false)
        _o.wrappedValue = Linear(width, width, bias: false)
        _upGate.wrappedValue = Linear(width, 2 * config.feedForwardWidth, bias: false)
        _down.wrappedValue = Linear(config.feedForwardWidth, width, bias: false)
    }
}

private final class MOSSImageBlock: Module {
    let norm1: RMSNorm
    let norm2: RMSNorm
    @ModuleInfo(key: "q_norm") var queryNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var keyNorm: RMSNorm
    let proj: MOSSImageProjections
    let heads: Int

    init(_ config: MOSSImageTransformerConfiguration) {
        heads = config.heads
        norm1 = RMSNorm(dimensions: config.width, eps: 1e-6)
        norm2 = RMSNorm(dimensions: config.width, eps: 1e-6)
        _queryNorm.wrappedValue = RMSNorm(dimensions: config.headWidth, eps: 1e-6)
        _keyNorm.wrappedValue = RMSNorm(dimensions: config.headWidth, eps: 1e-6)
        proj = MOSSImageProjections(config)
    }

    func callAsFunction(
        _ x: MLXArray, modulation: [MLXArray], rotary: MOSSImageRotary
    ) -> MLXArray {
        let attentionInput = (norm1(x) * (1 + modulation[1]) + modulation[0]).asType(x.dtype)
        let attention = mossImageAttention(
            proj.q(attentionInput), proj.k(attentionInput), proj.v(attentionInput),
            queryNorm: queryNorm, keyNorm: keyNorm, rotary: rotary, heads: heads)
        let residual = x + proj.o(attention) * modulation[2]
        let ffnInput = (norm2(residual) * (1 + modulation[4]) + modulation[3]).asType(x.dtype)
        let parts = split(proj.upGate(ffnInput), parts: 2, axis: -1)
        return residual + proj.down(mossImageSiTU(parts[0], gate: parts[1])) * modulation[5]
    }
}

struct MOSSImageCondition {
    let text: MLXArray
    let rotary: MOSSImageRotary
}

final class MOSSImageTransformer: Module {
    let config: MOSSImageTransformerConfiguration
    @ModuleInfo(key: "video_embedder") private var imageProjection: Linear
    @ModuleInfo(key: "text_embedder") private var textProjection: MOSSImageTextProjection
    @ModuleInfo(key: "time_embedding") private var timeEmbedding: Sequential
    @ModuleInfo(key: "shared_block_modulation") private var blockModulation: Linear
    @ModuleInfo(key: "video_head_modulation") private var headModulation: Linear
    @ModuleInfo(key: "final_norm_video") private var finalNorm: RMSNorm
    @ModuleInfo(key: "final_linear_video") private var finalProjection: Linear
    private let layers: [MOSSImageBlock]

    init(_ config: MOSSImageTransformerConfiguration) {
        self.config = config
        _imageProjection.wrappedValue = Linear(config.latentChannels, config.width)
        _textProjection.wrappedValue = MOSSImageTextProjection(config)
        _timeEmbedding.wrappedValue = Sequential {
            Linear(config.frequencyWidth, config.width)
            SiLU()
            Linear(config.width, config.width)
        }
        _blockModulation.wrappedValue = Linear(config.width, 6 * config.width)
        _headModulation.wrappedValue = Linear(config.width, 2 * config.width)
        _finalNorm.wrappedValue = RMSNorm(dimensions: config.width, eps: 1e-6)
        _finalProjection.wrappedValue = Linear(config.width, config.latentChannels, bias: false)
        layers = (0 ..< config.layers).map { _ in MOSSImageBlock(config) }
    }

    func condition(text: MLXArray, height: Int, width: Int) -> MOSSImageCondition {
        let dtype = finalNorm.weight.dtype
        let rotary = MOSSImageRotary(
            height: height, width: width, textLength: text.dim(1),
            headWidth: config.headWidth, base: config.ropeBase, dtype: dtype)
        let projected = textProjection(
            text, rotary: rotary.textSuffix(startingAt: height * width), dtype: dtype)
        eval(projected, rotary.cosine, rotary.sine)
        return MOSSImageCondition(text: projected, rotary: rotary)
    }

    func callAsFunction(
        _ latents: MLXArray, timestep: MLXArray, condition: MOSSImageCondition,
        skipLayers: Set<Int> = []
    ) -> MLXArray {
        let (batch, channels, height, width) =
            (latents.dim(0), latents.dim(1), latents.dim(2), latents.dim(3))
        let imageTokens = height * width
        let image = latents.reshaped(batch, channels, imageTokens).transposed(0, 2, 1)
        let dtype = finalNorm.weight.dtype
        let time = timeEmbedding(
            diffusionTimeEmbedding(timestep, dimensions: config.frequencyWidth).asType(dtype))
        let modulation = split(
            blockModulation(time).expandedDimensions(axis: 1), parts: 6, axis: -1)
        let head = split(headModulation(time).expandedDimensions(axis: 1), parts: 2, axis: -1)
        var tokens = concatenated([imageProjection(image.asType(dtype)), condition.text], axis: 1)
        for (index, layer) in layers.enumerated() where !skipLayers.contains(index) {
            tokens = layer(tokens, modulation: modulation, rotary: condition.rotary)
        }
        let normalized = finalNorm(tokens[0..., ..<imageTokens, 0...])
        let prediction = finalProjection((normalized * (1 + head[1]) + head[0]).asType(dtype))
        return prediction.transposed(0, 2, 1).reshaped(batch, channels, height, width)
    }

    static func checkpointKey(_ key: String) -> String? {
        guard key != "rope.bands" else { return nil }
        return
            key
            .replacingOccurrences(of: "time_embedding.0.", with: "time_embedding.layers.0.")
            .replacingOccurrences(of: "time_embedding.2.", with: "time_embedding.layers.2.")
    }
}
