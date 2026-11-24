import Foundation
import MLX
import MLXNN

struct ErnieImageTransformerConfig {
    let width: Int
    let feedForwardWidth: Int
    let channels: Int
    let layerCount: Int
    let heads: Int
    let textWidth: Int
    let ropeAxes: [Int]
    let ropeTheta: Float
    let eps: Float
    let normalizeQueriesAndKeys: Bool

    var headWidth: Int { width / heads }

    init(_ config: DiffusionConfiguration) throws {
        width = config.int("hidden_size", 4096)
        feedForwardWidth = config.int("ffn_hidden_size", 12288)
        channels = config.int("in_channels", 128)
        layerCount = config.int("num_layers", 36)
        heads = config.int("num_attention_heads", 32)
        textWidth = config.int("text_in_dim", 3072)
        ropeAxes = config.values["rope_axes_dim"] as? [Int] ?? [32, 48, 48]
        ropeTheta = config.float("rope_theta", 256)
        eps = config.float("eps", 1e-6)
        normalizeQueriesAndKeys = config.bool("qk_layernorm", true)
        guard width > 0, width.isMultiple(of: 2), feedForwardWidth > 0, channels > 0,
            layerCount > 0, heads > 0, width.isMultiple(of: heads), textWidth > 0,
            ropeAxes.count == 3, ropeAxes.allSatisfy({ $0 > 0 && $0.isMultiple(of: 2) }),
            ropeAxes.reduce(0, +) == headWidth, ropeTheta.isFinite, ropeTheta > 0,
            eps.isFinite, eps > 0, config.int("patch_size", 1) == 1,
            config.int("out_channels", 128) == channels
        else {
            throw ModelExecutionError.invalidOptions(
                "Invalid ERNIE-Image transformer configuration")
        }
    }
}

private final class ErnieImagePatchEmbedding: Module {
    @ModuleInfo var proj: Conv2d

    init(_ config: ErnieImageTransformerConfig) {
        _proj.wrappedValue = Conv2d(
            inputChannels: config.channels, outputChannels: config.width, kernelSize: 1)
    }
}

private final class ErnieImageTimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var input: Linear
    @ModuleInfo(key: "linear_2") var output: Linear
    let width: Int

    init(_ width: Int) {
        _input.wrappedValue = Linear(width, width)
        _output.wrappedValue = Linear(width, width)
        self.width = width
    }

    func callAsFunction(_ timestep: MLXArray) -> MLXArray {
        let embedding = diffusionTimeEmbedding(timestep, dimensions: width, sineFirst: true)
            .asType(timestep.dtype)
        return output(silu(input(embedding)))
    }
}

struct ErnieImageRotary {
    let cosine: MLXArray
    let sine: MLXArray

    init(config: ErnieImageTransformerConfig, positions: MLXArray, dtype: DType) {
        let axes = config.ropeAxes.enumerated().map { axis, width in
            let powers = MLXArray(stride(from: 0, to: width, by: 2).map(Float.init)) / width
            let omega = 1 / pow(MLXArray(config.ropeTheta), powers)
            let angles = positions[0..., 0..., axis, .newAxis].asType(.float32) * omega
            return stacked([angles, angles], axis: -1)
                .reshaped(positions.dim(0), positions.dim(1), width)
        }
        let phase = concatenated(axes, axis: -1).expandedDimensions(axis: 1)
        cosine = cos(phase).asType(dtype)
        sine = sin(phase).asType(dtype)
    }

    func apply(_ values: MLXArray) -> MLXArray {
        // Frequencies are interleaved within each axis. The rotation pairs the two head halves.
        values * cosine + RotaryEmbeddingOps.rotateHalf(values) * sine
    }
}

private final class ErnieImageAttention: Module {
    @ModuleInfo(key: "to_q") var query: Linear
    @ModuleInfo(key: "to_k") var key: Linear
    @ModuleInfo(key: "to_v") var value: Linear
    @ModuleInfo(key: "to_out") var output: Linear
    @ModuleInfo(key: "norm_q") var queryNorm: RMSNorm?
    @ModuleInfo(key: "norm_k") var keyNorm: RMSNorm?

    let config: ErnieImageTransformerConfig

    init(_ config: ErnieImageTransformerConfig) {
        self.config = config
        _query.wrappedValue = Linear(config.width, config.width, bias: false)
        _key.wrappedValue = Linear(config.width, config.width, bias: false)
        _value.wrappedValue = Linear(config.width, config.width, bias: false)
        _output.wrappedValue = Linear(config.width, config.width, bias: false)
        if config.normalizeQueriesAndKeys {
            _queryNorm.wrappedValue = RMSNorm(dimensions: config.headWidth, eps: config.eps)
            _keyNorm.wrappedValue = RMSNorm(dimensions: config.headWidth, eps: config.eps)
        }
    }

    func callAsFunction(
        _ input: MLXArray, rotary: ErnieImageRotary, mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let (batch, tokens) = (input.dim(0), input.dim(1))
        let shape = [batch, tokens, config.heads, config.headWidth]
        var q = query(input).reshaped(shape)
        var k = key(input).reshaped(shape)
        if let queryNorm, let keyNorm {
            q = queryNorm(q)
            k = keyNorm(k)
        }
        let v = value(input).reshaped(shape).transposed(0, 2, 1, 3)
        let attended = MLXFast.scaledDotProductAttention(
            queries: rotary.apply(q.transposed(0, 2, 1, 3)),
            keys: rotary.apply(k.transposed(0, 2, 1, 3)), values: v,
            scale: 1 / sqrt(Float(config.headWidth)), mask: mask)
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, tokens, config.width))
    }
}

private final class ErnieImageMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "linear_fc2") var down: Linear

    init(_ config: ErnieImageTransformerConfig) {
        _gate.wrappedValue = Linear(config.width, config.feedForwardWidth, bias: false)
        _up.wrappedValue = Linear(config.width, config.feedForwardWidth, bias: false)
        _down.wrappedValue = Linear(config.feedForwardWidth, config.width, bias: false)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        down(up(input) * gelu(gate(input)))
    }
}

private final class ErnieImageTransformerBlock: Module {
    @ModuleInfo(key: "adaLN_sa_ln") var attentionNorm: RMSNorm
    @ModuleInfo(key: "self_attention") var attention: ErnieImageAttention
    @ModuleInfo(key: "adaLN_mlp_ln") var mlpNorm: RMSNorm
    let mlp: ErnieImageMLP

    init(_ config: ErnieImageTransformerConfig) {
        _attentionNorm.wrappedValue = RMSNorm(dimensions: config.width, eps: config.eps)
        _attention.wrappedValue = ErnieImageAttention(config)
        _mlpNorm.wrappedValue = RMSNorm(dimensions: config.width, eps: config.eps)
        mlp = ErnieImageMLP(config)
    }

    func callAsFunction(
        _ input: MLXArray, modulation: [MLXArray], condition: ErnieImageCondition
    ) -> MLXArray {
        func modulate(_ input: MLXArray, shift: MLXArray, scale: MLXArray) -> MLXArray {
            (input.asType(.float32) * (1 + scale) + shift).asType(input.dtype)
        }
        let attended = attention(
            modulate(attentionNorm(input), shift: modulation[0], scale: modulation[1]),
            rotary: condition.rotary, mask: condition.mask)
        let hidden = input + (modulation[2] * attended.asType(.float32)).asType(input.dtype)
        let projected = mlp(modulate(mlpNorm(hidden), shift: modulation[3], scale: modulation[4]))
        return hidden + (modulation[5] * projected.asType(.float32)).asType(hidden.dtype)
    }
}

private final class ErnieImageFinalNorm: Module {
    let norm: LayerNorm
    @ModuleInfo var linear: Linear

    init(_ config: ErnieImageTransformerConfig) {
        norm = LayerNorm(dimensions: config.width, eps: config.eps, affine: false)
        _linear.wrappedValue = Linear(config.width, 2 * config.width)
    }

    func callAsFunction(_ input: MLXArray, timestep: MLXArray) -> MLXArray {
        let values = split(linear(timestep).expandedDimensions(axis: 1), parts: 2, axis: -1)
        return norm(input) * (1 + values[0]) + values[1]
    }
}

struct ErnieImageCondition {
    let projectedText: MLXArray
    let rotary: ErnieImageRotary
    let attentionMask: MLXArray

    var mask: MLXFast.ScaledDotProductAttentionMaskMode { .array(attentionMask) }
}

final class ErnieImageTransformer: Module {
    private let config: ErnieImageTransformerConfig
    @ModuleInfo(key: "x_embedder") private var imageEmbedding: ErnieImagePatchEmbedding
    @ModuleInfo(key: "text_proj") private var textProjection: Linear
    @ModuleInfo(key: "time_embedding") private var timestepEmbedding: ErnieImageTimestepEmbedding
    @ModuleInfo(key: "adaln_modulation") private var modulation: Linear
    private let layers: [ErnieImageTransformerBlock]
    @ModuleInfo(key: "final_norm") private var finalNorm: ErnieImageFinalNorm
    @ModuleInfo(key: "final_linear") private var finalLinear: Linear

    init(_ config: ErnieImageTransformerConfig) {
        self.config = config
        _imageEmbedding.wrappedValue = ErnieImagePatchEmbedding(config)
        _textProjection.wrappedValue = Linear(config.textWidth, config.width, bias: false)
        _timestepEmbedding.wrappedValue = ErnieImageTimestepEmbedding(config.width)
        _modulation.wrappedValue = Linear(config.width, 6 * config.width)
        layers = (0 ..< config.layerCount).map { _ in ErnieImageTransformerBlock(config) }
        _finalNorm.wrappedValue = ErnieImageFinalNorm(config)
        _finalLinear.wrappedValue = Linear(config.width, config.channels)
    }

    func condition(
        text: MLXArray, lengths: [Int], height: Int, width: Int, dtype: DType
    ) -> ErnieImageCondition {
        let batch = text.dim(0)
        let imageTokens = height * width
        let textTokens = text.dim(1)
        let grid = MLXArray(0 ..< imageTokens)[.newAxis, 0...]
        let imageShape = [batch, imageTokens]
        let textShape = [batch, textTokens]
        let ends = MLXArray(lengths)[0..., .newAxis]
        let imagePositions = stacked(
            [
                broadcast(ends, to: imageShape),
                broadcast(grid.floorDivide(width), to: imageShape),
                broadcast(grid % width, to: imageShape),
            ], axis: -1)
        let index = broadcast(MLXArray(0 ..< textTokens)[.newAxis, 0...], to: textShape)
        let textPositions = stacked(
            [index, MLXArray.zeros(like: index), MLXArray.zeros(like: index)], axis: -1)
        let rotary = ErnieImageRotary(
            config: config, positions: concatenated([imagePositions, textPositions], axis: 1),
            dtype: dtype)
        let valid = concatenated([MLXArray.ones(imageShape, dtype: .bool), index .< ends], axis: 1)
        let mask = MLX.where(valid, MLXArray(0 as Float), MLXArray(-Float.infinity))
            .reshaped(batch, 1, 1, imageTokens + textTokens).asType(dtype)
        let projectedText = textProjection(text)
        eval(rotary.cosine, rotary.sine, mask, projectedText)
        return ErnieImageCondition(
            projectedText: projectedText, rotary: rotary, attentionMask: mask)
    }

    func callAsFunction(
        _ latents: MLXArray, timestep: MLXArray, condition: ErnieImageCondition
    ) -> MLXArray {
        let (batch, height, width) = (latents.dim(0), latents.dim(2), latents.dim(3))
        let image = imageEmbedding.proj(latents.transposed(0, 2, 3, 1))
            .reshaped(batch, height * width, config.width)
        var sequence = concatenated([image, condition.projectedText], axis: 1)
        let time = timestepEmbedding(timestep)
        let mods = split(
            modulation(silu(time)).asType(.float32).expandedDimensions(axis: 1), parts: 6,
            axis: -1)
        for layer in layers {
            sequence = layer(sequence, modulation: mods, condition: condition)
        }
        let output = finalLinear(
            finalNorm(sequence[0..., ..<(height * width), 0...], timestep: time))
        return output.reshaped(batch, height, width, config.channels).transposed(0, 3, 1, 2)
    }
}
