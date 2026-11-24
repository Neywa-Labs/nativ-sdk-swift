import Foundation
import MLX
import MLXNN

/// The image path takes the last temporal kernel slice of each causal convolution.
private func imageConv(_ conv: Conv2d, _ x: MLXArray) -> MLXArray {
    conv(x.transposed(0, 2, 3, 1)).transposed(0, 3, 1, 2)
}

final class QwenImageVAENorm: Module, UnaryLayer {
    @ParameterInfo var gamma: MLXArray
    let scale: Float
    init(_ dimensions: Int) {
        scale = sqrt(Float(dimensions))
        _gamma.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let norm = rsqrt(sum(square(x.asType(.float32)), axis: 1, keepDims: true) + 1e-12)
        return (x.asType(.float32) * norm).asType(x.dtype) * scale * gamma.reshaped([1, -1, 1, 1])
    }
}

final class QwenImageVAEResidual: Module, UnaryLayer {
    @ModuleInfo var norm1: QwenImageVAENorm
    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var norm2: QwenImageVAENorm
    @ModuleInfo var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var shortcut: Conv2d?
    init(_ input: Int, _ output: Int) {
        _norm1.wrappedValue = QwenImageVAENorm(input)
        _norm2.wrappedValue = QwenImageVAENorm(output)
        _conv1.wrappedValue = Conv2d(
            inputChannels: input, outputChannels: output, kernelSize: 3, padding: 1)
        _conv2.wrappedValue = Conv2d(
            inputChannels: output, outputChannels: output, kernelSize: 3, padding: 1)
        _shortcut.wrappedValue =
            input == output
            ? nil : Conv2d(inputChannels: input, outputChannels: output, kernelSize: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = imageConv(conv1, silu(norm1(x)))
        return imageConv(conv2, silu(norm2(h))) + (shortcut.map { imageConv($0, x) } ?? x)
    }
}

final class QwenImageVAEAttention: Module, UnaryLayer {
    @ModuleInfo var norm: QwenImageVAENorm
    @ModuleInfo(key: "to_qkv") var qkv: Conv2d
    @ModuleInfo var proj: Conv2d
    init(_ dim: Int) {
        _norm.wrappedValue = QwenImageVAENorm(dim)
        _qkv.wrappedValue = Conv2d(inputChannels: dim, outputChannels: 3 * dim, kernelSize: 1)
        _proj.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let c = x.dim(1)
        let h = x.dim(2)
        let w = x.dim(3)
        let qkv = split(
            imageConv(qkv, norm(x)).reshaped([b, 3 * c, h * w]).transposed(0, 2, 1), parts: 3,
            axis: -1)
        let out = MLXFast.scaledDotProductAttention(
            queries: qkv[0].expandedDimensions(axis: 1),
            keys: qkv[1].expandedDimensions(axis: 1), values: qkv[2].expandedDimensions(axis: 1),
            scale: pow(Float(c), -0.5), mask: .none)
        return x
            + imageConv(proj, out.squeezed(axis: 1).transposed(0, 2, 1).reshaped([b, c, h, w]))
    }
}

final class QwenImageVAEMid: Module, UnaryLayer {
    @ModuleInfo var resnets: [QwenImageVAEResidual]
    @ModuleInfo var attentions: [QwenImageVAEAttention]
    init(_ dim: Int) {
        _resnets.wrappedValue = [QwenImageVAEResidual(dim, dim), QwenImageVAEResidual(dim, dim)]
        _attentions.wrappedValue = [QwenImageVAEAttention(dim)]
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { resnets[1](attentions[0](resnets[0](x))) }
}

final class QwenImageVAEResample: Module, UnaryLayer {
    @ModuleInfo var resample: [Module]
    @ModuleInfo(key: "time_conv") var timeConv: Conv2d?
    let up: Bool
    init(_ dim: Int, up: Bool, temporal: Bool, out: Int? = nil) {
        self.up = up
        _resample.wrappedValue = [
            Identity(),
            Conv2d(
                inputChannels: dim, outputChannels: up ? (out ?? dim / 2) : dim,
                kernelSize: 3, stride: up ? 1 : 2, padding: up ? 1 : 0),
        ]
        _timeConv.wrappedValue =
            temporal
            ? Conv2d(inputChannels: dim, outputChannels: up ? dim * 2 : dim, kernelSize: 1) : nil
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h =
            up
            ? repeated(repeated(x, count: 2, axis: 2), count: 2, axis: 3)
            : padded(x, widths: [.init(0), .init(0), .init((0, 1)), .init((0, 1))])
        return imageConv(resample[1] as! Conv2d, h)
    }
}

final class QwenImageVAEDown: Module, UnaryLayer {
    @ModuleInfo var resnets: [QwenImageVAEResidual]
    @ModuleInfo var downsampler: QwenImageVAEResample?
    let input: Int, output: Int, factor: Int, temporalFactor: Int
    init(input: Int, output: Int, layers: Int, down: Bool, temporal: Bool) {
        self.input = input
        self.output = output
        factor = down ? 2 : 1
        temporalFactor = temporal ? 2 : 1
        _resnets.wrappedValue = (0 ..< layers).map {
            QwenImageVAEResidual($0 == 0 ? input : output, output)
        }
        _downsampler.wrappedValue =
            down ? QwenImageVAEResample(output, up: false, temporal: temporal) : nil
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for block in resnets { h = block(h) }
        if let downsampler { h = downsampler(h) }
        let b = x.dim(0)
        let height = x.dim(2) / factor
        let width = x.dim(3) / factor
        // A first image frame is preceded by zeros in a temporal downsample shortcut.
        var shortcut = x.expandedDimensions(axis: 2)
        if temporalFactor == 2 {
            shortcut = concatenated(
                [MLXArray.zeros(shortcut.shape, dtype: x.dtype), shortcut], axis: 2)
        }
        shortcut = shortcut.reshaped([b, input, temporalFactor, height, factor, width, factor])
            .transposed(0, 1, 2, 4, 6, 3, 5).reshaped([
                b, output, input * temporalFactor * factor * factor / output, height, width,
            ])
        return h + mean(shortcut, axis: 2)
    }
}

final class QwenImageVAEUp: Module, UnaryLayer {
    @ModuleInfo var resnets: [QwenImageVAEResidual]
    @ModuleInfo var upsampler: QwenImageVAEResample?
    @ModuleInfo var upsamplers: [QwenImageVAEResample]?
    let input: Int, output: Int, temporalFactor: Int, residual: Bool, up: Bool
    init(input: Int, output: Int, layers: Int, up: Bool, temporal: Bool, residual: Bool) {
        self.input = input
        self.output = output
        self.temporalFactor = temporal ? 2 : 1
        self.residual = residual
        self.up = up
        _resnets.wrappedValue = (0 ... layers).map {
            QwenImageVAEResidual($0 == 0 ? input : output, output)
        }
        _upsampler.wrappedValue =
            up && residual
            ? QwenImageVAEResample(output, up: true, temporal: temporal, out: output) : nil
        _upsamplers.wrappedValue =
            up && !residual ? [QwenImageVAEResample(output, up: true, temporal: temporal)] : nil
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for block in resnets { h = block(h) }
        if let upsamplers { h = upsamplers[0](h) }
        if let upsampler { h = upsampler(h) }
        if residual && up {
            let b = x.dim(0)
            let height = x.dim(2)
            let width = x.dim(3)
            let duplicate = repeated(x, count: output * temporalFactor * 4 / input, axis: 1)
                .reshaped([b, output, temporalFactor, 2, 2, height, width])
                .transposed(0, 1, 2, 5, 3, 6, 4).reshaped([
                    b, output, temporalFactor, height * 2, width * 2,
                ])
            h = h + duplicate[0..., 0..., temporalFactor - 1, 0..., 0...]
        }
        return h
    }
}

final class QwenImageVAEEncoder: Module, UnaryLayer {
    @ModuleInfo(key: "conv_in") var input: Conv2d
    @ModuleInfo(key: "down_blocks") var blocks: [Module]
    @ModuleInfo(key: "mid_block") var middle: QwenImageVAEMid
    @ModuleInfo(key: "norm_out") var norm: QwenImageVAENorm
    @ModuleInfo(key: "conv_out") var output: Conv2d
    init(
        base: Int, channels: Int, latent: Int, multipliers: [Int], layers: Int, temporal: [Bool],
        residual: Bool
    ) {
        let dims = ([1] + multipliers).map { $0 * base }
        _input.wrappedValue = Conv2d(
            inputChannels: channels, outputChannels: dims[0], kernelSize: 3, padding: 1)
        var blocks = [Module]()
        for i in multipliers.indices {
            let down = i < multipliers.count - 1
            if residual {
                blocks.append(
                    QwenImageVAEDown(
                        input: dims[i], output: dims[i + 1], layers: layers, down: down,
                        temporal: down && temporal[i]))
            } else {
                for j in 0 ..< layers {
                    blocks.append(QwenImageVAEResidual(j == 0 ? dims[i] : dims[i + 1], dims[i + 1]))
                }
                if down {
                    blocks.append(
                        QwenImageVAEResample(dims[i + 1], up: false, temporal: temporal[i]))
                }
            }
        }
        _blocks.wrappedValue = blocks
        _middle.wrappedValue = QwenImageVAEMid(dims.last!)
        _norm.wrappedValue = QwenImageVAENorm(dims.last!)
        _output.wrappedValue = Conv2d(
            inputChannels: dims.last!, outputChannels: 2 * latent, kernelSize: 3, padding: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = imageConv(input, x)
        for block in blocks { h = (block as! any UnaryLayer)(h) }
        return imageConv(output, silu(norm(middle(h))))
    }
}

final class QwenImageVAEDecoder: Module, UnaryLayer {
    @ModuleInfo(key: "conv_in") var input: Conv2d
    @ModuleInfo(key: "mid_block") var middle: QwenImageVAEMid
    @ModuleInfo(key: "up_blocks") var blocks: [QwenImageVAEUp]
    @ModuleInfo(key: "norm_out") var norm: QwenImageVAENorm
    @ModuleInfo(key: "conv_out") var output: Conv2d
    init(
        base: Int, channels: Int, latent: Int, multipliers: [Int], layers: Int, temporal: [Bool],
        residual: Bool
    ) {
        let dims = ([multipliers.last!] + multipliers.reversed()).map { $0 * base }
        _input.wrappedValue = Conv2d(
            inputChannels: latent, outputChannels: dims[0], kernelSize: 3, padding: 1)
        _middle.wrappedValue = QwenImageVAEMid(dims[0])
        _blocks.wrappedValue = multipliers.indices.map { i in
            let up = i < multipliers.count - 1
            return QwenImageVAEUp(
                input: i > 0 && !residual ? dims[i] / 2 : dims[i], output: dims[i + 1],
                layers: layers, up: up, temporal: up && temporal[i], residual: residual)
        }
        _norm.wrappedValue = QwenImageVAENorm(dims.last!)
        _output.wrappedValue = Conv2d(
            inputChannels: dims.last!, outputChannels: channels, kernelSize: 3, padding: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = middle(imageConv(input, x))
        for block in blocks { h = block(h) }
        return imageConv(output, silu(norm(h)))
    }
}

final class QwenImageVAE: Module {
    @ModuleInfo var encoder: QwenImageVAEEncoder
    @ModuleInfo var decoder: QwenImageVAEDecoder
    @ModuleInfo(key: "quant_conv") var quant: Conv2d
    @ModuleInfo(key: "post_quant_conv") var postQuant: Conv2d
    init(_ c: DiffusionConfiguration, ming: Bool = false) throws {
        let base = c.int("base_dim", 96)
        let latent = c.int("z_dim", ming ? 16 : 64)
        let multipliers = c.ints("dim_mult", ming ? [1, 2, 4, 4] : [1, 2, 4, 8, 8])
        let layers = c.int("num_res_blocks", 2)
        let temporal =
            c.values["temperal_downsample"] as? [Bool]
            ?? (ming ? [false, true, true] : [false, true, true, true])
        let residual = c.bool("is_residual", !ming)
        let channels = c.int("input_channels", c.int("in_channels", 4))
        guard !multipliers.isEmpty, temporal.count == multipliers.count - 1 else {
            throw ModelExecutionError.invalidOptions("Invalid image VAE block configuration")
        }
        _encoder.wrappedValue = QwenImageVAEEncoder(
            base: base, channels: channels, latent: latent, multipliers: multipliers,
            layers: layers, temporal: temporal, residual: residual)
        _decoder.wrappedValue = QwenImageVAEDecoder(
            base: c.int("decoder_base_dim", base),
            channels: c.int("out_channels", channels), latent: latent,
            multipliers: multipliers, layers: layers, temporal: Array(temporal.reversed()),
            residual: residual)
        _quant.wrappedValue = Conv2d(
            inputChannels: latent * 2, outputChannels: latent * 2, kernelSize: 1)
        _postQuant.wrappedValue = Conv2d(
            inputChannels: latent, outputChannels: latent, kernelSize: 1)
        super.init()
    }
    func encode(_ x: MLXArray) -> MLXArray {
        split(imageConv(quant, encoder(x)), parts: 2, axis: 1)[0]
    }
    func decode(_ x: MLXArray) -> MLXArray {
        clip(decoder(imageConv(postQuant, x)), min: -1, max: 1)
    }
    static func sanitize(_ weights: [String: MLXArray], sourceLayout: Bool) -> [String: MLXArray] {
        weights.mapValues { value in
            if value.ndim == 5 { return value[0..., 0..., -1, 0..., 0...].transposed(0, 2, 3, 1) }
            return sourceLayout && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value
        }.reduce(into: [:]) { result, item in
            result[item.key] = item.key.hasSuffix(".gamma") ? item.value.reshaped([-1]) : item.value
        }
    }
}
