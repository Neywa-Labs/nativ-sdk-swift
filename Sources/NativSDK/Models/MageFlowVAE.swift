import Foundation
import MLX
import MLXNN

final class MageFlowVAE: Module {
    private static func groupNorm(_ width: Int) -> GroupNorm {
        GroupNorm(groupCount: 32, dimensions: width, eps: 1e-6, pytorchCompatible: true)
    }

    private static func conv(
        _ input: Int, _ output: Int, kernel: Int = 1, stride: Int = 1,
        padding: Int = 0, groups: Int = 1, bias: Bool = true
    ) -> Conv2d {
        Conv2d(
            inputChannels: input, outputChannels: output, kernelSize: .init(kernel),
            stride: .init(stride), padding: .init(padding), groups: groups, bias: bias)
    }

    final class DiCo: Module {
        @ModuleInfo var conv1: Conv2d
        @ModuleInfo var conv2: Conv2d
        @ModuleInfo var conv3: Conv2d
        @ModuleInfo(key: "ca_conv") var channelAttention: Conv2d
        @ModuleInfo var conv4: Conv2d
        @ModuleInfo var conv5: Conv2d
        @ModuleInfo var norm1: LayerNorm
        @ModuleInfo var norm2: LayerNorm
        @ModuleInfo(key: "adaLN_modulation") var modulation: MageFlowLayers.Modulation?

        init(_ width: Int, conditioned: Bool) {
            _conv1.wrappedValue = conv(width, width)
            _conv2.wrappedValue = conv(width, width, kernel: 3, padding: 1, groups: width)
            _conv3.wrappedValue = conv(width, width)
            _channelAttention.wrappedValue = conv(width, width)
            _conv4.wrappedValue = conv(width, 4 * width)
            _conv5.wrappedValue = conv(4 * width, width)
            _norm1.wrappedValue = LayerNorm(dimensions: width, eps: 1e-6, affine: !conditioned)
            _norm2.wrappedValue = LayerNorm(dimensions: width, eps: 1e-6, affine: !conditioned)
            if conditioned {
                _modulation.wrappedValue = MageFlowLayers.Modulation(width, chunks: 6)
            }
        }

        private func spatial(_ x: MLXArray) -> MLXArray {
            let hidden = gelu(conv2(conv1(x)))
            let gate = sigmoid(channelAttention(hidden.mean(axes: [1, 2], keepDims: true)))
            return conv3(hidden * gate)
        }

        func callAsFunction(_ x: MLXArray, time: MLXArray? = nil) -> MLXArray {
            if let modulation, let time {
                let weights = split(
                    modulation(time).expandedDimensions(axes: [1, 2]), parts: 6, axis: -1)
                let input = norm1(x) * (1 + weights[1]) + weights[0]
                let hidden = x + weights[2] * spatial(input)
                let residual = norm2(hidden) * (1 + weights[4]) + weights[3]
                return hidden + weights[5] * conv5(gelu(conv4(residual)))
            }
            let hidden = x + spatial(norm1(x))
            return hidden + conv5(gelu(conv4(norm2(hidden))))
        }
    }

    final class Encoder: Module {
        @ModuleInfo(key: "patch_cond_embed") var input: Conv2d
        @ModuleInfo(key: "head_blocks") var head: [DiCo]
        @ModuleInfo(key: "proj_down") var projectDown: Conv2d
        @ModuleInfo(key: "z_proj") var projectLatents: Conv2d
        @ModuleInfo(key: "fuse_proj") var fuse: Conv2d
        @ModuleInfo(key: "t_embedder") var timeEmbedding: MageFlowLayers.TimeEmbedding
        @ModuleInfo var blocks: [DiCo]
        @ModuleInfo(key: "norm_out") var norm: LayerNorm
        @ModuleInfo(key: "proj_out") var output: Conv2d

        override init() {
            _input.wrappedValue = conv(3, 768, kernel: 16, stride: 16)
            _head.wrappedValue = (0 ..< 2).map { _ in DiCo(768, conditioned: false) }
            _projectDown.wrappedValue = conv(768, 384)
            _projectLatents.wrappedValue = conv(128, 384)
            _fuse.wrappedValue = conv(768, 384)
            _timeEmbedding.wrappedValue = MageFlowLayers.TimeEmbedding(384)
            _blocks.wrappedValue = (0 ..< 21).map { _ in DiCo(384, conditioned: true) }
            _norm.wrappedValue = LayerNorm(dimensions: 384, eps: 1e-6)
            _output.wrappedValue = conv(384, 256)
        }

        func callAsFunction(_ image: MLXArray) -> (MLXArray, MLXArray) {
            let encoded = projectDown(head.reduce(input(image)) { $1($0) })
            let zeros = MLXArray.zeros(
                [image.dim(0), encoded.dim(1), encoded.dim(2), 128], dtype: image.dtype)
            let time = timeEmbedding(.zeros([image.dim(0)], dtype: image.dtype), dtype: image.dtype)
            let fused = fuse(concatenated([encoded, projectLatents(zeros)], axis: -1))
            let hidden = blocks.reduce(fused) { $1($0, time: time) }
            let moments = split(output(norm(hidden)), parts: 2, axis: -1)
            return (moments[0], clip(moments[1], min: -20, max: 10))
        }
    }

    final class Residual: Module, UnaryLayer {
        @ModuleInfo var norm1: GroupNorm
        @ModuleInfo var conv1: Conv2d
        @ModuleInfo var norm2: GroupNorm
        @ModuleInfo var conv2: Conv2d

        init(_ width: Int) {
            _norm1.wrappedValue = groupNorm(width)
            _conv1.wrappedValue = conv(width, width, kernel: 3, padding: 1)
            _norm2.wrappedValue = groupNorm(width)
            _conv2.wrappedValue = conv(width, width, kernel: 3, padding: 1)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            x + conv2(silu(norm2(conv1(silu(norm1(x))))))
        }
    }

    final class LocalAttention: Module, UnaryLayer {
        @ModuleInfo var norm: GroupNorm
        @ModuleInfo(key: "q") var query: Conv2d
        @ModuleInfo(key: "k") var key: Conv2d
        @ModuleInfo(key: "v") var value: Conv2d
        @ModuleInfo(key: "proj_out") var output: Conv2d
        private let patch = 32

        init(_ width: Int) {
            _norm.wrappedValue = groupNorm(width)
            _query.wrappedValue = conv(width, width)
            _key.wrappedValue = conv(width, width)
            _value.wrappedValue = conv(width, width)
            _output.wrappedValue = conv(width, width)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (batch, h, w, channels) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
            let rows = (h + patch - 1) / patch
            let columns = (w + patch - 1) / patch
            func pack(_ input: MLXArray) -> MLXArray {
                var input = input
                if h != rows * patch {
                    let index = clip(arange(rows * patch, dtype: .int32), min: 0, max: h - 1)
                    input = take(input, index, axis: 1)
                }
                if w != columns * patch {
                    let index = clip(arange(columns * patch, dtype: .int32), min: 0, max: w - 1)
                    input = take(input, index, axis: 2)
                }
                return input.reshaped(batch, rows, patch, columns, patch, channels)
                    .transposed(0, 1, 3, 2, 4, 5)
                    .reshaped(batch * rows * columns, patch * patch, channels).asType(.float32)
            }
            let normalized = norm(x)
            let q = pack(query(normalized))
            let k = pack(key(normalized))
            let v = pack(value(normalized))
            let logits = matmul(q, k.swappedAxes(-1, -2)) * (1 / sqrt(Float(channels)))
            let attended = matmul(softmax(logits, axis: -1), v).asType(x.dtype)
            let spatial = attended.reshaped(batch, rows, columns, patch, patch, channels)
                .transposed(0, 1, 3, 2, 4, 5)
                .reshaped(batch, rows * patch, columns * patch, channels)[0..., ..<h, ..<w]
            return x + output(spatial)
        }
    }

    final class CodecDecoder: Module {
        @ModuleInfo(key: "conv_in") var input: Conv2d
        @ModuleInfo(key: "block") var blocks: Sequential
        @ModuleInfo(key: "norm_out") var norm: GroupNorm
        @ModuleInfo(key: "conv_out") var output: Conv2d

        override init() {
            _input.wrappedValue = conv(128, 384, kernel: 3, padding: 1)
            _blocks.wrappedValue = Sequential(
                layers:
                    Residual(384), LocalAttention(384), Residual(384), LocalAttention(384),
                Residual(384))
            _norm.wrappedValue = groupNorm(384)
            _output.wrappedValue = conv(384, 384, kernel: 3, padding: 1)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            output(silu(norm(blocks(input(x)))))
        }
    }

    final class LatentConditioner: Module {
        @ModuleInfo var decoder: CodecDecoder

        override init() {
            _decoder.wrappedValue = CodecDecoder()
        }
    }

    final class SpatialEmbedding: Module {
        @ModuleInfo var proj1: Conv2d
        @ModuleInfo var proj2: Conv2d

        override init() {
            _proj1.wrappedValue = conv(3, 128, kernel: 16, stride: 16, bias: false)
            _proj2.wrappedValue = conv(512, 384)
        }

        func callAsFunction(_ image: MLXArray, condition: MLXArray) -> MLXArray {
            proj2(concatenated([proj1(image), condition], axis: -1))
        }
    }

    final class PixelEmbedding: Module {
        @ModuleInfo var linear: Linear

        override init() {
            _linear.wrappedValue = Linear(99, 32)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let patch = 16
            let coordinates = linspace(Float(0), Float(1), count: patch)
            let xPos = broadcast(coordinates.reshaped(1, patch), to: [patch, patch])
                .reshaped(patch * patch, 1, 1)
            let yPos = broadcast(coordinates.reshaped(patch, 1), to: [patch, patch])
                .reshaped(patch * patch, 1, 1)
            let frequencies = linspace(Float(0), Float(8), count: 8)
            let fx = frequencies.reshaped(1, 8, 1)
            let fy = frequencies.reshaped(1, 1, 8)
            let weights = 1 / (1 + fx * fy)
            let dct = (cos(Float.pi * xPos * fx) * cos(Float.pi * yPos * fy) * weights)
                .reshaped(1, patch * patch, 64).asType(x.dtype)
            return linear(
                concatenated([x, broadcast(dct, to: [x.dim(0), patch * patch, 64])], axis: -1))
        }
    }

    final class PixelBlock: Module {
        @ModuleInfo(key: "in_ln") var norm: LayerNorm
        @ModuleInfo(key: "linear_1") var first: Linear
        @ModuleInfo(key: "linear_2") var second: Linear
        @ModuleInfo(key: "adaLN_modulation") var modulation: MageFlowLayers.Modulation

        override init() {
            _norm.wrappedValue = LayerNorm(dimensions: 32, eps: 1e-6)
            _first.wrappedValue = Linear(32, 32)
            _second.wrappedValue = Linear(32, 32)
            _modulation.wrappedValue = MageFlowLayers.Modulation(32, chunks: 3)
        }

        func callAsFunction(_ x: MLXArray, condition: MLXArray) -> MLXArray {
            let fields = split(modulation(condition), parts: 3, axis: -1)
            let input = norm(x) * (1 + fields[1]) + fields[0]
            return x + fields[2] * second(silu(first(input)))
        }
    }

    final class PixelDecoder: Module {
        @ModuleInfo(key: "cond_embed") var conditionEmbedding: Linear
        @ModuleInfo(key: "input_proj") var input: Linear
        @ModuleInfo(key: "res_blocks") var blocks: [PixelBlock]

        override init() {
            _conditionEmbedding.wrappedValue = Linear(384, 16 * 16 * 32)
            _input.wrappedValue = Linear(32, 32)
            _blocks.wrappedValue = (0 ..< 3).map { _ in PixelBlock() }
        }

        func callAsFunction(_ pixels: MLXArray, condition: MLXArray) -> MLXArray {
            let projected = conditionEmbedding(condition).reshaped(condition.dim(0), 256, 32)
            return blocks.reduce(input(pixels)) { $1($0, condition: projected) }
        }
    }

    final class PixelOutput: Module {
        @ModuleInfo var norm: RMSNorm
        @ModuleInfo var linear: Linear

        override init() {
            _norm.wrappedValue = RMSNorm(dimensions: 32, eps: 1e-6)
            _linear.wrappedValue = Linear(32, 3)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear(norm(x))
        }
    }

    final class Decoder: Module {
        @ModuleInfo(key: "t_embedder") var timeEmbedding: MageFlowLayers.TimeEmbedding
        @ModuleInfo(key: "y_embedder_x") var pixelCondition: Conv2d
        @ModuleInfo(key: "x_embedder") var pixels: PixelEmbedding
        @ModuleInfo(key: "s_embedder") var spatialCondition: SpatialEmbedding
        @ModuleInfo var blocks: [DiCo]
        @ModuleInfo(key: "dec_net") var pixelDecoder: PixelDecoder
        @ModuleInfo(key: "final_layer") var output: PixelOutput
        @ModuleInfo(key: "y_embedder") var latentConditioner: LatentConditioner

        override init() {
            _timeEmbedding.wrappedValue = MageFlowLayers.TimeEmbedding(384)
            _pixelCondition.wrappedValue = conv(384, 32 * 256)
            _pixels.wrappedValue = PixelEmbedding()
            _spatialCondition.wrappedValue = SpatialEmbedding()
            _blocks.wrappedValue = (0 ..< 21).map { _ in DiCo(384, conditioned: true) }
            _pixelDecoder.wrappedValue = PixelDecoder()
            _output.wrappedValue = PixelOutput()
            _latentConditioner.wrappedValue = LatentConditioner()
        }

        func callAsFunction(_ latents: MLXArray) -> MLXArray {
            let condition = latentConditioner.decoder(latents)
            let (batch, h, w) = (condition.dim(0), condition.dim(1), condition.dim(2))
            let dtype = condition.dtype
            let noise = MLXArray.zeros([batch, h * 16, w * 16, 3], dtype: dtype)
            let time = timeEmbedding(.zeros([batch], dtype: dtype), dtype: dtype)
            let spatial = blocks.reduce(spatialCondition(noise, condition: condition)) {
                $1($0, time: time)
            }
            .reshaped(batch * h * w, 384)
            let pixelFeatures = pixelCondition(condition).reshaped(batch * h * w, 32, 256)
                .transposed(0, 2, 1)
            let rows = concatenated(
                [MLXArray.zeros([batch * h * w, 256, 3], dtype: dtype), pixelFeatures], axis: -1)
            let decoded = output(pixelDecoder(pixels(rows), condition: spatial))
            return decoded.reshaped(batch, h, w, 16, 16, 3)
                .transposed(0, 1, 3, 2, 4, 5).reshaped(batch, h * 16, w * 16, 3)
        }
    }

    @ModuleInfo(key: "dconv_encoder") var encoder: Encoder?
    @ModuleInfo(key: "decoder_model") var decoder: Decoder

    init(editing: Bool) {
        if editing { _encoder.wrappedValue = Encoder() }
        _decoder.wrappedValue = Decoder()
    }

    func encode(_ rgb: MLXArray, key: MLXArray) throws -> MLXArray {
        guard let encoder else {
            throw ModelExecutionError.invalidOptions(
                "Mage-Flow editing requires the Mage-VAE encoder")
        }
        let (mean, logVariance) = encoder(rgb)
        return mean + exp(0.5 * logVariance)
            * MLXRandom.normal(mean.shape, dtype: mean.dtype, key: key)
    }

    func decode(_ latents: MLXArray) -> MLXArray {
        decoder(latents)
    }

    static func checkpointWeights(
        _ weights: [String: MLXArray], editing: Bool, mlxLayout: Bool
    ) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (name, value) in weights where !name.hasSuffix(".num_batches_tracked") {
            var key = name
            if key.hasPrefix("student.dconv_encoder.") {
                key = String(key.dropFirst("student.".count))
            } else if key.hasPrefix("pipeline."), !key.hasPrefix("pipeline.y_embedder.encoder.") {
                key = "decoder_model." + key.dropFirst("pipeline.".count)
            }
            guard key.hasPrefix("decoder_model.") || (editing && key.hasPrefix("dconv_encoder."))
            else { continue }
            key = key.replacingOccurrences(
                of: ".adaLN_modulation.1.", with: ".adaLN_modulation.linear."
            )
            .replacingOccurrences(of: ".ca.1.", with: ".ca_conv.")
            .replacingOccurrences(of: ".t_embedder.mlp.0.", with: ".t_embedder.linear_1.")
            .replacingOccurrences(of: ".t_embedder.mlp.2.", with: ".t_embedder.linear_2.")
            .replacingOccurrences(of: ".x_embedder.embedder.0.", with: ".x_embedder.linear.")
            .replacingOccurrences(
                of: ".y_embedder.decoder.block.", with: ".y_embedder.decoder.block.layers.")
            if key.contains(".dec_net.res_blocks.") {
                key = key.replacingOccurrences(of: ".mlp.0.", with: ".linear_1.")
                    .replacingOccurrences(of: ".mlp.2.", with: ".linear_2.")
            }
            result[key] = value.ndim == 4 && !mlxLayout ? value.transposed(0, 2, 3, 1) : value
        }
        return result
    }
}
