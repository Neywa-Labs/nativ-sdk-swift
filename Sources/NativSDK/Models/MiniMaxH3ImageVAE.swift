import Foundation
import MLX
import MLXNN

struct MiniMaxH3ImageVAEConfiguration {
    let input: Int
    let output: Int
    let channels: Int
    let blocks: [Int]
    let layersPerBlock: Int
    let spatial: [Int]
    let temporal: [Int]
    let groups: Int
    let epsilon: Float
    let decoderLayers: Int
    let heads: Int
    let headDim: Int
    let registers: Int
    let ffn: Int
    let theta: Float
    let rotary: Int
    let decoderEpsilon: Float
    let mean: [Float]
    let deviation: [Float]
    var ratio: Int { spatial.reduce(1, *) }

    init(_ c: DiffusionConfiguration) throws {
        input = c.int("in_channels", 3)
        output = c.int("out_channels", 3)
        channels = c.int("latent_channels", 24)
        blocks = c.ints("block_out_channels", [128, 256, 256, 512, 512, 1024])
        layersPerBlock = c.int("layers_per_block", 2)
        spatial = c.ints("spatial_downsample_factors", [2, 2, 2, 2, 1, 1])
        temporal = c.ints("temporal_downsample_factors", [1, 2, 2, 1, 1, 1])
        groups = c.int("norm_num_groups", 32)
        epsilon = c.float("norm_eps", 1e-6)
        decoderLayers = c.int("decoder_num_layers", 24)
        heads = c.int("decoder_num_attention_heads", 16)
        headDim = c.int("decoder_attention_head_dim", 64)
        registers = c.int("decoder_num_register_tokens", 4)
        ffn = c.int("decoder_ffn_mult", 4)
        theta = c.float("decoder_rope_theta", 100)
        let fraction = c.float("decoder_rope_dim_ratio", 0.5)
        guard fraction.isFinite, fraction > 0, fraction <= 1, headDim > 0, headDim <= 1024 else {
            throw ModelExecutionError.invalidOptions("Invalid MiniMax-H3-Image VAE rotary geometry")
        }
        rotary = Int(Float(headDim) * fraction)
        decoderEpsilon = c.float("decoder_norm_eps", 1e-5)
        mean = (c.values["latents_mean"] as? [NSNumber])?.map(\.floatValue) ?? []
        deviation = (c.values["latents_std"] as? [NSNumber])?.map(\.floatValue) ?? []
        guard input == 3, output == 3, channels > 0, layersPerBlock > 0,
            !blocks.isEmpty, blocks.count <= 6, groups > 0,
            blocks.allSatisfy({ $0 > 0 && $0.isMultiple(of: groups) }),
            spatial.count == blocks.count, spatial.allSatisfy({ $0 == 1 || $0 == 2 }),
            temporal.count == blocks.count, temporal.allSatisfy({ $0 == 1 || $0 == 2 }),
            zip(spatial, temporal).allSatisfy({ $0.0 == 2 || $0.1 == 1 }),
            c.string("spatial_padding_mode", "reflect") == "reflect",
            decoderLayers > 0, heads > 0, registers >= 0, ffn > 0,
            rotary > 0, rotary.isMultiple(of: 4),
            [theta, epsilon, decoderEpsilon].allSatisfy({ $0.isFinite && $0 > 0 }),
            mean.count == channels, mean.allSatisfy(\.isFinite), deviation.count == channels,
            deviation.allSatisfy({ $0.isFinite && $0 > 0 })
        else {
            throw ModelExecutionError.invalidOptions("Invalid MiniMax-H3-Image VAE configuration")
        }
    }
}

final class MiniMaxH3ImageVAE: Module {
    typealias Config = MiniMaxH3ImageVAEConfiguration
    typealias Layers = MiniMaxH3ImageLayers

    /// With one frame, the temporal padding contributes only zeros. The last
    /// temporal tap of each causal convolution is its entire spatial kernel.
    final class ImageConv: Conv2d {
        let leading: Bool
        let trailing: Bool

        init(_ input: Int, _ output: Int, kernel: Int = 3, stride: Int = 1, reflect: Bool = true) {
            leading = reflect && kernel == 3 && stride == 1
            trailing = reflect && kernel == 3
            super.init(
                inputChannels: input, outputChannels: output, kernelSize: .init(kernel),
                stride: .init(stride))
        }

        override func callAsFunction(_ input: MLXArray) -> MLXArray {
            guard trailing else { return super.callAsFunction(input) }
            var x = input
            for axis in [1, 2] {
                let size = x.dim(axis)
                let indices = (leading ? [1] : []) + Array(0 ..< size) + [size - 2]
                x = take(x, MLXArray(indices), axis: axis)
            }
            return super.callAsFunction(x)
        }
    }

    final class GroupNorm: Module {
        @ParameterInfo var weight: MLXArray
        @ParameterInfo var bias: MLXArray
        let groups: Int
        let epsilon: Float

        init(_ width: Int, _ c: Config) {
            groups = c.groups
            epsilon = c.epsilon
            _weight.wrappedValue = .ones([width])
            _bias.wrappedValue = .zeros([width])
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let batch = x.dim(0)
            let columns = x.dim(-1) / groups
            let grouped = x.reshaped(batch, -1, groups, columns).transposed(0, 2, 1, 3)
                .reshaped(batch, groups, -1)
            let normalized = MLXFast.layerNorm(grouped, eps: epsilon)
            return normalized.reshaped(batch, groups, -1, columns).transposed(0, 2, 1, 3)
                .reshaped(x.shape) * weight + bias
        }
    }

    final class Resnet: Module {
        @ModuleInfo var norm1: GroupNorm
        @ModuleInfo var conv1: ImageConv
        @ModuleInfo var norm2: GroupNorm
        @ModuleInfo var conv2: ImageConv
        @ModuleInfo(key: "conv_shortcut") var shortcut: ImageConv?

        init(_ c: Config, input: Int, output: Int) {
            _norm1.wrappedValue = GroupNorm(input, c)
            _conv1.wrappedValue = ImageConv(input, output)
            _norm2.wrappedValue = GroupNorm(output, c)
            _conv2.wrappedValue = ImageConv(output, output)
            if input != output { _shortcut.wrappedValue = ImageConv(input, output, kernel: 1) }
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = conv1(silu(norm1(x)))
            return (shortcut?(x) ?? x) + conv2(silu(norm2(h)))
        }
    }

    final class Downsample: Module {
        @ModuleInfo var conv: ImageConv
        init(width: Int, stride: Int) {
            _conv.wrappedValue = ImageConv(width, width, stride: stride)
        }
    }

    final class EncoderBlock: Module {
        @ModuleInfo var resnets: [Resnet]
        @ModuleInfo var downsamplers: [Downsample]

        init(_ c: Config, input: Int, output: Int, index: Int) {
            _resnets.wrappedValue = (0 ..< c.layersPerBlock).map {
                Resnet(c, input: $0 == 0 ? input : output, output: output)
            }
            _downsamplers.wrappedValue =
                c.spatial[index] > 1 ? [Downsample(width: output, stride: c.spatial[index])] : []
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = resnets.reduce(x) { $1($0) }
            return downsamplers.reduce(h) { $1.conv($0) }
        }
    }

    final class Encoder: Module {
        @ModuleInfo(key: "conv_in") var input: ImageConv
        @ModuleInfo(key: "down_blocks") var blocks: [EncoderBlock]
        @ModuleInfo(key: "norm_out") var norm: GroupNorm
        @ModuleInfo(key: "conv_out") var output: ImageConv

        init(_ c: Config) {
            _input.wrappedValue = ImageConv(c.input, c.blocks[0])
            _blocks.wrappedValue = c.blocks.enumerated().map { index, channels in
                EncoderBlock(c, input: c.blocks[max(0, index - 1)], output: channels, index: index)
            }
            _norm.wrappedValue = GroupNorm(c.blocks.last!, c)
            _output.wrappedValue = ImageConv(c.blocks.last!, c.channels * 2)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            output(silu(norm(blocks.reduce(input(x)) { $1($0) })))
        }
    }

    final class DecoderBlock: Module {
        @ModuleInfo var norm1: RMSNorm
        @ModuleInfo var attn: Layers.Attention
        @ParameterInfo var scale1: MLXArray
        @ModuleInfo var norm2: RMSNorm
        @ModuleInfo var ff: Layers.FeedForward
        @ParameterInfo var scale2: MLXArray

        init(_ c: Config, width: Int) {
            _norm1.wrappedValue = RMSNorm(dimensions: width, eps: c.decoderEpsilon)
            _norm2.wrappedValue = RMSNorm(dimensions: width, eps: c.decoderEpsilon)
            _attn.wrappedValue = .init(
                width: width, heads: c.heads, dimension: c.headDim,
                epsilon: c.decoderEpsilon, bias: true)
            _ff.wrappedValue = .init(width: width, intermediate: width * c.ffn, bias: true)
            _scale1.wrappedValue = .zeros([width])
            _scale2.wrappedValue = .zeros([width])
        }

        func callAsFunction(_ x: MLXArray, rotary: Layers.Rotary) -> MLXArray {
            let h = x + attn(norm1(x.asType(.float32)).asType(x.dtype), rotary: rotary) * scale1
            return h + ff(norm2(h.asType(.float32)).asType(h.dtype)) * scale2
        }
    }

    final class Decoder: Module {
        @ModuleInfo(key: "proj_in") var input: Linear
        @ParameterInfo(key: "register_tokens") var registers: MLXArray
        @ModuleInfo(key: "transformer_blocks") var blocks: [DecoderBlock]
        @ModuleInfo(key: "norm_out") var norm: LayerNorm
        @ModuleInfo(key: "proj_out") var output: Linear
        let config: Config
        let width: Int

        init(_ c: Config) {
            config = c
            width = c.heads * c.headDim
            _input.wrappedValue = Linear(c.channels, width)
            _registers.wrappedValue = .zeros([1, c.registers, width])
            _blocks.wrappedValue = (0 ..< c.decoderLayers).map { _ in
                DecoderBlock(c, width: c.heads * c.headDim)
            }
            _norm.wrappedValue = LayerNorm(dimensions: width, eps: c.decoderEpsilon)
            _output.wrappedValue = Linear(width, c.output * c.ratio * c.ratio)
        }

        func callAsFunction(_ latents: MLXArray) -> MLXArray {
            let batch = latents.dim(0)
            let (height, columns) = (latents.dim(1), latents.dim(2))
            let count = height * columns
            let x = input(latents.reshaped(batch, count, config.channels))
            let special = broadcast(
                registers.asType(x.dtype), to: [batch, config.registers, width])
            let initial = concatenated(
                [x, special, MLXArray.zeros([batch, 1, width], dtype: x.dtype)], axis: 1)
            let y = 2 * (arange(0, height, dtype: .float32) + 0.5) / Float(height) - 1
            let xPositions = 2 * (arange(0, columns, dtype: .float32) + 0.5) / Float(columns) - 1
            let grid = stacked(
                [
                    broadcast(y.reshaped(height, 1), to: [height, columns]),
                    broadcast(xPositions.reshaped(1, columns), to: [height, columns]),
                ], axis: -1
            ).reshaped(1, count, 2)
            let positions = concatenated(
                [grid, MLXArray.zeros([1, config.registers + 1, 2])], axis: 1)
            let frequency =
                1
                / pow(
                    config.theta,
                    arange(0, config.rotary / 4, dtype: .float32) * (4 / Float(config.rotary)))
            let angles = (2 * Float.pi * positions[.ellipsis, .newAxis] * frequency)
                .reshaped(1, -1, 1, config.rotary / 2)
            let doubled = concatenated([angles, angles], axis: -1)
            let rotary = (cos(doubled), sin(doubled))
            let normalized = norm(blocks.reduce(initial) { $1($0, rotary: rotary) })
            let patch = config.ratio
            return output(normalized[0..., ..<count, 0...])
                .reshaped(batch, height, columns, config.output, patch, patch)
                .transposed(0, 1, 4, 2, 5, 3)
                .reshaped(batch, height * patch, columns * patch, config.output)
        }
    }

    @ModuleInfo var encoder: Encoder
    @ModuleInfo(key: "quant_conv") var quant: ImageConv
    @ModuleInfo(key: "post_quant_conv") var postQuant: ImageConv
    @ModuleInfo var decoder: Decoder
    let config: Config

    init(_ config: Config) {
        self.config = config
        _encoder.wrappedValue = Encoder(config)
        _quant.wrappedValue = ImageConv(2 * config.channels, 2 * config.channels, kernel: 1)
        _postQuant.wrappedValue = ImageConv(config.channels, config.channels, kernel: 1)
        _decoder.wrappedValue = Decoder(config)
    }

    func moments(_ pixels: MLXArray) throws -> MLXArray {
        try tiled(pixels, tile: 256, overlap: 64, alignment: config.ratio, divisor: config.ratio) {
            quant(encoder($0))
        }
    }

    func decode(_ latents: MLXArray) throws -> MLXArray {
        let ratio = config.ratio
        return try tiled(
            latents, tile: 256 / ratio, overlap: 64 / ratio, alignment: 1, multiplier: ratio
        ) {
            decoder(postQuant($0))
        }
    }

    private func tiled(
        _ x: MLXArray, tile: Int, overlap: Int, alignment: Int, divisor: Int = 1,
        multiplier: Int = 1, apply: (MLXArray) -> MLXArray
    ) throws -> MLXArray {
        let y = Self.tilePlan(x.dim(1), tile: tile, overlap: overlap, alignment: alignment)
        let h = Self.tilePlan(x.dim(2), tile: tile, overlap: overlap, alignment: alignment)
        var outputs: [[MLXArray]] = []
        for rows in y.ranges {
            var row: [MLXArray] = []
            for cols in h.ranges {
                try Task.checkCancellation()
                let tile = apply(x[0..., rows, cols, 0...])
                eval(tile)
                row.append(tile)
            }
            outputs.append(row)
        }

        let yOverlap = y.overlap.map { $0 * multiplier / divisor }
        let xOverlap = h.overlap.map { $0 * multiplier / divisor }
        var rows: [MLXArray] = []
        for (rowIndex, tiles) in outputs.enumerated() {
            var columns: [MLXArray] = []
            for (columnIndex, tile) in tiles.enumerated() {
                var tile = tile
                if rowIndex > 0 {
                    tile = Self.blend(
                        outputs[rowIndex - 1][columnIndex], tile, yOverlap[rowIndex - 1], axis: 1)
                }
                if columnIndex > 0 {
                    tile = Self.blend(
                        tiles[columnIndex - 1], tile, xOverlap[columnIndex - 1], axis: 2)
                }
                let height = tile.dim(1) - (rowIndex < yOverlap.count ? yOverlap[rowIndex] : 0)
                let width = tile.dim(2) - (columnIndex < xOverlap.count ? xOverlap[columnIndex] : 0)
                columns.append(tile[0..., ..<height, ..<width, 0...])
            }
            rows.append(concatenated(columns, axis: 2))
        }
        return concatenated(rows, axis: 1)
    }

    static func tilePlan(_ length: Int, tile: Int, overlap: Int, alignment: Int)
        -> (ranges: [Range<Int>], overlap: [Int])
    {
        guard length > tile else { return ([0 ..< length], []) }
        let stride = tile - overlap
        let count = 1 + (length - tile + stride - 1) / stride
        var seams = Array(repeating: overlap, count: count - 1)
        let unused = (tile * count - overlap * (count - 1) - length) / alignment
        for index in 0 ..< unused { seams[index % seams.count] += alignment }
        var range = 0 ..< tile
        var ranges = [range]
        for seam in seams {
            let start = range.upperBound - seam
            range = start ..< (start + tile)
            ranges.append(range)
        }
        return (ranges, seams)
    }

    private static func blend(_ left: MLXArray, _ right: MLXArray, _ extent: Int, axis: Int)
        -> MLXArray
    {
        guard extent > 0 else { return right }
        // Put the blend axis first so the same slice logic handles both seams.
        let a = left.swappedAxes(0, axis)
        let b = right.swappedAxes(0, axis)
        let mix = (arange(0, extent, dtype: b.dtype) / Float(extent)).reshaped(extent, 1, 1, 1)
        let seam = a[(a.dim(0) - extent)...] * (1 - mix) + b[..<extent] * mix
        return concatenated([seam, b[extent...]], axis: 0).swappedAxes(0, axis)
    }

    static func checkpointWeights(_ weights: [String: MLXArray], mlxLayout: Bool) -> [String:
        MLXArray]
    {
        Dictionary(
            uniqueKeysWithValues: weights.map { key, tensor in
                var value = tensor
                if tensor.ndim == 5 {
                    value =
                        mlxLayout
                        ? tensor[0..., -1, 0..., 0..., 0...]
                        : tensor[0..., 0..., -1, 0..., 0...].transposed(0, 2, 3, 1)
                    value = contiguous(value)
                } else if tensor.ndim == 4 && !mlxLayout {
                    value = tensor.transposed(0, 2, 3, 1)
                }
                return (Layers.checkpointKey(key), value)
            })
    }
}
