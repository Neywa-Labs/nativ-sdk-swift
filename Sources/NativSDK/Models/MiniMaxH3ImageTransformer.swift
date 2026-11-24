import Foundation
import MLX
import MLXNN

struct MiniMaxH3ImageTransformerConfiguration {
    let width: Int
    let heads: Int
    let headDim: Int
    let layers: Int
    let refinerLayers: Int
    let ffn: Int
    let channels: Int
    let patch: [Int]
    let text: Int
    let frequencies: Int
    let timeHidden: Int
    let timeWidth: Int
    let ropeWidth: Int
    let theta: Float
    let epsilon: Float
    let qkEpsilon: Float
    let finalEpsilon: Float

    init(_ c: DiffusionConfiguration) throws {
        width = c.int("hidden_size", 5376)
        heads = c.int("num_attention_heads", 56)
        headDim = c.int("attention_head_dim", 128)
        layers = c.int("num_layers", 50)
        refinerLayers = c.int("num_refiner_layers", 2)
        ffn = c.int("ffn_dim", 14336)
        channels = c.int("in_channels", 24)
        patch = c.ints("patch_size", [1, 2, 2])
        text = c.int("text_dim", 5120)
        frequencies = c.int("freq_dim", 256)
        timeHidden = c.int("time_embed_hidden_dim", 5376)
        timeWidth = c.int("time_embed_dim", 2688)
        ropeWidth = c.int("rope_freq_dim", 16)
        theta = c.float("rope_theta", 10000)
        epsilon = c.float("norm_eps", 1e-5)
        qkEpsilon = c.float("qk_norm_eps", 1e-5)
        finalEpsilon = c.float("final_norm_eps", 1e-5)
        guard
            [
                width, heads, headDim, layers, ffn, channels, text, frequencies, timeHidden,
                timeWidth, ropeWidth,
            ].allSatisfy({ $0 > 0 }), frequencies > 1, refinerLayers >= 0,
            patch.count == 3, patch[0] == 1, patch[1] > 0, patch[2] > 0,
            ropeWidth <= headDim / 6, c.int("num_pdd_heads", 6) == 6,
            [theta, epsilon, qkEpsilon, finalEpsilon].allSatisfy({ $0.isFinite && $0 > 0 })
        else {
            throw ModelExecutionError.invalidOptions(
                "Invalid MiniMax-H3-Image transformer geometry")
        }
    }

    var patchWidth: Int { channels * patch[1] * patch[2] }
}

enum MiniMaxH3ImageLayers {
    typealias Rotary = (cosine: MLXArray, sine: MLXArray)

    static func rotate(_ x: MLXArray, with rope: Rotary) -> MLXArray {
        let width = rope.cosine.dim(-1)
        let rotary = x[.ellipsis, ..<width]
        let half = width / 2
        let rotated = concatenated(
            [-rotary[.ellipsis, half...], rotary[.ellipsis, ..<half]], axis: -1)
        let combined = rotary * rope.cosine.asType(x.dtype) + rotated * rope.sine.asType(x.dtype)
        if width == x.dim(-1) { return combined }
        return concatenated([combined, x[.ellipsis, width...]], axis: -1)
    }

    final class Attention: Module {
        @ModuleInfo(key: "to_q") var query: Linear
        @ModuleInfo(key: "to_k") var key: Linear
        @ModuleInfo(key: "to_v") var value: Linear
        @ModuleInfo(key: "to_out") var output: Linear
        @ModuleInfo(key: "norm_q") var queryNorm: RMSNorm?
        @ModuleInfo(key: "norm_k") var keyNorm: RMSNorm?
        let heads: Int
        let dimension: Int
        let epsilon: Float

        init(width: Int, heads: Int, dimension: Int, epsilon: Float, bias: Bool) {
            self.heads = heads
            self.dimension = dimension
            self.epsilon = epsilon
            let inner = heads * dimension
            _query.wrappedValue = Linear(width, inner, bias: bias)
            _key.wrappedValue = Linear(width, inner, bias: bias)
            _value.wrappedValue = Linear(width, inner, bias: bias)
            _output.wrappedValue = Linear(inner, width, bias: bias)
            if !bias {
                _queryNorm.wrappedValue = RMSNorm(dimensions: dimension, eps: epsilon)
                _keyNorm.wrappedValue = RMSNorm(dimensions: dimension, eps: epsilon)
            }
        }

        func callAsFunction(_ x: MLXArray, rotary: Rotary? = nil) -> MLXArray {
            let batch = x.dim(0)
            let length = x.dim(1)
            func project(_ projection: Linear, norm: RMSNorm?) -> MLXArray {
                var h = projection(x).reshaped(batch, length, heads, dimension)
                if let norm {
                    h = norm(h)
                } else {
                    let fp = h.asType(.float32)
                    h = (fp * rsqrt(square(fp).mean(axis: -1, keepDims: true) + epsilon))
                        .asType(h.dtype)
                }
                if let rotary { h = rotate(h, with: rotary) }
                return h.transposed(0, 2, 1, 3)
            }
            let values = value(x).reshaped(batch, length, heads, dimension).transposed(0, 2, 1, 3)
            let attended = MLXFast.scaledDotProductAttention(
                queries: project(query, norm: queryNorm), keys: project(key, norm: keyNorm),
                values: values, scale: 1 / sqrt(Float(dimension)), mask: .none)
            return output(attended.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
        }
    }

    final class FeedForward: Module {
        @ModuleInfo(key: "in_proj") var expand: Linear
        @ModuleInfo(key: "out_proj") var reduce: Linear
        let intermediate: Int

        init(width: Int, intermediate: Int, bias: Bool) {
            self.intermediate = intermediate
            _expand.wrappedValue = Linear(width, 2 * intermediate, bias: bias)
            _reduce.wrappedValue = Linear(intermediate, width, bias: bias)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = expand(x)
            return reduce(h[.ellipsis, ..<intermediate] * silu(h[.ellipsis, intermediate...]))
        }
    }

    static func checkpointKey(_ key: String) -> String {
        key.replacingOccurrences(of: ".attn.to_out.0.", with: ".attn.to_out.")
            .replacingOccurrences(of: ".ff.net.0.proj.", with: ".ff.in_proj.")
            .replacingOccurrences(of: ".ff.net.2.", with: ".ff.out_proj.")
    }

    static func projectionDType(_ linear: Linear) -> DType {
        (linear as? QuantizedLinear)?.scales.dtype ?? linear.weight.dtype
    }
}

final class MiniMaxH3ImageTransformer: Module {
    typealias Layers = MiniMaxH3ImageLayers
    typealias Configuration = MiniMaxH3ImageTransformerConfiguration

    final class TimeEmbedding: Module {
        @ModuleInfo(key: "linear_1") var first: Linear
        @ModuleInfo(key: "linear_2") var second: Linear
        let dimensions: Int
        init(_ c: Configuration) {
            dimensions = c.frequencies
            _first.wrappedValue = Linear(c.frequencies, c.timeHidden)
            _second.wrappedValue = Linear(c.timeHidden, c.timeWidth)
        }

        func callAsFunction(_ times: MLXArray) -> MLXArray {
            let half = dimensions / 2
            let exponent = (-log(Float(10000)) * arange(0, half, dtype: .float32)) / Float(half)
            let angles = times[0..., .newAxis] * exp(exponent)[.newAxis, 0...]
            var embedding = concatenated([cos(angles), sin(angles)], axis: -1)
            if dimensions % 2 == 1 {
                embedding = MLX.padded(embedding, widths: [.init(0), .init((0, 1))])
            }
            return second(silu(first(embedding.asType(first.weight.dtype))))
        }
    }

    final class Modulation: Module {
        @ModuleInfo var linear: Linear
        let width: Int
        init(_ c: Configuration, pieces: Int, modalities: Int) {
            width = c.width * pieces
            _linear.wrappedValue = Linear(c.timeWidth, width * modalities)
        }

        func callAsFunction(_ embedding: MLXArray) -> MLXArray {
            linear(silu(embedding).asType(linear.weight.dtype)).reshaped(-1, width)
        }

        func release() {
            linear.weight._updateInternal(MLXArray.zeros([0]))
            linear.bias?._updateInternal(MLXArray.zeros([0]))
        }
    }

    final class OutputNorm: Module {
        @ModuleInfo var norm: RMSNorm
        @ModuleInfo var linear: Linear
        let width: Int

        init(_ c: Configuration) {
            width = c.width
            _norm.wrappedValue = RMSNorm(dimensions: width, eps: c.finalEpsilon)
            _linear.wrappedValue = Linear(c.timeWidth, 2 * width)
        }

        func callAsFunction(_ x: MLXArray, modulation: MLXArray, indices: MLXArray) -> MLXArray {
            let m = take(modulation, indices, axis: 0)
            return norm(x) * (1 + m[.ellipsis, width...]) + m[.ellipsis, ..<width]
        }
    }

    final class Block: Module {
        @ModuleInfo var norm1: RMSNorm
        @ModuleInfo var attn: Layers.Attention
        @ModuleInfo var norm2: RMSNorm
        @ModuleInfo var ff: Layers.FeedForward
        @ModuleInfo(key: "adaln_proj") var modulation: Modulation?
        let width: Int

        init(_ c: Configuration, denoiser: Bool) {
            width = c.width
            _norm1.wrappedValue = RMSNorm(dimensions: c.width, eps: c.epsilon)
            _norm2.wrappedValue = RMSNorm(dimensions: c.width, eps: c.epsilon)
            _attn.wrappedValue = .init(
                width: c.width, heads: c.heads, dimension: c.headDim,
                epsilon: c.qkEpsilon, bias: false)
            _ff.wrappedValue = .init(width: c.width, intermediate: c.ffn, bias: false)
            if denoiser { _modulation.wrappedValue = Modulation(c, pieces: 6, modalities: 3) }
        }

        func refine(_ x: MLXArray) -> MLXArray {
            let h = x + attn(norm1(x))
            return h + ff(norm2(h))
        }

        func callAsFunction(
            _ x: MLXArray, table: MLXArray, indices: MLXArray, rotary: Layers.Rotary
        ) -> MLXArray {
            let m = take(table, indices, axis: 0)
            let attnInput =
                norm1(x) * (1 + m[.ellipsis, width ..< (2 * width)])
                + m[.ellipsis, ..<width]
            let h = x + m[.ellipsis, (2 * width) ..< (3 * width)] * attn(attnInput, rotary: rotary)
            let ffnInput =
                norm2(h) * (1 + m[.ellipsis, (4 * width) ..< (5 * width)])
                + m[.ellipsis, (3 * width) ..< (4 * width)]
            return h + m[.ellipsis, (5 * width)...] * ff(ffnInput)
        }
    }

    final class Refiner: Module {
        @ModuleInfo(key: "refiner_blocks") var blocks: [Block]
        @ModuleInfo(key: "final_norm") var norm: RMSNorm

        init(_ c: Configuration) {
            _blocks.wrappedValue = (0 ..< c.refinerLayers).map { _ in Block(c, denoiser: false) }
            _norm.wrappedValue = RMSNorm(dimensions: c.width, eps: c.finalEpsilon)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            norm(blocks.reduce(x) { $1.refine($0) })
        }
    }

    @ModuleInfo(key: "proj_in") var projection: Linear
    @ModuleInfo(key: "context_embedder") var context: Linear
    @ModuleInfo(key: "time_embedder") var time: TimeEmbedding
    @ModuleInfo(key: "token_refiner") var refiner: Refiner
    @ModuleInfo(key: "transformer_blocks") var blocks: [Block]
    @ModuleInfo(key: "norm_out") var outputNorm: OutputNorm
    @ModuleInfo(key: "pdd_proj_out") var outputs: [Linear]
    let config: Configuration

    init(_ config: Configuration) {
        self.config = config
        _projection.wrappedValue = Linear(config.patchWidth, config.width)
        _context.wrappedValue = Linear(config.text, config.width)
        _time.wrappedValue = TimeEmbedding(config)
        _refiner.wrappedValue = Refiner(config)
        _blocks.wrappedValue = (0 ..< config.layers).map { _ in Block(config, denoiser: true) }
        _outputNorm.wrappedValue = OutputNorm(config)
        _outputs.wrappedValue = (0 ..< 6).map { _ in Linear(config.width, config.patchWidth) }
    }

    struct Step {
        var blocks: [MLXArray] = []
        var output: MLXArray
    }

    func prepareSteps(_ times: [Float]) throws -> [Step] {
        let embeddings = times.map { time(MLXArray([$0, 0.999])) }
        eval(embeddings)
        var steps = embeddings.map { embedding in
            Step(output: outputNorm.linear(silu(embedding).asType(outputNorm.linear.weight.dtype)))
        }
        eval(steps.map(\.output))
        for block in blocks {
            try Task.checkCancellation()
            let projected = embeddings.map { block.modulation!($0) }
            eval(projected)
            for (step, table) in projected.enumerated() { steps[step].blocks.append(table) }
            block.modulation!.release()
        }
        outputNorm.linear.weight._updateInternal(MLXArray.zeros([0]))
        outputNorm.linear.bias?._updateInternal(MLXArray.zeros([0]))
        return steps
    }

    func rotary(_ positions: MLXArray) -> Layers.Rotary {
        let dimension = config.ropeWidth
        let exponents = arange(0, dimension, dtype: .float32) / Float(dimension)
        let frequencies = 1 / pow(config.theta, exponents)
        let angles = (positions[.ellipsis, .newAxis] * frequencies).reshaped(
            1, -1, 1, 3 * dimension)
        let doubled = concatenated([angles, angles], axis: -1)
        return (cos(doubled), sin(doubled))
    }

    func encodeText(_ hidden: MLXArray) -> MLXArray {
        refiner(context(hidden.asType(Layers.projectionDType(context))))
    }

    func callAsFunction(
        _ image: MLXArray, text: MLXArray, rotary: Layers.Rotary, indices: MLXArray,
        timesteps: MLXArray, step: Step, head: Int, targetStart: Int
    ) throws -> MLXArray {
        let vision = projection(image.asType(Layers.projectionDType(projection)))
        var packed = concatenated([text, vision.asType(text.dtype)], axis: 1)
        for (block, modulation) in zip(blocks, step.blocks) {
            try Task.checkCancellation()
            packed = block(packed, table: modulation, indices: indices, rotary: rotary)
            eval(packed)
        }
        let final = outputNorm(
            packed[0..., targetStart..., 0...], modulation: step.output,
            indices: timesteps[targetStart...])
        let output = outputs[head]
        return output(final.asType(Layers.projectionDType(output)))
    }
}
