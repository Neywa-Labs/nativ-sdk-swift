import Foundation
import MLX
import MLXNN

struct MageFlowTransformerConfiguration {
    let channels: Int
    let output: Int
    let context: Int
    let hidden: Int
    let heads: Int
    let depth: Int
    let intermediate: Int
    let axes: [Int]
    let theta: Float
    let maxTokens: Int

    init(_ c: DiffusionConfiguration) throws {
        channels = c.int("in_channels", 128)
        output = c.int("out_channels", 128)
        context = c.int("context_in_dim", 2560)
        hidden = c.int("hidden_size", 3072)
        heads = c.int("num_heads", 24)
        depth = c.int("depth", 12)
        let mlpWidth = Double(hidden) * Double(c.float("mlp_ratio", 4))
        guard mlpWidth >= 1, mlpWidth < Double(Int.max) else {
            throw ModelExecutionError.invalidOptions("Invalid Mage-Flow feed-forward width")
        }
        intermediate = Int(mlpWidth)
        axes = c.ints("axes_dim", [16, 56, 56])
        theta = c.float("theta", 10_000)
        maxTokens = c.int("txt_max_length", c.int("max_sequence_length", 2048))
        guard channels > 0, output > 0, context > 0, hidden > 0, hidden <= Int.max / 6, heads > 0,
            hidden.isMultiple(of: heads), depth > 0, axes.count == 3,
            axes.allSatisfy({ $0 > 0 && $0 <= hidden && $0.isMultiple(of: 2) }),
            axes.reduce(0, +) == hidden / heads, theta.isFinite, theta > 0, maxTokens > 0,
            c.int("depth_single_blocks", 0) == 0, c.int("patch_size", 1) == 1,
            !c.bool("guidance_embed", false), !c.bool("apply_text_rotary_emb", false)
        else {
            throw ModelExecutionError.invalidOptions("Invalid Mage-Flow transformer configuration")
        }
    }
}

enum MageFlowLayers {
    final class TimeEmbedding: Module {
        @ModuleInfo(key: "linear_1") var first: Linear
        @ModuleInfo(key: "linear_2") var second: Linear
        private let scale: Float
        private let roundFrequencies: Bool

        init(_ width: Int, scale: Float = 1, roundFrequencies: Bool = false) {
            self.scale = scale
            self.roundFrequencies = roundFrequencies
            _first.wrappedValue = Linear(256, width)
            _second.wrappedValue = Linear(width, width)
        }

        func callAsFunction(_ t: MLXArray, dtype: DType) -> MLXArray {
            var frequencies = exp(-log(Float(10_000)) * arange(128, dtype: .float32) / 128)
            if roundFrequencies { frequencies = frequencies.asType(dtype).asType(.float32) }
            let angles = scale * (t.reshaped(-1, 1).asType(.float32) * frequencies)
            let embedding = concatenated([cos(angles), sin(angles)], axis: -1).asType(dtype)
            return second(silu(first(embedding)))
        }
    }

    final class Modulation: Module {
        @ModuleInfo var linear: Linear

        init(_ width: Int, chunks: Int) {
            _linear.wrappedValue = Linear(width, chunks * width)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear(silu(x))
        }
    }

    static func norm(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: nil, bias: nil, eps: 1e-6)
    }
}

final class MageFlowTransformer: Module {
    typealias RoPE = (cosine: MLXArray, sine: MLXArray)

    final class TimeProjection: Module {
        @ModuleInfo(key: "timestep_embedder") var time: MageFlowLayers.TimeEmbedding

        init(_ width: Int) {
            _time.wrappedValue = MageFlowLayers.TimeEmbedding(
                width, scale: 1000, roundFrequencies: true)
        }
    }

    final class FeedForward: Module {
        @ModuleInfo(key: "linear_in") var input: Linear
        @ModuleInfo(key: "linear_out") var output: Linear

        init(_ c: MageFlowTransformerConfiguration) {
            _input.wrappedValue = Linear(c.hidden, c.intermediate)
            _output.wrappedValue = Linear(c.intermediate, c.hidden)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            output(geluApproximate(input(x)))
        }
    }

    final class Attention: Module {
        @ModuleInfo(key: "to_q") var imageQuery: Linear
        @ModuleInfo(key: "to_k") var imageKey: Linear
        @ModuleInfo(key: "to_v") var imageValue: Linear
        @ModuleInfo(key: "add_q_proj") var textQuery: Linear
        @ModuleInfo(key: "add_k_proj") var textKey: Linear
        @ModuleInfo(key: "add_v_proj") var textValue: Linear
        @ModuleInfo(key: "norm_q") var imageQueryNorm: RMSNorm
        @ModuleInfo(key: "norm_k") var imageKeyNorm: RMSNorm
        @ModuleInfo(key: "norm_added_q") var textQueryNorm: RMSNorm
        @ModuleInfo(key: "norm_added_k") var textKeyNorm: RMSNorm
        @ModuleInfo(key: "to_out") var imageOut: Linear
        @ModuleInfo(key: "to_add_out") var textOut: Linear
        private let heads: Int
        private let headDim: Int

        init(_ c: MageFlowTransformerConfiguration) {
            heads = c.heads
            headDim = c.hidden / c.heads
            _imageQuery.wrappedValue = Linear(c.hidden, c.hidden)
            _imageKey.wrappedValue = Linear(c.hidden, c.hidden)
            _imageValue.wrappedValue = Linear(c.hidden, c.hidden)
            _textQuery.wrappedValue = Linear(c.hidden, c.hidden)
            _textKey.wrappedValue = Linear(c.hidden, c.hidden)
            _textValue.wrappedValue = Linear(c.hidden, c.hidden)
            _imageQueryNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
            _imageKeyNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
            _textQueryNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
            _textKeyNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: 1e-6)
            _imageOut.wrappedValue = Linear(c.hidden, c.hidden)
            _textOut.wrappedValue = Linear(c.hidden, c.hidden)
        }

        private func rotate(_ x: MLXArray, with rotary: RoPE) -> MLXArray {
            let even = x[.ellipsis, .stride(by: 2)].asType(.float32)
            let odd = x[.ellipsis, .stride(from: 1, by: 2)].asType(.float32)
            let co = rotary.cosine.expandedDimensions(axis: 2)
            let si = rotary.sine.expandedDimensions(axis: 2)
            return stacked([even * co - odd * si, even * si + odd * co], axis: -1)
                .reshaped(x.shape).asType(x.dtype)
        }

        func callAsFunction(
            _ image: MLXArray, _ text: MLXArray, rotary: RoPE
        ) -> (MLXArray, MLXArray) {
            func splitHeads(_ x: MLXArray) -> MLXArray {
                x.reshaped(x.dim(0), x.dim(1), heads, headDim)
            }
            func join(_ text: MLXArray, _ image: MLXArray) -> MLXArray {
                concatenated([text, image], axis: 1).transposed(0, 2, 1, 3)
            }
            let q = join(
                textQueryNorm(splitHeads(textQuery(text))),
                rotate(imageQueryNorm(splitHeads(imageQuery(image))), with: rotary))
            let k = join(
                textKeyNorm(splitHeads(textKey(text))),
                rotate(imageKeyNorm(splitHeads(imageKey(image))), with: rotary))
            let v = join(splitHeads(textValue(text)), splitHeads(imageValue(image)))
            let attended = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: 1 / sqrt(Float(headDim)), mask: .none)
            let output = attended.transposed(0, 2, 1, 3).reshaped(
                image.dim(0), text.dim(1) + image.dim(1), -1)
            return (imageOut(output[0..., text.dim(1)...]), textOut(output[0..., ..<text.dim(1)]))
        }
    }

    final class Block: Module {
        @ModuleInfo(key: "img_mod") var imageModulation: MageFlowLayers.Modulation
        @ModuleInfo(key: "txt_mod") var textModulation: MageFlowLayers.Modulation
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo(key: "img_mlp") var imageMLP: FeedForward
        @ModuleInfo(key: "txt_mlp") var textMLP: FeedForward

        init(_ c: MageFlowTransformerConfiguration) {
            _imageModulation.wrappedValue = MageFlowLayers.Modulation(c.hidden, chunks: 6)
            _textModulation.wrappedValue = MageFlowLayers.Modulation(c.hidden, chunks: 6)
            _attention.wrappedValue = Attention(c)
            _imageMLP.wrappedValue = FeedForward(c)
            _textMLP.wrappedValue = FeedForward(c)
        }

        func callAsFunction(
            _ image: MLXArray, _ text: MLXArray, time: MLXArray, rotary: RoPE
        ) -> (MLXArray, MLXArray) {
            let imageWeights = split(
                imageModulation(time).expandedDimensions(axis: 1), parts: 6, axis: -1)
            let textWeights = split(
                textModulation(time).expandedDimensions(axis: 1), parts: 6, axis: -1)
            func modulate(_ x: MLXArray, _ weights: [MLXArray], offset: Int) -> MLXArray {
                MageFlowLayers.norm(x) * (1 + weights[offset + 1]) + weights[offset]
            }
            let (img, txt) = attention(
                modulate(image, imageWeights, offset: 0), modulate(text, textWeights, offset: 0),
                rotary: rotary)
            let updatedImage = image + imageWeights[2] * img
            let updatedText = text + textWeights[2] * txt
            return (
                updatedImage + imageWeights[5]
                    * imageMLP(modulate(updatedImage, imageWeights, offset: 3)),
                updatedText + textWeights[5]
                    * textMLP(modulate(updatedText, textWeights, offset: 3))
            )
        }
    }

    @ModuleInfo(key: "img_in") var imageIn: Linear
    @ModuleInfo(key: "txt_norm") var textNorm: RMSNorm
    @ModuleInfo(key: "txt_in") var textIn: Linear
    @ModuleInfo(key: "time_text_embed") var timeProjection: TimeProjection
    @ModuleInfo(key: "transformer_blocks") var blocks: [Block]
    @ModuleInfo(key: "norm_out") var outputModulation: MageFlowLayers.Modulation
    @ModuleInfo(key: "proj_out") var output: Linear
    private let position: Flux2PosEmbed

    init(_ c: MageFlowTransformerConfiguration) {
        _imageIn.wrappedValue = Linear(c.channels, c.hidden)
        _textNorm.wrappedValue = RMSNorm(dimensions: c.context, eps: 1e-6)
        _textIn.wrappedValue = Linear(c.context, c.hidden)
        _timeProjection.wrappedValue = TimeProjection(c.hidden)
        _blocks.wrappedValue = (0 ..< c.depth).map { _ in Block(c) }
        _outputModulation.wrappedValue = MageFlowLayers.Modulation(c.hidden, chunks: 2)
        _output.wrappedValue = Linear(c.hidden, c.output)
        position = Flux2PosEmbed(theta: c.theta, axesDims: c.axes)
    }

    static func checkpointKey(_ key: String) -> String {
        key.replacingOccurrences(of: ".img_mod.1.", with: ".img_mod.linear.")
            .replacingOccurrences(of: ".txt_mod.1.", with: ".txt_mod.linear.")
            .replacingOccurrences(of: ".img_mlp.net.0.proj.", with: ".img_mlp.linear_in.")
            .replacingOccurrences(of: ".img_mlp.net.2.", with: ".img_mlp.linear_out.")
            .replacingOccurrences(of: ".txt_mlp.net.0.proj.", with: ".txt_mlp.linear_in.")
            .replacingOccurrences(of: ".txt_mlp.net.2.", with: ".txt_mlp.linear_out.")
            .replacingOccurrences(of: ".attn.to_out.0.", with: ".attn.to_out.")
    }

    func rotary(grids: [(height: Int, width: Int)]) -> RoPE {
        let positions = grids.enumerated().map { index, size in
            let grid = Flux2Latents.prepareGridIDs(
                .zeros([1, 1, size.height, size.width]), tCoordinate: index)
            return grid[.ellipsis, ..<3].asType(.float32)
                - MLXArray([
                    Float(0), Float(size.height - size.height / 2),
                    Float(size.width - size.width / 2),
                ])
        }
        return position(concatenated(positions, axis: 1))
    }

    func condition(_ text: MLXArray) -> MLXArray {
        textIn(textNorm(text))
    }

    func callAsFunction(_ latents: MLXArray, text: MLXArray, sigma: MLXArray, rotary: RoPE)
        -> MLXArray
    {
        var image = imageIn(latents)
        var text = text
        let time = timeProjection.time(
            broadcast(sigma.asType(latents.dtype), to: [latents.dim(0)]), dtype: image.dtype)
        for block in blocks {
            (image, text) = block(image, text, time: time, rotary: rotary)
        }
        let mod = split(outputModulation(time).expandedDimensions(axis: 1), parts: 2, axis: -1)
        return output(MageFlowLayers.norm(image) * (1 + mod[0]) + mod[1])
    }
}
