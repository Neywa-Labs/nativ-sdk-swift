import Foundation
import MLX
import MLXNN

struct Flux3Configuration {
    let hidden: Int
    let heads: Int
    let depth: Int
    let jointDepth: Int
    let intermediate: Int
    let axes: [Int]
    let theta: Float
    let imageChannels: Int
    let referenceChannels: Int
    let textChannels: Int
    let imageRoPEScale: Float
    let flowMap: Bool
    let textLayers: [Int]
    let padMultiple: Int
    let maxTokens: Int
    let steps: Int
    let guidance: Float
    let alpha: Float

    static func matches(_ config: DiffusionConfiguration) -> Bool {
        let channels = config.child("dit").child("in_channels").values
        return config.string("model").hasPrefix("flux-3-image")
            && ["image", "image_cond", "text"].allSatisfy { channels[$0] is Int }
    }

    init(_ config: DiffusionConfiguration) throws {
        let dit = config.child("dit")
        let channels = dit.child("in_channels")
        let text = config.child("text_encoder")
        let sampler = config.child("sampler")
        hidden = dit.int("hidden", 3072)
        heads = dit.int("heads", 24)
        depth = dit.int("depth", 5)
        jointDepth = dit.int("depth_single", 28)
        let mlpRatio = dit.float("mlp_ratio", 3)
        let mlpWidth = Double(hidden) * Double(mlpRatio)
        guard mlpWidth >= 1, mlpWidth < Double(Int.max / 2) else {
            throw ModelExecutionError.invalidOptions("Invalid FLUX.3 MLP width")
        }
        intermediate = Int(mlpWidth)
        axes = dit.ints("axes_dim", [32, 32, 32, 32])
        theta = dit.float("theta", 10_000)
        imageChannels = channels.int("image", 0)
        referenceChannels = channels.int("image_cond", 0)
        textChannels = channels.int("text", 0)
        imageRoPEScale = dit.float("image_rope_scale", 0.5)
        flowMap = dit.bool("flow_map", true)
        textLayers = text.ints("layers", [4, 8, 12, 16, 20, 24, 28, 32])
        padMultiple = text.int("pad_multiple", 80)
        maxTokens = min(4096, text.int("max_tokens", 4096))
        steps = sampler.int("steps", flowMap ? 16 : 50)
        guidance = sampler.float("guidance", flowMap ? 1 : 3.5)
        alpha = sampler.float("alpha", 6.93)
        guard Self.matches(config), config.int("format_version", 1) == 1,
            config.values["quantization"] == nil || config.values["quantization"] is NSNull,
            (dit.values["out_modes"] as? [String] ?? ["image"]) == ["image"],
            dit.values["vec_in_dim"] == nil || dit.values["vec_in_dim"] is NSNull,
            hidden > 0, hidden <= Int.max / 4, heads > 0, hidden.isMultiple(of: heads),
            axes.count == 4, axes.allSatisfy({ $0 > 0 && $0 <= hidden && $0.isMultiple(of: 2) }),
            axes.reduce(0, +) == hidden / heads, depth >= 0, jointDepth > 0,
            imageChannels > 0, referenceChannels > 0, textChannels > 0,
            theta.isFinite, theta > 0, imageRoPEScale.isFinite, imageRoPEScale > 0,
            !textLayers.isEmpty, Set(textLayers).count == textLayers.count,
            textLayers.allSatisfy({ $0 > 0 }), (1 ... 4096).contains(padMultiple),
            maxTokens > 0, steps > 0, guidance.isFinite, guidance >= 0, alpha.isFinite, alpha > 0
        else {
            throw ModelExecutionError.invalidOptions("Invalid or unsupported FLUX.3 checkpoint")
        }
    }
}

final class Flux3Transformer: Module {
    typealias Modulation = (shift: MLXArray, scale: MLXArray, gate: MLXArray)
    typealias RoPE = (cosine: MLXArray, sine: MLXArray)

    struct Stream {
        let tokens: MLXArray
        let rotary: RoPE
    }

    final class TimeEmbedding: Module {
        @ModuleInfo(key: "in_layer") var input: Linear
        @ModuleInfo(key: "out_layer") var output: Linear

        init(_ width: Int) {
            _input.wrappedValue = Linear(256, width, bias: false)
            _output.wrappedValue = Linear(width, width, bias: false)
        }

        func callAsFunction(_ time: Float, batch: Int, dtype: DType) -> MLXArray {
            let frequencies = exp(-log(Float(10_000)) * arange(128, dtype: .float32) / 128)
            let value = broadcast(MLXArray(time).asType(dtype), to: [batch, 1, 1])
            let angles = (1000 * value).asType(.float32) * frequencies
            let embedding = concatenated([cos(angles), sin(angles)], axis: -1).asType(dtype)
            return output(silu(input(embedding)))
        }
    }

    final class ModeModulation: Module {
        @ModuleInfo(key: "lin") var linear: Linear

        init(_ width: Int) {
            _linear.wrappedValue = Linear(width, 3 * width, bias: false)
        }

        func callAsFunction(_ time: MLXArray) -> Modulation {
            let values = split(linear(silu(time)), parts: 3, axis: -1)
            return (values[0], values[1], values[2])
        }
    }

    final class ModeModulations: Module {
        @ModuleInfo var image: ModeModulation
        @ModuleInfo(key: "image_cond") var reference: ModeModulation
        @ModuleInfo(key: "txt") var text: ModeModulation

        init(_ width: Int) {
            _image.wrappedValue = ModeModulation(width)
            _reference.wrappedValue = ModeModulation(width)
            _text.wrappedValue = ModeModulation(width)
        }
    }

    final class Scale: Module {
        @ParameterInfo var scale: MLXArray

        init(_ width: Int) {
            _scale.wrappedValue = .ones([width])
        }
    }

    final class QKNorm: Module {
        @ModuleInfo(key: "query_norm") var queryNorm: Scale
        @ModuleInfo(key: "key_norm") var keyNorm: Scale

        init(_ width: Int) {
            _queryNorm.wrappedValue = Scale(width)
            _keyNorm.wrappedValue = Scale(width)
        }

        func rotate(_ x: MLXArray, scale: MLXArray, rotary: RoPE) -> MLXArray {
            let y = MLXFast.rmsNorm(x.asType(.float32), weight: scale.asType(.float32), eps: 1e-6)
            let even = y[.ellipsis, .stride(by: 2)]
            let odd = y[.ellipsis, .stride(from: 1, by: 2)]
            let cosine = rotary.cosine.expandedDimensions(axis: 1)
            let sine = rotary.sine.expandedDimensions(axis: 1)
            return stacked([even * cosine - odd * sine, even * sine + odd * cosine], axis: -1)
                .reshaped(x.shape).asType(x.dtype)
        }
    }

    final class Block: Module {
        @ModuleInfo(key: "q_proj") var query: Linear
        @ModuleInfo(key: "k_proj") var key: Linear
        @ModuleInfo(key: "v_proj") var value: Linear
        @ModuleInfo(key: "mlp_in") var mlpIn: Linear
        @ModuleInfo(key: "attn_out") var attentionOut: Linear
        @ModuleInfo(key: "mlp_out") var mlpOut: Linear
        @ModuleInfo var norm: QKNorm
        let heads: Int
        let headWidth: Int

        init(_ c: Flux3Configuration) {
            heads = c.heads
            headWidth = c.hidden / c.heads
            _query.wrappedValue = Linear(c.hidden, c.hidden, bias: false)
            _key.wrappedValue = Linear(c.hidden, c.hidden, bias: false)
            _value.wrappedValue = Linear(c.hidden, c.hidden, bias: false)
            _mlpIn.wrappedValue = Linear(c.hidden, 2 * c.intermediate, bias: false)
            _attentionOut.wrappedValue = Linear(c.hidden, c.hidden, bias: false)
            _mlpOut.wrappedValue = Linear(c.intermediate, c.hidden, bias: false)
            _norm.wrappedValue = QKNorm(headWidth)
        }

        func callAsFunction(_ x: MLXArray, rotary: RoPE, mod: Modulation) -> MLXArray {
            let h = Flux3Transformer.modulate(x, scale: mod.scale, shift: mod.shift)
            func splitHeads(_ x: MLXArray) -> MLXArray {
                x.reshaped(h.dim(0), h.dim(1), heads, headWidth).transposed(0, 2, 1, 3)
            }
            let q = norm.rotate(splitHeads(query(h)), scale: norm.queryNorm.scale, rotary: rotary)
            let k = norm.rotate(splitHeads(key(h)), scale: norm.keyNorm.scale, rotary: rotary)
            let attended = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: splitHeads(value(h)),
                scale: 1 / sqrt(Float(headWidth)),
                mask: .none)
            let projected = attentionOut(attended.transposed(0, 2, 1, 3).reshaped(x.shape))
            let mlp = split(mlpIn(h), parts: 2, axis: -1)
            return x + mod.gate * (projected + mlpOut(silu(mlp[0]) * mlp[1]))
        }
    }

    final class InputProjection: Module {
        @ModuleInfo var image: Linear
        @ModuleInfo(key: "image_cond") var reference: Linear

        init(_ c: Flux3Configuration) {
            _image.wrappedValue = Linear(c.imageChannels, c.hidden, bias: false)
            _reference.wrappedValue = Linear(c.referenceChannels, c.hidden, bias: false)
        }
    }

    final class EarlyBlocks: Module {
        @ModuleInfo var image: [Block]
        @ModuleInfo(key: "image_cond") var reference: [Block]

        init(_ c: Flux3Configuration) {
            _image.wrappedValue = (0 ..< c.depth).map { _ in Block(c) }
            _reference.wrappedValue = (0 ..< c.depth).map { _ in Block(c) }
        }
    }

    final class FinalProjection: Module {
        final class Image: Module {
            @ModuleInfo var linear: Linear
            @ModuleInfo(key: "adaLN_modulation") var modulation: Linear

            init(_ c: Flux3Configuration) {
                _linear.wrappedValue = Linear(c.hidden, c.imageChannels, bias: false)
                _modulation.wrappedValue = Linear(c.hidden, 2 * c.hidden, bias: false)
            }

            func callAsFunction(_ x: MLXArray, time: MLXArray) -> MLXArray {
                let weights = split(modulation(silu(time)), parts: 2, axis: -1)
                return linear(Flux3Transformer.modulate(x, scale: weights[1], shift: weights[0]))
            }
        }

        @ModuleInfo var image: Image

        init(_ c: Flux3Configuration) {
            _image.wrappedValue = Image(c)
        }
    }

    @ModuleInfo(key: "emb_in") var input: InputProjection
    @ModuleInfo(key: "txt_in") var textIn: Linear
    @ModuleInfo(key: "time_in") var timeIn: TimeEmbedding
    @ModuleInfo(key: "ref_time_in") var refTimeIn: TimeEmbedding?
    @ModuleInfo(key: "early_stream_modulations") var early: ModeModulations
    @ModuleInfo(key: "single_stream_modulations") var single: ModeModulations
    @ModuleInfo(key: "content_mode_blocks") var contentBlocks: EarlyBlocks
    @ModuleInfo(key: "txt_mode_blocks") var textBlocks: [Block]
    @ModuleInfo(key: "single_blocks") var jointBlocks: [Block]
    @ModuleInfo(key: "final_layer") var final: FinalProjection
    private let position: Flux2PosEmbed

    init(_ c: Flux3Configuration) {
        _input.wrappedValue = InputProjection(c)
        _textIn.wrappedValue = Linear(c.textChannels, c.hidden, bias: false)
        _timeIn.wrappedValue = TimeEmbedding(c.hidden)
        if c.flowMap {
            _refTimeIn.wrappedValue = TimeEmbedding(c.hidden)
        }
        _early.wrappedValue = ModeModulations(c.hidden)
        _single.wrappedValue = ModeModulations(c.hidden)
        _contentBlocks.wrappedValue = EarlyBlocks(c)
        _textBlocks.wrappedValue = (0 ..< c.depth).map { _ in Block(c) }
        _jointBlocks.wrappedValue = (0 ..< c.jointDepth).map { _ in Block(c) }
        _final.wrappedValue = FinalProjection(c)
        position = Flux2PosEmbed(theta: c.theta, axesDims: c.axes)
    }

    static func checkpointWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        Dictionary(
            uniqueKeysWithValues: weights.map { key, tensor in
                (
                    key.replacingOccurrences(
                        of: ".adaLN_modulation.1.", with: ".adaLN_modulation."),
                    tensor.asType(.bfloat16)
                )
            })
    }

    private static func modulate(_ x: MLXArray, scale: MLXArray, shift: MLXArray) -> MLXArray {
        (1 + scale) * MLXFast.layerNorm(x, weight: nil, bias: nil, eps: 1e-6) + shift
    }

    func rotary(_ ids: MLXArray) -> RoPE {
        position(ids)
    }

    func prepareText(_ hidden: MLXArray, ids: MLXArray) -> Stream {
        Stream(tokens: textIn(hidden.asType(textIn.weight.dtype)), rotary: rotary(ids))
    }

    func prepareReferences(_ tokens: MLXArray, ids: MLXArray) -> Stream {
        Stream(tokens: input.reference(tokens.asType(textIn.weight.dtype)), rotary: rotary(ids))
    }

    func callAsFunction(
        _ latents: MLXArray, rotary: RoPE, text: Stream, references: Stream?,
        time: Float, target: Float?
    ) throws -> MLXArray {
        guard (target != nil) == (refTimeIn != nil) else {
            throw ModelExecutionError.invalidOptions("Invalid FLUX.3 flow-map timestep")
        }
        let dtype = textIn.weight.dtype
        let batch = latents.dim(0)
        let imageTime = timeIn(time, batch: batch, dtype: dtype)
        let conditionTime = timeIn(0, batch: batch, dtype: dtype)
        let times: (image: MLXArray, conditioning: MLXArray)
        if let refTimeIn, let target {
            let offset = refTimeIn(target, batch: batch, dtype: dtype)
            times = (imageTime + offset, conditionTime + offset)
        } else {
            times = (imageTime, conditionTime)
        }

        func process(
            _ stream: Stream, _ blocks: [Block], _ early: Modulation, _ joint: Modulation
        ) -> (Stream, Modulation) {
            let output = blocks.reduce(stream.tokens) { $1($0, rotary: stream.rotary, mod: early) }
            return (
                Stream(tokens: output, rotary: stream.rotary),
                (
                    broadcast(joint.shift, to: output.shape),
                    broadcast(joint.scale, to: output.shape),
                    broadcast(joint.gate, to: output.shape)
                )
            )
        }

        let image = Stream(tokens: input.image(latents.asType(dtype)), rotary: rotary)
        var streams = [
            process(image, contentBlocks.image, early.image(times.image), single.image(times.image))
        ]
        if let references {
            streams.append(
                process(
                    references, contentBlocks.reference,
                    early.reference(times.conditioning), single.reference(times.conditioning)))
        }
        streams.append(
            process(
                text, textBlocks, early.text(times.conditioning), single.text(times.conditioning)))
        let tokens = concatenated(streams.map { $0.0.tokens }, axis: 1)
        let jointRoPE = (
            concatenated(streams.map { $0.0.rotary.cosine }, axis: 1),
            concatenated(streams.map { $0.0.rotary.sine }, axis: 1)
        )
        let modulation = (
            concatenated(streams.map { $0.1.shift }, axis: 1),
            concatenated(streams.map { $0.1.scale }, axis: 1),
            concatenated(streams.map { $0.1.gate }, axis: 1)
        )
        let output = jointBlocks.reduce(tokens) { $1($0, rotary: jointRoPE, mod: modulation) }
        return final.image(output[0..., ..<latents.dim(1), 0...], time: times.image)
    }
}
