// Copyright 2026 The HuggingFace Team. All rights reserved.
// Adapted from the Apache-2.0 mlx-vlm Liquid codec implementation.
import Foundation
import MLX
import MLXNN

struct LiquidCodecConfiguration: Decodable {
    let encoderDim: Int
    let quantizerDim: Int
    let levels: [Int]
    let codebooks: Int
    let fft: Int
    let hop: Int
    let window: Int
    let stack: Int
    let decoderWindow: Int
    let sampleRate: Int
    enum CodingKeys: String, CodingKey {
        case encoderDim = "encoder_dim"
        case quantizerDim = "quantizer_dim"
        case levels = "fsq_levels"
        case codebooks = "n_q"
        case fft = "stft_n_fft"
        case hop = "stft_hop_length"
        case window = "stft_win_length"
        case stack = "stft_frame_stack"
        case decoderWindow = "decoder_window"
        case sampleRate = "sampling_rate"
    }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        encoderDim = try c.decodeIfPresent(Int.self, forKey: .encoderDim) ?? 512
        quantizerDim = try c.decodeIfPresent(Int.self, forKey: .quantizerDim) ?? 256
        levels = try c.decodeIfPresent([Int].self, forKey: .levels) ?? [8, 7, 6, 6]
        codebooks = try c.decodeIfPresent(Int.self, forKey: .codebooks) ?? 12
        fft = try c.decodeIfPresent(Int.self, forKey: .fft) ?? 1280
        hop = try c.decodeIfPresent(Int.self, forKey: .hop) ?? 320
        window = try c.decodeIfPresent(Int.self, forKey: .window) ?? 1280
        stack = try c.decodeIfPresent(Int.self, forKey: .stack) ?? 6
        decoderWindow = try c.decodeIfPresent(Int.self, forKey: .decoderWindow) ?? 30
        sampleRate = try c.decodeIfPresent(Int.self, forKey: .sampleRate) ?? 24000
        guard encoderDim > 0, quantizerDim > 0, codebooks > 0, !levels.isEmpty,
            levels.allSatisfy({ $0 > 1 }), hop > 0, fft > hop, fft % hop == 0,
            window == fft, stack > 0, decoderWindow > 0, sampleRate > 0
        else { throw ModelFactoryError.invalidConfiguration("Invalid Liquid codec configuration") }
    }
    func backbone() throws -> LFM3TextConfiguration {
        let data = try JSONSerialization.data(withJSONObject: [
            "hidden_size": encoderDim, "intermediate_size": 2304, "num_hidden_layers": 8,
            "num_attention_heads": 8, "num_key_value_heads": 2, "head_dim": 64, "vocab_size": 1,
            "layer_types": (0 ..< 8).map { [2, 4, 6].contains($0) ? "sliding_attention" : "conv" },
            "rope_parameters": ["sliding_attention": ["rope_theta": 1_000_000]],
            "sliding_window": decoderWindow, "tie_word_embeddings": false,
        ])
        return try JSONDecoder().decode(LFM3TextConfiguration.self, from: data)
    }
}

final class LiquidQuantizer: Module {
    let config: LiquidCodecConfiguration
    @ModuleInfo(key: "project_out") var output: Linear
    init(_ c: LiquidCodecConfiguration) {
        config = c
        _output.wrappedValue = Linear(c.codebooks * c.levels.count, c.quantizerDim)
    }
    func callAsFunction(_ codes: MLXArray) -> MLXArray {
        let levels = MLXArray(config.levels)
        var product = 1
        let basis = config.levels.map { n -> Int in
            defer { product *= n }
            return product
        }
        let half = levels.floorDivide(2).asType(.float32)
        let digits =
            codes.transposed(0, 2, 1).expandedDimensions(axis: -1).floorDivide(MLXArray(basis))
            % levels
        let latents = ((digits.asType(.float32) - half) / half).reshaped(
            codes.dim(0), codes.dim(2), -1)
        return output(latents.asType(output.weight.dtype))
    }
}

final class LiquidISTFT: Module {
    let config: LiquidCodecConfiguration
    let window: MLXArray
    init(_ c: LiquidCodecConfiguration) {
        config = c
        window = .zeros([c.window])
    }
    func overlapAdd(_ frames: MLXArray) -> MLXArray {
        let c = config
        let blocks = frames.reshaped(
            frames.dim(0), frames.dim(1), config.fft / config.hop, config.hop)
        let k = c.fft / c.hop
        var sum = MLXArray.zeros([frames.dim(0), frames.dim(1) + k - 1, c.hop])
        for j in 0 ..< k {
            sum =
                sum
                + padded(blocks[0..., 0..., j, 0...], widths: [0, IntOrPair((j, k - 1 - j)), 0])
        }
        return sum.reshaped(frames.dim(0), -1)
    }
    func callAsFunction(_ spectrum: MLXArray) -> MLXArray {
        let w = window.asType(.float32)
        let audio = overlapAdd(MLXFFT.irfft(spectrum, n: config.fft, axis: -1) * w)
        let envelope = overlapAdd(broadcast(w * w, to: [1, spectrum.dim(1), config.window]))
        let trim = (config.fft - config.hop) / 2
        return audio[0..., trim ..< (audio.dim(1) - trim)]
            / envelope[0..., trim ..< (envelope.dim(1) - trim)]
    }
}

final class LiquidDecoder: Module {
    let config: LiquidCodecConfiguration
    let lfm: LFM3Backbone
    let lin: Linear
    let istft: LiquidISTFT
    init(_ c: LiquidCodecConfiguration) throws {
        config = c
        lfm = LFM3Backbone(try c.backbone(), embeddings: false)
        lin = Linear(c.encoderDim, c.fft + 2)
        istft = LiquidISTFT(c)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = lfm(repeated(x, count: config.stack, axis: 1))
        let parts = lin(h).asType(.float32).split(parts: 2, axis: -1)
        let mag = exp(parts[0])
        let phase = parts[1]
        return istft(mag * cos(phase) + MLXArray(real: 0, imaginary: 1) * mag * sin(phase))
    }
}

/// Decode-only Liquid codec. Input is [codebooks, frames] without EOS or padding.
final class LiquidCodec: Module {
    let config: LiquidCodecConfiguration
    let quantizer: LiquidQuantizer
    @ModuleInfo(key: "from_quantizer") var projection: Linear
    let decoder: LiquidDecoder
    init(_ c: LiquidCodecConfiguration) throws {
        config = c
        quantizer = LiquidQuantizer(c)
        _projection.wrappedValue = Linear(c.quantizerDim, c.encoderDim)
        decoder = try LiquidDecoder(c)
    }
    func decode(_ codes: MLXArray) throws -> MLXArray {
        guard codes.ndim == 2, codes.dim(0) == config.codebooks else {
            throw ModelExecutionError.invalidOptions("Liquid codec expects [codebooks, frames]")
        }
        if codes.dim(1) == 0 { return MLXArray.zeros([0]) }
        guard codes.min().item(Int.self) >= 0,
            codes.max().item(Int.self) < config.levels.reduce(1, *)
        else {
            throw ModelExecutionError.invalidOptions(
                "Strip EOS and padding before Liquid codec decoding")
        }
        return decoder(projection(quantizer(codes.expandedDimensions(axis: 0))))[0]
    }
    static func load(from directory: URL, dtype: DType) throws -> LiquidCodec {
        let c = try JSONDecoder().decode(
            LiquidCodecConfiguration.self,
            from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let model = try LiquidCodec(c)
        var weights = [String: MLXArray]()
        for url in try safetensorWeightURLs(in: directory) {
            for (name, value) in try loadArrays(url: url) {
                let key = name.replacingOccurrences(of: "quantizer.fsq.", with: "quantizer.")
                guard
                    ["decoder.", "from_quantizer.", "quantizer.project_out."].contains(
                        where: key.hasPrefix)
                else { continue }
                var tensor = value
                if key.hasSuffix("conv.conv.weight"), tensor.dim(1) == 1 {
                    tensor = tensor.transposed(0, 2, 1)
                }
                weights[key] = tensor.asType(key == "decoder.istft.window" ? .float32 : dtype)
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        model.train(false)
        eval(model)
        return model
    }
}
