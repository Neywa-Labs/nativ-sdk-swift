import Accelerate
// Copyright 2026 The HuggingFace Team. All rights reserved.
// Adapted from the Apache-2.0 mlx-vlm LFM3-Audio implementation.
import Foundation
import MLX
import MLXNN

enum LFM3AudioFeatures {
    static func frameCount(_ samples: Int) -> Int { 1 + max(0, (samples - 400 + 159) / 160) }
    static func tokenCount(_ frames: Int) -> Int { (frames + 7) / 8 }

    // Slaney's piecewise linear/log mel scale and area-normalized triangles.
    static func filterValues() -> [Double] {
        func mel(_ hz: Double) -> Double {
            hz < 1000 ? hz / (200.0 / 3) : 15 + log(hz / 1000) / (log(6.4) / 27)
        }
        func hz(_ mel: Double) -> Double {
            mel < 15 ? mel * (200.0 / 3) : 1000 * exp((mel - 15) * log(6.4) / 27)
        }
        let edges = (0 ..< 130).map { hz(Double($0) * mel(8000) / 129) }
        var values = [Double]()
        for i in 0 ..< 128 {
            for j in 0 ..< 201 {
                let f = Double(j) * 40
                let triangle = max(
                    0,
                    min(
                        (f - edges[i]) / (edges[i + 1] - edges[i]),
                        (edges[i + 2] - f) / (edges[i + 2] - edges[i + 1])))
                values.append(triangle * 2 / (edges[i + 2] - edges[i]))
            }
        }
        return values
    }

    static func logMel(_ samples: MLXArray) throws -> MLXArray {
        guard samples.ndim == 1, samples.size > 0 else {
            throw ModelExecutionError.invalidOptions("LFM3 requires a nonempty mono audio clip")
        }
        let n = frameCount(samples.size)
        let raw = samples.asType(.float32).asArray(Float.self)
        var frames = [Double](repeating: 0, count: n * 400)
        for frame in 0 ..< n {
            for j in 0 ..< 400 where frame * 160 + j < raw.count {
                frames[frame * 400 + j] = Double(raw[frame * 160 + j]) * fourierWindow[j]
            }
        }
        // MLX has no float64 dtype. The reference evaluates this 400-point DFT
        // in float64; float32 FFT roundoff is visible after log compression of
        // quiet bins. Accelerate keeps the DFT and Slaney projection in double.
        var real = [Double](repeating: 0, count: n * 201)
        var imaginary = real
        vDSP_mmulD(frames, 1, fourierReal, 1, &real, 1, vDSP_Length(n), 201, 400)
        vDSP_mmulD(frames, 1, fourierImaginary, 1, &imaginary, 1, vDSP_Length(n), 201, 400)
        let power = zip(real, imaginary).map { $0 * $0 + $1 * $1 }
        var mel = [Double](repeating: 0, count: n * 128)
        vDSP_mmulD(power, 1, melProjection, 1, &mel, 1, vDSP_Length(n), 128, 201)
        let result = mel.map { Float((max(Foundation.log10($0), -12) + 4) / 4) }
        return MLXArray(result, [n, 128]).transposed()
    }
    private static let melProjection: [Double] = {
        let filters = filterValues()
        return (0 ..< 201).flatMap { bin in (0 ..< 128).map { filters[$0 * 201 + bin] } }
    }()
    private static let fourierWindow = (0 ..< 400).map {
        0.5 - 0.5 * cos(2 * Double.pi * Double($0) / 400)
    }
    private static let fourierReal = (0 ..< 400).flatMap { sample in
        (0 ..< 201).map { cos(2 * Double.pi * Double(sample * $0) / 400) }
    }
    private static let fourierImaginary = (0 ..< 400).flatMap { sample in
        (0 ..< 201).map { -sin(2 * Double.pi * Double(sample * $0) / 400) }
    }
}

final class LFM3Downsampler: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "conv_mid") var convMid: Conv2d
    @ModuleInfo(key: "lin_mid") var linMid: Conv2d
    @ModuleInfo(key: "conv_end") var convEnd: Conv2d
    @ModuleInfo(key: "lin_end") var linEnd: Conv2d
    @ModuleInfo(key: "lin_out") var linOut: Linear
    init(_ c: LFM3AudioEncoderConfiguration) {
        _convIn.wrappedValue = Conv2d(
            inputChannels: 1, outputChannels: c.channels, kernelSize: 3, stride: 2, padding: 1)
        _convMid.wrappedValue = Conv2d(
            inputChannels: c.channels, outputChannels: c.channels, kernelSize: 3, stride: 2,
            padding: 1, groups: c.channels)
        _linMid.wrappedValue = Conv2d(
            inputChannels: c.channels, outputChannels: c.channels, kernelSize: 1)
        _convEnd.wrappedValue = Conv2d(
            inputChannels: c.channels, outputChannels: c.channels, kernelSize: 3, stride: 2,
            padding: 1, groups: c.channels)
        _linEnd.wrappedValue = Conv2d(
            inputChannels: c.channels, outputChannels: c.channels, kernelSize: 1)
        _linOut.wrappedValue = Linear(c.melBins * c.channels / 8, c.hiddenSize)
    }
    func callAsFunction(_ features: MLXArray, lengths: MLXArray) -> MLXArray {
        func mask(_ x: MLXArray, _ n: MLXArray) -> MLXArray {
            x
                * (MLXArray(0 ..< x.dim(2)) .< n.expandedDimensions(axis: 1))[
                    0..., .newAxis, 0..., .newAxis]
        }
        func activate(_ x: MLXArray) -> MLXArray { silu(x.asType(.float32)).asType(x.dtype) }
        var n = lengths
        var x = mask(features.expandedDimensions(axis: -1).asType(convIn.weight.dtype), n)
        x = activate(convIn(x))
        n = (n + 1).floorDivide(2)
        x = activate(linMid(convMid(mask(x, n))))
        n = (n + 1).floorDivide(2)
        x = activate(linEnd(convEnd(mask(x, n))))
        return linOut(x.transposed(0, 2, 3, 1).reshaped(x.dim(0), x.dim(2), -1))
    }
}

final class LFM3AudioTrunk: Module {
    let layers: [LFM2DecoderLayer]
    @ModuleInfo(key: "embedding_norm") var norm: RMSNorm
    init(_ c: LFM2Configuration) {
        layers = (0 ..< c.hiddenLayers).map { LFM2DecoderLayer(c, layerIdx: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.normEps)
    }
    func callAsFunction(_ x: MLXArray, window: Int) -> MLXArray {
        var h = x
        let mask = createAttentionMask(h: x, cache: nil, windowSize: window)
        for layer in layers { h = layer(h, mask: mask, cache: nil) }
        return norm(h)
    }
}

final class LFM3AudioEncoder: Module {
    let downsample: LFM3Downsampler
    let lfm: LFM3AudioTrunk
    let window: Int
    init(_ c: LFM3AudioEncoderConfiguration) {
        downsample = LFM3Downsampler(c)
        lfm = LFM3AudioTrunk(c.text)
        window = c.window
    }
    func callAsFunction(_ features: MLXArray, lengths: [Int]) -> MLXArray {
        let h = lfm(downsample(features, lengths: MLXArray(lengths)), window: window)
        return concatenated(
            lengths.enumerated().map { i, n in h[i, ..<LFM3AudioFeatures.tokenCount(n)] })
    }
}

final class LFM3Projector: Module {
    let mergedSize: Int
    let fp32Activation: Bool
    @ModuleInfo(key: "layer_norm") var norm: LayerNorm
    @ModuleInfo(key: "linear_1") var first: Linear
    @ModuleInfo(key: "linear_2") var second: Linear
    init(
        input: Int, hidden: Int, output: Int, merge: Int = 1, bias: Bool = true, eps: Float = 1e-5,
        fp32Activation: Bool = true
    ) {
        mergedSize = input * merge * merge
        self.fp32Activation = fp32Activation
        _norm.wrappedValue = LayerNorm(dimensions: input, eps: eps)
        _first.wrappedValue = Linear(mergedSize, hidden, bias: bias)
        _second.wrappedValue = Linear(hidden, output, bias: bias)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = first(norm(x).reshaped(-1, mergedSize))
        return second(fp32Activation ? gelu(h.asType(.float32)).asType(h.dtype) : gelu(h))
    }
}

final class LFM3HeadNorm: Module, UnaryLayer {
    let weight: MLXArray
    let eps: Float
    init(_ dimensions: Int, eps: Float) {
        weight = .ones([dimensions])
        self.eps = eps
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x.asType(.float32), weight: weight.asType(.float32), eps: eps).asType(
            x.dtype)
    }
}

final class LFM3Codebook: Module {
    @ModuleInfo var embedding: Embedding
    @ModuleInfo(key: "embedding_norm") var norm: LFM3HeadNorm
    init(size: Int, dimensions: Int, eps: Float) {
        _embedding.wrappedValue = Embedding(embeddingCount: size, dimensions: dimensions)
        _norm.wrappedValue = LFM3HeadNorm(dimensions, eps: eps)
    }
    func logits(_ x: MLXArray) -> MLXArray { embedding.asLinear(norm(x)) }
}

final class LFM3DepthDecoder: Module {
    let config: LFM3Configuration
    @ModuleInfo(key: "depth_linear") var linear: Linear
    @ParameterInfo(key: "mask_embedding") var maskEmbedding: MLXArray
    let model: LFM3Backbone
    @ModuleInfo(key: "depth_embeddings") var embeddings: [LFM3Codebook]
    init(_ c: LFM3Configuration) {
        config = c
        _linear.wrappedValue = Linear(c.text.hiddenSize, c.numCodebooks * c.depth.hiddenSize)
        _maskEmbedding.wrappedValue = .zeros([c.depth.hiddenSize])
        model = LFM3Backbone(c.depth, embeddings: false)
        _embeddings.wrappedValue = (0 ..< c.numCodebooks).map { _ in
            LFM3Codebook(size: c.codebookSize, dimensions: c.depth.hiddenSize, eps: c.depth.normEps)
        }
    }
    func callAsFunction(_ hidden: MLXArray, codes: MLXArray, masked: MLXArray) -> MLXArray {
        let x = linear(hidden).reshaped(hidden.dim(0), config.numCodebooks, config.depth.hiddenSize)
        let e = stacked(
            embeddings.enumerated().map { i, e in e.embedding(codes[0..., i]) }, axis: 1)
        let h = model(
            x + which(masked.expandedDimensions(axis: -1), maskEmbedding.asType(e.dtype), e),
            causal: false)
        return stacked(embeddings.enumerated().map { i, e in e.logits(h[0..., i]) }, axis: 1)
    }
    func sampleFrame(
        _ hidden: MLXArray, temperature: Float, topK: Int, penalty: Float, history: [MLXArray],
        order: [Int]? = nil, random: MLXRandom.RandomState = .init()
    ) throws -> MLXArray {
        let count = config.numCodebooks
        let order =
            order ?? argSort(MLXRandom.uniform(0 ..< 1, [count], key: random)).asArray(Int.self)
        guard order.sorted() == Array(0 ..< count) else {
            throw ModelExecutionError.invalidOptions("MaskGIT order must be a codebook permutation")
        }
        var codes = MLXArray.zeros([1, count], dtype: .int32)
        var revealed = [Bool](repeating: false, count: count)
        var start = 0
        for size in config.schedule {
            try Task.checkCancellation()
            var logits = self(
                hidden, codes: codes, masked: MLXArray(revealed.map { !$0 }, [1, count])
            ).asType(.float32)
            if penalty != 1, !history.isEmpty {
                // Penalize each code only once per codebook, matching put_along_axis.
                let past = stacked(history).asArray(Int.self)
                for k in 0 ..< count {
                    let ids = Set(stride(from: k, to: past.count, by: count).map { past[$0] })
                    for id in ids {
                        let score = logits[0, k, id]
                        logits[0, k, id] = which(score .< 0, score * penalty, score / penalty)
                    }
                }
            }
            let sampled: MLXArray
            if temperature == 0 {
                sampled = argMax(logits, axis: -1)
            } else {
                logits = logits / temperature
                if topK > 0, topK < config.codebookSize {
                    let threshold = sorted(logits, axis: -1)[0..., 0..., config.codebookSize - topK]
                        .expandedDimensions(axis: -1)
                    logits = which(logits .< threshold, -Float.infinity, logits)
                }
                sampled = MLXRandom.categorical(logits, key: random)
            }
            var selected = [Bool](repeating: false, count: count)
            for k in order[start ..< (start + size)] {
                selected[k] = true
                revealed[k] = true
            }
            codes = which(MLXArray(selected, [1, count]), sampled.asType(.int32), codes)
            start += size
        }
        return which(
            any(codes .== config.audioEOS, axis: -1, keepDims: true), config.audioEOS, codes)[0]
    }
}
