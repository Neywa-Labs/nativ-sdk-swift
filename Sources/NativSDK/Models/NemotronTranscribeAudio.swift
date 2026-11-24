// Adapted from NVIDIA NeMo Speech and the MLX Audio Nemotron Transcribe port.
// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. Apache-2.0.
import Foundation
import MLX
import MLXNN

final class NemotronSpeechMel: Module {
    @ParameterInfo var window: MLXArray
    @ParameterInfo var fb: MLXArray
    let config: NemotronSpeechMelConfiguration

    init(_ config: NemotronSpeechMelConfiguration) {
        self.config = config
        _window.wrappedValue = MLXArray(
            (0 ..< config.windowLength).map {
                Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(config.windowLength - 1)))
            })
        let bins = config.fftLength / 2 + 1
        func mel(_ f: Float) -> Float {
            f < 1000 ? f * 3 / 200 : 15 + log(f / 1000) / (log(6.4) / 27)
        }
        func hz(_ m: Float) -> Float {
            m < 15 ? m * 200 / 3 : 1000 * exp((m - 15) * (log(6.4) / 27))
        }
        let maximum = mel(Float(config.sampleRate) / 2)
        let points = (0 ..< (config.features + 2)).map {
            hz(Float($0) * maximum / Float(config.features + 1))
        }
        var filters = [Float](repeating: 0, count: config.features * bins)
        for m in 0 ..< config.features {
            for k in 0 ..< bins {
                let f = Float(k * config.sampleRate) / Float(config.fftLength)
                let slope = min(
                    (f - points[m]) / (points[m + 1] - points[m]),
                    (points[m + 2] - f) / (points[m + 2] - points[m + 1]))
                filters[m * bins + k] = max(0, slope) * 2 / (points[m + 2] - points[m])
            }
        }
        _fb.wrappedValue = MLXArray(filters, [1, config.features, bins])
    }

    func callAsFunction(_ audio: MLXArray) -> MLXArray {
        let frames = audio.size / config.hopLength
        let positions =
            MLXArray(0 ..< frames).expandedDimensions(axis: 1) * config.hopLength
            + MLXArray(0 ..< config.fftLength).expandedDimensions(axis: 0) - config.fftLength / 2
        func gather(_ indices: MLXArray) -> MLXArray {
            let valid = logicalAnd(indices .>= 0, indices .< audio.size)
            return MLX.where(valid, audio[clip(indices, min: 0, max: audio.size - 1)], MLXArray(0))
        }
        let emphasized = gather(positions) - config.preemphasis * gather(positions - 1)
        let samples = MLX.where(
            logicalAnd(positions .>= 0, positions .< audio.size), emphasized, MLXArray(0))
        let padding = config.fftLength - config.windowLength
        let paddedWindow = padded(
            window.asType(.float32), widths: [.init((padding / 2, padding - padding / 2))])
        let power = MLXFFT.rfft(samples * paddedWindow, n: config.fftLength, axis: -1).abs()
            .square()
        let features = log(
            matmul(power, fb[0].asType(.float32).transposed()) + Float(pow(2.0, -24)))
        return contiguous(features.transposed()).expandedDimensions(axis: 0)
    }
}

final class NemotronSpeechStacking: Module {
    let factor: Int
    @ModuleInfo var proj: Linear
    init(_ config: NemotronSpeechEncoderConfiguration) {
        factor = config.factor
        _proj.wrappedValue = Linear(config.features * config.factor, config.width, bias: false)
    }
    func callAsFunction(_ features: MLXArray) -> MLXArray {
        var x = features.transposed(0, 2, 1)
        let remainder = (factor - x.dim(1) % factor) % factor
        x = padded(x, widths: [0, .init((0, remainder)), 0])
        return proj(x.reshaped(x.dim(0), -1, x.dim(2) * factor))
    }
}

final class NemotronSpeechAttention: Module {
    let heads: Int
    let headSize: Int
    @ModuleInfo(key: "w_qkv") var qkv: Linear
    @ModuleInfo(key: "out_proj") var output: Linear
    @ModuleInfo(key: "q_norm") var queryNorm: LayerNorm?
    @ModuleInfo(key: "k_norm") var keyNorm: LayerNorm?
    let rope: RoPE

    init(_ config: NemotronSpeechEncoderConfiguration) {
        heads = config.heads
        headSize = config.width / config.heads
        _qkv.wrappedValue = Linear(config.width, 3 * config.width, bias: config.qkvBias)
        _output.wrappedValue = Linear(config.width, config.width)
        if config.qkNorm {
            _queryNorm.wrappedValue = LayerNorm(dimensions: headSize)
            _keyNorm.wrappedValue = LayerNorm(dimensions: headSize)
        }
        rope = RoPE(
            dimensions: Int(Float(headSize) * config.rotaryFraction), traditional: false,
            base: config.ropeBase)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let packed = qkv(x).reshaped(x.dim(0), x.dim(1), 3, heads, headSize)
        var q = packed[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)
        var k = packed[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        let v = packed[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)
        if let queryNorm, let keyNorm {
            q = queryNorm(q)
            k = keyNorm(k)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: rope(q), keys: rope(k), values: v, scale: 1 / sqrt(Float(headSize)),
            mask: .array(MLXArray.ones([1, 1, 1, x.dim(1)], dtype: .bool)))
        return output(attended.transposed(0, 2, 1, 3).reshaped(x.shape))
    }
}

final class NemotronSpeechFeedForward: Module {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear
    init(_ config: NemotronSpeechEncoderConfiguration) {
        let hidden = Int(Float(config.width) * config.expansion)
        _linear1.wrappedValue = Linear(config.width, hidden)
        _linear2.wrappedValue = Linear(hidden, config.width)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(gelu(linear1(x))) }
}

final class NemotronSpeechBlock: Module {
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var attn: NemotronSpeechAttention
    @ModuleInfo var ffn: NemotronSpeechFeedForward
    init(_ config: NemotronSpeechEncoderConfiguration) {
        _norm1.wrappedValue = LayerNorm(dimensions: config.width)
        _norm2.wrappedValue = LayerNorm(dimensions: config.width)
        _attn.wrappedValue = NemotronSpeechAttention(config)
        _ffn.wrappedValue = NemotronSpeechFeedForward(config)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x = x + attn(norm1(x))
        return x + ffn(norm2(x))
    }
}

final class NemotronSpeechEncoder: Module {
    @ModuleInfo(key: "pre_encode") var preEncode: NemotronSpeechStacking
    @ModuleInfo(key: "embed_norm") var embedNorm: LayerNorm?
    @ModuleInfo var layers: [NemotronSpeechBlock]
    @ModuleInfo(key: "final_norm") var finalNorm: LayerNorm
    let scale: Float
    // The decoder's expert cache stays warm while speech layers load and release
    // one at a time. These replacement arrays are still lazy.
    var reloadLayerWeights: (() throws -> [ModuleParameters])?
    init(_ config: NemotronSpeechEncoderConfiguration) {
        _preEncode.wrappedValue = NemotronSpeechStacking(config)
        if config.preNorm { _embedNorm.wrappedValue = LayerNorm(dimensions: config.width) }
        _layers.wrappedValue = (0 ..< config.layers).map { _ in NemotronSpeechBlock(config) }
        _finalNorm.wrappedValue = LayerNorm(dimensions: config.width)
        scale = config.scaleInput ? sqrt(Float(config.width)) : 1
    }
    // The speech API operates on one recording, trimmed to its valid feature length.
    func callAsFunction(_ x: MLXArray) throws -> MLXArray {
        let replacementWeights = try reloadLayerWeights?()
        if let replacementWeights, replacementWeights.count != layers.count {
            throw VLMError.processing("Invalid streamed speech layer count")
        }
        var x = x * scale
        if let embedNorm { x = embedNorm(x) }
        for (index, layer) in layers.enumerated() {
            try Task.checkCancellation()
            x = try autoreleasepool {
                let output = layer(x)
                if let replacementWeights {
                    // Complete every use before replacing the module's weights.
                    // The next audio chunk reloads the exact source tensors.
                    eval(output)
                    try layer.update(parameters: replacementWeights[index], verify: [.all])
                }
                return output
            }
        }
        return finalNorm(x)
    }
}

final class NemotronSpeakerHead: Module {
    @ModuleInfo(key: "encoder_proj") var encoderProjection: Linear
    @ModuleInfo(key: "first_hidden_to_hidden") var hidden: Linear
    @ModuleInfo(key: "single_hidden_to_spks") var single: Linear
    @ModuleInfo(key: "hidden_to_spks") var combined: Linear
    @ModuleInfo(key: "subpixel_upsample") var upsample: Conv1d
    @ParameterInfo(key: "learnable_sil_emb") var silence: MLXArray
    init(_ config: NemotronTranscribeConfiguration) {
        let width = config.speakerHeadWidth
        _encoderProjection.wrappedValue = Linear(config.speaker.width, width)
        _hidden.wrappedValue = Linear(width, width)
        _single.wrappedValue = Linear(width, config.speakerCount)
        _combined.wrappedValue = Linear(2 * width, config.speakerCount)
        _upsample.wrappedValue = Conv1d(
            inputChannels: width, outputChannels: width * config.speaker.factor,
            kernelSize: 3, padding: 1)
        _silence.wrappedValue = MLXArray.zeros([config.speaker.width])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x = encoderProjection(x)
        let high = upsample(x).reshaped(x.dim(0), -1, x.dim(2))
        return sigmoid(single(relu(hidden(relu(high)))))
    }
}

final class NemotronSpeakerEncoder: Module {
    @ModuleInfo var encoder: NemotronSpeechEncoder
    @ModuleInfo var preprocessor: NemotronSpeechMel
    @ModuleInfo(key: "sortformer_modules") var head: NemotronSpeakerHead
    let config: NemotronTranscribeConfiguration

    struct State {
        var cache: MLXArray
        var predictions: MLXArray
        var fifo: MLXArray
        var fifoPredictions: MLXArray
        var compressed = false
    }
    init(_ config: NemotronTranscribeConfiguration) {
        self.config = config
        _encoder.wrappedValue = NemotronSpeechEncoder(config.speaker)
        _preprocessor.wrappedValue = NemotronSpeechMel(config.speakerMel)
        _head.wrappedValue = NemotronSpeakerHead(config)
    }
    func newState() -> State {
        let empty = MLXArray.zeros(
            [1, 0, config.speaker.width], dtype: encoder.preEncode.proj.weight.dtype)
        let predictions = MLXArray.zeros([1, 0, config.speakerCount])
        return State(
            cache: empty, predictions: predictions, fifo: empty, fifoPredictions: predictions)
    }
    func step(_ features: MLXArray, state: inout State, centralFrames: Int, leftFrames: Int) throws
        -> MLXArray
    {
        let factor = config.speaker.factor
        let chunk = encoder.preEncode(features.asType(encoder.preEncode.proj.weight.dtype))
        let cacheLength = state.cache.dim(1)
        let fifoLength = state.fifo.dim(1)
        let high = try head(encoder(concatenated([state.cache, state.fifo, chunk], axis: 1)))
        let low = high.asType(.float32).reshaped(1, -1, factor, config.speakerCount).mean(axis: 2)
        let count = (centralFrames + factor - 1) / factor
        let left = leftFrames / factor
        let start = cacheLength + fifoLength + left
        let result = high[0, (start * factor) ..< (start * factor + centralFrames), 0...].asType(
            .float32)
        state.fifo = concatenated(
            [state.fifo, chunk[0..., left ..< (left + count), 0...]], axis: 1)
        state.fifoPredictions = concatenated(
            [
                low[0..., cacheLength ..< (cacheLength + fifoLength), 0...],
                low[0..., start ..< (start + count), 0...],
            ], axis: 1)
        if state.fifo.dim(1) > config.fifoLength {
            let pop = min(
                state.fifo.dim(1),
                max(config.speakerCacheUpdatePeriod, state.fifo.dim(1) - config.fifoLength))
            state.cache = concatenated([state.cache, state.fifo[0..., ..<pop, 0...]], axis: 1)
            let previous = state.compressed ? state.predictions : low[0..., ..<cacheLength, 0...]
            state.predictions = concatenated(
                [previous, state.fifoPredictions[0..., ..<pop, 0...]], axis: 1)
            state.fifo = state.fifo[0..., pop..., 0...]
            state.fifoPredictions = state.fifoPredictions[0..., pop..., 0...]
            if state.cache.dim(1) > config.speakerCacheLength {
                (state.cache, state.predictions) = compress(
                    state.cache, predictions: state.predictions)
                state.compressed = true
            }
        }
        eval(result, state.cache, state.predictions, state.fifo, state.fifoPredictions)
        return result
    }

    // Arrival-order speaker cache: preserve the reference's per-speaker scoring,
    // global selection and silence replacement, including repeated frame indices.
    func compress(_ embeddings: MLXArray, predictions: MLXArray) -> (MLXArray, MLXArray) {
        let cfg = config
        let perSpeaker = cfg.speakerCacheLength / cfg.speakerCount - cfg.silenceFrames
        let positiveMinimum = Int(floor(Float(perSpeaker) * cfg.minimumPositive))
        let lp = log(maximum(predictions, cfg.scoreThreshold))
        let ln = log(maximum(1 - predictions, cfg.scoreThreshold))
        var scores = lp - ln + ln.sum(axis: 2, keepDims: true) - Float(log(0.5))
        scores = MLX.where(predictions .> 0.5, scores, MLXArray(-Float.infinity))
        let positive = scores .> 0
        let enough = positive.asType(.float32).sum(axis: 1, keepDims: true) .>= positiveMinimum
        let remove = logicalAnd(logicalAnd(logicalNot(positive), predictions .> 0.5), enough)
        scores = MLX.where(remove, MLXArray(-Float.infinity), scores)
        if cfg.latestBoost > 0, scores.dim(1) > cfg.speakerCacheLength {
            let boost = (MLXArray(0 ..< scores.dim(1)) .>= cfg.speakerCacheLength).asType(.float32)
            scores = scores + boost.reshaped(1, -1, 1) * cfg.latestBoost
        }
        func boosted(_ input: MLXArray, count: Int, scale: Float) -> MLXArray {
            guard count > 0 else { return input }
            let k = min(count, input.dim(1))
            return stacked(
                (0 ..< cfg.speakerCount).map { speaker in
                    let flat = input[0..., 0..., speaker]
                    let indices = argPartition(-flat, kth: k - 1, axis: 1)[0..., ..<k]
                    let mask = putAlong(
                        MLXArray.zeros(like: flat), indices, values: MLXArray.ones(indices.shape),
                        axis: 1)
                    return flat + mask * (-scale * Float(log(0.5)))
                        * (flat .> -Float.infinity).asType(.float32)
                }, axis: -1)
        }
        scores = boosted(scores, count: Int(floor(Float(perSpeaker) * cfg.strongBoost)), scale: 2)
        scores = boosted(scores, count: Int(floor(Float(perSpeaker) * cfg.weakBoost)), scale: 1)
        if cfg.silenceFrames > 0 {
            scores = concatenated(
                [
                    scores,
                    MLXArray.full(
                        [1, cfg.silenceFrames, cfg.speakerCount], values: MLXArray(Float.infinity)),
                ], axis: 1)
        }
        let frames = scores.dim(1)
        let flat = scores.transposed(0, 2, 1).reshaped(1, -1)
        var indices = argPartition(-flat, kth: cfg.speakerCacheLength - 1, axis: 1)[
            0..., ..<cfg.speakerCacheLength]
        indices = MLX.where(
            takeAlong(flat, indices, axis: 1) .> -Float.infinity, indices,
            MLXArray(cfg.maximumIndex))
        indices = sorted(indices, axis: 1)
        let disabled = logicalOr(
            indices .== cfg.maximumIndex, (indices % frames) .>= (frames - cfg.silenceFrames))
        indices = MLX.where(disabled, MLXArray(0), indices % frames)
        let mask = disabled.expandedDimensions(axis: -1)
        let gathered = takeAlong(
            embeddings,
            broadcast(
                indices.expandedDimensions(axis: -1),
                to: [1, cfg.speakerCacheLength, cfg.speaker.width]), axis: 1)
        let probs = takeAlong(
            predictions,
            broadcast(
                indices.expandedDimensions(axis: -1),
                to: [1, cfg.speakerCacheLength, cfg.speakerCount]), axis: 1)
        return (
            MLX.where(mask, head.silence.reshaped(1, 1, -1), gathered),
            MLX.where(mask, MLXArray(0), probs)
        )
    }
}

final class NemotronParallelSpeechEncoder: Module {
    @ModuleInfo(key: "asr_encoder") var speech: NemotronSpeechEncoder
    @ModuleInfo(key: "diarization_model") var speaker: NemotronSpeakerEncoder
    @ModuleInfo(key: "asr_norm") var speechNorm: LayerNorm
    @ModuleInfo(key: "diar_norm") var speakerNorm: LayerNorm
    let config: NemotronTranscribeConfiguration
    private let _speakerKernel: MLXArray
    init(_ config: NemotronTranscribeConfiguration) {
        self.config = config
        _speech.wrappedValue = NemotronSpeechEncoder(config.speech)
        _speaker.wrappedValue = NemotronSpeakerEncoder(config)
        _speechNorm.wrappedValue = LayerNorm(dimensions: config.speech.width)
        _speakerNorm.wrappedValue = LayerNorm(dimensions: config.speakerCount)
        let positions = MLXArray(0 ..< config.speakerCount).expandedDimensions(axis: 1)
        let frequencies = exp(
            MLXArray(stride(from: 0, to: config.speech.width, by: 2))
                * (-Foundation.log(Float(10000)) / Float(config.speech.width)))
        let angles = positions * frequencies
        _speakerKernel = stacked([sin(angles), cos(angles)], axis: -1).reshaped(
            config.speakerCount, config.speech.width)
    }
    static func normalize(_ features: MLXArray) -> MLXArray {
        let reference = features[.ellipsis, ..<1]
        let mean = reference + (features - reference).mean(axis: -1, keepDims: true)
        let centered = features - mean
        let variance =
            centered.square().sum(axis: -1, keepDims: true) / max(1, features.dim(-1) - 1)
        return centered / (sqrt(variance) + 1e-5)
    }
    func callAsFunction(_ features: MLXArray) throws -> MLXArray {
        let factor = config.speech.factor
        let total = features.dim(-1)
        let normalized = Self.normalize(features).asType(speechNorm.weight!.dtype)
        var state = speaker.newState()
        var outputs: [MLXArray] = []
        for start in stride(from: 0, to: total, by: config.chunkLength * factor) {
            try Task.checkCancellation()
            let end = min(start + config.chunkLength * factor, total)
            let contextStart = max(0, start - config.leftContext * factor)
            let contextEnd = min(total, end + config.rightContext * factor)
            let encoded = try speech(
                speech.preEncode(normalized[0..., 0..., contextStart ..< contextEnd]))
            let first = (start - contextStart) / factor
            let count = (end + factor - 1) / factor - (start + factor - 1) / factor
            let states = encoded[0..., first ..< (first + count), 0...]
            var probabilities = try speaker.step(
                features[0..., 0..., contextStart ..< contextEnd], state: &state,
                centralFrames: count * factor, leftFrames: start - contextStart
            )
            .reshaped(1, count, factor, config.speakerCount).mean(axis: 2).asType(states.dtype)
            if let threshold = config.speakerThreshold {
                probabilities = (probabilities .> threshold).asType(states.dtype)
            }
            let targets = speakerNorm(probabilities)
            let infusion = matmul(targets, _speakerKernel.asType(targets.dtype))
            let output = speechNorm(states) + config.speakerScale * infusion
            eval(output)
            outputs.append(output)
        }
        return concatenated(outputs, axis: 1)
    }
}

final class NemotronSpeechPerception: Module {
    @ModuleInfo var preprocessor: NemotronSpeechMel
    @ModuleInfo var encoder: NemotronParallelSpeechEncoder
    @ModuleInfo var proj: Linear
    init(_ config: NemotronTranscribeConfiguration) {
        _preprocessor.wrappedValue = NemotronSpeechMel(config.mel)
        _encoder.wrappedValue = NemotronParallelSpeechEncoder(config)
        _proj.wrappedValue = Linear(config.speech.width, config.language.hiddenSize)
    }
    func callAsFunction(_ audio: MLXArray) throws -> MLXArray {
        try proj(encoder(preprocessor(audio)))
    }
}
