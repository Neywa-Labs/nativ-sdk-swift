// Copyright 2026 The HuggingFace Team. All rights reserved.
// Adapted from the Apache-2.0 mlx-vlm LFM3-Audio and LFM3-Omni implementations.
import Foundation
import MLX
import MLXNN

public struct LFM3AudioOptions: AudioGenerationOptions {
    public enum Mode: String, Sendable { case automatic, text, audio, interleaved }
    public var mode: Mode
    public var maxTokens: Int
    public var temperature: Float?
    public var topK: Int?
    public var repetitionPenalty: Float?
    /// Nil uses the family default: 4096 frames for Audio, 64 frames for Omni.
    public var repetitionWindow: Int?
    public init(
        mode: Mode = .automatic, maxTokens: Int = 2048, temperature: Float? = nil, topK: Int? = nil,
        repetitionPenalty: Float? = nil, repetitionWindow: Int? = nil
    ) {
        self.mode = mode
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.repetitionWindow = repetitionWindow
    }
}

/// Shared speech model; LFM3-Omni additionally installs the vision tower.
public class LFM3Audio: Module, VLMModel, KVCacheDimensionProvider {
    let config: LFM3Configuration
    public let vocabularySize: Int
    public var kvHeads: [Int] {
        config.text.layerTypes.map { $0 == "conv" ? 0 : config.text.kvHeads }
    }
    public var loraLayers: [Module] { language.model.layers }
    public var toolCallFormat: ToolCallFormat? { .xmlFunction }
    @ModuleInfo(key: "language_model") var language: LFM3Language
    @ModuleInfo(key: "audio_tower") var audioTower: LFM3AudioEncoder
    @ModuleInfo(key: "audio_projector") var audioProjector: LFM3Projector
    @ModuleInfo(key: "codebook_embedding") var codebookEmbedding: Embedding
    @ModuleInfo(key: "depth_decoder") var depthDecoder: LFM3DepthDecoder
    @ModuleInfo(key: "vision_tower") var visionTower: LFM3Vision?
    @ModuleInfo(key: "multi_modal_projector") var visionProjector: LFM3Projector?
    // Codec assets are separate from the language checkpoint and loaded only for speech output.
    var codecDirectory: URL?
    @ModuleInfo var codec: LiquidCodec?

    public init(_ c: LFM3Configuration) {
        config = c
        vocabularySize = c.text.vocabularySize
        _language.wrappedValue = LFM3Language(c.text)
        _audioTower.wrappedValue = LFM3AudioEncoder(c.audio)
        _audioProjector.wrappedValue = LFM3Projector(
            input: c.audio.hiddenSize, hidden: c.text.hiddenSize, output: c.text.hiddenSize)
        _codebookEmbedding.wrappedValue = Embedding(
            embeddingCount: c.numCodebooks * c.codebookSize, dimensions: c.text.hiddenSize)
        _depthDecoder.wrappedValue = LFM3DepthDecoder(c)
        if let vision = c.vision {
            _visionTower.wrappedValue = LFM3Vision(vision)
            _visionProjector.wrappedValue = LFM3Projector(
                input: vision.hiddenSize, hidden: c.projectorSize,
                output: c.text.hiddenSize, merge: c.downsample, bias: c.projectorBias,
                eps: c.projectorEps, fp32Activation: false)
        }
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try language.model.newCache(configuration: configuration)
    }

    var placeholderIDs: [Int] {
        [config.audioToken, config.audioOutToken]
            + (config.vision == nil ? [] : [config.imageToken, config.videoToken])
    }
    func textLogits(_ hidden: MLXArray, allowSpeech: Bool = false) -> MLXArray {
        let logits = language.logits(hidden)
        let suppressed =
            placeholderIDs + (allowSpeech ? [] : [config.audioStart, config.mixedStart])
        logits[.ellipsis, MLXArray(suppressed)] = MLXArray(-Float.infinity).asType(logits.dtype)
        return logits
    }
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        textLogits(language.model(language.model.embedTokens!(inputs), cache: cache))
    }

    static func merge(_ embeddings: MLXArray, ids: [Int], features: MLXArray?, token: Int) throws
        -> MLXArray
    {
        let indices = ids.indices.filter { ids[$0] == token }
        guard indices.count == (features?.dim(0) ?? 0) else {
            throw ModelExecutionError.invalidOptions(
                "LFM3 token \(token) has \(indices.count) placeholders but \(features?.dim(0) ?? 0) feature rows"
            )
        }
        guard !indices.isEmpty, let features else { return embeddings }
        let flat = embeddings.reshaped(-1, embeddings.dim(-1))
        flat[MLXArray(indices)] = features.asType(embeddings.dtype)
        return flat.reshaped(embeddings.shape)
    }

    func inputEmbeddings(_ input: LMInput) throws -> MLXArray {
        guard
            input.text.tokens.ndim == 1
                || (input.text.tokens.ndim == 2 && input.text.tokens.dim(0) == 1),
            input.text.tokens.size > 0
        else {
            throw ModelExecutionError.invalidOptions(
                "LFM3 multimodal input requires one nonempty sequence")
        }
        let tokens = input.text.tokens.reshaped(1, -1)
        let ids = tokens.asArray(Int.self)
        guard ids.allSatisfy({ $0 >= 0 && $0 < vocabularySize }) else {
            throw ModelExecutionError.invalidOptions("LFM3 input contains out-of-vocabulary tokens")
        }
        var embeddings = language.model.embedTokens!(tokens)
        var speech: MLXArray?
        if let audio = input.audio {
            guard audio.features.ndim == 3, audio.features.dim(1) == config.audio.melBins else {
                throw ModelExecutionError.invalidOptions(
                    "LFM3 audio features must be [clips, mel bins, frames]")
            }
            if let mask = audio.mask, mask.shape != [audio.features.dim(0), audio.features.dim(2)] {
                throw ModelExecutionError.invalidOptions(
                    "LFM3 audio mask shape does not match features")
            }
            let lengths =
                audio.mask?.sum(axis: -1).asArray(Int.self)
                ?? Array(repeating: audio.features.dim(2), count: audio.features.dim(0))
            guard lengths.allSatisfy({ $0 > 0 && $0 <= audio.features.dim(2) }) else {
                throw ModelExecutionError.invalidOptions("LFM3 audio lengths must be positive")
            }
            speech = audioProjector(audioTower(audio.features, lengths: lengths))
        }
        embeddings = try Self.merge(
            embeddings, ids: ids, features: speech, token: config.audioToken)
        embeddings = try Self.merge(
            embeddings, ids: ids, features: nil, token: config.audioOutToken)
        func vision(_ pixels: MLXArray?, _ grids: [THW]?, _ token: Int) throws {
            var features: MLXArray?
            if let pixels {
                guard let visionTower, let visionProjector, let grids else {
                    throw ModelExecutionError.invalidOptions(
                        "LFM3-Audio cannot read images or video; use LFM3-Omni")
                }
                features = try visionProjector(visionTower(pixels, grids: grids))
            }
            if config.vision != nil {
                embeddings = try Self.merge(embeddings, ids: ids, features: features, token: token)
            }
        }
        try vision(input.image?.pixels, input.image?.frames, config.imageToken)
        try vision(input.video?.pixels, input.video?.frames, config.videoToken)
        return embeddings
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        let embeddings = try inputEmbeddings(input)
        return .logits(
            .init(
                logits: textLogits(
                    try prefillEmbeddings(embeddings, cache: cache, options: prefill))))
    }

    private func prefillEmbeddings(
        _ embeddings: MLXArray, cache: [KVCache], options: ModelPrefillOptions
    ) throws -> MLXArray {
        var last: MLXArray?
        let processed = try options.forEachChunk(total: embeddings.dim(1), reserving: 0) { range in
            last = language.model(embeddings[0..., range, 0...], cache: cache)
            eval(last!, cache)
        }
        if processed == 0 {
            try Task.checkCancellation()
            last = language.model(embeddings, cache: cache)
            eval(last!, cache)
            options.progress?(embeddings.dim(1), embeddings.dim(1))
        }
        return last!
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Self.sanitize(weights: weights, tied: config.text.tieEmbeddings)
    }

    static func sanitize(weights: [String: MLXArray], tied: Bool) -> [String: MLXArray] {
        let prefixes = [
            ("lfm.model.", "language_model.model."),
            ("encoder.downsample.", "audio_tower.downsample."),
            ("encoder.lfm.lfm.", "audio_tower.lfm."),
            ("mlp.model.0.", "audio_projector.layer_norm."),
            ("mlp.model.1.", "audio_projector.linear_1."),
            ("mlp.model.3.", "audio_projector.linear_2."),
            ("depthformer.model.", "depth_decoder.model."),
            ("depth_linear.", "depth_decoder.depth_linear."),
            ("mask_embedding", "depth_decoder.mask_embedding"),
            ("depth_embeddings.", "depth_decoder.depth_embeddings."),
            ("audio_encoder.", "audio_tower."),
        ]
        var result = [String: MLXArray]()
        for (rawName, tensor) in weights {
            var name = rawName
            var value = tensor
            if name.hasPrefix("model.language_model.") {
                name = "language_model.model." + name.dropFirst("model.language_model.".count)
            } else if name.hasPrefix("model.") {
                name = String(name.dropFirst(6))
            } else if name.hasPrefix("lm_head.") {
                name = "language_model." + name
            }
            for (old, new) in prefixes where name.hasPrefix(old) {
                name = new + name.dropFirst(old.count)
                break
            }
            if name.contains(".to_logits.")
                || (tied && name == "language_model.lm_head.weight")
            {
                continue
            }
            if name.hasSuffix("conv.conv.weight"), value.ndim == 3, value.dim(1) == 1 {
                value = value.transposed(0, 2, 1)
            }
            if name.hasSuffix("spatial_conv.weight"), value.ndim == 4, value.dim(1) == 1 {
                value = value.transposed(0, 2, 3, 1)
            }
            if name.hasPrefix("audio_tower.downsample.") {
                if value.ndim == 4, value.dim(2) == value.dim(3) {
                    value = value.transposed(0, 2, 3, 1)
                } else if value.ndim == 3 {
                    value = value[.ellipsis, 0]
                }
            }
            result[name] = value
        }
        return result
    }

    func loadCodec() throws -> LiquidCodec {
        if let codec { return codec }
        guard let codecDirectory else {
            throw ModelExecutionError.invalidOptions(
                "LFM3 speech output requires audio_tokenizer/config.json and codec weights")
        }
        let loaded = try LiquidCodec.load(
            from: codecDirectory, dtype: language.model.norm.weight.dtype)
        guard loaded.config.codebooks == config.numCodebooks,
            loaded.config.levels.reduce(1, *) == config.audioEOS
        else {
            throw ModelFactoryError.invalidConfiguration(
                "LFM3 and Liquid codec codebooks do not match")
        }
        codec = loaded
        return loaded
    }
    func embedFrame(_ codes: MLXArray) -> MLXArray {
        codebookEmbedding(codes + MLXArray(0 ..< config.numCodebooks) * config.codebookSize).sum(
            axis: -2)
    }
}

public final class LFM3Omni: LFM3Audio {}

extension LFM3Audio: JointAudioGeneratingLanguageModel {
    func validateAudioOptions(_ options: (any AudioGenerationOptions)?) throws {
        guard options == nil || options is LFM3AudioOptions else {
            throw ModelExecutionError.invalidOptions("LFM3 audio requires LFM3AudioOptions")
        }
        let speech = options as? LFM3AudioOptions ?? .init()
        guard speech.maxTokens > 0,
            speech.temperature.map({ $0.isFinite && $0 >= 0 }) ?? true,
            speech.topK.map({ $0 >= 0 }) ?? true,
            speech.repetitionPenalty.map({ $0.isFinite && $0 > 0 }) ?? true,
            speech.repetitionWindow.map({ $0 >= 0 }) ?? true
        else { throw ModelExecutionError.invalidOptions("Invalid LFM3 speech sampling options") }
    }
    func generateJointAudio(
        input: LMInput, options: ModelGenerationOptions, audio: (any AudioGenerationOptions)?,
        tokenizer: any Tokenizer
    ) throws -> ModelAudioResult {
        let codec = try loadCodec()
        let c = config
        try validateAudioOptions(audio)
        let speech = audio as? LFM3AudioOptions ?? .init()
        let temperature = speech.temperature ?? (c.vision == nil ? 1 : 0.4)
        let topK = speech.topK ?? (c.vision == nil ? 128 : 0)
        let penalty = speech.repetitionPenalty ?? (c.vision == nil ? 1.1 : 1.2)
        let repetitionWindow = speech.repetitionWindow ?? (c.vision == nil ? 4096 : 64)
        guard temperature.isFinite, temperature >= 0, topK >= 0, penalty.isFinite, penalty > 0,
            repetitionWindow >= 0
        else {
            throw ModelExecutionError.invalidOptions("Invalid LFM3 speech sampling options")
        }
        var mode = speech.mode
        var ids = input.text.tokens.asArray(Int.self)
        var embeddings = try inputEmbeddings(input)
        if mode == .automatic {
            mode =
                ids.last == c.audioStart || ids.last == c.audioOutToken
                ? .audio : ids.last == c.mixedStart ? .interleaved : .text
        } else if mode == .audio || mode == .interleaved {
            let marker = mode == .audio ? c.audioStart : c.mixedStart
            if ids.last != marker {
                ids.append(marker)
                embeddings = concatenated(
                    [embeddings, language.model.embedTokens!(MLXArray([marker], [1, 1]))], axis: 1)
            }
        }
        let cache = try newCache(configuration: options.cache)
        let start = Date()
        var hidden = try prefillEmbeddings(embeddings, cache: cache, options: options.prefill)[
            0..., -1, 0...]
        eval(hidden)
        let prefillSeconds = Date().timeIntervalSince(start)
        let generationStart = Date()
        let random =
            options.sampling.seed.map { MLXRandom.RandomState(seed: $0) } ?? MLXRandom.RandomState()
        let sampler = options.sampling.makeSampler()
        var processor = options.sampling.makeProcessor()
        processor.prompt(MLXArray(ids))
        var tokens = [Int]()
        var frames = [MLXArray]()
        var history = [MLXArray]()
        var slotsLeft = c.textSlots
        var speaking = false
        var textDone = false
        var stopped = false
        let eos = tokenizer.convertTokenToId("<|im_end|>") ?? 124900
        let limit = min(speech.maxTokens, options.output.maxTokens ?? speech.maxTokens)
        for step in 0 ..< limit {
            try Task.checkCancellation()
            let interleaving = mode == .interleaved
            let embed: MLXArray
            if mode == .audio || (interleaving && speaking) {
                let codes = try depthDecoder.sampleFrame(
                    hidden, temperature: temperature, topK: topK, penalty: penalty,
                    history: history, random: random)
                if codes[0].item(Int.self) == c.audioEOS {
                    mode = .text
                } else {
                    frames.append(codes)
                    history.append(codes)
                    if history.count > repetitionWindow {
                        history.removeFirst(history.count - repetitionWindow)
                    }
                }
                tokens.append(c.audioOutToken)
                embed = embedFrame(codes).reshaped(1, 1, -1)
            } else {
                let logits = processor.process(logits: textLogits(hidden, allowSpeech: true))
                var suppressed = options.sampling.suppressedTokenIDs
                if speech.mode == .text { suppressed.formUnion([c.audioStart, c.mixedStart]) }
                guard suppressed.allSatisfy({ $0 >= 0 && $0 < vocabularySize }),
                    suppressed.count + placeholderIDs.count < vocabularySize
                else {
                    throw ModelExecutionError.invalidOptions("Invalid LFM3 suppressed token IDs")
                }
                if !suppressed.isEmpty {
                    logits[.ellipsis, MLXArray(Array(suppressed))] = MLXArray(-Float.infinity)
                }
                let sampled = sampler.sample(logits: logits)
                let token = sampled.item(Int.self)
                processor.didSample(token: sampled)
                tokens.append(token)
                if token == c.textEnd {
                    if interleaving {
                        textDone = true
                        speaking = true
                    } else {
                        stopped = true
                    }
                } else if token == eos {
                    stopped = true
                } else if !interleaving {
                    if token == c.audioStart {
                        mode = .audio
                    } else if token == c.mixedStart {
                        mode = .interleaved
                        slotsLeft = c.textSlots
                        speaking = false
                        textDone = false
                    }
                }
                embed = language.model.embedTokens!(MLXArray([token], [1, 1]))
            }
            if interleaving && !textDone {
                slotsLeft -= 1
                if slotsLeft == 0 {
                    speaking.toggle()
                    slotsLeft = speaking ? c.audioSlots : c.textSlots
                }
            }
            if stopped || step == limit - 1 { break }
            hidden = language.model(embed, cache: cache)[0..., -1, 0...]
            eval(hidden)
        }
        let codes =
            frames.isEmpty
            ? MLXArray.zeros([c.numCodebooks, 0], dtype: .int32) : stacked(frames, axis: -1)
        try Task.checkCancellation()
        let waveform = try codec.decode(codes).materialized()
        let special = Set(placeholderIDs + [c.audioStart, c.mixedStart, c.textEnd, eos])
        return ModelAudioResult(
            waveform: waveform,
            sampleRate: codec.config.sampleRate, tokenIDs: tokens,
            text: tokenizer.decode(
                tokenIds: tokens.filter { !special.contains($0) }, skipSpecialTokens: true),
            summary: .init(
                promptTokenCount: ids.count, generatedTokenCount: tokens.count,
                promptSeconds: prefillSeconds,
                generationSeconds: Date().timeIntervalSince(generationStart),
                stopReason: stopped ? .stop : .length))
    }
}
