import Foundation
import MLX
import MLXNN

public enum NemotronTranscribeTask: String, CaseIterable, Sendable {
    case transcription = "asr"
    case translation = "translate"
    case speakerAttribution = "saa"
    case richTranscription = "rich"
    case audioQuestion = "qa"

    public func prompt(language: String? = nil) throws -> String {
        switch self {
        case .transcription:
            return
                "Produce a verbatim transcript of the audio. Preserve named entities, abbreviations, numbers, dates, measurements, acronyms, and technical terms as clearly as possible. Keep the same language as spoken and do not translate."
        case .translation:
            guard let language, !language.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else {
                throw VLMError.processing("Translation requires a target language")
            }
            return "Translate the speech in this audio into \(language)."
        case .speakerAttribution:
            return
                "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. Write numbers in their standard written form. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> <spk:4> <spk:5> <spk:6> <spk:7> on the transcription."
        case .richTranscription:
            return
                "Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. Write numbers in their standard written form. Use <ev:*> and <sty:*> special tokens to recognize all non-speech verbatims and speaking style."
        case .audioQuestion: return "<|audio|>"
        }
    }
}

/// Speech and speaker encoders feeding the shared Nemotron-H decoder. Routed
/// SwitchLinear modules use the shared expert streaming loader.
public final class NemotronTranscribe: Module, VLMModel, KVCacheDimensionProvider,
    MixedQuantizationPolicyProviding
{
    public let config: NemotronTranscribeConfiguration
    public let vocabularySize: Int
    public let kvHeads: [Int]
    @ModuleInfo var llm: NemotronHModel
    @ModuleInfo var perception: NemotronSpeechPerception

    // With expert streaming, speech weights can be released after encoding.
    // The next audio request reloads the same tensors.
    private var reloadPerceptionWeights: (() throws -> [String: MLXArray])?

    func configurePerceptionStreaming(urls: [URL]) throws -> Set<String> {
        let tensors = try ExpertTensorIndex(urls: urls).tensors.filter {
            $0.key.hasPrefix("perception.")
        }
        let files = Dictionary(grouping: tensors.keys, by: { tensors[$0]!.url })
        reloadPerceptionWeights = {
            var weights: [String: MLXArray] = [:]
            for (url, names) in files {
                let arrays = try loadArrays(url: url, stream: .cpu)
                for name in names { weights[name] = arrays[name] }
            }
            return weights
        }
        let reload = reloadPerceptionWeights!
        let count = perception.encoder.speech.layers.count
        perception.encoder.speech.reloadLayerWeights = { [weak self] in
            guard let self else { return [] }
            let weights = self.sanitize(weights: try reload())
            return (0 ..< count).map { index in
                let prefix = "perception.encoder.asr_encoder.layers.\(index)."
                return .unflattened(
                    Dictionary(
                        uniqueKeysWithValues: weights.compactMap {
                            name, value in
                            name.hasPrefix(prefix)
                                ? (String(name.dropFirst(prefix.count)), value) : nil
                        }))
            }
        }
        return Set(tensors.keys)
    }

    private func releasePerceptionWeights() throws {
        guard let reloadPerceptionWeights else { return }
        let weights = sanitize(weights: try reloadPerceptionWeights())
        let parameters = Dictionary(
            uniqueKeysWithValues: weights.map {
                (String($0.key.dropFirst("perception.".count)), $0.value)
            })
        try perception.update(parameters: .unflattened(parameters), verify: [.all])
    }

    public var loraLayers: [Module] { llm.loraLayers }

    public init(_ config: NemotronTranscribeConfiguration) {
        self.config = config
        let language = NemotronHModel(config.language)
        vocabularySize = language.vocabularySize
        kvHeads = language.kvHeads
        _llm.wrappedValue = language
        _perception.wrappedValue = NemotronSpeechPerception(config)
    }

    public var reasoningConfig: ReasoningConfig? {
        .init(
            startDelimiter: "<think>", endDelimiter: "</think>",
            promptStrategy: .templateFlag(key: "enable_thinking", defaultOn: false))
    }

    public func quantizationPredicate(path: String, module: Module) -> QuantizationDecision {
        // Match MLX Audio: speech and speaker encoders retain their source precision.
        path.hasPrefix("llm.") ? .useDefault : .skip
    }

    /// The mixed recipe uses 3-bit routed input projections in the middle half
    /// of the decoder. Boundary layers, output projections and resident weights
    /// retain the requested defaults; speech encoders retain source precision.
    public func mixedQuantizationPredicate(path: String, module: Module) -> QuantizationDecision {
        let count = config.language.numHiddenLayers
        let layer = path.split(separator: ".").compactMap { Int($0) }.first
        if path.hasPrefix("llm."), path.hasSuffix(".fc1"), module is SwitchLinear,
            let layer, layer >= count / 4, layer < 3 * count / 4
        {
            return .parameters(.init(groupSize: 64, bits: 3))
        }
        return quantizationPredicate(path: path, module: module)
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try llm.newCache(configuration: configuration)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        llm(inputs, cache: cache)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        guard input.image == nil, input.video == nil, let audio = input.audio,
            let mask = audio.mask, audio.features.ndim == 1
        else { throw VLMError.processing("Nemotron Transcribe requires one audio recording") }
        let tokens = input.text.tokens.reshaped(1, -1)
        let positions = mask.asType(.int32).asArray(Int32.self).enumerated().compactMap {
            $0.element == 0 ? nil : $0.offset
        }
        guard let first = positions.first, mask.size == tokens.size,
            positions == Array(first ..< (first + positions.count))
        else { throw VLMError.processing("Invalid speech placeholder mask") }
        let audioEmbeddings = try perception(audio.features)
        guard audioEmbeddings.dim(1) == positions.count else {
            throw VLMError.processing("Audio encoder length does not match its prompt placeholders")
        }
        let textEmbeddings = llm.inputEmbeddings(tokens)
        let embeddings = concatenated(
            [
                textEmbeddings[0..., ..<first, 0...], audioEmbeddings.asType(textEmbeddings.dtype),
                textEmbeddings[0..., (first + positions.count)..., 0...],
            ], axis: 1)
        eval(embeddings)
        try releasePerceptionWeights()
        let count = embeddings.dim(1)
        let step = prefill.chunkLength(forChunking: count) ?? count
        let errors = InferenceErrorCheck(self)
        var logits: MLXArray?
        try withPreparedCache(cache, lengths: input.text.sequenceLengths) {
            for start in stride(from: 0, to: count, by: max(1, step)) {
                try Task.checkCancellation()
                let end = min(count, start + step)
                logits = llm(
                    nil, cache: cache, inputEmbeddings: embeddings[0..., start ..< end, 0...])
                eval(logits!, cache)
                try errors.check()
                prefill.progress?(end, count)
            }
        }
        return .logits(.init(logits: logits!))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (source, tensor) in weights {
            if source.hasPrefix("llm.mtp.") || source.hasSuffix("._extra_state") { continue }
            var name =
                source
                .replacingOccurrences(
                    of: "llm.model.embed_tokens.", with: "llm.backbone.embeddings."
                )
                .replacingOccurrences(of: "llm.model.norm.", with: "llm.backbone.norm_f.")
                .replacingOccurrences(of: "llm.model.layers.", with: "llm.backbone.layers.")
                .replacingOccurrences(of: "._fp32_params.", with: ".")
                .replacingOccurrences(of: ".ffn.net.0.", with: ".ffn.linear1.")
                .replacingOccurrences(of: ".ffn.net.3.", with: ".ffn.linear2.")
                .replacingOccurrences(of: ".preprocessor.featurizer.", with: ".preprocessor.")
            var value = tensor
            if name.hasSuffix(".experts.gate_and_up_projs") {
                name = name.replacingOccurrences(
                    of: ".experts.gate_and_up_projs", with: ".switch_mlp.fc1.weight")
                value = value.swappedAxes(1, 2)
            } else if name.hasSuffix(".experts.down_projs") {
                name = name.replacingOccurrences(
                    of: ".experts.down_projs", with: ".switch_mlp.fc2.weight")
                value = value.swappedAxes(1, 2)
            } else if name.hasSuffix(".conv1d.weight"), value.dim(1) == 1 {
                value = value.swappedAxes(1, 2)
            } else if name.hasSuffix(".subpixel_upsample.weight"), value.dim(-1) == 3 {
                value = value.swappedAxes(1, 2)
            }
            if name.hasPrefix("llm."), name.hasSuffix(".gate.weight"),
                [.float16, .bfloat16, .float32].contains(value.dtype)
            {
                value = value.asType(.float32)
            }
            result[name] = value
        }
        return result
    }
}

/// Formats the checkpoint's native speech prompts without an intermediate transcript.
public struct NemotronTranscribeProcessor: UserInputProcessor {
    let config: NemotronTranscribeConfiguration
    let tokenizer: any Tokenizer

    public init(_ config: NemotronTranscribeConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.audios.count == 1, input.images.isEmpty, input.videos.isEmpty else {
            throw VLMError.processing("Pass exactly one recording to Nemotron Transcribe")
        }
        var processing = UserInput.AudioProcessing()
        processing.sampleRate = Double(config.mel.sampleRate)
        let audio = try await input.audios[0].asMLXArray(processing: processing).asType(.float32)
        guard audio.ndim == 1, audio.size >= config.mel.hopLength,
            all(isFinite(audio)).item(Bool.self)
        else {
            throw VLMError.processing(
                "Expected finite, nonempty mono audio at \(config.mel.sampleRate) Hz")
        }
        var system = ""
        var user = ""
        switch input.prompt {
        case .text(let text): user = text
        case .chat(let messages):
            guard messages.filter({ $0.role == .user }).count == 1,
                messages.allSatisfy({ $0.role == .user || $0.role == .system })
            else {
                throw VLMError.processing(
                    "Provide one user audio request and an optional system prompt")
            }
            system = messages.filter { $0.role == .system }.map(\.content).joined(separator: "\n")
            user = messages.first { $0.role == .user }!.content
        case .messages(let messages):
            guard messages.filter({ $0["role"] as? String == "user" }).count == 1,
                messages.allSatisfy({
                    ["system", "user"].contains($0["role"] as? String ?? "")
                        && $0["content"] is String
                })
            else { throw VLMError.processing("Expected plain system/user speech messages") }
            system = messages.filter { $0["role"] as? String == "system" }.compactMap {
                $0["content"] as? String
            }.joined(separator: "\n")
            user = messages.first { $0["role"] as? String == "user" }!["content"] as! String
        }
        let context = input.additionalContext ?? [:]
        let taskName = context["task"] as? String ?? "asr"
        guard
            let task = NemotronTranscribeTask(rawValue: taskName == "transcribe" ? "asr" : taskName)
        else {
            throw VLMError.processing("Unsupported speech task: \(taskName)")
        }
        if user.isEmpty { user = try task.prompt(language: context["language"] as? String) }
        if let words = context["hotwords"] as? [String], !words.isEmpty {
            user += "\nContextual vocabulary: " + words.joined(separator: ", ")
        }
        let parts = user.components(separatedBy: config.audioTag)
        guard parts.count <= 2 else {
            throw VLMError.processing("Pass exactly one audio placeholder")
        }
        if parts.count == 1 { user += " " + config.audioTag }
        let thinking =
            (context["enable_thinking"] as? Bool ?? false) ? "<think>\n" : "<think></think>"
        let turns = [
            "<|im_start|>system\n\(system)<|im_end|>\n",
            "<|im_start|>user\n\(user)<|im_end|>\n", "<|im_start|>assistant\n\(thinking)",
        ]
        let ids = turns.flatMap { tokenizer.encode(text: $0, addSpecialTokens: false) }
        guard let placeholder = tokenizer.convertTokenToId(config.audioTag),
            tokenizer.encode(text: config.audioTag, addSpecialTokens: false) == [placeholder],
            ids.filter({ $0 == placeholder }).count == 1,
            let position = ids.firstIndex(of: placeholder)
        else {
            throw VLMError.processing("Expected one native audio token in the formatted prompt")
        }
        let frames = audio.size / config.mel.hopLength
        let count = (frames + config.speech.factor - 1) / config.speech.factor
        let expanded =
            Array(ids[..<position]) + Array(repeating: config.padTokenID, count: count)
            + Array(ids[(position + 1)...])
        let mask = MLXArray(expanded.indices.map { $0 >= position && $0 < position + count })
        return LMInput(
            text: .init(tokens: MLXArray(expanded)), audio: .init(features: audio, mask: mask))
    }
}
