import Foundation

/// Configuration of the consolidated Nemotron 3.5 speech checkpoint.
public struct NemotronTranscribeConfiguration: Decodable, Sendable {
    public let language: NemotronHConfiguration
    let speech: NemotronSpeechEncoderConfiguration
    let speaker: NemotronSpeechEncoderConfiguration
    let mel: NemotronSpeechMelConfiguration
    let speakerMel: NemotronSpeechMelConfiguration
    let speakerCount: Int
    let speakerHeadWidth: Int
    let chunkLength: Int
    let leftContext: Int
    let rightContext: Int
    let fifoLength: Int
    let speakerCacheLength: Int
    let speakerCacheUpdatePeriod: Int
    let silenceFrames: Int
    let scoreThreshold: Float
    let latestBoost: Float
    let strongBoost: Float
    let weakBoost: Float
    let minimumPositive: Float
    let maximumIndex: Int
    let speakerThreshold: Float?
    let speakerScale: Float
    let audioTag: String
    let padTokenID: Int

    public init(from decoder: Decoder) throws {
        let root = try SpeechJSON(from: decoder)
        let text = try root.object("llm_config")
        guard try text.string("model_type", "") == "nemotron_h",
            try root.string("prompt_format", "nemotron3p5") == "nemotron3p5"
        else { throw Self.invalid("Expected a Nemotron-H backbone and nemotron3p5 prompts") }
        var language = try JSONDecoder().decode(
            NemotronHConfiguration.self, from: JSONEncoder().encode(text.values))
        language.speechPrecision = true
        self.language = language
        let pe = try root.object("pe_encoder_config")
        let perception = try root.object("perception")
        let diar = try pe.object("diarization_model_cfg")
        let modules = try diar.object("sortformer_modules")
        speech = try .init(pe.object("asr_encoder_cfg"))
        speaker = try .init(diar.object("encoder"))
        mel = try .init(perception.object("preprocessor"))
        speakerMel = try .init(diar.object("preprocessor"))
        speakerCount = try modules.int("num_spks", 8)
        speakerHeadWidth = try modules.int("tf_d_model", 192)
        chunkLength = try pe.int("online_inference_length", 500)
        leftContext = max(0, try pe.int("chunk_left_context", 50))
        rightContext = max(0, try pe.int("chunk_right_context", 50))
        fifoLength = try pe.int("diar_fifo_len", 40)
        speakerCacheLength = try pe.int("diar_spkcache_len", 188)
        speakerCacheUpdatePeriod = try pe.int("diar_spkcache_update_period", 300)
        silenceFrames = try modules.int("spkcache_sil_frames_per_spk", 1)
        scoreThreshold = try modules.float("pred_score_threshold", 0.25)
        latestBoost = try modules.float("scores_boost_latest", 0.05)
        strongBoost = try modules.float("strong_boost_rate", 0.75)
        weakBoost = try modules.float("weak_boost_rate", 1.5)
        minimumPositive = try modules.float("min_pos_scores_rate", 0.5)
        maximumIndex = try modules.int("max_index", 99999)
        speakerScale = try pe.float("spk_kernel_scale", 1)
        audioTag = try root.string("audio_locator_tag", "<|audio|>")
        padTokenID = try root.int("pad_token_id", 0)
        switch try pe.string("speaker_feature_mode", "") {
        case "thresholded":
            speakerThreshold = try pe.float("speaker_activity_threshold", -1)
            guard (0 ... 1).contains(speakerThreshold!) else {
                throw Self.invalid("Expected a speaker activity threshold in 0...1")
            }
        case "continuous":
            guard
                pe.values["speaker_activity_threshold"] == nil
                    || pe.values["speaker_activity_threshold"] == .null
            else { throw Self.invalid("Continuous speaker features cannot have a threshold") }
            speakerThreshold = nil
        default: throw Self.invalid("Unsupported speaker feature mode")
        }
        guard try pe.int("speaker_feature_config_version", 0) == 1,
            try pe.string("asr_encoder_type", "transformer") == "transformer",
            try pe.string("asr_normalize_type", "per_feature") == "per_feature",
            try ["NA", ""].contains(pe.string("diar_normalize_type", "NA")),
            try diar.bool("high_resolution", false),
            try modules.bool("use_learnable_sil_emb", true),
            try perception.int("output_dim", 0) == language.hiddenSize,
            try perception.object("modality_adapter").string("_target_", "").hasSuffix(
                "IdentityConnector"),
            speech.factor == speaker.factor, speech.features == mel.features,
            speaker.features == speakerMel.features,
            try modules.int("fc_d_model", speaker.width) == speaker.width,
            speakerCount > 0, speakerHeadWidth > 0, chunkLength > 0, fifoLength >= 0,
            speakerCacheUpdatePeriod > 0, silenceFrames >= 0,
            speakerCacheLength >= (silenceFrames + 1) * speakerCount,
            scoreThreshold > 0, scoreThreshold <= 1
        else { throw Self.invalid("Unsupported speech encoder or speaker-cache configuration") }
    }

    static func invalid(_ message: String) -> ModelFactoryError {
        .invalidConfiguration("Nemotron Transcribe: " + message)
    }
}

struct NemotronSpeechEncoderConfiguration: Sendable {
    let features: Int
    let width: Int
    let layers: Int
    let heads: Int
    let factor: Int
    let expansion: Float
    let qkvBias: Bool
    let qkNorm: Bool
    let preNorm: Bool
    let scaleInput: Bool
    let ropeBase: Float
    let rotaryFraction: Float

    fileprivate init(_ json: SpeechJSON) throws {
        features = try json.int("feat_in", 128)
        width = try json.int("d_model", 512)
        layers = try json.int("n_layers", 31)
        heads = try json.int("n_heads", 8)
        factor = try json.int("subsampling_factor", 8)
        expansion = try json.float("ff_expansion", 4)
        qkvBias = try json.bool("qkv_bias", false)
        qkNorm = try json.bool("qk_norm", false)
        preNorm = try json.bool("pre_block_norm", true)
        scaleInput = try json.bool("xscaling", false)
        ropeBase = try json.float("rope_base", 10000)
        rotaryFraction = try json.float("rotary_fraction", 1)
        guard try json.string("subsampling", "feature_stacking") == "feature_stacking",
            try json.string("self_attention_model", "rope") == "rope",
            try json.string("attn_mode", "full") == "full", try json.int("feat_out", -1) == -1,
            width > 0, heads > 0, width % heads == 0, factor > 0, layers > 0,
            features > 0, expansion > 0, rotaryFraction > 0, rotaryFraction <= 1,
            Int(Float(width / heads) * rotaryFraction) > 0,
            Int(Float(width / heads) * rotaryFraction) % 2 == 0
        else {
            throw NemotronTranscribeConfiguration.invalid("Unsupported feature-stacking encoder")
        }
    }
}

struct NemotronSpeechMelConfiguration: Sendable {
    let sampleRate: Int
    let features: Int
    let fftLength: Int
    let hopLength: Int
    let windowLength: Int
    let preemphasis: Float

    fileprivate init(_ json: SpeechJSON) throws {
        sampleRate = try json.int("sample_rate", 16000)
        features = try json.int("features", 128)
        fftLength = try json.int("n_fft", 512)
        hopLength = Int(Float(sampleRate) * (try json.float("window_stride", 0.01)))
        windowLength = Int(Float(sampleRate) * (try json.float("window_size", 0.025)))
        preemphasis = try json.float("preemph", 0.97)
        guard try json.string("window", "hann") == "hann",
            try json.int("frame_splicing", 1) == 1, try !json.bool("exact_pad", false),
            try json.float("mag_power", 2) == 2, try json.bool("log", true),
            try json.string("log_zero_guard_type", "add") == "add",
            try json.float("log_zero_guard_value", pow(2, -24)) == pow(2, -24),
            try ["", "NA"].contains(json.string("normalize", "")),
            sampleRate > 0, features > 0, hopLength > 0,
            windowLength > 0, windowLength <= fftLength
        else { throw NemotronTranscribeConfiguration.invalid("Unsupported log-mel preprocessing") }
    }
}

private struct SpeechJSON: Decodable {
    let values: [String: JSONValue]
    init(from decoder: Decoder) throws { values = try [String: JSONValue](from: decoder) }
    init(_ values: [String: JSONValue]) { self.values = values }
    func object(_ key: String) throws -> SpeechJSON {
        guard case .object(let value) = values[key] else { throw invalid(key) }
        return SpeechJSON(value)
    }
    func int(_ key: String, _ fallback: Int) throws -> Int {
        guard let value = values[key], value != .null else { return fallback }
        guard case .int(let number) = value else { throw invalid(key) }
        return number
    }
    func float(_ key: String, _ fallback: Float) throws -> Float {
        guard let value = values[key], value != .null else { return fallback }
        switch value {
        case .int(let number): return Float(number)
        case .double(let number): return Float(number)
        default: throw invalid(key)
        }
    }
    func bool(_ key: String, _ fallback: Bool) throws -> Bool {
        guard let value = values[key], value != .null else { return fallback }
        guard case .bool(let boolean) = value else { throw invalid(key) }
        return boolean
    }
    func string(_ key: String, _ fallback: String) throws -> String {
        guard let value = values[key], value != .null else { return fallback }
        guard case .string(let string) = value else { throw invalid(key) }
        return string
    }
    private func invalid(_ key: String) -> ModelFactoryError {
        NemotronTranscribeConfiguration.invalid("Missing or invalid \(key)")
    }
}
