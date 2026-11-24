import Foundation
import MLX
import MLXNN

/// Muse Spark Open text generation with optional image, video and waveform inputs.
public final class MuseSpark: Module, VLMModel, KVCacheDimensionProvider,
    UnwrappedLanguageModelWeights
{
    @ModuleInfo(key: "language_model") var language: MuseSparkLanguage.Model
    @ModuleInfo(key: "vision_tower") var vision: MuseSparkVision.Tower?
    @ModuleInfo(key: "vision_adapter") private var adapter: MuseSparkVision.Adapter?
    @ModuleInfo(key: "vision_projection") private var projection: Linear?
    @ModuleInfo(key: "audio_tower") var audio: MuseSparkAudio.Tower?
    public let config: MuseSparkConfiguration

    public var vocabularySize: Int { config.text.vocabularySize }
    public var kvHeads: [Int] { Array(repeating: config.text.kvHeads, count: config.text.layers) }
    public var loraLayers: [Module] { language.model.layers }
    public var toolCallFormat: ToolCallFormat? { .atem }
    public var reasoningConfig: ReasoningConfig? {
        .init(
            startDelimiter: "to=self<|message|>", endDelimiter: "<|eom|>",
            promptStrategy: .none, isSpecialToken: true)
    }

    public init(_ config: MuseSparkConfiguration) {
        self.config = config
        _language.wrappedValue = MuseSparkLanguage.Model(config.text)
        if let vision = config.vision {
            _vision.wrappedValue = MuseSparkVision.Tower(vision)
            _adapter.wrappedValue = MuseSparkVision.Adapter(
                input: config.visionOutputSize, width: config.projectorSize)
            _projection.wrappedValue = Linear(
                config.projectorSize, config.text.hiddenSize, bias: false)
        }
        _audio.wrappedValue = config.audio.map(MuseSparkAudio.Tower.init)
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try config.text.layerTypes.indices.map { index in
            try makeHybridAttentionKVCache(
                configuration: configuration,
                slidingWindow: config.text.slidingWindow,
                usesSlidingWindow: config.text.isSliding(index))
        }
    }

    private func normalizeFeatures(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: .mlxNone, eps: config.text.rmsEpsilon)
    }

    func visionFeatures(_ pixels: MLXArray, grid: [THW]) throws -> MLXArray {
        guard let vision, let adapter, let projection else {
            throw VLMError.processing(
                "Muse Spark requires a checkpoint with a vision tower for images and video.")
        }
        return try normalizeFeatures(projection(adapter(vision(pixels, grid: grid))))
    }

    func audioFeatures(_ waves: MLXArray, mask: MLXArray?) throws -> MLXArray {
        guard let audio else {
            throw VLMError.processing(
                "Muse Spark requires an audio-inclusive checkpoint for waveform input.")
        }
        return try normalizeFeatures(audio(waves, mask: mask))
    }

    func inputEmbeddings(_ input: LMInput) throws -> MLXArray {
        let tokens = input.text.tokens
        let ids = tokens.ndim == 1 ? tokens.expandedDimensions(axis: 0) : tokens
        guard ids.ndim == 2, ids.dim(0) > 0, ids.dim(1) > 0 else {
            throw VLMError.processing("Muse Spark requires a nonempty token prompt.")
        }
        let initial = language.model.embed(ids)
        guard input.image != nil || input.video != nil || input.audio != nil else { return initial }
        let flat = ids.asArray(Int32.self)
        let width = config.text.hiddenSize
        let embeddings = initial.reshaped(-1, width)
        func splice(_ features: MLXArray, token: Int) throws {
            let indices = flat.enumerated().compactMap { index, id in
                id == token ? Int32(index) : nil
            }
            guard indices.count == features.dim(0) else {
                throw VLMError.processing(
                    "Muse Spark has \(indices.count) placeholders for \(features.dim(0)) features (token \(token))."
                )
            }
            embeddings[MLXArray(indices), 0...] = features.asType(initial.dtype)
        }
        if let image = input.image {
            try splice(
                visionFeatures(image.pixels, grid: image.frames ?? []), token: config.imageToken)
        }
        if let video = input.video {
            try splice(
                visionFeatures(video.pixels, grid: video.frames ?? []), token: config.videoToken)
        }
        if let audio = input.audio {
            try splice(audioFeatures(audio.features, mask: audio.mask), token: config.audioToken)
        }
        return embeddings.reshaped(initial.shape)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        let embeddings = try inputEmbeddings(input)
        let length = embeddings.dim(1)
        let completed = try prefill.forEachChunk(total: length) { range in
            _ = language(embeddings[0..., range, 0...], cache: cache, lastOnly: true)
            asyncEval(cache)
        }
        if completed > 0 { eval(cache) }
        let logits = language(embeddings[0..., completed..., 0...], cache: cache, lastOnly: true)
        prefill.progress?(length, length)
        return .logits(.init(logits: logits))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let ids = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs
        return language(language.model.embed(ids), cache: cache)
    }

    public func quantizationPredicate(path: String, module: Module) -> QuantizationDecision {
        let pieces = path.split(separator: ".")
        guard pieces.count == 7, pieces[0] == "language_model", pieces[1] == "model",
            pieces[2] == "layers", let layer = Int(pieces[3]),
            layer > 0, layer < config.text.layers - 1, pieces[4] == "mlp", pieces[5] == "experts",
            ["gate_proj", "up_proj", "down_proj"].contains(pieces[6])
        else { return .skip }
        return .useDefault
    }

    private static func canonicalPath(_ path: String) -> String {
        if path.hasPrefix("model.language_model.model.") {
            return String(path.dropFirst("model.".count))
        }
        if path.hasPrefix("model.language_model.") {
            return "language_model.model." + path.dropFirst("model.language_model.".count)
        }
        if path.hasPrefix("lm_head.") || path == "lm_head" {
            return "language_model." + path
        }
        if ["model.layers.", "model.embed_tokens", "model.norm"].contains(where: path.hasPrefix) {
            return "language_model." + path
        }
        return path.hasPrefix("model.") ? String(path.dropFirst("model.".count)) : path
    }

    func checkpointQuantization(_ raw: BaseConfiguration.PerLayerQuantization?) throws
        -> BaseConfiguration.PerLayerQuantization?
    {
        guard let raw else { return nil }
        var options: [String: BaseConfiguration.QuantizationOption] = [:]
        for (key, value) in raw.perLayerQuantization {
            let path = Self.canonicalPath(key)
            guard options[path] == nil else {
                throw ModelExecutionError.invalidOptions(
                    "Duplicate Muse Spark quantization path: \(path)")
            }
            options[path] = value
        }
        for index in config.text.layerTypes.indices {
            let layer = "language_model.model.layers.\(index)."
            let experts = layer + "mlp.experts."
            if let fused = options[experts + "gate_up_proj"] {
                for name in ["gate_proj", "up_proj"] where options[experts + name] == nil {
                    options[experts + name] = fused
                }
            }
            let attn = layer + "self_attn."
            if options[attn + "qkvg_proj"] == nil,
                let source = options[attn + "q_proj"]
            {
                let decision:
                    (BaseConfiguration.QuantizationOption?) -> BaseConfiguration.Quantization? = {
                        switch $0 {
                        case .quantize(let value): value
                        case .skip: nil
                        case nil: raw.quantization
                        }
                    }
                let modes = ["q_proj", "k_proj", "v_proj", "gate_proj"].map {
                    decision(options[attn + $0])
                }
                if modes.dropFirst().allSatisfy({ $0 == modes[0] }) {
                    options[attn + "qkvg_proj"] = source
                } else {
                    throw ModelExecutionError.invalidOptions(
                        "Muse Spark fused attention requires matching projection quantization")
                }
            }
        }
        return .init(quantization: raw.quantization, perLayerQuantization: options)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        result.reserveCapacity(weights.count)
        for (key, value) in weights where !key.hasSuffix("rotary_emb.inv_freq") {
            let path = Self.canonicalPath(key)
            if path.hasSuffix(".global_scale") {
                let name = String(path.dropLast(".global_scale".count)) + ".weight_scale_2"
                result[name] = value.asType(.float32).reshaped(-1) / (6 * 448)
            } else if path.hasPrefix("audio_tower."), path.hasSuffix(".conv.weight"),
                value.ndim == 3, value.dim(-1) != 1
            {
                result[path] = value.swappedAxes(1, 2)
            } else {
                result[path] = value
            }
        }

        let norms = [
            "input_layernorm.weight", "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight", "mlp.pre_expert_norm.weight",
            "mlp.experts.post_expert_norm.weight",
        ]
        for index in config.text.layerTypes.indices {
            let layer = "language_model.model.layers.\(index)."
            let gates = [
                ("post_attention_residual_gate.gate", "add_attn"),
                ("post_feedforward_residual_gate.gate", "add_ffn"),
            ]
            let hasSourceGates = gates.contains { result[layer + $0.0] != nil }
            for (name, target) in gates {
                guard let gate = result.removeValue(forKey: layer + name) else { continue }
                let (alpha, beta) = MuseSparkLanguage.gateCoefficients(
                    gate, temperature: config.text.gateTemperature)
                result[layer + target + "_alpha"] = alpha
                result[layer + target + "_beta"] = beta
            }
            if hasSourceGates {
                for suffix in norms {
                    let key = layer + suffix
                    if let value = result[key] {
                        result[key] = (1 + value.asType(.float32)).asType(.bfloat16)
                    }
                }
            }
            Self.sanitizeExperts(layer + "mlp.experts.", weights: &result)
            Self.fuseAttention(
                layer + "self_attn.", projections: ["q", "k", "v", "gate"],
                target: "qkvg_proj", head: config.text.headDimension, weights: &result)
        }
        if let vision = config.vision?.geometry {
            for index in 0 ..< vision.hiddenLayers {
                Self.fuseAttention(
                    "vision_tower.layers.\(index).attn.", projections: ["q", "k", "v"],
                    target: "qkv_proj", head: vision.hiddenSize / vision.attentionHeads,
                    weights: &result)
            }
        }
        return result
    }

    private static func sanitizeExperts(_ experts: String, weights: inout [String: MLXArray]) {
        for (source, suffix) in [
            ("gate_up_proj", "weight"), ("gate_up_proj_weight_scale", "scales"),
            ("gate_up_proj_weight_scale_2", "weight_scale_2"),
        ] {
            guard let value = weights.removeValue(forKey: experts + source) else { continue }
            let parts: [MLXArray]
            if suffix == "weight_scale_2" {
                parts =
                    value.ndim == 2 && value.dim(1) == 2
                    ? [value[0..., 0], value[0..., 1]]
                    : [value.reshaped(-1), value.reshaped(-1)]
            } else {
                parts = split(value, parts: 2, axis: 1)
            }
            for (name, part) in zip(["gate_proj", "up_proj"], parts) {
                weights[experts + name + "." + suffix] =
                    suffix == "weight" && part.dtype == .uint8
                    ? part.view(dtype: .uint32) : part
            }
        }
        for (source, suffix) in [
            ("down_proj", "weight"), ("down_proj_weight_scale", "scales"),
            ("down_proj_weight_scale_2", "weight_scale_2"),
        ] {
            guard let value = weights.removeValue(forKey: experts + source) else { continue }
            weights[experts + "down_proj." + suffix] =
                if suffix == "weight" && value.dtype == .uint8 {
                    value.view(dtype: .uint32)
                } else if suffix == "weight_scale_2" { value.reshaped(-1) } else { value }
        }
        weights.removeValue(forKey: experts + "gate_up_proj_input_scale")
        weights.removeValue(forKey: experts + "down_proj_input_scale")
    }

    private static func fuseAttention(
        _ prefix: String, projections: [String], target: String, head: Int,
        weights: inout [String: MLXArray]
    ) {
        for suffix in ["weight", "bias", "scales", "biases"] {
            let names = projections.map { prefix + $0 + "_proj." + suffix }
            let parts = names.compactMap { weights[$0] }
            guard parts.count == names.count else { continue }
            func interleave(_ value: MLXArray) -> MLXArray {
                let shape = [-1, 2, head / 2] + value.shape.dropFirst()
                return value.reshaped(shape).swappedAxes(1, 2).reshaped(value.shape)
            }
            let ordered = [interleave(parts[0]), interleave(parts[1])] + parts.dropFirst(2)
            weights[prefix + target + "." + suffix] = concatenated(ordered, axis: 0)
            for name in names { weights.removeValue(forKey: name) }
        }
    }
}
