import Foundation
import MLX
import MLXNN

enum Flux3ImageSampling {
    static func schedule(steps: Int, alpha: Float) -> MLXArray {
        let time = linspace(Float(1), Float(0), count: steps + 1)
        return alpha * time / (1 + (alpha - 1) * time)
    }

    static func imageIDs(height: Int, width: Int, reference: Int? = nil, scale: Float) -> MLXArray {
        let grid = Flux2Latents.prepareGridIDs(
            .zeros([1, 1, height / 16, width / 16]), tCoordinate: 0)
        let ids = grid.asType(.float32) * MLXArray([Float(1), scale, scale, 1])
        guard let reference else { return ids }
        return ids + MLXArray([Float(0), 0, 0, -1 - Float(reference)])
    }

    static func reference(_ image: MLXArray, multi: Bool) -> MLXArray {
        let bytes = DiffusionRGBResize.rgbBytes(image)
        let (height, width) = (bytes.dim(1), bytes.dim(2))
        let area = Double(height) * Double(width)
        if multi {
            let scale = min(1, sqrt(1_048_576 / area))
            let w = max(16, Int(Double(width) * scale))
            let h = max(16, Int(Double(height) * scale))
            let resized = DiffusionRGBResize.resize(bytes, height: h, width: w, lanczos: true)
            let top = h % 16 / 2
            let left = w % 16 / 2
            return resized[0..., top ..< top + h / 16 * 16, left ..< left + w / 16 * 16, 0...]
        }

        var scale = min(1, sqrt(4_194_304 / area), 4096 / Double(max(height, width)))
        func fit(_ side: Int) -> Int {
            max(1, Int((Double(side) * scale).rounded(.toNearestOrEven)))
        }
        func aligned(_ size: Int) -> Int {
            min(4096, max(64, (size + 15) / 16 * 16))
        }
        var h = fit(height)
        var w = fit(width)
        while aligned(h) * aligned(w) > 4_194_304 {
            scale *= 0.995
            h = fit(height)
            w = fit(width)
        }
        let resized = DiffusionRGBResize.resize(bytes, height: h, width: w, lanczos: true)
        let symmetric = min(h, w) <= 16
        func reflectedIndices(_ size: Int, padded: Int) -> MLXArray {
            let pad = (padded - size) / 2
            let period = 2 * (symmetric ? size : size - 1)
            return MLXArray(
                (0 ..< padded).map { index -> Int32 in
                    let index = ((index - pad) % period + period) % period
                    let mirrored = index < size ? index : period - index - (symmetric ? 1 : 0)
                    return Int32(mirrored)
                })
        }
        if h == aligned(h), w == aligned(w) { return resized }
        let rows = take(resized, reflectedIndices(h, padded: aligned(h)), axis: 1)
        return take(rows, reflectedIndices(w, padded: aligned(w)), axis: 2)
    }
}

enum Flux3TextConditioning {
    static func tokenize(
        _ prompt: String, tokenizer: any Tokenizer, limit: Int, multiple: Int, bucketed: Bool
    ) throws -> (ids: MLXArray, count: Int) {
        var ids = Array(
            try tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": prompt]], tools: nil, additionalContext: nil
            ).prefix(limit))
        let count = ids.count
        guard count > 0 else {
            throw ModelExecutionError.invalidOptions("FLUX.3 requires conditioning tokens")
        }
        if bucketed {
            let length = min(limit, (count + multiple - 1) / multiple * multiple)
            if length > count {
                let pad = tokenizer.padToken ?? "<|endoftext|>"
                guard let padID = tokenizer.convertTokenToId(pad) else {
                    throw ModelExecutionError.invalidOptions("Missing FLUX.3 text padding token")
                }
                ids.append(contentsOf: repeatElement(padID, count: length - count))
            }
        }
        return (MLXArray(ids).reshaped(1, -1), count)
    }

    static func load(directory: URL, layers: [Int], channels: Int) throws -> Qwen3VLLanguage.Model {
        let root = try DiffusionConfiguration(contentsOf: directory.appending(path: "config.json"))
        var text = root.child("text_config").values
        let depth = (text["num_hidden_layers"] as? Int) ?? 0
        let hidden = (text["hidden_size"] as? Int) ?? 0
        guard let last = layers.max(), last <= depth, hidden > 0,
            channels.isMultiple(of: layers.count), channels / layers.count == hidden
        else {
            throw ModelExecutionError.invalidOptions("Invalid FLUX.3 text feature layers")
        }
        text["num_hidden_layers"] = last
        text["use_final_norm"] = last == depth
        let config = try JSONDecoder().decode(
            Qwen3VLConfiguration.TextConfiguration.self,
            from: JSONSerialization.data(withJSONObject: text))
        let model = Qwen3VLLanguage.Model(config)
        let prefix = "model.language_model."
        var weights: [String: MLXArray] = [:]
        for (key, tensor) in try DiffusionCheckpoint(directory: directory).weights {
            guard key.hasPrefix(prefix) else { continue }
            let name = String(key.dropFirst(prefix.count))
            if name.hasPrefix("norm."), last != depth { continue }
            if name.hasPrefix("layers."),
                let layer = name.split(separator: ".").dropFirst().first, let index = Int(layer),
                index >= last
            {
                continue
            }
            weights[name] = tensor.asType(.bfloat16)
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
        return model
    }

    static func features(
        _ model: Qwen3VLLanguage.Model, ids: MLXArray, count: Int, layers: [Int]
    ) -> MLXArray {
        let sequence = arange(ids.dim(1), dtype: .int32)
        let valid = sequence .< count
        let positions = MLX.where(valid, sequence, 1).reshaped(1, -1)
        let allowed = (sequence[0..., .newAxis] .>= sequence[.newAxis, 0...]) .&& valid
        let mask = allowed.expandedDimensions(axes: [0, 1])
        var hidden = model.embedTokens(ids)
        var captured: [MLXArray] = []
        let choices = Set(layers)
        for (index, block) in model.layers.enumerated() {
            hidden = block(hidden, mask: mask, cache: nil, positionIds: positions)
            if choices.contains(index + 1) {
                let output =
                    index + 1 == model.layers.count ? model.norm?(hidden) ?? hidden : hidden
                captured.append(output)
            }
        }
        return concatenated(captured, axis: -1).asType(.bfloat16)
    }
}

enum Flux3Autoencoder {
    static func load(url: URL) throws -> Flux2VAE {
        let vae = Flux2VAE(.init(includeEncoder: true), activationDType: .float32)
        let weights = try loadArrays(url: url)
        try vae.update(
            parameters: ModuleParameters.unflattened(checkpointWeights(weights)), verify: [.all])
        eval(vae)
        return vae
    }

    static func checkpointWeights(
        _ weights: [String: MLXArray], depth: Int = 4
    ) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]
        for (key, tensor) in weights where key != "bn.num_batches_tracked" {
            var parts = key.split(separator: ".").map(String.init)
            var linear = false
            if key.hasPrefix("encoder.quant_conv.") || key.hasPrefix("decoder.post_quant_conv.") {
                parts.removeFirst()
            } else if parts.count > 2 {
                switch parts[1] {
                case "conv_in", "conv_out":
                    parts.insert("conv", at: 2)
                case "norm_out":
                    parts.replaceSubrange(1 ... 1, with: ["conv_norm_out", "norm"])
                case "mid":
                    switch parts[2] {
                    case "block_1", "block_2":
                        let index = parts[2] == "block_1" ? "0" : "1"
                        parts.replaceSubrange(1 ... 2, with: ["mid_block", "resnets", index])
                    case "attn_1" where parts.count > 3:
                        let names = [
                            "norm": "group_norm", "q": "to_q", "k": "to_k", "v": "to_v",
                            "proj_out": "to_out",
                        ]
                        let name = parts[3]
                        linear = name != "norm"
                        parts.replaceSubrange(
                            1 ... 3, with: ["mid_block", "attentions", "0", names[name] ?? name])
                    default: break
                    }
                case "down", "up":
                    guard parts.count > 3 else { break }
                    let down = parts[1] == "down"
                    parts[1] = down ? "down_blocks" : "up_blocks"
                    if !down, let index = Int(parts[2]) { parts[2] = String(depth - 1 - index) }
                    switch parts[3] {
                    case "block": parts[3] = "resnets"
                    case "downsample":
                        parts.replaceSubrange(3 ... 3, with: ["downsamplers", "0"])
                    case "upsample":
                        parts.replaceSubrange(3 ... 3, with: ["upsamplers", "0"])
                    default: break
                    }
                default: break
                }
            }
            if let shortcut = parts.firstIndex(of: "nin_shortcut") {
                parts[shortcut] = "conv_shortcut"
            }
            let value = tensor.asType(.float32)
            let converted =
                if value.ndim != 4 { value } else if linear { value[0..., 0..., 0, 0] } else {
                    value.transposed(0, 2, 3, 1)
                }
            mapped[parts.joined(separator: ".")] = converted
        }
        return mapped
    }
}

final class Flux3ImagePipeline: PromptImageGenerationModel {
    let supportsImageEditing = true
    private let directory: URL
    private let config: Flux3Configuration
    private var transformer: Flux3Transformer?
    private var autoencoder: Flux2VAE?

    private struct CachedPrompt {
        let prompt: String
        let limit: Int
        let bucketed: Bool
        let features: MLXArray
    }
    private var promptCache: [CachedPrompt] = []

    init(directory: URL, config: Flux3Configuration) throws {
        self.directory = directory
        self.config = config
        guard config.imageChannels == 128, config.referenceChannels == 128 else {
            throw ModelExecutionError.invalidOptions(
                "FLUX.3 autoencoder requires 128 packed channels")
        }
        for path in ["model.safetensors", "ae.safetensors", "text_encoder/config.json"] {
            guard FileManager.default.fileExists(atPath: directory.appending(path: path).path)
            else { throw ModelExecutionError.invalidOptions("Missing FLUX.3 component: \(path)") }
        }
    }

    func validate(_ p: ImageGenerationParameters, references: Int) throws {
        guard references <= 10, (64 ... 4096).contains(p.width), (64 ... 4096).contains(p.height),
            p.width.isMultiple(of: 16), p.height.isMultiple(of: 16)
        else {
            throw ModelExecutionError.invalidOptions(
                "FLUX.3 dimensions must be multiples of 16 in [64, 4096] with up to 10 references")
        }
        guard (p.steps ?? config.steps) > 0,
            !config.flowMap || (p.guidance ?? config.guidance) == 1,
            !config.flowMap || p.negativePrompt?.isEmpty != false
        else {
            throw ModelExecutionError.invalidOptions(
                "FLUX.3 requires positive steps. The distilled checkpoint uses guidance 1 and no negative prompt."
            )
        }
    }

    func releaseComponents() {
        transformer = nil
        autoencoder = nil
        MLX.Memory.clearCache()
    }

    private func promptFeatures(
        _ prompts: [String], tokenizer: any Tokenizer, parameters p: ImageGenerationParameters
    ) throws -> [MLXArray] {
        let limit = min(p.maxPromptLength, config.maxTokens)
        var encoder: Qwen3VLLanguage.Model?
        return try prompts.map { prompt in
            if let cached = promptCache.first(where: {
                $0.prompt == prompt && $0.limit == limit && $0.bucketed == p.bucketedPrompt
            }) {
                return cached.features
            }
            let model =
                try encoder
                ?? Flux3TextConditioning.load(
                    directory: directory.appending(path: "text_encoder"), layers: config.textLayers,
                    channels: config.textChannels)
            encoder = model
            let tokens = try Flux3TextConditioning.tokenize(
                prompt, tokenizer: tokenizer, limit: limit, multiple: config.padMultiple,
                bucketed: p.bucketedPrompt)
            let features = Flux3TextConditioning.features(
                model, ids: tokens.ids, count: tokens.count, layers: config.textLayers)
            eval(features)
            if promptCache.count >= 4 { promptCache.removeFirst() }
            promptCache.append(
                CachedPrompt(
                    prompt: prompt, limit: limit, bucketed: p.bucketedPrompt, features: features))
            return features
        }
    }

    func generateImage(
        prompt: String, referenceImages: [MLXArray], tokenizer: any Tokenizer,
        parameters p: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        try validate(p, references: referenceImages.count)
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ModelExecutionError.invalidOptions("FLUX.3 requires a nonempty prompt")
        }
        let guidance = p.guidance ?? config.guidance
        let prompts = guidance == 1 ? [prompt] : [prompt, p.negativePrompt ?? ""]
        let features = try promptFeatures(prompts, tokenizer: tokenizer, parameters: p)
        MLX.Memory.clearCache()
        let text = features.map { ($0, Flux2PromptIDs.prepareTextIDs(for: $0)) }
        return try sample(text, images: referenceImages, parameters: p)
    }

    func generate(
        promptEmbeddings: MLXArray, textIDs: MLXArray, parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        try validate(parameters, references: 0)
        guard (parameters.guidance ?? config.guidance) == 1,
            promptEmbeddings.ndim == 3, promptEmbeddings.dim(0) == 1,
            promptEmbeddings.dim(-1) == config.textChannels, promptEmbeddings.dim(1) > 0,
            textIDs.shape == [1, promptEmbeddings.dim(1), 4]
        else {
            throw ModelExecutionError.invalidOptions(
                "FLUX.3 direct embeddings require compatible caption features and guidance 1")
        }
        return try sample([(promptEmbeddings, textIDs)], images: [], parameters: parameters)
    }

    private func loadTransformer() throws -> Flux3Transformer {
        if let transformer { return transformer }
        let model = Flux3Transformer(config)
        let raw = try loadArrays(url: directory.appending(path: "model.safetensors"))
        try model.update(
            parameters: ModuleParameters.unflattened(Flux3Transformer.checkpointWeights(raw)),
            verify: [.all])
        eval(model)
        transformer = model
        return model
    }

    private func loadAutoencoder() throws -> Flux2VAE {
        if let autoencoder { return autoencoder }
        let model = try Flux3Autoencoder.load(url: directory.appending(path: "ae.safetensors"))
        autoencoder = model
        return model
    }

    private func sample(
        _ captions: [(MLXArray, MLXArray)], images: [MLXArray],
        parameters p: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        let vae = try loadAutoencoder()
        var referenceTokens: [MLXArray] = []
        var referenceIDs: [MLXArray] = []
        for (index, pixels) in images.enumerated() {
            try Task.checkCancellation()
            let image = Flux3ImageSampling.reference(pixels, multi: images.count > 1)
            let values = (image.asType(.float32) / 127.5 - 1).transposed(0, 3, 1, 2)
            let tokens = Flux2Latents.pack(try vae.encodePackedLatents(values)).asType(.bfloat16)
            eval(tokens)
            referenceTokens.append(tokens)
            referenceIDs.append(
                Flux3ImageSampling.imageIDs(
                    height: image.dim(1), width: image.dim(2), reference: index,
                    scale: config.imageRoPEScale))
        }
        let model = try loadTransformer()
        let reference =
            referenceTokens.isEmpty
            ? nil
            : model.prepareReferences(
                concatenated(referenceTokens, axis: 1), ids: concatenated(referenceIDs, axis: 1))
        let text = captions.map { model.prepareText($0.0, ids: $0.1) }
        let rotary = model.rotary(
            Flux3ImageSampling.imageIDs(
                height: p.height, width: p.width, scale: config.imageRoPEScale))
        eval(rotary.cosine, rotary.sine)
        for stream in text + (reference.map { [$0] } ?? []) {
            eval(stream.tokens, stream.rotary.cosine, stream.rotary.sine)
        }

        let steps = p.steps ?? config.steps
        let guidance = p.guidance ?? config.guidance
        let seed = p.seed ?? UInt64.random(in: UInt64.min ... UInt64.max)
        let h = p.height / 16
        let w = p.width / 16
        var latents = MLXRandom.normal(
            [1, h * w, config.imageChannels], key: MLXRandom.key(seed)
        ).asType(.bfloat16).asType(.float32)
        let times = Flux3ImageSampling.schedule(steps: steps, alpha: config.alpha).asArray(
            Float.self)
        for (time, next) in zip(times, times.dropFirst()) {
            try Task.checkCancellation()
            var velocity = try model(
                latents, rotary: rotary, text: text[0], references: reference,
                time: time, target: config.flowMap ? next : nil)
            if text.count > 1 {
                let unconditioned = try model(
                    latents, rotary: rotary, text: text[1], references: reference,
                    time: time, target: nil)
                velocity = unconditioned + guidance * (velocity - unconditioned)
            }
            latents = latents + (next - time) * velocity.asType(.float32)
            eval(latents)
        }
        let unpacked = Flux2Latents.unpack(latents, latentHeight: h, latentWidth: w)
        let decoded = vae.decodePackedLatents(unpacked).transposed(0, 2, 3, 1)
        let image = ((clip(decoded, min: -1, max: 1) + 1) * 127.5).asType(.uint8)[0]
        eval(image)
        return .init(
            image: image, seed: seed, width: p.width, height: p.height, steps: steps,
            guidance: guidance,
            metadata: ["flow_map": String(config.flowMap), "reference_count": String(images.count)])
    }
}
