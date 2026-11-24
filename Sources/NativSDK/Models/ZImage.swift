import Foundation
import MLX
import MLXNN

final class ZImageVAE: Module {
    @ModuleInfo var encoder: Flux2Encoder
    @ModuleInfo var decoder: Flux2Decoder
    let scale: Float, shift: Float, channels: Int
    init(_ c: DiffusionConfiguration) {
        channels = c.int("latent_channels", 16)
        scale = c.float("scaling_factor", 0.3611)
        shift = c.float("shift_factor", 0.1159)
        let blocks = c.ints("block_out_channels", [128, 256, 512, 512])
        let count = c.int("layers_per_block", 2)
        _encoder.wrappedValue = Flux2Encoder(
            inChannels: c.int("in_channels", 3), outChannels: channels,
            blockOutChannels: blocks, layersPerBlock: count, normGroups: 32, eps: 1e-6,
            midBlockAddAttention: true)
        _decoder.wrappedValue = Flux2Decoder(
            inChannels: channels, outChannels: c.int("out_channels", 3),
            blockOutChannels: blocks, layersPerBlock: count, normGroups: 32, eps: 1e-6,
            midBlockAddAttention: true)
        super.init()
    }
    func encode(_ pixels: MLXArray) -> MLXArray { split(encoder(pixels), parts: 2, axis: 1)[0] }
    func decode(_ latents: MLXArray) -> MLXArray { decoder(latents) }
    static func sanitize(_ weights: [String: MLXArray], sourceLayout: Bool) -> [String: MLXArray] {
        var normalized = [String: MLXArray]()
        for (key, value) in weights {
            var key = key.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
            for part in ["encoder", "decoder"] {
                for conv in ["conv_in", "conv_out"] {
                    key = key.replacingOccurrences(
                        of: "\(part).\(conv).conv2d.", with: "\(part).\(conv).")
                    key = key.replacingOccurrences(
                        of: "\(part).\(conv).conv.", with: "\(part).\(conv).")
                    key = key.replacingOccurrences(
                        of: "\(part).\(conv).", with: "\(part).\(conv).conv.")
                }
                key = key.replacingOccurrences(
                    of: "\(part).conv_norm_out.norm.", with: "\(part).conv_norm_out."
                )
                .replacingOccurrences(
                    of: "\(part).conv_norm_out.", with: "\(part).conv_norm_out.norm.")
            }
            normalized[key] = sourceLayout && value.ndim == 4 ? value.transposed(0, 2, 3, 1) : value
        }
        return normalized
    }
}

/// Flow-matching image generation and strength-controlled image-to-image sampling.
final class ZImagePipeline: PromptImageGenerationModel {
    var supportsImageEditing: Bool { true }
    let directory: URL
    let schedulerShift: Float
    private var textEncoder: Flux2TextEncoder?
    private var transformer: ZImageTransformer?
    private var vae: ZImageVAE?

    init(directory: URL) throws {
        self.directory = directory
        let scheduler = try DiffusionConfiguration(
            contentsOf: directory.appending(path: "scheduler/scheduler_config.json"))
        schedulerShift = scheduler.float("shift", 3)
    }
    private func encode(_ prompt: String, tokenizer: any Tokenizer) throws -> MLXArray {
        if textEncoder == nil {
            // The encoder and diffusion components need not occupy device memory together.
            transformer = nil
            vae = nil
            let config = try DiffusionConfiguration(
                contentsOf: directory.appending(path: "text_encoder/config.json"))
            let encoder = Flux2TextEncoder(.init(diffusion: config))
            var checkpoint = try DiffusionCheckpoint(
                directory: directory.appending(path: "text_encoder"))
            checkpoint.weights = Dictionary(
                uniqueKeysWithValues: checkpoint.weights.compactMap { key, tensor in
                    let key = key.hasPrefix("model.") ? String(key.dropFirst(6)) : key
                    guard !key.hasPrefix("lm_head."), !key.contains("rotary_emb") else {
                        return nil
                    }
                    return (key, tensor)
                })
            try checkpoint.apply(to: encoder)
            textEncoder = encoder
        }
        let tokens = try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": prompt]], tools: nil,
            additionalContext: ["enable_thinking": true, "add_generation_prompt": true])
        guard !tokens.isEmpty else {
            throw ModelExecutionError.invalidOptions("The image prompt produced no tokens")
        }
        // Causal attention makes right padding irrelevant to the retained token positions.
        let input = MLXArray(Array(tokens.prefix(512))).reshaped([1, -1])
        let (_, states) = try textEncoder!(inputIDs: input, outputHiddenStates: true)
        let features = states![states!.count - 2]
        eval(features)
        return features
    }
    private func loadDiffusionComponents() throws {
        textEncoder = nil
        if transformer == nil {
            let c = try DiffusionConfiguration(
                contentsOf: directory.appending(path: "transformer/config.json"))
            let model = ZImageTransformer(try .init(c))
            var checkpoint = try DiffusionCheckpoint(
                directory: directory.appending(path: "transformer"))
            checkpoint.weights = ZImageTransformer.sanitize(checkpoint.weights)
            try checkpoint.apply(to: model)
            transformer = model
        }
        if vae == nil {
            let model = ZImageVAE(
                try .init(contentsOf: directory.appending(path: "vae/config.json")))
            var checkpoint = try DiffusionCheckpoint(directory: directory.appending(path: "vae"))
            let sourceLayout =
                checkpoint.metadata.string("mlx_vlm_format") != "z_image"
                && checkpoint.weights["encoder.conv_in.weight"] != nil
            checkpoint.weights = ZImageVAE.sanitize(checkpoint.weights, sourceLayout: sourceLayout)
            try checkpoint.apply(to: model)
            vae = model
        }
    }
    func generate(
        promptEmbeddings: MLXArray, textIDs: MLXArray, parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        try sample(
            features: promptEmbeddings, negative: nil, reference: nil, parameters: parameters)
    }
    func generateImage(
        prompt: String, referenceImages: [MLXArray], tokenizer: any Tokenizer,
        parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        guard referenceImages.count <= 1 else {
            throw ModelExecutionError.invalidOptions("Z-Image img2img accepts one source image")
        }
        let features = try encode(prompt, tokenizer: tokenizer)
        let guidance = parameters.guidance ?? (schedulerShift <= 3 ? 0 : 4)
        let negative =
            guidance > 1 ? try encode(parameters.negativePrompt ?? "", tokenizer: tokenizer) : nil
        return try sample(
            features: features, negative: negative, reference: referenceImages.first,
            parameters: parameters)
    }
    private func sample(
        features: MLXArray, negative: MLXArray?, reference: MLXArray?,
        parameters p: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        let editing = reference != nil
        let steps = p.steps ?? (schedulerShift <= 3 ? (editing ? 8 : 9) : 50)
        let guidance = p.guidance ?? (schedulerShift <= 3 ? 0 : 4)
        guard p.width >= 16, p.height >= 16, p.width % 16 == 0, p.height % 16 == 0,
            steps >= (editing ? 1 : 2), p.cfgTruncation.isFinite,
            (0 ... 1).contains(p.cfgTruncation),
            p.strength.isFinite, p.strength > 0, p.strength <= 1
        else {
            throw ModelExecutionError.invalidOptions(
                "Z-Image requires dimensions divisible by 16, valid steps, strength in (0, 1] and CFG truncation in [0, 1]"
            )
        }
        if guidance > 1 && negative == nil {
            throw ModelExecutionError.invalidOptions(
                "Guidance requires negative prompt embeddings; use prompt-based generation")
        }
        try loadDiffusionComponents()
        let model = transformer!
        let vae = vae!
        let seed = p.seed ?? UInt64.random(in: 0 ... UInt64.max)
        let schedule = FlowEulerSchedule(steps: editing ? steps : steps - 1, shift: schedulerShift)
        let noise = MLXRandom.normal(
            [1, vae.channels, 1, p.height / 8, p.width / 8], key: MLXRandom.key(seed))
        var latents = noise
        let start =
            editing ? Int(max(Float(steps) - min(Float(steps) * p.strength, Float(steps)), 0)) : 0
        if let reference {
            let reference = try diffusionResizeRGB(reference, height: p.height, width: p.width)
            let encoded = ((vae.encode(reference.asType(.bfloat16)) - vae.shift) * vae.scale)
                .expandedDimensions(axis: 2)
            let sigma = schedule.sigmas[start].asType(encoded.dtype)
            latents = (1 - sigma) * encoded + sigma * noise
        }
        for i in start ..< (schedule.sigmas.size - 1) {
            try Task.checkCancellation()
            let timestep = (1 - schedule.sigmas[i]).reshaped([1])
            var prediction = model(latents, timestep: timestep, caption: features)
            if let negative, timestep.item(Float.self) <= p.cfgTruncation {
                let unconditional = model(latents, timestep: timestep, caption: negative)
                prediction = prediction + guidance * (prediction - unconditional)
            }
            latents = schedule.step(-prediction, at: i, latents: latents)
            eval(latents)
        }
        let decoded = vae.decode(
            (latents.squeezed(axis: 2) / vae.scale + vae.shift).asType(.bfloat16))
        let pixels = clip(decoded / 2 + 0.5, min: 0, max: 1).transposed(0, 2, 3, 1)[0]
        eval(pixels)
        return .init(
            image: round(pixels * 255).asType(.uint8), seed: seed, width: p.width, height: p.height,
            steps: steps,
            guidance: guidance,
            metadata: ["mode": editing ? "img2img" : "text2img"])
    }
}
