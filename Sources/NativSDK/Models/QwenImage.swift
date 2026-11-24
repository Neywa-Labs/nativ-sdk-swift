import Foundation
import MLX

final class QwenImagePipeline: PromptImageGenerationModel {
    let supportsImageEditing = true
    let directory: URL
    let latentChannels: Int
    let latentMean, latentStd: MLXArray
    let scheduler: DiffusionConfiguration
    private var conditioner: DiffusionVisionConditioner?
    private var transformer: QwenImageTransformer?
    private var vae: QwenImageVAE?
    init(directory: URL) throws {
        self.directory = directory
        let c = try DiffusionConfiguration(contentsOf: directory.appending(path: "vae/config.json"))
        latentChannels = c.int("z_dim", 64)
        guard let means = c.values["latents_mean"] as? [NSNumber],
            let stds = c.values["latents_std"] as? [NSNumber],
            means.count == latentChannels, stds.count == latentChannels,
            stds.allSatisfy({ $0.floatValue > 0 && $0.floatValue.isFinite })
        else {
            throw ModelExecutionError.invalidOptions(
                "Qwen-Image VAE requires per-channel latent means and standard deviations")
        }
        latentMean = MLXArray(means.map(\.floatValue)).reshaped([1, latentChannels, 1, 1])
        latentStd = MLXArray(stds.map(\.floatValue)).reshaped([1, latentChannels, 1, 1])
        scheduler =
            (try? .init(contentsOf: directory.appending(path: "scheduler/scheduler_config.json")))
            ?? .init()
    }
    private func loadComponents() throws {
        conditioner = nil
        if transformer == nil {
            let model = try QwenImageTransformer(
                .init(contentsOf: directory.appending(path: "transformer/config.json")))
            var checkpoint = try DiffusionCheckpoint(
                directory: directory.appending(path: "transformer"))
            checkpoint.weights = Dictionary(
                uniqueKeysWithValues: checkpoint.weights.map { key, value in
                    (
                        key.replacingOccurrences(
                            of: "time_text_embed.timestep_embedder.", with: "time_text_embed."
                        ).replacingOccurrences(of: "modulation.1.", with: "modulation.0."), value
                    )
                })
            try checkpoint.apply(to: model)
            transformer = model
        }
        if vae == nil {
            let config = try DiffusionConfiguration(
                contentsOf: directory.appending(path: "vae/config.json"))
            let model = try QwenImageVAE(config)
            var checkpoint = try DiffusionCheckpoint(directory: directory.appending(path: "vae"))
            checkpoint.weights = QwenImageVAE.sanitize(
                checkpoint.weights, sourceLayout: !config.bool("mlx_format", false))
            try checkpoint.apply(to: model)
            vae = model
        }
    }
    func generateImage(
        prompt: String, referenceImages: [MLXArray], tokenizer: any Tokenizer,
        parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        guard parameters.referenceResolution >= 256 else {
            throw ModelExecutionError.invalidOptions(
                "Qwen-Image reference resolution must be at least 256")
        }
        let resolution = Float(parameters.referenceResolution)
        let referenceImages = try referenceImages.map { image in
            let ratio = Float(image.dim(3)) / Float(image.dim(2))
            let width = max(32, Int((resolution * sqrt(ratio) / 32).rounded(.toNearestOrEven)) * 32)
            let height = max(
                32, Int((resolution / sqrt(ratio) / 32).rounded(.toNearestOrEven)) * 32)
            return try diffusionResizeRGB(image, height: height, width: width, preserveAlpha: true)
        }
        if conditioner == nil {
            transformer = nil
            vae = nil
            conditioner = try DiffusionVisionConditioner(
                directory: directory.appending(path: "text_encoder"),
                processorDirectory: directory.appending(path: "processor"))
        }
        let (features, slots) = try conditioner!.encode(
            prompt, references: referenceImages, tokenizer: tokenizer)
        let negative =
            (parameters.guidance ?? 1) > 1
            ? try conditioner!.encode(
                parameters.negativePrompt ?? " ", references: referenceImages, tokenizer: tokenizer)
            : nil
        return try sample(
            features, negative: negative?.0, references: referenceImages, slots: slots,
            negativeSlots: negative?.1, parameters: parameters)
    }
    func generate(
        promptEmbeddings: MLXArray, textIDs: MLXArray, parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        try sample(
            promptEmbeddings, negative: nil, references: [], slots: nil, negativeSlots: nil,
            parameters: parameters)
    }
    private func sample(
        _ features: MLXArray, negative: MLXArray?, references: [MLXArray], slots: [Bool]?,
        negativeSlots: [Bool]?, parameters p: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        let steps = p.steps ?? (references.isEmpty ? 30 : 40)
        let guidance = p.guidance ?? 1
        guard steps > 0, p.width >= 16, p.height >= 16, p.width % 16 == 0, p.height % 16 == 0,
            guidance.isFinite, guidance >= 0
        else {
            throw ModelExecutionError.invalidOptions(
                "Invalid Qwen-Image dimensions, steps or guidance")
        }
        if guidance > 1 && negative == nil {
            throw ModelExecutionError.invalidOptions(
                "Use prompt-based generation for classifier-free guidance")
        }
        try loadComponents()
        let model = transformer!
        let vae = vae!
        let h = p.height / 16
        let w = p.width / 16
        let seed = p.seed ?? UInt64.random(in: 0 ... UInt64.max)
        var referenceLatents = [MLXArray]()
        var shapes = [(Int, Int)]()
        for reference in references {
            guard reference.ndim == 4, reference.dim(0) == 1, [3, 4].contains(reference.dim(1)),
                reference.dim(2) % 32 == 0, reference.dim(3) % 32 == 0
            else {
                throw ModelExecutionError.invalidOptions(
                    "Qwen-Image references require RGB/RGBA dimensions divisible by 32")
            }
            let rgba =
                reference.dim(1) == 4
                ? reference
                : concatenated(
                    [reference, MLXArray.ones([1, 1, reference.dim(2), reference.dim(3)])], axis: 1)
            let encoded = vae.encode(rgba.asType(.bfloat16))
            let normalized =
                (encoded - latentMean.asType(encoded.dtype)) / latentStd.asType(encoded.dtype)
            shapes.append((encoded.dim(2), encoded.dim(3)))
            referenceLatents.append(
                normalized.reshaped([1, latentChannels, -1]).transposed(0, 2, 1))
        }
        var latents = MLXRandom.normal([1, latentChannels, h, w], key: MLXRandom.key(seed)).asType(
            .bfloat16
        ).reshaped([1, latentChannels, h * w]).transposed(0, 2, 1)
        let sigmas = try Self.sigmas(steps: steps, tokens: h * w, config: scheduler)
        let cache =
            !references.isEmpty && p.useReferenceCache && steps > 1 && model.causal
            ? QwenImageReferenceCache() : nil
        let negativeCache = cache != nil && negative != nil ? QwenImageReferenceCache() : nil
        for i in 0 ..< steps {
            try Task.checkCancellation()
            let t =
                (sigmas[i] * Float(scheduler.int("num_train_timesteps", 1000))).asType(
                    latents.dtype
                ).reshaped([1]) / 1000
            let input =
                references.isEmpty || (cache != nil && i > 0)
                ? latents : concatenated(referenceLatents + [latents], axis: 1)
            var prediction = try model(
                input, text: features.asType(.bfloat16), timestep: t, height: h, width: w,
                referenceShapes: shapes, imageSlots: slots, cache: cache)
            if let negative {
                let uncond = try model(
                    input, text: negative.asType(.bfloat16), timestep: t, height: h, width: w,
                    referenceShapes: shapes, imageSlots: negativeSlots, cache: negativeCache)
                prediction = uncond + guidance * (prediction - uncond)
            }
            let dt = (sigmas[i + 1] - sigmas[i]).asType(prediction.dtype)
            latents = (latents.asType(.float32) + dt * prediction).asType(prediction.dtype)
            eval(latents)
            if i == 0 {
                if let cache { eval(cache.keys + cache.values) }
                if let negativeCache { eval(negativeCache.keys + negativeCache.values) }
            }
        }
        cache?.keys = []
        cache?.values = []
        negativeCache?.keys = []
        negativeCache?.values = []
        let z =
            latents.transposed(0, 2, 1).reshaped([1, latentChannels, h, w]).asType(.float32)
            * latentStd + latentMean
        var pixels = clip((vae.decode(z) + 1) / 2, min: 0, max: 1).transposed(0, 2, 3, 1)[0]
        if references.isEmpty { pixels = pixels[0..., 0..., ..<3] }
        eval(pixels)
        return .init(
            image: round(pixels * 255).asType(.uint8), seed: seed, width: p.width, height: p.height,
            steps: steps, guidance: guidance)
    }
    static func sigmas(steps: Int, tokens: Int, config c: DiffusionConfiguration) throws -> MLXArray
    {
        let baseLength = c.int("base_image_seq_len", 256)
        let maxLength = c.int("max_image_seq_len", 8192)
        guard steps > 0, maxLength > baseLength else {
            throw ModelExecutionError.invalidOptions("Invalid Qwen-Image schedule")
        }
        let base = c.float("base_shift", 0.5)
        let maximum = c.float("max_shift", 0.9)
        let mu =
            base + Float(tokens - baseLength) * (maximum - base) / Float(maxLength - baseLength)
        let unshifted = linspace(1, 1 / Float(steps), count: steps)
        let exponential = exp(MLXArray(mu))
        var sigmas = exponential / (exponential + (1 / unshifted - 1))
        let terminal = c.values["shift_terminal"] is NSNull ? 0 : c.float("shift_terminal", 0.02)
        if terminal != 0 && steps > 1 {
            sigmas = 1 - (1 - sigmas) / ((1 - sigmas[-1]) / (1 - terminal))
        }
        return concatenated([sigmas, MLXArray.zeros([1])])
    }
}
