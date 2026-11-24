import Foundation
import MLX

final class MingImagePipeline: PromptImageGenerationModel {
    let supportsImageEditing = false
    let directory: URL
    private var encoder: MingImageTextEncoder?
    private var transformer: ZImageTransformer?
    private var vae: QwenImageVAE?
    init(directory: URL) { self.directory = directory }
    func generateImage(
        prompt: String, referenceImages: [MLXArray], tokenizer: any Tokenizer,
        parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        guard referenceImages.isEmpty else {
            throw ModelExecutionError.invalidOptions("Ming-Image supports text-to-image generation")
        }
        transformer = nil
        vae = nil
        if encoder == nil { encoder = try MingImageTextEncoder.load(directory: directory) }
        let (features, direct) = try encoder!.encode(prompt, tokenizer: tokenizer)
        encoder = nil
        return try sample(features, direct: direct, parameters: parameters)
    }
    func generate(
        promptEmbeddings: MLXArray, textIDs: MLXArray, parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        throw ModelExecutionError.invalidOptions(
            "Ming-Image requires both caption and direct language features; use prompt-based generation"
        )
    }
    private func sample(
        _ features: MLXArray, direct: MLXArray, parameters p: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        let steps = p.steps ?? 12
        let guidance = p.guidance ?? 1
        guard (256 ... 2048).contains(p.width), (256 ... 2048).contains(p.height),
            p.width % 16 == 0, p.height % 16 == 0, steps > 0, guidance == 1
        else {
            throw ModelExecutionError.invalidOptions(
                "Ming-Image requires dimensions in [256, 2048] divisible by 16, positive steps and guidance 1"
            )
        }
        if transformer == nil {
            let c = try DiffusionConfiguration(
                contentsOf: directory.appending(path: "transformer/config.json"))
            let model = ZImageTransformer(try .init(c), learnedPadding: false)
            var checkpoint = try DiffusionCheckpoint(
                directory: directory.appending(path: "transformer"))
            checkpoint.weights = ZImageTransformer.sanitize(checkpoint.weights)
            try checkpoint.apply(to: model)
            transformer = model
        }
        let vc = try DiffusionConfiguration(
            contentsOf: directory.appending(path: "vae/config.json"))
        if vae == nil {
            let model = try QwenImageVAE(vc, ming: true)
            var checkpoint = try DiffusionCheckpoint(directory: directory.appending(path: "vae"))
            checkpoint.weights = QwenImageVAE.sanitize(
                checkpoint.weights, sourceLayout: !vc.bool("mlx_format", false))
            try checkpoint.apply(to: model)
            vae = model
        }
        let model = transformer!
        let vae = vae!
        let seed = p.seed ?? UInt64.random(in: 0 ... UInt64.max)
        let h = p.height / 8
        let w = p.width / 8
        let channels = vc.int("z_dim", 16)
        let tokens = (h / model.config.patch) * (w / model.config.patch)
        let schedule = try QwenImagePipeline.sigmas(
            steps: steps, tokens: tokens,
            config: .init([
                "max_image_seq_len": tokens >= 4096 ? tokens : 4096,
                "max_shift": tokens >= 4096 ? 1.35 : 1.15, "shift_terminal": NSNull(),
            ]))
        var latents = MLXRandom.normal([1, channels, 1, h, w], key: MLXRandom.key(seed))
        for i in 0 ..< steps {
            try Task.checkCancellation()
            let t = (1 - schedule[i]).reshaped([1]).asType(.bfloat16)
            let prediction = model(
                latents.asType(.bfloat16), timestep: t, caption: features, directFeatures: direct
            ).asType(.float32)
            latents = latents - (schedule[i + 1] - schedule[i]) * prediction
            eval(latents)
        }
        let z =
            latents.squeezed(axis: 2) / vc.float("scaling_factor", 8.0064)
            + vc.float("shift_factor", 0)
        let pixels = ((vae.decode(z) + 1) / 2).transposed(0, 2, 3, 1)[0]
        eval(pixels)
        return .init(
            image: round(clip(pixels, min: 0, max: 1) * 255).asType(.uint8), seed: seed,
            width: p.width, height: p.height, steps: steps, guidance: guidance)
    }
}
