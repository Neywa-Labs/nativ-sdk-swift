import Foundation
import MLX
import MLXNN

final class LLaDAImagePipeline: PromptImageGenerationModel {
    let directory: URL, schedule: LLaDAImageSchedule
    let supportsImageEditing = true
    private var encoder: LLaDAImageTextEncoder?
    private var queries: LLaDAQueryFormer?
    private var projection: LLaDATextProjection?
    private var transformer: LLaDAImageTransformer?
    private var vae: Flux2VAE?
    private var prompts: [(String?, MLXArray)] = []
    init(directory: URL) throws {
        self.directory = directory
        schedule = try .init(
            .init(contentsOf: directory.appending(path: "scheduler/scheduler_config.json")))
    }
    private func config(_ component: String) throws -> DiffusionConfiguration {
        try .init(contentsOf: directory.appending(path: "\(component)/config.json"))
    }
    private func loadEncoder() throws {
        if encoder != nil { return }
        transformer = nil
        vae = nil
        let c = try config("text_encoder")
        let model = LLaDAImageTextEncoder(
            try JSONDecoder().decode(
                LLaDA2Configuration.self, from: JSONSerialization.data(withJSONObject: c.values)))
        var checkpoint = try DiffusionCheckpoint(
            directory: directory.appending(path: "text_encoder"))
        let restored = try lladaImageRestoreFP8(checkpoint.weights, config: c)
        checkpoint.weights = Dictionary(
            uniqueKeysWithValues: restored.compactMap { key, value in
                guard key != "model.lm_head.weight" else { return nil }
                var name =
                    key.hasPrefix("model.language_model.")
                    ? String(key.dropFirst("model.language_model.".count)) : key
                if name.contains(".mlp.experts.") {
                    name =
                        name.replacingOccurrences(of: ".mlp.experts.", with: ".mlp.switch_mlp.")
                        + ".weight"
                }
                return (name, value)
            })
        try checkpoint.apply(to: model)
        encoder = model
    }
    private func encode(_ prompt: String?, tokenizer: any Tokenizer) throws -> MLXArray {
        if let index = prompts.firstIndex(where: { $0.0 == prompt }) {
            let value = prompts.remove(at: index)
            prompts.append(value)
            return value.1
        }
        try loadEncoder()
        if queries == nil {
            let model = LLaDAQueryFormer(try config("queryformer"))
            try DiffusionCheckpoint(directory: directory.appending(path: "queryformer")).apply(
                to: model)
            queries = model
        }
        if projection == nil {
            let model = LLaDATextProjection(try config("text_projection"))
            try DiffusionCheckpoint(directory: directory.appending(path: "text_projection")).apply(
                to: model)
            projection = model
        }
        let instruction =
            prompt.map {
                "Generate an image: \($0.trimmingCharacters(in: .whitespacesAndNewlines))"
            } ?? "Generate an image."
        let tokens = Array(
            tokenizer.encode(
                text: "<role>HUMAN</role> \(instruction)\n<role>ASSISTANT</role>\n<IMAGE1>",
                addSpecialTokens: true
            ).prefix(2048))
        let ids = MLXArray(tokens, [1, tokens.count])
        let embeddings = encoder!.wordEmbeddings(ids)
        let query = queries!(embeddings)
        let combined = concatenated([embeddings, query], axis: 1)
        let positions = arange(combined.dim(1))
        let mask =
            ((positions.expandedDimensions(axis: 1) .>= tokens.count)
            .|| (positions .< tokens.count)).reshaped([1, 1, combined.dim(1), combined.dim(1)])
        let output = projection!(encoder!.encode(ids, embeddings: combined, mask: mask))
        eval(output)
        prompts.append((prompt, output))
        if prompts.count > 2 { prompts.removeFirst() }
        return output
    }
    private func releaseText() {
        encoder = nil
        queries = nil
        projection = nil
    }
    private func loadSigVQ(encoder: Bool) throws -> LLaDASigVQ {
        let model = LLaDASigVQ(try config("sigvq"), includeEncoder: encoder)
        var checkpoint = try DiffusionCheckpoint(directory: directory.appending(path: "sigvq"))
        checkpoint.weights = LLaDASigVQ.sanitize(checkpoint.weights, includeEncoder: encoder)
        try checkpoint.apply(to: model)
        return model
    }
    private func loadVAE(encoder: Bool) throws -> Flux2VAE {
        if let vae, !encoder || vae.supportsEncoding { return vae }
        let model = Flux2VAE(.init(includeEncoder: encoder))
        try Flux2WeightLoader.loadVAEWeights(
            from: directory.appending(path: "vae"), into: model, includeEncoder: encoder)
        vae = model
        return model
    }
    func generateImage(
        prompt: String, referenceImages: [MLXArray], tokenizer: any Tokenizer,
        parameters p: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        guard referenceImages.count <= 1 else {
            throw ModelExecutionError.invalidOptions("LLaDA editing accepts one source image")
        }
        let caption = try encode(prompt, tokenizer: tokenizer)
        let guidance = p.guidance ?? (schedule.uniform ? 1 : 5)
        let negative = guidance > 1 ? try encode(p.negativePrompt, tokenizer: tokenizer) : nil
        var semantic: MLXArray?
        var source: MLXArray?
        if p.useImageTokenPrior {
            guard referenceImages.isEmpty else {
                throw ModelExecutionError.invalidOptions(
                    "The image token prior is used for generation; editing encodes the source image"
                )
            }
            try loadEncoder()
            let checkpoint = try DiffusionCheckpoint(
                directory: directory.appending(path: "text_encoder"))
            guard let weight = checkpoint.weights["model.lm_head.weight"] else {
                throw ModelExecutionError.invalidOptions("Missing LLaDA image token output head")
            }
            let head = Linear(weight.dim(1), weight.dim(0), bias: false)
            try head.update(
                parameters: ModuleParameters.unflattened(["weight": weight]), verify: [.all])
            let ids = try lladaImageVQTokens(
                prompt: prompt, height: p.height, width: p.width, tokenizer: tokenizer,
                encoder: encoder!, head: head,
                codebookSize: config("sigvq").int("codebook_size", 16384))
            releaseText()
            semantic = try loadSigVQ(encoder: false)(ids)
            eval(semantic!)
        }
        releaseText()
        if let reference = referenceImages.first {
            guard p.height % 32 == 0, p.width % 32 == 0 else {
                throw ModelExecutionError.invalidOptions(
                    "LLaDA editing requires dimensions divisible by 32")
            }
            let pixels = try diffusionResizeRGB(reference, height: p.height, width: p.width)
            let half = pixels.transposed(0, 2, 3, 1).reshaped([
                1, p.height / 2, 2, p.width / 2, 2, 3,
            ]).mean(axes: [2, 4]).asType(.bfloat16)
            let sigvq = try loadSigVQ(encoder: true)
            semantic = sigvq(try sigvq.encode(half))
            eval(semantic!)
            source = try loadVAE(encoder: true).encodePackedLatents(pixels.asType(.bfloat16))
            eval(source!)
        }
        return try sample(
            caption, negative: negative, semantic: semantic, source: source, parameters: p)
    }
    func generate(
        promptEmbeddings: MLXArray, textIDs: MLXArray, parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        try sample(
            promptEmbeddings, negative: nil, semantic: nil, source: nil, parameters: parameters)
    }
    private func sample(
        _ caption: MLXArray, negative: MLXArray?, semantic: MLXArray?, source: MLXArray?,
        parameters p: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        let steps = p.steps ?? (schedule.uniform ? 4 : 50)
        let guidance = p.guidance ?? (schedule.uniform ? 1 : 5)
        guard p.height >= 16, p.width >= 16, p.height % 16 == 0, p.width % 16 == 0, steps > 0,
            guidance.isFinite, guidance >= 0, guidance <= 1 || negative != nil
        else {
            throw ModelExecutionError.invalidOptions(
                "Invalid LLaDA dimensions, steps or guidance; CFG needs prompt-based generation")
        }
        if transformer == nil {
            let c = try config("transformer")
            let model = try LLaDAImageTransformer(c)
            var checkpoint = try DiffusionCheckpoint(
                directory: directory.appending(path: "transformer"))
            checkpoint.weights = LLaDAImageTransformer.sanitize(
                try lladaImageRestoreFP8(checkpoint.weights, config: c))
            try checkpoint.apply(to: model)
            transformer = model
        }
        let model = transformer!
        let dtype = model.image.weight.dtype
        let seed = p.seed ?? UInt64.random(in: 0 ... UInt64.max)
        var key = MLXRandom.key(seed)
        var latent = MLXRandom.normal([1, 128, p.height / 16, p.width / 16], key: key).asType(dtype)
            .asType(.float32)
        let sigmas = schedule.sigmas(steps)
        let cap = caption.asType(dtype)
        let neg = negative?.asType(dtype)
        let sem = semantic?.asType(dtype)
        let src = source?.asType(dtype)
        for i in 0 ..< steps {
            try Task.checkCancellation()
            let t = MLXArray([sigmas[i]]).asType(dtype)
            let input = latent.asType(dtype)
            var prediction =
                -(try model(input, timestep: t, caption: cap, semantics: sem, source: src)).asType(
                    .float32)
            if let neg {
                let unc =
                    -(try model(
                        input, timestep: t, caption: neg,
                        semantics: sem.map { $0[0..., ..<0, 0...] }, source: src)).asType(.float32)
                prediction = unc + guidance * (prediction - unc)
            }
            if p.stochasticSampling ?? schedule.stochastic {
                key = MLXRandom.split(key: key).1
                latent =
                    (1 - sigmas[i + 1]) * (latent - sigmas[i] * prediction) + sigmas[i + 1]
                    * MLXRandom.normal(latent.shape, key: key)
            } else {
                latent = latent + (sigmas[i + 1] - sigmas[i]) * prediction
            }
            eval(latent)
        }
        transformer = nil
        let decoded = try loadVAE(encoder: false).decodePackedLatents(latent.asType(.bfloat16))
        let pixels = clip(decoded.asType(.float32) / 2 + 0.5, min: 0, max: 1).transposed(
            0, 2, 3, 1)[0]
        eval(pixels)
        return .init(
            image: round(pixels * 255).asType(.uint8), seed: seed, width: p.width, height: p.height,
            steps: steps, guidance: guidance,
            metadata: ["mode": source != nil ? "editing" : semantic != nil ? "vq" : "text"])
    }
}
