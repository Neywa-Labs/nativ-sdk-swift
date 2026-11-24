import Foundation
import MLX
import MLXNN

public enum BonsaiError: LocalizedError {
    case missingWeights(URL)
    case invalidStepCount(Int)
    case emptyPrompt
    case invalidQuantizationConfig(URL)

    public var errorDescription: String? {
        switch self {
        case .missingWeights(let url):
            "Missing Bonsai weights under \(url.path)."
        case .invalidStepCount(let steps):
            "Bonsai image generation requires at least one denoising step; got \(steps)."
        case .emptyPrompt:
            "Bonsai prompt must not be empty."
        case .invalidQuantizationConfig(let url):
            "Invalid Bonsai quantization config at \(url.path)."
        }
    }
}

public enum BonsaiDimensions {
    public static func validate(width: Int, height: Int) throws {
        try Flux2Dimensions.validate(width: width, height: height)
    }
}

public final class BonsaiImagePipeline: ImageGenerationModel {

    public let textEncoder: Flux2TextEncoder
    public let transformer: Flux2Transformer
    public let vae: Flux2VAE

    public init(
        textEncoder: Flux2TextEncoder,
        transformer: Flux2Transformer,
        vae: Flux2VAE,
    ) {
        self.textEncoder = textEncoder
        self.transformer = transformer
        self.vae = vae
    }

    public func encodePrompt(
        _ prompt: String,
        tokenizer: any Tokenizer,
        maxLength: Int = 512,
        bucketed: Bool = true,
        hiddenStateLayers: [Int] = [9, 18, 27],
    )
        throws -> (embeddings: MLXArray, textIDs: MLXArray)
    {
        let tokens = try Flux2PromptTokenizer.tokenize(
            prompt: prompt,
            tokenizer: tokenizer,
            maxLength: maxLength,
            bucketed: bucketed,
        )
        let embeddings = try textEncoder.promptEmbeddings(
            inputIDs: tokens.inputIDs,
            attentionMask: tokens.attentionMask,
            hiddenStateLayers: hiddenStateLayers,
        )
        return (embeddings, Flux2PromptIDs.prepareTextIDs(for: embeddings))
    }

    public func generate(
        prompt: String,
        tokenizer: any Tokenizer,
        parameters: ImageGenerationParameters = ImageGenerationParameters(),
        maxLength: Int = 512,
        bucketed: Bool = true,
        hiddenStateLayers: [Int] = [9, 18, 27],
    )
        throws -> ImageGenerationResult
    {
        guard !prompt.isEmpty else {
            throw BonsaiError.emptyPrompt
        }
        let promptInputs = try encodePrompt(
            prompt,
            tokenizer: tokenizer,
            maxLength: maxLength,
            bucketed: bucketed,
            hiddenStateLayers: hiddenStateLayers,
        )
        var negativeInputs: (embeddings: MLXArray, textIDs: MLXArray)?
        if (parameters.guidance ?? 1) > 1.0 {
            negativeInputs = try encodePrompt(
                parameters.negativePrompt ?? " ",
                tokenizer: tokenizer,
                maxLength: maxLength,
                bucketed: bucketed,
                hiddenStateLayers: hiddenStateLayers,
            )
        }
        return try generate(
            promptEmbeddings: promptInputs.embeddings,
            textIDs: promptInputs.textIDs,
            negativePromptEmbeddings: negativeInputs?.embeddings,
            negativeTextIDs: negativeInputs?.textIDs,
            parameters: parameters,
        )
    }

    public func generate(
        promptEmbeddings: MLXArray,
        textIDs: MLXArray,
        parameters: ImageGenerationParameters,
    )
        throws -> ImageGenerationResult
    {
        try generate(
            promptEmbeddings: promptEmbeddings,
            textIDs: textIDs,
            negativePromptEmbeddings: nil,
            negativeTextIDs: nil,
            parameters: parameters,
        )
    }

    public func generate(
        promptEmbeddings: MLXArray,
        textIDs: MLXArray,
        negativePromptEmbeddings: MLXArray? = nil,
        negativeTextIDs: MLXArray? = nil,
        parameters: ImageGenerationParameters = ImageGenerationParameters(),
    )
        throws -> ImageGenerationResult
    {
        try BonsaiDimensions.validate(width: parameters.width, height: parameters.height)
        guard (parameters.steps ?? 4) >= 1 else {
            throw BonsaiError.invalidStepCount((parameters.steps ?? 4))
        }

        let seed = parameters.seed ?? 0
        let prepared = Flux2Latents.preparePackedLatents(
            seed: seed,
            height: parameters.height,
            width: parameters.width,
            numLatentChannels: vae.config.latentChannels,
        )
        var latents = prepared.latents
        let scheduler = Flux2FlowMatchEulerDiscreteScheduler(
            imageSequenceLength: (parameters.height / 16) * (parameters.width / 16),
            inferenceSteps: (parameters.steps ?? 4),
        )

        for index in 0 ..< (parameters.steps ?? 4) {
            try Task.checkCancellation()
            var noise = predict(
                latents: latents,
                latentIDs: prepared.ids,
                promptEmbeddings: promptEmbeddings,
                textIDs: textIDs,
                timestep: scheduler.timesteps[index],
            )
            if (parameters.guidance ?? 1) > 1.0,
                let negativePromptEmbeddings,
                let negativeTextIDs
            {
                let negativeNoise = predict(
                    latents: latents,
                    latentIDs: prepared.ids,
                    promptEmbeddings: negativePromptEmbeddings,
                    textIDs: negativeTextIDs,
                    timestep: scheduler.timesteps[index],
                )
                noise = negativeNoise + (parameters.guidance ?? 1) * (noise - negativeNoise)
            }
            latents = scheduler.step(noise: noise, stepIndex: index, latents: latents)
            eval(latents)
        }

        var packed = latents.reshaped(
            latents.dim(0),
            prepared.latentHeight,
            prepared.latentWidth,
            latents.dim(-1),
        )
        packed = packed.transposed(0, 3, 1, 2)
        let decoded = vae.decodePackedLatents(packed)
        let image = Flux2ImagePipeline.imageArray(fromDecodedLatents: decoded)
        return ImageGenerationResult(
            image: image,
            seed: seed,
            width: parameters.width,
            height: parameters.height,
            steps: (parameters.steps ?? 4),
            guidance: (parameters.guidance ?? 1),
            metadata: [
                "architecture": "flux2-klein-packed",
                "transformer_precision": "2bit",
                "text_encoder_precision": "4bit",
                "vae_variant": "flux2",
            ],
        )
    }

    public func predict(
        latents: MLXArray,
        latentIDs: MLXArray,
        promptEmbeddings: MLXArray,
        textIDs: MLXArray,
        timestep: MLXArray,
    )
        -> MLXArray
    {
        transformer(
            hiddenStates: latents,
            encoderHiddenStates: promptEmbeddings,
            timestep: timestep,
            imageIDs: latentIDs,
            textIDs: textIDs,
            guidance: nil,
        )
    }
}

public enum BonsaiWeightLoader {
    public static func loadPipeline(
        from modelDirectory: URL,
    )
        throws -> BonsaiImagePipeline
    {
        let textEncoder = Flux2TextEncoder(
            try .read(from: modelDirectory, path: "text_encoder-mlx-4bit/config.json"))
        let transformer = Flux2Transformer(
            try .read(from: modelDirectory, path: "transformer-packed-mflux/config.json"))
        let vae = Flux2VAE(try .read(from: modelDirectory, includeEncoder: false))
        try loadTextEncoderWeights(
            from: modelDirectory.appending(path: "text_encoder-mlx-4bit"),
            into: textEncoder,
        )
        try loadTransformerWeights(
            from: modelDirectory.appending(path: "transformer-packed-mflux"),
            into: transformer,
        )
        try Flux2WeightLoader.loadVAEWeights(
            from: modelDirectory.appending(path: "vae"),
            into: vae,
            includeEncoder: false,
        )
        return BonsaiImagePipeline(
            textEncoder: textEncoder,
            transformer: transformer,
            vae: vae,
        )
    }

    public static func loadTextEncoderWeights(from directory: URL, into model: Flux2TextEncoder)
        throws
    {
        quantize(model: model, groupSize: 64, bits: 4)
        let raw = try loadSafetensors(from: directory)
        let weights = sanitizeTextEncoderWeights(raw)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
    }

    public static func loadTransformerWeights(from directory: URL, into model: Flux2Transformer)
        throws
    {
        let quantizedModules = try readQuantizedTransformerModules(from: directory)
        quantize(model: model, groupSize: 128, bits: 2) { path, module in
            module is Linear && quantizedModules.contains(path)
        }
        let raw = try loadSafetensors(from: directory)
        let weights = sanitizeTransformerWeights(raw)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: [.all])
        eval(model)
    }

    public static func sanitizeTextEncoderWeights(_ weights: [String: MLXArray])
        -> [String: MLXArray]
    {
        sanitizeQuantizedWeights(
            weights,
            strippingPrefix: "model.",
            keeping: { stripped in
                stripped.hasPrefix("embed_tokens.")
                    || stripped.hasPrefix("layers.")
                    || stripped.hasPrefix("norm.")
            },
        )
    }

    public static func sanitizeTransformerWeights(_ weights: [String: MLXArray])
        -> [String: MLXArray]
    {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            var mapped = key.replacingOccurrences(
                of: "time_guidance_embed.timestep_embedder.",
                with: "time_guidance_embed.",
            )
            mapped = mapped.replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
            sanitized[mapped] = castQuantizedTensor(key: mapped, value: value)
        }
        return sanitized
    }

    public static func sanitizeQuantizedWeights(
        _ weights: [String: MLXArray],
        strippingPrefix prefix: String,
        keeping keep: (String) -> Bool,
    )
        -> [String: MLXArray]
    {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            guard key.hasPrefix(prefix) else {
                continue
            }
            let stripped = String(key.dropFirst(prefix.count))
            guard keep(stripped) else {
                continue
            }
            sanitized[stripped] = castQuantizedTensor(key: stripped, value: value)
        }
        return sanitized
    }

    public static func castQuantizedTensor(key: String, value: MLXArray) -> MLXArray {
        if value.dtype == .uint32 || value.dtype == .int32 {
            return value
        }
        if key.hasSuffix(".weight"), value.dtype == .uint8 {
            return value
        }
        return value.asType(Flux2Precision.dtype)
    }

    private static func readQuantizedTransformerModules(from directory: URL) throws -> Set<String> {
        let url = directory.appending(path: "quantization_config.json")
        guard let data = try? Data(contentsOf: url),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let modules = object["quantized_modules"] as? [String]
        else {
            throw BonsaiError.invalidQuantizationConfig(url)
        }
        return Set(
            modules.map { module in
                module.replacingOccurrences(of: ".to_out.0", with: ".to_out")
            })
    }

    private static func loadSafetensors(from directory: URL) throws -> [String: MLXArray] {
        guard
            let enumerator = FileManager.default.enumerator(
                at: directory,
                includingPropertiesForKeys: nil,
            )
        else {
            throw BonsaiError.missingWeights(directory)
        }
        var weights = [String: MLXArray]()
        for case let url as URL in enumerator {
            guard url.pathExtension == "safetensors",
                !url.lastPathComponent.hasPrefix("._")
            else {
                continue
            }
            let (arrays, _) = try loadArraysAndMetadata(url: url)
            for (key, value) in arrays {
                weights[key] = value
            }
        }
        if weights.isEmpty {
            throw BonsaiError.missingWeights(directory)
        }
        return weights
    }
}
