import Foundation
import MLX

public struct MOSSImageGenerationOptions: Sendable {
    /// Block indices to skip in the negative guidance pass. The default uses block 22 if present.
    public var unconditionalSkipLayers: [Int]?
    /// Block indices to skip in the additional positive guidance pass.
    public var skipLayerGuidanceLayers: [Int]
    public var skipLayerGuidanceScale: Float
    /// Inclusive start and exclusive end, measured as fractions of the denoising schedule.
    public var skipLayerGuidanceStart: Float
    public var skipLayerGuidanceStop: Float

    public init(
        unconditionalSkipLayers: [Int]? = nil,
        skipLayerGuidanceLayers: [Int] = [],
        skipLayerGuidanceScale: Float = 0,
        skipLayerGuidanceStart: Float = 0,
        skipLayerGuidanceStop: Float = 1
    ) {
        self.unconditionalSkipLayers = unconditionalSkipLayers
        self.skipLayerGuidanceLayers = skipLayerGuidanceLayers
        self.skipLayerGuidanceScale = skipLayerGuidanceScale
        self.skipLayerGuidanceStart = skipLayerGuidanceStart
        self.skipLayerGuidanceStop = skipLayerGuidanceStop
    }
}

struct MOSSImageScheduleConfiguration {
    let trainingSteps: Int
    let shift: Float
    let trainingShift: Float

    init(_ metadata: DiffusionConfiguration) throws {
        trainingSteps = metadata.int("num_train_timesteps", 1000)
        shift = metadata.float("shift", 5)
        trainingShift = metadata.float("training_shift", 1)
        guard trainingSteps > 0, shift.isFinite, shift > 0,
            trainingShift.isFinite, trainingShift > 0
        else {
            throw ModelExecutionError.invalidOptions("Invalid MOSS-Image scheduler configuration")
        }
    }
}

struct MOSSImageSchedule {
    let sigmas: [Float]
    let timesteps: [Float]

    init(steps: Int, configuration: MOSSImageScheduleConfiguration) {
        func grid(count: Int, shift: Float) -> [Float] {
            let end = Double(shift) / (Double(configuration.trainingSteps) + Double(shift) - 1)
            let step = count == 1 ? 0 : (end - 1) / Double(count - 1)
            return (0 ..< count).map { index in
                let value = Float(count > 1 && index == count - 1 ? end : 1 + Double(index) * step)
                return shift * value / (1 + (shift - 1) * value)
            }
        }

        let training = grid(count: configuration.trainingSteps, shift: configuration.trainingShift)
        let inference = grid(count: steps, shift: configuration.shift)
        var index = 0
        var snapped = [Float]()
        for sigma in inference {
            // The grids descend. Resolve ties toward the lower training sigma.
            while index + 1 < training.count,
                abs(sigma - training[index + 1]) <= abs(sigma - training[index])
            {
                index += 1
            }
            let value = training[index]
            if snapped.last != value { snapped.append(value) }
        }
        timesteps = snapped.map { $0 * Float(configuration.trainingSteps) }
        sigmas = snapped + [0]
    }
}

struct MOSSImageLayerGuidance {
    let unconditionalLayers: Set<Int>
    let positiveLayers: Set<Int>
    let scale: Float
    let start: Float
    let stop: Float

    init(_ options: MOSSImageGenerationOptions, depth: Int) throws {
        unconditionalLayers = Set(options.unconditionalSkipLayers ?? (depth > 22 ? [22] : []))
        positiveLayers = Set(options.skipLayerGuidanceLayers)
        scale = options.skipLayerGuidanceScale
        start = options.skipLayerGuidanceStart
        stop = options.skipLayerGuidanceStop
        guard unconditionalLayers.union(positiveLayers).allSatisfy({ (0 ..< depth).contains($0) }),
            scale.isFinite, scale >= 0, 0 <= start, start <= stop, stop <= 1
        else {
            throw ModelExecutionError.invalidOptions(
                "MOSS-Image requires valid block indices, a nonnegative skip scale, and 0 ≤ start ≤ stop ≤ 1"
            )
        }
    }

    func isActive(step: Int, count: Int) -> Bool {
        let progress = Float(step) / Float(count)
        return start <= progress && progress < stop
    }
}

final class MOSSImagePipeline: PromptImageGenerationModel {
    var supportsImageEditing: Bool { false }

    private let directory: URL
    private let transformerConfig: MOSSImageTransformerConfiguration
    private let schedulerConfig: MOSSImageScheduleConfiguration
    private let textConfig: Qwen35TextConfiguration
    private let convertedText: Bool
    private let vaeConfig: Flux2VAEConfiguration
    private let convertedVAE: Bool

    private struct TextConfiguration: Decodable {
        let text: Qwen35TextConfiguration
        let format: String?

        private enum CodingKeys: String, CodingKey {
            case text = "text_config"
            case format = "mlx_vlm_format"
        }
    }

    init(directory: URL) throws {
        self.directory = directory
        transformerConfig = try MOSSImageTransformerConfiguration(
            ImageCheckpointMetadata.required("transformer/config.json", in: directory))
        schedulerConfig = try MOSSImageScheduleConfiguration(
            ImageCheckpointMetadata.required("scheduler/scheduler_config.json", in: directory))
        let conditioner = try JSONDecoder().decode(
            TextConfiguration.self,
            from: Data(contentsOf: directory.appending(path: "text_encoder/config.json")))
        var text = conditioner.text
        text.linearQKUsesL2Epsilon = true
        textConfig = text
        convertedText = conditioner.format == "moss_image"
        let vaeMetadata = try ImageCheckpointMetadata.required("vae/config.json", in: directory)
        convertedVAE = vaeMetadata.string("mlx_vlm_format") == "moss_image"
        vaeConfig = try Flux2VAEConfiguration.read(from: directory)
        guard textConfig.hiddenSize == transformerConfig.textWidth,
            textConfig.fullAttentionInterval > 0,
            textConfig.hiddenLayers >= textConfig.fullAttentionInterval,
            vaeConfig.latentChannels * 4 == transformerConfig.latentChannels,
            vaeConfig.spatialScale == 8
        else {
            throw ModelExecutionError.invalidOptions("Incompatible MOSS-Image component dimensions")
        }
    }

    func generateImage(
        prompt: String, referenceImages: [MLXArray], tokenizer: any Tokenizer,
        parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        guard referenceImages.isEmpty else {
            throw ModelExecutionError.invalidOptions("MOSS-Image supports text-to-image generation")
        }
        let layerGuidance = try validate(parameters)
        let guidance = parameters.guidance ?? 3
        let prompts = guidance == 1 ? [prompt] : [prompt, parameters.negativePrompt ?? ""]
        let tokens = try prompts.map { caption in
            let ids = try tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": caption]],
                tools: nil, additionalContext: ["enable_thinking": true])
            guard !ids.isEmpty else {
                throw ModelExecutionError.invalidOptions(
                    "MOSS-Image tokenizer returned an empty prompt")
            }
            return Array(ids.prefix(parameters.maxPromptLength))
        }
        let captions = try encode(tokens)
        MLX.Memory.clearCache()
        return try sample(
            captions, parameters: parameters, guidance: guidance, layers: layerGuidance)
    }

    func generate(
        promptEmbeddings: MLXArray, textIDs: MLXArray, parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        let layerGuidance = try validate(parameters)
        let guidance = parameters.guidance ?? 3
        guard guidance == 1, promptEmbeddings.ndim == 3, promptEmbeddings.dim(0) == 1,
            promptEmbeddings.dim(1) > 0,
            promptEmbeddings.dim(2) == transformerConfig.textWidth
        else {
            throw ModelExecutionError.invalidOptions(
                "MOSS-Image direct embeddings require compatible caption features and guidance 1")
        }
        return try sample(
            [promptEmbeddings], parameters: parameters, guidance: guidance, layers: layerGuidance)
    }

    private func validate(_ parameters: ImageGenerationParameters) throws -> MOSSImageLayerGuidance
    {
        guard parameters.width > 0, parameters.height > 0,
            parameters.width.isMultiple(of: 16), parameters.height.isMultiple(of: 16)
        else {
            throw ModelExecutionError.invalidOptions(
                "MOSS-Image width and height must be positive multiples of 16")
        }
        guard (parameters.steps ?? 1) > 0, (parameters.guidance ?? 3).isFinite,
            (parameters.guidance ?? 3) >= 0, (1 ... 4096).contains(parameters.maxPromptLength)
        else {
            throw ModelExecutionError.invalidOptions("Invalid MOSS-Image sampling options")
        }
        return try MOSSImageLayerGuidance(parameters.mossImage, depth: transformerConfig.layers)
    }

    private func encode(_ tokens: [[Int]]) throws -> [MLXArray] {
        try Task.checkCancellation()
        let model = Qwen35TextModelInner(textConfig)
        var checkpoint = try DiffusionCheckpoint(
            directory: directory.appending(path: "text_encoder"))
        if !convertedText {
            checkpoint.weights = Self.sanitizeTextEncoder(checkpoint.weights)
        }
        try checkpoint.apply(to: model)
        model.train(false)
        // Qwen's recurrent layers need separate, unpadded caption forwards.
        return try tokens.map { ids in
            try Task.checkCancellation()
            let features = model(MLXArray(ids).reshaped(1, -1))
            eval(features)
            return features
        }
    }

    static func sanitizeTextEncoder(_ raw: [String: MLXArray]) -> [String: MLXArray] {
        let prefix = "language_model."
        let shiftedNorms = [
            ".input_layernorm.weight", ".post_attention_layernorm.weight",
            ".q_norm.weight", ".k_norm.weight",
        ]
        var weights = [String: MLXArray]()
        for (key, value) in raw where key.hasPrefix(prefix) {
            let name = String(key.dropFirst(prefix.count))
            if name.hasSuffix("conv1d.weight") {
                weights[name] = value.movedAxis(source: 2, destination: 1)
            } else if name == "norm.weight" || shiftedNorms.contains(where: name.hasSuffix) {
                weights[name] = value + MLXArray(1, dtype: value.dtype)
            } else {
                weights[name] = value
            }
        }
        return weights
    }

    private func denoise(
        _ captions: [MLXArray], schedule: MOSSImageSchedule, height: Int, width: Int,
        seed: UInt64, guidance: Float, layers: MOSSImageLayerGuidance
    ) throws -> MLXArray {
        try Task.checkCancellation()
        let model = MOSSImageTransformer(transformerConfig)
        var checkpoint = try DiffusionCheckpoint(
            directory: directory.appending(path: "transformer"))
        checkpoint.weights = Dictionary(
            uniqueKeysWithValues: checkpoint.weights.compactMap { key, value in
                MOSSImageTransformer.checkpointKey(key).map { ($0, value) }
            })
        try checkpoint.apply(to: model)
        let conditions = captions.map { model.condition(text: $0, height: height, width: width) }
        var latents = MLXRandom.normal(
            [1, transformerConfig.latentChannels, height, width], dtype: .float32,
            key: MLXRandom.key(seed))
        for (index, time) in schedule.timesteps.enumerated() {
            try Task.checkCancellation()
            let timestep = MLXArray(time)
            let active = layers.isActive(step: index, count: schedule.timesteps.count)
            let conditional = model(latents, timestep: timestep, condition: conditions[0]).asType(
                .float32)
            var prediction = conditional
            if conditions.count == 2 {
                let unconditional = model(
                    latents, timestep: timestep, condition: conditions[1],
                    skipLayers: active ? layers.unconditionalLayers : []
                ).asType(.float32)
                prediction = unconditional + guidance * (conditional - unconditional)
            }
            if active, layers.scale > 0, !layers.positiveLayers.isEmpty {
                let skipped = model(
                    latents, timestep: timestep, condition: conditions[0],
                    skipLayers: layers.positiveLayers
                ).asType(.float32)
                if conditions.count == 1 { prediction = skipped }
                prediction = prediction + layers.scale * (conditional - skipped)
            }
            let delta = schedule.sigmas[index + 1] - schedule.sigmas[index]
            latents = latents + delta * prediction
            eval(latents)
        }
        return latents
    }

    private func sample(
        _ captions: [MLXArray], parameters: ImageGenerationParameters,
        guidance: Float, layers: MOSSImageLayerGuidance
    ) throws -> ImageGenerationResult {
        let schedule = MOSSImageSchedule(
            steps: parameters.steps ?? 50, configuration: schedulerConfig)
        let seed = parameters.seed ?? UInt64.random(in: UInt64.min ... UInt64.max)
        let latents = try denoise(
            captions, schedule: schedule, height: parameters.height / 16,
            width: parameters.width / 16,
            seed: seed, guidance: guidance, layers: layers)
        MLX.Memory.clearCache()
        try Task.checkCancellation()
        let vae = Flux2VAE(vaeConfig, activationDType: .float32, evaluateDecoderBlocks: true)
        var checkpoint = try DiffusionCheckpoint(directory: directory.appending(path: "vae"))
        checkpoint.weights = Flux2WeightLoader.sanitizeVAEWeights(
            checkpoint.weights, includeEncoder: false, channelsLast: convertedVAE, dtype: .float32)
        try checkpoint.apply(to: vae)
        let decoded = vae.decodePackedLatents(latents).transposed(0, 2, 3, 1)[0]
        let image = round(clip(decoded / 2 + 0.5, min: 0, max: 1) * 255).asType(.uint8)
        eval(image)
        return .init(
            image: image, seed: seed, width: parameters.width, height: parameters.height,
            steps: schedule.timesteps.count, guidance: guidance,
            metadata: ["architecture": "moss-image-dit"])
    }
}
