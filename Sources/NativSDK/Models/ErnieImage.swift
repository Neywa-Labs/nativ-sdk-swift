import Foundation
import MLX

enum ErnieImageVariant: String {
    case base = "ernie-image"
    case turbo = "ernie-image-turbo"

    var steps: Int { self == .turbo ? 8 : 50 }
    var repository: String { self == .turbo ? "baidu/ERNIE-Image-Turbo" : "baidu/ERNIE-Image" }

    func guidance(editing: Bool) -> Float {
        switch self {
        case .base: 4
        case .turbo: editing ? 3 : 1
        }
    }

    static func named(_ name: String) -> Self? {
        switch name.lowercased() {
        case "ernie-image", "ernie-image-base", "baidu/ernie-image": .base
        case "ernie-image-turbo", "ernie-turbo", "baidu/ernie-image-turbo": .turbo
        default: nil
        }
    }

    static func resolve(
        in directory: URL, source: ModelSource, transformer: DiffusionConfiguration
    ) throws -> Self {
        if let metadata = try ImageCheckpointMetadata.optional(
            "mlx_ernie_image.json", in: directory)
        {
            if let variant = named(metadata.string("source")) { return variant }
            let name = metadata.string("variant")
            if !name.isEmpty {
                guard let variant = named(name) else {
                    throw ModelExecutionError.invalidOptions("Unknown ERNIE-Image variant: \(name)")
                }
                return variant
            }
        }
        if case .repository(let id, _) = source, let variant = named(id) { return variant }
        let parent = directory.deletingLastPathComponent()
        let hubName = parent.deletingLastPathComponent().lastPathComponent.lowercased()
        if parent.lastPathComponent == "snapshots", hubName.hasPrefix("models--"),
            let variant = named(
                String(hubName.dropFirst(8)).replacingOccurrences(of: "--", with: "/"))
        {
            return variant
        }
        if let variant = named(directory.lastPathComponent) { return variant }
        // Native Turbo configurations carry LoRA fields left over from distillation.
        return transformer.values["use_lora"] != nil && transformer.values["lora_rank"] != nil
            ? .turbo : .base
    }
}

struct ErnieImageSchedule {
    let sigmas: MLXArray
    let trainTimesteps: Float

    init(steps: Int, shift: Float, trainTimesteps: Float) {
        let base = linspace(1, 0, count: steps + 1)
        sigmas = shift * base / (1 + (shift - 1) * base)
        self.trainTimesteps = trainTimesteps
    }

    static func editStart(steps: Int, strength: Float) -> Int {
        let count = (Float(steps) * strength).rounded(.toNearestOrEven)
        return count >= Float(steps) ? 0 : steps - max(1, Int(count))
    }

    func timestep(at index: Int, batch: Int, dtype: DType) -> MLXArray {
        broadcast((trainTimesteps * sigmas[index]).asType(dtype), to: [batch])
    }

    func step(_ velocity: MLXArray, at index: Int, latents: MLXArray) -> MLXArray {
        let delta = sigmas[index + 1] - sigmas[index]
        return (latents.asType(.float32) + delta * velocity.asType(.float32)).asType(latents.dtype)
    }
}

final class ErnieImagePipeline: PromptImageGenerationModel {
    let directory: URL
    let supportsImageEditing: Bool
    let bundledEnhancer: (@Sendable () async throws -> URL)?
    let enhancerTokenizerLoader: any TokenizerLoader

    private let variant: ErnieImageVariant
    private let transformerConfiguration: ErnieImageTransformerConfig
    private let textConfiguration: Ministral3Configuration
    private let decoderConfiguration: Flux2VAEConfiguration
    private let shift: Float
    private let trainTimesteps: Float
    private var transformer: ErnieImageTransformer?
    private var vae: Flux2VAE?
    private var promptFeatures = [(tokens: [Int], features: MLXArray)]()

    init(
        directory: URL, source: ModelSource,
        bundledEnhancer: (@Sendable () async throws -> URL)? = nil,
        enhancerTokenizerLoader: any TokenizerLoader = DefaultTokenizerLoader()
    ) throws {
        self.directory = directory
        self.bundledEnhancer = bundledEnhancer
        self.enhancerTokenizerLoader = enhancerTokenizerLoader
        let config = try ImageCheckpointMetadata.required("transformer/config.json", in: directory)
        variant = try ErnieImageVariant.resolve(in: directory, source: source, transformer: config)
        transformerConfiguration = try ErnieImageTransformerConfig(config)
        textConfiguration = try ErnieImageTextConfiguration.read(
            in: directory.appending(path: "text_encoder"))
        decoderConfiguration = try Flux2VAEConfiguration.read(from: directory, batchNormEps: 1e-5)
        guard textConfiguration.hiddenSize == transformerConfiguration.textWidth,
            decoderConfiguration.latentChannels * 4 == transformerConfiguration.channels,
            decoderConfiguration.spatialScale == 8
        else {
            throw ModelExecutionError.invalidOptions(
                "Incompatible ERNIE-Image component dimensions")
        }
        let scheduler = try ImageCheckpointMetadata.optional(
            "scheduler/scheduler_config.json", in: directory)
        shift = scheduler?.float("shift", 4) ?? 4
        trainTimesteps = scheduler?.float("num_train_timesteps", 1000) ?? 1000
        guard shift.isFinite, shift > 0, trainTimesteps.isFinite, trainTimesteps > 0 else {
            throw ModelExecutionError.invalidOptions("Invalid ERNIE-Image scheduler configuration")
        }
        let vaeDirectory = directory.appending(path: "vae").resolvingSymlinksInPath()
        let vaeFiles = try safetensorWeightURLs(in: vaeDirectory)
        let tensors = try vaeFiles.flatMap { try safetensorSpansInFileOrder(url: $0) }
        supportsImageEditing =
            tensors.contains { $0.name.hasPrefix("encoder.") }
            && tensors.contains { $0.name.hasPrefix("quant_conv.") }
    }

    func releaseComponents() {
        transformer = nil
        vae = nil
        MLX.Memory.clearCache()
    }

    func validate(_ parameters: ImageGenerationParameters, references: Int) throws {
        guard parameters.width > 0, parameters.height > 0,
            parameters.width.isMultiple(of: 16), parameters.height.isMultiple(of: 16)
        else {
            throw ModelExecutionError.invalidOptions(
                "ERNIE-Image width and height must be positive multiples of 16")
        }
        guard references <= 1 else {
            throw ModelExecutionError.invalidOptions("ERNIE-Image accepts one reference image")
        }
        guard references == 0 || supportsImageEditing else {
            throw ModelExecutionError.invalidOptions(
                "ERNIE-Image editing requires VAE encoder weights")
        }
        guard (parameters.steps ?? 1) > 0, (parameters.steps ?? 1) < Int.max,
            (parameters.guidance ?? 1).isFinite, (parameters.guidance ?? 1) >= 0,
            parameters.maxPromptLength > 0, parameters.strength > 0, parameters.strength <= 1
        else { throw ModelExecutionError.invalidOptions("Invalid ERNIE-Image sampling options") }
    }

    func generateImage(
        prompt: String, referenceImages: [MLXArray], tokenizer: any Tokenizer,
        parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        try validate(parameters, references: referenceImages.count)
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ModelExecutionError.invalidOptions("ERNIE-Image requires a nonempty prompt")
        }
        let guidance = parameters.guidance ?? variant.guidance(editing: !referenceImages.isEmpty)
        let prompts = guidance > 1 ? [parameters.negativePrompt ?? "", prompt] : [prompt]
        let tokenLimit = min(parameters.maxPromptLength, 2048)
        let tokens = prompts.map { caption in
            let tokens = tokenizer.encode(text: caption, addSpecialTokens: true)
            return Array(tokens.prefix(tokenLimit))
        }
        guard tokens.allSatisfy({ !$0.isEmpty }) else {
            throw ModelExecutionError.invalidOptions(
                "ERNIE-Image tokenizer returned an empty caption")
        }
        let captions = try encode(tokens)
        return try sample(
            captions, reference: referenceImages.first, guidance: guidance, parameters: parameters)
    }

    func generate(
        promptEmbeddings: MLXArray, textIDs: MLXArray, parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        try validate(parameters, references: 0)
        let guidance = parameters.guidance ?? variant.guidance(editing: false)
        guard guidance <= 1, promptEmbeddings.ndim == 3, promptEmbeddings.dim(0) == 1,
            promptEmbeddings.dim(1) > 0,
            promptEmbeddings.dim(2) == transformerConfiguration.textWidth
        else {
            throw ModelExecutionError.invalidOptions(
                "ERNIE-Image direct embeddings require compatible caption features and guidance at most 1"
            )
        }
        return try sample(
            [promptEmbeddings.asType(.bfloat16)], reference: nil, guidance: guidance,
            parameters: parameters)
    }

    private func encode(_ tokens: [[Int]]) throws -> [MLXArray] {
        var encoder: Ministral3ModelInner?
        defer {
            if encoder != nil {
                encoder = nil
                MLX.Memory.clearCache()
            }
        }
        return try tokens.map { ids in
            try Task.checkCancellation()
            if let index = promptFeatures.firstIndex(where: { $0.tokens == ids }) {
                let entry = promptFeatures.remove(at: index)
                promptFeatures.append(entry)
                return entry.features
            }
            if encoder == nil {
                encoder = try ErnieImageTextConfiguration.load(
                    from: directory.appending(path: "text_encoder"),
                    configuration: textConfiguration)
            }
            let features = encoder!.encode(
                MLXArray(ids).reshaped(1, -1), penultimate: true, useQueryTemperature: false)
            eval(features)
            promptFeatures.append((ids, features))
            if promptFeatures.count > 2 { promptFeatures.removeFirst() }
            return features
        }
    }

    private func loadTransformer() throws -> ErnieImageTransformer {
        if let transformer { return transformer }
        try Task.checkCancellation()
        let model = ErnieImageTransformer(transformerConfiguration)
        var checkpoint = try DiffusionCheckpoint(
            directory: directory.appending(path: "transformer"))
        let channelsLast = Self.channelsLast(checkpoint.metadata)
        var weights = [String: MLXArray]()
        for (key, value) in checkpoint.weights {
            let name =
                key
                .replacingOccurrences(of: "adaLN_modulation.1.", with: "adaln_modulation.")
                .replacingOccurrences(of: ".to_out.0.", with: ".to_out.")
            var tensor = value.dtype.isFloatingPoint ? value.asType(.bfloat16) : value
            if name == "x_embedder.proj.weight", !channelsLast {
                tensor = tensor.transposed(0, 2, 3, 1)
            }
            weights[name] = tensor
        }
        checkpoint.weights = weights
        try checkpoint.apply(to: model)
        transformer = model
        return model
    }

    private func loadVAE(editing: Bool) throws -> Flux2VAE {
        if let vae, !editing || vae.supportsEncoding { return vae }
        try Task.checkCancellation()
        let config =
            editing
            ? try Flux2VAEConfiguration.read(
                from: directory, includeEncoder: true, batchNormEps: 1e-5)
            : decoderConfiguration
        let model = Flux2VAE(config, activationDType: .bfloat16, encoderHiddenDType: .float32)
        var checkpoint = try DiffusionCheckpoint(directory: directory.appending(path: "vae"))
        checkpoint.weights = Flux2WeightLoader.sanitizeVAEWeights(
            checkpoint.weights, includeEncoder: editing,
            channelsLast: Self.channelsLast(checkpoint.metadata),
            dtype: .bfloat16)
        try checkpoint.apply(to: model)
        vae = model
        return model
    }

    private static func channelsLast(_ metadata: DiffusionConfiguration) -> Bool {
        switch metadata.string("tensor_layout") {
        case "mlx_nhwc": true
        case "pytorch_nchw": false
        default: metadata.values["mflux_version"] != nil || metadata.string("format") == "mlx"
        }
    }

    private func sample(
        _ captions: [MLXArray], reference: MLXArray?, guidance: Float,
        parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        let editing = reference != nil
        let steps = parameters.steps ?? variant.steps
        let start =
            editing ? ErnieImageSchedule.editStart(steps: steps, strength: parameters.strength) : 0
        let schedule = ErnieImageSchedule(
            steps: steps, shift: shift, trainTimesteps: trainTimesteps)
        let seed = parameters.seed ?? UInt64.random(in: UInt64.min ... UInt64.max)
        let key = MLXRandom.key(seed)
        let (h, w) = (parameters.height / 16, parameters.width / 16)
        let vae = try loadVAE(editing: editing)
        var latents: MLXArray
        if let reference {
            let rgb = DiffusionRGBResize.resize(
                DiffusionRGBResize.rgbBytes(reference),
                height: parameters.height, width: parameters.width, filter: .lanczos)
            let pixels = (rgb.asType(.float32) / 127.5 - 1).transposed(0, 3, 1, 2)
            let source = try vae.encodePackedLatents(pixels)
            let noise = MLXRandom.normal(source.shape, dtype: source.dtype, key: key)
            let sigma = schedule.sigmas[start].asType(source.dtype)
            latents = (1 - sigma) * source + sigma * noise
            eval(latents)
        } else {
            latents = MLXRandom.normal(
                [1, transformerConfiguration.channels, h, w], dtype: .bfloat16, key: key)
        }
        let model = try loadTransformer()
        let lengths = captions.map { $0.dim(1) }
        let maxLength = lengths.reduce(0, max)
        let paddedCaptions = zip(captions, lengths).map { features, length in
            padded(features, widths: [0, .init((0, maxLength - length)), 0])
        }
        let condition = model.condition(
            text: concatenated(paddedCaptions, axis: 0), lengths: lengths,
            height: h, width: w, dtype: latents.dtype)
        for index in start ..< steps {
            try Task.checkCancellation()
            let input = captions.count == 2 ? concatenated([latents, latents], axis: 0) : latents
            var velocity = model(
                input,
                timestep: schedule.timestep(at: index, batch: captions.count, dtype: latents.dtype),
                condition: condition)
            if captions.count == 2 {
                let parts = split(velocity, parts: 2, axis: 0)
                velocity = parts[0] + guidance * (parts[1] - parts[0])
            }
            latents = schedule.step(velocity, at: index, latents: latents)
            eval(latents)
        }
        let decoded = vae.decodePackedLatents(latents)
        let rgb = clip(decoded / 2 + 0.5, min: 0, max: 1)[0]
            .transposed(1, 2, 0).asType(.float32)
        let image = round(rgb * 255).asType(.uint8)
        eval(image)
        return .init(
            image: image, seed: seed, width: parameters.width, height: parameters.height,
            steps: steps - start, guidance: guidance,
            metadata: ["variant": variant.rawValue, "reference_count": editing ? "1" : "0"])
    }
}
