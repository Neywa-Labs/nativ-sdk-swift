import Foundation
import MLX

enum MageFlowVariant: String, CaseIterable {
    case generation = "mage-flow"
    case generationBase = "mage-flow-base"
    case generationTurbo = "mage-flow-turbo"
    case edit = "mage-flow-edit"
    case editBase = "mage-flow-edit-base"
    case editTurbo = "mage-flow-edit-turbo"

    var editing: Bool {
        switch self {
        case .generation, .generationBase, .generationTurbo: false
        case .edit, .editBase, .editTurbo: true
        }
    }

    var steps: Int {
        switch self {
        case .generation: 20
        case .generationTurbo, .editTurbo: 4
        default: 30
        }
    }

    var guidance: Float {
        self == .generationTurbo || self == .editTurbo ? 1 : 5
    }

    static func named(_ name: String) -> Self? {
        let slug = name.lowercased().replacingOccurrences(of: "_", with: "-")
            .replacingOccurrences(of: "-4b", with: "")
        if let exact = Self(rawValue: slug) { return exact }
        return allCases.sorted { $0.rawValue.count > $1.rawValue.count }
            .first { slug.hasPrefix($0.rawValue + "-") }
    }

    var repository: String { "mage-flow-community/" + rawValue.capitalized }

    static func resolve(directory: URL, source: ModelSource) throws -> Self {
        if let sidecar = try ImageCheckpointMetadata.optional("mlx_mage_flow.json", in: directory),
            let name = sidecar.values["variant"] as? String
        {
            guard let variant = named(name) else {
                throw ModelExecutionError.invalidOptions("Unknown Mage-Flow variant: \(name)")
            }
            return variant
        }
        if case .repository(let repo, _) = source,
            let name = repo.split(separator: "/").last, let variant = named(String(name))
        {
            return variant
        }
        if let variant = named(directory.lastPathComponent) { return variant }
        // A local Hub snapshot ends in a commit hash. Its enclosing cache directory
        // still carries the original checkpoint name.
        for part in directory.pathComponents.reversed() where part.hasPrefix("models--") {
            if let name = part.components(separatedBy: "--").last, let variant = named(name) {
                return variant
            }
        }
        return .generation
    }
}

struct MageFlowSampling {
    let sigmas: [MLXArray]
    let deltas: [MLXArray]

    init(steps: Int, shift: Float, dtype: DType) {
        let base = linspace(Float(1), 1 / Float(steps), count: steps)
        let sigmas = shift * base / (1 + (shift - 1) * base)
        let padded = concatenated([sigmas, MLXArray.zeros([1], dtype: .float32)])
        self.sigmas = sigmas.asType(dtype).split(parts: steps).map { $0.squeezed(axis: 0) }
        self.deltas = (padded[1...] - padded[..<steps]).asType(dtype)
            .split(parts: steps).map { $0.squeezed(axis: 0) }
    }

    static func referenceTokens(_ references: MLXArray) -> MLXArray {
        references.reshaped(1, -1, references.dim(-1))
    }
}

final class MageFlowConditioner {
    private static let generationInstructions =
        "Describe the image by detailing the color, shape, size, texture, quantity, "
        + "text, spatial relationships of the objects and background:"
    private static let editInstructions =
        "Describe the key features of the input image (color, shape, size, texture,"
        + " objects, background), then explain how the user's text instruction should "
        + "alter or modify the image. Generate a new image that meets the user's "
        + "requirements while maintaining consistency with the original input where appropriate."
    private static let imagePlaceholder = "<|vision_start|><|image_pad|><|vision_end|>"

    let model: Qwen3VL
    let processor: Qwen3VLProcessorConfiguration?

    init(directory: URL, editing: Bool, context: Int) throws {
        let config = try JSONDecoder().decode(
            Qwen3VLConfiguration.self,
            from: Data(contentsOf: directory.appending(path: "config.json")))
        guard config.textConfiguration.hiddenSize == context else {
            throw ModelExecutionError.invalidOptions("Mage-Flow conditioner dimensions disagree")
        }
        model = Qwen3VL(config)
        model.removeLanguageHead()
        let processorConfig = directory.appending(path: "preprocessor_config.json")
        processor =
            editing
            ? try JSONDecoder().decode(
                Qwen3VLProcessorConfiguration.self, from: Data(contentsOf: processorConfig))
            : nil
        if let processor {
            let vision = config.visionConfiguration
            guard processor.patchSize == vision.patchSize,
                processor.mergeSize == vision.spatialMergeSize,
                processor.temporalPatchSize == vision.temporalPatchSize,
                processor.patchSize > 0, processor.mergeSize > 0, processor.temporalPatchSize > 0,
                processor.minPixels > 0, processor.maxPixels >= processor.minPixels,
                processor.imageMean.count == 3, processor.imageStd.count == 3,
                processor.imageStd.allSatisfy({ $0.isFinite && $0 > 0 })
            else {
                throw ModelExecutionError.invalidOptions(
                    "Invalid Mage-Flow image processor configuration")
            }
        }
        var checkpoint = try DiffusionCheckpoint(directory: directory)
        checkpoint.weights = model.sanitize(weights: checkpoint.weights).filter {
            !$0.key.hasPrefix("language_model.lm_head.")
        }
        try checkpoint.apply(to: model)
    }

    static func formattedPrompt(_ prompt: String, references: Int) -> String {
        let instructions = references == 0 ? generationInstructions : editInstructions
        let parts = (0 ..< references).map { "Image \($0 + 1): \(imagePlaceholder)" }.joined()
        return "<|im_start|>system\n\(instructions)<|im_end|>\n<|im_start|>user\n"
            + parts + prompt + "<|im_end|>\n<|im_start|>assistant\n"
    }

    static func visionSize(
        height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int
    ) throws -> (Int, Int) {
        guard Double(max(height, width)) / Double(min(height, width)) <= 200 else {
            throw ModelExecutionError.invalidOptions(
                "Mage-Flow reference aspect ratio exceeds 200:1")
        }
        let alignment = Double(factor)
        let h = Int((Double(height) / alignment).rounded(.toNearestOrEven)) * factor
        let w = Int((Double(width) / alignment).rounded(.toNearestOrEven)) * factor
        let area = Double(height) * Double(width)
        if h * w > maxPixels {
            let scale = sqrt(Double(maxPixels) / area) / alignment
            return (
                max(factor, Int(floor(Double(height) * scale)) * factor),
                max(factor, Int(floor(Double(width) * scale)) * factor)
            )
        }
        if h * w < minPixels {
            let scale = sqrt(Double(minPixels) / area) / alignment
            return (
                Int(ceil(Double(height) * scale)) * factor,
                Int(ceil(Double(width) * scale)) * factor
            )
        }
        return (h, w)
    }

    private func vision(
        _ images: [MLXArray], config: Qwen3VLProcessorConfiguration
    ) throws -> LMInput.ProcessedImage {
        let factor = config.patchSize * config.mergeSize
        let batches = try images.map { rgb in
            try Task.checkCancellation()
            let (h, w) = (rgb.dim(1), rgb.dim(2))
            let scale = min(1, 384 / Double(max(h, w)))
            let height = max(1, Int((Double(h) * scale).rounded(.toNearestOrEven)))
            let width = max(1, Int((Double(w) * scale).rounded(.toNearestOrEven)))
            let preview = DiffusionRGBResize.resize(
                rgb, height: height, width: width, filter: .pillowBicubic)
            let (targetH, targetW) = try Self.visionSize(
                height: height, width: width, factor: factor, minPixels: config.minPixels,
                maxPixels: config.maxPixels)
            let resized = DiffusionRGBResize.resize(
                preview, height: targetH, width: targetW, filter: .pillowBicubic)
            let mean = MLXArray(config.imageMean.map(Float.init))
            let deviation = MLXArray(config.imageStd.map(Float.init))
            let pixels = (resized.asType(.float32) / 255 - mean) / deviation
            return try QwenVL.patchify(
                images: [pixels.transposed(0, 3, 1, 2)], mergeSize: config.mergeSize,
                patchSize: config.patchSize, temporalPatchSize: config.temporalPatchSize)
        }
        return LMInput.ProcessedImage(
            pixels: concatenated(batches.map { $0.0 }), frames: batches.map { $0.1 })
    }

    func encode(
        prompts: [String], tokenizer: any Tokenizer, images: [MLXArray], maxTokens: Int
    ) throws -> [MLXArray] {
        let vision = try processor.map { try self.vision(images, config: $0) }
        return try prompts.map { prompt in
            try Task.checkCancellation()
            let formatted = Self.formattedPrompt(prompt, references: images.count)
            let ids = tokenizer.encode(text: formatted, addSpecialTokens: true)
            let trim = images.isEmpty ? 34 : 64
            let tokens: [Int]
            if let vision, let processor {
                tokens = try QwenVL.replacePaddingTokens(
                    in: ids, frames: vision.frames ?? [], paddingToken: "<|image_pad|>",
                    mergeSize: processor.mergeSize, tokenizer: tokenizer)
            } else {
                tokens = Array(ids.prefix(maxTokens + trim))
            }
            guard tokens.count > trim else {
                throw ModelExecutionError.invalidOptions(
                    "Mage-Flow caption is empty after formatting")
            }
            let input = LMInput(
                text: .init(tokens: MLXArray(tokens).reshaped(1, -1)), image: vision)
            let hidden = try model.imageConditioning(input, applyFinalNorm: true)
            let result = contiguous(hidden[0..., trim..., 0...])
            eval(result)
            return result
        }
    }
}

final class MageFlowImagePipeline: PromptImageGenerationModel {
    var supportsImageEditing: Bool { variant.editing }
    private let directory: URL
    private let variant: MageFlowVariant
    private let config: MageFlowTransformerConfiguration
    private let shift: Float
    private var transformer: MageFlowTransformer?
    private var autoencoder: MageFlowVAE?

    init(directory: URL, source: ModelSource) throws {
        self.directory = directory
        variant = try MageFlowVariant.resolve(directory: directory, source: source)
        let metadata = try ImageCheckpointMetadata.required(
            "transformer/config.json", in: directory)
        config = try MageFlowTransformerConfiguration(metadata)
        let vae = try ImageCheckpointMetadata.required("vae/config.json", in: directory)
        guard vae.int("latent_channels", 128) == 128, vae.int("downsample_factor", 16) == 16,
            config.channels == 128, config.output == 128
        else { throw ModelExecutionError.invalidOptions("Invalid Mage-Flow latent geometry") }
        let scheduler = try ImageCheckpointMetadata.optional(
            "scheduler/scheduler_config.json", in: directory)
        shift =
            scheduler?.float("shift", metadata.float("static_shift", 6))
            ?? metadata.float("static_shift", 6)
        guard shift.isFinite, shift > 0 else {
            throw ModelExecutionError.invalidOptions("Invalid Mage-Flow scheduler shift")
        }
    }

    private func validate(_ p: ImageGenerationParameters, referenceCount: Int) throws {
        guard (512 ... 2048).contains(p.width), (512 ... 2048).contains(p.height),
            p.width.isMultiple(of: 16), p.height.isMultiple(of: 16)
        else {
            throw ModelExecutionError.invalidOptions(
                "Mage-Flow width and height must be multiples of 16 in [512, 2048]")
        }
        guard variant.editing == (referenceCount > 0) else {
            throw ModelExecutionError.invalidOptions(
                variant.editing
                    ? "Mage-Flow-Edit requires at least one reference image"
                    : "Select a Mage-Flow-Edit checkpoint to edit reference images")
        }
    }

    func generateImage(
        prompt: String, referenceImages: [MLXArray], tokenizer: any Tokenizer,
        parameters p: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        try validate(p, referenceCount: referenceImages.count)
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ModelExecutionError.invalidOptions("Mage-Flow requires a nonempty prompt")
        }
        let images = referenceImages.map { DiffusionRGBResize.rgbBytes($0) }
        let guidance = p.guidance ?? variant.guidance
        var negative = p.negativePrompt ?? " "
        if variant.editing, negative.isEmpty { negative = " " }
        let prompts = guidance > 1 && !negative.isEmpty ? [prompt, negative] : [prompt]
        let text = try MageFlowConditioner(
            directory: directory.appending(path: "text_encoder"), editing: variant.editing,
            context: config.context
        ).encode(
            prompts: prompts, tokenizer: tokenizer, images: images,
            maxTokens: min(p.maxPromptLength, config.maxTokens))
        MLX.Memory.clearCache()
        return try sample(features: text, images: images, parameters: p)
    }

    func generate(
        promptEmbeddings: MLXArray, textIDs: MLXArray, parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        try validate(parameters, referenceCount: 0)
        guard (parameters.guidance ?? variant.guidance) <= 1,
            promptEmbeddings.ndim == 3, promptEmbeddings.dim(0) == 1,
            promptEmbeddings.dim(1) > 0, promptEmbeddings.dim(2) == config.context
        else {
            throw ModelExecutionError.invalidOptions(
                "Mage-Flow direct embeddings require compatible caption features and guidance at most 1"
            )
        }
        return try sample(features: [promptEmbeddings], images: [], parameters: parameters)
    }

    private func loadTransformer() throws -> MageFlowTransformer {
        if let transformer { return transformer }
        let model = MageFlowTransformer(config)
        var checkpoint = try DiffusionCheckpoint(
            directory: directory.appending(path: "transformer"))
        checkpoint.weights = Dictionary(
            uniqueKeysWithValues: checkpoint.weights.map {
                (MageFlowTransformer.checkpointKey($0.key), $0.value)
            })
        try checkpoint.apply(to: model)
        transformer = model
        return model
    }

    private func loadAutoencoder() throws -> MageFlowVAE {
        if let autoencoder { return autoencoder }
        let model = MageFlowVAE(editing: variant.editing)
        var checkpoint = try DiffusionCheckpoint(directory: directory.appending(path: "vae"))
        let mlxLayout =
            checkpoint.metadata.string("tensor_layout") == "mlx_nhwc"
            || checkpoint.metadata.string("mlx_vlm_format") == "mage_flow"
        checkpoint.weights = MageFlowVAE.checkpointWeights(
            checkpoint.weights, editing: variant.editing, mlxLayout: mlxLayout)
        try checkpoint.apply(to: model)
        autoencoder = model
        return model
    }

    private func sample(
        features: [MLXArray], images: [MLXArray], parameters p: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        let steps = p.steps ?? variant.steps
        let guidance = p.guidance ?? variant.guidance
        let seed = p.seed ?? UInt64.random(in: UInt64.min ... UInt64.max)
        let key = MLXRandom.key(seed)
        let (h, w) = (p.height / 16, p.width / 16)
        let targetCount = h * w
        let vae = try loadAutoencoder()
        let references: MLXArray?
        if images.isEmpty {
            references = nil
        } else {
            let batch = concatenated(
                images.map { image in
                    DiffusionRGBResize.resize(
                        image, height: p.height, width: p.width, filter: .pillowBicubic)
                })
            let normalized = (batch.asType(.float32) / 127.5 - 1).asType(.bfloat16)
            let tokens = try MageFlowSampling.referenceTokens(vae.encode(normalized, key: key))
            eval(tokens)
            references = tokens
        }

        let model = try loadTransformer()
        let text = features.map(model.condition)
        let rotary = model.rotary(grids: Array(repeating: (h, w), count: images.count + 1))
        eval(rotary.cosine, rotary.sine)
        for condition in text { eval(condition) }
        var target = MLXRandom.normal([1, targetCount, 128], dtype: .bfloat16, key: key)
        let schedule = MageFlowSampling(steps: steps, shift: shift, dtype: target.dtype)
        for (sigma, delta) in zip(schedule.sigmas, schedule.deltas) {
            try Task.checkCancellation()
            let latents = references.map { concatenated([target, $0], axis: 1) } ?? target
            var velocity = model(latents, text: text[0], sigma: sigma, rotary: rotary)
            if text.count > 1 {
                let unconditional = model(latents, text: text[1], sigma: sigma, rotary: rotary)
                velocity = unconditional + guidance * (velocity - unconditional)
            }
            target = target + delta * velocity[0..., ..<targetCount, 0...].asType(target.dtype)
            eval(target)
        }
        let decoded = vae.decode(target.reshaped(1, h, w, 128))
        let rgb = round(127.5 * (clip(decoded[0], min: -1, max: 1) + 1)).asType(.uint8)
        eval(rgb)
        return .init(
            image: rgb, seed: seed, width: p.width, height: p.height, steps: steps,
            guidance: guidance,
            metadata: ["variant": variant.rawValue, "reference_count": String(images.count)])
    }
}
