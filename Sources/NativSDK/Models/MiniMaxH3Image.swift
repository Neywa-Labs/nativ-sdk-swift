import Foundation
import MLX

public struct MiniMaxH3ImageGenerationOptions: Sendable {
    /// Internal canvas pixel budget. The output is resized to the requested dimensions.
    public var minimumCanvasArea: Int

    public init(minimumCanvasArea: Int = 4_194_304) {
        self.minimumCanvasArea = minimumCanvasArea
    }
}

enum MiniMaxH3ImagePacking {
    static let times: [Float] = [
        0, 1 - 0.9675236940383911, 1 - 0.8875485062599182, 0.25,
        1 - 0.5327739119529724, 1 - 0.23200887441635132,
    ]

    static func canvas(width: Int, height: Int, minimumArea: Int, alignment: Int) -> (Int, Int) {
        let scale = max(1, sqrt(Double(minimumArea) / (Double(width) * Double(height))))
        return (
            aligned(Double(height) * scale, alignment), aligned(Double(width) * scale, alignment)
        )
    }

    private static func aligned(_ length: Double, _ factor: Int) -> Int {
        max(factor, Int((length / Double(factor)).rounded(.toNearestOrEven)) * factor)
    }

    static func referenceSize(_ image: MLXArray, shortest: ClosedRange<Int>) -> (Int, Int) {
        let (height, width) = (image.dim(1), image.dim(2))
        let small = min(height, width)
        if shortest.contains(small) { return (height, width) }
        let ratio =
            Double(min(max(small, shortest.lowerBound), shortest.upperBound)) / Double(small)
        return (aligned(Double(height) * ratio, 32), aligned(Double(width) * ratio, 32))
    }

    struct Layout {
        let positions: MLXArray
        let modulationRows: MLXArray
        let timeRows: MLXArray
        let referenceCount: Int
        let targetStart: Int
    }

    static func layout(text: [Int32], grids: [(Int, Int)], patch: [Int]) -> Layout {
        var positions = text.indices.flatMap { [Float($0), Float(0), Float(0)] }
        var tags = text
        var timesteps = Array(repeating: Int32(0), count: text.count)
        var references = 0
        for (index, grid) in grids.enumerated() {
            let root = sqrt(Double(grid.0 * grid.1))
            let rows = grid.0 / patch[1]
            let columns = grid.1 / patch[2]
            func axis(size: Int, count: Int) -> [Float] {
                let ratio = Double(size) / root
                let first = (1 - ratio) / 2
                let delta = (first + ratio - first) / Double(count)
                return (0 ..< count).map { Float((first + Double($0) * delta) * 32) }
            }
            let y = axis(size: grid.0, count: rows)
            let x = axis(size: grid.1, count: columns)
            let reference = index < grids.count - 1
            for row in y {
                for column in x {
                    positions.append(contentsOf: [Float(text.count + index), row, column])
                }
            }
            let count = rows * columns
            tags.append(contentsOf: repeatElement(0, count: count))
            timesteps.append(contentsOf: repeatElement(reference ? 1 : 0, count: count))
            if reference { references += count }
        }
        let time = MLXArray(timesteps)
        return Layout(
            positions: MLXArray(positions, [tags.count, 3]),
            modulationRows: time * 3 + MLXArray(tags), timeRows: time,
            referenceCount: references, targetStart: text.count + references)
    }

    // Patches are row-major. Each row groups the two spatial axes inside each channel.
    static func pack(_ latents: MLXArray, patch: [Int]) -> MLXArray {
        let h = latents.dim(1)
        let w = latents.dim(2)
        let channels = latents.dim(3)
        return latents.reshaped(1, h / patch[1], patch[1], w / patch[2], patch[2], channels)
            .transposed(0, 1, 3, 5, 2, 4).reshaped(-1, channels * patch[1] * patch[2])
    }

    static func unpack(_ rows: MLXArray, height: Int, width: Int, channels: Int, patch: [Int])
        -> MLXArray
    {
        rows.reshaped(1, height / patch[1], width / patch[2], channels, patch[1], patch[2])
            .transposed(0, 1, 4, 2, 5, 3).reshaped(1, height, width, channels)
    }
}

final class MiniMaxH3ImageConditioner {
    let model: Qwen3VL

    init(directory: URL) throws {
        let metadata = try DiffusionConfiguration(
            contentsOf: directory.appending(path: "config.json"))
        var root = metadata.values
        guard var text = root["text_config"] as? [String: Any],
            let depth = text["num_hidden_layers"] as? Int
        else {
            throw ModelExecutionError.invalidOptions(
                "MiniMax-H3-Image requires Qwen3-VL text configuration")
        }
        let layers = metadata.int("h3_conditioner_layer", 50)
        guard layers > 0, layers <= depth else {
            throw ModelExecutionError.invalidOptions("Invalid MiniMax-H3-Image conditioner depth")
        }
        text["num_hidden_layers"] = layers
        text["tie_word_embeddings"] = true
        text["use_final_norm"] = false
        root["text_config"] = text
        let config = try JSONDecoder().decode(
            Qwen3VLConfiguration.self, from: JSONSerialization.data(withJSONObject: root))
        model = Qwen3VL(config)
        var checkpoint = try DiffusionCheckpoint(directory: directory)
        checkpoint.weights = model.sanitize(weights: checkpoint.weights).filter { key, _ in
            if key == "language_model.model.norm.weight" { return false }
            let prefix = "language_model.model.layers."
            guard key.hasPrefix(prefix),
                let layer = Int(key.dropFirst(prefix.count).prefix(while: { $0 != "." }))
            else { return true }
            return layer < layers
        }
        try checkpoint.apply(to: model)
    }

    func encode(
        _ prompt: String, images: [MLXArray], tokenizer: any Tokenizer, maxLength: Int
    ) throws -> (MLXArray, [Int32]) {
        let config = model.config
        let vision = config.visionConfiguration
        let factor = vision.patchSize * vision.spatialMergeSize
        var tokens: [Int] = []
        var tags: [Int32] = []
        var patches: [MLXArray] = []
        var grids: [THW] = []

        func addText(_ text: String) {
            let encoded = tokenizer.encode(text: text, addSpecialTokens: false).prefix(maxLength)
            tokens.append(contentsOf: encoded)
            tags.append(contentsOf: repeatElement(1, count: encoded.count))
        }

        for (index, image) in images.enumerated() {
            try Task.checkCancellation()
            let (height, width) = MiniMaxH3ImagePacking.referenceSize(
                image, shortest: 1024 ... 1024)
            let resized = DiffusionRGBResize.resize(
                image, height: height, width: width, lanczos: true)
            let (h, w) = try QwenVL.targetSize(
                height: height, width: width, factor: factor, minPixels: 65536,
                maxPixels: 16_777_216)
            let values = DiffusionRGBResize.resize(resized, height: h, width: w, lanczos: false)
            let normalized = (values.asType(.float32) / 255 - 0.5) / 0.5
            let (pixels, grid) = try QwenVL.patchify(
                images: [normalized.transposed(0, 3, 1, 2)], mergeSize: vision.spatialMergeSize,
                patchSize: vision.patchSize, temporalPatchSize: vision.temporalPatchSize)
            patches.append(pixels)
            grids.append(grid)
            addText("Picture \(index + 1): ")
            let count = grid.product / (vision.spatialMergeSize * vision.spatialMergeSize)
            tokens.append(config.visionStartTokenId)
            tokens.append(contentsOf: repeatElement(config.imageTokenIndex, count: count))
            tokens.append(config.visionEndTokenId)
            tags.append(contentsOf: repeatElement(0, count: count + 2))
        }
        addText(prompt)
        guard !tokens.isEmpty else {
            throw ModelExecutionError.invalidOptions("MiniMax-H3-Image requires prompt tokens")
        }
        let input = LMInput(
            text: .init(tokens: MLXArray(tokens).reshaped(1, -1)),
            image: patches.isEmpty ? nil : .init(pixels: concatenated(patches), frames: grids))
        let hidden = try model.imageConditioning(input, applyFinalNorm: false)
        eval(hidden)
        return (hidden, tags)
    }
}

final class MiniMaxH3ImagePipeline: PromptImageGenerationModel {
    let supportsImageEditing = true
    let directory: URL
    let conditionerDirectory: URL
    let transformerDirectory: URL
    let vaeDirectory: URL
    let transformerConfig: MiniMaxH3ImageTransformerConfiguration
    let vaeConfig: MiniMaxH3ImageVAEConfiguration
    private var transformer: (MiniMaxH3ImageTransformer, [MiniMaxH3ImageTransformer.Step])?
    private var vae: MiniMaxH3ImageVAE?

    private static let pixelMean: [Float] = [0.485, 0.456, 0.406]
    private static let pixelDeviation: [Float] = [0.229, 0.224, 0.225]

    init(directory: URL, conditionerDirectory: URL? = nil) throws {
        self.directory = directory
        self.conditionerDirectory =
            conditionerDirectory ?? directory.appending(path: "text_encoder")
        transformerDirectory = Self.component("transformer", in: directory)
        vaeDirectory = Self.component("vae", in: directory)
        let t = try ImageCheckpointMetadata.required("config.json", in: transformerDirectory)
        let v = try ImageCheckpointMetadata.required("config.json", in: vaeDirectory)
        transformerConfig = try .init(t)
        vaeConfig = try .init(v)
        guard transformerConfig.channels == vaeConfig.channels else {
            throw ModelExecutionError.invalidOptions(
                "MiniMax-H3-Image latent channel counts disagree")
        }
    }

    private static func component(_ name: String, in root: URL) -> URL {
        let diffusers = root.appending(path: name + "_diffusers")
        return FileManager.default.fileExists(atPath: diffusers.appending(path: "config.json").path)
            ? diffusers : root.appending(path: name)
    }

    private func loadTransformerWeights() throws -> MiniMaxH3ImageTransformer {
        let model = MiniMaxH3ImageTransformer(transformerConfig)
        var checkpoint = try DiffusionCheckpoint(directory: transformerDirectory)
        checkpoint.weights = Dictionary(
            uniqueKeysWithValues: checkpoint.weights.map {
                (MiniMaxH3ImageLayers.checkpointKey($0.key), $0.value)
            })
        try checkpoint.apply(to: model, evaluate: false)
        return model
    }

    private func denoiser() throws -> (MiniMaxH3ImageTransformer, [MiniMaxH3ImageTransformer.Step])
    {
        if let transformer { return transformer }
        let model = try loadTransformerWeights()
        let steps = try model.prepareSteps(MiniMaxH3ImagePacking.times)
        MLX.Memory.clearCache()
        transformer = (model, steps)
        return (model, steps)
    }

    private func autoencoder() throws -> MiniMaxH3ImageVAE {
        if let vae { return vae }
        let model = MiniMaxH3ImageVAE(vaeConfig)
        let metadata = try ImageCheckpointMetadata.required("config.json", in: vaeDirectory)
        var checkpoint = try DiffusionCheckpoint(directory: vaeDirectory)
        checkpoint.weights = MiniMaxH3ImageVAE.checkpointWeights(
            checkpoint.weights, mlxLayout: metadata.string("mlx_vlm_format") == "minimax_h3_image")
        try checkpoint.apply(to: model)
        vae = model
        return model
    }

    private func condition(
        _ prompt: String, images: [MLXArray], tokenizer: any Tokenizer, maxLength: Int
    ) throws -> (MLXArray, [Int32]) {
        try MiniMaxH3ImageConditioner(directory: conditionerDirectory)
            .encode(prompt, images: images, tokenizer: tokenizer, maxLength: maxLength)
    }

    func generateImage(
        prompt: String, referenceImages: [MLXArray], tokenizer: any Tokenizer,
        parameters p: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        try validate(p)
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            referenceImages.count <= 16
        else {
            throw ModelExecutionError.invalidOptions(
                "MiniMax-H3-Image requires a prompt and accepts up to 16 reference images")
        }

        let images = try referenceImages.map { image in
            guard image.ndim == 4, image.dim(0) == 1, [3, 4].contains(image.dim(1)),
                min(image.dim(2), image.dim(3)) >= 32,
                Double(max(image.dim(2), image.dim(3))) / Double(min(image.dim(2), image.dim(3)))
                    <= 200
            else {
                throw ModelExecutionError.invalidOptions(
                    "MiniMax-H3-Image requires RGB or RGBA references with sides of at least 32 pixels"
                )
            }
            let x = (image.transposed(0, 2, 3, 1).asType(.float32) + 1) / 2
            var rgb = x[.ellipsis, ..<3]
            if x.dim(-1) == 4 {
                let alpha = x[.ellipsis, 3...]
                rgb = rgb * alpha + (1 - alpha)
            }
            return clip(round(rgb * 255), min: 0, max: 255).asType(.uint8)
        }
        let (hidden, tags) = try condition(
            prompt, images: images, tokenizer: tokenizer, maxLength: p.maxPromptLength)
        MLX.Memory.clearCache()
        return try sample(hidden, tags: tags, images: images, p: p)
    }

    func generate(
        promptEmbeddings: MLXArray, textIDs: MLXArray, parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        try validate(parameters)
        guard promptEmbeddings.ndim == 3, promptEmbeddings.dim(0) == 1,
            promptEmbeddings.dim(1) > 0, promptEmbeddings.dim(2) == transformerConfig.text
        else {
            throw ModelExecutionError.invalidOptions("Invalid MiniMax-H3-Image caption embeddings")
        }
        return try sample(
            promptEmbeddings, tags: Array(repeating: 1, count: promptEmbeddings.dim(1)), images: [],
            p: parameters)
    }

    func releaseComponents() {
        transformer = nil
        vae = nil
        MLX.Memory.clearCache()
    }

    func validate(_ p: ImageGenerationParameters) throws {
        guard (p.steps ?? 6) == 6, (p.guidance ?? 1) == 1,
            p.negativePrompt?.isEmpty != false,
            (1 ... 4096).contains(p.maxPromptLength), (1 ... 8192).contains(p.width),
            (1 ... 8192).contains(p.height),
            (1024 ... 16_777_216).contains(p.minimaxH3Image.minimumCanvasArea)
        else {
            throw ModelExecutionError.invalidOptions(
                "MiniMax-H3-Image requires six steps, guidance 1, and valid dimensions without a negative prompt"
            )
        }
    }

    private func sample(
        _ hidden: MLXArray, tags: [Int32], images: [MLXArray], p: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        let ratio = vaeConfig.ratio
        let channels = vaeConfig.channels
        let patch = transformerConfig.patch
        let alignment = ratio * max(patch[1], patch[2])
        let (height, width) = MiniMaxH3ImagePacking.canvas(
            width: p.width, height: p.height,
            minimumArea: p.minimaxH3Image.minimumCanvasArea, alignment: alignment)
        let (lh, lw) = (height / ratio, width / ratio)
        guard height <= 16384, width <= 16384, lh.isMultiple(of: patch[1]),
            lw.isMultiple(of: patch[2])
        else {
            throw ModelExecutionError.invalidOptions(
                "MiniMax-H3-Image canvas geometry exceeds the supported size")
        }
        let seed = p.seed ?? UInt64.random(in: UInt64.min ... UInt64.max)
        var key = MLXRandom.key(seed)
        func randomImage(_ h: Int, _ w: Int) -> MLXArray {
            let (next, sample) = MLXRandom.split(key: key)
            key = next
            return MLXRandom.normal([1, channels, h, w], key: sample).transposed(0, 2, 3, 1)
        }
        let latentMean = MLXArray(vaeConfig.mean)
        let latentDeviation = MLXArray(vaeConfig.deviation)
        let pixelMean = MLXArray(Self.pixelMean)
        let pixelDeviation = MLXArray(Self.pixelDeviation)
        let mutator = try autoencoder()
        var references: [MLXArray] = []
        for image in images {
            let (h, w) = MiniMaxH3ImagePacking.referenceSize(image, shortest: 1024 ... 2048)
            let resized = DiffusionRGBResize.resize(image, height: h, width: w, lanczos: true)
            let pixels = resized[0..., ..<(h / ratio * ratio), ..<(w / ratio * ratio), 0...]
            let moments = try mutator.moments(
                (pixels.asType(.float32) / 255 - pixelMean) / pixelDeviation)
            let mean = moments[.ellipsis, ..<channels]
            let variance = clip(moments[.ellipsis, channels...], min: -30, max: 20)
            let z = (mean + exp(0.5 * variance) * randomImage(mean.dim(1), mean.dim(2)))
                .asType(.float16).asType(.float32)
            let rh = z.dim(1) / patch[1] * patch[1]
            let rw = z.dim(2) / patch[2] * patch[2]
            let norm = (z[0..., ..<rh, ..<rw, 0...] - latentMean) / latentDeviation
            eval(norm)
            references.append(norm)
        }
        let packing = MiniMaxH3ImagePacking.layout(
            text: tags, grids: references.map { ($0.dim(1), $0.dim(2)) } + [(lh, lw)], patch: patch)
        var target = MiniMaxH3ImagePacking.pack(randomImage(lh, lw), patch: patch)
        let frozen = references.map { z in
            let referenceTime = MLXArray(Float(0.999))
            return MiniMaxH3ImagePacking.pack(
                referenceTime * z + (1 - referenceTime) * randomImage(z.dim(1), z.dim(2)),
                patch: patch)
        }
        let referenceRows = frozen.isEmpty ? nil : concatenated(frozen)
        let (model, steps) = try denoiser()
        let rotary = model.rotary(packing.positions)
        let caption = model.encodeText(hidden)
        eval(caption, rotary.cosine, rotary.sine)
        let sigmas = MiniMaxH3ImagePacking.times.map { 1 - $0 } + [0]
        for i in steps.indices {
            try Task.checkCancellation()
            let input = referenceRows.map { concatenated([$0, target]) } ?? target
            let flow = try model(
                input.expandedDimensions(axis: 0), text: caption, rotary: rotary,
                indices: packing.modulationRows, timesteps: packing.timeRows, step: steps[i],
                head: i, targetStart: packing.targetStart)[0]
            target = target + (sigmas[i] - sigmas[i + 1]) * flow.asType(.float32)
            eval(target)
        }
        let latents = MiniMaxH3ImagePacking.unpack(
            target, height: lh, width: lw, channels: channels, patch: patch)
        let decoded = try mutator.decode(latents * latentDeviation + latentMean)
        let pixels = clip(
            (decoded.asType(.float32) * pixelDeviation + pixelMean) * 255, min: 0, max: 255
        )
        .asType(.uint8)
        let output = DiffusionRGBResize.resize(
            pixels, height: p.height, width: p.width, lanczos: true)[0]
        eval(output)
        return .init(
            image: output, seed: seed, width: p.width, height: p.height, steps: 6, guidance: 1)
    }
}
