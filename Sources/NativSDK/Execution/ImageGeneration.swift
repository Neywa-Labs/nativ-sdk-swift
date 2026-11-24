import Foundation
import MLX

public enum ImagePromptExpansion: Sendable, Equatable {
    /// Generate from the supplied prompt.
    case disabled
    /// Use the image pipeline's caption model when it provides one.
    case defaultModel
    /// Use a compatible language model from a local directory or repository.
    case model(ModelSource)
}

public struct ImageGenerationParameters: Sendable {
    public var maxPromptLength: Int
    public var bucketedPrompt: Bool
    public var ideogram4: Ideogram4GenerationOptions
    public var useImageTokenPrior: Bool
    public var stochasticSampling: Bool?
    public var seed: UInt64?
    public var steps: Int?
    public var width: Int
    public var height: Int
    public var guidance: Float?
    public var negativePrompt: String?
    public var strength: Float
    public var cfgTruncation: Float
    /// Qwen-Image reference pixel-area resolution, independent of output dimensions.
    public var referenceResolution: Int
    public var useReferenceCache: Bool
    public var minimaxH3Image: MiniMaxH3ImageGenerationOptions
    public var mossImage: MOSSImageGenerationOptions
    private var requestedPromptExpansion: ImagePromptExpansion?

    /// Controls whether to rewrite the prompt before image generation.
    public var promptExpansion: ImagePromptExpansion {
        get { requestedPromptExpansion ?? .disabled }
        set { requestedPromptExpansion = newValue }
    }

    fileprivate var ideogramPromptExpansion: ImagePromptExpansion {
        if let requestedPromptExpansion { return requestedPromptExpansion }
        return ideogram4.legacyPromptExpansionModel.map(ImagePromptExpansion.model) ?? .disabled
    }

    public init(
        maxPromptLength: Int = 512,
        bucketedPrompt: Bool = true,
        ideogram4: Ideogram4GenerationOptions = .init(),
        useImageTokenPrior: Bool = false,
        stochasticSampling: Bool? = nil,
        seed: UInt64? = nil,
        steps: Int? = nil,
        width: Int = 1024,
        height: Int = 1024,
        guidance: Float? = nil,
        negativePrompt: String? = nil,
        strength: Float = 0.6,
        cfgTruncation: Float = 1,
        referenceResolution: Int = 1024,
        useReferenceCache: Bool = true,
        minimaxH3Image: MiniMaxH3ImageGenerationOptions = .init(),
        promptExpansion: ImagePromptExpansion? = nil,
        mossImage: MOSSImageGenerationOptions = .init(),
    ) {
        self.maxPromptLength = maxPromptLength
        self.bucketedPrompt = bucketedPrompt
        self.ideogram4 = ideogram4
        self.useImageTokenPrior = useImageTokenPrior
        self.stochasticSampling = stochasticSampling
        self.seed = seed
        self.steps = steps
        self.width = width
        self.height = height
        self.guidance = guidance
        self.negativePrompt = negativePrompt
        self.strength = strength
        self.cfgTruncation = cfgTruncation
        self.referenceResolution = referenceResolution
        self.useReferenceCache = useReferenceCache
        self.minimaxH3Image = minimaxH3Image
        self.mossImage = mossImage
        self.requestedPromptExpansion = promptExpansion
    }
}

public struct ImageGenerationResult {
    public let image: MLXArray
    public let seed: UInt64
    public let width: Int
    public let height: Int
    public let steps: Int
    public let guidance: Float
    public let metadata: [String: String]

    public init(
        image: MLXArray,
        seed: UInt64,
        width: Int,
        height: Int,
        steps: Int,
        guidance: Float,
        metadata: [String: String] = [:],
    ) {
        self.image = image
        self.seed = seed
        self.width = width
        self.height = height
        self.steps = steps
        self.guidance = guidance
        self.metadata = metadata
    }
}

public protocol ImageGenerationModel {
    func generate(
        promptEmbeddings: MLXArray,
        textIDs: MLXArray,
        parameters: ImageGenerationParameters,
    )
        throws -> ImageGenerationResult
}

/// Image pixels safe to share across tasks and actors. Reference images use
/// normalized NCHW [1, channels, height, width] RGB or RGBA values in [-1, 1].
/// Generated images use HWC [height, width, channels] values in 0...255.
/// Use ``cgImage()``, ``pngData()``, or ``savePNG(to:)`` to display or export generated images.
public struct ModelImageData: Sendable {
    private enum Storage: Sendable {
        case pixels([Float], TensorConversionCache<MaterializedArray>)
        case tensor(MaterializedArray, TensorConversionCache<[Float]>)
    }

    private let storage: Storage
    public let shape: [Int]

    /// Evaluated pixels in their original shape and dtype.
    public var tensor: MaterializedArray {
        switch storage {
        case .tensor(let tensor, _): tensor
        case .pixels(let pixels, let cache): cache.value { MLXArray(pixels, shape).materialized() }
        }
    }

    public var pixels: [Float] {
        switch storage {
        case .pixels(let pixels, _): pixels
        case .tensor(let tensor, let cache): cache.value { tensor.asArray(Float.self) }
        }
    }

    /// Takes an immutable snapshot. An existing materialized tensor is reused.
    public init(tensor: MLXArray) throws {
        let shape = tensor.shape
        guard !shape.isEmpty, shape.allSatisfy({ $0 > 0 }) else {
            throw ModelExecutionError.invalidOptions("invalid image shape")
        }
        storage = .tensor(tensor.materialized(), .init())
        self.shape = shape
    }

    fileprivate var hasNormalizedPixels: Bool {
        switch storage {
        case .pixels(let pixels, _):
            return pixels.allSatisfy { (-1 ... 1).contains($0) }
        case .tensor(let tensor, _):
            let values = tensor.asType(.float32)
            return all((values .>= -1) .&& (values .<= 1)).item(Bool.self)
        }
    }

    public init(pixels: [Float], shape: [Int]) throws {
        var count = 1
        for dimension in shape {
            let next = count.multipliedReportingOverflow(by: dimension)
            guard dimension > 0, !next.overflow else {
                throw ModelExecutionError.invalidOptions("invalid image shape")
            }
            count = next.partialValue
        }
        guard !shape.isEmpty, count == pixels.count else {
            throw ModelExecutionError.invalidOptions("image shape/pixel mismatch")
        }
        storage = .pixels(pixels, .init())
        self.shape = shape
    }
}

public struct ModelImageResult: Sendable {
    public let source: ModelSource
    public let image: ModelImageData
    public let seed: UInt64
    public let steps: Int
    public let width: Int
    public let height: Int
    public let guidance: Float
    public let metadata: [String: String]
}

public protocol PromptImageGenerationModel: ImageGenerationModel, AnyObject {
    var supportsImageEditing: Bool { get }
    func generateImage(
        prompt: String, referenceImages: [MLXArray], tokenizer: any Tokenizer,
        parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult
}

extension Flux2ImagePipeline: PromptImageGenerationModel {
    public var supportsImageEditing: Bool { vae.supportsEncoding || vae.encoderDirectory != nil }
    public func generateImage(
        prompt: String, referenceImages: [MLXArray], tokenizer: any Tokenizer,
        parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        if referenceImages.isEmpty {
            return try generate(
                prompt: prompt, tokenizer: tokenizer, parameters: parameters,
                maxLength: parameters.maxPromptLength, bucketed: parameters.bucketedPrompt)
        }
        return try edit(
            prompt: prompt, referenceImages: referenceImages, tokenizer: tokenizer,
            parameters: parameters, maxLength: parameters.maxPromptLength,
            bucketed: parameters.bucketedPrompt)
    }
}
extension BonsaiImagePipeline: PromptImageGenerationModel {
    public var supportsImageEditing: Bool { false }
    public func generateImage(
        prompt: String, referenceImages: [MLXArray], tokenizer: any Tokenizer,
        parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        guard referenceImages.isEmpty else {
            throw ModelExecutionError.invalidOptions(
                "image editing is unsupported by this pipeline")
        }
        return try generate(
            prompt: prompt, tokenizer: tokenizer, parameters: parameters,
            maxLength: parameters.maxPromptLength, bucketed: parameters.bucketedPrompt)
    }
}
extension Ideogram4ImagePipeline: PromptImageGenerationModel {
    public var supportsImageEditing: Bool { false }
    public func generateImage(
        prompt: String, referenceImages: [MLXArray], tokenizer: any Tokenizer,
        parameters: ImageGenerationParameters
    ) throws -> ImageGenerationResult {
        guard referenceImages.isEmpty else {
            throw ModelExecutionError.invalidOptions(
                "image editing is unsupported by this pipeline")
        }
        return try generate(
            prompt: prompt, tokenizer: tokenizer, parameters: parameters,
            autoJSONCaption: parameters.ideogram4.autoJSONCaption)
    }
}

public actor NativImageModel {
    public nonisolated let source: ModelSource
    public nonisolated let modelDirectory: URL
    public nonisolated let supportsImageEditing: Bool
    private let model: any PromptImageGenerationModel
    private let tokenizer: any Tokenizer
    private let policy: ModelExecutionPolicy
    private let admission = ExecutionAdmission()

    public init(
        model: sending any PromptImageGenerationModel, tokenizer: sending any Tokenizer,
        source: ModelSource, modelDirectory: URL, policy: ModelExecutionPolicy = .init()
    ) {
        self.supportsImageEditing = model.supportsImageEditing
        self.model = model
        self.tokenizer = tokenizer
        self.source = source
        self.modelDirectory = modelDirectory
        self.policy = policy
    }

    init(
        resources: ExecutionTransfer<(any PromptImageGenerationModel, any Tokenizer)>,
        source: ModelSource, modelDirectory: URL, policy: ModelExecutionPolicy
    ) {
        let (model, tokenizer) = resources.take()
        self.supportsImageEditing = model.supportsImageEditing
        self.model = model
        self.tokenizer = tokenizer
        self.source = source
        self.modelDirectory = modelDirectory
        self.policy = policy
    }

    public nonisolated func generate(
        prompt: String, referenceImages: [ModelImageData] = [],
        parameters: ImageGenerationParameters = .init()
    ) -> ModelOperation<ModelImageResult> {
        ModelOperation(
            Task {
                try await self.run(
                    prompt: prompt, references: referenceImages, parameters: parameters)
            })
    }

    package func promptShapes(prompt: String, parameters: ImageGenerationParameters) async throws
        -> [[Int]]
    {
        guard parameters.maxPromptLength > 0 else {
            throw ModelExecutionError.invalidOptions("maxPromptLength must be positive")
        }
        try await admission.acquire()
        defer {
            Stream.defaultStream.synchronize()
            admission.release()
        }
        let transfer = ExecutionTransfer((model, tokenizer))
        return try await runModelWorker(policy: policy) {
            let (model, tokenizer) = transfer.take()
            switch model {
            case let pipeline as Flux2ImagePipeline:
                let inputs = try pipeline.encodePrompt(
                    prompt, tokenizer: tokenizer,
                    maxLength: parameters.maxPromptLength, bucketed: parameters.bucketedPrompt)
                eval(inputs.embeddings, inputs.textIDs)
                return [inputs.embeddings.shape, inputs.textIDs.shape]
            case let pipeline as BonsaiImagePipeline:
                let inputs = try pipeline.encodePrompt(
                    prompt, tokenizer: tokenizer,
                    maxLength: parameters.maxPromptLength, bucketed: parameters.bucketedPrompt)
                eval(inputs.embeddings, inputs.textIDs)
                return [inputs.embeddings.shape, inputs.textIDs.shape]
            case let pipeline as Ideogram4ImagePipeline:
                let inputs = try pipeline.encodePrompt(
                    prompt, tokenizer: tokenizer,
                    width: parameters.width, height: parameters.height,
                    autoJSONCaption: parameters.ideogram4.autoJSONCaption)
                eval(inputs.llmFeatures, inputs.positionIDs)
                return [inputs.llmFeatures.shape, inputs.positionIDs.shape]
            default:
                throw ModelExecutionError.invalidOptions(
                    "Prompt-only encoding is available for Flux2, Bonsai and Ideogram4")
            }
        }
    }

    private func run(
        prompt: String, references: [ModelImageData], parameters: ImageGenerationParameters
    ) async throws -> ModelImageResult {
        try await admission.acquire()
        defer {
            Stream.defaultStream.synchronize()
            admission.release()
        }
        guard !prompt.isEmpty, parameters.maxPromptLength > 0, (parameters.steps ?? 1) > 0,
            parameters.width > 0,
            parameters.height > 0,
            (parameters.guidance ?? 1).isFinite, (parameters.guidance ?? 1) >= 0,
            parameters.strength.isFinite, parameters.strength > 0, parameters.strength <= 1,
            parameters.cfgTruncation.isFinite, (0 ... 1).contains(parameters.cfgTruncation),
            parameters.referenceResolution >= 256
        else { throw ModelExecutionError.invalidOptions("invalid image generation options") }
        guard references.isEmpty || supportsImageEditing else {
            throw ModelExecutionError.invalidOptions(
                "image editing is unsupported by this pipeline")
        }
        for image in references {
            guard image.shape.count == 4, image.shape[0] == 1, [3, 4].contains(image.shape[1]),
                image.hasNormalizedPixels
            else {
                throw ModelExecutionError.invalidOptions(
                    "reference images require normalized NCHW RGB or RGBA pixels")
            }
        }
        let (finalPrompt, metadata) = try await preparePrompt(
            prompt, references: references, parameters: parameters)
        let transfer = ExecutionTransfer((model, tokenizer))
        let source = self.source
        return try await runModelWorker(policy: policy) {
            let (model, tokenizer) = transfer.take()
            // Give each pipeline its own mutable array handle for the shared pixels.
            let images = references.map { $0.tensor.asType(.float32).reshaped($0.shape) }
            let result = try model.generateImage(
                prompt: finalPrompt, referenceImages: images, tokenizer: tokenizer,
                parameters: parameters)
            return ModelImageResult(
                source: source,
                image: try .init(tensor: result.image),
                seed: result.seed, steps: result.steps, width: result.width, height: result.height,
                guidance: result.guidance,
                metadata: result.metadata.merging(metadata) { _, new in new })
        }
    }

    private func preparePrompt(
        _ prompt: String, references: [ModelImageData], parameters: ImageGenerationParameters
    ) async throws -> (String, [String: String]) {
        let isIdeogram = model is Ideogram4ImagePipeline
        let mode = isIdeogram ? parameters.ideogramPromptExpansion : parameters.promptExpansion
        guard mode != .disabled else { return (prompt, [:]) }
        let override: ModelSource?
        if case .model(let source) = mode { override = source } else { override = nil }

        if let h3 = model as? MiniMaxH3ImagePipeline {
            try h3.validate(parameters)
            guard references.count <= 16 else {
                throw ModelExecutionError.invalidOptions(
                    "MiniMax-H3-Image accepts up to 16 reference images")
            }
            let source = override ?? MiniMaxH3ImagePromptExpansion.defaultModel
            let resources = ExecutionTransfer(h3)
            try await runModelWorker(policy: policy) { resources.take().releaseComponents() }
            let expansion = try await MiniMaxH3ImagePromptExpansion.expand(
                prompt, model: source, references: references,
                width: parameters.width, height: parameters.height,
                seed: parameters.seed, policy: policy)
            return (
                expansion.text,
                [
                    "prompt_expansion_used": "true", "prompt_expansion_model": source.name,
                    "revised_prompt": expansion.text,
                    "suggested_aspect_ratio": expansion.aspectRatio,
                ]
            )
        }

        if let flux3 = model as? Flux3ImagePipeline {
            try flux3.validate(parameters, references: references.count)
            guard !Flux3LayoutPrompt.isFinished(prompt) else { return (prompt, [:]) }
            let source = override ?? Flux3PromptExpansion.defaultModel
            let resources = ExecutionTransfer(flux3)
            try await runModelWorker(policy: policy) { resources.take().releaseComponents() }
            let expanded = try await Flux3PromptExpansion.expand(
                prompt, model: source, references: references, seed: parameters.seed, policy: policy
            )
            return (
                expanded,
                [
                    "prompt_expansion_used": "true", "prompt_expansion_model": source.name,
                    "revised_prompt": expanded,
                ]
            )
        }

        if let ernie = model as? ErnieImagePipeline {
            try ernie.validate(parameters, references: references.count)
            let tokenizerLoader = ernie.enhancerTokenizerLoader
            let assets: ErnieImagePromptExpansionAssets
            let enhancerName: String
            if let override {
                let resolved = try await ModelArtifactResolver.resolve(
                    model: override, modelPatterns: ErnieImagePromptExpansion.overridePatterns)
                assets = .init(directory: resolved.modelDirectory)
                enhancerName = override.name
            } else {
                let bundled = ErnieImagePromptExpansionAssets(
                    directory: ernie.directory, bundledOnly: true)
                if !bundled.isAvailable, let fetch = ernie.bundledEnhancer {
                    let directory = try await fetch()
                    assets = .init(directory: directory, bundledOnly: true)
                } else {
                    assets = bundled
                }
                enhancerName = source.name + "/pe"
            }
            let resources = ExecutionTransfer(ernie)
            try await runModelWorker(policy: policy) { resources.take().releaseComponents() }
            let expanded = try await ErnieImagePromptExpansion.expand(
                prompt, assets: assets, tokenizerLoader: tokenizerLoader,
                width: parameters.width, height: parameters.height, seed: parameters.seed,
                policy: policy)
            return (
                expanded,
                [
                    "prompt_expansion_used": "true", "prompt_expansion_model": enhancerName,
                    "revised_prompt": expanded,
                ]
            )
        }

        guard isIdeogram else {
            throw ModelExecutionError.unsupportedOperation("image prompt expansion")
        }
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).hasPrefix("{") else {
            return (prompt, [:])
        }
        guard let source = override else {
            throw ModelExecutionError.invalidOptions(
                "Ideogram 4 requires an explicit prompt expansion model")
        }
        do {
            let expanded = try await Ideogram4Prompting.expanded(
                prompt, model: source, width: parameters.width,
                height: parameters.height, seed: parameters.seed, policy: policy)
            return (expanded, ["prompt_expansion_used": "true", "revised_prompt": expanded])
        } catch {
            try Task.checkCancellation()
            guard parameters.ideogram4.autoJSONCaption else { throw error }
            return (
                prompt,
                [
                    "prompt_expansion_used": "false",
                    "prompt_expansion_error": String(describing: error),
                ]
            )
        }
    }
}
