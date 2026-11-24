import Foundation
import MLX

public enum EmbeddingInput: Sendable {
    /// Raw text, including any retrieval instruction required by the checkpoint.
    case text(String)
    /// An image file, optionally accompanied by text.
    case image(URL, prompt: String = "")
    /// A video file for visual encoders that support video frames.
    case video(URL, prompt: String = "")
    /// An audio file for encoders with an audio tower.
    case audio(URL, prompt: String = "")
    /// Combined media. Manual placeholders in the prompt control interleaving.
    case multimodal(prompt: String = "", images: [URL] = [], videos: [URL] = [], audios: [URL] = [])
}

/// Host-owned embeddings distinguish pooled encoders from token-level retrieval models.
public enum EmbeddingRepresentation: Sendable, Equatable {
    case vector([Float])
    case tokenVectors([[Float]])
}

public struct ModelEmbeddingResult: Sendable {
    /// A pooled vector [dimensions] or unpadded token vectors [tokenCount, dimensions].
    public let tensor: MaterializedArray
    public let tokenCount: Int

    private let embeddingCache = TensorConversionCache<EmbeddingRepresentation>()

    public var embeddings: EmbeddingRepresentation {
        embeddingCache.value {
            let values = tensor.asArray(Float.self)
            if tensor.ndim == 1 { return .vector(values) }
            let width = tensor.dim(1)
            return .tokenVectors(
                stride(from: 0, to: values.count, by: width).map {
                    Array(values[$0 ..< ($0 + width)])
                })
        }
    }
}

extension NativEmbeddingModel {
    public nonisolated func embed(input: EmbeddingInput) -> ModelOperation<ModelEmbeddingResult> {
        embed(input: input, dimensions: nil)
    }

    /// Select and renormalize a Matryoshka prefix for a compatible embedding encoder.
    public nonisolated func embed(input: EmbeddingInput, dimensions: Int?) -> ModelOperation<
        ModelEmbeddingResult
    > {
        let userInput: UserInput
        switch input {
        case .text(let text): userInput = UserInput(prompt: .text(text))
        case .image(let url, let prompt):
            userInput = UserInput(prompt: .text(prompt), images: [.url(url)])
        case .video(let url, let prompt):
            userInput = UserInput(prompt: .text(prompt), videos: [.url(url)])
        case .audio(let url, let prompt):
            userInput = UserInput(prompt: .text(prompt), audios: [.url(url)])
        case .multimodal(let prompt, let images, let videos, let audios):
            userInput = UserInput(
                prompt: .text(prompt), images: images.map { .url($0) },
                videos: videos.map { .url($0) }, audios: audios.map { .url($0) })
        }
        let transfer = ExecutionTransfer(RequestInput.unprepared(userInput))
        return ModelOperation(
            Task { try await self.runEmbedding(input: transfer, dimensions: dimensions) })
    }

    public nonisolated func embed(prepared: consuming sending PreparedModelInput)
        -> ModelOperation<ModelEmbeddingResult>
    {
        embed(prepared: prepared, dimensions: nil)
    }

    public nonisolated func embed(
        prepared: consuming sending PreparedModelInput, dimensions: Int?
    )
        -> ModelOperation<ModelEmbeddingResult>
    {
        let transfer = ExecutionTransfer(
            RequestInput.prepared(
                ownerID: prepared.ownerID, input: prepared.input, tools: prepared.tools))
        return ModelOperation(
            Task { try await self.runEmbedding(input: transfer, dimensions: dimensions) })
    }

    private func runEmbedding(input: ExecutionTransfer<RequestInput>, dimensions: Int?) async throws
        -> ModelEmbeddingResult
    {
        try await admission.acquire()
        defer {
            Stream.defaultStream.synchronize()
            admission.release()
        }
        let model = try context.embeddingModel
        try Task.checkCancellation()
        let value: LMInput
        switch input.take() {
        case .prepared(let owner, let prepared, let tools):
            guard owner == id else { throw ModelExecutionError.foreignPreparedInput }
            guard tools?.isEmpty != false else {
                throw ModelExecutionError.invalidOptions("Embedding inputs cannot contain tools")
            }
            value = prepared
        case .unprepared(let input):
            value = try await context.processor.prepare(input: input)
        }
        let transfer = ExecutionTransfer((model, value))
        return try await runModelWorker(policy: policy) {
            let (model, input) = transfer.take()
            let tokenCount = input.text.tokens.size
            guard
                input.text.tokens.ndim == 1
                    || (input.text.tokens.ndim == 2 && input.text.tokens.dim(0) == 1)
            else {
                throw ModelExecutionError.invalidOptions("embed requires one input sequence")
            }
            if let mask = input.text.mask {
                guard all((mask .== 0) .|| (mask .== 1)).item(Bool.self) else {
                    throw ModelExecutionError.invalidOptions("Embedding masks must be binary")
                }
            }
            let mask = input.text.mask?.flattened().asArray(Int.self)
            let activeTokenCount = mask?.reduce(0, +) ?? tokenCount
            guard (mask?.count ?? tokenCount) == tokenCount, activeTokenCount > 0 else {
                throw ModelExecutionError.invalidOptions("Invalid embedding input mask")
            }
            let result: MLXArray
            switch try model.embed(input) {
            case .pooled(let vector):
                guard vector.ndim == 2, vector.dim(0) == 1, vector.dim(1) > 0 else {
                    throw ModelExecutionError.invalidOptions("Invalid pooled embedding shape")
                }
                if let dimensions {
                    guard model is EmbeddingGemma2Model, dimensions > 0, dimensions <= vector.dim(1)
                    else {
                        throw ModelExecutionError.invalidOptions(
                            "Invalid Matryoshka embedding dimensions for this model")
                    }
                    result = normalizeEmbeddings(vector[0..., ..<dimensions])[0]
                } else {
                    result = vector[0]
                }
            case .tokens(let vectors):
                guard dimensions == nil else {
                    throw ModelExecutionError.invalidOptions(
                        "Token embeddings do not support Matryoshka dimensions")
                }
                guard vectors.ndim == 3, vectors.dim(0) == 1, vectors.dim(1) == tokenCount,
                    vectors.dim(2) > 0
                else {
                    throw ModelExecutionError.invalidOptions("Invalid token embedding shape")
                }
                let rows = vectors[0]
                if let mask, activeTokenCount < tokenCount {
                    let indices = mask.indices.filter { mask[$0] == 1 }.map(Int32.init)
                    result = rows.take(MLXArray(indices), axis: 0)
                } else {
                    result = rows
                }
            }
            return ModelEmbeddingResult(tensor: result.materialized(), tokenCount: activeTokenCount)
        }
    }
}
