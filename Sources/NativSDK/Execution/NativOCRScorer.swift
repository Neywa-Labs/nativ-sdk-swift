import Foundation
import MLX

/// Binary consistency scores; class 1 means the OCR candidate matches the original crop.
public struct OCRScoringResult: Sendable {
    public let consistentProbability: Float
    public let inconsistentProbability: Float
    public let consistentLogit: Float
    public let inconsistentLogit: Float
    public let tokenCount: Int
}

extension LoadedModel {
    consuming func intoNativOCRScorer(policy: ModelExecutionPolicy) -> NativOCRScorer {
        NativOCRScorer(
            assets: assets, context: ExecutionTransfer(intoModelContext()), policy: policy)
    }
}

/// A loaded MiniCPM-OCR Jev model. Each request owns preparation and inference exclusively.
public actor NativOCRScorer {
    public nonisolated let assets: ResolvedModelAssets
    let context: ModelContext
    let policy: ModelExecutionPolicy
    let admission = ExecutionAdmission()

    init(
        assets: ResolvedModelAssets, context: ExecutionTransfer<ModelContext>,
        policy: ModelExecutionPolicy
    ) {
        self.assets = assets
        self.context = context.take()
        self.policy = policy
    }

    /// Scores an original crop against candidate text and a separately rendered image of that text.
    public nonisolated func score(
        original: URL, text: String, rendering: URL, processing: UserInput.Processing = .init()
    ) -> ModelOperation<OCRScoringResult> {
        score(
            original: .url(original), text: text, rendering: .url(rendering), processing: processing
        )
    }

    /// Accepts decoded images as well as file URLs. Candidate whitespace is preserved verbatim.
    public nonisolated func score(
        original: sending UserInput.Image, text: String, rendering: sending UserInput.Image,
        processing: UserInput.Processing = .init()
    ) -> ModelOperation<OCRScoringResult> {
        let input = ExecutionTransfer(
            UserInput(
                prompt: .text(text), images: [original, rendering], processing: processing))
        return ModelOperation(Task { try await self.runScore(input) })
    }

    private func runScore(_ input: ExecutionTransfer<UserInput>) async throws -> OCRScoringResult {
        try await admission.acquire()
        defer {
            Stream.defaultStream.synchronize()
            admission.release()
        }
        try Task.checkCancellation()
        let prepared = try await context.processor.prepare(input: input.take())
        guard let model = context.model as? MiniCPMOCRJev else {
            throw ModelExecutionError.unsupportedOperation("OCR scoring")
        }
        let transfer = ExecutionTransfer((model, prepared))
        return try await runModelWorker(policy: policy) {
            try Task.checkCancellation()
            let (model, input) = transfer.take()
            let logits = try model.logits(input)
            let probabilities = softmax(logits, axis: -1)
            eval(logits, probabilities)
            let values = logits.asArray(Float.self)
            let probs = probabilities.asArray(Float.self)
            try Task.checkCancellation()
            return OCRScoringResult(
                consistentProbability: probs[1], inconsistentProbability: probs[0],
                consistentLogit: values[1], inconsistentLogit: values[0],
                tokenCount: input.text.tokens.size)
        }
    }
}
