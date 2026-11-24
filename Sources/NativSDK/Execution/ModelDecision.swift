import Foundation
import MLX

extension NativDecisionModel {
    /// Dictionary questions run in alphabetical order; use named rows to control packed order.
    public nonisolated func predict(
        state: DecisionState, questions: [String: DecisionQuestion],
        options: DecisionPredictionOptions = .init(), media: [DecisionMedia] = []
    ) -> ModelOperation<ModelDecisionResult> {
        predict(
            state: state,
            questions: questions.keys.sorted().map {
                NamedDecisionQuestion($0, question: questions[$0]!)
            }, options: options, media: media)
    }

    public nonisolated func predict(
        state: DecisionState, questions: [NamedDecisionQuestion],
        options: DecisionPredictionOptions = .init(), media: [DecisionMedia] = []
    ) -> ModelOperation<ModelDecisionResult> {
        ModelOperation(
            Task {
                try await self.runPrediction(
                    state: state, questions: questions, options: options, media: media)
            })
    }

    private func runPrediction(
        state: DecisionState, questions: [NamedDecisionQuestion],
        options: DecisionPredictionOptions,
        media: [DecisionMedia]
    ) async throws -> ModelDecisionResult {
        try await admission.acquire()
        defer {
            Stream.defaultStream.synchronize()
            admission.release()
        }
        try Task.checkCancellation()
        let model = try context.decisionModel
        try DecisionValidation.validate(questions, supported: model.decisionTypes, options: options)
        if let multimodal = model as? any MultimodalDecisionModel {
            let prepared = try await multimodal.preparePrediction(
                state: state, questions: questions, tokenizer: context.tokenizer,
                options: options, media: media, isolation: self)
            let transfer = ExecutionTransfer(prepared)
            return try await runModelWorker(policy: policy) { try transfer.take().execute() }
        }
        guard media.isEmpty else {
            throw ModelExecutionError.unsupportedOperation(
                "This decision model does not accept media")
        }
        guard options.modelOptions == nil else {
            throw ModelExecutionError.unsupportedOperation(
                "This decision model does not accept additional options")
        }
        let transfer = ExecutionTransfer((model, context.tokenizer))
        return try await runModelWorker(policy: policy) {
            let (model, tokenizer) = transfer.take()
            return try model.predict(
                state: state, questions: questions, tokenizer: tokenizer, options: options)
        }
    }
}
