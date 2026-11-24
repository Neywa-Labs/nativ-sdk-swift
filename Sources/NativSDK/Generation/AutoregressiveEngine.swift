import Foundation
import MLX

enum AutoregressiveEngine {
    static func run(
        input: LMInput, tools: [[String: any Sendable]]?, options: ModelGenerationOptions,
        context: ModelContext, wiredMemoryTicket: WiredMemoryTicket?,
        emit: AsyncThrowingStream<ModelGenerationEvent, Error>.Continuation,
        sessionCache: PromptCacheVault? = nil,
        cacheReuse: PromptCacheReuseConfiguration = .init(),
        observe: @escaping @Sendable (ModelGenerationEvent) -> Void = { _ in },
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> ModelGenerationSummary {
        let transfer = ExecutionTransfer((input, context))
        let worker = ModelWorker()
        let operation: @Sendable () async throws -> ModelGenerationSummary = {
            try await worker.run {
                let (input, context) = transfer.take()
                defer { Stream.defaultStream.synchronize() }
                try Task.checkCancellation()
                if let sessionCache {
                    return try PromptCacheEngine.run(
                        context: context, prepared: input, tools: tools, options: options,
                        configuration: cacheReuse, vault: sessionCache
                    ) {
                        observe($0)
                        try GenerationDriver.emit($0, to: emit)
                    }
                }
                let iterator = try AutoregressiveIterator(
                    input: input, model: context.languageModel, options: options)
                return try consume(
                    iterator, input: input, context: context, tools: tools, options: options
                ) {
                    observe($0)
                    try GenerationDriver.emit($0, to: emit)
                }
            }
        }
        if let wiredMemoryTicket {
            return try await WiredMemoryTicket.withWiredLimit(wiredMemoryTicket, operation)
        }
        return try await operation()
    }

    static func consume(
        _ iterator: any GenerationTokenIterator, input: LMInput, context: ModelContext,
        tools: [[String: any Sendable]]?, options: ModelGenerationOptions,
        throwOnCancellation: Bool = true, includeStopToken: Bool = false,
        additionalPrefillTime: TimeInterval = 0, didAdvance: (Int) -> Void = { _ in },
        emit: (ModelGenerationEvent) throws -> Void
    ) throws -> ModelGenerationSummary {
        let inferenceErrors = try InferenceErrorCheck(context.languageModel)
        var stopTokens = StopPolicy.tokenIDs(context: context)
        let reasoning =
            options.output.format == .text && options.output.separateReasoning
            ? context.metadata.reasoningConfig : nil
        let promptTail =
            reasoning == nil
            ? ""
            : context.tokenizer.decode(
                tokenIds: input.text.tokens.flattened()[max(0, input.text.tokens.size - 256)...]
                    .asArray(Int.self))
        let primed =
            reasoning.map {
                !$0.startDelimiter.isEmpty
                    && promptTail.trimmingCharacters(in: .whitespacesAndNewlines)
                        .hasSuffix($0.startDelimiter)
            } ?? false
        let decoder =
            options.output.format == .text
            ? ModelOutputDecoder(
                tokenizer: context.tokenizer, format: context.metadata.toolCallFormat ?? .json,
                tools: tools, stopStrings: context.metadata.stopStrings,
                toolCallPolicy: options.output.toolCallPolicy, reasoning: reasoning,
                reasoningPrimed: primed)
            : nil
        stopTokens.formUnion(decoder?.additionalStopTokenIDs ?? [])
        func forward(_ events: [ModelOutputEvent]) throws {
            for event in events {
                switch event {
                case .response(let text): try emit(.chunk(text))
                case .reasoning(let text): try emit(.reasoning(text))
                case .toolCall(let call): try emit(.toolCall(call))
                case .rejectedToolCall(let rejection): try emit(.rejectedToolCall(rejection))
                case .protocolError(let message): try emit(.protocolError(message))
                case .stop: break
                }
            }
        }
        var count = 0
        var reason = ModelGenerationSummary.StopReason.length
        var promptTime = iterator.promptPrefillTime + additionalPrefillTime
        var generationStart = Date.timeIntervalSinceReferenceDate
        var first = true
        try inferenceErrors.check()
        while let token = iterator.next() {
            try inferenceErrors.check()
            didAdvance(token)
            if Task.isCancelled {
                if throwOnCancellation { throw CancellationError() }
                reason = .cancelled
                break
            }
            if first {
                let now = Date.timeIntervalSinceReferenceDate
                promptTime += now - generationStart
                generationStart = now
                first = false
            }
            if stopTokens.contains(token) || token == context.tokenizer.unknownTokenId {
                if includeStopToken, options.output.format == .tokens {
                    count += 1
                    try emit(.token(token))
                }
                if stopTokens.contains(token), decoder?.receivesStopTokens == true {
                    try forward(decoder?.push(token) ?? [])
                }
                reason = .stop
                break
            }
            count += 1
            if options.output.format == .tokens {
                try emit(.token(token))
            } else {
                try forward(decoder?.push(token) ?? [])
                if decoder?.isStopped == true {
                    reason = .stop
                    break
                }
            }
        }
        try inferenceErrors.check()
        if let failure = iterator.failure {
            if !throwOnCancellation && failure is CancellationError {
                reason = .cancelled
            } else {
                throw failure
            }
        }
        if Task.isCancelled {
            if throwOnCancellation { throw CancellationError() }
            reason = .cancelled
        }
        if reason == .cancelled { decoder?.cancel() } else { try forward(decoder?.finish() ?? []) }
        return ModelGenerationSummary(
            promptTokenCount: input.text.tokens.size, generatedTokenCount: count,
            promptSeconds: promptTime,
            generationSeconds: Date.timeIntervalSinceReferenceDate - generationStart,
            stopReason: reason, rejectedToolCallCount: decoder?.rejectedToolCallCount ?? 0,
            recoveredToolCallCount: decoder?.recoveredToolCallCount ?? 0)
    }
}
