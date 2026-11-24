import Foundation

public enum GenerationMode: String, Sendable {
    case automatic
    case autoregressive
    case diffusion
}

public enum ModelGenerationStrategy: Sendable {
    case autoregressive, maskedDiffusion, blockDiffusion

    public static func resolve(_ mode: GenerationMode, model: any LanguageModel) throws -> Self {
        let masked = model as? any MaskedDiffusionLanguageModel
        let block = model as? any BlockDiffusionLanguageModel
        if mode == .diffusion {
            if masked != nil { return .maskedDiffusion }
            if block != nil { return .blockDiffusion }
            throw GenerationError.diffusionUnsupported(String(describing: type(of: model)))
        }
        if mode == .automatic {
            if masked?.diffusionPrefersDiffusionGeneration == true { return .maskedDiffusion }
            if block?.diffusionPrefersDiffusionGeneration == true { return .blockDiffusion }
        }
        return .autoregressive
    }
}

public struct ModelSamplingOptions: Sendable {
    public struct Penalty: Sendable {
        public var value: Float
        public var contextSize: Int
        public init(_ value: Float, contextSize: Int = 20) {
            self.value = value
            self.contextSize = contextSize
        }
    }
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var minP: Float
    public var seed: UInt64?
    /// Masked after penalties and before sampling; invalid IDs or a fully masked vocabulary throw.
    public var suppressedTokenIDs: Set<Int>
    public var repetition: Penalty?
    public var presence: Penalty?
    public var frequency: Penalty?
    public init(
        temperature: Float = 0.6, topP: Float = 1, topK: Int = 0, minP: Float = 0,
        seed: UInt64? = nil, suppressedTokenIDs: Set<Int> = [], repetition: Penalty? = nil,
        presence: Penalty? = nil, frequency: Penalty? = nil
    ) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.minP = minP
        self.seed = seed
        self.suppressedTokenIDs = suppressedTokenIDs
        self.repetition = repetition
        self.presence = presence
        self.frequency = frequency
    }
}

public struct ModelOutputOptions: Sendable {
    /// Tokens bypass interpretation and exclude EOS/unknown IDs. Autoregressive only.
    public enum Format: Sendable { case text, tokens }
    public var format: Format
    /// Maximum generated tokens, defaulting to 8192. Pass nil for unbounded autoregressive output.
    public var maxTokens: Int?
    /// The owned event buffer is bounded. A slow consumer fails explicitly instead of losing output.
    public var bufferCapacity: Int
    /// Separate tagged reasoning into reasoning events. Framed protocols always separate it.
    public var separateReasoning: Bool
    public var toolCallPolicy: ToolCallPolicy
    public init(
        maxTokens: Int? = 8192, bufferCapacity: Int = 256, format: Format = .text,
        separateReasoning: Bool = true, toolCallPolicy: ToolCallPolicy = .init()
    ) {
        self.maxTokens = maxTokens
        self.format = format
        self.bufferCapacity = bufferCapacity
        self.toolCallPolicy = toolCallPolicy
        self.separateReasoning = separateReasoning
    }
}

public struct ModelGenerationOptions: Sendable {
    public var cache: KVCacheConfiguration?
    public var prefill: ModelPrefillOptions
    public var sampling: ModelSamplingOptions
    public var output: ModelOutputOptions
    public var mode: GenerationMode
    public var diffusion: DiffusionGenerateParameters?
    public init(
        cache: KVCacheConfiguration? = nil, prefill: ModelPrefillOptions = .init(),
        sampling: ModelSamplingOptions = .init(), output: ModelOutputOptions = .init(),
        mode: GenerationMode = .automatic, diffusion: DiffusionGenerateParameters? = nil
    ) {
        self.cache = cache
        self.prefill = prefill
        self.sampling = sampling
        self.output = output
        self.mode = mode
        self.diffusion = diffusion
    }
}

public enum ModelExecutionError: LocalizedError, Equatable {
    case unsupportedOperation(String)
    case foreignPreparedInput
    case sessionClosed
    case multipleConsumers
    case outputBufferOverflow
    case missingCompletion
    case invalidOptions(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedOperation(let operation):
            String(localized: "This model does not support the requested operation: \(operation)")
        case .foreignPreparedInput:
            String(localized: "The prepared input belongs to a different model.")
        case .sessionClosed:
            String(localized: "This conversation session is closed.")
        case .multipleConsumers:
            String(localized: "This response is already being consumed or has been consumed.")
        case .outputBufferOverflow:
            String(
                localized:
                    "Generation stopped because response events were not consumed fast enough.")
        case .missingCompletion:
            String(localized: "Generation ended without a completion result.")
        case .invalidOptions(let message):
            String(localized: "Invalid model options: \(message)")
        }
    }

    public var recoverySuggestion: String? {
        switch self {
        case .unsupportedOperation:
            String(localized: "Use a model that supports this operation.")
        case .foreignPreparedInput:
            String(localized: "Prepare the input with the same model that will process it.")
        case .sessionClosed:
            String(localized: "Create a new ModelSession to continue the conversation.")
        case .multipleConsumers:
            String(
                localized:
                    "Use either response() or one event iterator per request. Create a new request to generate again."
            )
        case .outputBufferOverflow:
            String(
                localized:
                    "Consume events promptly using response() or iteration, or increase output.bufferCapacity. Await metrics() after consuming the response."
            )
        case .missingCompletion:
            String(
                localized:
                    "Try a new request. If the problem persists, report it with the model ID and generation options."
            )
        case .invalidOptions:
            String(
                localized:
                    "Check the supplied options and the model's supported operations, then try again."
            )
        }
    }
}

public struct ModelGenerationSummary: Sendable {
    public enum StopReason: String, Sendable { case stop, length, cancelled }
    public let promptTokenCount: Int
    public let generatedTokenCount: Int
    public let promptSeconds: TimeInterval
    public let generationSeconds: TimeInterval
    public let stopReason: StopReason
    public let rejectedToolCallCount: Int
    public let recoveredToolCallCount: Int
    /// Prefix reuse statistics for an autoregressive session request.
    public var cache: ModelCacheStatistics? = nil
    /// Verification counters include proposed tokens beyond an encountered stop token.
    public var speculative: SpeculativeGenerationStatistics? = nil

    init(
        promptTokenCount: Int, generatedTokenCount: Int,
        promptSeconds: TimeInterval, generationSeconds: TimeInterval,
        stopReason: StopReason, rejectedToolCallCount: Int = 0,
        recoveredToolCallCount: Int = 0, speculative: SpeculativeGenerationStatistics? = nil
    ) {
        self.promptTokenCount = promptTokenCount
        self.generatedTokenCount = generatedTokenCount
        self.promptSeconds = promptSeconds
        self.generationSeconds = generationSeconds
        self.stopReason = stopReason
        self.rejectedToolCallCount = rejectedToolCallCount
        self.recoveredToolCallCount = recoveredToolCallCount
        self.speculative = speculative
    }

}

public enum ModelGenerationEvent: Sendable {
    /// Raw autoregressive token output; excludes EOS/unknown tokens and bypasses text parsing.
    case token(Int)
    case chunk(String)
    case reasoning(String)
    /// Framing diagnostics are delivered as data and never logged automatically.
    case protocolError(String)
    /// Emitted once after the worker has finished and released its resources.
    case completion(ModelGenerationSummary)
    case toolCall(ToolCall)
    case rejectedToolCall(RejectedToolCall)
}

/// A single-consumer request. Cancel explicitly, cancel the consuming task, or
/// release the request/iterator to stop work. `metrics()` waits for worker cleanup.
/// Keep the iterator alive only while consuming it; retaining an idle iterator
/// intentionally retains the request. Errors/cancellation finish by throwing,
/// without a completion event. Buffered events may precede that error.
public final class ModelGeneration: AsyncSequence, Sendable {
    public typealias Element = ModelGenerationEvent
    public let id = UUID()
    private let stream: AsyncThrowingStream<Element, Error>
    let task: Task<ModelGenerationSummary, Error>
    private let consumer = ConsumerClaim()

    init(
        stream: AsyncThrowingStream<Element, Error>, task: Task<ModelGenerationSummary, Error>
    ) {
        self.stream = stream
        self.task = task
    }

    deinit { task.cancel() }

    public func cancel() { task.cancel() }

    /// Waits for completion and worker cleanup without consuming the event stream.
    /// Consume events using iteration or `response()` to avoid filling the bounded buffer.
    public func metrics() async throws -> ModelGenerationSummary {
        try await withTaskCancellationHandler {
            let summary = try await task.value
            try Task.checkCancellation()
            return summary
        } onCancel: {
            task.cancel()
        }
    }

    /// Consumes the request and returns its complete response text after worker cleanup.
    /// Only `.chunk` events contribute text; reasoning, tool calls, and diagnostics are
    /// discarded. Use streaming to inspect those events. Raw token events cannot be
    /// collected as text.
    /// This is the request's single consumer and cannot follow or accompany iteration.
    /// Failures and cancellation throw rather than returning a partial response.
    public func response() async throws -> String {
        guard consumer.claim() else { throw ModelExecutionError.multipleConsumers }
        return try await withTaskCancellationHandler {
            do {
                try Task.checkCancellation()
                var iterator = AsyncIterator(stream: stream, request: self, claimed: true)
                var text = ""
                while let event = try await iterator.next() {
                    switch event {
                    case .chunk(let chunk): text += chunk
                    case .token:
                        throw ModelExecutionError.unsupportedOperation(
                            "response requires text output")
                    default: break
                    }
                }
                return text
            } catch {
                task.cancel()
                _ = try? await task.value
                throw error
            }
        } onCancel: {
            task.cancel()
        }
    }

    public struct AsyncIterator: AsyncIteratorProtocol {
        private var iterator: AsyncThrowingStream<Element, Error>.Iterator
        private let lifetime: IteratorLifetime
        private let request: ModelGeneration
        private let claimed: Bool
        fileprivate init(
            stream: AsyncThrowingStream<Element, Error>,
            request: ModelGeneration, claimed: Bool
        ) {
            iterator = stream.makeAsyncIterator()
            lifetime = IteratorLifetime(task: claimed ? request.task : nil)
            self.request = request
            self.claimed = claimed
        }
        public mutating func next() async throws -> Element? {
            guard claimed else { throw ModelExecutionError.multipleConsumers }
            let event = try await iterator.next()
            if event == nil { _ = try await request.metrics() }
            return event
        }
    }

    public func makeAsyncIterator() -> AsyncIterator {
        AsyncIterator(stream: stream, request: self, claimed: consumer.claim())
    }
}

private final class IteratorLifetime: Sendable {
    let task: Task<ModelGenerationSummary, Error>?
    init(task: Task<ModelGenerationSummary, Error>?) { self.task = task }
    deinit { task?.cancel() }
}

private final class ConsumerClaim: @unchecked Sendable {
    private let lock = NSLock()
    private var claimed = false
    func claim() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard !claimed else { return false }
        claimed = true
        return true
    }
}

/// A fixed collection of independent requests. A row failure does not cancel
/// other rows. `metrics()` joins all workers without consuming their event streams.
/// Use `responses()` to collect text safely, or consume every row concurrently.
/// This API schedules requests; it does not combine their tensors into a batch.
public final class ModelGenerationBatch: Sendable {
    public let requests: [ModelGeneration]
    init(requests: [ModelGeneration]) { self.requests = requests }
    deinit { requests.forEach { $0.cancel() } }
    public func cancel() { requests.forEach { $0.cancel() } }

    /// Consumes all requests concurrently and returns text results in input order.
    ///
    /// Each row succeeds or fails independently. Cancelling the calling task cancels
    /// all requests and waits for their workers to finish. Already completed rows
    /// may still succeed; cancelled rows contain a `CancellationError`.
    /// Like ``ModelGeneration/response()``, this collects only text chunks and claims
    /// each request's single consumer. Do not combine it with individual response
    /// collection or iteration. Start consuming promptly: requests begin eagerly,
    /// and this method cannot recover a stream that has already overflowed.
    public func responses() async -> [Result<String, Error>] {
        await withTaskCancellationHandler {
            await withTaskGroup(of: (Int, Result<String, Error>).self) { group in
                for (index, request) in requests.enumerated() {
                    group.addTask {
                        do {
                            // Swift 6.4 optimized builds can lose the row index if
                            // the await is nested inside the tuple construction.
                            let text = try await request.response()
                            return (index, .success(text))
                        } catch {
                            return (index, .failure(error))
                        }
                    }
                }
                var results = Array(
                    repeating: Result<String, Error>.success(""), count: requests.count)
                for await (index, result) in group {
                    results[index] = result
                }
                return results
            }
        } onCancel: {
            self.cancel()
        }
    }

    public func metrics() async -> [Result<ModelGenerationSummary, Error>] {
        await withTaskCancellationHandler {
            var results: [Result<ModelGenerationSummary, Error>] = []
            for request in requests {
                do { results.append(.success(try await request.metrics())) } catch {
                    results.append(.failure(error))
                }
            }
            return results
        } onCancel: {
            self.cancel()
        }
    }
}
