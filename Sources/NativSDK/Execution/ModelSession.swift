import Foundation
import MLX

/// A conversation with its own transcript and generation defaults.
/// Both standard and draft models reuse supported text prompt prefixes.
/// Inputs contain new messages, not the full conversation. Successful turns append their
/// input and assistant response atomically; failed or cancelled turns leave history intact.
public actor ModelSession {
    /// The loaded backend. Keep the original model reference for architecture-specific access.
    public enum Model: Sendable {
        case standard(NativModel)
        case draft(NativDraftModel)
    }
    public let model: Model
    /// Proposal settings for draft sessions; nil for standard sessions.
    public let speculation: SpeculativeGenerationOptions?
    public private(set) var options: ModelGenerationOptions
    public let cacheReuse: PromptCacheReuseConfiguration
    private let state: SessionState
    private let admission = ExecutionAdmission()
    var queuedRequestCount: Int { admission.queuedCount }
    private var closed = false
    private var active: [UUID: Task<ModelGenerationSummary, Error>] = [:]

    public init(
        model: NativModel, instructions: String? = nil,
        history: sending [Chat.Message] = [], options: ModelGenerationOptions = .init(),
        cacheReuse: PromptCacheReuseConfiguration = .init()
    ) {
        self.model = .standard(model)
        self.speculation = nil
        self.options = options
        self.cacheReuse = cacheReuse
        self.state = SessionState(instructions: instructions, history: history)
    }

    /// Creates a conversation backed by a target/draft pair with paired prefix caches.
    /// Greedy sampling is the default because independent drafts require it. DFlash
    /// and DSpark sessions may opt into supported nonzero-temperature sampling.
    public init(
        model: NativDraftModel, instructions: String? = nil,
        history: sending [Chat.Message] = [],
        options: ModelGenerationOptions = .init(sampling: .init(temperature: 0)),
        speculation: SpeculativeGenerationOptions = .init(),
        cacheReuse: PromptCacheReuseConfiguration = .init()
    ) {
        self.model = .draft(model)
        self.speculation = speculation
        self.options = options
        self.cacheReuse = cacheReuse
        self.state = SessionState(instructions: instructions, history: history)
    }

    deinit { for task in active.values { task.cancel() } }

    /// Changes defaults for subsequently submitted turns. Per-request overrides do not persist.
    public func setOptions(_ options: ModelGenerationOptions) { self.options = options }

    public func respond(to prompt: String, options: ModelGenerationOptions? = nil) throws
        -> ModelGeneration
    {
        try generate(input: UserInput(prompt: prompt), options: options)
    }

    /// Appends the input's structured messages as one turn. Supply tool results with
    /// `UserInput(chat: [.tool(...)])`. Tool calls are retained but never executed here.
    /// Raw dictionary prompts and raw-token output belong on NativModel directly.
    public func generate(
        input: sending UserInput, options: ModelGenerationOptions? = nil
    ) throws -> ModelGeneration {
        guard !closed else { throw ModelExecutionError.sessionClosed }
        let options = options ?? self.options
        let transfer = ExecutionTransfer(input)
        let id = UUID()
        let (stream, continuation) = AsyncThrowingStream<ModelGenerationEvent, Error>.makeStream(
            bufferingPolicy: .bufferingOldest(max(1, options.output.bufferCapacity)))
        let model = self.model
        let state = self.state
        let admission = self.admission
        let cacheReuse = self.cacheReuse
        let speculation = self.speculation
        let ticket = admission.reserve()
        let task = Task { [weak self] in
            do {
                try await admission.acquire(ticket)
                defer { admission.release() }
                let result: ModelGenerationSummary
                switch model {
                case .standard(let model):
                    result = try await model.runSession(
                        input: transfer, state: state, options: options,
                        cacheReuse: cacheReuse, emit: continuation)
                case .draft(let model):
                    result = try await model.runSession(
                        input: transfer, state: state, options: options,
                        speculation: speculation ?? .init(), cacheReuse: cacheReuse,
                        emit: continuation)
                }
                continuation.finish()
                await self?.remove(id)
                return result
            } catch {
                continuation.finish(throwing: error)
                await self?.remove(id)
                throw error
            }
        }
        active[id] = task
        continuation.onTermination = { if case .cancelled = $0 { task.cancel() } }
        return ModelGeneration(stream: stream, task: task)
    }

    private func remove(_ id: UUID) { active.removeValue(forKey: id) }

    /// Inspects the committed transcript, including instructions, between turns. Media may
    /// contain non-Sendable tensors: return only Sendable values and never retain resources.
    public func withTranscript<R: Sendable>(
        _ body: @escaping @Sendable ([Chat.Message]) throws -> R
    ) async throws -> R {
        try await admission.acquire()
        defer { admission.release() }
        let state = self.state
        return try await model.withState(state) { try body($0.transcript) }
    }

    /// Replaces conversation history while preserving the session instructions. Cached
    /// prefixes are checked against the new prompt on the next turn, including edits.
    public func replaceTranscript(_ messages: sending [Chat.Message]) async throws {
        let transfer = ExecutionTransfer(messages)
        try await admission.acquire()
        defer { admission.release() }
        guard !closed else { throw ModelExecutionError.sessionClosed }
        let state = self.state
        try await model.withState(state) { $0.history = transfer.take() }
    }

    /// Releases KV state and checkpoints without removing messages. Call after changing
    /// weights, adapters, tokenizer, or processor settings through exclusive model access.
    public func clearCache() async throws {
        try await admission.acquire()
        defer { admission.release() }
        let state = self.state
        try await model.withState(state) { $0.resetCaches() }
    }

    /// Clears the conversation and cache, preserving instructions and generation defaults.
    public func reset() async throws {
        try await admission.acquire()
        defer { admission.release() }
        guard !closed else { throw ModelExecutionError.sessionClosed }
        let state = self.state
        try await model.withState(state) {
            $0.history = []
            $0.resetCaches()
        }
    }

    /// Cancels and joins all session requests, then releases its transcript and cache.
    /// Other sessions and directly submitted model requests are unaffected.
    public func close() async {
        closed = true
        let tasks = Array(active.values)
        for task in tasks { task.cancel() }
        for task in tasks { _ = await task.result }
        let state = self.state
        let model = self.model
        // Cleanup must also complete when the caller's task has already been cancelled.
        await Task {
            try? await model.withState(state) {
                $0.history = []
                $0.resetCaches()
            }
        }.value
        active.removeAll()
    }
}

/// All access is protected by the owning model's admission lock. Worker output is
/// read only after the worker has joined. No media or MLX tensors escape that boundary.
private final class SessionState: @unchecked Sendable {
    let instructions: String?
    var history: [Chat.Message]
    let cache = PromptCacheVault()
    let draftCache = DraftCacheVault()
    func resetCaches() {
        cache.reset()
        draftCache.reset()
    }
    var modelRevision: UInt64?
    var transcript: [Chat.Message] {
        (instructions.map { [Chat.Message.system($0)] } ?? []) + history
    }
    init(instructions: String?, history: [Chat.Message]) {
        self.instructions = instructions
        self.history = history
    }

    func prepare(_ input: UserInput, options: ModelGenerationOptions) throws
        -> ExecutionTransfer<(UserInput, [Chat.Message])>
    {
        try GenerationDriver.validate(options)
        guard options.output.format == .text else {
            throw ModelExecutionError.invalidOptions("sessions require text output")
        }
        var input = input
        let messages: [Chat.Message]
        switch input.prompt {
        case .chat(let chat): messages = chat
        case .text(let text):
            messages = [
                .user(text, images: input.images, videos: input.videos, audios: input.audios)
            ]
        case .messages:
            throw ModelExecutionError.invalidOptions("sessions require structured chat messages")
        }
        guard !messages.isEmpty else {
            throw ModelExecutionError.invalidOptions("a session turn requires at least one message")
        }
        input.prompt = .chat(transcript + messages)
        return ExecutionTransfer((input, messages))
    }

    func commit(_ messages: [Chat.Message], response: SessionResponse) {
        history.append(contentsOf: messages)
        history.append(
            .assistant(response.text, toolCalls: response.calls.isEmpty ? nil : response.calls))
    }

}

/// Written by a single generation worker, then read after that worker has joined.
private final class SessionResponse: @unchecked Sendable {
    var text = ""
    var calls: [ToolCall] = []
    func observe(_ event: ModelGenerationEvent) {
        switch event {
        case .chunk(let text): self.text += text
        case .toolCall(let call): calls.append(call)
        default: break
        }
    }
}

extension NativModel {
    fileprivate func runSession(
        input: ExecutionTransfer<UserInput>, state: SessionState,
        options: ModelGenerationOptions, cacheReuse: PromptCacheReuseConfiguration,
        emit: AsyncThrowingStream<ModelGenerationEvent, Error>.Continuation
    ) async throws -> ModelGenerationSummary {
        try await admission.acquire()
        defer {
            Stream.defaultStream.synchronize()
            admission.release()
        }
        do {
            try Task.checkCancellation()
            if state.modelRevision != modelRevision {
                state.cache.reset()
                state.modelRevision = modelRevision
            }
            let (input, messages) = try state.prepare(input.take(), options: options).take()
            try GenerationDriver.validateStrategy(
                options, model: context.languageModel, tools: input.tools)
            let prepared = try await context.processor.prepare(input: input)
            try Task.checkCancellation()
            let response = SessionResponse()
            let summary = try await GenerationDriver.run(
                input: prepared, tools: input.tools, options: options, context: context,
                wiredMemoryTicket: policy.wiredMemoryTicket, emit: emit,
                sessionCache: state.cache, cacheReuse: cacheReuse,
                observe: { response.observe($0) })
            try Task.checkCancellation()
            Stream.defaultStream.synchronize()
            // Enqueue completion before publishing history: overflow is a failed turn too.
            try GenerationDriver.emit(.completion(summary), to: emit)
            state.commit(messages, response: response)
            return summary
        } catch {
            state.cache.reset()
            throw error
        }
    }
}

extension ModelSession.Model {
    /// Transcript media and cache state stay under the owning model's admission lock.
    fileprivate func withState<R: Sendable>(
        _ state: SessionState, _ body: @escaping @Sendable (SessionState) throws -> R
    ) async throws -> R {
        switch self {
        case .standard(let model):
            return try await model.withModel { _, _ in try body(state) }
        case .draft(let model):
            return try await model.withSessionState(state, body)
        }
    }
}

extension NativDraftModel {
    fileprivate func withSessionState<R: Sendable>(
        _ state: SessionState, _ body: @Sendable (SessionState) throws -> R
    ) async throws -> R {
        try await admission.acquire()
        defer {
            Stream.defaultStream.synchronize()
            admission.release()
        }
        try Task.checkCancellation()
        return try body(state)
    }

    fileprivate func runSession(
        input: ExecutionTransfer<UserInput>, state: SessionState,
        options: ModelGenerationOptions, speculation: SpeculativeGenerationOptions,
        cacheReuse: PromptCacheReuseConfiguration,
        emit: AsyncThrowingStream<ModelGenerationEvent, Error>.Continuation
    ) async throws -> ModelGenerationSummary {
        try await admission.acquire()
        defer {
            Stream.defaultStream.synchronize()
            admission.release()
        }
        try Task.checkCancellation()
        do {
            let (input, messages) = try state.prepare(input.take(), options: options).take()
            let response = SessionResponse()
            let summary = try await runGeneration(
                input: ExecutionTransfer(input), options: options, speculation: speculation,
                emit: emit,
                observe: { response.observe($0) }, sessionCache: state.draftCache,
                cacheReuse: cacheReuse)
            try Task.checkCancellation()
            Stream.defaultStream.synchronize()
            // Completion overflow must fail without publishing the new transcript.
            try GenerationDriver.emit(.completion(summary), to: emit)
            state.commit(messages, response: response)
            return summary
        } catch {
            state.draftCache.reset()
            throw error
        }
    }
}
