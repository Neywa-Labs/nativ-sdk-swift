import Foundation
import MLX

public struct ModelCapabilities: OptionSet, Sendable {
    public let rawValue: Int
    public init(rawValue: Int) { self.rawValue = rawValue }
    public static let embedding = Self(rawValue: 1 << 5)
    public static let decision = Self(rawValue: 1 << 6)
    public static let autoregressive = Self(rawValue: 1 << 0)
    public static let maskedDiffusion = Self(rawValue: 1 << 1)
    public static let blockDiffusion = Self(rawValue: 1 << 2)
    public static let audio = Self(rawValue: 1 << 3)
    /// Model opt-in; a compatible draft and snapshot-capable caches are also required.
    public static let speculative = Self(rawValue: 1 << 4)
}

/// Prepared tensors are exclusively transferred and can only run on their source model.
public struct PreparedModelInput: ~Copyable {
    public let tokenCount: Int
    let ownerID: UUID
    let input: LMInput
    let tools: [[String: any Sendable]]?

    /// Inspects the exclusively owned tensors before transferring them to generation or embedding.
    public borrowing func inspect<R: Sendable>(_ body: @Sendable (LMInput) throws -> R) rethrows
        -> R
    {
        try body(input)
    }
}

extension LoadedModel {
    consuming func intoNativModel(policy: ModelExecutionPolicy = .init()) -> NativModel {
        NativModel(
            assets: assets, metadata: metadata, context: ExecutionTransfer(intoModelContext()),
            policy: policy)
    }
}

/// A loaded, runnable model with reusable resources. Admission stays exclusive across async preparation,
/// prefill, decoding, and worker cleanup, including cancellation and failure paths.
/// Direct requests use fresh caches; ModelSession owns reusable conversation caches.
public actor NativModel {
    public nonisolated let assets: ResolvedModelAssets
    public nonisolated let metadata: ModelMetadata
    nonisolated let id = UUID()
    let context: ModelContext
    let policy: ModelExecutionPolicy
    let admission = ExecutionAdmission()
    var installedAdapter: InstalledAdapter?
    var modelRevision: UInt64 = 0
    var queuedRequestCount: Int { admission.queuedCount }

    init(
        assets: ResolvedModelAssets, metadata: ModelMetadata,
        context: ExecutionTransfer<ModelContext>, policy: ModelExecutionPolicy
    ) {
        self.assets = assets
        self.metadata = metadata
        self.context = context.take()
        self.policy = policy
    }

    /// Runs an operation with exclusive access to the model resources. Do not retain
    /// resource references or return before work using them has completed.
    /// Do not await another request on this model from inside the closure.
    public func withModel<R: Sendable>(
        _ body: @Sendable (isolated NativModel, ModelContext) async throws -> R
    ) async throws -> R {
        try await acquire()
        defer {
            Stream.defaultStream.synchronize()
            release()
        }
        try Task.checkCancellation()
        let result = try await body(self, context)
        try Task.checkCancellation()
        return result
    }

    public func prepare(input: sending UserInput) async throws -> sending PreparedModelInput {
        try await acquire()
        defer {
            Stream.defaultStream.synchronize()
            release()
        }
        try Task.checkCancellation()
        let tools = input.tools
        let prepared = try await context.processor.prepare(input: input)
        try Task.checkCancellation()
        return PreparedModelInput(
            tokenCount: prepared.text.tokens.size, ownerID: id, input: prepared, tools: tools)
    }

    public nonisolated func generate(
        input: sending UserInput, options: ModelGenerationOptions = .init()
    ) -> ModelGeneration {
        start(ExecutionTransfer(RequestInput.unprepared(input)), options: options)
    }

    public nonisolated func generate(
        prepared: consuming sending PreparedModelInput, options: ModelGenerationOptions = .init()
    ) -> ModelGeneration {
        start(
            ExecutionTransfer(
                RequestInput.prepared(
                    ownerID: prepared.ownerID, input: prepared.input, tools: prepared.tools)),
            options: options)
    }

    /// Independent requests with fresh per-sequence state. The model admits one
    /// at a time; use `responses()` or consume each stream concurrently to respect
    /// bounded buffering.
    public nonisolated func generateBatch(
        inputs: sending [UserInput], options: ModelGenerationOptions = .init()
    ) -> ModelGenerationBatch {
        ModelGenerationBatch(
            requests: inputs.map {
                start(ExecutionTransfer(RequestInput.unprepared($0)), options: options)
            })
    }

    enum RequestInput {
        case unprepared(UserInput)
        case prepared(ownerID: UUID, input: LMInput, tools: [[String: any Sendable]]?)
    }

    private nonisolated func start(
        _ input: ExecutionTransfer<RequestInput>, options: ModelGenerationOptions
    ) -> ModelGeneration {
        let (stream, continuation) = AsyncThrowingStream<ModelGenerationEvent, Error>.makeStream(
            bufferingPolicy: .bufferingOldest(max(1, options.output.bufferCapacity)))
        let task = Task {
            do {
                let summary = try await self.run(input: input, options: options, emit: continuation)
                try Self.finish(summary, continuation)
                return summary
            } catch {
                continuation.finish(throwing: error)
                throw error
            }
        }
        continuation.onTermination = { if case .cancelled = $0 { task.cancel() } }
        return ModelGeneration(stream: stream, task: task)
    }

    private nonisolated static func finish(
        _ summary: ModelGenerationSummary,
        _ continuation: AsyncThrowingStream<ModelGenerationEvent, Error>.Continuation
    ) throws {
        try Task.checkCancellation()
        try GenerationDriver.emit(.completion(summary), to: continuation)
        continuation.finish()
    }

    private func run(
        input: ExecutionTransfer<RequestInput>, options: ModelGenerationOptions,
        emit: AsyncThrowingStream<ModelGenerationEvent, Error>.Continuation
    ) async throws -> ModelGenerationSummary {
        try await acquire()
        defer {
            Stream.defaultStream.synchronize()
            release()
        }
        try Task.checkCancellation()
        try GenerationDriver.validate(options)
        let prepared: LMInput
        let tools: [[String: any Sendable]]?
        switch input.take() {
        case .unprepared(let input):
            tools = input.tools
            try GenerationDriver.validateStrategy(
                options, model: context.languageModel, tools: tools)
            prepared = try await context.processor.prepare(input: input)
        case .prepared(let ownerID, let input, let inputTools):
            guard ownerID == id else { throw ModelExecutionError.foreignPreparedInput }
            try GenerationDriver.validateStrategy(
                options, model: context.languageModel, tools: inputTools)
            prepared = input
            tools = inputTools
        }
        try Task.checkCancellation()
        return try await decode(prepared, tools: tools, options: options, emit: emit)
    }

    private func decode(
        _ input: LMInput, tools: [[String: any Sendable]]?, options: ModelGenerationOptions,
        emit: AsyncThrowingStream<ModelGenerationEvent, Error>.Continuation
    ) async throws -> ModelGenerationSummary {
        // The generation worker retains exclusive model access. Its worker is
        // joined before this method returns and the next request can be admitted.
        try await GenerationDriver.run(
            input: input, tools: tools, options: options, context: context,
            wiredMemoryTicket: policy.wiredMemoryTicket, emit: emit)
    }

    private func acquire() async throws { try await admission.acquire() }
    private func release() { admission.release() }

}

extension NativModel {
    public var capabilities: ModelCapabilities {
        var result: ModelCapabilities = []
        if context.model is any LanguageModel { result.insert(.autoregressive) }
        if context.model is any MaskedDiffusionLanguageModel { result.insert(.maskedDiffusion) }
        if context.model is any BlockDiffusionLanguageModel { result.insert(.blockDiffusion) }
        if context.model is any AudioGeneratingLanguageModel { result.insert(.audio) }
        if context.model is any SpeculativeDecodingModel { result.insert(.speculative) }
        return result
    }
}

/// Validates generation options and selects the numerical engine.
enum GenerationDriver {
    static func validate(_ options: ModelGenerationOptions) throws {
        let sampling = options.sampling
        guard sampling.temperature.isFinite && sampling.temperature >= 0 else {
            throw ModelExecutionError.invalidOptions("temperature must be finite and nonnegative")
        }
        guard sampling.topP.isFinite && (0 ... 1).contains(sampling.topP) else {
            throw ModelExecutionError.invalidOptions("topP must be finite and between 0 and 1")
        }
        guard sampling.topK >= 0 else {
            throw ModelExecutionError.invalidOptions("topK cannot be negative")
        }
        guard sampling.minP.isFinite && (0 ... 1).contains(sampling.minP) else {
            throw ModelExecutionError.invalidOptions("minP must be finite and between 0 and 1")
        }
        guard options.output.bufferCapacity > 0 else {
            throw ModelExecutionError.invalidOptions("bufferCapacity must be positive")
        }
        guard options.output.maxTokens.map({ $0 >= 0 }) ?? true else {
            throw ModelExecutionError.invalidOptions("maxTokens cannot be negative")
        }
        guard options.prefill.stepSize.map({ $0 > 0 }) ?? true else {
            throw ModelExecutionError.invalidOptions("prefill stepSize must be positive")
        }
    }

    static func validateStrategy(
        _ options: ModelGenerationOptions, model: any LanguageModel,
        tools: [[String: any Sendable]]?
    ) throws {
        try validate(options)
        guard try ModelGenerationStrategy.resolve(options.mode, model: model) != .autoregressive
        else {
            guard options.diffusion == nil else {
                throw ModelExecutionError.invalidOptions(
                    "diffusion options require a diffusion strategy")
            }
            return
        }
        if options.cache != nil {
            throw ModelExecutionError.invalidOptions(
                "owned cache configurations require autoregressive generation")
        }
        if options.output.format == .tokens {
            throw ModelExecutionError.invalidOptions(
                "raw token output requires autoregressive generation")
        }
        guard options.cache == nil,
            options.sampling.repetition == nil, options.sampling.presence == nil,
            options.sampling.frequency == nil, options.sampling.suppressedTokenIDs.isEmpty,
            tools?.isEmpty != false
        else {
            throw ModelExecutionError.invalidOptions(
                "unsupported diffusion cache, penalty, suppression or tool options")
        }
        guard options.prefill.stepSize == nil, options.prefill.progress == nil,
            options.prefill.chunking == .balanced
        else {
            throw ModelExecutionError.invalidOptions("diffusion manages its own prefill")
        }
        try
            (options.diffusion
            ?? DiffusionGenerateParameters(maxTokens: options.output.maxTokens ?? 128))
            .validate()
    }

    static func run(
        input: LMInput, tools: [[String: any Sendable]]?, options: ModelGenerationOptions,
        context: ModelContext, wiredMemoryTicket: WiredMemoryTicket?,
        emit continuation: AsyncThrowingStream<ModelGenerationEvent, Error>.Continuation,
        sessionCache: PromptCacheVault? = nil,
        cacheReuse: PromptCacheReuseConfiguration = .init(),
        observe: @escaping @Sendable (ModelGenerationEvent) -> Void = { _ in },
        isolation: isolated (any Actor)? = #isolation
    ) async throws -> ModelGenerationSummary {
        try validateStrategy(options, model: context.languageModel, tools: tools)
        if try ModelGenerationStrategy.resolve(options.mode, model: context.languageModel)
            == .autoregressive
        {
            return try await AutoregressiveEngine.run(
                input: input, tools: tools, options: options, context: context,
                wiredMemoryTicket: wiredMemoryTicket, emit: continuation,
                sessionCache: sessionCache, cacheReuse: cacheReuse, observe: observe
            )
        }
        let transfer = ExecutionTransfer((input, context))
        return try await runModelWorker(policy: .init(wiredMemoryTicket: wiredMemoryTicket)) {
            let (input, context) = transfer.take()
            let diffusion =
                options.diffusion
                ?? DiffusionGenerateParameters(maxTokens: options.output.maxTokens ?? 128)
            let info = try runDiffusion(
                input: input, options: options,
                diffusionParameters: diffusion, context: context, mode: options.mode
            ) { tokens in
                try Task.checkCancellation()
                if !tokens.isEmpty {
                    let text = context.tokenizer.decode(
                        tokenIds: tokens,
                        skipSpecialTokens: diffusion.skipSpecialTokens)
                    if !text.isEmpty {
                        observe(.chunk(text))
                        try emit(.chunk(text), to: continuation)
                    }
                }
            }
            return info
        }
    }

    static func emit(
        _ event: ModelGenerationEvent,
        to continuation: AsyncThrowingStream<ModelGenerationEvent, Error>.Continuation
    ) throws {
        switch continuation.yield(event) {
        case .enqueued: break
        case .dropped: throw ModelExecutionError.outputBufferOverflow
        case .terminated: throw CancellationError()
        @unknown default: throw CancellationError()
        }
    }
}
