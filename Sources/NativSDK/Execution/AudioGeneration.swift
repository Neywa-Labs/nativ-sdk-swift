import Foundation
import MLX

public enum AudioGenerationError: LocalizedError {
    case unsupportedModel(String)
    case batchUnsupported

    public var errorDescription: String? {
        switch self {
        case .unsupportedModel(let modelType):
            "Model \(modelType) does not support audio output generation."
        case .batchUnsupported:
            "Audio output currently supports a single generated sequence."
        }
    }
}

/// Model-owned parameters for audio generation. Each model validates its own type.
public protocol AudioGenerationOptions: Sendable {}

public struct ModelAudioOptions: AudioGenerationOptions {
    public var speaker: String
    public var maxTokens: Int
    public var chunkSize: Int
    public var leftContextSize: Int
    public init(
        speaker: String = "ethan", maxTokens: Int = 2048, chunkSize: Int = 300,
        leftContextSize: Int = 25
    ) {
        self.speaker = speaker
        self.maxTokens = maxTokens
        self.chunkSize = chunkSize
        self.leftContextSize = leftContextSize
    }
}

/// Audio synthesis consumes a committed thinker sequence on the same worker.
public protocol AudioGeneratingLanguageModel: LanguageModel {
    var outputAudioSampleRate: Int { get }
    func synthesizeAudio(input: LMInput, generatedTokens: [Int], options: ModelAudioOptions) throws
        -> MLXArray
}

/// Models whose text tokens and audio frames must share one autoregressive pass.
protocol JointAudioGeneratingLanguageModel: LanguageModel {
    func validateAudioOptions(_ options: (any AudioGenerationOptions)?) throws
    func generateJointAudio(
        input: LMInput, options: ModelGenerationOptions,
        audio: (any AudioGenerationOptions)?, tokenizer: any Tokenizer
    ) throws -> ModelAudioResult
}

public struct ModelAudioResult: Sendable {
    /// Evaluated samples in the audio decoder's native shape and dtype.
    public let waveform: MaterializedArray
    private let sampleCache = TensorConversionCache<[Float]>()

    public var samples: [Float] { sampleCache.value { waveform.asArray(Float.self) } }
    public var shape: [Int] { waveform.shape }
    public let sampleRate: Int
    public let tokenIDs: [Int]
    public let text: String
    public let summary: ModelGenerationSummary
}

extension Qwen3OmniMoE: AudioGeneratingLanguageModel {
    public var outputAudioSampleRate: Int { 24000 }
    public func synthesizeAudio(input: LMInput, generatedTokens: [Int], options: ModelAudioOptions)
        throws -> MLXArray
    {
        let tokens = input.text.tokens.asArray(Int.self) + generatedTokens
        let sequences = MLXArray(tokens, [1, tokens.count])
        let trace = try extractThinkerTrace(input: input, sequences: sequences)
        try Task.checkCancellation()
        return try generateAudioFromThinker(
            sequences: trace.sequences,
            thinkerHiddenStates: trace.hiddenStates, thinkerInputEmbeddings: trace.inputEmbeddings,
            speaker: options.speaker, maxNewTokens: options.maxTokens, sampler: ArgMaxSampler(),
            chunkSize: options.chunkSize, leftContextSize: options.leftContextSize)
    }
}

enum AudioGenerationEngine {
    static func validate(
        _ options: ModelGenerationOptions, audio: (any AudioGenerationOptions)?,
        model: any LanguageModel
    ) throws {
        guard
            model is any AudioGeneratingLanguageModel
                || model is any JointAudioGeneratingLanguageModel
        else {
            throw AudioGenerationError.unsupportedModel(String(describing: type(of: model)))
        }
        try GenerationDriver.validate(options)
        guard try ModelGenerationStrategy.resolve(options.mode, model: model) == .autoregressive,
            options.diffusion == nil
        else {
            throw ModelExecutionError.invalidOptions(
                "audio requires autoregressive generation")
        }
        if let joint = model as? any JointAudioGeneratingLanguageModel {
            try joint.validateAudioOptions(audio)
        } else {
            guard audio == nil || audio is ModelAudioOptions else {
                throw ModelExecutionError.invalidOptions(
                    "This audio model requires ModelAudioOptions")
            }
            let options = audio as? ModelAudioOptions ?? .init()
            guard options.maxTokens > 0, options.chunkSize > 0, options.leftContextSize >= 0 else {
                throw ModelExecutionError.invalidOptions("audio requires positive limits")
            }
        }
    }

    static func run(
        input: LMInput, options: ModelGenerationOptions, audio: (any AudioGenerationOptions)?,
        context: ModelContext, includeStopToken: Bool = true
    ) throws -> ModelAudioResult {
        try validate(options, audio: audio, model: context.languageModel)
        guard
            input.text.tokens.ndim == 1
                || (input.text.tokens.ndim == 2 && input.text.tokens.dim(0) == 1)
        else {
            throw AudioGenerationError.batchUnsupported
        }
        if let model = context.model as? any JointAudioGeneratingLanguageModel {
            return try model.generateJointAudio(
                input: input, options: options, audio: audio, tokenizer: context.tokenizer)
        }
        let model = context.model as! any AudioGeneratingLanguageModel
        var tokenOptions = options
        tokenOptions.output.format = .tokens
        let iterator = try AutoregressiveIterator(
            input: input, model: model, options: tokenOptions)
        var tokens: [Int] = []
        let summary = try AutoregressiveEngine.consume(
            iterator, input: input, context: context,
            tools: nil, options: tokenOptions, includeStopToken: includeStopToken
        ) {
            if case .token(let token) = $0 { tokens.append(token) }
        }
        try Task.checkCancellation()
        let waveform = try model.synthesizeAudio(
            input: input, generatedTokens: tokens, options: audio as? ModelAudioOptions ?? .init())
        let snapshot = waveform.materialized()
        try Task.checkCancellation()
        return ModelAudioResult(
            waveform: snapshot,
            sampleRate: model.outputAudioSampleRate, tokenIDs: tokens,
            text: context.tokenizer.decode(tokenIds: tokens), summary: summary)
    }
}

extension NativModel {
    public nonisolated func generateAudio(
        input: sending UserInput,
        options: ModelGenerationOptions = .init(), audio: ModelAudioOptions? = nil
    ) -> ModelOperation<ModelAudioResult> {
        let transfer = ExecutionTransfer(RequestInput.unprepared(input))
        return ModelOperation(
            Task { try await self.runAudio(input: transfer, options: options, audio: audio) })
    }

    public nonisolated func generateAudio(
        prepared: consuming sending PreparedModelInput,
        options: ModelGenerationOptions = .init(), audio: ModelAudioOptions? = nil
    ) -> ModelOperation<ModelAudioResult> {
        let transfer = ExecutionTransfer(
            RequestInput.prepared(
                ownerID: prepared.ownerID, input: prepared.input, tools: prepared.tools))
        return ModelOperation(
            Task { try await self.runAudio(input: transfer, options: options, audio: audio) })
    }

    /// Pass the model family's own options without adding family fields to the common options.
    public nonisolated func generateAudio<Options: AudioGenerationOptions>(
        input: sending UserInput, options: ModelGenerationOptions = .init(), audio: Options
    ) -> ModelOperation<ModelAudioResult> {
        let transfer = ExecutionTransfer(RequestInput.unprepared(input))
        return ModelOperation(
            Task { try await self.runAudio(input: transfer, options: options, audio: audio) })
    }

    public nonisolated func generateAudio<Options: AudioGenerationOptions>(
        prepared: consuming sending PreparedModelInput, options: ModelGenerationOptions = .init(),
        audio: Options
    ) -> ModelOperation<ModelAudioResult> {
        let transfer = ExecutionTransfer(
            RequestInput.prepared(
                ownerID: prepared.ownerID, input: prepared.input, tools: prepared.tools))
        return ModelOperation(
            Task { try await self.runAudio(input: transfer, options: options, audio: audio) })
    }

    private func runAudio(
        input: ExecutionTransfer<RequestInput>, options: ModelGenerationOptions,
        audio: (any AudioGenerationOptions)?
    ) async throws -> ModelAudioResult {
        try await admission.acquire()
        defer {
            Stream.defaultStream.synchronize()
            admission.release()
        }
        try Task.checkCancellation()
        try AudioGenerationEngine.validate(options, audio: audio, model: context.languageModel)
        let prepared: LMInput
        switch input.take() {
        case .unprepared(let userInput):
            guard userInput.tools?.isEmpty != false else {
                throw ModelExecutionError.invalidOptions("audio tool execution is unsupported")
            }
            prepared = try await context.processor.prepare(input: userInput)
        case .prepared(let ownerID, let value, let tools):
            guard ownerID == id else { throw ModelExecutionError.foreignPreparedInput }
            guard tools?.isEmpty != false else {
                throw ModelExecutionError.invalidOptions("audio tool execution is unsupported")
            }
            prepared = value
        }
        let transfer = ExecutionTransfer((prepared, context))
        return try await runModelWorker(policy: policy) {
            let (input, context) = transfer.take()
            return try AudioGenerationEngine.run(
                input: input, options: options, audio: audio, context: context)
        }
    }
}
