// Copyright © 2024 Apple Inc.

import Foundation
import MLX

/// Owned one-token-lookahead stepping, independent of output interpretation.
protocol GenerationTokenIterator: AnyObject {
    var failure: (any Error)? { get }
    var promptPrefillTime: TimeInterval { get }
    func next() -> Int?
}

protocol GenerationTokenConstraint: AnyObject {
    func apply(to logits: MLXArray) throws -> MLXArray
    func sampled(_ token: Int) throws
}

final class AutoregressiveIterator: Sequence, IteratorProtocol, GenerationTokenIterator {
    let model: any LanguageModel
    private let inferenceErrors: InferenceErrorCheck
    private(set) var state: LMOutput.State?
    private(set) var cache: [KVCache]
    private var policy: GenerationCachePolicy
    private var y: LMInput.Text
    private var processor: PenaltyProcessor
    private let sampler: any LogitSampler
    private let constraint: (any GenerationTokenConstraint)?
    private let suppressedTokens: Set<Int>
    private(set) var tokenCount = 0
    let maxTokens: Int?
    private(set) var promptPrefillTime: TimeInterval = 0
    private(set) var failure: (any Error)?

    init(
        input: LMInput, model: any LanguageModel, cache: [KVCache]? = nil,
        state: LMOutput.State? = nil, options: ModelGenerationOptions,
        samplingPrompt: MLXArray? = nil, constraint: (any GenerationTokenConstraint)? = nil
    ) throws {
        guard input.text.tokens.size > 0 else {
            throw ModelExecutionError.invalidOptions("empty prompt")
        }
        try GenerationDriver.validate(options)
        self.model = model
        self.inferenceErrors = InferenceErrorCheck(model)
        self.state = state
        self.y = input.text
        self.policy = GenerationCachePolicy(options.cache)
        self.cache =
            try cache ?? model.newCache(configuration: options.cache)
        try policy.validate(self.cache)
        self.processor = options.sampling.makeProcessor()
        self.sampler = options.sampling.makeSampler()
        self.constraint = constraint
        self.suppressedTokens = options.sampling.suppressedTokenIDs
        self.maxTokens = options.output.maxTokens
        let start = Date.timeIntervalSinceReferenceDate
        processor.prompt(samplingPrompt ?? input.text.tokens)
        try Task.checkCancellation()
        let prefill = options.prefill
        switch try model.prepare(input, cache: self.cache, state: state, prefill: prefill) {
        case .tokens(let tokens):
            guard tokens.tokens.size > 0, tokens.tokens.dim(-1) <= input.text.tokens.dim(-1) else {
                throw ModelExecutionError.invalidOptions("invalid prepared token remainder")
            }
            y = .init(tokens: try step(tokens))
            asyncEval(y.tokens)
            prefill.progress?(input.text.tokens.size, input.text.tokens.size)
        case .logits(let output):
            self.state = output.state
            y = .init(tokens: try sample(output.logits))
            asyncEval(y.tokens)
        }
        try policy.apply(&self.cache)
        try inferenceErrors.check()
        try Task.checkCancellation()
        promptPrefillTime = Date.timeIntervalSinceReferenceDate - start
    }

    private func sample(_ output: MLXArray) throws -> MLXArray {
        var logits = processor.process(logits: output[0..., -1, 0...])
        if !suppressedTokens.isEmpty {
            let vocabulary = logits.dim(-1)
            guard suppressedTokens.count < vocabulary,
                suppressedTokens.allSatisfy({ $0 >= 0 && $0 < vocabulary })
            else {
                throw ModelExecutionError.invalidOptions("invalid or fully suppressed vocabulary")
            }
            logits = putAlong(
                logits, MLXArray(suppressedTokens.sorted())[.newAxis],
                values: MLXArray(-Float.infinity), axis: -1)
        }
        if let constraint { logits = try constraint.apply(to: logits) }
        let token = sampler.sample(logits: logits)
        if let constraint { try constraint.sampled(token.item(Int.self)) }
        processor.didSample(token: token)
        return token
    }

    private func step(_ previous: LMInput.Text) throws -> MLXArray {
        let output = withPreparedCache(cache, lengths: previous.sequenceLengths) {
            model(previous[text: .newAxis], cache: cache.isEmpty ? nil : cache, state: state)
        }
        state = output.state
        try inferenceErrors.check()
        try policy.apply(&cache)
        return try sample(output.logits)
    }

    func next() -> Int? {
        guard failure == nil, maxTokens.map({ tokenCount < $0 }) ?? true else { return nil }
        do {
            try Task.checkCancellation()
            return try autoreleasepool {
                let previous = y
                let token = try step(previous)
                y = .init(tokens: token)
                asyncEval([token] + cache.flatMap { $0.state })
                if tokenCount % 256 == 0 { Memory.clearCache() }
                tokenCount += 1
                return previous.tokens.item(Int.self)
            }
        } catch {
            failure = error
            return nil
        }
    }
}
