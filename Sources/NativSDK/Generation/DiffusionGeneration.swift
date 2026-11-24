import Foundation
import MLX

public struct DiffusionGenerateParameters: Sendable {
    public var maxTokens: Int
    public var blockLength: Int
    public var steps: Int
    public var threshold: Float
    public var minThreshold: Float?
    public var maxTransferPerStep: Int?
    public var eosEarlyStop: Bool
    public var skipSpecialTokens: Bool

    public init(
        maxTokens: Int = 128,
        blockLength: Int = 32,
        steps: Int = 32,
        threshold: Float = 0.95,
        minThreshold: Float? = nil,
        maxTransferPerStep: Int? = nil,
        eosEarlyStop: Bool = true,
        skipSpecialTokens: Bool = true,
    ) {
        self.maxTokens = max(0, maxTokens)
        self.blockLength = max(1, blockLength)
        self.steps = max(1, steps)
        self.threshold = threshold
        self.minThreshold = minThreshold
        self.maxTransferPerStep = maxTransferPerStep
        self.eosEarlyStop = eosEarlyStop
        self.skipSpecialTokens = skipSpecialTokens
    }

}

extension DiffusionGenerateParameters {
    func validate() throws {
        guard maxTokens >= 0, blockLength > 0, steps > 0,
            threshold.isFinite, (0 ... 1).contains(threshold),
            minThreshold.map({ $0.isFinite && (0 ... 1).contains($0) && $0 <= threshold }) ?? true,
            maxTransferPerStep.map({ $0 > 0 }) ?? true
        else {
            throw ModelExecutionError.invalidOptions(
                "invalid diffusion limits or confidence thresholds")
        }
    }
}

public protocol MaskedDiffusionLanguageModel: LanguageModel {
    var diffusionMaskTokenID: Int { get }
    var diffusionEOSTokenIDs: [Int] { get }
    var diffusionPrefersDiffusionGeneration: Bool { get }

    func diffusionLogits(_ tokens: MLXArray) -> MLXArray
}

public protocol BlockDiffusionLanguageModel: LanguageModel {
    var diffusionCanvasLength: Int { get }
    var diffusionEOSTokenIDs: [Int] { get }
    var diffusionPrefersDiffusionGeneration: Bool { get }
    var diffusionVocabularySize: Int { get }

    func diffusionMakeCache(parameters: DiffusionGenerateParameters?) -> [KVCache]
    func diffusionPrefill(input: LMInput, cache: [KVCache]) throws
    func diffusionUpdateCache(tokens: MLXArray, cache: [KVCache])
    func diffusionDecoderLogits(
        canvas: MLXArray,
        cache: [KVCache],
        selfConditioning: MLXArray?,
    )
        -> MLXArray
    func diffusionSelfConditioning(logits: MLXArray) -> MLXArray?
}

public enum GenerationError: LocalizedError {
    case diffusionUnsupported(String)

    public var errorDescription: String? {
        switch self {
        case .diffusionUnsupported(let modelType):
            "Model \(modelType) does not support diffusion generation."
        }
    }
}

extension MaskedDiffusionLanguageModel {
    public var diffusionPrefersDiffusionGeneration: Bool {
        true
    }

    public func diffusionLogits(_ tokens: MLXArray) -> MLXArray {
        callAsFunction(tokens, cache: nil)
    }
}

extension BlockDiffusionLanguageModel {
    public var diffusionPrefersDiffusionGeneration: Bool {
        true
    }
}

func runMaskedDiffusion(
    input: LMInput,
    options: ModelGenerationOptions,
    diffusionParameters: DiffusionGenerateParameters,
    model: any MaskedDiffusionLanguageModel,
    context: ModelContext,
    emit: ([Int]) throws -> Void,
)
    throws -> ModelGenerationSummary
{
    let inferenceErrors = InferenceErrorCheck(model)
    let promptStart = Date()
    let promptTokens = input.text.tokens.asArray(Int.self)
    let promptTime = max(Date().timeIntervalSince(promptStart), 1e-9)
    let generationStart = Date()
    let eosTokenIDs = resolvedEOSTokenIDs(model: model, context: context)
    let sampler = options.sampling.makeSampler()
    var tokenBlocks: [[Int]] = []
    var stopReason = ModelGenerationSummary.StopReason.length

    if diffusionParameters.maxTokens > 0, !Task.isCancelled {
        var generatedTokens: [Int] = []
        var sequence = promptTokens
        let blockLength = diffusionParameters.blockLength
        let numberOfBlocks =
            (diffusionParameters.maxTokens + blockLength - 1) / blockLength

        for _ in 0 ..< numberOfBlocks {
            if Task.isCancelled {
                stopReason = .cancelled
                break
            }

            let remaining = diffusionParameters.maxTokens - generatedTokens.count
            if remaining <= 0 {
                break
            }
            let currentBlockLength = min(blockLength, remaining)
            let blockStart = sequence.count
            sequence.append(
                contentsOf: Array(
                    repeating: model.diffusionMaskTokenID,
                    count: currentBlockLength,
                ),
            )

            try denoiseMaskedBlock(
                sequence: &sequence,
                blockStart: blockStart,
                blockLength: currentBlockLength,
                sampler: sampler,
                diffusionParameters: diffusionParameters,
                model: model,
            )

            let blockTokens = Array(sequence[blockStart ..< blockStart + currentBlockLength])
            let committed = prefixBeforeStop(
                blockTokens,
                eosTokenIDs: eosTokenIDs,
                eosEarlyStop: diffusionParameters.eosEarlyStop,
            )
            generatedTokens.append(contentsOf: committed.tokens)
            tokenBlocks.append(committed.tokens)
            try inferenceErrors.check()
            try emit(committed.tokens)

            if committed.didStop {
                stopReason = .stop
                break
            }
        }
    }

    let generationTime = max(Date().timeIntervalSince(generationStart), 1e-9)
    try Task.checkCancellation()
    return ModelGenerationSummary(
        promptTokenCount: promptTokens.count,
        generatedTokenCount: tokenBlocks.reduce(0) { $0 + $1.count },
        promptSeconds: promptTime, generationSeconds: generationTime, stopReason: stopReason)
}

func runBlockDiffusion(
    input: LMInput,
    options: ModelGenerationOptions,
    diffusionParameters: DiffusionGenerateParameters,
    model: any BlockDiffusionLanguageModel,
    context: ModelContext,
    emit: ([Int]) throws -> Void,
)
    throws -> ModelGenerationSummary
{
    let inferenceErrors = InferenceErrorCheck(model)
    let promptStart = Date()
    let promptTokens = input.text.tokens.asArray(Int.self)
    let cache = model.diffusionMakeCache(parameters: diffusionParameters)
    try model.diffusionPrefill(input: input, cache: cache)
    try inferenceErrors.check()
    let promptTime = max(Date().timeIntervalSince(promptStart), 1e-9)

    let generationStart = Date()
    let eosTokenIDs = resolvedBlockEOSTokenIDs(model: model, context: context)
    let sampler = options.sampling.makeSampler()
    let randomState =
        options.sampling.seed.map { MLXRandom.RandomState(seed: $0) } ?? MLXRandom.RandomState()

    var tokenBlocks: [[Int]] = []
    var generatedTokenCount = 0
    var stopReason = ModelGenerationSummary.StopReason.length

    while generatedTokenCount < diffusionParameters.maxTokens, !Task.isCancelled {
        let remaining = diffusionParameters.maxTokens - generatedTokenCount
        let canvasLength = min(
            remaining,
            max(1, min(diffusionParameters.blockLength, model.diffusionCanvasLength)),
        )
        var canvasTokens = withRandomState(randomState) {
            MLXRandom.randInt(0 ..< max(model.diffusionVocabularySize, 1), [canvasLength]).asArray(
                Int.self)
        }
        try denoiseBlockCanvas(
            canvasTokens: &canvasTokens,
            canvasLength: canvasLength,
            sampler: sampler,
            diffusionParameters: diffusionParameters,
            model: model,
            cache: cache,
        )

        let committed = prefixBeforeStop(
            canvasTokens,
            eosTokenIDs: eosTokenIDs,
            eosEarlyStop: diffusionParameters.eosEarlyStop,
        )
        if !committed.tokens.isEmpty {
            let committedArray = MLXArray(committed.tokens, [1, committed.tokens.count])
            model.diffusionUpdateCache(tokens: committedArray, cache: cache)
            try inferenceErrors.check()
            tokenBlocks.append(committed.tokens)
            try emit(committed.tokens)
            generatedTokenCount += committed.tokens.count
        }

        if committed.didStop {
            stopReason = .stop
            break
        }
        if committed.tokens.isEmpty {
            break
        }
    }

    let generationTime = max(Date().timeIntervalSince(generationStart), 1e-9)
    try Task.checkCancellation()
    return ModelGenerationSummary(
        promptTokenCount: promptTokens.count,
        generatedTokenCount: tokenBlocks.reduce(0) { $0 + $1.count },
        promptSeconds: promptTime, generationSeconds: generationTime, stopReason: stopReason)
}

private func denoiseBlockCanvas(
    canvasTokens: inout [Int],
    canvasLength: Int,
    sampler: any LogitSampler,
    diffusionParameters: DiffusionGenerateParameters,
    model: any BlockDiffusionLanguageModel,
    cache: [KVCache],
) throws {
    let inferenceErrors = InferenceErrorCheck(model)
    var fixed = Array(repeating: false, count: canvasLength)
    var unresolvedCount = canvasLength
    var selfConditioning: MLXArray?
    let totalSteps = max(1, diffusionParameters.steps)

    for step in 0 ..< totalSteps {
        try Task.checkCancellation()
        if unresolvedCount == 0 || Task.isCancelled {
            break
        }

        let canvas = MLXArray(canvasTokens, [1, canvasLength])
        let logits =
            model
            .diffusionDecoderLogits(
                canvas: canvas,
                cache: cache,
                selfConditioning: selfConditioning,
            )
            .asType(.float32)
        try inferenceErrors.check()
        let sampled = sampler.sample(logits: logits)
        let probabilities = softmax(logits, axis: -1, precise: true)
        let tokenProbabilities = takeAlong(
            probabilities,
            sampled[.ellipsis, .newAxis],
            axis: -1,
        )[.ellipsis, 0]

        let sampledTokens = sampled.asArray(Int.self)
        let confidence = tokenProbabilities.asArray(Float.self)
        let forceCompletion = step == totalSteps - 1
        let progress = Float(step) / Float(max(totalSteps - 1, 1))
        let minThreshold = diffusionParameters.minThreshold ?? diffusionParameters.threshold
        let stepThreshold =
            diffusionParameters.threshold - (diffusionParameters.threshold - minThreshold)
            * progress

        var candidates: [(position: Int, confidence: Float)] = []
        candidates.reserveCapacity(unresolvedCount)
        for position in 0 ..< canvasLength where !fixed[position] {
            candidates.append((position, confidence[position]))
        }
        candidates.sort { lhs, rhs in
            if lhs.confidence == rhs.confidence {
                return lhs.position < rhs.position
            }
            return lhs.confidence > rhs.confidence
        }

        let remainingSteps = max(1, totalSteps - step)
        let scheduledTransferCount = max(1, (unresolvedCount + remainingSteps - 1) / remainingSteps)
        let maxTransferCount = diffusionParameters.maxTransferPerStep.map {
            max(1, min($0, canvasLength))
        }
        let transferLimit =
            forceCompletion
            ? unresolvedCount
            : min(maxTransferCount ?? unresolvedCount, max(scheduledTransferCount, 1))

        var selected: [Int]
        if forceCompletion {
            selected = candidates.map(\.position)
        } else {
            selected =
                candidates
                .filter { $0.confidence >= stepThreshold }
                .prefix(transferLimit)
                .map(\.position)
            if selected.isEmpty, let best = candidates.first {
                selected = [best.position]
            }
        }

        for position in selected {
            canvasTokens[position] = sampledTokens[position]
            fixed[position] = true
        }
        unresolvedCount -= selected.count
        selfConditioning = model.diffusionSelfConditioning(logits: logits)
    }
}

private func denoiseMaskedBlock(
    sequence: inout [Int],
    blockStart: Int,
    blockLength: Int,
    sampler: any LogitSampler,
    diffusionParameters: DiffusionGenerateParameters,
    model: any MaskedDiffusionLanguageModel,
) throws {
    let inferenceErrors = InferenceErrorCheck(model)
    var maskedCount = sequence[blockStart ..< blockStart + blockLength]
        .count(where: { $0 == model.diffusionMaskTokenID })

    if maskedCount == 0 {
        return
    }

    let totalSteps = max(1, min(diffusionParameters.steps, blockLength))
    for step in 0 ..< totalSteps {
        try Task.checkCancellation()
        if maskedCount == 0 || Task.isCancelled {
            break
        }

        let inputArray = MLXArray(sequence, [1, sequence.count])
        let logits = model.diffusionLogits(inputArray).asType(.float32)
        try inferenceErrors.check()
        try Task.checkCancellation()
        let sampled = sampler.sample(logits: logits)
        let probabilities = softmax(logits, axis: -1, precise: true)
        let tokenProbabilities = takeAlong(
            probabilities,
            sampled[.ellipsis, .newAxis],
            axis: -1,
        )[.ellipsis, 0]

        let sampledTokens = sampled.asArray(Int.self)
        let confidence = tokenProbabilities.asArray(Float.self)
        let blockRange = blockStart ..< blockStart + blockLength
        let forceCompletion = step == totalSteps - 1
        let progress = Float(step) / Float(max(totalSteps - 1, 1))
        let minThreshold = diffusionParameters.minThreshold ?? diffusionParameters.threshold
        let stepThreshold =
            diffusionParameters.threshold - (diffusionParameters.threshold - minThreshold)
            * progress

        var candidates: [(position: Int, confidence: Float)] = []
        candidates.reserveCapacity(maskedCount)
        for position in blockRange where sequence[position] == model.diffusionMaskTokenID {
            candidates.append((position, confidence[position]))
        }
        candidates.sort { lhs, rhs in
            if lhs.confidence == rhs.confidence {
                return lhs.position < rhs.position
            }
            return lhs.confidence > rhs.confidence
        }

        let remainingSteps = max(1, totalSteps - step)
        let scheduledTransferCount = max(1, (maskedCount + remainingSteps - 1) / remainingSteps)
        let maxTransferCount = diffusionParameters.maxTransferPerStep.map {
            max(1, min($0, blockLength))
        }
        let transferLimit =
            forceCompletion
            ? maskedCount
            : min(maxTransferCount ?? maskedCount, max(scheduledTransferCount, 1))

        var selected: [Int]
        if forceCompletion {
            selected = candidates.map(\.position)
        } else {
            selected =
                candidates
                .filter { $0.confidence >= stepThreshold }
                .prefix(transferLimit)
                .map(\.position)
            if selected.isEmpty, let best = candidates.first {
                selected = [best.position]
            }
        }

        for position in selected {
            sequence[position] = sampledTokens[position]
        }
        maskedCount -= selected.count
    }
}

private func prefixBeforeStop(
    _ tokens: [Int],
    eosTokenIDs: Set<Int>,
    eosEarlyStop: Bool,
)
    -> (tokens: [Int], didStop: Bool)
{
    guard eosEarlyStop else {
        return (tokens, false)
    }
    if let stopIndex = tokens.firstIndex(where: { eosTokenIDs.contains($0) }) {
        return (Array(tokens[...stopIndex]), true)
    }
    return (tokens, false)
}

private func resolvedBlockEOSTokenIDs(
    model: any BlockDiffusionLanguageModel,
    context: ModelContext,
)
    -> Set<Int>
{
    var eosTokenIDs = Set(model.diffusionEOSTokenIDs)
    eosTokenIDs.formUnion(StopPolicy.tokenIDs(context: context))
    return eosTokenIDs
}

private func resolvedEOSTokenIDs(
    model: any MaskedDiffusionLanguageModel,
    context: ModelContext,
)
    -> Set<Int>
{
    var eosTokenIDs = Set(model.diffusionEOSTokenIDs)
    eosTokenIDs.formUnion(StopPolicy.tokenIDs(context: context))
    eosTokenIDs.remove(model.diffusionMaskTokenID)
    return eosTokenIDs
}

func runDiffusion(
    input: LMInput, options: ModelGenerationOptions,
    diffusionParameters: DiffusionGenerateParameters, context: ModelContext,
    mode: GenerationMode = .diffusion, emit: ([Int]) throws -> Void
) throws -> ModelGenerationSummary {
    try Task.checkCancellation()
    try diffusionParameters.validate()
    guard input.text.tokens.size > 0,
        input.text.tokens.ndim == 1
            || (input.text.tokens.ndim == 2 && input.text.tokens.dim(0) == 1)
    else {
        throw ModelExecutionError.invalidOptions("diffusion requires one nonempty sequence")
    }
    let strategy = try ModelGenerationStrategy.resolve(mode, model: context.languageModel)
    if strategy == .maskedDiffusion, let model = context.model as? any MaskedDiffusionLanguageModel
    {
        guard input.image == nil, input.video == nil, input.audio == nil else {
            throw ModelExecutionError.invalidOptions(
                "masked diffusion does not consume media inputs")
        }
        return try runMaskedDiffusion(
            input: input, options: options,
            diffusionParameters: diffusionParameters, model: model, context: context, emit: emit)
    }
    if strategy == .blockDiffusion, let model = context.model as? any BlockDiffusionLanguageModel {
        return try runBlockDiffusion(
            input: input, options: options,
            diffusionParameters: diffusionParameters, model: model, context: context, emit: emit)
    }
    throw GenerationError.diffusionUnsupported(String(describing: type(of: context.model)))
}
