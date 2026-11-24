import MLX
import MLXNN

private let gemmaDraftPLE = LMOutput.Key<MLXArray>("gemma.draft.perLayerInputs")
private let gemmaDraftTokenTypes = LMOutput.Key<MLXArray>("gemma.draft.tokenTypes")

protocol Gemma4DraftTarget: DFlashTarget {
    var draftLanguageModel: Gemma4TextLanguageModel { get }
    func prepareGemmaDraftPrompt(_ input: LMInput) throws -> DFlashPreparedPrompt?
}

extension Gemma4DraftTarget {
    var draftHiddenSize: Int { draftLanguageModel.config.hiddenSize }
    var draftLayerCount: Int { draftLanguageModel.config.hiddenLayers }
    var draftEmbedding: Embedding { draftLanguageModel.model.embedTokens }
    var draftEmbeddingScale: Float { draftLanguageModel.model.embedScale }
    // The drafter applies its own soft cap once, after the raw target readout.
    func draftReadout(_ hidden: MLXArray) -> MLXArray {
        draftLanguageModel.lmHead.map { $0(hidden) } ?? draftEmbedding.asLinear(hidden)
    }
    func draftForward(
        _ tokens: MLXArray, cache: [KVCache], capture: [Int], verify: Bool,
        inputEmbeddings: MLXArray? = nil, recordTransaction: Bool = false,
        state: LMOutput.State? = nil, positionIds: MLXArray? = nil
    ) throws -> DFlashTargetOutput {
        let transaction =
            try verify || recordTransaction
            ? AttentionDraftTransaction(cache, length: tokens.dim(1)) : nil
        do {
            try Task.checkCancellation()
            var features: [MLXArray] = []
            let selected = Set(capture)
            let output = draftLanguageModel.model(
                tokens, inputsEmbeds: inputEmbeddings,
                cache: (transaction?.caches ?? cache).map { $0 as KVCache? },
                perLayerInputs: inputEmbeddings == nil ? nil : state?[gemmaDraftPLE],
                tokenTypeIds: inputEmbeddings == nil ? nil : state?[gemmaDraftTokenTypes],
                hiddenSink: { index, hidden in
                    if selected.contains(index) { features.append(hidden) }
                })
            var logits = draftReadout(output.hidden)
            if let cap = draftLanguageModel.finalLogitSoftcapping, cap > 0 {
                logits = gemma4LogitSoftcap(logits, MLXArray(cap))
            }
            return DFlashTargetOutput(
                logits: logits, features: concatenated(features, axis: -1),
                commit: transaction.map { value in { try value.commit($0) } }, state: .init())
        } catch {
            try transaction?.commit(0)
            throw error
        }
    }
}

extension Gemma4: Gemma4DraftTarget {
    var draftLanguageModel: Gemma4TextLanguageModel { languageModel }
    func prepareGemmaDraftPrompt(_ input: LMInput) throws -> DFlashPreparedPrompt? {
        guard input.image != nil else { return nil }
        let tokens = input.text.tokens.reshaped(1, -1)
        let (embeddings, ple) = try getInputEmbeddings(inputIds: tokens, image: input.image)
        var state = LMOutput.State()
        state[gemmaDraftPLE] = ple
        state[gemmaDraftTokenTypes] = gemma4TokenTypeIds(
            inputIds: tokens, imageTokenId: config.imageTokenId,
            videoTokenId: config.videoTokenId, audioTokenId: config.audioTokenId)
        return DFlashPreparedPrompt(embeddings: embeddings, positions: nil, state: state)
    }
}
extension Gemma4Unified: Gemma4DraftTarget {
    var draftLanguageModel: Gemma4TextLanguageModel { languageModel }
    func prepareGemmaDraftPrompt(_ input: LMInput) throws -> DFlashPreparedPrompt? {
        guard input.image != nil else { return nil }
        let tokens = input.text.tokens.reshaped(1, -1)
        let (embeddings, ple) = try getInputEmbeddings(
            inputIds: tokens,
            pixelValues: input.image?.pixels, imagePositionIds: input.image?.positionIds)
        var state = LMOutput.State()
        state[gemmaDraftPLE] = ple
        state[gemmaDraftTokenTypes] = gemma4TokenTypeIds(
            inputIds: tokens, imageTokenId: config.imageTokenId,
            videoTokenId: config.videoTokenId, audioTokenId: config.audioTokenId)
        return DFlashPreparedPrompt(embeddings: embeddings, positions: nil, state: state)
    }
}
extension Gemma4UnifiedAudioModel: Gemma4DraftTarget {
    var draftLanguageModel: Gemma4TextLanguageModel { base.languageModel }
    func prepareGemmaDraftPrompt(_ input: LMInput) throws -> DFlashPreparedPrompt? {
        try base.prepareGemmaDraftPrompt(input)
    }
}

extension Gemma4TextModel: DFlashTarget {
    var draftHiddenSize: Int { model.config.hiddenSize }
    var draftLayerCount: Int { model.config.numHiddenLayers }
    var draftEmbedding: Embedding { model.embedTokens }
    var draftEmbeddingScale: Float { model.embedScale }
    func draftReadout(_ hidden: MLXArray) -> MLXArray {
        lmHead.map { $0(hidden) } ?? draftEmbedding.asLinear(hidden)
    }
    func draftForward(
        _ tokens: MLXArray, cache: [KVCache], capture: [Int], verify: Bool,
        inputEmbeddings: MLXArray? = nil, recordTransaction: Bool = false,
        state: LMOutput.State? = nil, positionIds: MLXArray? = nil
    ) throws -> DFlashTargetOutput {
        guard inputEmbeddings == nil else {
            throw ModelExecutionError.invalidOptions("Gemma4 text drafting requires token inputs")
        }
        let transaction =
            try verify || recordTransaction
            ? AttentionDraftTransaction(cache, length: tokens.dim(1)) : nil
        do {
            try Task.checkCancellation()
            var features: [MLXArray] = []
            let selected = Set(capture)
            let hidden = model(
                tokens, cache: transaction?.caches ?? cache,
                hiddenSink: { index, h in
                    if selected.contains(index) { features.append(h) }
                })
            let cap = model.config.finalLogitSoftcapping
            let logits = tanh(draftReadout(hidden) / cap) * cap
            return DFlashTargetOutput(
                logits: logits, features: concatenated(features, axis: -1),
                commit: transaction.map { value in { try value.commit($0) } })
        } catch {
            try transaction?.commit(0)
            throw error
        }
    }
}
