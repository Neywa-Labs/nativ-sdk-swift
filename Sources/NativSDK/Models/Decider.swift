import Foundation
import MLX
import MLXNN

struct DeciderConfiguration: Decodable {
    let text: Qwen35TextConfiguration
    let calibration: DeciderCalibration
    let padTokenID: Int?
    enum CodingKeys: String, CodingKey {
        case calibration = "decision_config"
        case padTokenID = "pad_token_id"
        case attentionHeads = "num_attention_heads"
    }
    init(from decoder: any Decoder) throws {
        let probe = try decoder.container(keyedBy: CodingKeys.self)
        if try probe.decodeIfPresent(Int.self, forKey: .attentionHeads) == 0 {
            throw ModelExecutionError.invalidOptions("Invalid Decider attention head count")
        }
        text = try Qwen35TextConfiguration(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        calibration = try c.decode(DeciderCalibration.self, forKey: .calibration)
        padTokenID = try c.decodeIfPresent(Int.self, forKey: .padTokenID)
        guard text.hiddenSize > 0, text.hiddenLayers > 0, text.intermediateSize > 0,
            text.attentionHeads > 0, text.kvHeads > 0, text.attentionHeads % text.kvHeads == 0,
            text.fullAttentionInterval > 0, text.fullAttentionInterval <= text.hiddenLayers,
            text.vocabularySize > 255, text.vocabularySize <= Int(Int32.max),
            text.linearNumKeyHeads > 0, text.linearNumValueHeads > 0,
            text.linearNumValueHeads % text.linearNumKeyHeads == 0,
            text.linearKeyHeadDim > 0, text.linearValueHeadDim > 0, text.linearConvKernelDim > 0,
            (text.headDim ?? 0) > 0, text.partialRotaryFactor > 0, text.partialRotaryFactor <= 1,
            text.rmsNormEps.isFinite, text.rmsNormEps > 0,
            padTokenID.map({ (0 ..< text.vocabularySize).contains($0) }) ?? true
        else {
            throw ModelExecutionError.invalidOptions("Invalid Decider backbone configuration")
        }
    }
}

/// Decider-2b's Qwen3.5 backbone, scored directly against the input label embeddings.
final class DeciderModel: Module, DecisionModel {
    @ModuleInfo(key: "language_model") var languageModel: Qwen35TextModel
    let configuration: DeciderConfiguration
    let decisionTypes: Set<DecisionType> = [.choice, .score, .bool, .noul]
    init(_ configuration: DeciderConfiguration) {
        self.configuration = configuration
        _languageModel.wrappedValue = Qwen35TextModel(configuration.text)
    }
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var mapped: [String: MLXArray] = [:]
        for (key, value) in weights { mapped[Self.canonicalPath(key)] = value }
        return languageModel.sanitize(weights: mapped)
    }
    static func canonicalPath(_ key: String) -> String {
        if key.hasPrefix("model.language_model.") {
            return "language_model.model." + key.dropFirst("model.language_model.".count)
        }
        if key.hasPrefix("language_model.") { return key }
        return "language_model." + key
    }
    func predict(
        state: DecisionState, questions: [NamedDecisionQuestion], tokenizer: any Tokenizer,
        options: DecisionPredictionOptions
    ) throws -> ModelDecisionResult {
        try DecisionValidation.validate(questions, supported: decisionTypes, options: options)
        let plan = try DeciderPredictionPlan(
            questions: questions, calibration: configuration.calibration, options: options)
        let builder = try DeciderPromptBuilder(
            tokenizer: tokenizer, state: state, maxStateTokens: options.maxStateTokens)
        let pad = configuration.padTokenID ?? tokenizer.eosTokenId ?? 0
        let labelIDs = builder.labelTokenIDs
        try validateTokens(labelIDs + [pad])
        let weights = languageModel.model.embedTokens(MLXArray(labelIDs)).T
        var probabilities: [[Double]] = []
        let inputTokenCount: Int
        if options.independent {
            let prompts = plan.rows.map(builder.prompt)
            var start = 0
            while start < prompts.count {
                try Task.checkCancellation()
                var end = start + 1
                while end < prompts.count,
                    (end - start + 1) * prompts[start ... end].map(\.count).max()! <= 8192
                {
                    end += 1
                }
                let group = Array(prompts[start ..< end])
                let positions = group.map { $0.count - 1 }
                probabilities += try score(
                    group, positions: positions, rows: Array(plan.rows[start ..< end]),
                    pad: pad, weights: weights, packed: false)
                start = end
            }
            var common = 0
            while common < prompts.map(\.count).min()!,
                prompts.allSatisfy({ $0[common] == prompts[0][common] })
            {
                common += 1
            }
            inputTokenCount = common + prompts.reduce(0) { $0 + $1.count - common }
        } else {
            let packed = builder.packed(plan.rows)
            probabilities = try score(
                [packed.tokens], positions: packed.positions, rows: plan.rows,
                pad: pad, weights: weights, packed: true)
            inputTokenCount = packed.tokens.count
        }
        return ModelDecisionResult(
            answers: try plan.answers(probabilities: probabilities),
            inputTokenCount: inputTokenCount)
    }
    private func validateTokens(_ ids: [Int]) throws {
        guard ids.allSatisfy({ (0 ..< configuration.text.vocabularySize).contains($0) }) else {
            throw ModelExecutionError.invalidOptions(
                "Decision tokenizer produced out-of-vocabulary tokens")
        }
    }
    private func score(
        _ prompts: [[Int]], positions: [Int], rows: [DeciderRow], pad: Int,
        weights: MLXArray, packed: Bool
    ) throws -> [[Double]] {
        try Task.checkCancellation()
        let length = ((prompts.map(\.count).max()! + 63) / 64) * 64
        var tokens: [Int32] = []
        for prompt in prompts {
            try validateTokens(prompt)
            tokens +=
                prompt.map(Int32.init) + Array(repeating: Int32(pad), count: length - prompt.count)
        }
        let hidden = languageModel.model(MLXArray(tokens, [prompts.count, length]))
        let slots =
            packed
            ? hidden[0, MLXArray(positions)]
            : hidden[MLXArray(Array(prompts.indices)), MLXArray(positions)]
        let logits = matmul(slots, weights)
        eval(logits)
        try Task.checkCancellation()
        return rows.enumerated().map { index, row in
            let values =
                logits[index, 0 ..< row.options.count].asType(.float32) / Float(row.temperature)
            return softmax(values).asArray(Float.self).map(Double.init)
        }
    }
}
