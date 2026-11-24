import MLX
import MLXNN

final class LlamaBidirectionalEmbeddingModel: Module, EmbeddingModel {
    let model: LlamaModelInner
    let maximumSequenceLength: Int
    let vocabularySize: Int
    init(_ config: LlamaConfiguration) throws {
        guard config.hiddenSize > 0, config.attentionHeads > 0, config.kvHeads > 0,
            config.attentionHeads % config.kvHeads == 0, config.hiddenLayers > 0,
            config.vocabularySize > 0
        else {
            throw ModelExecutionError.invalidOptions("Invalid bidirectional Llama configuration")
        }
        model = LlamaModelInner(config)
        maximumSequenceLength = config.maxPositionEmbeddings ?? 131072
        vocabularySize = config.vocabularySize
    }
    func embed(_ input: LMInput) throws -> EmbeddingOutput {
        let (ids, valid) = try embeddingTokens(
            input, vocabularySize: vocabularySize, maximumLength: maximumSequenceLength)
        var h = model.embedTokens(ids)
        let mask = embeddingAttentionMask(valid)
        for layer in model.layers {
            try Task.checkCancellation()
            h = layer(h, mask: .array(mask), cache: nil)
        }
        return .pooled(normalizeEmbeddings(EmbeddingPooling.mean.apply(model.norm(h), mask: valid)))
    }
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights
        where !key.contains("rotary_emb.inv_freq") && !key.contains("lm_head")
            && !key.contains("position_ids")
        {
            let key =
                key.hasPrefix("language_model.")
                ? String(key.dropFirst("language_model.".count)) : key
            result[key.hasPrefix("model.") ? key : "model." + key] = value
        }
        return result
    }
}
