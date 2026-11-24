import Foundation
import MLX
import MLXNN

/// Qwen3 embedding checkpoints use causal attention and checkpoint-selected pooling.
final class Qwen3EmbeddingModel: Module, EmbeddingModel {
    let model: Qwen3ModelInner
    let maximumSequenceLength: Int
    let vocabularySize: Int
    let pooling: EmbeddingPooling

    init(_ config: Qwen3Configuration, pooling: EmbeddingPooling) throws {
        guard config.vocabularySize > 0, config.hiddenSize > 0, config.hiddenLayers > 0,
            config.intermediateSize > 0, config.attentionHeads > 0, config.kvHeads > 0,
            config.attentionHeads % config.kvHeads == 0, config.headDim > 0,
            config.headDim % 2 == 0, config.maxPositionEmbeddings > 0
        else {
            throw ModelExecutionError.invalidOptions("Invalid Qwen3 embedding configuration")
        }
        try config.validateModelConfiguration()
        model = Qwen3ModelInner(config)
        maximumSequenceLength = config.maxPositionEmbeddings
        vocabularySize = config.vocabularySize
        self.pooling = pooling
        super.init()
    }

    func embed(_ input: LMInput) throws -> EmbeddingOutput {
        let (ids, valid) = try embeddingTokens(
            input, vocabularySize: vocabularySize,
            maximumLength: maximumSequenceLength)
        let attention = embeddingAttentionMask(valid, causal: true)
        var hidden = model.embedTokens(ids)
        for layer in model.layers {
            try Task.checkCancellation()
            hidden = layer(hidden, mask: .array(attention), cache: nil)
        }
        return .pooled(normalizeEmbeddings(pooling.apply(model.norm(hidden), mask: valid)))
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        embeddingBackboneWeights(weights)
    }
}
