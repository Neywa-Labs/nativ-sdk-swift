import Foundation
import MLX
import MLXNN

final class Ministral3EmbeddingModel: Module, EmbeddingModel {
    let model: Ministral3ModelInner
    let maximumSequenceLength: Int
    let vocabularySize: Int
    let pooling: EmbeddingPooling

    init(_ config: Ministral3Configuration, pooling: EmbeddingPooling) throws {
        guard config.hiddenSize > 0, config.hiddenLayers > 0, config.attentionHeads > 0,
            config.kvHeads > 0, config.attentionHeads % config.kvHeads == 0,
            config.resolvedHeadDimensions > 0, config.vocabularySize > 0
        else {
            throw ModelExecutionError.invalidOptions("Invalid Ministral3 embedding configuration")
        }
        model = Ministral3ModelInner(config)
        maximumSequenceLength = config.maxPositionEmbeddings ?? 128000
        vocabularySize = config.vocabularySize
        self.pooling = pooling
    }

    func embed(_ input: LMInput) throws -> EmbeddingOutput {
        let (ids, valid) = try embeddingTokens(
            input, vocabularySize: vocabularySize,
            maximumLength: maximumSequenceLength)
        var h = model.embedTokens(ids)
        let mask = embeddingAttentionMask(valid)
        let config = model.configuration
        let scale = llama4AttentionScale(
            size: ids.dim(1), offset: MLXArray(0),
            beta: config.llama4ScalingBeta,
            maxPositionEmbeddings: config.originalMaxPositionEmbeddings
        ).asType(h.dtype)
        for layer in model.layers {
            try Task.checkCancellation()
            h = layer(h, attentionScale: scale, mask: .array(mask), cache: nil)
        }
        return .pooled(normalizeEmbeddings(pooling.apply(model.norm(h), mask: valid)))
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        embeddingBackboneWeights(weights)
    }
}
