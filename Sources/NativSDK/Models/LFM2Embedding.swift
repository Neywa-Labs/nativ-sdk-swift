import Foundation
import MLX
import MLXNN

/// Bidirectional attention and centered convolutions for LFM2 retrieval checkpoints.
final class LFM2EmbeddingModel: Module, EmbeddingModel {
    let model: LFM2ModelInner
    @ModuleInfo var projection: Linear?
    let maximumSequenceLength: Int
    let vocabularySize: Int
    let pooling: EmbeddingPooling

    init(_ config: LFM2Configuration, pooling: EmbeddingPooling, embeddingDimension: Int? = nil)
        throws
    {
        guard config.hiddenSize > 0, config.attentionHeads > 0, config.kvHeads > 0,
            config.hiddenSize % config.attentionHeads == 0,
            config.attentionHeads % config.kvHeads == 0, config.hiddenLayers > 0,
            config.vocabularySize > 0, config.convLCache > 0, config.convLCache % 2 == 1,
            config.blockDim == config.hiddenSize, config.blockMultipleOf > 0,
            embeddingDimension == nil || embeddingDimension! > 0
        else {
            throw ModelExecutionError.invalidOptions("Invalid LFM2 embedding configuration")
        }
        model = LFM2ModelInner(config)
        maximumSequenceLength = config.maxPositionEmbeddings ?? 128000
        vocabularySize = config.vocabularySize
        self.pooling = pooling
        if let embeddingDimension {
            _projection.wrappedValue = Linear(config.hiddenSize, embeddingDimension, bias: false)
        }
    }

    func embed(_ input: LMInput) throws -> EmbeddingOutput {
        let (ids, valid) = try embeddingTokens(
            input, vocabularySize: vocabularySize,
            maximumLength: maximumSequenceLength)
        let attention = embeddingAttentionMask(valid)
        var h = model.embedTokens(ids)
        for layer in model.layers {
            try Task.checkCancellation()
            let normalized = layer.operatorNorm(h)
            let mixed: MLXArray
            if let attentionLayer = layer.attention {
                mixed = attentionLayer(normalized, mask: .array(attention), cache: nil)
            } else if let conv = layer.conv {
                let parts = conv.inProj(normalized).split(parts: 3, axis: -1)
                var gated = parts[0] * parts[2]
                // ColBERT uses unmasked convolution inputs; the pooled encoder masks padding.
                if projection == nil { gated = which(valid.expandedDimensions(axis: -1), gated, 0) }
                let pad = conv.lCache / 2
                mixed = conv.outProj(
                    parts[1] * conv.conv(padded(gated, widths: [0, [pad, pad], 0])))
            } else {
                throw ModelExecutionError.invalidOptions("Missing LFM2 mixing layer")
            }
            h = h + mixed
            h = h + layer.feedForward(layer.ffnNorm(h))
        }
        h = model.embeddingNorm(h)
        if let projection { return .tokens(normalizeEmbeddings(projection(h))) }
        return .pooled(normalizeEmbeddings(pooling.apply(h, mask: valid)))
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights where !key.hasPrefix("lm_head.") {
            var name = key
            var value = value
            if name.hasPrefix("1_Dense.linear.") {
                name = "projection." + name.dropFirst("1_Dense.linear.".count)
            } else if !name.hasPrefix("projection.") && !name.hasPrefix("model.") {
                name = "model." + name
            }
            if name.contains("conv.weight"), value.ndim == 3, value.dim(-1) > value.dim(1) {
                value = value.transposed(0, 2, 1)
            }
            result[name] = value
        }
        return result
    }
}
