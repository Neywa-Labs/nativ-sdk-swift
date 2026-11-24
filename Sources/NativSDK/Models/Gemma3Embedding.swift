import Foundation
import MLX
import MLXNN

final class Gemma3EmbeddingModel: Module, EmbeddingModel {
    let model: Gemma3Model
    @ModuleInfo var dense: [Linear]
    let maximumSequenceLength: Int
    let vocabularySize: Int
    let pooling: EmbeddingPooling

    init(_ config: Gemma3TextModelConfiguration, pooling: EmbeddingPooling) throws {
        guard config.hiddenSize > 0, config.hiddenLayers > 0, config.attentionHeads > 0,
            config.kvHeads > 0, config.attentionHeads % config.kvHeads == 0,
            config.headDim > 0, config.headDim % 2 == 0, config.slidingWindowPattern > 0,
            config.vocabularySize > 0, config.intermediateSize > 0
        else {
            throw ModelExecutionError.invalidOptions("Invalid Gemma3 embedding configuration")
        }
        model = Gemma3Model(config)
        _dense.wrappedValue = [
            Linear(config.hiddenSize, config.hiddenSize * 4, bias: false),
            Linear(config.hiddenSize * 4, config.hiddenSize, bias: false),
        ]
        maximumSequenceLength = config.maxPositionEmbeddings
        vocabularySize = config.vocabularySize
        self.pooling = pooling
    }

    func embed(_ input: LMInput) throws -> EmbeddingOutput {
        let (ids, valid) = try embeddingTokens(
            input, vocabularySize: vocabularySize,
            maximumLength: maximumSequenceLength)
        var h = model.embedTokens(ids)
        h = h * MLXArray(sqrt(Float(model.config.hiddenSize)), dtype: .bfloat16).asType(h.dtype)
        let mask = embeddingAttentionMask(valid)
        // Encoder checkpoints use full bidirectional attention, including in local-RoPE layers.
        for layer in model.layers {
            try Task.checkCancellation()
            h = layer(h, mask: .array(mask))
        }
        var pooled = pooling.apply(model.norm(h), mask: valid)
        for layer in dense { pooled = layer(pooled) }
        return .pooled(normalizeEmbeddings(pooled))
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        // Resolve sidecar order from module numbers, independent of quantized tensor shapes.
        let folders = Set(
            weights.keys.compactMap { key -> String? in
                let folder = String(key.prefix { $0 != "." })
                return folder.hasSuffix("_Dense") ? folder : nil
            }
        ).sorted { (Int($0.prefix { $0 != "_" }) ?? 0) < (Int($1.prefix { $0 != "_" }) ?? 0) }
        for (key, value) in weights where !key.hasPrefix("lm_head.") && !key.hasPrefix("head.") {
            var name = key
            if let index = folders.firstIndex(where: { key.hasPrefix($0 + ".linear.") }) {
                name = "dense.\(index)." + key.dropFirst((folders[index] + ".linear.").count)
            } else if !name.hasPrefix("dense.") && !name.hasPrefix("model.") {
                name = "model." + name
            }
            result[name] = value
        }
        return result
    }
}
