import Foundation
import MLX
import MLXNN

final class Qwen3VLEmbeddingModel: Module, EmbeddingModel {
    let backbone: Qwen3VL
    let maximumSequenceLength: Int
    let normalize: Bool
    init(_ config: Qwen3VLConfiguration, normalize: Bool) throws {
        backbone = Qwen3VL(config)
        backbone.removeLanguageHead()
        maximumSequenceLength = config.textConfiguration.maxPositionEmbeddings
        self.normalize = normalize
    }
    func embed(_ input: LMInput) throws -> EmbeddingOutput {
        try validateQwenEmbeddingMedia(input)
        let (ids, valid) = try embeddingTokens(
            input, vocabularySize: backbone.config.textConfiguration.vocabSize,
            maximumLength: maximumSequenceLength, allowsMedia: true)
        let h = try backbone.embeddingHiddenStates(input, ids: ids, valid: valid)
        let pooled = EmbeddingPooling.lasttoken.apply(h, mask: valid)
        return .pooled(normalize ? normalizeEmbeddings(pooled) : pooled)
    }
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let unwrapped = Dictionary(
            uniqueKeysWithValues: weights.map {
                ($0.key.hasPrefix("backbone.") ? String($0.key.dropFirst(9)) : $0.key, $0.value)
            })
        return Dictionary(
            uniqueKeysWithValues: backbone.sanitize(weights: unwrapped)
                .filter { !$0.key.contains("lm_head.") }.map { ("backbone." + $0.key, $0.value) })
    }
}

final class ColQwen25EmbeddingModel: Module, EmbeddingModel {
    let backbone: Qwen25VL
    @ModuleInfo(key: "custom_text_proj") var projection: Linear
    let maximumSequenceLength: Int
    init(_ config: Qwen25VLConfiguration, embeddingDimension: Int) throws {
        try config.validateModelConfiguration()
        guard embeddingDimension > 0 else {
            throw ModelExecutionError.invalidOptions("Invalid embedding dimension")
        }
        backbone = Qwen25VL(config)
        backbone.removeLanguageHead()
        _projection.wrappedValue = Linear(config.textConfiguration.hiddenSize, embeddingDimension)
        maximumSequenceLength = config.textConfiguration.maxPositionEmbeddings
    }
    func embed(_ input: LMInput) throws -> EmbeddingOutput {
        try validateQwenEmbeddingMedia(input)
        let (ids, valid) = try embeddingTokens(
            input, vocabularySize: backbone.config.textConfiguration.vocabularySize,
            maximumLength: maximumSequenceLength, allowsMedia: true)
        let h = try backbone.embeddingHiddenStates(input, ids: ids, valid: valid)
        return .tokens(normalizeEmbeddings(projection(h)) * valid.expandedDimensions(axis: -1))
    }
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var remapped: [String: MLXArray] = [:]
        var head: [String: MLXArray] = [:]
        for (key, value) in weights {
            var key = key
            if key.hasPrefix("embedding_proj_layer.") {
                key = key.replacingOccurrences(
                    of: "embedding_proj_layer.", with: "custom_text_proj.")
            }
            if key.hasPrefix("custom_text_proj.") {
                head[key] = value
                continue
            }
            if key.hasPrefix("backbone.") { key = String(key.dropFirst(9)) }
            if key.hasPrefix("vlm.model.language_model.") {
                key = "model." + key.dropFirst("vlm.model.language_model.".count)
            } else if key.hasPrefix("vlm.model.visual.") {
                key = "visual." + key.dropFirst("vlm.model.visual.".count)
            } else if key.hasPrefix("vlm.") {
                key = String(key.dropFirst(4))
            }
            if !key.contains("lm_head.") {
                let config = backbone.config.visionConfiguration
                let expected = [
                    config.hiddenSize, config.temporalPatchSize, config.patchSize, config.patchSize,
                    config.inChannels,
                ]
                remapped[key] =
                    key.hasSuffix("patch_embed.proj.weight") && value.ndim == 5
                        && value.shape != expected
                    ? value.transposed(0, 2, 3, 4, 1) : value
            }
        }
        for (key, value) in backbone.sanitize(weights: remapped) { head["backbone." + key] = value }
        return head
    }
}

private func validateQwenEmbeddingMedia(_ input: LMInput) throws {
    let media = [
        (input.image?.pixels, input.image?.frames), (input.video?.pixels, input.video?.frames),
    ]
    for (pixels, frames) in media {
        guard let pixels else { continue }
        guard input.text.tokens.ndim == 2, input.text.tokens.dim(0) == 1,
            pixels.ndim == 2, let frames, !frames.isEmpty,
            frames.allSatisfy({ $0.t > 0 && $0.h > 0 && $0.w > 0 }),
            frames.reduce(0, { $0 + $1.product }) == pixels.dim(0)
        else {
            throw ModelExecutionError.invalidOptions(
                "Qwen visual embeddings require valid patch tensors and frame grids")
        }
    }
}
