import Foundation
import MLX
import MLXNN

public struct MistralLarge3Configuration: Codable, Sendable {
    public let textConfiguration: DeepseekV3Configuration
    public let visionConfiguration: PixtralVisionConfiguration
    public let modelType: String
    private let _imageTokenId: Int?
    public var imageTokenId: Int { _imageTokenId ?? 10 }
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int { _imageTokenIndex ?? imageTokenId }
    private let _visionFeatureLayer: Int?
    public var visionFeatureLayer: Int { _visionFeatureLayer ?? -1 }
    private let _spatialMergeSize: Int?
    public var spatialMergeSize: Int { _spatialMergeSize ?? 2 }
    private let _multimodalProjectorBias: Bool?
    public var multimodalProjectorBias: Bool { _multimodalProjectorBias ?? false }
    private let _vocabSize: Int?
    public var vocabSize: Int { _vocabSize ?? textConfiguration.vocabSize }
    private let _eosTokenId: IntOrIntArray?
    public var eosTokenId: [Int]? { _eosTokenId?.values }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _imageTokenId = "image_token_id"
        case _imageTokenIndex = "image_token_index"
        case _visionFeatureLayer = "vision_feature_layer"
        case _spatialMergeSize = "spatial_merge_size"
        case _multimodalProjectorBias = "multimodal_projector_bias"
        case _vocabSize = "vocab_size"
        case _eosTokenId = "eos_token_id"
    }
}

public class MistralLarge3: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: PixtralVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: DeepseekV3Model
    @ModuleInfo(key: "multi_modal_projector") private var multiModalProjector:
        Mistral3MultiModalProjector

    public let config: MistralLarge3Configuration

    public var vocabularySize: Int { config.vocabSize }
    public var toolCallFormat: ToolCallFormat? { .mistral }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: MistralLarge3Configuration) {
        self.config = config
        self._visionTower.wrappedValue = PixtralVision.VisionModel(config.visionConfiguration)
        self._languageModel.wrappedValue = DeepseekV3Model(config.textConfiguration)
        self._multiModalProjector.wrappedValue = Mistral3MultiModalProjector(
            visionHiddenSize: config.visionConfiguration.hiddenSize,
            textHiddenSize: config.textConfiguration.hiddenSize,
            patchSize: config.visionConfiguration.patchSize,
            spatialMergeSize: config.spatialMergeSize,
            bias: config.multimodalProjectorBias)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }

        guard let pixelValues = input.image?.pixels else {
            let total = tokens.dim(1)
            let processed = try prefill.forEachChunk(total: total) { range in
                _ = languageModel(tokens[0..., range], cache: cache)
                asyncEval(cache)
            }
            if processed > 0 { eval(cache) }
            prefill.progress?(total, total)
            return .logits(.init(logits: languageModel(tokens[0..., processed...], cache: cache)))
        }

        let imageSizes: [(Int, Int)] =
            input.image?.frames?.map { ($0.h, $0.w) }
            ?? [(config.visionConfiguration.imageSize, config.visionConfiguration.imageSize)]
        let embeddings = try inputEmbeddings(
            inputIds: tokens, pixelValues: pixelValues, imageSizes: imageSizes)

        let total = embeddings.dim(1)
        let processed = try prefill.forEachChunk(total: total) { range in
            _ = languageModel(
                tokens[0..., range], cache: cache, inputsEmbeds: embeddings[0..., range, 0...])
            asyncEval(cache)
        }
        if processed > 0 { eval(cache) }
        prefill.progress?(total, total)
        return .logits(
            .init(
                logits: languageModel(
                    tokens[0..., processed...], cache: cache,
                    inputsEmbeds: embeddings[0..., processed..., 0...])))
    }

    func inputEmbeddings(
        inputIds: MLXArray, pixelValues: MLXArray, imageSizes: [(Int, Int)]
    ) throws -> MLXArray {
        var pixelValues = pixelValues
        if pixelValues.ndim == 3 { pixelValues = pixelValues.expandedDimensions(axis: 0) }

        let alignment = config.visionConfiguration.patchSize * config.spatialMergeSize
        guard inputIds.ndim == 2, inputIds.dim(0) == 1,
            pixelValues.ndim == 4, pixelValues.dim(0) == 1, pixelValues.dim(1) == 3,
            alignment > 0, pixelValues.dim(2) > 0, pixelValues.dim(3) > 0,
            pixelValues.dim(2).isMultiple(of: alignment),
            pixelValues.dim(3).isMultiple(of: alignment),
            imageSizes.count == 1, imageSizes[0].0 == pixelValues.dim(2),
            imageSizes[0].1 == pixelValues.dim(3)
        else {
            throw VLMError.processing(
                "Mistral Large 3 requires one image aligned to its merged-patch grid.")
        }
        let (_, _, hiddenStates) = visionTower(
            pixelValues.transposed(0, 2, 3, 1), outputHiddenStates: true)
        guard let hiddenStates else {
            throw VLMError.processing("The vision tower did not return hidden states.")
        }
        let layer =
            config.visionFeatureLayer < 0
            ? hiddenStates.count + config.visionFeatureLayer : config.visionFeatureLayer

        guard hiddenStates.indices.contains(layer) else {
            throw VLMError.processing("vision_feature_layer is outside the vision tower.")
        }
        let imageFeatures = multiModalProjector(hiddenStates[layer], imageSizes: imageSizes)
        return try merge(
            imageFeatures: imageFeatures,
            inputsEmbeds: languageModel.model.embedTokens(inputIds),
            inputIds: inputIds)
    }

    private func merge(imageFeatures: MLXArray, inputsEmbeds: MLXArray, inputIds: MLXArray)
        throws -> MLXArray
    {
        guard inputIds.ndim == 2, inputIds.dim(0) == 1 else {
            throw VLMError.processing("Image input requires a single sequence.")
        }
        let patches = imageFeatures.dim(1)
        let ids: [Int32] = inputIds[0].asArray(Int32.self)
        let positions = ids.enumerated().compactMap {
            $1 == Int32(config.imageTokenIndex) ? $0 : nil
        }
        guard positions.count == patches, patches > 0 else {
            throw VLMError.processing(
                "the prompt holds \(positions.count) image tokens and the tower produced "
                    + "\(patches) patches; the processor must emit one token per patch")
        }

        var segments: [MLXArray] = []
        var start = 0
        for position in positions {
            segments.append(inputsEmbeds[0..., start ..< position, 0...])
            start = position + 1
        }
        let perPatch = MLX.split(imageFeatures, indices: Array(1 ..< patches), axis: 1)
        var merged: [MLXArray] = []
        for (text, patch) in zip(segments, perPatch) {
            merged.append(text)
            merged.append(patch)
        }
        merged.append(inputsEmbeds[0..., start..., 0...])
        return MLX.concatenated(merged, axis: 1)
    }
}

extension MistralLarge3 {
    /// Mistral publishes this model in its own layout, which names almost nothing the way
    /// the three component modules read it. A checkpoint already in the converted layout
    /// is left alone; otherwise every tensor is routed to one of the three and renamed.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let alreadyConverted = weights.keys.contains {
            $0.hasPrefix("language_model.") || $0.hasPrefix("vision_tower.")
                || $0.hasPrefix("multi_modal_projector.")
        }
        if alreadyConverted {
            let prefix = "language_model."
            let language = weights.filter { $0.key.hasPrefix(prefix) }
                .reduce(into: [String: MLXArray]()) {
                    $0[String($1.key.dropFirst(prefix.count))] = $1.value
                }
            var result = weights.filter { !$0.key.hasPrefix(prefix) }
            for (key, value) in languageModel.sanitize(weights: restoreCombinedProjection(language))
            {
                result[prefix + key] = value
            }
            return result
        }

        var language: [String: MLXArray] = [:]
        var vision: [String: MLXArray] = [:]
        var projector: [String: MLXArray] = [:]
        var unmapped: [String: MLXArray] = [:]

        for (key, value) in weights {
            if key == "pre_mm_projector_norm.weight" {
                projector["multi_modal_projector.norm.weight"] = value
            } else if key.hasPrefix("patch_merger.") {
                projector["multi_modal_projector." + key] = value
            } else if key.hasPrefix("vision_language_adapter.") {
                let parts = key.split(separator: ".", maxSplits: 2).map(String.init)
                guard parts.count == 3,
                    let name = ["w_in": "linear_1", "w_out": "linear_2"][parts[1]]
                else {
                    unmapped[key] = value
                    continue
                }
                projector["multi_modal_projector.\(name).\(parts[2])"] = value
            } else if key.hasPrefix("vision_encoder.") {
                var renamed = String(key.dropFirst("vision_encoder.".count))
                for (source, destination) in Self.visionRenames {
                    renamed = renamed.replacingOccurrences(of: source, with: destination)
                }
                vision["vision_model." + renamed] = value
            } else if let mapped = Self.languageKey(key) {
                language[mapped] = value
            } else {
                unmapped[key] = value
            }
        }

        // The router's correction bias is optional in the source and required by the
        // module, which reads it on every mixture layer.
        let text = config.textConfiguration
        if let experts = text.nRoutedExperts {
            for layer in text.firstKDenseReplace ..< text.numHiddenLayers
            where layer % text.moeLayerFreq == 0 {
                let key = "model.layers.\(layer).mlp.gate.e_score_correction_bias"
                if language[key] == nil {
                    language[key] = MLXArray.zeros([experts])
                }
            }
        }

        var out = unmapped
        for (key, value) in languageModel.sanitize(weights: language) {
            out["language_model." + key] = value
        }
        for (key, value) in visionTower.sanitize(weights: vision) {
            out["vision_tower." + key] = value
        }
        out.merge(projector) { _, new in new }
        return out
    }

    /// Converted latent-attention checkpoints store the key and value projections separately.
    /// Reconstruct the combined projection consumed by the decoder.
    private func restoreCombinedProjection(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var result = weights
        for layer in 0 ..< config.textConfiguration.numHiddenLayers {
            let prefix = "model.layers.\(layer).self_attn."
            func unpack(_ name: String, inputDimensions: Int) -> MLXArray? {
                let key = prefix + name
                guard let weight = result.removeValue(forKey: key + ".weight") else { return nil }
                guard let scales = result.removeValue(forKey: key + ".scales") else {
                    return weight
                }
                let biases = result.removeValue(forKey: key + ".biases")
                let bits = weight.dim(-1) * 32 / inputDimensions
                let groupSize = inputDimensions / scales.dim(-1)
                return dequantized(
                    weight, scales: scales, biases: biases, groupSize: groupSize, bits: bits)
            }
            let text = config.textConfiguration
            guard result[prefix + "embed_q.weight"] != nil,
                result[prefix + "unembed_out.weight"] != nil
            else { continue }
            if let key = unpack("embed_q", inputDimensions: text.qkNopeHeadDim),
                let value = unpack("unembed_out", inputDimensions: text.kvLoraRank)
            {
                result[prefix + "kv_b_proj.weight"] = concatenated(
                    [key.transposed(0, 2, 1), value], axis: 1
                )
                .reshaped(-1, text.kvLoraRank)
            }
        }
        return result
    }

    private static let visionRenames: [(String, String)] = [
        (".attention.wq.", ".attention.q_proj."),
        (".attention.wk.", ".attention.k_proj."),
        (".attention.wv.", ".attention.v_proj."),
        (".attention.wo.", ".attention.o_proj."),
        (".feed_forward.w1.", ".feed_forward.gate_proj."),
        (".feed_forward.w2.", ".feed_forward.down_proj."),
        (".feed_forward.w3.", ".feed_forward.up_proj."),
    ]

    private static let expertProjections = [
        "w1": "gate_proj", "w2": "down_proj", "w3": "up_proj",
    ]

    private static let attentionRenames = [
        "attention_norm": "input_layernorm",
        "ffn_norm": "post_attention_layernorm",
        "attention.wq_a": "self_attn.q_a_proj",
        "attention.q_a_norm": "self_attn.q_a_layernorm",
        "attention.wq_b": "self_attn.q_b_proj",
        "attention.wkv_a_with_mqa": "self_attn.kv_a_proj_with_mqa",
        "attention.kv_a_norm": "self_attn.kv_a_layernorm",
        "attention.wkv_b": "self_attn.kv_b_proj",
        "attention.wo": "self_attn.o_proj",
        "gate": "mlp.gate",
    ]

    /// The tensor's suffix decides nothing but how much of the name to map. A source fp8
    /// block scale becomes `weight_scale_inv`, which is what the backbone's dequantization
    /// looks for; affine `scales` and `biases` from a re-quantized checkpoint pass through.
    fileprivate static func languageKey(_ key: String) -> String? {
        for (suffix, mapped) in [
            (".weight_scale", ".weight_scale_inv"), (".scales", ".scales"),
            (".biases", ".biases"), (".weight", ".weight"),
        ] where key.hasSuffix(suffix) {
            guard let base = languageBase(String(key.dropLast(suffix.count))) else {
                return nil
            }
            return base + mapped
        }
        return languageBase(key)
    }

    private static func languageBase(_ base: String) -> String? {
        switch base {
        case "tok_embeddings": return "model.embed_tokens"
        case "norm": return "model.norm"
        case "output": return "lm_head"
        default: break
        }

        let parts = base.split(separator: ".").map(String.init)
        guard parts.count >= 2, parts[0] == "layers" else { return nil }
        let prefix = "model.layers.\(parts[1])"
        let rest = parts.dropFirst(2).joined(separator: ".")

        if rest == "gate.e_score_correction_bias" {
            return "\(prefix).mlp.gate.e_score_correction_bias"
        }
        if let renamed = attentionRenames[rest] {
            return "\(prefix).\(renamed)"
        }
        let tail = rest.split(separator: ".").map(String.init)
        if tail.first == "experts", tail.count == 3,
            let projection = expertProjections[tail[2]]
        {
            return "\(prefix).mlp.experts.\(tail[1]).\(projection)"
        }
        if tail.first == "shared_experts", tail.count == 2,
            let projection = expertProjections[tail[1]]
        {
            return "\(prefix).mlp.shared_experts.\(projection)"
        }
        if tail.first == "feed_forward", tail.count == 2,
            let projection = expertProjections[tail[1]]
        {
            return "\(prefix).mlp.\(projection)"
        }
        return nil
    }
}

extension MistralLarge3: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
