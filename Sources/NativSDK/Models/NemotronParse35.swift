// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: OpenMDW-1.1
// Ported from NVIDIA's RadioMBartPrefixForCausalLM and the Python MLX implementation.
import Foundation
import MLX
import MLXNN

public struct NemotronParse35Configuration: Decodable, Sendable {
    let vision: NemotronParseConfiguration.VisionConfiguration
    let text: NemotronParseConfiguration.TextConfiguration
    let kvHeads: Int
    let maxSequenceLength: Int
    private enum CodingKeys: String, CodingKey {
        case vision = "vision_config"
        case text = "text_config"
        case decoder
        case maxSequenceLength = "max_sequence_length"
        case kvHeads = "self_attention_kv_heads"
    }
    private struct TextOptions: Decodable {
        var kvHeads: Int?
        var vocabularySize: Int?
        var maxPositions: Int?
        enum CodingKeys: String, CodingKey {
            case kvHeads = "self_attention_kv_heads"
            case vocabularySize = "vocab_size"
            case maxPositions = "max_position_embeddings"
        }
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        var v =
            try c.decodeIfPresent(
                NemotronParseConfiguration.VisionConfiguration.self, forKey: .vision)
            ?? NemotronParseConfiguration.VisionConfiguration()
        let values = try c.decodeIfPresent([String: JSONValue].self, forKey: .vision) ?? [:]
        if values["hidden_size"] == nil { v.hiddenSize = 1152 }
        if values["mlp_ratio"] == nil { v.mlpRatio = 4304 / Float(1152) }
        if values["num_layers"] == nil { v.layers = 27 }
        if values["num_cls_tokens"] == nil { v.clsTokens = 2 }
        if values["num_register_tokens"] == nil { v.registerTokens = 8 }
        if values["summary_idxs"] == nil { v.summaryIndices = [0, 1] }
        vision = v
        let key: CodingKeys
        if c.contains(.text), !(try c.decodeNil(forKey: .text)) {
            key = .text
        } else {
            key = .decoder
        }
        var t =
            try c.decodeIfPresent(NemotronParseConfiguration.TextConfiguration.self, forKey: key)
            ?? NemotronParseConfiguration.TextConfiguration()
        let options = try c.decodeIfPresent(TextOptions.self, forKey: key)
        if options?.vocabularySize == nil { t.vocabularySize = 72512 }
        if options?.maxPositions == nil { t.maxPositionEmbeddings = 16384 }
        text = t
        kvHeads = try options?.kvHeads ?? c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        maxSequenceLength = try c.decodeIfPresent(Int.self, forKey: .maxSequenceLength) ?? 16384
    }
    static func matches(_ data: Data) -> Bool {
        guard let object = try? JSONDecoder.json5().decode([String: JSONValue].self, from: data)
        else { return false }
        if object["model_type"] == .string("nemotron_parse_v3_5") { return true }
        if case .array(let architectures) = object["architectures"] {
            return architectures.contains(.string("RadioMBartPrefixForCausalLM"))
        }
        return false
    }
}

extension NemotronParse35Configuration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard vision.patchSize == 16, vision.hiddenSize > 0, vision.heads > 0,
            vision.hiddenSize.isMultiple(of: vision.heads), vision.layers > 0,
            vision.mlpRatio.isFinite, vision.mlpRatio > 0,
            vision.clsTokens > 0, vision.registerTokens >= 0,
            !vision.summaryIndices.isEmpty,
            vision.summaryIndices.allSatisfy({ $0 >= 0 && $0 < vision.skippedTokens }),
            vision.neckDimensions == text.modelDimension,
            text.modelDimension > 0, text.attentionHeads > 0, kvHeads > 0,
            text.modelDimension.isMultiple(of: text.attentionHeads),
            text.attentionHeads.isMultiple(of: kvHeads),
            text.layers > 0, text.feedForwardDimension > 0, text.vocabularySize > 0,
            maxSequenceLength > 0
        else {
            throw ModelExecutionError.invalidOptions("Invalid Nemotron 3.5 Parse configuration")
        }
    }
}

enum NemotronParse35Vision {
    static func resizeWeights(source: Int, target: Int, area: Bool) -> MLXArray {
        if area {
            var values = [Float](repeating: 0, count: source * target)
            for row in 0 ..< target {
                let first = row * source / target
                let last = ((row + 1) * source + target - 1) / target
                for column in first ..< last {
                    values[row * source + column] = 1 / Float(last - first)
                }
            }
            return MLXArray(values, [target, source])
        }
        let scale = Float(source) / Float(target)
        let centers = (MLXArray(0 ..< target).asType(.float32) + 0.5) * scale - 0.5
        let distance = abs(
            centers[0..., .newAxis] - MLXArray(0 ..< source).asType(.float32)[.newAxis])
        let weights = maximum(1 - distance / max(scale, 1), 0)
        return weights / weights.sum(axis: 1, keepDims: true)
    }

    final class Model: Module {
        @ModuleInfo(key: "patch_embed") var patchEmbed: Linear
        @ParameterInfo(key: "pos_embed") var positionEmbedding: MLXArray
        @ParameterInfo(key: "cls_token") var clsToken: MLXArray
        let blocks: [NemotronParseVision.ViTBlock]
        let neck: NemotronParseVision.Neck
        let config: NemotronParseConfiguration.VisionConfiguration

        init(_ config: NemotronParseConfiguration.VisionConfiguration) {
            self.config = config
            _patchEmbed.wrappedValue = Linear(768, config.hiddenSize, bias: false)
            _positionEmbedding.wrappedValue = MLXArray.zeros([1, 128 * 128, config.hiddenSize])
            _clsToken.wrappedValue = MLXArray.zeros([config.skippedTokens, config.hiddenSize])
            blocks = (0 ..< config.layers).map { _ in NemotronParseVision.ViTBlock(config) }
            neck = NemotronParseVision.Neck(config)
        }

        func positions(rows: Int, columns: Int, contentHeight: Int, contentWidth: Int, dtype: DType)
            -> MLXArray
        {
            let contentRows = (contentHeight + 15) / 16
            let contentColumns = (contentWidth + 15) / 16
            let area = min(contentRows, contentColumns) == 1
            let grid = einsum(
                "rh,hwc,sw->rsc",
                resizeWeights(source: 128, target: contentRows, area: area),
                positionEmbedding.reshaped(128, 128, config.hiddenSize).asType(.float32),
                resizeWeights(source: 128, target: contentColumns, area: area))
            func coordinates(_ count: Int, _ content: Int, _ target: Int) -> MLXArray {
                let starts = MLXArray(0 ..< count).asType(.float32) * 16
                let extent = clip(Float(content) - starts, min: 0, max: 16)
                return clip(
                    minimum(starts + 0.5 * extent, Float(content)) / Float(content) * Float(target)
                        - 0.5, min: 0, max: Float(target - 1))
            }
            let y = coordinates(rows, contentHeight, contentRows)
            let x = coordinates(columns, contentWidth, contentColumns)
            let y0 = floor(y).asType(.int32)
            let x0 = floor(x).asType(.int32)
            let y1 = minimum(y0 + 1, contentRows - 1)
            let x1 = minimum(x0 + 1, contentColumns - 1)
            let wy = (y - y0)[0..., .newAxis, .newAxis]
            let wx = (x - x0)[.newAxis, 0..., .newAxis]
            let top = grid[y0][0..., x0] * (1 - wx) + grid[y0][0..., x1] * wx
            let bottom = grid[y1][0..., x0] * (1 - wx) + grid[y1][0..., x1] * wx
            return (top * (1 - wy) + bottom * wy).reshaped(1, rows * columns, -1).asType(dtype)
        }

        func callAsFunction(_ image: LMInput.ProcessedImage) throws -> MLXArray {
            guard let frames = image.frames, frames.count == 1, frames[0].t == 1,
                let contents = image.contentSizes, contents.count == 1,
                contents[0].width.isFinite, contents[0].height.isFinite,
                contents[0].width >= 1, contents[0].height >= 1,
                frames[0].h > 0, frames[0].w > 0, frames[0].w.isMultiple(of: 4),
                contents[0].height <= CGFloat(frames[0].h * 16),
                contents[0].width <= CGFloat(frames[0].w * 16),
                image.pixels.shape == [frames[0].h * frames[0].w, 768]
            else { throw VLMError.processing("Invalid Nemotron 3.5 packed page geometry") }
            let frame = frames[0]
            var x = patchEmbed(image.pixels.asType(patchEmbed.weight.dtype))[.newAxis]
            x =
                x
                + positions(
                    rows: frame.h, columns: frame.w,
                    contentHeight: Int(contents[0].height), contentWidth: Int(contents[0].width),
                    dtype: x.dtype)
            x = concatenated([clsToken[.newAxis].asType(x.dtype), x], axis: 1)
            for block in blocks { x = block(x) }
            let summary = x[0..., MLXArray(config.summaryIndices), 0...].reshaped(1, -1)
            return neck(
                features: x[0..., config.skippedTokens..., 0...], summary: summary,
                patchGrid: (frame.h, frame.w))
        }
    }
}

enum NemotronParse35Language {
    final class Attention: Module {
        @ModuleInfo(key: "q_proj") var query: Linear
        @ModuleInfo(key: "k_proj") var key: Linear
        @ModuleInfo(key: "v_proj") var value: Linear
        @ModuleInfo(key: "out_proj") var output: Linear
        let heads: Int
        let kvHeads: Int
        let headSize: Int
        init(_ config: NemotronParse35Configuration) {
            let size = config.text.modelDimension
            heads = config.text.attentionHeads
            kvHeads = config.kvHeads
            headSize = size / heads
            _query.wrappedValue = Linear(size, size)
            _key.wrappedValue = Linear(size, kvHeads * headSize)
            _value.wrappedValue = Linear(size, kvHeads * headSize)
            _output.wrappedValue = Linear(size, size)
        }
        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let q = query(x).reshaped(x.dim(0), x.dim(1), heads, headSize).transposed(0, 2, 1, 3)
            let k = key(x).reshaped(x.dim(0), x.dim(1), kvHeads, headSize).transposed(0, 2, 1, 3)
            let v = value(x).reshaped(x.dim(0), x.dim(1), kvHeads, headSize).transposed(0, 2, 1, 3)
            let attended = attentionWithCacheUpdate(
                queries: q, keys: k, values: v, cache: cache,
                scale: pow(Float(headSize), -0.5), mask: mask)
            return output(attended.transposed(0, 2, 1, 3).reshaped(x.shape))
        }
    }
    final class Layer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "self_attn_layer_norm") var attentionNorm: LayerNorm
        @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        init(_ config: NemotronParse35Configuration) {
            let text = config.text
            _attention.wrappedValue = Attention(config)
            _attentionNorm.wrappedValue = LayerNorm(dimensions: text.modelDimension)
            _finalNorm.wrappedValue = LayerNorm(dimensions: text.modelDimension)
            _fc1.wrappedValue = Linear(text.modelDimension, text.feedForwardDimension)
            _fc2.wrappedValue = Linear(text.feedForwardDimension, text.modelDimension)
        }
        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let h = x + attention(attentionNorm(x), mask: mask, cache: cache)
            return h + fc2(gelu(fc1(finalNorm(h))))
        }
    }
    final class Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layernorm_embedding") var embeddingNorm: LayerNorm
        @ModuleInfo(key: "layer_norm") var finalNorm: LayerNorm
        let layers: [Layer]
        let scale: Float
        init(_ config: NemotronParse35Configuration) {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.text.vocabularySize, dimensions: config.text.modelDimension)
            _embeddingNorm.wrappedValue = LayerNorm(dimensions: config.text.modelDimension)
            _finalNorm.wrappedValue = LayerNorm(dimensions: config.text.modelDimension)
            layers = (0 ..< config.text.layers).map { _ in Layer(config) }
            scale = config.text.scaleEmbedding ? sqrt(Float(config.text.modelDimension)) : 1
        }
        func embed(_ ids: MLXArray) -> MLXArray { embedTokens(ids) * scale }
        func callAsFunction(_ embeddings: MLXArray, cache: [KVCache]?, logitsToKeep: Int? = nil)
            -> MLXArray
        {
            let mask = createAttentionMask(h: embeddings, cache: cache?.first)
            var x = embeddingNorm(embeddings)
            for (index, layer) in layers.enumerated() {
                x = layer(x, mask: mask, cache: cache?[index])
            }
            x = finalNorm(x)
            if let count = logitsToKeep { x = x[0..., (-count)..., 0...] }
            return embedTokens.asLinear(x)
        }
    }
}

public final class NemotronParse35: Module, VLMModel, InferenceErrorProviding {
    @ModuleInfo(key: "vision_tower") var visionTower: NemotronParse35Vision.Model
    @ModuleInfo(key: "language_model") var languageModel: NemotronParse35Language.Model
    let config: NemotronParse35Configuration
    private var inferenceError: (any Error)?
    public var vocabularySize: Int { config.text.vocabularySize }
    public var loraLayers: [Module] { languageModel.layers }
    public init(_ config: NemotronParse35Configuration) {
        self.config = config
        _visionTower.wrappedValue = NemotronParse35Vision.Model(config.vision)
        _languageModel.wrappedValue = NemotronParse35Language.Model(config)
    }
    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        // This model has no position embeddings: dropping old tokens changes its prefix semantics.
        guard configuration?.capacity == nil else {
            throw ModelExecutionError.invalidOptions(
                "Nemotron 3.5 Parse requires a full context cache")
        }
        return (0 ..< config.text.layers).map { _ in KVCacheSimple() }
    }
    func checkInferenceError() throws {
        if let error = inferenceError {
            inferenceError = nil
            throw error
        }
    }
    func inputEmbeddings(_ input: LMInput) throws -> (MLXArray, Int) {
        let ids = input.text.tokens.ndim == 1 ? input.text.tokens[.newAxis] : input.text.tokens
        guard ids.ndim == 2, ids.dim(0) == 1, ids.dim(1) > 0,
            input.video == nil, input.audio == nil,
            input.text.mask == nil || all(input.text.mask! .== 1).item(Bool.self)
        else { throw VLMError.processing("Nemotron 3.5 Parse requires one unpadded page prompt") }
        let text = languageModel.embed(ids)
        guard let image = input.image else { return (text, ids.dim(1)) }
        return (concatenated([try visionTower(image), text], axis: 1), ids.dim(1))
    }
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        inferenceError = nil
        guard input.image != nil || (cache.first?.offset ?? 0) > 0 else {
            throw VLMError.processing("Nemotron 3.5 Parse requires a document image")
        }
        let (embeddings, promptLength) = try inputEmbeddings(input)
        let total = embeddings.dim(1)
        guard (cache.first?.offset ?? 0) + total <= config.maxSequenceLength else {
            throw VLMError.processing("Visual prefix plus text exceeds the model context limit")
        }
        let processed = try prefill.forEachChunk(total: total) { range in
            _ = languageModel(embeddings[0..., range, 0...], cache: cache)
            asyncEval(cache)
        }
        eval(cache)
        let logits = languageModel(
            embeddings[0..., processed..., 0...], cache: cache,
            logitsToKeep: min(promptLength, total - processed))
        prefill.progress?(total, total)
        return .logits(.init(logits: logits))
    }
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let ids = inputs.ndim == 1 ? inputs[.newAxis] : inputs
        guard (cache?.first?.offset ?? 0) + ids.dim(1) <= config.maxSequenceLength else {
            inferenceError = VLMError.processing("Nemotron 3.5 Parse context limit exceeded")
            return MLXArray.zeros([ids.dim(0), 1, vocabularySize])
        }
        return languageModel(languageModel.embed(ids), cache: cache)
    }
    public func quantizationPredicate(path: String, module: Module) -> QuantizationDecision {
        path.hasPrefix("vision_tower") ? .skip : .useDefault
    }
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("lm_head.") || key.hasPrefix("decoder.extra_heads.")
                || key.hasPrefix("decoder.extra_proj.")
            {
                continue
            }
            let native =
                key.hasPrefix("decoder.")
                ? "language_model." + key.dropFirst(8) : nemotronParseVisionWeightKey(key)
            result[native] = value
        }
        return nemotronParseVisionLayouts(result)
    }
}
