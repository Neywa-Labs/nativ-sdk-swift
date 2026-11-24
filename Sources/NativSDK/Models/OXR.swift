// OXR follows SoMarkAI's reference architecture and the Python MLX port.
// See Documentation/Models/OXR/README.md for attribution and license links.
import Foundation
import MLX
import MLXNN

public struct OXRConfiguration: Decodable, Sendable {
    public struct VisionConfiguration: Decodable, Sendable {
        let hiddenSize: Int
        let intermediateSize: Int
        let hiddenLayers: Int
        let attentionHeads: Int
        let channels: Int
        let patchSize: Int
        let imageSize: Int
        let hiddenStride: Int
        let temporalPatchSize: Int
        let layerNormEps: Float

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case channels = "num_channels"
            case patchSize = "patch_size"
            case imageSize = "image_size"
            case hiddenStride = "hidden_stride"
            case temporalPatchSize = "temporal_patch_size"
            case layerNormEps = "layer_norm_eps"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 27
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 512
            hiddenStride = try c.decodeIfPresent(Int.self, forKey: .hiddenStride) ?? 2
            temporalPatchSize = try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 1
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        }
    }

    let text: Qwen3Configuration
    let vision: VisionConfiguration
    let visualVocabularySize: Int
    let imageTokenId: Int
    let visionStartTokenId: Int
    let visionEndTokenId: Int

    enum CodingKeys: String, CodingKey {
        case text = "text_config"
        case llm = "llm_config"
        case vision = "vision_config"
        case vit = "vit_config"
        case visualVocabularySize = "visual_vocab_size"
        case imageTokenId = "image_token_id"
        case visionStartTokenId = "vision_start_token_id"
        case visionEndTokenId = "vision_end_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        text =
            try c.decodeIfPresent(Qwen3Configuration.self, forKey: .text)
            ?? c.decode(Qwen3Configuration.self, forKey: .llm)
        vision =
            try c.decodeIfPresent(VisionConfiguration.self, forKey: .vision)
            ?? c.decode(VisionConfiguration.self, forKey: .vit)
        visualVocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .visualVocabularySize) ?? 65536
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 151655
        visionStartTokenId = try c.decodeIfPresent(Int.self, forKey: .visionStartTokenId) ?? 151652
        visionEndTokenId = try c.decodeIfPresent(Int.self, forKey: .visionEndTokenId) ?? 151653
    }
}

extension OXRConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        try text.validateModelConfiguration()
        guard vision.hiddenSize > 0, vision.attentionHeads > 0,
            vision.hiddenSize.isMultiple(of: vision.attentionHeads),
            (vision.hiddenSize / vision.attentionHeads).isMultiple(of: 4),
            vision.hiddenLayers > 0, vision.intermediateSize > 0,
            vision.channels == 3, vision.patchSize > 0, vision.hiddenStride > 0,
            vision.imageSize > 0, vision.imageSize.isMultiple(of: vision.patchSize),
            vision.temporalPatchSize == 1, vision.layerNormEps > 0,
            visualVocabularySize > 4,
            text.hiddenSize > 0, text.hiddenLayers > 0, text.attentionHeads > 0,
            text.kvHeads > 0, text.attentionHeads.isMultiple(of: text.kvHeads),
            text.headDim > 0, text.headDim.isMultiple(of: 2), text.intermediateSize > 0,
            text.rmsNormEps > 0,
            [imageTokenId, visionStartTokenId, visionEndTokenId].allSatisfy({
                $0 >= 0 && $0 < text.vocabularySize
            })
        else {
            throw ModelExecutionError.invalidOptions("Invalid OXR image or language configuration")
        }
    }
}

/// Torch-compatible bicubic interpolation of the learned vision position grid.
func oxrBicubicMatrix(input: Int, output: Int) -> MLXArray {
    var matrix = [Float](repeating: 0, count: input * output)
    func near(_ x: Double) -> Double { ((1.25 * x - 2.25) * x) * x + 1 }
    func far(_ x: Double) -> Double { ((-0.75 * x + 3.75) * x - 6) * x + 3 }
    for row in 0 ..< output {
        let source = (Double(row) + 0.5) * Double(input) / Double(output) - 0.5
        let base = Int(floor(source))
        let fraction = source - floor(source)
        let weights = [far(fraction + 1), near(fraction), near(1 - fraction), far(2 - fraction)]
        // Accumulate boundary taps in Double, then round once like the reference matrix.
        var values = [Double](repeating: 0, count: input)
        for tap in 0 ..< 4 { values[min(input - 1, max(0, base - 1 + tap))] += weights[tap] }
        for column in 0 ..< input { matrix[row * input + column] = Float(values[column]) }
    }
    return MLXArray(matrix, [output, input])
}

enum OXRVision {
    final class Attention: Module {
        @ModuleInfo(key: "q_proj") var query: Linear
        @ModuleInfo(key: "k_proj") var key: Linear
        @ModuleInfo(key: "v_proj") var value: Linear
        @ModuleInfo(key: "out_proj") var output: Linear
        let heads: Int

        init(_ config: OXRConfiguration.VisionConfiguration) {
            heads = config.attentionHeads
            _query.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
            _key.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
            _value.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
            _output.wrappedValue = Linear(config.hiddenSize, config.hiddenSize)
        }

        func callAsFunction(_ x: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
            let shape = [1, x.dim(1), heads, x.dim(-1) / heads]
            func rotate(_ projection: MLXArray) -> MLXArray {
                let projected = projection.reshaped(shape)
                let full = projected.asType(.float32)
                let half = full.dim(-1) / 2
                let rotated = concatenated(
                    [-full[.ellipsis, half...], full[.ellipsis, ..<half]], axis: -1)
                return
                    (full * cosine[0..., 0..., .newAxis, 0...] + rotated
                    * sine[0..., 0..., .newAxis, 0...])
                    .asType(projected.dtype).transposed(0, 2, 1, 3)
            }
            let q = rotate(query(x))
            let k = rotate(key(x))
            let v = value(x).reshaped(shape).transposed(0, 2, 1, 3)
            let result = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v,
                scale: pow(Float(shape[3]), -0.5), mask: .none)
            return output(result.transposed(0, 2, 1, 3).reshaped(x.shape))
        }
    }

    final class MLP: Module, UnaryLayer {
        let fc1: Linear
        let fc2: Linear
        init(_ config: OXRConfiguration.VisionConfiguration) {
            fc1 = Linear(config.hiddenSize, config.intermediateSize)
            fc2 = Linear(config.intermediateSize, config.hiddenSize)
        }
        func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(geluApproximate(fc1(x))) }
    }

    final class Layer: Module {
        @ModuleInfo(key: "layer_norm1") var norm1: LayerNorm
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "layer_norm2") var norm2: LayerNorm
        let mlp: MLP
        init(_ config: OXRConfiguration.VisionConfiguration) {
            _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            _attention.wrappedValue = Attention(config)
            mlp = MLP(config)
        }
        func callAsFunction(_ x: MLXArray, cosine: MLXArray, sine: MLXArray) -> MLXArray {
            let h = x + attention(norm1(x), cosine: cosine, sine: sine)
            return h + mlp(norm2(h))
        }
    }

    final class Embeddings: Module {
        @ModuleInfo(key: "patch_embedding") var patch: Linear
        @ModuleInfo(key: "position_embedding") var position: Embedding
        let config: OXRConfiguration.VisionConfiguration
        init(_ config: OXRConfiguration.VisionConfiguration) {
            self.config = config
            _patch.wrappedValue = Linear(
                config.channels * config.patchSize * config.patchSize, config.hiddenSize)
            let side = config.imageSize / config.patchSize
            _position.wrappedValue = Embedding(
                embeddingCount: side * side, dimensions: config.hiddenSize)
        }
        func positionEmbeddings(_ frame: THW) -> MLXArray {
            let side = config.imageSize / config.patchSize
            let weight = position.weight.asType(.float32).reshaped(side, side, -1)
            let height = oxrBicubicMatrix(input: side, output: frame.h)
            let width = oxrBicubicMatrix(input: side, output: frame.w)
            let resized = einsum("hi,ijc,wj->hwc", height, weight, width).asType(
                position.weight.dtype)
            let stride = config.hiddenStride
            return resized.reshaped(frame.h / stride, stride, frame.w / stride, stride, -1)
                .transposed(0, 2, 1, 3, 4).reshaped(frame.h * frame.w, -1)
        }
    }

    final class Model: Module {
        let embeddings: Embeddings
        let layers: [Layer]
        let config: OXRConfiguration.VisionConfiguration
        init(_ config: OXRConfiguration.VisionConfiguration) {
            self.config = config
            embeddings = Embeddings(config)
            layers = (0 ..< config.hiddenLayers).map { _ in Layer(config) }
        }
        func rotary(_ frame: THW) -> (MLXArray, MLXArray) {
            let stride = config.hiddenStride
            var rows = [Float]()
            var columns = [Float]()
            for blockY in 0 ..< frame.h / stride {
                for blockX in 0 ..< frame.w / stride {
                    for y in 0 ..< stride {
                        for x in 0 ..< stride {
                            rows.append(Float(blockY * stride + y))
                            columns.append(Float(blockX * stride + x))
                        }
                    }
                }
            }
            let dimensions = config.hiddenSize / config.attentionHeads / 2
            let inverse =
                1
                / pow(
                    Float(10000),
                    MLXArray(Array(Swift.stride(from: 0, to: dimensions, by: 2))).asType(.float32)
                        / dimensions)
            let frequencies = concatenated(
                [outer(MLXArray(rows), inverse), outer(MLXArray(columns), inverse)], axis: -1)
            let phases = concatenated([frequencies, frequencies], axis: -1)[.newAxis]
            return (cos(phases), sin(phases))
        }
        func callAsFunction(_ pixels: MLXArray, frames: [THW]) -> MLXArray {
            var start = 0
            let encoded = frames.map { frame in
                let end = start + frame.product
                var x =
                    (embeddings.patch(pixels[start ..< end].asType(embeddings.patch.weight.dtype))
                    + embeddings.positionEmbeddings(frame))[.newAxis]
                let (cosine, sine) = rotary(frame)
                for layer in layers { x = layer(x, cosine: cosine, sine: sine) }
                start = end
                return x[0]
            }
            return concatenated(encoded)
        }
    }

    final class HeadLayers: Module {
        @ModuleInfo(key: "0") var projection: Linear
        @ModuleInfo(key: "1") var norm: LayerNorm
        init(input: Int, vocabulary: Int) {
            _projection.wrappedValue = Linear(input, vocabulary, bias: false)
            _norm.wrappedValue = LayerNorm(dimensions: vocabulary)
        }
        func callAsFunction(_ x: MLXArray) -> MLXArray { norm(projection(x)) }
    }

    final class Head: Module, UnaryLayer {
        let layers: HeadLayers
        init(input: Int, vocabulary: Int) {
            layers = HeadLayers(input: input, vocabulary: vocabulary)
        }
        func callAsFunction(_ x: MLXArray) -> MLXArray { layers(x) }
    }
}

public final class OXR: Module, VLMModel {
    @ModuleInfo(key: "vision_tower") var vision: OXRVision.Model
    @ModuleInfo(key: "vision_head") var visionHead: OXRVision.Head
    @ModuleInfo(key: "vision_embed") var visionEmbedding: Embedding
    @ModuleInfo(key: "language_model") var languageModel: Qwen3Model
    let config: OXRConfiguration
    public var vocabularySize: Int { config.text.vocabularySize }
    public var loraLayers: [Module] { languageModel.loraLayers }

    public init(_ config: OXRConfiguration) {
        self.config = config
        _vision.wrappedValue = OXRVision.Model(config.vision)
        _visionHead.wrappedValue = OXRVision.Head(
            input: config.vision.hiddenSize * config.vision.hiddenStride
                * config.vision.hiddenStride,
            vocabulary: config.visualVocabularySize - 4)
        _visionEmbedding.wrappedValue = Embedding(
            embeddingCount: config.visualVocabularySize, dimensions: config.text.hiddenSize)
        _languageModel.wrappedValue = Qwen3Model(config.text)
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try languageModel.newCache(configuration: configuration)
    }

    func encodeImages(_ pixels: MLXArray, frames: [THW]) -> MLXArray {
        let features = vision(pixels, frames: frames)
        let stride = config.vision.hiddenStride
        let logits = visionHead(features.reshaped(features.dim(0) / (stride * stride), -1))
        let probabilities = softmax(logits.asType(.float32), axis: -1).asType(logits.dtype)
        return matmul(probabilities, visionEmbedding.weight[..<(config.visualVocabularySize - 4)])
    }

    func inputEmbeddings(_ input: LMInput) throws -> (MLXArray, MLXArray) {
        guard input.video == nil, input.audio == nil else {
            throw VLMError.processing("OXR accepts image and text input")
        }
        var ids = input.text.tokens.ndim == 1 ? input.text.tokens[.newAxis] : input.text.tokens
        guard ids.ndim == 2, ids.dim(0) == 1, ids.dim(1) > 0 else {
            throw VLMError.processing("OXR requires one nonempty input sequence")
        }
        if let mask = input.text.mask {
            guard mask.size == ids.size, all((mask .== 0) .|| (mask .== 1)).item(Bool.self) else {
                throw VLMError.processing("Invalid OXR attention mask")
            }
            let valid = mask.flattened().asArray(Int.self).enumerated().filter { $0.element == 1 }
                .map(\.offset)
            guard let first = valid.first, let last = valid.last, last - first + 1 == valid.count
            else { throw VLMError.processing("OXR requires contiguous non-padding tokens") }
            ids = ids[0..., first ..< last + 1]
        }
        var embeddings = languageModel.model.embedTokens(ids)
        guard let image = input.image else { return (ids, embeddings) }
        let stride = config.vision.hiddenStride
        guard let frames = image.frames, !frames.isEmpty,
            frames.allSatisfy({
                $0.t == 1 && $0.h > 0 && $0.w > 0 && $0.h.isMultiple(of: stride)
                    && $0.w.isMultiple(of: stride)
            }),
            image.pixels.ndim == 2, image.pixels.dim(0) == frames.reduce(0, { $0 + $1.product }),
            image.pixels.dim(1) == config.vision.channels * config.vision.patchSize
                * config.vision.patchSize
        else { throw VLMError.processing("Invalid OXR packed image patches") }
        let visual = encodeImages(image.pixels, frames: frames).asType(embeddings.dtype)
        let tokens = ids.asArray(Int.self)
        let pads = tokens.indices.filter { tokens[$0] == config.imageTokenId }
        let starts = tokens.indices.filter { tokens[$0] == config.visionStartTokenId }
        let ends = tokens.indices.filter { tokens[$0] == config.visionEndTokenId }
        guard pads.count == visual.dim(0), starts.count == frames.count, ends.count == frames.count
        else { throw VLMError.processing("OXR image features and placeholders do not match") }
        let indicators = visionEmbedding.weight[(config.visualVocabularySize - 4)...].asType(
            embeddings.dtype)
        let flat = embeddings.reshaped(-1, config.text.hiddenSize)
        flat[MLXArray(pads)] = visual
        flat[MLXArray(starts)] = broadcast(
            indicators[0], to: [starts.count, config.text.hiddenSize])
        flat[MLXArray(ends)] = broadcast(indicators[1], to: [ends.count, config.text.hiddenSize])
        embeddings = flat.reshaped(embeddings.shape)
        return (ids, embeddings)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        let (ids, embeddings) = try inputEmbeddings(input)
        let processed = try prefill.forEachChunk(total: ids.dim(1)) { range in
            _ = languageModel(
                ids[0..., range], cache: cache, inputEmbedding: embeddings[0..., range, 0...])
            asyncEval(cache)
        }
        eval(cache)
        let logits = languageModel(
            ids[0..., processed...], cache: cache,
            inputEmbedding: embeddings[0..., processed..., 0...])
        prefill.progress?(ids.dim(1), ids.dim(1))
        return .logits(LMOutput(logits: logits))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func quantizationPredicate(path: String, module: Module) -> QuantizationDecision {
        path.hasPrefix("vision_tower") || path.hasPrefix("vision_head")
            || path.hasPrefix("vision_embed") ? .skip : .useDefault
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (original, value) in weights {
            var key = original
            if key.hasPrefix("visual_tokenizer.vit.vision_model.") {
                key = String(key.dropFirst("visual_tokenizer.vit.vision_model.".count))
                if key.hasPrefix("post_layernorm.") { continue }
                key =
                    "vision_tower."
                    + key.replacingOccurrences(of: "encoder.layers.", with: "layers.")
            } else if key.hasPrefix("visual_tokenizer.head.") {
                key = "vision_head.layers." + key.dropFirst("visual_tokenizer.head.".count)
            } else if key.hasPrefix("vte.") {
                key = "vision_embed." + key.dropFirst(4)
            } else if key.hasPrefix("llm.lm_head.") {
                key = "language_model.lm_head." + key.dropFirst("llm.lm_head.".count)
            } else if key.hasPrefix("llm.model.") {
                key = "language_model.model." + key.dropFirst("llm.model.".count)
            }
            if config.text.tieWordEmbeddings, key.hasPrefix("language_model.lm_head.") { continue }
            result[key] =
                key == "vision_tower.embeddings.patch_embedding.weight" && value.ndim == 4
                ? value.reshaped(value.dim(0), -1) : value
        }
        return result
    }
}
