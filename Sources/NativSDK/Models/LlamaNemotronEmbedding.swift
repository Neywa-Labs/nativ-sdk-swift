import CoreImage
import Foundation
import MLX
import MLXNN

struct LlamaNemotronEmbeddingConfiguration: Decodable {
    let text: LlamaConfiguration
    let vision: GraniteVisionConfiguration.VisionConfiguration
    let downsampleRatio: Float
    let imageTokenID: Int
    let normalize: Bool
    enum CodingKeys: String, CodingKey {
        case text = "text_config"
        case llm = "llm_config"
        case vision = "vision_config"
        case downsampleRatio = "downsample_ratio"
        case imageTokenID = "img_context_token_id"
        case normalize
    }
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        text =
            try c.decodeIfPresent(LlamaConfiguration.self, forKey: .text)
            ?? c.decode(LlamaConfiguration.self, forKey: .llm)
        vision = try c.decode(GraniteVisionConfiguration.VisionConfiguration.self, forKey: .vision)
        downsampleRatio = try c.decodeIfPresent(Float.self, forKey: .downsampleRatio) ?? 0.5
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? 128258
        normalize = try c.decodeIfPresent(Bool.self, forKey: .normalize) ?? true
        guard downsampleRatio > 0, downsampleRatio <= 1,
            (1 / downsampleRatio).rounded() == 1 / downsampleRatio,
            vision.patchSize > 0, vision.imageSize > 0,
            vision.imageSize % vision.patchSize == 0,
            (vision.imageSize / vision.patchSize) % Int(1 / downsampleRatio) == 0,
            text.hiddenSize > 0, text.attentionHeads > 0, text.kvHeads > 0,
            text.attentionHeads % text.kvHeads == 0, text.hiddenLayers > 0,
            imageTokenID >= 0, imageTokenID < text.vocabularySize
        else {
            throw ModelExecutionError.invalidOptions("Invalid Nemotron embedding configuration")
        }
    }
}

final class LlamaNemotronEmbeddingModel: Module, EmbeddingModel {
    @ModuleInfo(key: "vision_model") var vision: GraniteVisionVision.VisionModel
    @ModuleInfo(key: "language_model") var language: Language
    let mlp1: Projection
    let config: LlamaNemotronEmbeddingConfiguration
    var maximumSequenceLength: Int { config.text.maxPositionEmbeddings ?? 131072 }

    final class Projection: Module {
        let norm: LayerNorm
        @ModuleInfo var input: Linear
        @ModuleInfo var output: Linear
        init(inputSize: Int, hiddenSize: Int) {
            norm = LayerNorm(dimensions: inputSize)
            _input.wrappedValue = Linear(inputSize, hiddenSize)
            _output.wrappedValue = Linear(hiddenSize, hiddenSize)
        }
        func callAsFunction(_ x: MLXArray) -> MLXArray { output(gelu(input(norm(x)))) }
    }

    final class Language: Module {
        let model: LlamaModelInner
        init(_ c: LlamaConfiguration) { model = LlamaModelInner(c) }
    }
    init(_ config: LlamaNemotronEmbeddingConfiguration) {
        self.config = config
        _vision.wrappedValue = GraniteVisionVision.VisionModel(config.vision)
        _language.wrappedValue = Language(config.text)
        let factor = Int(1 / config.downsampleRatio)
        let size = config.vision.hiddenSize * factor * factor
        mlp1 = Projection(inputSize: size, hiddenSize: config.text.hiddenSize)
    }
    func embed(_ input: LMInput) throws -> EmbeddingOutput {
        let (ids, valid) = try embeddingTokens(
            input, vocabularySize: config.text.vocabularySize,
            maximumLength: maximumSequenceLength, allowsMedia: true)
        guard input.video == nil, ids.dim(0) == 1 else {
            throw ModelExecutionError.invalidOptions(
                "Nemotron embeddings require one text or image input")
        }
        let model = language.model
        var h = model.embedTokens(ids)
        if let image = input.image {
            var pixels = image.pixels
            if pixels.ndim == 5, pixels.dim(0) == 1 { pixels = pixels[0] }
            guard pixels.ndim == 4, pixels.dim(1) == config.vision.imageSize,
                pixels.dim(2) == config.vision.imageSize, pixels.dim(3) == config.vision.channels
            else {
                throw ModelExecutionError.invalidOptions("Invalid Nemotron image tensor")
            }
            var features = vision(
                pixels.asType(vision.visionModel.embeddings.patchEmbedding.weight.dtype)
            ).0
            features = Self.pixelShuffle(features, factor: Int(1 / config.downsampleRatio))
            features = mlp1(features)
            features = features.reshaped(-1, config.text.hiddenSize)
            guard
                sum((ids .== config.imageTokenID).asType(.int32)).item(Int.self) == features.dim(0)
            else {
                throw ModelExecutionError.invalidOptions(
                    "Nemotron image features do not match placeholders")
            }
            h = MultimodalEmbedding.mergeInputIDsWithImageFeatures(
                inputIDs: ids, inputEmbeds: h,
                imageFeatures: features.asType(h.dtype), imageTokenID: config.imageTokenID,
                videoTokenID: -1)
        }
        let mask = embeddingAttentionMask(valid)
        for layer in model.layers {
            try Task.checkCancellation()
            h = layer(h, mask: .array(mask), cache: nil)
        }
        let pooled = EmbeddingPooling.mean.apply(model.norm(h), mask: valid)
        return .pooled(config.normalize ? normalizeEmbeddings(pooled) : pooled)
    }
    static func pixelShuffle(_ x: MLXArray, factor: Int) -> MLXArray {
        let b = x.dim(0)
        let c = x.dim(2)
        let side = Int(sqrt(Double(x.dim(1))))
        return x.reshaped(b, side, side / factor, c * factor).transposed(0, 2, 1, 3)
            .reshaped(b, side / factor, side / factor, c * factor * factor)
            .transposed(0, 2, 1, 3).reshaped(b, -1, c * factor * factor)
    }
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights
        where !key.contains("rotary_emb.inv_freq") && !key.contains("lm_head") {
            let name =
                key.hasPrefix("language_model.") && !key.hasPrefix("language_model.model.")
                ? "language_model.model." + key.dropFirst("language_model.".count) : key
            result[
                name.replacingOccurrences(of: "mlp1.0.", with: "mlp1.norm.")
                    .replacingOccurrences(of: "mlp1.1.", with: "mlp1.input.")
                    .replacingOccurrences(of: "mlp1.3.", with: "mlp1.output.")] = value
        }
        return vision.sanitize(weights: result)
    }
}

struct LlamaNemotronEmbeddingProcessor: UserInputProcessor {
    let tokenizer: any Tokenizer
    let config: LlamaNemotronEmbeddingConfiguration
    let settings: [String: JSONValue]

    func prepare(input: UserInput) async throws -> LMInput {
        guard input.images.count == 1, input.videos.isEmpty, input.audios.isEmpty,
            case .text(let text) = input.prompt
        else {
            throw ModelExecutionError.invalidOptions(
                "Nemotron visual embeddings require one image and optional text")
        }
        let size = config.vision.imageSize
        let image = try input.images[0].asCIImage().toSRGB()
        let width = image.extent.width
        let height = image.extent.height
        guard width > 0, height > 0 else {
            throw ModelExecutionError.invalidOptions("Empty embedding image")
        }
        let maxTiles = {
            if case .int(let n) = settings["max_input_tiles"] { return n }
            return 6
        }()
        guard maxTiles > 0, maxTiles <= 128 else {
            throw ModelExecutionError.invalidOptions("Invalid image tile limit")
        }
        var columns = 1
        var rows = 1
        if settings["dynamic_image_size"] != .bool(false) {
            var best: CGFloat = -1
            for count in 1 ... maxTiles {
                for x in 1 ... count where count % x == 0 {
                    let y = count / x
                    let ratio = CGFloat(x) / CGFloat(y)
                    let aspect = width / height
                    let score =
                        min(CGFloat(count * size * size) / (width * height), 0.6)
                        * min(ratio / aspect, aspect / ratio)
                    if score > best {
                        best = score
                        columns = x
                        rows = y
                    }
                }
            }
        }
        let resized = image.resampled(
            to: CGSize(width: columns * size, height: rows * size), method: .bicubic)
        var tiles: [CIImage] = []
        for index in 0 ..< (columns * rows) {
            let rect = CGRect(
                x: (index % columns) * size, y: (rows - index / columns - 1) * size,
                width: size, height: size)
            tiles.append(
                resized.cropped(to: rect).transformed(
                    by: .init(translationX: -rect.minX, y: -rect.minY)))
        }
        if tiles.count > 1, settings["use_thumbnail"] != .bool(false) {
            tiles.append(image.resampled(to: CGSize(width: size, height: size), method: .bicubic))
        }
        let siglip = settings["norm_type"] == nil || settings["norm_type"] == .string("siglip")
        guard siglip || settings["norm_type"] == .string("imagenet") else {
            throw ModelExecutionError.invalidOptions("Unsupported image normalization")
        }
        let pixels = concatenated(
            tiles.map {
                $0.normalized(
                    mean: siglip ? (0.5, 0.5, 0.5) : (0.485, 0.456, 0.406),
                    std: siglip ? (0.5, 0.5, 0.5) : (0.229, 0.224, 0.225)
                ).asMLXArray()
            }, axis: 0
        ).transposed(0, 2, 3, 1)
        let factor = Int(1 / config.downsampleRatio)
        let tokensPerTile =
            (size / config.vision.patchSize / factor) * (size / config.vision.patchSize / factor)
        let imageTokens =
            "<img>" + String(repeating: "<IMG_CONTEXT>", count: tiles.count * tokensPerTile)
            + "</img>"
        let prefix = {
            if case .string(let s) = settings["passage_prefix"] { return s }
            return "passage:"
        }()
        let system = {
            if case .string(let s) = settings["system_message"] { return s }
            return ""
        }()
        let prompt = system + (prefix.isEmpty ? "" : prefix + " ") + imageTokens + " " + text
        let ids = tokenizer.encode(text: prompt, addSpecialTokens: true)
        guard ids.filter({ $0 == config.imageTokenID }).count == tiles.count * tokensPerTile else {
            throw ModelExecutionError.invalidOptions("Tokenizer is missing Nemotron image tokens")
        }
        return LMInput(
            text: .init(tokens: MLXArray(ids).reshaped(1, -1)), image: .init(pixels: pixels))
    }
}
