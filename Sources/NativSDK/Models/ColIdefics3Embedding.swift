import CoreImage
import Foundation
import MLX
import MLXNN

final class ColIdefics3EmbeddingModel: Module, EmbeddingModel {
    let backbone: Idefics3
    @ModuleInfo var linear: Linear
    let maximumSequenceLength: Int
    init(configuration: Data) throws {
        struct Options: Decodable {
            let embedding_dim: Int?
            let mask_non_image_embeddings: Bool?
            let text_config: Text
            struct Text: Decodable { let max_position_embeddings: Int? }
        }
        let options = try JSONDecoder.json5().decode(Options.self, from: configuration)
        let config = try JSONDecoder.json5().decode(Idefics3Configuration.self, from: configuration)
        let dimension = options.embedding_dim ?? 128
        guard options.mask_non_image_embeddings != true, dimension > 0,
            config.scaleFactor > 0, config.visionConfig.patchSize > 0,
            (config.visionConfig.imageSize / config.visionConfig.patchSize) % config.scaleFactor
                == 0
        else {
            throw ModelExecutionError.invalidOptions("Unsupported ColIdefics3 configuration")
        }
        backbone = Idefics3(config)
        backbone.removeLanguageHead()
        _linear.wrappedValue = Linear(config.textConfig.hiddenSize, dimension)
        maximumSequenceLength = options.text_config.max_position_embeddings ?? 4096
    }
    func embed(_ input: LMInput) throws -> EmbeddingOutput {
        let (ids, valid) = try embeddingTokens(
            input, vocabularySize: backbone.config.textConfig.vocabSize,
            maximumLength: maximumSequenceLength, allowsMedia: true)
        guard input.video == nil, ids.dim(0) == 1 else {
            throw ModelExecutionError.invalidOptions("ColIdefics3 requires one text or image input")
        }
        let hidden = try backbone.embeddingHiddenStates(input, ids: ids, valid: valid)
        return .tokens(normalizeEmbeddings(linear(hidden)))
    }
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var body: [String: MLXArray] = [:]
        var result: [String: MLXArray] = [:]
        for (key, value) in weights where !key.contains("lm_head") {
            if key.hasPrefix("linear.") {
                result[key] = value
            } else {
                body[key.hasPrefix("backbone.") ? String(key.dropFirst(9)) : key] = value
            }
        }
        for (key, value) in backbone.sanitizeEmbeddingWeights(body) {
            result["backbone." + key] = value
        }
        return result
    }
}

struct ColIdefics3EmbeddingProcessor: UserInputProcessor {
    let tokenizer: any Tokenizer
    let config: Idefics3Configuration
    let settings: SmolVLMProcessorConfiguration
    let splitImages: Bool

    func prepare(input: UserInput) async throws -> LMInput {
        guard input.images.count == 1, input.videos.isEmpty, input.audios.isEmpty else {
            throw ModelExecutionError.invalidOptions(
                "ColIdefics3 visual embeddings require one image")
        }
        let helper = SmolVLMProcessor(settings, tokenizer: tokenizer)
        let image = try input.images[0].asCIImage().toSRGB()
        let size = config.visionConfig.imageSize
        let global = image.resampled(to: CGSize(width: size, height: size), method: .lanczos)
        var images = [global]
        let imagePrompt: String
        if splitImages, max(image.extent.width, image.extent.height) > CGFloat(size) {
            let (tiles, rows, columns) = helper.tiles(from: image)
            images = tiles + [global]
            imagePrompt = helper.getImagePromptString(
                rows: rows, cols: columns,
                seqLen: settings.imageSequenceLength, fakeToken: helper.fakeImageToken,
                imageToken: helper.imageToken, globalImageToken: helper.globalImageToken)
        } else {
            imagePrompt =
                helper.fakeImageToken + helper.globalImageToken
                + String(repeating: helper.imageToken, count: settings.imageSequenceLength)
                + helper.fakeImageToken
        }
        let messages = Qwen2VLMessageGenerator().generate(from: input)
        let prompt = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: input.additionalContext)
        let rendered = tokenizer.decode(tokenIds: prompt, skipSpecialTokens: false)
        guard rendered.components(separatedBy: "<image>").count == 2 else {
            throw ModelExecutionError.invalidOptions(
                "The ColIdefics3 chat template must contain one image placeholder")
        }
        let ids = tokenizer.encode(
            text: rendered.replacingOccurrences(of: "<image>", with: imagePrompt),
            addSpecialTokens: false)
        guard
            ids.filter({ $0 == config.imageTokenIndex }).count == images.count
                * settings.imageSequenceLength
        else {
            throw ModelExecutionError.invalidOptions(
                "Tokenizer is missing ColIdefics3 image tokens")
        }
        let pixels = concatenated(
            images.map {
                $0.normalized(mean: settings.imageMeanTuple, std: settings.imageStdTuple)
                    .asMLXArray()
            }, axis: 0
        ).transposed(0, 2, 3, 1)
        return LMInput(
            text: .init(tokens: MLXArray(ids).reshaped(1, -1)), image: .init(pixels: pixels))
    }
}
