// MiniCPM-OCR Jev follows OpenBMB's Apache-2.0 scoring contract.
// See Documentation/Models/MiniCPMOCRJev/README.md for attribution.
import CoreImage
import Foundation
import MLX
import MLXNN

struct MiniCPMOCRJevConfiguration: Decodable {
    let backbone: Qwen35Configuration
    let padTokenId: Int

    private enum CodingKeys: String, CodingKey {
        case padTokenId = "pad_token_id"
        case numLabels = "num_labels"
        case id2label
    }

    init(from decoder: Decoder) throws {
        backbone = try Qwen35Configuration(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 248_044
        let labels =
            try c.decodeIfPresent(Int.self, forKey: .numLabels)
            ?? c.decodeIfPresent([String: String].self, forKey: .id2label)?.count ?? 2
        guard labels == 2 else {
            throw ModelExecutionError.invalidOptions("MiniCPM-OCR Jev requires two labels")
        }
    }

    static func matches(_ data: Data) -> Bool {
        guard let c = try? JSONDecoder.json5().decode([String: JSONValue].self, from: data)
        else { return false }
        if c["model_type"] == .string("minicpm_ocr_jev") { return true }
        let sequenceClassifier: Bool
        if case .array(let architectures) = c["architectures"] {
            sequenceClassifier = architectures.contains(.string("Qwen3_5ForSequenceClassification"))
        } else {
            sequenceClassifier = false
        }
        return c["model_type"] == .string("qwen3_5")
            && (c["problem_type"] == .string("single_label_classification")
                || sequenceClassifier)
    }
}

/// A sequence classifier, with no vocabulary readout or generation interface.
final class MiniCPMOCRJev: Module, BaseLanguageModel {
    let backbone: Qwen35
    @ModuleInfo(key: "score") var scoreHead: Linear
    let configuration: MiniCPMOCRJevConfiguration

    init(_ configuration: MiniCPMOCRJevConfiguration) throws {
        self.configuration = configuration
        backbone = Qwen35(configuration.backbone)
        _scoreHead.wrappedValue = Linear(
            configuration.backbone.textConfiguration.hiddenSize, 2, bias: false)
        super.init()
        try backbone.languageModel.update(
            modules: ModuleChildren(values: ["lm_head": .none]), verify: .noUnusedKeys)
        let norms: [(String, Module)] = backbone.languageModel.namedModules().compactMap {
            path, module in
            guard let norm = module as? RMSNorm else { return nil }
            return (
                path,
                MiniCPMOCRRMSNorm(
                    dimensions: norm.weight.size,
                    eps: configuration.backbone.textConfiguration.rmsNormEps)
            )
        }
        try backbone.languageModel.update(
            modules: ModuleChildren.unflattened(norms), verify: .noUnusedKeys)
    }

    func logits(_ input: LMInput) throws -> MLXArray {
        let ids = input.text.tokens
        guard ids.ndim == 2, ids.dim(0) > 0, ids.dim(1) > 0,
            input.text.mask == nil || input.text.mask?.shape == ids.shape,
            input.video == nil, input.audio == nil
        else {
            throw ModelExecutionError.invalidOptions(
                "Jev requires nonempty batched tokens and a matching mask")
        }
        if input.image != nil, ids.dim(0) != 1 {
            throw ModelExecutionError.invalidOptions("Jev scores one image pair per request")
        }
        if let image = input.image {
            let vision = configuration.backbone.visionConfiguration
            guard let frames = image.frames, frames.count == 2,
                frames.allSatisfy({
                    $0.t == 1 && $0.h > 0 && $0.w > 0
                        && $0.h.isMultiple(of: vision.spatialMergeSize)
                        && $0.w.isMultiple(of: vision.spatialMergeSize)
                }),
                image.pixels.ndim == 2,
                image.pixels.dim(0) == frames.reduce(0, { $0 + $1.product }),
                image.pixels.dim(1) == vision.inChannels * vision.temporalPatchSize
                    * vision.patchSize * vision.patchSize
            else {
                throw ModelExecutionError.invalidOptions("Jev requires two valid image patch grids")
            }
        }
        let rows = ids.asArray(Int.self)
        guard rows.allSatisfy({ $0 >= 0 && $0 < configuration.backbone.vocabSize }) else {
            throw ModelExecutionError.invalidOptions("Jev token IDs exceed the vocabulary")
        }
        if let mask = input.text.mask,
            !all((mask .== 0) .|| (mask .== 1)).item(Bool.self)
        {
            throw ModelExecutionError.invalidOptions("Jev attention masks must be binary")
        }
        let mask = input.text.mask?.asArray(Int.self)
        var ranges = [Range<Int>]()
        for row in 0 ..< ids.dim(0) {
            let offset = row * ids.dim(1)
            let valid = (0 ..< ids.dim(1)).filter { index in
                let i = offset + index
                return mask.map { $0[i] != 0 } ?? (rows[i] != configuration.padTokenId)
            }
            guard let first = valid.first, let last = valid.last,
                valid.count == last - first + 1,
                (first ... last).contains(where: { rows[offset + $0] != configuration.padTokenId })
            else {
                throw ModelExecutionError.invalidOptions(
                    "Jev requires contiguous non-padding tokens")
            }
            ranges.append(first ..< last + 1)
        }
        // Encode images before slicing each row; the visual placeholders index this full input.
        let (_, images, videos, embeddings) = try backbone.visionInputEmbeddings(input)
        let pooled = ranges.enumerated().map { row, range in
            let rowIDs = ids[row ..< row + 1, range]
            // Position the unpadded sequence as well as trimming its recurrent inputs.
            let (positions, _) = Qwen3VLLanguage.getRopeIndex(
                inputIds: rowIDs, imageGridTHW: images, videoGridTHW: videos,
                spatialMergeSize: configuration.backbone.visionConfiguration.spatialMergeSize,
                imageTokenId: configuration.backbone.imageTokenId,
                videoTokenId: configuration.backbone.videoTokenId,
                visionStartTokenId: configuration.backbone.visionStartTokenId)
            let hidden = backbone.languageModel.model(
                rowIDs,
                inputsEmbeds: embeddings.map { $0[row ..< row + 1, range, 0...] },
                positionIds: positions)
            let last = range.last { rows[row * ids.dim(1) + $0] != configuration.padTokenId }!
            return hidden[0, last - range.lowerBound]
        }
        return scoreHead(stacked(pooled)).asType(.float32)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        sanitize(weights: weights, metadata: [:])
    }

    func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray] {
        var unwrapped: [String: MLXArray] = [:]
        var head: [String: MLXArray] = [:]
        for (key, original) in weights {
            if key.hasPrefix("score.") {
                head[key] = original
                continue
            }
            let key = key.hasPrefix("backbone.") ? String(key.dropFirst(9)) : key
            guard !key.contains("lm_head.") else { continue }
            unwrapped[key] =
                MiniCPMV.normWeightSuffixes.contains(where: key.hasSuffix)
                ? original.asType(.float32) : original
        }
        for (key, value) in backbone.sanitize(weights: unwrapped, metadata: metadata) {
            head["backbone." + key] = value
        }
        return head
    }

    func quantizationPredicate(path: String, module: Module) -> QuantizationDecision {
        path == "score" ? .skip : .useDefault
    }
}

struct MiniCPMOCRJevProcessor: UserInputProcessor {
    let config: Qwen3VLProcessorConfiguration
    let tokenizer: any Tokenizer

    static func prompt(text: String) -> String {
        "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>\nOCR文本：\n\(text)\n"
            + "<|vision_start|><|image_pad|><|vision_end|>\n"
            + "第一张图是原图，第二张图是上述 OCR 文本渲染得到的图片。\n"
            + "请判断该 OCR 文本及其对应渲染图是否与原图内容一致。\n"
            + "这是一个二分类任务：一致标为 1，不一致标为 0。<|im_end|>\n"
            + "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    }

    static func padExtremeAspectRatio(_ source: CIImage) throws -> CIImage {
        let extent = source.extent
        guard extent.width.isFinite, extent.height.isFinite, extent.width >= 1, extent.height >= 1
        else {
            throw VLMError.imageProcessingFailure("Jev requires a finite nonempty image")
        }
        let width = Int(extent.width)
        let height = Int(extent.height)
        let image = source.transformed(by: .init(translationX: -extent.minX, y: -extent.minY))
        guard Double(max(width, height)) / Double(min(width, height)) > 199 else { return image }
        let w = max(width, Int(ceil(Double(height) / 199)))
        let h = max(height, Int(ceil(Double(width) / 199)))
        return image.transformed(
            by: .init(translationX: CGFloat((w - width) / 2), y: CGFloat((h - height) / 2))
        )
        .composited(
            over: CIImage(color: .white).cropped(to: CGRect(x: 0, y: 0, width: w, height: h)))
    }

    func prepare(input: UserInput) async throws -> LMInput {
        guard case .text(let text) = input.prompt, input.images.count == 2,
            input.videos.isEmpty, input.audios.isEmpty, input.tools?.isEmpty != false
        else {
            throw ModelExecutionError.invalidOptions(
                "Jev requires candidate text, the original crop, and its rendering")
        }
        try Task.checkCancellation()
        var processing = input.processing
        // Match the checkpoint's pixel budget, rather than the general Qwen VQA default cap.
        processing.maxPixels = processing.maxPixels ?? config.maxPixels
        let processor = Qwen3VLProcessor(config, tokenizer: tokenizer)
        let images = try input.images.map {
            try processor.preprocess(
                images: [Self.padExtremeAspectRatio($0.asCIImage())], processing: processing)
        }
        let tokens = try QwenVL.replacePaddingTokens(
            in: tokenizer.encode(text: Self.prompt(text: text), addSpecialTokens: false),
            frames: images.map(\.1), paddingToken: "<|image_pad|>", mergeSize: config.mergeSize,
            tokenizer: tokenizer)
        return LMInput(
            text: .init(tokens: MLXArray(tokens, [1, tokens.count])),
            image: .init(pixels: concatenated(images.map(\.0)), frames: images.map(\.1)))
    }
}
