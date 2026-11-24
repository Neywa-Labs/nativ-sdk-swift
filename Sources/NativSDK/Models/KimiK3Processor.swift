import CoreImage
import Foundation
import MLX

struct KimiK3ProcessorConfiguration: Decodable, Sendable {
    let vision: KimiK3Configuration.VisionConfiguration
    let patchLimit: Int
    let sideLimit: Int
    let imageMean: [Float]
    let imageStd: [Float]
    enum CodingKeys: String, CodingKey {
        case vision = "vision_config"
        case patchLimit = "in_patch_limit"
        case sideLimit = "patch_limit_on_one_side"
        case imageMean = "image_mean"
        case imageStd = "image_std"
    }
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vision = try c.decode(KimiK3Configuration.VisionConfiguration.self, forKey: .vision)
        patchLimit = try c.decodeIfPresent(Int.self, forKey: .patchLimit) ?? 65536
        sideLimit = try c.decodeIfPresent(Int.self, forKey: .sideLimit) ?? 512
        imageMean = try c.decodeIfPresent([Float].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd = try c.decodeIfPresent([Float].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        guard patchLimit > 0, sideLimit > 0, vision.patchSize > 0,
            vision.mergeKernelSize.count == 2, vision.mergeKernelSize.allSatisfy({ $0 > 0 }),
            imageMean.count == 3, imageStd.count == 3, imageStd.allSatisfy({ $0 > 0 })
        else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath, debugDescription: "Invalid Kimi image geometry")
            )
        }
    }
}

struct KimiK3Processor: UserInputProcessor {
    let config: KimiK3ProcessorConfiguration
    let tokenizer: any Tokenizer
    init(_ config: KimiK3ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }
    func preprocess(_ image: CIImage, processing: UserInput.Processing) -> (MLXArray, THW) {
        let source = MediaProcessing.apply(image, processing: processing)
        let h = Int(source.extent.height)
        let w = Int(source.extent.width)
        let p = config.vision.patchSize
        let limit = min(config.patchLimit, (processing.maxPixels ?? Int.max) / (p * p))
        let scale = min(
            1, sqrt(Double(limit) / Double(max(1, w / p) * max(1, h / p))),
            Double(config.sideLimit * p) / Double(w), Double(config.sideLimit * p) / Double(h))
        let contentH = max(1, Int(Double(h) * scale))
        let contentW = max(1, Int(Double(w) * scale))
        let factorH = p * config.vision.mergeKernelSize[0]
        let factorW = p * config.vision.mergeKernelSize[1]
        let targetH = (contentH + factorH - 1) / factorH * factorH
        let targetW = (contentW + factorW - 1) / factorW * factorW
        let resized = MediaProcessing.resampleBicubic(
            MediaProcessing.inSRGBToneCurveSpace(source),
            to: CGSize(width: contentW, height: contentH))
        let canvas = MLXArray.zeros([1, 3, targetH, targetW])
        canvas[0..., 0..., ..<contentH, ..<contentW] = MediaProcessing.asMLXArray(resized)
        let normalized =
            (canvas - MLXArray(config.imageMean).reshaped(1, 3, 1, 1))
            / MLXArray(config.imageStd).reshaped(1, 3, 1, 1)
        let gh = targetH / p
        let gw = targetW / p
        return (
            normalized.reshaped(3, gh, p, gw, p).transposed(1, 3, 0, 2, 4).reshaped(
                gh * gw, 3, p, p), THW(1, gh, gw)
        )
    }
    func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty, input.audios.isEmpty else {
            throw VLMError.processing("Kimi K3 accepts still images and text")
        }
        let images = MediaProcessing.images(from: input)
        let processed = try images.map {
            try preprocess($0.asCIImage(), processing: input.processing)
        }
        var prompts: [String] = []
        for (image, result) in zip(images, processed) {
            let size = try image.asCIImage().extent.size
            let count = result.1.product / config.vision.mergeUnit
            prompts.append(
                "<|media_begin|>image \(Int(size.width))x\(Int(size.height))<|media_content|>"
                    + String(repeating: "<|media_pad|>", count: count) + "<|media_end|>")
        }
        let messages = try ImagePromptMessages.generate(input: input, imagePrompts: prompts)
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        return LMInput(
            text: .init(tokens: MLXArray(tokens)[.newAxis]),
            image: processed.isEmpty
                ? nil
                : .init(
                    pixels: concatenated(processed.map { $0.0 }), frames: processed.map { $0.1 }))
    }
}

/// String-content templates need the complete image span before tokenization.
enum ImagePromptMessages {
    static func generate(input: UserInput, imagePrompts: [String], placeholder: String? = nil)
        throws -> [Message]
    {
        var next = 0
        func content(_ text: String, count: Int) throws -> String {
            guard next + count <= imagePrompts.count else {
                throw VLMError.processing("Image count does not match the chat")
            }
            let prompts = Array(imagePrompts[next ..< next + count])
            next += count
            if let placeholder, text.contains(placeholder) {
                let pieces = text.components(separatedBy: placeholder)
                guard pieces.count == prompts.count + 1 else {
                    throw VLMError.processing("Image placeholder count does not match the chat")
                }
                return zip(pieces.dropLast(), prompts).map { $0.0 + $0.1 }.joined() + pieces.last!
            }
            return prompts.joined(separator: "\n") + (prompts.isEmpty ? "" : "\n") + text
        }
        let messages: [Message]
        switch input.prompt {
        case .text(let text):
            messages = [["role": "user", "content": try content(text, count: imagePrompts.count)]]
        case .chat(let chat):
            messages = try chat.map { message in
                var result = DefaultMessageGenerator().generate(message: message)
                result["content"] = try content(message.content, count: message.images.count)
                return result
            }
        case .messages(let raw):
            guard imagePrompts.isEmpty else {
                throw VLMError.processing("Use structured chat for image input with this processor")
            }
            messages = raw
        }
        guard next == imagePrompts.count else { throw VLMError.processing("Unused image input") }
        return messages
    }
}
