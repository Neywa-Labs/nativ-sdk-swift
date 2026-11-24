import CoreImage
import Foundation
import MLX

struct Phi4SigLipProcessorConfiguration: Decodable, Sendable {
    let vision: Phi4SigLipConfiguration.VisionConfiguration
    let maxPatches: Int
    enum CodingKeys: String, CodingKey {
        case vision = "vision_config"
        case maxPatches = "max_num_patches"
    }
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vision = try c.decode(Phi4SigLipConfiguration.VisionConfiguration.self, forKey: .vision)
        maxPatches = try c.decodeIfPresent(Int.self, forKey: .maxPatches) ?? 3600
        guard maxPatches > 0, vision.patchSize > 0 else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "Invalid Phi4 SigLIP patch geometry"))
        }
    }
}

struct Phi4SigLipProcessor: UserInputProcessor {
    let config: Phi4SigLipProcessorConfiguration
    let tokenizer: any Tokenizer
    init(_ config: Phi4SigLipProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }
    func targetSize(height: Int, width: Int, maxPatches: Int) -> (Int, Int) {
        let p = config.vision.patchSize
        func aligned(_ n: Int, _ scale: Double) -> Int {
            max(p, Int(ceil(Double(n) * scale / Double(p))) * p)
        }
        var low = 1e-6
        var high = 100.0
        while high - low >= 1e-5 {
            let scale = (low + high) / 2
            if aligned(height, scale) / p * (aligned(width, scale) / p) <= maxPatches {
                low = scale
            } else {
                high = scale
            }
        }
        return (aligned(height, low), aligned(width, low))
    }
    func preprocess(_ image: CIImage, processing: UserInput.Processing) -> (MLXArray, THW) {
        let source = MediaProcessing.apply(image, processing: processing)
        let p = config.vision.patchSize
        let limit = max(1, min(config.maxPatches, (processing.maxPixels ?? Int.max) / (p * p)))
        let (h, w) = targetSize(
            height: Int(source.extent.height), width: Int(source.extent.width), maxPatches: limit)
        let srgb = MediaProcessing.asMLXArray(MediaProcessing.inSRGBToneCurveSpace(source))
        let normalized = (resizeRGBBilinear(srgb, size: (h, w)) - 0.5) / 0.5
        let patches = normalized.reshaped(3, h / p, p, w / p, p).transposed(1, 3, 2, 4, 0).reshaped(
            h / p * (w / p), 3 * p * p)
        let padded = MLXArray.zeros([config.maxPatches, 3 * p * p])
        padded[..<patches.dim(0)] = patches
        return (padded, THW(1, h / p, w / p))
    }
    func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty, input.audios.isEmpty else {
            throw VLMError.processing("Phi4 SigLIP accepts still images and text")
        }
        let images = MediaProcessing.images(from: input)
        let messages = try ImagePromptMessages.generate(
            input: input, imagePrompts: Array(repeating: "<image>", count: images.count),
            placeholder: "<image>")
        let encoded = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        if images.isEmpty { return LMInput(tokens: MLXArray(encoded)[.newAxis]) }
        // The placeholder is outside the language vocabulary. Split it after rendering
        // the template so it can be replaced by the model's negative sentinel.
        let rendered = tokenizer.decode(tokenIds: encoded, skipSpecialTokens: false)
        let pieces = rendered.components(separatedBy: "<image>")
        guard pieces.count == images.count + 1 else {
            throw VLMError.processing(
                "Phi4 SigLIP image placeholders do not match the supplied images")
        }
        var tokens: [Int] = []
        for (index, piece) in pieces.enumerated() {
            if index > 0 { tokens.append(-200) }
            tokens += tokenizer.encode(text: piece, addSpecialTokens: false)
        }
        let processed = try images.map {
            try preprocess($0.asCIImage(), processing: input.processing)
        }
        return LMInput(
            text: .init(tokens: MLXArray(tokens)[.newAxis]),
            image: processed.isEmpty
                ? nil
                : .init(pixels: stacked(processed.map { $0.0 }), frames: processed.map { $0.1 }))
    }
}
