import CoreImage
import Foundation
import MLX

struct DeepSeekV41Processor: UserInputProcessor {
    let config: DeepSeekV41Configuration
    let tokenizer: any Tokenizer

    init(_ config: DeepSeekV41Configuration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    static func tokenTypes(height: Int, width: Int) -> [DeepSeekV41ImageSpan.Code] {
        [.start]
            + (0 ..< height).flatMap { _ in Array(repeating: .image, count: width) + [.newline] }
            + [.end]
    }

    func targetSize(height: Int, width: Int) throws -> (Int, Int) {
        let vision = config.visionConfiguration
        let p = vision.patchSize
        let ratio = vision.downsampleRatio
        guard height > 0, width > 0, p > 0, ratio > 0, vision.maxImageTokens >= 4 else {
            throw VLMError.imageProcessingFailure("Invalid DeepSeek image geometry")
        }
        var h = height
        var w = width
        if let maximum = vision.maxWidthHeightRatio { w = min(w, Int(Float(h) * maximum)) }
        if h * w < vision.minPixels {
            let scale = sqrt(Double(vision.minPixels) / Double(h * w))
            h = Int(Double(h) * scale)
            w = Int(Double(w) * scale)
        }
        var bestH = (h + p - 1) / p * p
        var bestW = (w + p - 1) / p * p
        let cell = p * ratio
        let rows = (bestH + cell - 1) / cell
        let columns = (bestW + cell - 1) / cell
        if rows * (columns + 1) + 2 > vision.maxImageTokens {
            let aspect = Double(h) / Double(w)
            let maxW = sqrt(Double(vision.maxImageTokens - 2) / aspect + 0.25) - 0.5
            let maxH = maxW * aspect
            if maxW < 1 {
                bestH = (vision.maxImageTokens - 2) / 2 * cell
                bestW = cell
            } else if maxH < 1 {
                bestH = cell
                bestW = (vision.maxImageTokens - 3) * cell
            } else {
                let scale = min(
                    floor(maxW) * Double(cell) / Double(w), floor(maxH) * Double(cell) / Double(h))
                bestH = Int(floor(Double(h) * scale / Double(p))) * p
                bestW = Int(floor(Double(w) * scale / Double(p))) * p
            }
        }
        return (bestH, bestW)
    }

    func preprocess(_ image: CIImage, processing: UserInput.Processing) throws -> (MLXArray, THW) {
        let source = MediaProcessing.apply(image, processing: processing)
        let h = Int(source.extent.height)
        let w = Int(source.extent.width)
        let (targetH, targetW) = try targetSize(height: h, width: w)
        let stretch =
            config.visionConfiguration.maxWidthHeightRatio.map { Float(w) >= $0 * Float(h) }
            ?? false
        let scale = min(Double(targetH) / Double(h), Double(targetW) / Double(w))
        let resizedH =
            stretch ? targetH : max(1, Int((Double(h) * scale).rounded(.toNearestOrEven)))
        let resizedW =
            stretch ? targetW : max(1, Int((Double(w) * scale).rounded(.toNearestOrEven)))
        let image = MediaProcessing.resampleBicubic(
            MediaProcessing.inSRGBToneCurveSpace(source),
            to: CGSize(width: resizedW, height: resizedH))
        var pixels = MediaProcessing.asMLXArray(image)
        if resizedH != targetH || resizedW != targetW {
            let canvas = MLXArray.full(
                [1, 3, targetH, targetW], values: MLXArray(Float(127.0 / 255.0)))
            let y = (targetH - resizedH) / 2
            let x = (targetW - resizedW) / 2
            canvas[0..., 0..., y ..< y + resizedH, x ..< x + resizedW] = pixels
            pixels = canvas
        }
        let p = config.visionConfiguration.patchSize
        let gh = targetH / p
        let gw = targetW / p
        pixels = ((pixels - 0.5) / 0.5).reshaped(3, gh, p, gw, p).transposed(1, 3, 0, 2, 4)
            .reshaped(gh * gw, 3, p, p)
        return (pixels, THW(1, gh, gw))
    }

    func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty, input.audios.isEmpty else {
            throw VLMError.processing("DeepSeek V4.1 accepts still images and text.")
        }
        let tokens = try tokenizer.applyChatTemplate(
            messages: Qwen3VLMessageGenerator().generate(from: input), tools: input.tools,
            additionalContext: input.additionalContext)
        let images = MediaProcessing.images(from: input)
        guard tokens.filter({ $0 == config.imageTokenId }).count == images.count else {
            throw VLMError.processing(
                "DeepSeek image placeholders do not match the supplied images")
        }
        guard !images.isEmpty else { return LMInput(tokens: MLXArray(tokens)[.newAxis]) }
        let processed = try images.map {
            try preprocess($0.asCIImage(), processing: input.processing)
        }
        var expanded: [Int] = []
        var index = 0
        let ratio = config.visionConfiguration.downsampleRatio
        for token in tokens {
            if token == config.imageTokenId {
                let grid = processed[index].1
                let count = Self.tokenTypes(
                    height: (grid.h + ratio - 1) / ratio, width: (grid.w + ratio - 1) / ratio
                ).count
                expanded += Array(repeating: token, count: count)
                index += 1
            } else {
                expanded.append(token)
            }
        }
        return LMInput(
            text: .init(tokens: MLXArray(expanded)[.newAxis]),
            image: .init(
                pixels: concatenated(processed.map { $0.0 }), frames: processed.map { $0.1 }))
    }
}
