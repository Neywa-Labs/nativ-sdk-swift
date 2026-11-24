// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: OpenMDW-1.1
import CoreImage
import Foundation
import MLX

struct NemotronParse35Processor: UserInputProcessor {
    static let defaultPrompt =
        "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
    let config: NemotronParseProcessorConfiguration
    let tokenizer: any Tokenizer
    init(_ config: NemotronParseProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }
    func preprocess(_ image: CIImage, processing: UserInput.Processing) throws
        -> LMInput.ProcessedImage
    {
        let image = MediaProcessing.apply(image, processing: processing)
        let extent = image.extent
        guard extent.width.isFinite, extent.height.isFinite, extent.width >= 1, extent.height >= 1
        else {
            throw VLMError.imageProcessingFailure(
                "Nemotron 3.5 Parse requires a finite nonempty page")
        }
        let width = Int(extent.width)
        let height = Int(extent.height)
        var h = height
        var w = width
        let ratio = Double(width) / Double(height)
        if h > config.finalSize[0] {
            h = config.finalSize[0]
            w = Int(Double(h) * ratio)
        }
        if w > config.finalSize[1] {
            w = config.finalSize[1]
            h = Int(Double(w) / ratio)
        }
        guard h > 0, w > 0 else {
            throw VLMError.imageProcessingFailure(
                "Page aspect ratio exceeds the Nemotron Parse image budget")
        }
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let context = CIContext(options: [.cacheIntermediates: false])
        context.render(
            image, toBitmap: &rgba, rowBytes: width * 4, bounds: extent, format: .RGBA8,
            colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for pixel in 0 ..< width * height {
            for channel in 0 ..< 3 {
                let alpha = Int(rgba[pixel * 4 + 3])
                // The reference accepts RGB pages; discard alpha rather than compositing it.
                rgb[pixel * 3 + channel] =
                    alpha == 0
                    ? 0
                    : UInt8(clamping: (Int(rgba[pixel * 4 + channel]) * 255 + alpha / 2) / alpha)
            }
        }
        let resized = PillowImageProcessing.resizeRGB(
            rgb, width: width, height: height, targetWidth: w, targetHeight: h)
        let ph = (h + 15) / 16 * 16
        let pw = (w + 63) / 64 * 64
        let canvas = MLXArray.full([ph, pw, 3], values: MLXArray(UInt8(255)))
        canvas[..<h, ..<w, 0...] = MLXArray(resized, [h, w, 3])
        let normalized =
            (canvas.asType(.float32) / Float(255) - MLXArray(config.imageMean))
            / MLXArray(config.imageStd)
        let patches = normalized.transposed(2, 0, 1).reshaped(3, ph / 16, 16, pw / 16, 16)
            .transposed(1, 3, 0, 2, 4).reshaped(-1, 768)
        return .init(
            pixels: patches, frames: [THW(1, ph / 16, pw / 16)],
            contentSizes: [CGSize(width: w, height: h)])
    }
    func prepare(input: UserInput) async throws -> LMInput {
        guard input.images.count == 1, input.videos.isEmpty, input.audios.isEmpty,
            input.tools?.isEmpty != false
        else {
            throw VLMError.processing(
                "Nemotron 3.5 Parse requires one page without audio, video, or tools")
        }
        let prompt: String
        switch input.prompt {
        case .text(let text): prompt = text
        case .chat(let chat):
            guard chat.count == 1, chat[0].role == .user else {
                throw VLMError.processing("Nemotron 3.5 Parse takes a raw page task prompt")
            }
            prompt = chat[0].content
        case .messages(let messages):
            guard messages.count == 1, let text = messages[0]["content"] as? String else {
                throw VLMError.processing("Nemotron 3.5 Parse takes a raw page task prompt")
            }
            prompt = text
        }
        try Task.checkCancellation()
        let tokens = tokenizer.encode(
            text: prompt.isEmpty ? Self.defaultPrompt : prompt, addSpecialTokens: false)
        guard !tokens.isEmpty else { throw VLMError.processing("Empty Nemotron Parse task prompt") }
        let image = try preprocess(input.images[0].asCIImage(), processing: input.processing)
        try Task.checkCancellation()
        return LMInput(text: .init(tokens: MLXArray(tokens, [1, tokens.count])), image: image)
    }
}
