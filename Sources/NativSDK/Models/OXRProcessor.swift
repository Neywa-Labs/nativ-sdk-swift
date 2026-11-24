import CoreImage
import Foundation
import MLX

public struct OXRProcessorConfiguration: Decodable, Sendable {
    let patchSize: Int
    let hiddenStride: Int
    let minPixels: Int
    let maxPixels: Int
    let imageMean: [Float]
    let imageStd: [Float]

    private struct Settings: Decodable {
        var patchSize: Int?
        var hiddenStride: Int?
        var minPixels: Int?
        var maxPixels: Int?
        var imageMean: [Float]?
        var imageStd: [Float]?
        enum CodingKeys: String, CodingKey {
            case patchSize = "patch_size"
            case hiddenStride = "hidden_stride"
            case minPixels = "min_pixels"
            case maxPixels = "max_pixels"
            case imageMean = "image_mean"
            case imageStd = "image_std"
        }
    }
    private enum CodingKeys: String, CodingKey {
        case image = "image_processor"
        case vision = "vision_config"
        case vit = "vit_config"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let top = try Settings(from: decoder)
        let nested = try c.decodeIfPresent(Settings.self, forKey: .image)
        let vision =
            try c.decodeIfPresent(OXRConfiguration.VisionConfiguration.self, forKey: .vision)
            ?? c.decodeIfPresent(OXRConfiguration.VisionConfiguration.self, forKey: .vit)
        patchSize = nested?.patchSize ?? top.patchSize ?? vision?.patchSize ?? 16
        hiddenStride = nested?.hiddenStride ?? top.hiddenStride ?? vision?.hiddenStride ?? 2
        minPixels = nested?.minPixels ?? top.minPixels ?? 448 * 448
        maxPixels = nested?.maxPixels ?? top.maxPixels ?? 960 * 960
        imageMean = nested?.imageMean ?? top.imageMean ?? [0.5, 0.5, 0.5]
        imageStd = nested?.imageStd ?? top.imageStd ?? [0.5, 0.5, 0.5]
    }
}

extension OXRProcessorConfiguration: ModelConfigurationValidating {
    public func validateModelConfiguration() throws {
        guard patchSize > 0, hiddenStride > 0, minPixels > 0, maxPixels > 0,
            imageMean.count == 3, imageStd.count == 3,
            imageMean.allSatisfy(\.isFinite), imageStd.allSatisfy({ $0.isFinite && $0 > 0 })
        else { throw VLMError.processing("Invalid OXR image processor configuration") }
    }
}

public struct OXRProcessor: UserInputProcessor {
    let config: OXRProcessorConfiguration
    let tokenizer: any Tokenizer
    public init(_ config: OXRProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    static func targetSize(height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int)
        throws -> (Int, Int)
    {
        guard height > 0, width > 0, factor > 0, minPixels > 0, maxPixels > 0 else {
            throw VLMError.imageProcessingFailure(
                "OXR image dimensions and pixel budgets must be positive")
        }
        var h = Double(height)
        var w = Double(width)
        let f = Double(factor)
        if h < f || w < f {
            if h < w {
                w = (f / h * w).rounded(.toNearestOrEven)
                h = f
            } else {
                h = (f / w * h).rounded(.toNearestOrEven)
                w = f
            }
        } else if max(h, w) / min(h, w) > 200 {
            if h > w { h = 200 * w } else { w = 200 * h }
        }
        var rh = (h / f).rounded(.toNearestOrEven) * f
        var rw = (w / f).rounded(.toNearestOrEven) * f
        if rh * rw > Double(maxPixels) {
            let beta = sqrt(h * w / Double(maxPixels))
            rh = floor(h / beta / f) * f
            rw = floor(w / beta / f) * f
        } else if rh * rw < Double(min(minPixels, maxPixels)) {
            let beta = sqrt(Double(min(minPixels, maxPixels)) / (h * w))
            rh = ceil(h * beta / f) * f
            rw = ceil(w * beta / f) * f
        }
        guard rh >= f, rw >= f, rh < Double(Int.max), rw < Double(Int.max) else {
            throw VLMError.imageProcessingFailure(
                "OXR pixel budget is too small for this aspect ratio")
        }
        return (Int(rh), Int(rw))
    }

    func preprocess(_ image: CIImage, processing: UserInput.Processing) throws -> (MLXArray, THW) {
        let applied = MediaProcessing.apply(image, processing: processing)
        let extent = applied.extent
        guard extent.width.isFinite, extent.height.isFinite, extent.width >= 1, extent.height >= 1,
            extent.width < CGFloat(Int.max), extent.height < CGFloat(Int.max)
        else { throw VLMError.imageProcessingFailure("OXR requires a finite nonempty image") }
        let width = Int(extent.width)
        let height = Int(extent.height)
        let (rh, rw) = try Self.targetSize(
            height: height, width: width,
            factor: config.patchSize * config.hiddenStride,
            minPixels: processing.minPixels ?? config.minPixels,
            maxPixels: processing.maxPixels ?? config.maxPixels)
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let context = CIContext(options: [.cacheIntermediates: false])
        context.render(
            applied, toBitmap: &rgba, rowBytes: width * 4, bounds: extent,
            format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        // Rendered RGBA bytes are premultiplied; composite over white in sRGB byte space.
        for pixel in 0 ..< width * height {
            for channel in 0 ..< 3 {
                rgb[pixel * 3 + channel] = UInt8(
                    clamping: Int(rgba[pixel * 4 + channel]) + 255 - Int(rgba[pixel * 4 + 3]))
            }
        }
        let resized = PillowImageProcessing.resizeRGB(
            rgb, width: width, height: height, targetWidth: rw, targetHeight: rh)
        let pixels = MLXArray(resized, [rh, rw, 3]).asType(.float32) / Float(255)
        let normalized = (pixels - MLXArray(config.imageMean)) / MLXArray(config.imageStd)
        return try QwenVL.patchify(
            images: [normalized.transposed(2, 0, 1)[.newAxis]],
            mergeSize: config.hiddenStride, patchSize: config.patchSize, temporalPatchSize: 1)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty, input.audios.isEmpty, input.tools?.isEmpty != false else {
            throw VLMError.processing("OXR accepts images and text without tools")
        }
        try Task.checkCancellation()
        let images = try input.images.map {
            try preprocess($0.asCIImage(), processing: input.processing)
        }
        let counts = images.map { $0.1.product / (config.hiddenStride * config.hiddenStride) }
        var index = 0
        func expand(_ text: String) throws -> String {
            let parts = text.components(separatedBy: "<image>")
            var result = parts[0]
            for part in parts.dropFirst() {
                guard index < counts.count else {
                    throw VLMError.processing("More OXR image placeholders than images")
                }
                result +=
                    "<|vision_start|>" + String(repeating: "<|image_pad|>", count: counts[index])
                    + "<|vision_end|>" + part
                index += 1
            }
            return result
        }
        var messages: [Message]
        func content(_ text: String, count: Int) -> String {
            text.contains("<image>") ? text : String(repeating: "<image>", count: count) + text
        }
        switch input.prompt {
        case .text(let text):
            messages = [["role": "user", "content": content(text, count: images.count)]]
        case .chat(let chat):
            messages = chat.map { message in
                [
                    "role": message.role.rawValue,
                    "content": content(message.content, count: message.images.count),
                ]
            }
        case .messages(let raw): messages = raw
        }
        for i in messages.indices {
            if let text = messages[i]["content"] as? String {
                messages[i]["content"] = try expand(text)
            } else if let parts = messages[i]["content"] as? [[String: any Sendable]] {
                messages[i]["content"] = try parts.map { part in
                    let type = part["type"] as? String
                    if type == "image" || type == "image_url" { return try expand("<image>") }
                    if type == "text" { return try expand(part["text"] as? String ?? "") }
                    throw VLMError.processing("Unsupported OXR message content")
                }.joined()
            } else {
                throw VLMError.processing("Invalid OXR message content")
            }
        }
        guard index == images.count else {
            throw VLMError.processing("OXR images and placeholders do not match")
        }
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: nil, additionalContext: input.additionalContext)
        guard !tokens.isEmpty else { throw VLMError.processing("OXR prompt produced no tokens") }
        try Task.checkCancellation()
        return LMInput(
            text: .init(tokens: MLXArray(tokens, [1, tokens.count])),
            image: images.isEmpty
                ? nil : .init(pixels: concatenated(images.map(\.0)), frames: images.map(\.1)))
    }
}
