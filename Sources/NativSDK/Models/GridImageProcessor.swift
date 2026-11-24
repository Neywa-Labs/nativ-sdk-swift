import CoreImage
import Foundation
import MLX

/// Shared patch packing for models whose vision encoder consumes a merged spatial grid.
struct GridImageProcessorConfiguration: Decodable, Sendable {
    let modelType: String
    let patchSize: Int
    let temporalPatchSize: Int
    let mergeSize: Int
    let patchExpandFactor: Int
    let minPixels: Int
    let maxPixels: Int
    let maxLongSide: Int?
    let maxTotalPixels: Int?
    let minImageTokens: Int
    let maxImageTokens: Int
    let imageMean: [Float]
    let imageStd: [Float]
    let imageTokenID: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case mergeSize = "merge_size"
        case patchExpandFactor = "patch_expand_factor"
        case minPixels = "min_pixels"
        case maxPixels = "max_pixels"
        case maxLongSide = "max_long_side_pixel"
        case maxTotalPixels = "max_total_pixels"
        case minImageTokens = "min_image_tokens"
        case maxImageTokens = "max_image_tokens"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case imageTokenID = "image_token_id"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        let compass = modelType == "cohere_compass"
        let hunyuan = modelType == "hunyuan_vl"
        patchSize =
            try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? (compass || hunyuan ? 16 : 14)
        temporalPatchSize =
            try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? (hunyuan ? 1 : 2)
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
        patchExpandFactor = try c.decodeIfPresent(Int.self, forKey: .patchExpandFactor) ?? 1
        minPixels =
            try c.decodeIfPresent(Int.self, forKey: .minPixels)
            ?? (compass ? 16384 : hunyuan ? 262144 : 3136)
        maxPixels =
            try c.decodeIfPresent(Int.self, forKey: .maxPixels)
            ?? (compass ? 3_868_706 : hunyuan ? 4_194_304 : 451584)
        maxLongSide = try c.decodeIfPresent(Int.self, forKey: .maxLongSide)
        maxTotalPixels = try c.decodeIfPresent(Int.self, forKey: .maxTotalPixels)
        minImageTokens = try c.decodeIfPresent(Int.self, forKey: .minImageTokens) ?? 16
        maxImageTokens = try c.decodeIfPresent(Int.self, forKey: .maxImageTokens) ?? 8000
        imageMean =
            try c.decodeIfPresent([Float].self, forKey: .imageMean)
            ?? (compass ? [0.5, 0.5, 0.5] : [0.48145466, 0.4578275, 0.40821073])
        imageStd =
            try c.decodeIfPresent([Float].self, forKey: .imageStd)
            ?? (compass ? [0.5, 0.5, 0.5] : [0.26862954, 0.26130258, 0.27577711])
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID)
        guard patchSize > 0, temporalPatchSize > 0, mergeSize > 0, patchExpandFactor > 0,
            minPixels > 0, maxPixels >= minPixels, minImageTokens > 0,
            maxImageTokens >= minImageTokens, imageMean.count == 3, imageStd.count == 3,
            imageStd.allSatisfy({ $0 > 0 })
        else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "Invalid image grid geometry or normalization"))
        }
    }
}

struct GridImageProcessor: UserInputProcessor {
    let config: GridImageProcessorConfiguration
    let tokenizer: any Tokenizer

    init(_ config: GridImageProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    var imageToken: String {
        switch config.modelType {
        case "cohere_compass": "<|IMAGE_PAD|>"
        case "hunyuan_vl": tokenizer.convertIdToToken(config.imageTokenID ?? 120120) ?? "<image>"
        case "glm5_next": "<|image|>"
        default: "]<]image[>["
        }
    }

    func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty, input.audios.isEmpty else {
            throw VLMError.processing("This processor accepts still images and text.")
        }
        let messages = Qwen3VLMessageGenerator().generate(from: input)
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        let images = MediaProcessing.images(from: input)
        guard !images.isEmpty else { return LMInput(tokens: MLXArray(tokens)[.newAxis]) }
        let processed = try images.map {
            try preprocess($0.asCIImage(), processing: input.processing)
        }
        guard let tokenID = config.imageTokenID ?? tokenizer.convertTokenToId(imageToken) else {
            throw VLMError.processing("Missing image token \(imageToken)")
        }
        let counts = processed.map { item in
            let grid = item.1
            if config.modelType == "hunyuan_vl" {
                return grid.h / config.mergeSize * (grid.w / config.mergeSize + 1) + 2
            }
            return grid.product / (config.mergeSize * config.mergeSize)
        }
        let expanded = try expand(tokens: tokens, imageTokenID: tokenID, counts: counts)
        return LMInput(
            text: .init(tokens: MLXArray(expanded)[.newAxis]),
            image: .init(
                pixels: concatenated(processed.map { $0.0 }), frames: processed.map { $0.1 }))
    }

    func expand(tokens: [Int], imageTokenID: Int, counts: [Int]) throws -> [Int] {
        guard tokens.filter({ $0 == imageTokenID }).count == counts.count else {
            throw VLMError.processing(
                "The chat template must emit one image placeholder per image.")
        }
        var output: [Int] = []
        var index = 0
        for (position, token) in tokens.enumerated() {
            guard token == imageTokenID else {
                output.append(token)
                continue
            }
            let minimax = config.modelType.hasPrefix("minimax_m3")
            let start = tokenizer.convertTokenToId("]<]start of image[>[")
            let end = tokenizer.convertTokenToId("]<]end of image[>[")
            if minimax, let start, position == 0 || tokens[position - 1] != start {
                output.append(start)
            }
            output += Array(repeating: token, count: counts[index])
            if minimax, let end, position + 1 == tokens.count || tokens[position + 1] != end {
                output.append(end)
            }
            index += 1
        }
        return output
    }

    func geometry(height: Int, width: Int, processing: UserInput.Processing) throws -> (
        height: Int, width: Int, contentHeight: Int, contentWidth: Int
    ) {
        guard height > 0, width > 0 else { throw VLMError.imageProcessingFailure("Empty image") }
        let factor = config.patchSize * config.mergeSize * config.patchExpandFactor
        if config.modelType == "glm5_next" {
            let minArea = config.minImageTokens * factor * factor
            let maxArea = min(
                config.maxImageTokens * factor * factor, processing.maxPixels ?? Int.max)
            func align(_ n: Int) -> Int { (n + factor - 1) / factor * factor }
            var h = align(height)
            var w = align(width)
            if h * w < minArea {
                let scale = sqrt(Double(minArea) / Double(height * width))
                h = align(Int(ceil(Double(height) * scale)))
                w = align(Int(ceil(Double(width) * scale)))
            }
            if h * w > maxArea {
                guard maxArea >= factor * factor else {
                    throw VLMError.imageProcessingFailure(
                        "Image budget is smaller than one merged patch")
                }
                var low = 1
                var high = height
                h = factor
                w = factor
                while low <= high {
                    let candidateH = (low + high) / 2
                    let candidateW = max(1, width * candidateH / height)
                    if align(candidateH) * align(candidateW) <= maxArea {
                        h = align(candidateH)
                        w = align(candidateW)
                        low = candidateH + 1
                    } else {
                        high = candidateH - 1
                    }
                }
            }
            var scale = min(Double(h) / Double(height), Double(w) / Double(width))
            if height * width >= minArea { scale = min(1, scale) }
            return (
                h, w, max(1, min(h, Int(Double(height) * scale))),
                max(1, min(w, Int(Double(width) * scale)))
            )
        }
        let h: Int
        let w: Int
        if let longSide = config.maxLongSide, config.modelType.hasPrefix("minimax_m3") {
            let scale =
                max(height, width) > longSide
                ? Double(longSide) / Double(max(height, width))
                : min(height, width) < 112 ? 112.0 / Double(min(height, width)) : 1
            h = max(
                factor,
                Int((Double(height) * scale / Double(factor)).rounded(.toNearestOrEven)) * factor)
            w = max(
                factor,
                Int((Double(width) * scale / Double(factor)).rounded(.toNearestOrEven)) * factor)
        } else {
            (h, w) = try QwenVL.targetSize(
                height: height, width: width, factor: factor,
                minPixels: processing.minPixels ?? config.minPixels,
                maxPixels: processing.maxPixels ?? config.maxPixels)
        }
        if let maximum = config.maxTotalPixels, h * w > maximum {
            throw VLMError.imageProcessingFailure("Image exceeds total pixel budget")
        }
        return (h, w, h, w)
    }

    func preprocess(_ image: CIImage, processing: UserInput.Processing) throws -> (MLXArray, THW) {
        let image = MediaProcessing.apply(image, processing: processing)
        let shape = try geometry(
            height: Int(image.extent.height), width: Int(image.extent.width), processing: processing
        )
        let resized = MediaProcessing.resampleBicubic(
            MediaProcessing.inSRGBToneCurveSpace(image),
            to: CGSize(width: shape.contentWidth, height: shape.contentHeight))
        var pixels = MediaProcessing.asMLXArray(resized)
        if shape.contentHeight != shape.height || shape.contentWidth != shape.width {
            let canvas = MLXArray.zeros([1, 3, shape.height, shape.width])
            canvas[0..., 0..., ..<shape.contentHeight, ..<shape.contentWidth] = pixels
            pixels = canvas
        }
        pixels =
            (pixels - MLXArray(config.imageMean).reshaped(1, 3, 1, 1))
            / MLXArray(config.imageStd).reshaped(1, 3, 1, 1)
        return try QwenVL.patchify(
            images: [pixels], mergeSize: config.mergeSize, patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize)
    }
}
