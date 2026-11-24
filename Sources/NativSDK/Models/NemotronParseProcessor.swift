import CoreImage
import Foundation
import MLX

struct NemotronParseProcessorConfiguration: Decodable, Sendable {
    let finalSize: [Int]
    let imageMean: [Float]
    let imageStd: [Float]
    enum CodingKeys: String, CodingKey {
        case finalSize = "final_size"
        case imageMean = "image_mean"
        case imageStd = "image_std"
    }
    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        finalSize = try c.decodeIfPresent([Int].self, forKey: .finalSize) ?? [2048, 1664]
        imageMean =
            try c.decodeIfPresent([Float].self, forKey: .imageMean) ?? [
                0.48145466, 0.4578275, 0.40821073,
            ]
        imageStd =
            try c.decodeIfPresent([Float].self, forKey: .imageStd) ?? [
                0.26862954, 0.26130258, 0.27577711,
            ]
        guard finalSize.count == 2, finalSize.allSatisfy({ $0 > 0 }), imageMean.count == 3,
            imageStd.count == 3, imageStd.allSatisfy({ $0 > 0 })
        else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: decoder.codingPath,
                    debugDescription: "Invalid Nemotron Parse image dimensions or normalization"))
        }
    }
}

struct NemotronParseProcessor: UserInputProcessor {
    let config: NemotronParseProcessorConfiguration
    let tokenizer: any Tokenizer
    init(_ config: NemotronParseProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }
    func preprocess(_ image: CIImage, processing: UserInput.Processing) -> MLXArray {
        let source = MediaProcessing.apply(image, processing: processing)
        let height = config.finalSize[0]
        let width = config.finalSize[1]
        let scale = min(
            1, Double(height) / source.extent.height, Double(width) / source.extent.width)
        let h = max(1, Int(source.extent.height * scale))
        let w = max(1, Int(source.extent.width * scale))
        let srgb = MediaProcessing.asMLXArray(MediaProcessing.inSRGBToneCurveSpace(source))
        let resized = resizeRGBBilinear(srgb, size: (h, w))
        let canvas = MLXArray.ones([1, 3, height, width])
        let y = (height - h) / 2
        let x = (width - w) / 2
        canvas[0..., 0..., y ..< y + h, x ..< x + w] = resized
        return (canvas - MLXArray(config.imageMean).reshaped(1, 3, 1, 1))
            / MLXArray(config.imageStd).reshaped(1, 3, 1, 1)
    }
    func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        guard images.count == 1, input.videos.isEmpty, input.audios.isEmpty else {
            throw VLMError.processing("Nemotron Parse requires exactly one document image")
        }
        let pixels = try preprocess(images[0].asCIImage(), processing: input.processing)
        return LMInput(
            text: .init(
                tokens: MLXArray(tokenizer.encode(text: input.prompt.description))[.newAxis]),
            image: .init(pixels: pixels))
    }
}
