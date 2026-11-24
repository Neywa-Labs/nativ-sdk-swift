import CoreImage
import Foundation
import MLX

public struct Inkling2ProcessorConfiguration: Decodable, Sendable {
    struct Image: Decodable, Sendable {
        let mean: [Float]?
        let deviation: [Float]?
        let rescale: Float?
        let size: Size?

        struct Size: Decodable, Sendable {
            let height: Int
            let width: Int
        }

        enum CodingKeys: String, CodingKey {
            case mean = "image_mean"
            case deviation = "image_std"
            case rescale = "rescale_factor"
            case size
        }
    }

    let model: Inkling2Configuration
    let imageToken: String
    let mean: [Float]
    let deviation: [Float]
    let rescale: Float

    enum CodingKeys: String, CodingKey {
        case image = "image_processor"
        case imageToken = "image_token"
    }

    public init(from decoder: Decoder) throws {
        model = try Inkling2Configuration(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        imageToken = try c.decodeIfPresent(String.self, forKey: .imageToken) ?? "<|unused_200054|>"
        let image = try c.decodeIfPresent(Image.self, forKey: .image)
        mean = image?.mean ?? [0.5, 0.5, 0.5]
        deviation = image?.deviation ?? [0.5, 0.5, 0.5]
        rescale = image?.rescale ?? 1 / 255
        let size = image?.size
        let tile = model.vision?.tile
        guard mean.count == 3, mean.allSatisfy(\.isFinite), deviation.count == 3,
            deviation.allSatisfy({ $0 > 0 && $0.isFinite }), rescale > 0, rescale.isFinite,
            size == nil || tile == nil || (size?.height == tile && size?.width == tile)
        else {
            throw VLMError.processing("Invalid Inkling 2 image processor configuration.")
        }
    }
}

public struct Inkling2Processor: UserInputProcessor {
    private static let context = CIContext(options: [
        .outputPremultiplied: false,
        .cacheIntermediates: false,
    ])

    let config: Inkling2ProcessorConfiguration
    let tokenizer: any Tokenizer

    public init(_ config: Inkling2ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        guard input.videos.isEmpty, MediaProcessing.audios(from: input).isEmpty else {
            throw VLMError.processing("Inkling 2 accepts text and still images.")
        }
        let images = MediaProcessing.images(from: input)
        let messages = MediaFirstMessageGenerator(textOnlyContentAsString: true).generate(
            from: input)
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        let imageToken = tokenizer.convertTokenToId(config.imageToken) ?? config.model.imageToken
        guard imageToken == config.model.imageToken,
            tokens.count(where: { $0 == imageToken }) == images.count
        else {
            throw VLMError.processing("Inkling 2 requires one matching placeholder per image.")
        }
        guard !images.isEmpty else { return LMInput(tokens: MLXArray(tokens)) }
        guard let vision = config.model.vision else {
            throw VLMError.processing("Image input requires an Inkling 2 vision tower.")
        }

        var tiles: [MLXArray] = []
        var grids: [THW] = []
        for image in images {
            try Task.checkCancellation()
            let frame = MediaProcessing.apply(try image.asCIImage(), processing: input.processing)
            let (patches, grid) = try imageTiles(frame, tile: vision.tile)
            tiles.append(patches)
            grids.append(grid)
        }
        var index = 0
        var expanded: [Int] = []
        expanded.reserveCapacity(tokens.count - images.count + grids.reduce(0) { $0 + $1.product })
        for token in tokens {
            if token == imageToken {
                expanded.append(contentsOf: repeatElement(token, count: grids[index].product))
                index += 1
            } else {
                expanded.append(token)
            }
        }

        return LMInput(
            text: .init(tokens: MLXArray(expanded).expandedDimensions(axis: 0)),
            image: .init(pixels: concatenated(tiles, axis: 0), frames: grids))
    }

    func imageTiles(_ image: CIImage, tile: Int) throws -> (MLXArray, THW) {
        let extent = image.extent
        guard !extent.isEmpty, !extent.isInfinite,
            let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        else {
            throw VLMError.processing("Inkling 2 requires a finite, nonempty image.")
        }
        let width = Int(extent.width.rounded())
        let height = Int(extent.height.rounded())
        guard width > 0, height > 0 else {
            throw VLMError.processing("Inkling 2 requires a finite, nonempty image.")
        }
        var data = Data(count: height * width * 4)
        data.withUnsafeMutableBytes { data in
            Self.context.render(
                image, toBitmap: data.baseAddress!, rowBytes: width * 4, bounds: extent,
                format: .RGBA8, colorSpace: colorSpace)
        }
        let rgba = MLXArray(data, [height, width, 4], type: UInt8.self).asType(.int32)
        let alpha = rgba[.ellipsis, 3...]
        // Transparent pixels are composited over the reference's 8-pixel checkerboard.
        let y = floorDivide(MLXArray(0 ..< height), 8).reshaped(height, 1)
        let x = floorDivide(MLXArray(0 ..< width), 8).reshaped(1, width)
        let background = MLX.where((y + x) % 2 .== 0, 204, 153).expandedDimensions(axis: -1)
        let rgb = floorDivide(rgba[.ellipsis, ..<3] * alpha + background * (255 - alpha) + 127, 255)
            .asType(.float32)
        let rows = (height + tile - 1) / tile
        let cols = (width + tile - 1) / tile
        let padded = MLX.padded(
            rgb,
            widths: [.init((0, rows * tile - height)), .init((0, cols * tile - width)), .init(0)],
            value: MLXArray(Float(-1)))
        let normalized =
            (padded * config.rescale - MLXArray(config.mean))
            / MLXArray(config.deviation)
        let tiles = normalized.reshaped(rows, tile, cols, tile, 3)
            .transposed(0, 2, 1, 3, 4).reshaped(rows * cols, tile, tile, 3)
        return (tiles, THW(1, rows, cols))
    }
}
