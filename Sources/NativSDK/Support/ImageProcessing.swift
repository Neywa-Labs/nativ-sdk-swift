import CoreImage
import Foundation
import MLX

extension MediaProcessing {
    /// Select explicit images first, otherwise collect chat images in message order.
    static func images(from input: UserInput) -> [UserInput.Image] {
        if !input.images.isEmpty {
            return input.images
        }
        guard case .chat(let messages) = input.prompt else {
            return []
        }
        return messages.flatMap(\.images)
    }

    /// Crop to integral bounds before translating the image to the origin.
    static func zeroOrigin(_ image: CIImage) -> CIImage {
        let rect = image.extent.integral
        return
            image
            .cropped(to: rect)
            .transformed(by: CGAffineTransform(translationX: -rect.minX, y: -rect.minY))
    }

    /// Normalize interleaved RGB pixels and convert them to planar CHW storage.
    static func normalizeCHW(
        _ pixels: [Float],
        width: Int,
        height: Int,
        mean: [CGFloat],
        std: [CGFloat],
    )
        -> [Float]
    {
        var output = Array(repeating: Float(0), count: 3 * width * height)
        for row in 0 ..< height {
            for col in 0 ..< width {
                let source = (row * width + col) * 3
                let spatial = row * width + col
                output[spatial] = (pixels[source] - Float(mean[0])) / Float(std[0])
                output[width * height + spatial] =
                    (pixels[source + 1] - Float(mean[1])) / Float(std[1])
                output[2 * width * height + spatial] =
                    (pixels[source + 2] - Float(mean[2])) / Float(std[2])
            }
        }
        return output
    }

    /// Render sRGB through RGBA8 and return unpremultiplied RGB values in 0...1.
    static func renderRGB8(_ image: CIImage, width: Int, height: Int) -> [Float] {
        let componentsPerPixel = 4
        let bytesPerPixel = componentsPerPixel * MemoryLayout<UInt8>.size
        let bytesPerRow = width * bytesPerPixel
        var data = Data(count: width * height * bytesPerPixel)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        data.withUnsafeMutableBytes { pointer in
            CIContext().render(
                image,
                toBitmap: pointer.baseAddress!,
                rowBytes: bytesPerRow,
                bounds: CGRect(x: 0, y: 0, width: width, height: height),
                format: .RGBA8,
                colorSpace: colorSpace,
            )
        }
        let rgba = data.withUnsafeBytes { Array($0.bindMemory(to: UInt8.self)) }
        var rgb = Array(repeating: Float(0), count: width * height * 3)
        for i in 0 ..< (width * height) {
            let source = i * 4
            let target = i * 3
            let alpha = max(Float(rgba[source + 3]) / 255, Float(1e-6))
            rgb[target] = min(max((Float(rgba[source]) / 255) / alpha, 0), 1)
            rgb[target + 1] = min(max((Float(rgba[source + 1]) / 255) / alpha, 0), 1)
            rgb[target + 2] = min(max((Float(rgba[source + 2]) / 255) / alpha, 0), 1)
        }
        return rgb
    }

    /// Render sRGB through RGBAf and return unpremultiplied RGB values in 0...1.
    static func renderRGBFloat(_ image: CIImage, width: Int, height: Int) -> [Float] {
        let componentsPerPixel = 4
        let bytesPerPixel = componentsPerPixel * MemoryLayout<Float32>.size
        let bytesPerRow = width * bytesPerPixel
        var data = Data(count: width * height * bytesPerPixel)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        data.withUnsafeMutableBytes { pointer in
            CIContext().render(
                image,
                toBitmap: pointer.baseAddress!,
                rowBytes: bytesPerRow,
                bounds: CGRect(x: 0, y: 0, width: width, height: height),
                format: .RGBAf,
                colorSpace: colorSpace,
            )
        }

        let rgba = data.withUnsafeBytes { Array($0.bindMemory(to: Float32.self)) }
        var rgb = Array(repeating: Float(0), count: width * height * 3)
        for i in 0 ..< (width * height) {
            let source = i * 4
            let target = i * 3
            let alpha = max(rgba[source + 3], Float32(1e-6))
            rgb[target] = min(max(rgba[source] / alpha, 0), 1)
            rgb[target + 1] = min(max(rgba[source + 1] / alpha, 0), 1)
            rgb[target + 2] = min(max(rgba[source + 2] / alpha, 0), 1)
        }
        return rgb
    }

    /// Render the image extent through RGBAf and return unpremultiplied HWC RGB.
    static func renderRGBArray(_ image: CIImage) -> MLXArray {
        let width = Int(image.extent.width.rounded())
        let height = Int(image.extent.height.rounded())
        let format = CIFormat.RGBAf
        let componentsPerPixel = 4
        let bytesPerPixel = componentsPerPixel * MemoryLayout<Float32>.stride
        let bytesPerRow = width * bytesPerPixel
        var data = Data(count: width * height * bytesPerPixel)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        let context = CIContext(options: [.cacheIntermediates: false])
        data.withUnsafeMutableBytes { ptr in
            context.render(
                image,
                toBitmap: ptr.baseAddress!,
                rowBytes: bytesPerRow,
                bounds: image.extent,
                format: format,
                colorSpace: colorSpace,
            )
        }
        let rgba = MLXArray(data, [height, width, 4], type: Float32.self)
        let rgb = rgba[0..., 0..., ..<3]
        let alpha = maximum(rgba[0..., 0..., 3...], MLXArray(Float32(1e-6)))
        return clip(rgb / alpha, min: Float32(0), max: Float32(1))
    }

    /// Crop an interleaved pixel buffer without changing channel or row order.
    static func cropPixels(
        _ pixels: [Float], sourceWidth: Int, x: Int, y: Int,
        width: Int, height: Int, channels: Int
    ) -> [Float] {
        var output = Array(repeating: Float(0), count: width * height * channels)
        for row in 0 ..< height {
            let sourceStart = ((y + row) * sourceWidth + x) * channels
            let targetStart = row * width * channels
            output[targetStart ..< targetStart + width * channels] =
                pixels[sourceStart ..< sourceStart + width * channels]
        }
        return output
    }

    /// Flatten complete patches in row-major order, keeping channels interleaved.
    static func patchifyPixels(
        _ pixels: [Float],
        width: Int,
        height: Int,
        channels: Int,
        patchSize: Int,
    )
        -> [Float]
    {
        let patchesH = height / patchSize
        let patchesW = width / patchSize
        var output = [Float]()
        output.reserveCapacity(patchesH * patchesW * patchSize * patchSize * channels)
        for patchRow in 0 ..< patchesH {
            for patchCol in 0 ..< patchesW {
                for dy in 0 ..< patchSize {
                    for dx in 0 ..< patchSize {
                        let pixel =
                            ((patchRow * patchSize + dy) * width + patchCol * patchSize + dx)
                            * channels
                        for channel in 0 ..< channels {
                            output.append(pixels[pixel + channel])
                        }
                    }
                }
            }
        }
        return output
    }

    /// Average each complete patch to obtain its valid-pixel fraction.
    static func patchifyMask(
        _ mask: [Float],
        width: Int,
        height: Int,
        patchSize: Int,
    )
        -> [Float]
    {
        let patchesH = height / patchSize
        let patchesW = width / patchSize
        var output = [Float]()
        output.reserveCapacity(patchesH * patchesW)
        let denominator = Float(patchSize * patchSize)
        for patchRow in 0 ..< patchesH {
            for patchCol in 0 ..< patchesW {
                var sum = Float(0)
                for dy in 0 ..< patchSize {
                    for dx in 0 ..< patchSize {
                        sum += mask[(patchRow * patchSize + dy) * width + patchCol * patchSize + dx]
                    }
                }
                output.append(sum / denominator)
            }
        }
        return output
    }
}
