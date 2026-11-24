import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Failures when converting generated pixels into a displayable or encoded image.
public enum ModelImageEncodingError: LocalizedError, Equatable {
    case unsupportedShape([Int])
    case nonFinitePixels
    case imageCreationFailed
    case pngEncodingFailed
    case nonFileURL

    public var errorDescription: String? {
        switch self {
        case .unsupportedShape(let shape):
            "Expected generated RGB or RGBA pixels with shape [height, width, 3 or 4], got \(shape)."
        case .nonFinitePixels:
            "The image contains non-finite pixel values. Replace NaN and infinity before exporting."
        case .imageCreationFailed:
            "Couldn’t create an image from the generated pixels."
        case .pngEncodingFailed:
            "Couldn’t encode the generated image as PNG."
        case .nonFileURL:
            "Saving an image requires a local file URL."
        }
    }
}

extension ModelImageData {
    /// Creates an 8-bit sRGB image for SwiftUI, UIKit, AppKit, or Core Graphics.
    ///
    /// Expects generated HWC pixels: `[height, width, 3]` for RGB or
    /// `[height, width, 4]` for RGBA, with values in `0...255`. Finite values
    /// are clamped and rounded to the nearest byte; NaN and infinity throw.
    /// RGB becomes opaque RGBA; RGBA preserves straight (unpremultiplied) alpha.
    /// Normalized NCHW reference images must be converted before calling this method.
    public func cgImage() throws -> CGImage {
        guard shape.count == 3, shape[2] == 3 || shape[2] == 4 else {
            throw ModelImageEncodingError.unsupportedShape(shape)
        }
        let height = shape[0]
        let width = shape[1]
        let channels = shape[2]
        let pixels = self.pixels
        let (byteCount, overflow) = (pixels.count / channels).multipliedReportingOverflow(by: 4)
        guard !overflow else { throw ModelImageEncodingError.imageCreationFailed }
        var rgba = [UInt8]()
        rgba.reserveCapacity(byteCount)
        for offset in stride(from: 0, to: pixels.count, by: channels) {
            for channel in 0 ..< channels {
                let value = pixels[offset + channel]
                guard value.isFinite else { throw ModelImageEncodingError.nonFinitePixels }
                rgba.append(UInt8(min(255, max(0, value)).rounded()))
            }
            if channels == 3 { rgba.append(255) }
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
            let colorSpace = CGColorSpace(name: CGColorSpace.sRGB),
            let image = CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                bytesPerRow: width * 4, space: colorSpace,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
        else { throw ModelImageEncodingError.imageCreationFailed }
        return image
    }

    /// Encodes generated HWC pixels as PNG data, preserving RGBA transparency.
    /// Uses the same pixel conversion and validation as ``cgImage()``.
    public func pngData() throws -> Data {
        let image = try cgImage()
        let data = NSMutableData()
        guard
            let destination = CGImageDestinationCreateWithData(
                data, UTType.png.identifier as CFString, 1, nil)
        else { throw ModelImageEncodingError.pngEncodingFailed }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else {
            throw ModelImageEncodingError.pngEncodingFailed
        }
        return data as Data
    }

    /// Saves generated pixels as a PNG at a local file URL.
    ///
    /// Creates missing parent directories and atomically replaces an existing file.
    /// The encoding is always PNG, regardless of the filename extension. File system
    /// errors are propagated to the caller. Generation metadata is not embedded.
    public func savePNG(to url: URL) throws {
        guard url.isFileURL else { throw ModelImageEncodingError.nonFileURL }
        let data = try pngData()
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: url, options: .atomic)
    }
}
