import CoreGraphics
import Foundation
import ImageIO
import MLX
import NativSDK
import XCTest

final class ModelImageDataTests: XCTestCase {
    func testTensorSnapshotPreservesStridesDtypeAndPixelsAcrossTasks() async throws {
        let source = MLXArray([UInt8(1), 2, 3, 4, 5, 6], [1, 3, 2]).transposed(2, 0, 1)
        let image = try ModelImageData(tensor: source)
        source[0, 0, 0] = MLXArray(UInt8(99))
        let snapshot = await Task.detached { image.tensor }.value
        XCTAssertEqual(snapshot.dtype, .uint8)
        XCTAssertEqual(snapshot.shape, [2, 1, 3])
        XCTAssertEqual(snapshot.asArray(UInt8.self), [1, 3, 5, 2, 4, 6])
        XCTAssertTrue(try ModelImageData(tensor: snapshot).tensor === snapshot)
        XCTAssertEqual(image.pixels, [1, 3, 5, 2, 4, 6])
        let rgba = try XCTUnwrap(image.cgImage().dataProvider?.data) as Data
        XCTAssertEqual(Array(rgba), [1, 3, 5, 255, 2, 4, 6, 255])
        XCTAssertThrowsError(try ModelImageData(tensor: MLXArray(42)))
        XCTAssertThrowsError(try ModelImageData(tensor: MLXArray.zeros([0, 3])))
    }

    func testRGBConversionPreservesLayoutAndClampsAndRoundsPixels() throws {
        let pixels: [Float] = [
            -1, 0.49, 0.5, 255.9, 127.5, 42,
            255, 0, 0, 0, 255, 0,
            0, 0, 255, 255, 255, 255,
        ]
        let data = try ModelImageData(pixels: pixels, shape: [3, 2, 3])
        let image = try data.cgImage()
        XCTAssertEqual(image.width, 2)
        XCTAssertEqual(image.height, 3)
        XCTAssertEqual(image.colorSpace?.name, CGColorSpace.sRGB)
        XCTAssertEqual(image.alphaInfo, .last)
        let bytes = try XCTUnwrap(image.dataProvider?.data) as Data
        XCTAssertEqual(
            Array(bytes),
            [
                0, 0, 1, 255, 255, 128, 42, 255,
                255, 0, 0, 255, 0, 255, 0, 255,
                0, 0, 255, 255, 255, 255, 255, 255,
            ])
    }

    func testPNGCanBeDecodedWithTransparency() throws {
        let data = try ModelImageData(
            pixels: [255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0], shape: [1, 3, 4])
        let png = try data.pngData()
        XCTAssertEqual(Array(png.prefix(8)), [137, 80, 78, 71, 13, 10, 26, 10])
        let source = try XCTUnwrap(CGImageSourceCreateWithData(png as CFData, nil))
        let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil))
        XCTAssertEqual(image.width, 3)
        XCTAssertEqual(image.height, 1)
        let context = try XCTUnwrap(
            CGContext(
                data: nil, width: 3, height: 1, bitsPerComponent: 8, bytesPerRow: 12,
                space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue
                    | CGImageAlphaInfo.premultipliedLast.rawValue))
        context.draw(image, in: CGRect(x: 0, y: 0, width: 3, height: 1))
        let bytes = try XCTUnwrap(context.data).assumingMemoryBound(to: UInt8.self)
        XCTAssertEqual(
            Array(UnsafeBufferPointer(start: bytes, count: 12)),
            [
                255, 0, 0, 255, 0, 128, 0, 128, 0, 0, 0, 0,
            ])
    }

    func testReferenceAndUnsupportedChannelLayoutsAreRejected() throws {
        for shape in [[1, 3, 2, 2], [2, 2, 1], [2, 2], [2, 2, 5]] {
            let data = try ModelImageData(
                pixels: Array(repeating: 0, count: shape.reduce(1, *)), shape: shape)
            XCTAssertThrowsError(try data.pngData()) { error in
                XCTAssertEqual(error as? ModelImageEncodingError, .unsupportedShape(shape))
            }
        }
    }

    func testNonFiniteColorAndAlphaValuesAreRejected() throws {
        for value: Float in [.nan, .infinity, -.infinity] {
            for pixels: [Float] in [[value, 0, 0, 255], [0, 0, 0, value]] {
                let data = try ModelImageData(pixels: pixels, shape: [1, 1, 4])
                XCTAssertThrowsError(try data.pngData()) { error in
                    XCTAssertEqual(error as? ModelImageEncodingError, .nonFinitePixels)
                }
            }
        }
    }

    func testSaveCreatesDirectoriesAndReplacesExistingFile() throws {
        let directory = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appending(path: "nested/image.png")
        let red = try ModelImageData(pixels: [255, 0, 0], shape: [1, 1, 3])
        let green = try ModelImageData(pixels: [0, 255, 0], shape: [1, 1, 3])
        try red.savePNG(to: url)
        XCTAssertEqual(try Data(contentsOf: url), try red.pngData())
        try green.savePNG(to: url)
        XCTAssertEqual(try Data(contentsOf: url), try green.pngData())

        let invalid = try ModelImageData(pixels: [.nan, 0, 0], shape: [1, 1, 3])
        XCTAssertThrowsError(try invalid.savePNG(to: url))
        XCTAssertEqual(try Data(contentsOf: url), try green.pngData())
    }

    func testSaveRejectsNonFileURLsAndPropagatesFilesystemErrors() throws {
        let image = try ModelImageData(pixels: [0, 0, 0], shape: [1, 1, 3])
        XCTAssertThrowsError(try image.savePNG(to: URL(string: "https://example.com/image.png")!)) {
            XCTAssertEqual($0 as? ModelImageEncodingError, .nonFileURL)
        }
        let url = FileManager.default.temporaryDirectory.appending(path: UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: url) }
        try Data([42]).write(to: url)
        XCTAssertThrowsError(try image.savePNG(to: url.appending(path: "image.png")))
        XCTAssertEqual(try Data(contentsOf: url), Data([42]))
    }
}
