import AVFoundation
import CoreImage
import CoreMedia
import Foundation
import MLX
import XCTest

@testable import NativSDK

final class MediaProcessingUtilitiesTests: XCTestCase {
    @MainActor
    func testProcessedFrameSnapshotsCrossTasksWithTiming() async throws {
        let rect = CGRect(x: 0, y: 0, width: 2, height: 2)
        let end = CMTime(value: 1, timescale: 1)
        let frames = [
            UserInput.VideoFrame(
                image: .ciImage(CIImage(color: .red).cropped(to: rect)), timeStamp: .zero),
            UserInput.VideoFrame(
                image: .ciImage(CIImage(color: .green).cropped(to: rect)), timeStamp: end),
        ]
        let sequence = try await MediaProcessing.asProcessedSequence(
            .frames(frames), processing: .init(targetFrames: 2))
        let expected = sequence.frames.map { $0.asArray(Float.self) }
        let snapshot = sequence.materialized()
        sequence.frames[0][0, 0, 0, 0] = MLXArray(Float(-1))
        let values = await Task.detached { snapshot.frames.map { $0.asArray(Float.self) } }.value
        XCTAssertEqual(values, expected)
        XCTAssertNotEqual(values.first, values.last)
        XCTAssertEqual(snapshot.timestamps, [.zero, end])
        XCTAssertEqual(snapshot.totalDuration, end)
    }

    @MainActor
    func testAudioInputReusesMaterializedSamples() async throws {
        let samples = MLXArray([Float(0.5), -0.5]).materialized()
        let decoded = try await UserInput.Audio.array(samples).asMaterializedArray()
        XCTAssertTrue(samples === decoded)
        let tagged = try await UserInput.Audio.array(samples, sampleRate: 48_000)
            .asMaterializedArray()
        XCTAssertTrue(samples === tagged)
        let column = try await UserInput.Audio.array(samples.reshaped(2, 1), sampleRate: 48_000)
            .asMLXArray()
        XCTAssertEqual(column.shape, [2])
        XCTAssertEqual(column.asArray(Float.self), samples.asArray(Float.self))
    }

    @MainActor
    func testFileAndInMemoryWaveformsKeepTheirSourceRate() async throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let format = try XCTUnwrap(AVAudioFormat(standardFormatWithSampleRate: 24_000, channels: 2))
        let frames = 2_400
        let pcm = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: UInt32(frames)))
        pcm.frameLength = pcm.frameCapacity
        let channels = try XCTUnwrap(pcm.floatChannelData)
        let signal = (0 ..< frames).map { Float(sin(2 * Double.pi * 440 * Double($0) / 24_000)) }
        for (index, sample) in signal.enumerated() {
            channels[0][index] = sample
            channels[1][index] = sample * 0.5
        }
        try AVAudioFile(forWriting: url, settings: format.settings).write(from: pcm)

        let waveform = try await Task.detached { try UserInput.AudioWaveform(contentsOf: url) }
            .value
        XCTAssertEqual(waveform.sampleRate, 24_000)
        let mono = signal.map { $0 * 0.75 }
        assertClose(waveform.samples.asArray(Float.self), mono, accuracy: 1e-6)

        var processing = UserInput.AudioProcessing()
        processing.sampleRate = 16_000
        let decoded = try await UserInput.Audio.waveform(waveform)
            .asMaterializedArray(processing: processing)
        let array = MLXArray(mono, [1, 1, frames])
        let input = UserInput.Audio.array(array, sampleRate: 24_000)
        array[0, 0, 0] = MLXArray(Float(100))
        let resampled = try await input.asMLXArray(processing: processing)
        XCTAssertEqual(decoded.shape, [1_600])
        assertClose(resampled.asArray(Float.self), decoded.asArray(Float.self), accuracy: 1e-6)
        XCTAssertEqual(
            decoded[100].item(Float.self), 0.75 * sin(2 * .pi * 440 * 100 / 16_000),
            accuracy: 0.01)

        processing.sampleRate = 24_000
        processing.channels = 2
        let stereo = try await UserInput.Audio.url(url).asMLXArray(processing: processing)
        assertClose(stereo.asArray(Float.self), signal.flatMap { [$0, $0 * 0.5] }, accuracy: 1e-6)
    }

    @MainActor
    func testWaveformRejectsInvalidRatesAndAmbiguousChannels() async throws {
        let samples = MLXArray([Float(0), 1])
        for (input, targetRate) in [
            (UserInput.Audio.array(samples, sampleRate: 0), 16_000.0),
            (.array(samples, sampleRate: 24_000), .nan),
            (.array(samples, sampleRate: 24_000), 0),
            (.array(MLXArray.zeros([2, 8]), sampleRate: 24_000), 16_000),
        ] {
            var processing = UserInput.AudioProcessing()
            processing.sampleRate = targetRate
            do {
                _ = try await input.asMLXArray(processing: processing)
                XCTFail("Expected invalid waveform metadata to throw")
            } catch VLMError.processing {}
        }
        let empty = try await UserInput.Audio.array(MLXArray([Float]()), sampleRate: 24_000)
            .asMLXArray()
        XCTAssertEqual(empty.shape, [0])
    }

    func testAudioSelectionPreservesExplicitPrecedenceAndChatOrder() throws {
        let first = URL(fileURLWithPath: "/first.wav")
        let second = URL(fileURLWithPath: "/second.wav")
        let explicit = URL(fileURLWithPath: "/explicit.wav")
        var input = UserInput(chat: [
            .user("first", audios: [.url(first)]),
            .user("second", audios: [.url(second)]),
        ])
        func audioURLs() throws -> [URL] {
            try MediaProcessing.audios(from: input).map {
                guard case .url(let url) = $0 else { throw CocoaError(.fileReadUnknown) }
                return url
            }
        }
        input.audios = []
        XCTAssertEqual(try audioURLs(), [first, second])
        input.audios = [.url(explicit)]
        XCTAssertEqual(try audioURLs(), [explicit])
        input.audios = []
        input.prompt = .text("no audio")
        XCTAssertTrue(MediaProcessing.audios(from: input).isEmpty)
        input.prompt = .messages([["role": "user", "content": "no audio"]])
        XCTAssertTrue(MediaProcessing.audios(from: input).isEmpty)
    }

    func testPixelCropsPreserveRowsAndChannels() {
        let rgb = (0 ..< 36).map(Float.init)
        XCTAssertEqual(
            MediaProcessing.cropPixels(
                rgb, sourceWidth: 4, x: 1, y: 1, width: 2, height: 2, channels: 3),
            [15, 16, 17, 18, 19, 20, 27, 28, 29, 30, 31, 32])
        let mask = (0 ..< 12).map(Float.init)
        XCTAssertEqual(
            MediaProcessing.cropPixels(
                mask, sourceWidth: 4, x: 1, y: 1, width: 2, height: 2, channels: 1),
            [5, 6, 9, 10])
    }

    func testPixelPatchesPreserveOrderAndDiscardIncompleteEdges() {
        let pixels = (0 ..< 30).map(Float.init)
        XCTAssertEqual(
            MediaProcessing.patchifyPixels(
                pixels, width: 5, height: 3, channels: 2, patchSize: 2),
            [0, 1, 2, 3, 10, 11, 12, 13, 4, 5, 6, 7, 14, 15, 16, 17])
        XCTAssertEqual(
            MediaProcessing.patchifyPixels(
                pixels, width: 5, height: 3, channels: 2, patchSize: 1), pixels)
    }

    func testPatchMaskAveragesRetainPartialCoverage() {
        let mask: [Float] = [
            1, 1, 0, 0, 1,
            1, 0, 1, 0, 1,
            1, 1, 1, 1, 1,
        ]
        XCTAssertEqual(
            MediaProcessing.patchifyMask(mask, width: 5, height: 3, patchSize: 2),
            [0.75, 0.25])
    }

    func testImageSelectionPreservesExplicitPrecedenceAndChatOrder() throws {
        let first = URL(fileURLWithPath: "/first.png")
        let second = URL(fileURLWithPath: "/second.png")
        let explicit = URL(fileURLWithPath: "/explicit.png")
        var input = UserInput(chat: [
            .user("first", images: [.url(first)]),
            .user("second", images: [.url(second)]),
        ])
        input.images = []
        let fromChat = MediaProcessing.images(from: input)
        XCTAssertEqual(try urls(fromChat), [first, second])
        input.images = [.url(explicit)]
        XCTAssertEqual(try urls(MediaProcessing.images(from: input)), [explicit])
        input.prompt = .text("no images")
        input.images = []
        XCTAssertTrue(MediaProcessing.images(from: input).isEmpty)
        input.prompt = .messages([["role": "user", "content": "no images"]])
        XCTAssertTrue(MediaProcessing.images(from: input).isEmpty)
    }

    func testZeroOriginRetainsIntegralCropBounds() {
        let input = CIImage(color: .red).cropped(to: CGRect(x: 1.25, y: -2.5, width: 2, height: 3))
        let result = MediaProcessing.zeroOrigin(input)
        XCTAssertEqual(result.extent, CGRect(x: 0, y: 0, width: 3, height: 4))
        XCTAssertEqual(MediaProcessing.zeroOrigin(result).extent, result.extent)
    }

    func testNormalizeCHWPreservesSpatialAndChannelOrder() {
        let pixels = (1 ... 12).map(Float.init)
        let result = MediaProcessing.normalizeCHW(
            pixels, width: 2, height: 2, mean: [1, 2, 3], std: [1, 2, 3])
        XCTAssertEqual(result, [0, 3, 6, 9, 0, 1.5, 3, 4.5, 0, 1, 2, 3])
    }

    func testRGBRenderersPreserveColorAlphaAndPixelOrder() {
        let bytes: [UInt8] = [
            64, 32, 0, 128, 0, 0, 0, 0,
            255, 128, 64, 255, 0, 255, 0, 255,
        ]
        let input = CIImage(
            bitmapData: Data(bytes), bytesPerRow: 8, size: CGSize(width: 2, height: 2),
            format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        let expected: [Float] = [0.5, 0.25, 0, 0, 0, 0, 1, 128.0 / 255, 64.0 / 255, 0, 1, 0]
        assertClose(
            MediaProcessing.renderRGB8(input, width: 2, height: 2), expected, accuracy: 1e-6)
        assertClose(
            MediaProcessing.renderRGBFloat(input, width: 2, height: 2), expected, accuracy: 1e-4)
        let tensor = MediaProcessing.renderRGBArray(input)
        XCTAssertEqual(tensor.shape, [2, 2, 3])
        assertClose(tensor.asArray(Float.self), expected, accuracy: 1e-4)

        let translated = input.transformed(by: CGAffineTransform(translationX: 7, y: -3))
        assertClose(
            MediaProcessing.renderRGBArray(translated).asArray(Float.self), expected, accuracy: 1e-4
        )
    }

    func testFloatRendererRetainsSubBytePrecision() {
        let rgba: [Float] = [0.123456, 0.234567, 0.345678, 1]
        let data = rgba.withUnsafeBytes { Data($0) }
        let input = CIImage(
            bitmapData: data, bytesPerRow: 16, size: CGSize(width: 1, height: 1),
            format: .RGBAf, colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        let expected = Array(rgba.prefix(3))
        assertClose(
            MediaProcessing.renderRGBFloat(input, width: 1, height: 1), expected, accuracy: 1e-4)
        assertClose(
            MediaProcessing.renderRGBArray(input).asArray(Float.self), expected, accuracy: 1e-4)
        let bytes = MediaProcessing.renderRGB8(input, width: 1, height: 1)
        XCTAssertGreaterThan(abs(bytes[0] - expected[0]), 1e-4)
    }

    func testSmartResizeRoundingAndPixelBudgets() throws {
        let rounded = try QwenVL.targetSize(
            height: 57, width: 85, factor: 28, minPixels: 1, maxPixels: 100000)
        XCTAssertEqual([rounded.0, rounded.1], [56, 84])
        let reduced = try QwenVL.targetSize(
            height: 57, width: 85, factor: 28, minPixels: 1, maxPixels: 3136)
        XCTAssertEqual([reduced.0, reduced.1], [28, 56])
        let enlarged = try QwenVL.targetSize(
            height: 57, width: 85, factor: 28, minPixels: 12544, maxPixels: 100000)
        XCTAssertEqual([enlarged.0, enlarged.1], [112, 140])
        for (height, width, maxPixels) in [
            (27, 28, 100000), (28, 27, 100000), (28, 5628, 100000), (28, 28, 1),
        ] {
            XCTAssertThrowsError(
                try QwenVL.targetSize(
                    height: height, width: width, factor: 28, minPixels: 1, maxPixels: maxPixels)
            ) {
                guard case VLMError.imageProcessingFailure = $0 else {
                    return XCTFail("Unexpected error: \($0)")
                }
            }
        }
    }

    func testPatchificationPreservesMergeOrderAndRepeatsLastVideoFrame() throws {
        let frames = (0 ..< 3).map { frame in
            MLXArray((frame * 16) ..< ((frame + 1) * 16), [1, 1, 4, 4])
        }
        let (patches, grid) = try QwenVL.patchify(
            images: frames, mergeSize: 2, patchSize: 1, temporalPatchSize: 2)
        XCTAssertEqual(patches.shape, [32, 2])
        XCTAssertEqual([grid.t, grid.h, grid.w], [2, 4, 4])
        let spatialOrder = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
        let expected =
            spatialOrder.flatMap { [$0, $0 + 16] } + spatialOrder.flatMap { [$0 + 32, $0 + 32] }
        XCTAssertEqual(patches.asArray(Int.self), expected)
    }

    func testPatchificationPreservesChannelsAndPixelsWithinPatches() throws {
        let (patches, grid) = try QwenVL.patchify(
            images: [MLXArray(0 ..< 16, [1, 2, 2, 4])], mergeSize: 1, patchSize: 2,
            temporalPatchSize: 1)
        XCTAssertEqual(patches.shape, [2, 8])
        XCTAssertEqual([grid.t, grid.h, grid.w], [1, 1, 2])
        XCTAssertEqual(
            patches.asArray(Int.self), [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15])
        XCTAssertThrowsError(
            try QwenVL.patchify(images: [], mergeSize: 2, patchSize: 2, temporalPatchSize: 2)
        ) {
            XCTAssertEqual($0 as? VLMError, .imageProcessingFailure("No images in video sequence"))
        }
    }

    private func urls(_ images: [UserInput.Image]) throws -> [URL] {
        try images.map {
            guard case .url(let url) = $0 else { throw CocoaError(.fileReadUnknown) }
            return url
        }
    }

    private func assertClose(
        _ actual: [Float], _ expected: [Float], accuracy: Float, file: StaticString = #filePath,
        line: UInt = #line
    ) {
        XCTAssertEqual(actual.count, expected.count, file: file, line: line)
        for (value, reference) in zip(actual, expected) {
            XCTAssertEqual(value, reference, accuracy: accuracy, file: file, line: line)
        }
    }
}
