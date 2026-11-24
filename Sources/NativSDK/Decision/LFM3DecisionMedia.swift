import AVFoundation
import CoreImage
import Foundation
import MLX

struct LFM3DecisionFeatures {
    let token: Int
    let values: MLXArray
}
struct LFM3DecisionMediaPart {
    let token: Int
    let values: MLXArray
    let grid: THW?
}
struct LFM3DecisionPreparedMedia {
    var markup = ""
    var parts: [LFM3DecisionMediaPart] = []
}
struct LFM3DecisionMediaProcessor {
    let config: LFM3DecisionConfiguration
    let tokenizer: any Tokenizer
    func span(token: Int, count: Int) throws -> String {
        guard let marker = tokenizer.convertIdToToken(token), count > 0 else {
            throw ModelExecutionError.invalidOptions("d1 media has no valid placeholder")
        }
        return "<|vision_start|>" + String(repeating: marker, count: count) + "<|vision_end|>"
    }
    func prepare(_ media: [DecisionMedia]) async throws -> LFM3DecisionPreparedMedia {
        var result = LFM3DecisionPreparedMedia()
        let processing = UserInput.Processing()
        for item in media {
            try Task.checkCancellation()
            switch item {
            case .image(let url):
                var image = try UserInput.Image.url(url).asCIImage()
                let pixels = image.extent.width * image.extent.height
                if pixels > Double(config.maxPixels) {
                    let scale = sqrt(Double(config.maxPixels) / pixels)
                    image = MediaProcessing.resampleBicubic(
                        image,
                        to: .init(
                            width: max(1, floor(image.extent.width * scale)),
                            height: max(1, floor(image.extent.height * scale))))
                }
                let (values, grid) = try LFM3Processor.preprocess(
                    [image], video: false,
                    processing: processing, vision: config.vision, settings: .init(video: false))
                result.parts.append(.init(token: config.imageToken, values: values, grid: grid))
                result.markup += try span(
                    token: config.imageToken,
                    count: grid.product / (config.vision.mergeSize * config.vision.mergeSize))
            case .audio(let url):
                guard let omni = config.omni else {
                    throw ModelExecutionError.unsupportedOperation(
                        "d1-2B audio input; use d1-Omni-2B")
                }
                var audio = processing.audio
                audio.sampleRate = 16000
                audio.channels = 1
                let samples = try await UserInput.Audio.url(url).asMLXArray(processing: audio)
                let mel = try LFM3AudioFeatures.logMel(samples.reshaped(-1))
                result.parts.append(
                    .init(
                        token: omni.audioToken, values: mel.expandedDimensions(axis: 0), grid: nil))
                result.markup += try span(
                    token: omni.audioToken, count: LFM3AudioFeatures.tokenCount(mel.dim(1)))
            case .video(let url):
                let settings = LFM3ProcessorConfiguration.Pixels(video: true)
                var frames: [CIImage] = []
                let sequence = try await MediaProcessing.asProcessedSequence(
                    .url(url), userProcessing: processing,
                    targetFPS: { duration in
                        Double(max(4, Int(duration.seconds * settings.fps)))
                            / max(duration.seconds, 0.001)
                    },
                    maxFrames: settings.maxFrames,
                    frameProcessing: { frame in
                        frames.append(try frame.image.asCIImage())
                        return frame
                    })
                let (values, grid) = try LFM3Processor.preprocess(
                    frames, video: true,
                    processing: processing, vision: config.vision, settings: settings)
                result.parts.append(.init(token: config.videoToken, values: values, grid: grid))
                let temporal = config.vision.temporalPatchSize
                var times = sequence.timestamps.map(CMTimeGetSeconds)
                guard !times.isEmpty else {
                    throw ModelExecutionError.invalidOptions("Empty d1 video")
                }
                while times.count % temporal != 0 { times.append(times.last!) }
                // The outer wrapper is emitted by the LFM3 chat template; each
                // temporal patch has its own timestamp and inner vision span.
                result.markup += "<|vision_start|>"
                for group in 0 ..< grid.t {
                    let seconds = (times[group * temporal] + times[(group + 1) * temporal - 1]) / 2
                    result.markup += String(
                        format: "<%.1f seconds>", locale: Locale(identifier: "en_US_POSIX"), seconds
                    )
                    result.markup += try span(
                        token: config.videoToken,
                        count: grid.h * grid.w / (config.vision.mergeSize * config.vision.mergeSize)
                    )
                }
                result.markup += "<|vision_end|>"
            }
        }
        return result
    }
}
