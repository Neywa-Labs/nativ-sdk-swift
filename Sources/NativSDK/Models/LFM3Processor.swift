import AVFoundation
import CoreImage
import Foundation
import MLX

public struct LFM3ProcessorConfiguration: Decodable, Sendable {
    let model: LFM3Configuration
    let image: Pixels
    let video: Pixels
    struct Pixels: Decodable, Sendable {
        let minPixels: Int
        let maxPixels: Int
        let mean: [CGFloat]
        let std: [CGFloat]
        let fps: Double
        let maxFrames: Int
        enum CodingKeys: String, CodingKey {
            case size
            case mean = "image_mean"
            case std = "image_std"
            case fps
            case maxFrames = "max_frames"
        }
        init(video: Bool) {
            minPixels = video ? 4096 : 65536
            maxPixels = video ? 25_165_824 : 2_097_152
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
            fps = 2
            maxFrames = 384
        }
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            let size = try c.decodeIfPresent([String: Int].self, forKey: .size) ?? [:]
            minPixels = size["shortest_edge"] ?? 65536
            maxPixels = size["longest_edge"] ?? 2_097_152
            mean = try c.decodeIfPresent([CGFloat].self, forKey: .mean) ?? [0.5, 0.5, 0.5]
            std = try c.decodeIfPresent([CGFloat].self, forKey: .std) ?? [0.5, 0.5, 0.5]
            fps = try c.decodeIfPresent(Double.self, forKey: .fps) ?? 2
            maxFrames = try c.decodeIfPresent(Int.self, forKey: .maxFrames) ?? 384
            guard minPixels > 0, maxPixels >= minPixels, mean.count == 3, std.count == 3,
                std.allSatisfy({ $0 > 0 }), fps > 0, maxFrames > 0
            else {
                throw ModelFactoryError.invalidConfiguration(
                    "Invalid LFM3 image/video preprocessing settings")
            }
        }
    }
    enum CodingKeys: String, CodingKey {
        case image = "image_processor"
        case video = "video_processor"
    }
    public init(from decoder: Decoder) throws {
        model = try LFM3Configuration(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        image = try c.decodeIfPresent(Pixels.self, forKey: .image) ?? Pixels(video: false)
        video = try c.decodeIfPresent(Pixels.self, forKey: .video) ?? Pixels(video: true)
    }
}

struct LFM3MessageGenerator: MessageGenerator {
    func generate(message: Chat.Message) -> Message {
        var content: [Message] = message.images.map { _ in ["type": "image"] }
        content += message.audios.map { _ in ["type": "audio"] }
        content += message.videos.map { _ in ["type": "video"] }
        content.append(["type": "text", "text": message.content])
        var result: Message = ["role": message.role.rawValue, "content": content]
        addToolMetadata(to: &result, for: message)
        return result
    }
    func generate(from input: UserInput) -> [Message] {
        switch input.prompt {
        case .text(let text):
            var content: [Message] = input.images.map { _ in ["type": "image"] }
            content += input.audios.map { _ in ["type": "audio"] }
            content += input.videos.map { _ in ["type": "video"] }
            content.append(["type": "text", "text": text])
            return [["role": "user", "content": content]]
        case .messages(let messages): return messages
        case .chat(let messages): return generate(messages: messages)
        }
    }
}

public struct LFM3Processor: UserInputProcessor {
    let config: LFM3ProcessorConfiguration
    let tokenizer: any Tokenizer
    public init(_ config: LFM3ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }
    static func expand(_ tokens: [Int], token: Int, replacements: [[Int]]) throws -> [Int] {
        guard tokens.filter({ $0 == token }).count == replacements.count else {
            throw ModelExecutionError.invalidOptions(
                "LFM3 media placeholders do not match supplied inputs")
        }
        var index = 0
        return tokens.flatMap { id -> [Int] in
            guard id == token else { return [id] }
            defer { index += 1 }
            return replacements[index]
        }
    }
    static func targetSize(
        height: Int, width: Int, factor: Int, frames: Int = 1,
        temporal: Int = 1, minPixels: Int, maxPixels: Int
    ) throws -> (Int, Int) {
        guard height > 0, width > 0, frames > 0,
            Double(max(height, width)) / Double(min(height, width)) <= 200
        else {
            throw ModelExecutionError.invalidOptions(
                "Invalid LFM3 media dimensions or aspect ratio")
        }
        var h = Double(height)
        var w = Double(width)
        if temporal > 1, min(h, w) < Double(factor) {
            let scale = Double(factor) / min(h, w)
            h = floor(h * scale)
            w = floor(w * scale)
        }
        let f = Double(factor)
        var targetH = Int((h / f).rounded(.toNearestOrEven)) * factor
        var targetW = Int((w / f).rounded(.toNearestOrEven)) * factor
        let roundedFrames =
            Int((Double(frames) / Double(temporal)).rounded(.toNearestOrEven)) * temporal
        let pixels = Double(roundedFrames * targetH * targetW)
        if pixels > Double(maxPixels) {
            let beta = sqrt(Double(frames) * h * w / Double(maxPixels))
            targetH = max(factor, Int(floor(h / beta / f)) * factor)
            targetW = max(factor, Int(floor(w / beta / f)) * factor)
        } else if pixels < Double(minPixels) {
            let beta = sqrt(Double(minPixels) / (Double(frames) * h * w))
            targetH = Int(ceil(h * beta / f)) * factor
            targetW = Int(ceil(w * beta / f)) * factor
        }
        return (targetH, targetW)
    }
    func preprocess(_ images: [CIImage], video: Bool, processing: UserInput.Processing) throws -> (
        MLXArray, THW
    ) {
        guard let vision = config.model.vision else {
            throw ModelExecutionError.invalidOptions("LFM3-Audio cannot read images/video")
        }
        return try Self.preprocess(
            images, video: video, processing: processing,
            vision: vision, settings: video ? config.video : config.image)
    }
    static func preprocess(
        _ images: [CIImage], video: Bool, processing: UserInput.Processing,
        vision v: LFM3VisionConfiguration,
        settings: LFM3ProcessorConfiguration.Pixels
    ) throws -> (MLXArray, THW) {
        guard let first = images.first else {
            throw ModelExecutionError.invalidOptions("Empty LFM3 image sequence")
        }
        let frames = images.map { MediaProcessing.apply($0, processing: processing) }
        let size = frames.first?.extent.size ?? first.extent.size
        let factor = v.patchSize * v.mergeSize
        let count = video ? max(v.temporalPatchSize, images.count) : 1
        let (height, width) = try Self.targetSize(
            height: Int(size.height), width: Int(size.width), factor: factor,
            frames: count, temporal: video ? v.temporalPatchSize : 1,
            minPixels: settings.minPixels, maxPixels: settings.maxPixels)
        let arrays = frames.map { image in
            MediaProcessing.asMLXArray(
                MediaProcessing.normalize(
                    MediaProcessing.resampleBicubic(
                        image, to: CGSize(width: width, height: height)),
                    mean: (settings.mean[0], settings.mean[1], settings.mean[2]),
                    std: (settings.std[0], settings.std[1], settings.std[2])))
        }
        return try QwenVL.patchify(
            images: arrays, mergeSize: v.mergeSize, patchSize: v.patchSize,
            temporalPatchSize: v.temporalPatchSize)
    }
    public func prepare(input: UserInput) async throws -> LMInput {
        let c = config.model
        if c.vision == nil, !input.images.isEmpty || !input.videos.isEmpty {
            throw ModelExecutionError.invalidOptions("Use LFM3-Omni for image and video inputs")
        }
        var tokens = try tokenizer.applyChatTemplate(
            messages: LFM3MessageGenerator().generate(from: input), tools: input.tools,
            additionalContext: input.additionalContext)
        // Older templates and raw message content may omit Omni's media span.
        if c.vision != nil, let start = tokenizer.convertTokenToId("<|vision_start|>"),
            let end = tokenizer.convertTokenToId("<|vision_end|>")
        {
            let original = tokens
            tokens = original.enumerated().flatMap { i, token -> [Int] in
                guard [c.imageToken, c.videoToken, c.audioToken].contains(token) else {
                    return [token]
                }
                let wrapped =
                    i > 0 && i + 1 < original.count && original[i - 1] == start
                    && original[i + 1] == end
                return wrapped ? [token] : [start, token, end]
            }
        }
        var audio: LMInput.ProcessedAudio?
        var features = [MLXArray]()
        var audioProcessing = input.processing.audio
        audioProcessing.sampleRate = 16000
        audioProcessing.channels = 1
        for clip in input.audios {
            try Task.checkCancellation()
            let samples = try await clip.asMLXArray(processing: audioProcessing)
            features.append(try LFM3AudioFeatures.logMel(samples.reshaped(-1)))
        }
        tokens = try Self.expand(
            tokens, token: c.audioToken,
            replacements: features.map {
                Array(repeating: c.audioToken, count: LFM3AudioFeatures.tokenCount($0.dim(1)))
            })
        if let length = features.map({ $0.dim(1) }).max() {
            let paddedFeatures = features.map {
                padded($0, widths: [0, IntOrPair((0, length - $0.dim(1)))])
            }
            let masks = features.map { MLXArray(0 ..< length) .< $0.dim(1) }
            audio = .init(features: stacked(paddedFeatures), mask: stacked(masks))
        }
        var image: LMInput.ProcessedImage?
        let images = try input.images.map {
            try preprocess([$0.asCIImage()], video: false, processing: input.processing)
        }
        tokens = try Self.expand(
            tokens, token: c.imageToken,
            replacements: images.map {
                Array(
                    repeating: c.imageToken,
                    count: $0.1.t * $0.1.h * $0.1.w / (c.downsample * c.downsample))
            })
        if !images.isEmpty {
            image = .init(pixels: concatenated(images.map(\.0)), frames: images.map(\.1))
        }
        var video: LMInput.ProcessedVideo?
        var videos = [(MLXArray, THW)]()
        var replacements = [[Int]]()
        for clip in input.videos {
            var originals = [CIImage]()
            let sequence = try await MediaProcessing.asProcessedSequence(
                clip, userProcessing: input.processing,
                targetFPS: { duration in
                    Double(max(4, Int(duration.seconds * config.video.fps)))
                        / max(duration.seconds, 0.001)
                }, maxFrames: config.video.maxFrames,
                frameProcessing: { frame in
                    originals.append(try frame.image.asCIImage())
                    return frame
                })
            let processed = try preprocess(originals, video: true, processing: input.processing)
            videos.append(processed)
            let temporal = c.vision!.temporalPatchSize
            var timestamps = sequence.timestamps.map(CMTimeGetSeconds)
            guard !timestamps.isEmpty else {
                throw ModelExecutionError.invalidOptions("Empty LFM3 video")
            }
            while timestamps.count % temporal != 0 { timestamps.append(timestamps.last!) }
            var replacement = [Int]()
            for group in 0 ..< processed.1.t {
                let seconds =
                    (timestamps[group * temporal] + timestamps[(group + 1) * temporal - 1]) / 2
                let marker = String(
                    format: "<%.1f seconds><|vision_start|>",
                    locale: Locale(identifier: "en_US_POSIX"), seconds)
                replacement += tokenizer.encode(text: marker, addSpecialTokens: false)
                replacement += Array(
                    repeating: c.videoToken,
                    count: processed.1.h * processed.1.w / (c.downsample * c.downsample))
                replacement += tokenizer.encode(text: "<|vision_end|>", addSpecialTokens: false)
            }
            replacements.append(replacement)
        }
        tokens = try Self.expand(tokens, token: c.videoToken, replacements: replacements)
        if !videos.isEmpty {
            video = .init(pixels: concatenated(videos.map(\.0)), frames: videos.map(\.1))
        }
        guard !tokens.contains(c.audioOutToken) else {
            throw ModelExecutionError.invalidOptions(
                "Audio output history requires codec codes; waveform history is not supported")
        }
        return LMInput(
            text: .init(tokens: MLXArray(tokens)), image: image, video: video, audio: audio)
    }
}
