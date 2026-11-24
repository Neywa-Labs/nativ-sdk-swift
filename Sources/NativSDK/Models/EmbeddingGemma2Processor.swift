import AVFoundation
import CoreImage
import Foundation
import MLX

struct EmbeddingGemma2Processor: UserInputProcessor {
    let tokenizer: any Tokenizer
    let model: EmbeddingGemma2Configuration
    let maximumLength: Int
    let imageConfiguration: Gemma4ProcessorConfiguration
    let videoConfiguration: Gemma4ProcessorConfiguration
    let audioConfiguration: Gemma4AudioFeatureExtractorConfiguration
    let videoFPS: Double?
    let maximumFrames: Int
    let timestamps: Bool
    let overflowStrategy: String

    init(
        tokenizer: any Tokenizer, model: EmbeddingGemma2Configuration,
        settings: [String: JSONValue], maximumLength: Int
    ) throws {
        self.tokenizer = tokenizer
        self.model = model
        self.maximumLength = maximumLength
        func imageConfig(_ value: JSONValue?, fallbackBudget: Int) throws
            -> Gemma4ProcessorConfiguration
        {
            var object: [String: JSONValue]
            if case .object(let nested) = value { object = nested } else { object = settings }
            object["processor_class"] = .string("EmbeddingGemma2Processor")
            object["patch_size"] = object["patch_size"] ?? .int(model.vision?.patchSize ?? 16)
            object["pooling_kernel_size"] =
                object["pooling_kernel_size"] ?? .int(model.vision?.poolingKernelSize ?? 3)
            object["max_soft_tokens"] = object["max_soft_tokens"] ?? .int(fallbackBudget)
            let config = try JSONDecoder().decode(
                Gemma4ProcessorConfiguration.self, from: JSONEncoder().encode(object))
            guard config.patchSize > 0, config.poolingKernelSize > 0, config.maxSoftTokens > 0,
                config.imageMean.count == 3, config.imageStd.count == 3,
                config.imageStd.allSatisfy({ $0 > 0 }),
                model.vision == nil
                    || (config.patchSize == model.vision!.patchSize
                        && config.poolingKernelSize == model.vision!.poolingKernelSize),
                object["do_resize"] != .bool(false), object["do_rescale"] != .bool(false)
            else {
                throw ModelExecutionError.invalidOptions(
                    "Invalid EmbeddingGemma 2 image processor configuration")
            }
            return config
        }
        imageConfiguration = try imageConfig(
            settings["image_processor"], fallbackBudget: model.visionSoftTokens)
        videoConfiguration = try imageConfig(settings["video_processor"], fallbackBudget: 140)
        var audio: [String: JSONValue]
        if case .object(let nested) = settings["feature_extractor"] {
            audio = nested
        } else {
            audio = settings
        }
        let sampleRate: Int
        if case .int(let rate) = audio["sampling_rate"] {
            sampleRate = rate
        } else {
            sampleRate = 16000
        }
        for (samples, milliseconds) in [
            ("frame_length", "frame_length_ms"), ("hop_length", "hop_length_ms"),
        ] {
            if audio[milliseconds] == nil, case .int(let count) = audio[samples] {
                audio[milliseconds] = .double(Double(count) * 1000 / Double(max(sampleRate, 1)))
            }
        }
        audioConfiguration = try JSONDecoder().decode(
            Gemma4AudioFeatureExtractorConfiguration.self, from: JSONEncoder().encode(audio))
        guard sampleRate > 0, audioConfiguration.featureSize > 0,
            audioConfiguration.frameLengthMs > 0, audioConfiguration.hopLengthMs > 0,
            model.audio == nil || audioConfiguration.featureSize == model.audio!.inputFeatureSize
        else {
            throw ModelExecutionError.invalidOptions(
                "Invalid EmbeddingGemma 2 audio processor configuration")
        }
        let video: [String: JSONValue]
        if case .object(let nested) = settings["video_processor"] {
            video = nested
        } else {
            video = [:]
        }
        switch video["fps"] {
        case .double(let value): videoFPS = value
        case .int(let value): videoFPS = Double(value)
        case .null: videoFPS = nil
        default: videoFPS = 1
        }
        if case .int(let count) = video["max_frames"] {
            maximumFrames = count
        } else {
            maximumFrames = 32
        }
        timestamps = video["add_timestamps"] == .bool(true)
        if case .string(let strategy) = video["overflow_strategy"] {
            overflowStrategy = strategy
        } else {
            overflowStrategy = "uniform"
        }
        guard videoFPS == nil || (videoFPS! > 0 && videoFPS!.isFinite), maximumFrames > 0,
            ["uniform", "truncate"].contains(overflowStrategy)
        else {
            throw ModelExecutionError.invalidOptions(
                "Unsupported EmbeddingGemma 2 video sampling configuration")
        }
    }

    /// Match the reference's floor-based sampling before applying the frame budget.
    static func sampleIndices(
        totalFrames: Int, sourceFPS: Double?, duration: Double?,
        targetFPS: Double?, maximumFrames: Int, strategy: String
    ) throws -> [Int] {
        guard totalFrames > 0, maximumFrames > 0, ["uniform", "truncate"].contains(strategy),
            targetFPS == nil || (targetFPS! > 0 && targetFPS!.isFinite)
        else {
            throw ModelExecutionError.invalidOptions("Invalid embedding video sampling options")
        }
        let indices: [Int]
        if let targetFPS, let sourceFPS, let duration,
            sourceFPS > 0, sourceFPS.isFinite, duration > 0, duration.isFinite
        {
            let count = max(1, Int(duration * targetFPS))
            indices = (0 ..< count).map {
                min(totalFrames - 1, Int(Double($0) * sourceFPS / targetFPS))
            }
        } else {
            indices = Array(0 ..< totalFrames)
        }
        guard indices.count > maximumFrames else { return indices }
        if strategy == "truncate" { return Array(indices.prefix(maximumFrames)) }
        if maximumFrames == 1 { return [indices[0]] }
        return (0 ..< maximumFrames).map {
            indices[Int(Double($0) * Double(indices.count - 1) / Double(maximumFrames - 1))]
        }
    }

    private func videoFrames(_ video: UserInput.Video, processing: UserInput.VideoProcessing)
        async throws -> [UserInput.VideoFrame]
    {
        func indices(count: Int, fps: Double?, duration: Double?) throws -> [Int] {
            if let requested = processing.targetFrames {
                return try Self.sampleIndices(
                    totalFrames: count, sourceFPS: nil, duration: nil,
                    targetFPS: nil, maximumFrames: min(requested, maximumFrames),
                    strategy: "uniform")
            }
            return try Self.sampleIndices(
                totalFrames: count, sourceFPS: fps, duration: duration,
                targetFPS: processing.targetFramesPerSecond ?? videoFPS,
                maximumFrames: maximumFrames, strategy: overflowStrategy)
        }
        if case .frames(let frames) = video {
            guard !frames.isEmpty else {
                throw ModelExecutionError.invalidOptions("Embedding video has no frames")
            }
            let interval =
                frames.count > 1
                ? (frames.last!.timeStamp.seconds - frames.first!.timeStamp.seconds)
                    / Double(frames.count - 1) : 0
            let fps = interval > 0 ? 1 / interval : nil
            return try indices(
                count: frames.count, fps: fps, duration: fps.map { Double(frames.count) / $0 }
            ).map { frames[$0] }
        }
        let asset: AVAsset
        switch video {
        case .url(let url): asset = AVURLAsset(url: url)
        case .avAsset(let value): asset = value
        case .frames: preconditionFailure("Decoded frames handled above")
        }
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw ModelExecutionError.invalidOptions("Embedding video has no video track")
        }
        let duration = try await asset.load(.duration).seconds
        let fps = Double(try await track.load(.nominalFrameRate))
        guard duration > 0, duration.isFinite, fps > 0, fps.isFinite else {
            throw ModelExecutionError.invalidOptions("Embedding video has invalid frame metadata")
        }
        let selected = try indices(
            count: max(1, Int((duration * fps).rounded())), fps: fps, duration: duration)
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero
        var frames: [UserInput.VideoFrame] = []
        for index in selected {
            try Task.checkCancellation()
            let timestamp = CMTime(seconds: Double(index) / fps, preferredTimescale: 60000)
            let result = try await generator.image(at: timestamp)
            frames.append(
                .init(image: .ciImage(CIImage(cgImage: result.image)), timeStamp: timestamp))
        }
        return frames
    }

    private func pack(_ pixels: [MLXArray]) -> MLXArray {
        let height = pixels.map { $0.dim(2) }.max()!
        let width = pixels.map { $0.dim(3) }.max()!
        return concatenated(
            pixels.map {
                padded($0, widths: [0, 0, [0, height - $0.dim(2)], [0, width - $0.dim(3)]])
            }, axis: 0)
    }

    func prepare(input: UserInput) async throws -> LMInput {
        try Task.checkCancellation()
        guard input.tools?.isEmpty != false else {
            throw ModelExecutionError.invalidOptions("Embedding inputs cannot contain tools")
        }
        guard input.images.isEmpty && input.videos.isEmpty || model.vision != nil else {
            throw ModelExecutionError.invalidOptions(
                "This EmbeddingGemma 2 checkpoint has no vision tower")
        }
        guard input.audios.isEmpty || model.audio != nil else {
            throw ModelExecutionError.invalidOptions(
                "This EmbeddingGemma 2 checkpoint has no audio tower")
        }
        var tokens: [Int]
        switch input.prompt {
        case .text(var text):
            let placeholders = ["<|image|>", "<|video|>", "<|audio|>"]
            if !placeholders.contains(where: { text.contains($0) }) {
                text =
                    String(repeating: placeholders[0], count: input.images.count)
                    + String(repeating: placeholders[1], count: input.videos.count)
                    + String(repeating: placeholders[2], count: input.audios.count) + text
            }
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ModelExecutionError.invalidOptions(
                    "Embedding input must contain text or media")
            }
            tokens = tokenizer.encode(text: text, addSpecialTokens: true)
        case .messages, .chat:
            tokens = try tokenizer.applyChatTemplate(
                messages: Gemma4UnifiedAudioMessageGenerator().generate(from: input),
                tools: nil, additionalContext: input.additionalContext)
        }
        let imageProcessor = Gemma4Processor(imageConfiguration, tokenizer: tokenizer)
        var imagePixels: [MLXArray] = []
        var imageSizes: [THW] = []
        var imageCounts: [Int] = []
        for image in input.images {
            try Task.checkCancellation()
            let (pixels, size) = try imageProcessor.preprocess(
                image: image.asCIImage(), processing: input.processing)
            imagePixels.append(pixels)
            imageSizes.append(size)
            imageCounts.append(imageConfiguration.softTokenCount(height: size.h, width: size.w))
        }
        let videoProcessor = Gemma4Processor(videoConfiguration, tokenizer: tokenizer)
        var videoPixels: [MLXArray] = []
        var videoSizes: [THW] = []
        var videoReplacements: [[Int]] = []
        for video in input.videos {
            let frames = try await videoFrames(video, processing: input.processing.video)
            var replacement: [Int] = []
            for (index, frame) in frames.enumerated() {
                let image = try frame.image.asCIImage()
                let (pixels, size) = try videoProcessor.preprocess(
                    image: image, processing: input.processing)
                videoPixels.append(pixels)
                videoSizes.append(size)
                if timestamps {
                    let seconds = max(0, Int(frame.timeStamp.seconds))
                    let prefix =
                        (index > 0 ? " " : "")
                        + String(format: "%02d:%02d ", seconds / 60, seconds % 60)
                    replacement += tokenizer.encode(text: prefix, addSpecialTokens: false)
                }
                replacement +=
                    [model.beginImageTokenId]
                    + Array(
                        repeating: model.videoTokenId,
                        count: videoConfiguration.softTokenCount(height: size.h, width: size.w))
                    + [model.endImageTokenId]
            }
            videoReplacements.append(replacement)
        }
        var audio: LMInput.ProcessedAudio?
        var audioCounts: [Int] = []
        if !input.audios.isEmpty {
            var processing = input.processing.audio
            processing.sampleRate = Double(audioConfiguration.samplingRate)
            processing.channels = 1
            var samples: [MLXArray] = []
            for item in input.audios {
                let waveform = Gemma4AudioFeatureExtractor.monoFloatSamples(
                    try await item.asMLXArray(processing: processing))
                if let limit = audioConfiguration.maxLength, waveform.size > limit {
                    throw ModelExecutionError.invalidOptions(
                        "Embedding audio exceeds the checkpoint sample limit")
                }
                samples.append(waveform)
            }
            let extracted = try Gemma4AudioFeatureExtractor(audioConfiguration)(samples)
            // Finish the FFT graph on the preparing thread before transferring arrays to inference.
            eval(extracted.inputFeatures, extracted.inputFeaturesMask)
            guard extracted.inputFeatures.dim(1) > 0 else {
                throw ModelExecutionError.invalidOptions("Embedding audio is too short")
            }
            audio = .init(features: extracted.inputFeatures, mask: extracted.inputFeaturesMask)
            var mask = extracted.inputFeaturesMask
            for _ in 0 ..< 2 { mask = mask[0..., .stride(by: 2)] }
            audioCounts = sum(mask.asType(.int32), axis: 1).asArray(Int.self)
        }
        let imageReplacements = imageCounts.map {
            [model.beginImageTokenId] + Array(repeating: model.imageTokenId, count: $0) + [
                model.endImageTokenId
            ]
        }
        let audioReplacements = audioCounts.map {
            [model.beginAudioTokenId] + Array(repeating: model.audioTokenId, count: $0) + [
                model.endAudioTokenId
            ]
        }
        let replacements = [
            model.imageTokenId: imageReplacements, model.videoTokenId: videoReplacements,
            model.audioTokenId: audioReplacements,
        ]
        var used: [Int: Int] = [:]
        var expanded: [Int] = []
        for token in tokens {
            if let rows = replacements[token] {
                let index = used[token, default: 0]
                guard index < rows.count else {
                    throw ModelExecutionError.invalidOptions(
                        "Embedding media placeholders do not match inputs")
                }
                expanded += rows[index]
                used[token] = index + 1
            } else {
                expanded.append(token)
            }
            guard expanded.count <= maximumLength else {
                throw ModelExecutionError.invalidOptions(
                    "Embedding input exceeds the checkpoint token limit")
            }
        }
        guard replacements.allSatisfy({ used[$0.key, default: 0] == $0.value.count }),
            !expanded.isEmpty
        else {
            throw ModelExecutionError.invalidOptions(
                "Embedding media placeholders do not match inputs")
        }
        return LMInput(
            text: .init(tokens: MLXArray(expanded).reshaped(1, -1)),
            image: imagePixels.isEmpty ? nil : .init(pixels: pack(imagePixels), frames: imageSizes),
            video: videoPixels.isEmpty ? nil : .init(pixels: pack(videoPixels), frames: videoSizes),
            audio: audio)
    }
}
