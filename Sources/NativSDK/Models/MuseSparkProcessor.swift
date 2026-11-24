import AVFoundation
import CoreImage
import Foundation
import MLX

public struct MuseSparkProcessorConfiguration: Decodable, Sendable {
    struct Media: Decodable, Sendable {
        let minimum: Int
        let maximum: Int
        let mean: [CGFloat]
        let deviation: [CGFloat]
        let frames: Int
        let fps: Double
        let timestampFormat: String

        enum CodingKeys: String, CodingKey {
            case minImage = "min_image_tokens"
            case minImageAlias = "min_num_image_tokens"
            case maxImage = "max_image_tokens"
            case maxImageAlias = "max_num_image_tokens"
            case maxVideo = "max_video_frame_tokens"
            case mean = "image_mean"
            case deviation = "image_std"
            case frames = "num_frames"
            case fps
            case timestampFormat = "frame_timestamp_format"
        }

        init(video: Bool) {
            minimum = 1
            maximum = video ? 256 : 4096
            mean = [0.5, 0.5, 0.5]
            deviation = [0.5, 0.5, 0.5]
            frames = 600
            fps = 2
            timestampFormat = "Time: {timestamp:.2f}s"
        }

        init(from decoder: Decoder) throws { try self.init(from: decoder, video: false) }

        init(from decoder: Decoder, video: Bool) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            let defaults = Self(video: video)
            if video {
                minimum = 1
                maximum = try c.decodeIfPresent(Int.self, forKey: .maxVideo) ?? defaults.maximum
            } else {
                minimum =
                    try c.decodeIfPresent(Int.self, forKey: .minImage)
                    ?? c.decodeIfPresent(Int.self, forKey: .minImageAlias) ?? defaults.minimum
                maximum =
                    try c.decodeIfPresent(Int.self, forKey: .maxImage)
                    ?? c.decodeIfPresent(Int.self, forKey: .maxImageAlias) ?? defaults.maximum
            }
            mean = try c.decodeIfPresent([CGFloat].self, forKey: .mean) ?? defaults.mean
            deviation =
                try c.decodeIfPresent([CGFloat].self, forKey: .deviation) ?? defaults.deviation
            frames = try c.decodeIfPresent(Int.self, forKey: .frames) ?? defaults.frames
            fps = try c.decodeIfPresent(Double.self, forKey: .fps) ?? defaults.fps
            timestampFormat =
                try c.decodeIfPresent(String.self, forKey: .timestampFormat)
                ?? defaults.timestampFormat
            guard minimum > 0, maximum >= minimum, frames > 0, fps.isFinite, fps > 0,
                mean.count == 3, deviation.count == 3, mean.allSatisfy(\.isFinite),
                deviation.allSatisfy({ $0 > 0 && $0.isFinite }),
                !video || timestampFormat.contains("{timestamp:.2f}")
            else { throw VLMError.processing("Invalid Muse Spark image or video settings.") }
        }
    }

    let model: MuseSparkConfiguration
    let image: Media
    let video: Media
    let sampleRate: Double

    enum CodingKeys: String, CodingKey {
        case image = "image_processor"
        case video = "video_processor"
        case featureExtractor = "feature_extractor"
        case sampleRate = "sampling_rate"
    }

    public init(from decoder: Decoder) throws {
        model = try MuseSparkConfiguration(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        image = try c.decodeIfPresent(Media.self, forKey: .image) ?? Media(video: false)
        if c.contains(.video), try !c.decodeNil(forKey: .video) {
            video = try Media(from: c.superDecoder(forKey: .video), video: true)
        } else {
            video = Media(video: true)
        }
        if c.contains(.featureExtractor), try !c.decodeNil(forKey: .featureExtractor) {
            let audio = try c.nestedContainer(keyedBy: CodingKeys.self, forKey: .featureExtractor)
            sampleRate = try audio.decodeIfPresent(Double.self, forKey: .sampleRate) ?? 24000
        } else {
            sampleRate = 24000
        }
        guard sampleRate.isFinite, sampleRate > 0 else {
            throw VLMError.processing("Invalid Muse Spark audio sampling rate.")
        }
    }
}

public struct MuseSparkProcessor: UserInputProcessor {
    let config: MuseSparkProcessorConfiguration
    let tokenizer: any Tokenizer

    public init(_ config: MuseSparkProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    private func special(_ text: String, fallback: Int) -> Int {
        tokenizer.convertTokenToId(text) ?? fallback
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let images = MediaProcessing.images(from: input)
        let videos = input.videos
        let audios = MediaProcessing.audios(from: input)
        if (!images.isEmpty || !videos.isEmpty) && config.model.vision == nil {
            throw VLMError.processing(
                "Muse Spark requires a checkpoint with a vision tower for images and video.")
        }
        if !audios.isEmpty && config.model.audio == nil {
            throw VLMError.processing(
                "Muse Spark requires an audio-inclusive checkpoint for waveform input.")
        }
        let messages = MediaFirstMessageGenerator().generate(from: input)
        let tokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools, additionalContext: input.additionalContext)
        let imToken = config.model.imageToken
        let vidToken = config.model.videoToken
        let auToken = config.model.audioToken
        guard tokens.count(where: { $0 == imToken }) == images.count,
            tokens.count(where: { $0 == vidToken }) == videos.count,
            tokens.count(where: { $0 == auToken }) == audios.count
        else {
            throw VLMError.processing(
                "Muse Spark requires one matching placeholder for each media input.")
        }

        var image: LMInput.ProcessedImage?
        var video: LMInput.ProcessedVideo?
        var audio: LMInput.ProcessedAudio?
        var replacements: [Int: [[Int]]] = [:]
        if let geometry = config.model.vision?.geometry {
            let mergeArea = geometry.mergeSize * geometry.mergeSize
            if !images.isEmpty {
                var tensors: [MLXArray] = []
                var grids: [THW] = []
                var spans: [[Int]] = []
                for item in images {
                    try Task.checkCancellation()
                    let frame = try item.asCIImage()
                    let dims = try targetSize(
                        frame, settings: config.image,
                        geometry: geometry, processing: input.processing)
                    let pixels = resample(
                        frame, height: dims.0, width: dims.1, settings: config.image)
                    let grid = THW(1, dims.0 / geometry.patchSize, dims.1 / geometry.patchSize)
                    grids.append(grid)
                    tensors.append(
                        museGlimmerPatchify(
                            pixels, gridH: grid.h, gridW: grid.w, patchSize: geometry.patchSize,
                            temporalPatchSize: geometry.patchTemporal))
                    spans.append(
                        [special("<|image_start|>", fallback: 200080)]
                            + Array(repeating: imToken, count: grid.product / mergeArea)
                            + [special("<|image_end|>", fallback: 200081)])
                }
                image = .init(pixels: concatenated(tensors, axis: 0), frames: grids)
                replacements[imToken] = spans
            }

            if !videos.isEmpty {
                var tensors: [MLXArray] = []
                var grids: [THW] = []
                var spans: [[Int]] = []
                for item in videos {
                    try Task.checkCancellation()
                    let clip = try await prepareVideo(
                        item, geometry: geometry, processing: input.processing)
                    tensors.append(clip.patches)
                    grids.append(clip.grid)
                    let perGroup = clip.grid.h * clip.grid.w / mergeArea
                    var span = [special("<|vid_start|>", fallback: 200082)]
                    for (index, time) in clip.timestamps.enumerated() {
                        if index > 0 {
                            span.append(special("<|vid_frame_separator|>", fallback: 200087))
                        }
                        let seconds = String(
                            format: "%.2f", locale: Locale(identifier: "en_US_POSIX"), time)
                        let timestamp = config.video.timestampFormat.replacingOccurrences(
                            of: "{timestamp:.2f}", with: seconds)
                        span.append(
                            contentsOf: tokenizer.encode(text: timestamp, addSpecialTokens: false))
                        span.append(contentsOf: repeatElement(vidToken, count: perGroup))
                    }
                    span.append(special("<|vid_end|>", fallback: 200083))
                    spans.append(span)
                }
                video = .init(pixels: concatenated(tensors, axis: 0), frames: grids)
                replacements[vidToken] = spans
            }
        }

        if !audios.isEmpty, let audioConfig = config.model.audio {
            var processing = input.processing.audio
            processing.sampleRate = config.sampleRate
            processing.channels = 1
            var waves: [MLXArray] = []
            for item in audios {
                try Task.checkCancellation()
                let waveform = try await item.asMLXArray(processing: processing)
                guard waveform.ndim == 1, waveform.dim(0) > 0 else {
                    throw VLMError.processing(
                        "Muse Spark audio inputs must be nonempty mono waveforms.")
                }
                waves.append(waveform.asType(.float32))
            }
            let stride = audioConfig.samplesPerToken
            let lengths = waves.map { $0.dim(0) }
            let longest = lengths.max()!
            let paddedLength = (longest + stride - 1) / stride * stride
            let batch = stacked(
                waves.map { padded($0, widths: [[0, paddedLength - $0.dim(0)]]) }, axis: 0)
            let mask =
                MLXArray(0 ..< paddedLength)[.newAxis, 0...]
                .< MLXArray(lengths)[0..., .newAxis]
            audio = .init(features: batch, mask: mask)
            replacements[auToken] = lengths.map { length in
                [special("<|soft_audio_start|>", fallback: 201463)]
                    + Array(repeating: auToken, count: (length + stride - 1) / stride)
                    + [special("<|soft_audio_end|>", fallback: 201464)]
            }
        }

        var expanded: [Int] = []
        var used: [Int: Int] = [:]
        for token in tokens {
            if let spans = replacements[token] {
                let index = used[token, default: 0]
                expanded.append(contentsOf: spans[index])
                used[token] = index + 1
            } else {
                expanded.append(token)
            }
        }
        let ids = MLXArray(expanded).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: ids), image: image, video: video, audio: audio)
    }

    static func resizeTarget(
        height: Int, width: Int, factor: Int, minimum: Int, maximum: Int
    ) -> (Int, Int) {
        var h = Double(height) / Double(factor)
        var w = Double(width) / Double(factor)
        let ratio = w / h
        let target = min(max(h * w, Double(minimum)), Double(maximum))
        if target != h * w {
            h = sqrt(target / ratio)
            w = h * ratio
        }
        let source = Double(height) / Double(width)
        let rows = [Int(floor(h)), Int(ceil(h))]
        let columns = [Int(floor(w)), Int(ceil(w))]
        var best: (Int, Int)?
        var difference = Double.infinity
        for row in rows where row >= 1 {
            for column in columns where column >= 1 {
                let count = row * column
                let distortion = abs(Double(row) / Double(column) - source)
                if (minimum ... maximum).contains(count), distortion < difference {
                    best = (row, column)
                    difference = distortion
                }
            }
        }
        if best == nil {
            for row in 1 ... maximum {
                let low = (minimum + row - 1) / row
                let high = maximum / row
                guard low <= high else { continue }
                let column = min(max(w, Double(low)), Double(high))
                let dh = Double(row) - h
                let dw = column - w
                let distance = dh * dh + dw * dw
                if distance < difference {
                    best = (row, Int(column))
                    difference = distance
                }
            }
        }
        return (best!.0 * factor, best!.1 * factor)
    }

    private func targetSize(
        _ image: CIImage, settings: MuseSparkProcessorConfiguration.Media,
        geometry: MuseGlimmerVisionConfiguration, processing: UserInput.Processing
    ) throws -> (Int, Int) {
        let extent = image.extent.integral
        guard extent.height.isFinite, extent.width.isFinite, extent.height > 0, extent.width > 0
        else {
            throw VLMError.processing(
                "Muse Spark images and video frames must have finite, nonzero bounds.")
        }
        let factor = geometry.patchSize * geometry.mergeSize
        let area = factor * factor
        let bound = processing.resize ?? extent.size
        let height = max(1, Int(min(extent.height, bound.height).rounded()))
        let width = max(1, Int(min(extent.width, bound.width).rounded()))
        let maximum = min(settings.maximum, max(1, (processing.maxPixels ?? Int.max) / area))
        let minimum = min(maximum, max(settings.minimum, (processing.minPixels ?? 0) / area))
        return Self.resizeTarget(
            height: height, width: width, factor: factor, minimum: minimum, maximum: maximum)
    }

    private func resample(
        _ image: CIImage, height: Int, width: Int, settings: MuseSparkProcessorConfiguration.Media
    ) -> MLXArray {
        let source = MediaProcessing.zeroOrigin(image)
        let size = CGSize(width: width, height: height)
        let canvas =
            source.extent.size == size ? source : MediaProcessing.resampleLanczos(source, to: size)
        let pixels = MediaProcessing.renderRGB8(canvas, width: width, height: height)
        return MLXArray(
            MediaProcessing.normalizeCHW(
                pixels, width: width, height: height, mean: settings.mean, std: settings.deviation),
            [1, 3, height, width])
    }

    private struct VideoPatches {
        let patches: MLXArray
        let grid: THW
        let timestamps: [Double]
    }

    private func prepareVideo(
        _ video: UserInput.Video, geometry: MuseGlimmerVisionConfiguration,
        processing: UserInput.Processing
    ) async throws -> VideoPatches {
        let frames: [UserInput.VideoFrame]
        switch video {
        case .frames(let decoded):
            guard let first = decoded.first, let last = decoded.last else {
                throw VLMError.processing("Muse Spark video input contains no frames.")
            }
            if processing.video.sampling == nil {
                frames =
                    decoded.count <= config.video.frames
                    ? decoded
                    : Self.uniformIndices(
                        total: decoded.count, count: config.video.frames
                    ).map { decoded[$0] }
            } else {
                let duration = max(0, last.timeStamp.seconds - first.timeStamp.seconds)
                let count = try videoFrameCount(
                    total: decoded.count, duration: duration, temporal: geometry.patchTemporal,
                    processing: processing.video)
                frames = Self.uniformIndices(total: decoded.count, count: count).map { decoded[$0] }
            }
        case .url(let url):
            frames = try await readVideo(
                AVURLAsset(url: url), temporal: geometry.patchTemporal, processing: processing.video
            )
        case .avAsset(let asset):
            frames = try await readVideo(
                asset, temporal: geometry.patchTemporal, processing: processing.video)
        }
        let times = frames.map { $0.timeStamp.seconds }
        guard !times.isEmpty, times.allSatisfy({ $0.isFinite && $0 >= 0 }),
            zip(times, times.dropFirst()).allSatisfy({ $0 <= $1 })
        else {
            throw VLMError.processing(
                "Muse Spark video timestamps must be nonnegative and ordered.")
        }
        let images = try frames.map { try $0.image.asCIImage() }
        let dimensions = try targetSize(
            images[0], settings: config.video, geometry: geometry, processing: processing)
        var pixels = images.map {
            resample($0, height: dimensions.0, width: dimensions.1, settings: config.video)
        }
        let temporal = geometry.patchTemporal
        let padding = (temporal - pixels.count % temporal) % temporal
        if padding > 0 { pixels.append(contentsOf: repeatElement(pixels.last!, count: padding)) }
        let t = pixels.count / temporal
        let patch = geometry.patchSize
        let h = dimensions.0 / patch
        let w = dimensions.1 / patch
        let patches = concatenated(pixels, axis: 0).reshaped(t, temporal, 3, h, patch, w, patch)
            .transposed(0, 3, 5, 1, 2, 4, 6).reshaped(t * h * w, temporal * 3 * patch * patch)
        return .init(
            patches: patches, grid: THW(t, h, w),
            timestamps: stride(from: 0, to: times.count, by: temporal).map { times[$0] })
    }

    private func videoFrameCount(
        total: Int, duration: Double, temporal: Int, processing: UserInput.VideoProcessing
    ) throws -> Int {
        let limit = min(total, config.video.frames)
        if let target = processing.targetFrames {
            guard target > 0 else {
                throw VLMError.processing("Muse Spark video frame count must be positive.")
            }
            return min(limit, target)
        }
        let fps = processing.targetFramesPerSecond ?? config.video.fps
        guard fps.isFinite, fps > 0, duration.isFinite else {
            throw VLMError.processing("Muse Spark video sampling rate must be positive and finite.")
        }
        let target = Int(min(Double(limit), duration * fps))
        return max(min(limit, temporal), target / temporal * temporal)
    }

    private static func uniformIndices(total: Int, count: Int) -> [Int] {
        guard count > 1 else { return [0] }
        let step = Double(total - 1) / Double(count - 1)
        return (0 ..< (count - 1)).map { Int(Double($0) * step) } + [total - 1]
    }

    private func readVideo(
        _ asset: AVAsset, temporal: Int, processing: UserInput.VideoProcessing
    ) async throws -> [UserInput.VideoFrame] {
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw VLMError.noVideoTrackFound
        }
        guard try await track.load(.isDecodable) else { throw VLMError.videoNotDecodable }
        let range = try await track.load(.timeRange)
        let nominal = Double(try await track.load(.nominalFrameRate))
        let sourceFPS = nominal.isFinite && nominal > 0 ? nominal : 24
        let duration = range.duration.seconds
        guard duration.isFinite, duration > 0, duration * sourceFPS < Double(Int.max) else {
            throw VLMError.videoNotDecodable
        }
        let total = max(1, Int((duration * sourceFPS).rounded()))
        let count = try videoFrameCount(
            total: total, duration: Double(total) / sourceFPS, temporal: temporal,
            processing: processing)
        let times = Self.uniformIndices(total: total, count: count).map {
            CMTimeAdd(
                range.start, CMTime(seconds: Double($0) / sourceFPS, preferredTimescale: 60000))
        }
        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        let tolerance = CMTime(seconds: 0.45 / sourceFPS, preferredTimescale: 60000)
        generator.requestedTimeToleranceBefore = tolerance
        generator.requestedTimeToleranceAfter = tolerance
        var result: [(CMTime, UserInput.VideoFrame)] = []
        for await item in generator.images(for: times) {
            try Task.checkCancellation()
            switch item {
            case .success(let requested, let image, let actual):
                let timestamp = CMTimeMaximum(.zero, CMTimeSubtract(actual, range.start))
                result.append(
                    (
                        requested,
                        .init(image: .ciImage(CIImage(cgImage: image)), timeStamp: timestamp)
                    ))
            case .failure(_, let error):
                throw error
            }
        }
        guard result.count == times.count else { throw VLMError.videoNotDecodable }
        return result.sorted { $0.0 < $1.0 }.map(\.1)
    }
}
