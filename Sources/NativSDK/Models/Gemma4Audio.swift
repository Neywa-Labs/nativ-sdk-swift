import CoreImage
import Foundation
import MLX

public struct Gemma4AudioFeatureExtractorConfiguration: Codable, Sendable {
    public let featureSize: Int
    public let samplingRate: Int
    public let paddingValue: Float
    public let frameLengthMs: Float
    public let hopLengthMs: Float
    public let minFrequency: Float
    public let maxFrequency: Float
    public let preemphasis: Float
    public let preemphasisHTKFlavor: Bool
    public let fftOverdrive: Bool
    public let dither: Float
    public let inputScaleFactor: Float
    public let melFloor: Float
    public let perBinMean: [Float]?
    public let perBinStddev: [Float]?
    public let maxLength: Int?
    public let padToMultipleOf: Int?

    enum CodingKeys: String, CodingKey {
        case featureSize = "feature_size"
        case samplingRate = "sampling_rate"
        case paddingValue = "padding_value"
        case frameLengthMs = "frame_length_ms"
        case hopLengthMs = "hop_length_ms"
        case minFrequency = "min_frequency"
        case maxFrequency = "max_frequency"
        case preemphasis
        case preemphasisHTKFlavor = "preemphasis_htk_flavor"
        case fftOverdrive = "fft_overdrive"
        case dither
        case inputScaleFactor = "input_scale_factor"
        case melFloor = "mel_floor"
        case perBinMean = "per_bin_mean"
        case perBinStddev = "per_bin_stddev"
        case maxLength = "max_length"
        case padToMultipleOf = "pad_to_multiple_of"
    }

    public init(
        featureSize: Int = 128,
        samplingRate: Int = 16000,
        paddingValue: Float = 0,
        frameLengthMs: Float = 20,
        hopLengthMs: Float = 10,
        minFrequency: Float = 0,
        maxFrequency: Float = 8000,
        preemphasis: Float = 0,
        preemphasisHTKFlavor: Bool = true,
        fftOverdrive: Bool = false,
        dither: Float = 0,
        inputScaleFactor: Float = 1,
        melFloor: Float = 1e-3,
        perBinMean: [Float]? = nil,
        perBinStddev: [Float]? = nil,
        maxLength: Int? = 480_000,
        padToMultipleOf: Int? = 128,
    ) {
        self.featureSize = featureSize
        self.samplingRate = samplingRate
        self.paddingValue = paddingValue
        self.frameLengthMs = frameLengthMs
        self.hopLengthMs = hopLengthMs
        self.minFrequency = minFrequency
        self.maxFrequency = maxFrequency
        self.preemphasis = preemphasis
        self.preemphasisHTKFlavor = preemphasisHTKFlavor
        self.fftOverdrive = fftOverdrive
        self.dither = dither
        self.inputScaleFactor = inputScaleFactor
        self.melFloor = melFloor
        self.perBinMean = perBinMean
        self.perBinStddev = perBinStddev
        self.maxLength = maxLength
        self.padToMultipleOf = padToMultipleOf
    }

    public init(from decoder: any Decoder) throws {
        try self.init(from: decoder, defaults: .init())
    }

    init(from decoder: any Decoder, defaults: Self) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        featureSize = try c.decodeIfPresent(Int.self, forKey: .featureSize) ?? defaults.featureSize
        samplingRate =
            try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? defaults.samplingRate
        paddingValue =
            try c.decodeIfPresent(Float.self, forKey: .paddingValue) ?? defaults.paddingValue
        frameLengthMs =
            try c.decodeIfPresent(Float.self, forKey: .frameLengthMs) ?? defaults.frameLengthMs
        hopLengthMs =
            try c.decodeIfPresent(Float.self, forKey: .hopLengthMs) ?? defaults.hopLengthMs
        minFrequency =
            try c.decodeIfPresent(Float.self, forKey: .minFrequency) ?? defaults.minFrequency
        maxFrequency =
            try c.decodeIfPresent(Float.self, forKey: .maxFrequency) ?? defaults.maxFrequency
        preemphasis =
            try c.decodeIfPresent(Float.self, forKey: .preemphasis) ?? defaults.preemphasis
        preemphasisHTKFlavor =
            try c.decodeIfPresent(Bool.self, forKey: .preemphasisHTKFlavor)
            ?? defaults.preemphasisHTKFlavor
        fftOverdrive =
            try c.decodeIfPresent(Bool.self, forKey: .fftOverdrive) ?? defaults.fftOverdrive
        dither = try c.decodeIfPresent(Float.self, forKey: .dither) ?? defaults.dither
        inputScaleFactor =
            try c.decodeIfPresent(Float.self, forKey: .inputScaleFactor)
            ?? defaults.inputScaleFactor
        melFloor = try c.decodeIfPresent(Float.self, forKey: .melFloor) ?? defaults.melFloor
        perBinMean = try c.decodeIfPresent([Float].self, forKey: .perBinMean) ?? defaults.perBinMean
        perBinStddev =
            try c.decodeIfPresent([Float].self, forKey: .perBinStddev) ?? defaults.perBinStddev
        maxLength = try c.decodeIfPresent(Int.self, forKey: .maxLength) ?? defaults.maxLength
        padToMultipleOf =
            try c.decodeIfPresent(Int.self, forKey: .padToMultipleOf) ?? defaults.padToMultipleOf
    }
}

public struct Gemma4AudioFeatureExtractor: Sendable {
    public let configuration: Gemma4AudioFeatureExtractorConfiguration

    enum Variant {
        case gemma3n, gemma4
    }

    private let variant: Variant
    private let frameLength: Int
    private let padLeft: Int
    private let hopLength: Int
    private let fftLength: Int
    private let window: MaterializedArray
    private let melFilters: MaterializedArray

    public init(_ configuration: Gemma4AudioFeatureExtractorConfiguration = .init()) {
        self.init(configuration, variant: .gemma4)
    }

    init(_ configuration: Gemma4AudioFeatureExtractorConfiguration, variant: Variant) {
        self.configuration = configuration
        self.variant = variant
        let computedFrameLength = max(
            1,
            Int(round(Float(configuration.samplingRate) * configuration.frameLengthMs / 1000)),
        )
        let computedHopLength = max(
            1,
            Int(round(Float(configuration.samplingRate) * configuration.hopLengthMs / 1000)),
        )
        var computedFFTLength = Self.nextPowerOfTwo(computedFrameLength)
        if configuration.fftOverdrive {
            computedFFTLength *= 2
        }
        frameLength = computedFrameLength
        padLeft = variant == .gemma4 ? computedFrameLength / 2 : 0
        hopLength = computedHopLength
        fftLength = computedFFTLength

        let windowValues = (0 ..< computedFrameLength).map { index in
            Float(
                0.5 - 0.5 * cos(2.0 * Double.pi * Double(index) / Double(computedFrameLength)),
            )
        }
        let window = MLXArray(windowValues, [computedFrameLength])
        let melFilters = Self.makeMelFilterBank(
            frequencyBins: computedFFTLength / 2 + 1,
            melFilters: configuration.featureSize,
            minFrequency: configuration.minFrequency,
            maxFrequency: configuration.maxFrequency,
            samplingRate: configuration.samplingRate,
        )
        let tables = materialize([window, melFilters])
        self.window = tables[0]
        self.melFilters = tables[1]
    }

    public func callAsFunction(_ rawSpeech: [MLXArray]) throws -> (
        inputFeatures: MLXArray, inputFeaturesMask: MLXArray,
    ) {
        guard !rawSpeech.isEmpty else {
            throw VLMError.processing("No audio samples were provided.")
        }

        let waveforms = rawSpeech.map { Self.monoFloatSamples($0) }
        let rawLengths = waveforms.map { $0.dim(0) }
        guard let longest = rawLengths.max(), longest > 0 else {
            throw VLMError.processing("Audio input is empty.")
        }

        var targetLength = longest
        if let maxLength = configuration.maxLength {
            targetLength = min(targetLength, maxLength)
        }
        if let multiple = configuration.padToMultipleOf, multiple > 0,
            targetLength % multiple != 0
        {
            targetLength = ((targetLength / multiple) + 1) * multiple
        }

        var features: [MLXArray] = []
        var masks: [MLXArray] = []
        features.reserveCapacity(waveforms.count)
        masks.reserveCapacity(waveforms.count)

        for waveform in waveforms {
            let validLength = min(waveform.dim(0), targetLength)
            var paddedWaveform =
                validLength < waveform.dim(0) ? waveform[..<validLength] : waveform

            if validLength < targetLength {
                paddedWaveform = padded(
                    paddedWaveform,
                    widths: [.init((0, targetLength - validLength))],
                    value: MLXArray(configuration.paddingValue, dtype: paddedWaveform.dtype),
                )
            }

            let attentionMask =
                Array(repeating: true, count: validLength)
                + Array(repeating: false, count: max(0, targetLength - validLength))
            let (spectrogram, spectrogramMask) = try extractSpectrogram(
                paddedWaveform, attentionMask: attentionMask,
            )
            features.append(spectrogram)
            masks.append(spectrogramMask)
        }

        return (stacked(features, axis: 0), stacked(masks, axis: 0))
    }

    func prepare(audios: [UserInput.Audio], processing: UserInput.Processing) async throws
        -> (audio: LMInput.ProcessedAudio, tokenCounts: [Int])
    {
        var audioProcessing = processing.audio
        audioProcessing.sampleRate = Double(configuration.samplingRate)
        audioProcessing.channels = 1
        var samples: [MLXArray] = []
        for audio in audios {
            samples.append(try await audio.asMLXArray(processing: audioProcessing))
        }
        let extracted = try self(samples)
        // FFT padding can create thread-local MLX work. Finish it before the
        // prepared tensors cross an async boundary into the generation worker.
        eval(extracted.inputFeatures, extracted.inputFeaturesMask)
        // The two stride-2 convolutions keep mask entries 0, 4, 8, ... .
        // Count those valid rows, including for unequal-length clips, rather
        // than estimating from duration and rounding at frame boundaries.
        let tokenCounts = (0 ..< audios.count).map { index in
            let mask = extracted.inputFeaturesMask[index].asArray(Bool.self)
            return stride(from: 0, to: mask.count, by: 4).filter { mask[$0] }.count
        }
        guard tokenCounts.allSatisfy({ $0 > 0 }) else {
            throw VLMError.processing("Gemma4 audio is too short to produce encoder features.")
        }
        return (
            .init(features: extracted.inputFeatures, mask: extracted.inputFeaturesMask),
            tokenCounts
        )
    }

    private func extractSpectrogram(
        _ waveform: MLXArray, attentionMask: [Bool],
    )
        throws -> (MLXArray, MLXArray)
    {
        var paddedWaveform = padded(waveform, widths: [.init((padLeft, 0))])
        if configuration.dither > 0 {
            throw VLMError.processing("Gemma4 audio dither is not supported in Swift.")
        }
        if configuration.inputScaleFactor != 1 {
            paddedWaveform = paddedWaveform * MLXArray(configuration.inputScaleFactor)
        }

        let paddedAttentionMask =
            Array(repeating: false, count: padLeft) + attentionMask
        let frameSizeForUnfold = frameLength + 1
        let available = paddedWaveform.dim(0) - frameSizeForUnfold
        guard available >= 0 else {
            return (
                MLXArray.zeros([0, configuration.featureSize], dtype: .float32),
                MLXArray([Bool](), [0]),
            )
        }

        let frameCount = available / hopLength + 1
        let frameRows = (0 ..< frameCount).map { frameIndex in
            let start = frameIndex * hopLength
            return paddedWaveform[start ..< (start + frameSizeForUnfold)]
        }
        let framesToProcess = stacked(frameRows, axis: 0)

        let frames: MLXArray
        if configuration.preemphasis > 0 {
            if configuration.preemphasisHTKFlavor {
                let first =
                    framesToProcess[0..., ..<1]
                    * MLXArray(1 - configuration.preemphasis)
                let rest =
                    framesToProcess[0..., 1 ..< frameLength]
                    - MLXArray(configuration.preemphasis)
                    * framesToProcess[0..., 0 ..< (frameLength - 1)]
                frames = concatenated([first, rest], axis: -1)
            } else {
                frames =
                    framesToProcess[0..., 1...]
                    - MLXArray(configuration.preemphasis)
                    * framesToProcess[0..., ..<frameLength]
            }
        } else {
            frames = framesToProcess[0..., ..<frameLength]
        }

        let windowedFrames = frames * window.asType(frames.dtype)
        let stft = MLXFFT.rfft(windowedFrames, n: fftLength, axis: -1)
        let magnitude = stft.abs()
        let melSpectrogram = magnitude.matmul(melFilters.asType(magnitude.dtype))
        // Gemma 3n clamps the magnitude floor; Gemma 4 adds it before the log.
        var logMelSpectrogram = log(
            variant == .gemma3n
                ? maximum(melSpectrogram, MLXArray(configuration.melFloor))
                : melSpectrogram + MLXArray(configuration.melFloor))

        if let perBinMean = configuration.perBinMean {
            guard perBinMean.count == configuration.featureSize else {
                throw VLMError.processing("Gemma4 per_bin_mean size does not match feature_size.")
            }
            logMelSpectrogram =
                logMelSpectrogram - MLXArray(perBinMean, [1, configuration.featureSize])
        }
        if let perBinStddev = configuration.perBinStddev {
            guard perBinStddev.count == configuration.featureSize else {
                throw VLMError.processing(
                    "Gemma4 per_bin_stddev size does not match feature_size.",
                )
            }
            logMelSpectrogram =
                logMelSpectrogram / MLXArray(perBinStddev, [1, configuration.featureSize])
        }

        let frameMaskValues = (0 ..< frameCount).map { frameIndex in
            let maskIndex =
                frameIndex * hopLength
                + (variant == .gemma4 ? frameSizeForUnfold - 1 : 0)
            return paddedAttentionMask[maskIndex]
        }
        let frameMask = MLXArray(frameMaskValues, [frameCount])
        if variant == .gemma3n {
            return (logMelSpectrogram.asType(.float32), frameMask)
        }
        let maskedSpectrogram =
            logMelSpectrogram.asType(DType.float32)
            * expandedDimensions(frameMask.asType(DType.float32), axis: -1)
        return (maskedSpectrogram, frameMask)
    }

    static func monoFloatSamples(_ input: MLXArray) -> MLXArray {
        var samples = input.asType(.float32)
        if samples.ndim == 2 {
            if samples.dim(-1) <= 8 {
                samples = mean(samples, axis: -1)
            } else if samples.dim(0) <= 8 {
                samples = mean(samples, axis: 0)
            }
        } else if samples.ndim > 2 {
            samples = samples.flattened()
        }
        return samples.flattened()
    }

    private static func nextPowerOfTwo(_ value: Int) -> Int {
        var result = 1
        while result < value {
            result <<= 1
        }
        return result
    }

    private static func hertzToMel(_ frequency: Float) -> Float {
        2595 * log10(1 + frequency / 700)
    }

    private static func melToHertz(_ mel: Float) -> Float {
        700 * (pow(10, mel / 2595) - 1)
    }

    private static func makeMelFilterBank(
        frequencyBins: Int,
        melFilters: Int,
        minFrequency: Float,
        maxFrequency: Float,
        samplingRate: Int,
    )
        -> MLXArray
    {
        let melMin = hertzToMel(minFrequency)
        let melMax = hertzToMel(maxFrequency)
        let melPoints = (0 ..< melFilters + 2).map { index in
            melMin + (melMax - melMin) * Float(index) / Float(melFilters + 1)
        }
        let frequencyPoints = melPoints.map(melToHertz)
        let allFrequencies = (0 ..< frequencyBins).map { index in
            Float(index) * Float(samplingRate) / Float(2 * max(frequencyBins - 1, 1))
        }

        var values = Array(repeating: Float(0), count: frequencyBins * melFilters)
        for filterIndex in 0 ..< melFilters {
            let lower = frequencyPoints[filterIndex]
            let center = frequencyPoints[filterIndex + 1]
            let upper = frequencyPoints[filterIndex + 2]
            for binIndex in 0 ..< frequencyBins {
                let frequency = allFrequencies[binIndex]
                let rising = (frequency - lower) / max(center - lower, 1e-10)
                let falling = (upper - frequency) / max(upper - center, 1e-10)
                values[binIndex * melFilters + filterIndex] = max(0, min(rising, falling))
            }
        }
        return MLXArray(values, [frequencyBins, melFilters])
    }
}

public struct Gemma4UnifiedAudioProcessorConfiguration: Decodable, Sendable {
    public let base: Gemma4UnifiedProcessorConfiguration
    public let boaTokenId: Int
    public let eoaTokenId: Int?
    public let audioFeatureExtractor: Gemma4AudioFeatureExtractorConfiguration

    private struct FeatureExtractorBox: Decodable {
        let featureExtractor: Gemma4AudioFeatureExtractorConfiguration?

        enum CodingKeys: String, CodingKey {
            case featureExtractor = "feature_extractor"
        }
    }

    enum CodingKeys: String, CodingKey {
        case boaTokenId = "boa_token_id"
        case eoaTokenId = "eoa_token_id"
        case eoaTokenIndex = "eoa_token_index"
        case audioSampleRate = "audio_sample_rate"
        case samplingRate = "sampling_rate"
        case audioFeatureSize = "audio_feature_size"
    }

    public init(from decoder: any Decoder) throws {
        base = try Gemma4UnifiedProcessorConfiguration(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        boaTokenId = try c.decodeIfPresent(Int.self, forKey: .boaTokenId) ?? 256_000
        eoaTokenId =
            try c.decodeIfPresent(Int.self, forKey: .eoaTokenId)
            ?? c.decodeIfPresent(Int.self, forKey: .eoaTokenIndex)
            ?? 258_883

        let boxed = try FeatureExtractorBox(from: decoder)
        if let featureExtractor = boxed.featureExtractor {
            audioFeatureExtractor = featureExtractor
        } else {
            audioFeatureExtractor = try Gemma4AudioFeatureExtractorConfiguration(
                featureSize: c.decodeIfPresent(Int.self, forKey: .audioFeatureSize) ?? 128,
                samplingRate: c.decodeIfPresent(Int.self, forKey: .audioSampleRate)
                    ?? c.decodeIfPresent(Int.self, forKey: .samplingRate)
                    ?? 16000,
            )
        }
    }
}

public struct Gemma4UnifiedAudioMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        var dictionary: Message =
            if message.role == .system {
                [
                    "role": message.role.rawValue,
                    "content": message.content,
                ]
            } else {
                [
                    "role": message.role.rawValue,
                    "content": message.images.map { _ in
                        ["type": "image"]
                    }
                        + message.videos.map { _ in
                            ["type": "video"]
                        }
                        + message.audios.map { _ in
                            ["type": "audio"]
                        }
                        + [
                            ["type": "text", "text": message.content]
                        ],
                ]
            }
        addToolMetadata(to: &dictionary, for: message)
        return dictionary
    }
}

public struct Gemma4UnifiedAudioProcessor: UserInputProcessor {
    private let config: Gemma4UnifiedAudioProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let featureExtractor: Gemma4AudioFeatureExtractor

    public init(_ config: Gemma4UnifiedAudioProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        featureExtractor = Gemma4AudioFeatureExtractor(config.audioFeatureExtractor)
    }

    private func patchify(_ pixelValues: MLXArray) -> (MLXArray, MLXArray, Int, THW) {
        let base = config.base
        let channels = pixelValues.dim(1)
        let height = pixelValues.dim(2)
        let width = pixelValues.dim(3)
        let patchHeight = height / base.modelPatchSize
        let patchWidth = width / base.modelPatchSize
        let realCount = min(patchHeight * patchWidth, base.maxSoftTokens)
        let patchDim = base.modelPatchSize * base.modelPatchSize * channels

        var patches = pixelValues.reshaped(
            1, channels, patchHeight, base.modelPatchSize, patchWidth, base.modelPatchSize,
        )
        patches = patches.transposed(0, 2, 4, 3, 5, 1)
        patches = patches.reshaped(patchHeight * patchWidth, patchDim)
        if realCount < patches.dim(0) {
            patches = patches[..<realCount, 0...]
        }
        if realCount < base.maxSoftTokens {
            patches = padded(patches, widths: [.init((0, base.maxSoftTokens - realCount)), 0])
        }

        var positionValues: [Int32] = []
        positionValues.reserveCapacity(base.maxSoftTokens * 2)
        var emitted = 0
        for y in 0 ..< patchHeight {
            for x in 0 ..< patchWidth where emitted < realCount {
                positionValues.append(Int32(x))
                positionValues.append(Int32(y))
                emitted += 1
            }
        }
        while emitted < base.maxSoftTokens {
            positionValues.append(-1)
            positionValues.append(-1)
            emitted += 1
        }
        let positions = MLXArray(positionValues, [base.maxSoftTokens, 2])
        return (patches, positions, realCount, THW(1, height, width))
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        pixels: MLXArray, positionIds: MLXArray, tokenCounts: [Int], frames: [THW],
    ) {
        let base = config.base
        var patchRows: [MLXArray] = []
        var positionRows: [MLXArray] = []
        var tokenCounts: [Int] = []
        var frames: [THW] = []

        for image in images {
            let processedImage = MediaProcessing.apply(image, processing: processing)
            let srgbImage = MediaProcessing.inSRGBToneCurveSpace(processedImage)
            let resizedImage =
                if base.doResize {
                    try MediaProcessing.resampleBicubic(
                        srgbImage,
                        to: base.aspectRatioPreservingSize(for: srgbImage.extent.size),
                    )
                } else {
                    srgbImage
                }

            var pixelValues = MediaProcessing.asMLXArray(resizedImage)
            let rescaleMultiplier = Float(base.doRescale ? base.rescaleFactor * 255 : 255)
            if rescaleMultiplier != 1 {
                pixelValues = pixelValues * MLXArray(rescaleMultiplier, dtype: pixelValues.dtype)
            }
            if base.doNormalize {
                let mean = MLXArray(
                    base.imageMean.map { Float($0) }, [1, base.imageMean.count, 1, 1],
                )
                .asType(pixelValues.dtype)
                let std = MLXArray(
                    base.imageStd.map { Float($0) }, [1, base.imageStd.count, 1, 1],
                )
                .asType(pixelValues.dtype)
                pixelValues = (pixelValues - mean) / std
            }
            let (patches, positions, tokenCount, frame) = patchify(pixelValues)
            patchRows.append(patches)
            positionRows.append(positions)
            tokenCounts.append(tokenCount)
            frames.append(frame)
        }

        return (
            pixels: stacked(patchRows, axis: 0),
            positionIds: stacked(positionRows, axis: 0),
            tokenCounts: tokenCounts,
            frames: frames,
        )
    }

    public func preprocess(audios: [UserInput.Audio], processing: UserInput.Processing) async throws
        -> (audio: LMInput.ProcessedAudio, tokenCounts: [Int])
    {
        var audioProcessing = processing.audio
        audioProcessing.sampleRate = Double(config.audioFeatureExtractor.samplingRate)
        audioProcessing.channels = 1

        var samples: [MLXArray] = []
        samples.reserveCapacity(audios.count)
        for audio in audios {
            let array = try await audio.asMLXArray(processing: audioProcessing)
            samples.append(array)
        }

        let monoSamples = samples.map { Gemma4AudioFeatureExtractor.monoFloatSamples($0) }
        let tokenCounts = monoSamples.map { sample -> Int in
            let maxLength = config.audioFeatureExtractor.maxLength ?? sample.dim(0)
            let sampleCount = min(sample.dim(0), maxLength)
            let durationMs =
                Double(sampleCount) / Double(config.audioFeatureExtractor.samplingRate) * 1000
            return min(
                max(1, Int(ceil(durationMs / Double(max(config.base.audioMsPerToken, 1))))),
                config.base.audioSeqLength,
            )
        }

        let extracted = try featureExtractor(monoSamples)
        return (
            audio: LMInput.ProcessedAudio(
                features: extracted.inputFeatures,
                mask: extracted.inputFeaturesMask,
            ),
            tokenCounts: tokenCounts,
        )
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        if !input.videos.isEmpty {
            throw VLMError.processing("Gemma4 unified video processing is not implemented.")
        }

        let messages = Gemma4UnifiedAudioMessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages, tools: input.tools,
            additionalContext: input.additionalContext,
        )

        var processedImage: LMInput.ProcessedImage?
        var imageTokenCounts: [Int] = []
        if !input.images.isEmpty {
            let imageData = try preprocess(
                images: input.images.map { try $0.asCIImage() },
                processing: input.processing,
            )
            processedImage = LMInput.ProcessedImage(
                pixels: imageData.pixels,
                positionIds: imageData.positionIds,
                frames: imageData.frames,
            )
            imageTokenCounts = imageData.tokenCounts
        }

        var processedAudio: LMInput.ProcessedAudio?
        var audioTokenCounts: [Int] = []
        if !input.audios.isEmpty {
            let audioData = try await preprocess(audios: input.audios, processing: input.processing)
            processedAudio = audioData.audio
            audioTokenCounts = audioData.tokenCounts
        }

        if !imageTokenCounts.isEmpty || !audioTokenCounts.isEmpty {
            var imageIndex = 0
            var audioIndex = 0
            var expandedTokens: [Int] = []
            expandedTokens.reserveCapacity(
                promptTokens.count + imageTokenCounts.reduce(0, +)
                    + audioTokenCounts.reduce(0, +),
            )

            for token in promptTokens {
                if token == config.base.imageTokenId, !imageTokenCounts.isEmpty {
                    let count =
                        imageIndex < imageTokenCounts.count
                        ? imageTokenCounts[imageIndex]
                        : config.base.imageSeqLength
                    expandedTokens.append(config.base.boiTokenId)
                    expandedTokens.append(
                        contentsOf: Array(repeating: config.base.imageTokenId, count: count),
                    )
                    if let eoiTokenId = config.base.eoiTokenId {
                        expandedTokens.append(eoiTokenId)
                    }
                    imageIndex += 1
                } else if token == config.base.audioTokenId, !audioTokenCounts.isEmpty {
                    let count =
                        audioIndex < audioTokenCounts.count
                        ? audioTokenCounts[audioIndex]
                        : config.base.audioSeqLength
                    expandedTokens.append(config.boaTokenId)
                    expandedTokens.append(
                        contentsOf: Array(repeating: config.base.audioTokenId, count: count),
                    )
                    if let eoaTokenId = config.eoaTokenId {
                        expandedTokens.append(eoaTokenId)
                    }
                    audioIndex += 1
                } else {
                    expandedTokens.append(token)
                }
            }

            guard imageIndex == imageTokenCounts.count else {
                throw VLMError.processing("Image prompt tokens do not match image inputs.")
            }
            guard audioIndex == audioTokenCounts.count else {
                throw VLMError.processing("Audio prompt tokens do not match audio inputs.")
            }
            promptTokens = expandedTokens
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage,
            audio: processedAudio,
        )
    }
}
