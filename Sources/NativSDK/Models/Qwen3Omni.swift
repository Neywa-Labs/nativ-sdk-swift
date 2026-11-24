import CoreImage
import Foundation
import MLX
import MLXNN

private enum Qwen3OmniMoEError: Error {
    case featureTokenMismatch(expected: Int, actual: Int)
    case audioRequiresAudioTower
    case audioHiddenSizeMismatch(expected: Int, actual: Int)
}

private enum Qwen3OmniSupport {
    static func replacePaddingTokens(
        in promptTokens: [Int],
        frames: [THW],
        paddingToken: String,
        mergeSize: Int,
        tokenizer: any Tokenizer,
    )
        throws -> [Int]
    {
        let placeholderTokens = tokenizer.encode(
            text: "<|vision_start|>\(paddingToken)<|vision_end|>",
            addSpecialTokens: false,
        )
        let placeholderRanges = promptTokens.ranges(of: placeholderTokens)
        guard placeholderRanges.count == frames.count else {
            throw VLMError.processing(
                "Number of placeholder tokens does not match number of frames",
            )
        }

        let mergeLength = mergeSize * mergeSize
        let replacementSequences = frames.map { frame in
            let paddingCount = frame.product / mergeLength
            return tokenizer.encode(
                text:
                    "<|vision_start|>\(Array(repeating: paddingToken, count: paddingCount).joined())<|vision_end|>",
                addSpecialTokens: false,
            )
        }

        var result: [Int] = []
        var currentIndex = promptTokens.startIndex
        for (range, replacement) in zip(placeholderRanges, replacementSequences) {
            result.append(contentsOf: promptTokens[currentIndex ..< range.lowerBound])
            result.append(contentsOf: replacement)
            currentIndex = range.upperBound
        }
        if currentIndex < promptTokens.endIndex {
            result.append(contentsOf: promptTokens[currentIndex...])
        }
        return result
    }

    static func replaceAudioPaddingTokens(
        in promptTokens: [Int],
        tokenCounts: [Int],
        paddingToken: String,
        tokenizer: any Tokenizer,
    )
        throws -> [Int]
    {
        let placeholderTokens = tokenizer.encode(text: paddingToken, addSpecialTokens: false)
        let placeholderRanges = promptTokens.ranges(of: placeholderTokens)
        guard placeholderRanges.count == tokenCounts.count else {
            throw VLMError.processing(
                "Number of audio placeholder tokens does not match number of audio inputs",
            )
        }

        let replacementSequences = tokenCounts.map { count in
            tokenizer.encode(
                text: String(repeating: paddingToken, count: max(count, 1)),
                addSpecialTokens: false,
            )
        }

        var result: [Int] = []
        var currentIndex = promptTokens.startIndex
        for (range, replacement) in zip(placeholderRanges, replacementSequences) {
            result.append(contentsOf: promptTokens[currentIndex ..< range.lowerBound])
            result.append(contentsOf: replacement)
            currentIndex = range.upperBound
        }
        if currentIndex < promptTokens.endIndex {
            result.append(contentsOf: promptTokens[currentIndex...])
        }
        return result
    }

    static func audioFeatureOutputLength(_ inputLength: Int) -> Int {
        let fullWindows = max(inputLength, 0) / 100
        let remainder = max(inputLength, 0) % 100

        func strideTwoConvLength(_ value: Int) -> Int {
            value <= 0 ? 0 : (value + 1) / 2
        }

        let tail =
            strideTwoConvLength(strideTwoConvLength(strideTwoConvLength(remainder)))
        return fullWindows * 13 + tail
    }

}

// MARK: - Processor

public struct Qwen3OmniMoEProcessor: UserInputProcessor {
    private let config: Qwen3OmniMoEProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let audioFeatureExtractor: Qwen3OmniMoEAudioFeatureExtractor

    public init(_ config: Qwen3OmniMoEProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        audioFeatureExtractor = Qwen3OmniMoEAudioFeatureExtractor(config.audioFeatureExtractor)
    }

    private func preprocess(image: CIImage, resizedSize: CGSize) -> CIImage {
        image
            .toSRGB()
            .resampled(to: resizedSize, method: .bicubic)
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW,
    ) {
        let processed = images.map { MediaProcessing.apply($0, processing: processing) }

        guard let first = processed.first else {
            throw VLMError.imageProcessingFailure("No image provided")
        }

        let extent = first.extent.size
        let (resizedHeight, resizedWidth) = try QwenVL.targetSize(
            height: Int(extent.height),
            width: Int(extent.width),
            factor: config.patchSize * config.mergeSize,
            minPixels: config.size.minPixels,
            maxPixels: config.size.maxPixels,
        )

        let targetSize = CGSize(width: resizedWidth, height: resizedHeight)

        let resampled = processed.map { MediaProcessing.resampleBicubic($0, to: targetSize) }

        let normalized =
            resampled
            .map {
                MediaProcessing.normalize(
                    $0,
                    mean: config.imageMeanTuple,
                    std: config.imageStdTuple,
                )
            }
            .map { MediaProcessing.asMLXArray($0) }

        return try QwenVL.patchify(
            images: normalized,
            mergeSize: config.mergeSize,
            patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize,
        )
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Qwen3OmniMoEMessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext,
        )

        if input.images.isEmpty, input.videos.isEmpty, input.audios.isEmpty {
            let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: promptArray).asType(.int8)
            return LMInput(text: .init(tokens: promptArray, mask: mask))
        }

        var processedAudio: LMInput.ProcessedAudio?
        if !input.audios.isEmpty {
            let audioData = try await preprocess(audios: input.audios, processing: input.processing)
            processedAudio = audioData.audio
            promptTokens = try Qwen3OmniSupport.replaceAudioPaddingTokens(
                in: promptTokens,
                tokenCounts: audioData.tokenCounts,
                paddingToken: config.audioToken,
                tokenizer: tokenizer,
            )
        }

        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imageFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            let concatenated = concatenated(imageFrames.map(\.0))
            processedImage = .init(pixels: concatenated, frames: imageFrames.map(\.1))

            if let frames = processedImage?.frames {
                promptTokens = try Qwen3OmniSupport.replacePaddingTokens(
                    in: promptTokens,
                    frames: frames,
                    paddingToken: "<|image_pad|>",
                    mergeSize: config.mergeSize,
                    tokenizer: tokenizer,
                )
            }
        }

        var processedVideo: LMInput.ProcessedVideo?
        if !input.videos.isEmpty {
            var accumulatedFrames: [[MLXArray]] = []

            for video in input.videos {
                var resizedSize: CGSize = .zero
                let sequence = try await MediaProcessing.asProcessedSequence(
                    video, targetFPS: { _ in Double(2) },
                ) { frame in
                    let processed = try MediaProcessing.apply(
                        frame.image.asCIImage(), processing: input.processing)
                    if resizedSize == .zero {
                        let size = processed.extent.size
                        let (height, width) = try QwenVL.targetSize(
                            height: Int(size.height),
                            width: Int(size.width),
                            factor: config.patchSize * config.mergeSize,
                            minPixels: config.minPixels,
                            maxPixels: config.maxPixels,
                        )
                        resizedSize = CGSize(width: width, height: height)
                    }
                    let finalImage = preprocess(image: processed, resizedSize: resizedSize)
                    return UserInput.VideoFrame(
                        image: .ciImage(finalImage), timeStamp: frame.timeStamp)
                }
                accumulatedFrames.append(sequence.frames)
            }

            let videoFrames = try accumulatedFrames.map {
                try QwenVL.patchify(
                    images: $0,
                    mergeSize: config.mergeSize,
                    patchSize: config.patchSize,
                    temporalPatchSize: config.temporalPatchSize,
                )
            }

            let concatenated = concatenated(videoFrames.map(\.0))
            processedVideo = .init(pixels: concatenated, frames: videoFrames.map(\.1))

            if let frames = processedVideo?.frames {
                promptTokens = try Qwen3OmniSupport.replacePaddingTokens(
                    in: promptTokens,
                    frames: frames,
                    paddingToken: "<|video_pad|>",
                    mergeSize: config.mergeSize,
                    tokenizer: tokenizer,
                )
            }
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)

        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage,
            video: processedVideo,
            audio: processedAudio,
        )
    }

    private func preprocess(
        audios: [UserInput.Audio],
        processing: UserInput.Processing,
    )
        async throws -> (audio: LMInput.ProcessedAudio, tokenCounts: [Int])
    {
        var audioProcessing = processing.audio
        audioProcessing.sampleRate = Double(config.audioFeatureExtractor.samplingRate)
        audioProcessing.channels = 1

        var samples: [MLXArray] = []
        samples.reserveCapacity(audios.count)
        for audio in audios {
            try await samples.append(audio.asMLXArray(processing: audioProcessing))
        }

        let extracted = try audioFeatureExtractor(samples)
        let tokenCounts = extracted.featureLengths.map(Qwen3OmniSupport.audioFeatureOutputLength)
        return (
            audio: .init(features: extracted.inputFeatures, mask: extracted.inputFeaturesMask),
            tokenCounts: tokenCounts,
        )
    }
}

public struct Qwen3OmniMoEAudioFeatureExtractorConfiguration: Decodable, Sendable {
    public let featureSize: Int
    public let samplingRate: Int
    public let hopLength: Int
    public let fftLength: Int
    public let chunkLength: Float
    public let maxSamples: Int?
    public let paddingValue: Float

    enum CodingKeys: String, CodingKey {
        case featureSize = "feature_size"
        case samplingRate = "sampling_rate"
        case hopLength = "hop_length"
        case fftLength = "n_fft"
        case chunkLength = "chunk_length"
        case maxSamples = "n_samples"
        case paddingValue = "padding_value"
    }

    public init(
        featureSize: Int = 128,
        samplingRate: Int = 16000,
        hopLength: Int = 160,
        fftLength: Int = 400,
        chunkLength: Float = 30,
        maxSamples: Int? = 480_000,
        paddingValue: Float = 0,
    ) {
        self.featureSize = featureSize
        self.samplingRate = samplingRate
        self.hopLength = hopLength
        self.fftLength = fftLength
        self.chunkLength = chunkLength
        self.maxSamples = maxSamples
        self.paddingValue = paddingValue
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        featureSize = try c.decodeIfPresent(Int.self, forKey: .featureSize) ?? 128
        samplingRate = try c.decodeIfPresent(Int.self, forKey: .samplingRate) ?? 16000
        hopLength = try c.decodeIfPresent(Int.self, forKey: .hopLength) ?? 160
        fftLength = try c.decodeIfPresent(Int.self, forKey: .fftLength) ?? 400
        chunkLength = try c.decodeIfPresent(Float.self, forKey: .chunkLength) ?? 30
        maxSamples = try c.decodeIfPresent(Int.self, forKey: .maxSamples) ?? 480_000
        paddingValue = try c.decodeIfPresent(Float.self, forKey: .paddingValue) ?? 0
    }
}

private struct Qwen3OmniMoEAudioFeatureExtractor: Sendable {
    let configuration: Qwen3OmniMoEAudioFeatureExtractorConfiguration
    let window: MaterializedArray
    let melFilters: MaterializedArray

    init(_ configuration: Qwen3OmniMoEAudioFeatureExtractorConfiguration) {
        self.configuration = configuration
        let window = MLXArray(
            (0 ..< configuration.fftLength).map { index in
                Float(
                    0.5 - 0.5
                        * cos(2.0 * Double.pi * Double(index) / Double(configuration.fftLength)),
                )
            },
            [configuration.fftLength],
        )
        let melFilters = Self.makeMelFilterBank(
            frequencyBins: configuration.fftLength / 2 + 1,
            melFilters: configuration.featureSize,
            minFrequency: 0,
            maxFrequency: Float(configuration.samplingRate) / 2,
            samplingRate: configuration.samplingRate,
        )
        let tables = materialize([window, melFilters])
        self.window = tables[0]
        self.melFilters = tables[1]
    }

    func callAsFunction(_ rawAudio: [MLXArray]) throws -> (
        inputFeatures: MLXArray,
        inputFeaturesMask: MLXArray,
        featureLengths: [Int],
    ) {
        guard !rawAudio.isEmpty else {
            throw VLMError.processing("No audio samples were provided.")
        }

        let configuredMaxSamples =
            configuration.maxSamples
            ?? max(1, Int(Float(configuration.samplingRate) * configuration.chunkLength))
        let maxSamples = max(1, configuredMaxSamples)
        var spectrograms: [MLXArray] = []
        var masks: [MLXArray] = []
        var featureLengths: [Int] = []

        spectrograms.reserveCapacity(rawAudio.count)
        masks.reserveCapacity(rawAudio.count)
        featureLengths.reserveCapacity(rawAudio.count)

        for audio in rawAudio {
            let samples = Gemma4AudioFeatureExtractor.monoFloatSamples(audio)
            let validSamples = min(max(samples.dim(0), 1), maxSamples)
            let waveform =
                samples.dim(0) > validSamples
                ? samples[..<validSamples]
                : samples
            let featureLength = min(
                max(1, Int(ceil(Double(validSamples) / Double(configuration.hopLength)))),
                max(1, maxSamples / configuration.hopLength),
            )
            featureLengths.append(featureLength)
            spectrograms.append(extractLogMel(waveform, frameCount: featureLength))
        }

        let maxFrames = featureLengths.max() ?? 1
        for index in spectrograms.indices {
            let pad = max(0, maxFrames - spectrograms[index].dim(0))
            if pad > 0 {
                spectrograms[index] = padded(
                    spectrograms[index],
                    widths: [.init((0, pad)), 0],
                    value: MLXArray(configuration.paddingValue, dtype: spectrograms[index].dtype),
                )
            }
            masks.append(
                MLXArray(
                    Array(repeating: true, count: featureLengths[index])
                        + Array(repeating: false, count: pad),
                    [maxFrames],
                ),
            )
        }

        return (stacked(spectrograms, axis: 0), stacked(masks, axis: 0), featureLengths)
    }

    private func extractLogMel(_ waveform: MLXArray, frameCount: Int) -> MLXArray {
        var rows: [MLXArray] = []
        rows.reserveCapacity(frameCount)
        for frameIndex in 0 ..< frameCount {
            let start = frameIndex * configuration.hopLength
            let available = max(0, min(configuration.fftLength, waveform.dim(0) - start))
            var frame =
                available > 0
                ? waveform[start ..< (start + available)]
                : MLXArray.zeros([0], dtype: .float32)
            if available < configuration.fftLength {
                frame = padded(
                    frame,
                    widths: [.init((0, configuration.fftLength - available))],
                    value: MLXArray(configuration.paddingValue, dtype: waveform.dtype),
                )
            }
            rows.append(frame)
        }

        let frames = stacked(rows, axis: 0) * window.asType(waveform.dtype)
        let spectrum = MLXFFT.rfft(frames, n: configuration.fftLength, axis: -1)
        let power = spectrum.abs().square()
        var mel = power.matmul(melFilters.asType(power.dtype))
        mel = maximum(mel, MLXArray(Float32(1e-10), dtype: mel.dtype))
        var logMel = log10(mel)
        let floor = logMel.max() - MLXArray(Float32(8), dtype: logMel.dtype)
        logMel = maximum(logMel, floor)
        return
            ((logMel + MLXArray(Float32(4), dtype: logMel.dtype))
            / MLXArray(Float32(4), dtype: logMel.dtype)).asType(.float32)
    }

    private static func hertzToMel(_ frequency: Float) -> Float {
        let minLogHertz: Float = 1000
        let minLogMel: Float = 15
        let logStep = log(Float(6.4)) / 27
        if frequency < minLogHertz {
            return frequency / (200 / 3)
        }
        return minLogMel + log(frequency / minLogHertz) / logStep
    }

    private static func melToHertz(_ mel: Float) -> Float {
        let minLogHertz: Float = 1000
        let minLogMel: Float = 15
        let logStep = log(Float(6.4)) / 27
        if mel < minLogMel {
            return mel * (200 / 3)
        }
        return minLogHertz * exp(logStep * (mel - minLogMel))
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

public struct Qwen3OmniMoEProcessorConfiguration: Decodable, Sendable {
    public struct Size: Sendable {
        public let maxPixels: Int
        public let minPixels: Int
    }

    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    private let _minPixels: Int?
    private let _maxPixels: Int?
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let imageProcessorType: String
    public let audioToken: String
    public let audioStartToken: String
    public let audioEndToken: String
    public let audioFeatureExtractor: Qwen3OmniMoEAudioFeatureExtractorConfiguration

    public var minPixels: Int {
        _minPixels ?? 4 * 28 * 28
    }  // 3,136
    public var maxPixels: Int {
        _maxPixels ?? 16384 * 28 * 28
    }  // 12,845,056

    public var size: Size {
        .init(maxPixels: maxPixels, minPixels: minPixels)
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }

    enum CodingKeys: String, CodingKey {
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case _minPixels = "min_pixels"
        case _maxPixels = "max_pixels"
        case mergeSize = "merge_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case imageProcessorType = "image_processor_type"
        case imageProcessor = "image_processor"
        case featureExtractor = "feature_extractor"
        case audioToken = "audio_token"
        case audioStartToken = "audio_bos_token"
        case audioEndToken = "audio_eos_token"
        case processorClass = "processor_class"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let source =
            (try? container.nestedContainer(keyedBy: CodingKeys.self, forKey: .imageProcessor))
            ?? container

        imageMean =
            try source.decodeIfPresent([CGFloat].self, forKey: .imageMean)
            ?? [0.5, 0.5, 0.5]
        imageStd =
            try source.decodeIfPresent([CGFloat].self, forKey: .imageStd)
            ?? [0.5, 0.5, 0.5]
        _minPixels = try source.decodeIfPresent(Int.self, forKey: ._minPixels)
        _maxPixels = try source.decodeIfPresent(Int.self, forKey: ._maxPixels)
        mergeSize = try source.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
        patchSize = try source.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        temporalPatchSize =
            try source.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
        imageProcessorType =
            try source.decodeIfPresent(String.self, forKey: .imageProcessorType)
            ?? "Qwen2VLImageProcessor"
        audioToken =
            try container.decodeIfPresent(String.self, forKey: .audioToken)
            ?? "<|audio_pad|>"
        audioStartToken =
            try container.decodeIfPresent(String.self, forKey: .audioStartToken)
            ?? "<|audio_start|>"
        audioEndToken =
            try container.decodeIfPresent(String.self, forKey: .audioEndToken)
            ?? "<|audio_end|>"
        audioFeatureExtractor =
            try container.decodeIfPresent(
                Qwen3OmniMoEAudioFeatureExtractorConfiguration.self,
                forKey: .featureExtractor,
            ) ?? .init()
    }
}

// MARK: - Model Configuration

public struct Qwen3OmniMoEConfiguration: Decodable, Sendable {
    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        private let _numKeyValueHeads: Int?
        public var numKeyValueHeads: Int {
            _numKeyValueHeads ?? numAttentionHeads
        }

        public let headDim: Int
        private let _ropeTheta: Double?
        public var ropeTheta: Double {
            _ropeTheta ?? 1_000_000
        }

        public let maxPositionEmbeddings: Int
        private let _rmsNormEps: Double?
        public var rmsNormEps: Double {
            _rmsNormEps ?? 1e-6
        }

        private let _ropeScaling: RoPEScaling?
        public var ropeScaling: RoPEScaling? {
            _ropeScaling
        }

        private let _normTopKProb: Bool?
        public var normTopKProb: Bool {
            _normTopKProb ?? true
        }

        private let _tieWordEmbeddings: Bool?
        public var tieWordEmbeddings: Bool {
            _tieWordEmbeddings ?? false
        }

        private let _attentionBias: Bool?
        public var attentionBias: Bool {
            _attentionBias ?? false
        }

        private let _hiddenAct: String?
        public var hiddenAct: String {
            _hiddenAct ?? "silu"
        }

        private let _numExperts: Int?
        public var numExperts: Int {
            _numExperts ?? 0
        }

        private let _numExpertsPerToken: Int?
        public var numExpertsPerToken: Int {
            _numExpertsPerToken ?? 1
        }

        private let _decoderSparseStep: Int?
        public var decoderSparseStep: Int {
            _decoderSparseStep ?? 1
        }

        private let _mlpOnlyLayers: [Int]?
        public var mlpOnlyLayers: [Int] {
            _mlpOnlyLayers ?? []
        }

        private let _moeIntermediateSize: Int?
        public var moeIntermediateSize: Int {
            _moeIntermediateSize ?? intermediateSize
        }

        private let _sharedExpertIntermediateSize: Int?
        public var sharedExpertIntermediateSize: Int {
            _sharedExpertIntermediateSize ?? intermediateSize
        }

        public let vocabSize: Int

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case _numKeyValueHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case _ropeTheta = "rope_theta"
            case maxPositionEmbeddings = "max_position_embeddings"
            case _rmsNormEps = "rms_norm_eps"
            case _ropeScaling = "rope_scaling"
            case _normTopKProb = "norm_topk_prob"
            case _tieWordEmbeddings = "tie_word_embeddings"
            case _attentionBias = "attention_bias"
            case _hiddenAct = "hidden_act"
            case _numExperts = "num_experts"
            case _numExpertsPerToken = "num_experts_per_tok"
            case _decoderSparseStep = "decoder_sparse_step"
            case _mlpOnlyLayers = "mlp_only_layers"
            case _moeIntermediateSize = "moe_intermediate_size"
            case _sharedExpertIntermediateSize = "shared_expert_intermediate_size"
            case vocabSize = "vocab_size"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "qwen3_omni_moe_thinker_text"
            hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
            intermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? hiddenSize * 4
            numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
            numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
            _numKeyValueHeads =
                try container.decodeIfPresent(Int.self, forKey: ._numKeyValueHeads)
            headDim =
                try container.decodeIfPresent(Int.self, forKey: .headDim)
                ?? hiddenSize / numAttentionHeads
            _ropeTheta = try container.decodeIfPresent(Double.self, forKey: ._ropeTheta)
            maxPositionEmbeddings =
                try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
            _rmsNormEps = try container.decodeIfPresent(Double.self, forKey: ._rmsNormEps)
            _ropeScaling =
                try container.decodeIfPresent(RoPEScaling.self, forKey: ._ropeScaling)
            _normTopKProb =
                try container.decodeIfPresent(Bool.self, forKey: ._normTopKProb)
            _tieWordEmbeddings =
                try container.decodeIfPresent(Bool.self, forKey: ._tieWordEmbeddings)
            _attentionBias =
                try container.decodeIfPresent(Bool.self, forKey: ._attentionBias)
            _hiddenAct = try container.decodeIfPresent(String.self, forKey: ._hiddenAct)
            _numExperts = try container.decodeIfPresent(Int.self, forKey: ._numExperts)
            _numExpertsPerToken =
                try container.decodeIfPresent(Int.self, forKey: ._numExpertsPerToken)
            _decoderSparseStep =
                try container.decodeIfPresent(Int.self, forKey: ._decoderSparseStep)
            _mlpOnlyLayers =
                try container.decodeIfPresent([Int].self, forKey: ._mlpOnlyLayers)
            _moeIntermediateSize =
                try container.decodeIfPresent(Int.self, forKey: ._moeIntermediateSize)
            _sharedExpertIntermediateSize =
                try container.decodeIfPresent(Int.self, forKey: ._sharedExpertIntermediateSize)
            vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let depth: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let outHiddenSize: Int
        public let numHeads: Int
        public let patchSize: Int
        public let spatialMergeSize: Int
        public let temporalPatchSize: Int
        public let numPositionEmbeddings: Int
        private let _inChannels: Int?
        public var inChannels: Int {
            _inChannels ?? 3
        }

        private let _hiddenAct: String?
        public var hiddenAct: String {
            _hiddenAct ?? "gelu"
        }

        private let _deepstackVisualIndexes: [Int]?
        public var deepstackVisualIndexes: [Int] {
            _deepstackVisualIndexes ?? []
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case depth
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case outHiddenSize = "out_hidden_size"
            case numHeads = "num_heads"
            case patchSize = "patch_size"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
            case numPositionEmbeddings = "num_position_embeddings"
            case _inChannels = "in_channels"
            case _hiddenAct = "hidden_act"
            case _deepstackVisualIndexes = "deepstack_visual_indexes"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try container.decodeIfPresent(String.self, forKey: .modelType)
                ?? "qwen3_omni_moe_thinker_vision"
            depth = try container.decode(Int.self, forKey: .depth)
            hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
            intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
            outHiddenSize = try container.decode(Int.self, forKey: .outHiddenSize)
            numHeads = try container.decode(Int.self, forKey: .numHeads)
            patchSize = try container.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
            spatialMergeSize =
                try container.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
            temporalPatchSize =
                try container.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
            numPositionEmbeddings =
                try container.decodeIfPresent(Int.self, forKey: .numPositionEmbeddings) ?? 2304
            _inChannels = try container.decodeIfPresent(Int.self, forKey: ._inChannels)
            _hiddenAct = try container.decodeIfPresent(String.self, forKey: ._hiddenAct)
            _deepstackVisualIndexes =
                try container.decodeIfPresent([Int].self, forKey: ._deepstackVisualIndexes)
        }
    }

    public struct AudioConfiguration: Decodable, Sendable {
        public let modelType: String
        public let modelDimensions: Int
        public let encoderLayers: Int
        public let encoderAttentionHeads: Int
        public let encoderFFNDimensions: Int
        public let melBins: Int
        public let outputDimensions: Int
        public let downsampleHiddenSize: Int
        public let activationFunction: String
        public let dropout: Float
        public let scaleEmbedding: Bool
        public let convChunkSize: Int
        public let windowSize: Int
        public let inferenceWindowSize: Int
        public let maxSourcePositions: Int

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case modelDimensions = "d_model"
            case encoderLayers = "encoder_layers"
            case encoderAttentionHeads = "encoder_attention_heads"
            case encoderFFNDimensions = "encoder_ffn_dim"
            case melBins = "num_mel_bins"
            case outputDimensions = "output_dim"
            case downsampleHiddenSize = "downsample_hidden_size"
            case activationFunction = "activation_function"
            case dropout
            case scaleEmbedding = "scale_embedding"
            case convChunkSize = "conv_chunksize"
            case windowSize = "n_window"
            case inferenceWindowSize = "n_window_infer"
            case maxSourcePositions = "max_source_positions"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType)
                ?? "qwen3_omni_moe_audio_encoder"
            modelDimensions = try c.decodeIfPresent(Int.self, forKey: .modelDimensions) ?? 1280
            encoderLayers = try c.decodeIfPresent(Int.self, forKey: .encoderLayers) ?? 32
            encoderAttentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .encoderAttentionHeads) ?? 20
            encoderFFNDimensions =
                try c.decodeIfPresent(Int.self, forKey: .encoderFFNDimensions) ?? 5120
            melBins = try c.decodeIfPresent(Int.self, forKey: .melBins) ?? 128
            outputDimensions =
                try c.decodeIfPresent(Int.self, forKey: .outputDimensions) ?? 2048
            downsampleHiddenSize =
                try c.decodeIfPresent(Int.self, forKey: .downsampleHiddenSize) ?? 480
            activationFunction =
                try c.decodeIfPresent(String.self, forKey: .activationFunction) ?? "gelu"
            dropout = try c.decodeIfPresent(Float.self, forKey: .dropout) ?? 0
            scaleEmbedding = try c.decodeIfPresent(Bool.self, forKey: .scaleEmbedding) ?? false
            convChunkSize = try c.decodeIfPresent(Int.self, forKey: .convChunkSize) ?? 500
            windowSize = try c.decodeIfPresent(Int.self, forKey: .windowSize) ?? 50
            inferenceWindowSize =
                try c.decodeIfPresent(Int.self, forKey: .inferenceWindowSize) ?? 800
            maxSourcePositions =
                try c.decodeIfPresent(Int.self, forKey: .maxSourcePositions) ?? 1500
        }
    }

    public struct Code2WavConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numHiddenLayers: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let decoderDimension: Int
        public let hiddenActivation: String
        public let attentionBias: Bool
        public let attentionDropout: Float
        public let rmsNormEps: Float
        public let ropeTheta: Float
        public let maxPositionEmbeddings: Int
        public let slidingWindow: Int?
        public let codebookDimension: Int
        public let codebookSize: Int
        public let numQuantizers: Int
        public let numSemanticQuantizers: Int
        public let semanticCodebookSize: Int
        public let vectorQuantizationHiddenDimension: Int
        public let layerScaleInitialScale: Float
        public let upsampleRates: [Int]
        public let upsamplingRatios: [Int]

        public var headDimension: Int {
            hiddenSize / max(numAttentionHeads, 1)
        }

        public var totalUpsampleFactor: Int {
            (upsampleRates + upsamplingRatios).reduce(1) { $0 * max($1, 1) }
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numHiddenLayers = "num_hidden_layers"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case decoderDimension = "decoder_dim"
            case hiddenActivation = "hidden_act"
            case attentionBias = "attention_bias"
            case attentionDropout = "attention_dropout"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case maxPositionEmbeddings = "max_position_embeddings"
            case slidingWindow = "sliding_window"
            case codebookDimension = "codebook_dim"
            case codebookSize = "codebook_size"
            case numQuantizers = "num_quantizers"
            case numSemanticQuantizers = "num_semantic_quantizers"
            case semanticCodebookSize = "semantic_codebook_size"
            case vectorQuantizationHiddenDimension = "vector_quantization_hidden_dimension"
            case layerScaleInitialScale = "layer_scale_initial_scale"
            case upsampleRates = "upsample_rates"
            case upsamplingRatios = "upsampling_ratios"
        }

        public init(
            modelType: String = "",
            hiddenSize: Int = 1024,
            intermediateSize: Int = 3072,
            numHiddenLayers: Int = 8,
            numAttentionHeads: Int = 16,
            numKeyValueHeads: Int = 16,
            decoderDimension: Int = 1536,
            hiddenActivation: String = "silu",
            attentionBias: Bool = false,
            attentionDropout: Float = 0,
            rmsNormEps: Float = 1e-5,
            ropeTheta: Float = 10000,
            maxPositionEmbeddings: Int = 8000,
            slidingWindow: Int? = 72,
            codebookDimension: Int = 512,
            codebookSize: Int = 2048,
            numQuantizers: Int = 16,
            numSemanticQuantizers: Int = 1,
            semanticCodebookSize: Int = 4096,
            vectorQuantizationHiddenDimension: Int = 512,
            layerScaleInitialScale: Float = 0.01,
            upsampleRates: [Int] = [8, 5, 4, 3],
            upsamplingRatios: [Int] = [2, 2],
        ) {
            self.modelType = modelType
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.numHiddenLayers = numHiddenLayers
            self.numAttentionHeads = numAttentionHeads
            self.numKeyValueHeads = numKeyValueHeads
            self.decoderDimension = decoderDimension
            self.hiddenActivation = hiddenActivation
            self.attentionBias = attentionBias
            self.attentionDropout = attentionDropout
            self.rmsNormEps = rmsNormEps
            self.ropeTheta = ropeTheta
            self.maxPositionEmbeddings = maxPositionEmbeddings
            self.slidingWindow = slidingWindow
            self.codebookDimension = codebookDimension
            self.codebookSize = codebookSize
            self.numQuantizers = numQuantizers
            self.numSemanticQuantizers = numSemanticQuantizers
            self.semanticCodebookSize = semanticCodebookSize
            self.vectorQuantizationHiddenDimension = vectorQuantizationHiddenDimension
            self.layerScaleInitialScale = layerScaleInitialScale
            self.upsampleRates = upsampleRates
            self.upsamplingRatios = upsamplingRatios
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? ""
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
            numHiddenLayers =
                try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 8
            numAttentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
            numKeyValueHeads =
                try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 16
            decoderDimension =
                try c.decodeIfPresent(Int.self, forKey: .decoderDimension) ?? 1536
            hiddenActivation =
                try c.decodeIfPresent(String.self, forKey: .hiddenActivation) ?? "silu"
            attentionBias =
                try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            attentionDropout =
                try c.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0
            rmsNormEps =
                try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8000
            slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
            codebookDimension =
                try c.decodeIfPresent(Int.self, forKey: .codebookDimension) ?? 512
            codebookSize = try c.decodeIfPresent(Int.self, forKey: .codebookSize) ?? 2048
            numQuantizers =
                try c.decodeIfPresent(Int.self, forKey: .numQuantizers) ?? 16
            numSemanticQuantizers =
                try c.decodeIfPresent(Int.self, forKey: .numSemanticQuantizers) ?? 1
            semanticCodebookSize =
                try c.decodeIfPresent(Int.self, forKey: .semanticCodebookSize) ?? 4096
            vectorQuantizationHiddenDimension =
                try c.decodeIfPresent(
                    Int.self,
                    forKey: .vectorQuantizationHiddenDimension,
                ) ?? 512
            layerScaleInitialScale =
                try c.decodeIfPresent(Float.self, forKey: .layerScaleInitialScale) ?? 0.01
            upsampleRates =
                try c.decodeIfPresent([Int].self, forKey: .upsampleRates) ?? [8, 5, 4, 3]
            upsamplingRatios =
                try c.decodeIfPresent([Int].self, forKey: .upsamplingRatios) ?? [2, 2]
        }
    }

    public struct CodePredictorConfiguration: Decodable, Sendable {
        public let modelType: String
        public let numHiddenLayers: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let headDim: Int
        public let hiddenActivation: String
        public let rmsNormEps: Float
        public let ropeTheta: Float
        public let maxPositionEmbeddings: Int
        public let vocabSize: Int
        public let numCodeGroups: Int
        public let maxWindowLayers: Int
        public let attentionBias: Bool
        public let attentionDropout: Float
        public let initializerRange: Float
        public let useCache: Bool
        public let layerTypes: [String]
        public let useSlidingWindow: Bool
        public let slidingWindow: Int?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case numHiddenLayers = "num_hidden_layers"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case hiddenActivation = "hidden_act"
            case rmsNormEps = "rms_norm_eps"
            case ropeTheta = "rope_theta"
            case maxPositionEmbeddings = "max_position_embeddings"
            case vocabSize = "vocab_size"
            case numCodeGroups = "num_code_groups"
            case maxWindowLayers = "max_window_layers"
            case attentionBias = "attention_bias"
            case attentionDropout = "attention_dropout"
            case initializerRange = "initializer_range"
            case useCache = "use_cache"
            case layerTypes = "layer_types"
            case useSlidingWindow = "use_sliding_window"
            case slidingWindow = "sliding_window"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType)
                ?? "qwen3_omni_moe_talker_code_predictor"
            numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 5
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3072
            numAttentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
            numKeyValueHeads =
                try c.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 8
            headDim =
                try c.decodeIfPresent(Int.self, forKey: .headDim)
                ?? hiddenSize / max(numAttentionHeads, 1)
            hiddenActivation =
                try c.decodeIfPresent(String.self, forKey: .hiddenActivation) ?? "silu"
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
            vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 2048
            numCodeGroups = try c.decodeIfPresent(Int.self, forKey: .numCodeGroups) ?? 16
            maxWindowLayers = try c.decodeIfPresent(Int.self, forKey: .maxWindowLayers) ?? 28
            attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            attentionDropout =
                try c.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0
            initializerRange =
                try c.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
            useCache = try c.decodeIfPresent(Bool.self, forKey: .useCache) ?? true
            layerTypes =
                try c.decodeIfPresent([String].self, forKey: .layerTypes)
                ?? Array(repeating: "full_attention", count: numHiddenLayers)
            useSlidingWindow =
                try c.decodeIfPresent(Bool.self, forKey: .useSlidingWindow) ?? false
            slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
        }
    }

    public struct TalkerConfiguration: Decodable, Sendable {
        public let textConfiguration: TextConfiguration
        public let codePredictorConfiguration: CodePredictorConfiguration
        public let modelType: String
        public let acceptHiddenLayer: Int
        public let imageTokenID: Int
        public let videoTokenID: Int
        public let audioTokenID: Int
        public let audioStartTokenID: Int
        public let audioEndTokenID: Int
        public let visionStartTokenID: Int
        public let numCodeGroups: Int
        public let outputRouterLogits: Bool
        public let positionIDPerSeconds: Int
        public let secondsPerChunk: Int
        public let spatialMergeSize: Int
        public let thinkerHiddenSize: Int
        public let codecBOSID: Int
        public let codecEOSTokenID: Int
        public let codecNoThinkID: Int
        public let codecPadID: Int
        public let codecThinkBOSID: Int
        public let codecThinkEOSID: Int
        public let speakerID: [String: Int]

        enum CodingKeys: String, CodingKey {
            case textConfiguration = "text_config"
            case codePredictorConfiguration = "code_predictor_config"
            case modelType = "model_type"
            case acceptHiddenLayer = "accept_hidden_layer"
            case imageTokenID = "image_token_id"
            case videoTokenID = "video_token_id"
            case audioTokenID = "audio_token_id"
            case audioStartTokenID = "audio_start_token_id"
            case audioEndTokenID = "audio_end_token_id"
            case visionStartTokenID = "vision_start_token_id"
            case numCodeGroups = "num_code_groups"
            case outputRouterLogits = "output_router_logits"
            case positionIDPerSeconds = "position_id_per_seconds"
            case secondsPerChunk = "seconds_per_chunk"
            case spatialMergeSize = "spatial_merge_size"
            case thinkerHiddenSize = "thinker_hidden_size"
            case codecBOSID = "codec_bos_id"
            case codecEOSTokenID = "codec_eos_token_id"
            case codecNoThinkID = "codec_nothink_id"
            case codecPadID = "codec_pad_id"
            case codecThinkBOSID = "codec_think_bos_id"
            case codecThinkEOSID = "codec_think_eos_id"
            case speakerID = "speaker_id"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            textConfiguration = try c.decode(TextConfiguration.self, forKey: .textConfiguration)
            codePredictorConfiguration =
                try c.decode(
                    CodePredictorConfiguration.self,
                    forKey: .codePredictorConfiguration,
                )
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType)
                ?? "qwen3_omni_moe_talker"
            acceptHiddenLayer = try c.decodeIfPresent(Int.self, forKey: .acceptHiddenLayer) ?? 24
            imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? 151_655
            videoTokenID = try c.decodeIfPresent(Int.self, forKey: .videoTokenID) ?? 151_656
            audioTokenID = try c.decodeIfPresent(Int.self, forKey: .audioTokenID) ?? 151_675
            audioStartTokenID =
                try c.decodeIfPresent(Int.self, forKey: .audioStartTokenID) ?? 151_669
            audioEndTokenID =
                try c.decodeIfPresent(Int.self, forKey: .audioEndTokenID) ?? 151_670
            visionStartTokenID =
                try c.decodeIfPresent(Int.self, forKey: .visionStartTokenID) ?? 151_652
            numCodeGroups = try c.decodeIfPresent(Int.self, forKey: .numCodeGroups) ?? 16
            outputRouterLogits =
                try c.decodeIfPresent(Bool.self, forKey: .outputRouterLogits) ?? false
            positionIDPerSeconds =
                try c.decodeIfPresent(Int.self, forKey: .positionIDPerSeconds) ?? 13
            secondsPerChunk = try c.decodeIfPresent(Int.self, forKey: .secondsPerChunk) ?? 2
            spatialMergeSize = try c.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
            thinkerHiddenSize =
                try c.decodeIfPresent(Int.self, forKey: .thinkerHiddenSize) ?? 2048
            codecBOSID = try c.decodeIfPresent(Int.self, forKey: .codecBOSID) ?? 2149
            codecEOSTokenID =
                try c.decodeIfPresent(Int.self, forKey: .codecEOSTokenID) ?? 2150
            codecNoThinkID =
                try c.decodeIfPresent(Int.self, forKey: .codecNoThinkID) ?? 2155
            codecPadID = try c.decodeIfPresent(Int.self, forKey: .codecPadID) ?? 2148
            codecThinkBOSID =
                try c.decodeIfPresent(Int.self, forKey: .codecThinkBOSID) ?? 2156
            codecThinkEOSID =
                try c.decodeIfPresent(Int.self, forKey: .codecThinkEOSID) ?? 2157
            speakerID =
                try c.decodeIfPresent([String: Int].self, forKey: .speakerID)
                ?? ["chelsie": 2301, "ethan": 2302, "aiden": 2303]
        }
    }

    public struct RoPEScaling: Decodable, Sendable {
        public let type: String?
        public let mropeInterleaved: Bool?
        public let mropeSection: [Int]?

        enum CodingKeys: String, CodingKey {
            case type
            case mropeInterleaved = "mrope_interleaved"
            case mropeSection = "mrope_section"
        }

        public init(type: String? = nil, mropeInterleaved: Bool? = nil, mropeSection: [Int]? = nil)
        {
            self.type = type
            self.mropeInterleaved = mropeInterleaved
            self.mropeSection = mropeSection
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let audioConfiguration: AudioConfiguration?
    public let code2WavConfiguration: Code2WavConfiguration?
    public let talkerConfiguration: TalkerConfiguration?
    public let modelType: String
    public let enableAudioOutput: Bool
    public let imStartTokenID: Int
    public let imEndTokenID: Int
    public let systemTokenID: Int
    public let userTokenID: Int
    public let assistantTokenID: Int
    public let ttsBOSTokenID: Int
    public let ttsEOSTokenID: Int
    public let ttsPadTokenID: Int
    private let _ignoreIndex: Int?
    public var ignoreIndex: Int {
        _ignoreIndex ?? -100
    }

    private let _imageTokenID: Int?
    public var imageTokenID: Int {
        _imageTokenID ?? 151_655
    }

    private let _videoTokenID: Int?
    public var videoTokenID: Int {
        _videoTokenID ?? 151_656
    }

    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int {
        _imageTokenIndex ?? imageTokenID
    }

    private let _videoTokenIndex: Int?
    public var videoTokenIndex: Int {
        _videoTokenIndex ?? videoTokenID
    }

    private let _visionStartTokenID: Int?
    public var visionStartTokenID: Int {
        _visionStartTokenID ?? 151_652
    }

    private let _visionEndTokenID: Int?
    public var visionEndTokenID: Int {
        _visionEndTokenID ?? 151_653
    }

    private let _visionTokenID: Int?
    public var visionTokenID: Int {
        _visionTokenID ?? 151_654
    }

    private let _audioTokenID: Int?
    public var audioTokenID: Int {
        _audioTokenID ?? 151_675
    }

    private let _audioStartTokenID: Int?
    public var audioStartTokenID: Int {
        _audioStartTokenID ?? 151_669
    }

    private let _audioEndTokenID: Int?
    public var audioEndTokenID: Int {
        _audioEndTokenID ?? 151_670
    }

    private let _vocabSize: Int?
    public var vocabSize: Int {
        _vocabSize ?? textConfiguration.vocabSize
    }

    private let _eosTokenID: IntOrIntArray?
    public var eosTokenID: [Int]? {
        _eosTokenID?.values
    }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case audioConfiguration = "audio_config"
        case code2WavConfiguration = "code2wav_config"
        case talkerConfiguration = "talker_config"
        case thinkerConfiguration = "thinker_config"
        case modelType = "model_type"
        case enableAudioOutput = "enable_audio_output"
        case imStartTokenID = "im_start_token_id"
        case imEndTokenID = "im_end_token_id"
        case systemTokenID = "system_token_id"
        case userTokenID = "user_token_id"
        case assistantTokenID = "assistant_token_id"
        case ttsBOSTokenID = "tts_bos_token_id"
        case ttsEOSTokenID = "tts_eos_token_id"
        case ttsPadTokenID = "tts_pad_token_id"
        case _ignoreIndex = "ignore_index"
        case _imageTokenID = "image_token_id"
        case _videoTokenID = "video_token_id"
        case _imageTokenIndex = "image_token_index"
        case _videoTokenIndex = "video_token_index"
        case _visionStartTokenID = "vision_start_token_id"
        case _visionEndTokenID = "vision_end_token_id"
        case _visionTokenID = "vision_token_id"
        case _audioTokenID = "audio_token_id"
        case _audioStartTokenID = "audio_start_token_id"
        case _audioEndTokenID = "audio_end_token_id"
        case _vocabSize = "vocab_size"
        case _eosTokenID = "eos_token_id"
    }

    public init(
        textConfiguration: TextConfiguration, visionConfiguration: VisionConfiguration,
        audioConfiguration: AudioConfiguration? = nil,
        code2WavConfiguration: Code2WavConfiguration? = nil,
        talkerConfiguration: TalkerConfiguration? = nil,
        modelType: String = "qwen3_omni_moe", ignoreIndex: Int = -100, imageTokenID: Int = 151_655,
        videoTokenID: Int = 151_656, imageTokenIndex: Int? = nil, videoTokenIndex: Int? = nil,
        visionStartTokenID: Int = 151_652, visionEndTokenID: Int = 151_653,
        visionTokenID: Int = 151_654, audioTokenID: Int = 151_675,
        audioStartTokenID: Int = 151_669, audioEndTokenID: Int = 151_670,
        vocabSize: Int? = nil, eosTokenID: [Int]? = nil, enableAudioOutput: Bool = true,
        imStartTokenID: Int = 151_644, imEndTokenID: Int = 151_645,
        systemTokenID: Int = 8948, userTokenID: Int = 872, assistantTokenID: Int = 77091,
        ttsBOSTokenID: Int = 151_672, ttsEOSTokenID: Int = 151_673,
        ttsPadTokenID: Int = 151_671,
    ) {
        self.textConfiguration = textConfiguration
        self.visionConfiguration = visionConfiguration
        self.audioConfiguration = audioConfiguration
        self.code2WavConfiguration = code2WavConfiguration
        self.talkerConfiguration = talkerConfiguration
        self.modelType = modelType
        self.enableAudioOutput = enableAudioOutput
        self.imStartTokenID = imStartTokenID
        self.imEndTokenID = imEndTokenID
        self.systemTokenID = systemTokenID
        self.userTokenID = userTokenID
        self.assistantTokenID = assistantTokenID
        self.ttsBOSTokenID = ttsBOSTokenID
        self.ttsEOSTokenID = ttsEOSTokenID
        self.ttsPadTokenID = ttsPadTokenID
        _ignoreIndex = ignoreIndex
        _imageTokenID = imageTokenID
        _videoTokenID = videoTokenID
        _imageTokenIndex = imageTokenIndex
        _videoTokenIndex = videoTokenIndex
        _visionStartTokenID = visionStartTokenID
        _visionEndTokenID = visionEndTokenID
        _visionTokenID = visionTokenID
        _audioTokenID = audioTokenID
        _audioStartTokenID = audioStartTokenID
        _audioEndTokenID = audioEndTokenID
        _vocabSize = vocabSize
        _eosTokenID = eosTokenID.map(IntOrIntArray.init)
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let source: KeyedDecodingContainer<CodingKeys> =
            if let thinker = try? container.nestedContainer(
                keyedBy: CodingKeys.self,
                forKey: .thinkerConfiguration,
            ) {
                thinker
            } else {
                container
            }

        textConfiguration = try source.decode(
            TextConfiguration.self,
            forKey: .textConfiguration,
        )
        visionConfiguration = try source.decode(
            VisionConfiguration.self,
            forKey: .visionConfiguration,
        )
        audioConfiguration =
            try source.decodeIfPresent(AudioConfiguration.self, forKey: .audioConfiguration)
            ?? container.decodeIfPresent(AudioConfiguration.self, forKey: .audioConfiguration)
        code2WavConfiguration =
            try container.decodeIfPresent(
                Code2WavConfiguration.self, forKey: .code2WavConfiguration)
        talkerConfiguration =
            try container.decodeIfPresent(TalkerConfiguration.self, forKey: .talkerConfiguration)
        modelType =
            try container.decodeIfPresent(String.self, forKey: .modelType)
            ?? (source.decodeIfPresent(String.self, forKey: .modelType))
            ?? "qwen3_omni_moe"
        enableAudioOutput =
            try container.decodeIfPresent(Bool.self, forKey: .enableAudioOutput) ?? true
        imStartTokenID =
            try container.decodeIfPresent(Int.self, forKey: .imStartTokenID) ?? 151_644
        imEndTokenID =
            try container.decodeIfPresent(Int.self, forKey: .imEndTokenID) ?? 151_645
        systemTokenID =
            try container.decodeIfPresent(Int.self, forKey: .systemTokenID) ?? 8948
        userTokenID =
            try container.decodeIfPresent(Int.self, forKey: .userTokenID) ?? 872
        assistantTokenID =
            try container.decodeIfPresent(Int.self, forKey: .assistantTokenID) ?? 77091
        ttsBOSTokenID =
            try container.decodeIfPresent(Int.self, forKey: .ttsBOSTokenID) ?? 151_672
        ttsEOSTokenID =
            try container.decodeIfPresent(Int.self, forKey: .ttsEOSTokenID) ?? 151_673
        ttsPadTokenID =
            try container.decodeIfPresent(Int.self, forKey: .ttsPadTokenID) ?? 151_671
        _ignoreIndex =
            try source.decodeIfPresent(Int.self, forKey: ._ignoreIndex)
            ?? container.decodeIfPresent(Int.self, forKey: ._ignoreIndex)
        _imageTokenID =
            try source.decodeIfPresent(Int.self, forKey: ._imageTokenID)
            ?? container.decodeIfPresent(Int.self, forKey: ._imageTokenID)
        _videoTokenID =
            try source.decodeIfPresent(Int.self, forKey: ._videoTokenID)
            ?? container.decodeIfPresent(Int.self, forKey: ._videoTokenID)
        _imageTokenIndex =
            try source.decodeIfPresent(Int.self, forKey: ._imageTokenIndex)
            ?? container.decodeIfPresent(Int.self, forKey: ._imageTokenIndex)
        _videoTokenIndex =
            try source.decodeIfPresent(Int.self, forKey: ._videoTokenIndex)
            ?? container.decodeIfPresent(Int.self, forKey: ._videoTokenIndex)
        _visionStartTokenID =
            try source.decodeIfPresent(Int.self, forKey: ._visionStartTokenID)
            ?? container.decodeIfPresent(Int.self, forKey: ._visionStartTokenID)
        _visionEndTokenID =
            try source.decodeIfPresent(Int.self, forKey: ._visionEndTokenID)
            ?? container.decodeIfPresent(Int.self, forKey: ._visionEndTokenID)
        _visionTokenID =
            try source.decodeIfPresent(Int.self, forKey: ._visionTokenID)
            ?? container.decodeIfPresent(Int.self, forKey: ._visionTokenID)
        _audioTokenID =
            try source.decodeIfPresent(Int.self, forKey: ._audioTokenID)
            ?? container.decodeIfPresent(Int.self, forKey: ._audioTokenID)
        _audioStartTokenID =
            try source.decodeIfPresent(Int.self, forKey: ._audioStartTokenID)
            ?? container.decodeIfPresent(Int.self, forKey: ._audioStartTokenID)
        _audioEndTokenID =
            try source.decodeIfPresent(Int.self, forKey: ._audioEndTokenID)
            ?? container.decodeIfPresent(Int.self, forKey: ._audioEndTokenID)
        _vocabSize =
            try source.decodeIfPresent(Int.self, forKey: ._vocabSize)
            ?? container.decodeIfPresent(Int.self, forKey: ._vocabSize)
        _eosTokenID =
            try source.decodeIfPresent(IntOrIntArray.self, forKey: ._eosTokenID)
            ?? container.decodeIfPresent(IntOrIntArray.self, forKey: ._eosTokenID)
    }
}

// MARK: - Vision

enum Qwen3OmniMoEVision {
    static func applyRotary(_ tensor: MLXArray, freqs: MLXArray) -> MLXArray {
        var cosVals = cos(freqs)
        var sinVals = sin(freqs)

        cosVals = expandedDimensions(cosVals, axis: 1)
        cosVals = tiled(cosVals, repetitions: [1, 1, 2])
        cosVals = expandedDimensions(cosVals, axis: 0)

        sinVals = expandedDimensions(sinVals, axis: 1)
        sinVals = tiled(sinVals, repetitions: [1, 1, 2])
        sinVals = expandedDimensions(sinVals, axis: 0)

        let rotated = (tensor * cosVals) + (RotaryEmbeddingOps.rotateHalf(tensor) * sinVals)
        return rotated.asType(tensor.dtype)
    }

    final class VisionRotaryEmbedding {
        let dimension: Int
        let theta: Float

        init(dimension: Int, theta: Float = 10000) {
            self.dimension = dimension
            self.theta = theta
        }

        func callAsFunction(sequenceLength: Int) -> MLXArray {
            let invFreq =
                1.0
                / pow(
                    MLXArray(theta),
                    MLXArray(stride(from: 0, to: dimension, by: 2)).asType(.float32)
                        / Float(dimension),
                )
            let seq = MLXArray(0 ..< sequenceLength).asType(invFreq.dtype)
            return outer(seq, invFreq)
        }
    }

    final class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo(key: "proj") var proj: Conv3d

        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let hiddenSize: Int

        init(
            patchSize: Int,
            temporalPatchSize: Int,
            inChannels: Int,
            hiddenSize: Int,
        ) {
            self.patchSize = patchSize
            self.temporalPatchSize = temporalPatchSize
            self.inChannels = inChannels
            self.hiddenSize = hiddenSize

            let kernel = IntOrTriple([temporalPatchSize, patchSize, patchSize])
            _proj.wrappedValue = Conv3d(
                inputChannels: inChannels,
                outputChannels: hiddenSize,
                kernelSize: kernel,
                stride: kernel,
                bias: true,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var states = x.reshaped(
                -1,
                inChannels,
                temporalPatchSize,
                patchSize,
                patchSize,
            ).movedAxis(source: 1, destination: 4)

            states = proj(states)
            states = states.reshaped(-1, hiddenSize)
            return states
        }
    }

    final class PatchMerger: Module, UnaryLayer {
        let hiddenSize: Int
        let usePostShuffleNorm: Bool

        @ModuleInfo(key: "norm") var norm: LayerNorm
        @ModuleInfo(key: "linear_fc1") var linear1: Linear
        @ModuleInfo(key: "linear_fc2") var linear2: Linear
        @ModuleInfo(key: "act") var activation: GELU

        init(config: Qwen3OmniMoEConfiguration.VisionConfiguration, usePostShuffleNorm: Bool) {
            hiddenSize =
                config.hiddenSize * (config.spatialMergeSize * config.spatialMergeSize)
            self.usePostShuffleNorm = usePostShuffleNorm

            let normDim = usePostShuffleNorm ? hiddenSize : config.hiddenSize
            _norm.wrappedValue = LayerNorm(dimensions: normDim, eps: 1e-6)
            _linear1.wrappedValue = Linear(hiddenSize, hiddenSize)
            _linear2.wrappedValue = Linear(hiddenSize, config.outHiddenSize)
            _activation.wrappedValue = GELU()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var states = x
            if usePostShuffleNorm {
                states = states.reshaped(-1, hiddenSize)
            }
            states = norm(states)
            states = states.reshaped(-1, hiddenSize)
            states = linear1(states)
            states = activation(states)
            states = linear2(states)
            return states
        }
    }

    final class Attention: Module {
        let numHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "qkv") var qkv: Linear
        @ModuleInfo(key: "proj") var proj: Linear

        init(dim: Int, numHeads: Int) {
            self.numHeads = numHeads
            headDim = dim / numHeads
            scale = pow(Float(headDim), -0.5)

            _qkv.wrappedValue = Linear(dim, 3 * dim, bias: true)
            _proj.wrappedValue = Linear(dim, dim)
        }

        func callAsFunction(
            _ x: MLXArray,
            cuSeqlens: MLXArray,
            rotaryPosEmb: MLXArray,
        )
            -> MLXArray
        {
            let sequenceLength = x.dim(0)

            var qkvStates = qkv(x)
            qkvStates = qkvStates.reshaped(sequenceLength, 3, numHeads, headDim)
            qkvStates = qkvStates.transposed(1, 0, 2, 3)

            let parts = split(qkvStates, parts: 3, axis: 0)
            var queries = parts[0][0, 0..., 0..., 0...]
            var keys = parts[1][0, 0..., 0..., 0...]
            var values = parts[2][0, 0..., 0..., 0...]

            queries = applyRotary(queries, freqs: rotaryPosEmb)
            keys = applyRotary(keys, freqs: rotaryPosEmb)

            queries = queries.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)

            var mask = ones([1, sequenceLength, sequenceLength], dtype: queries.dtype)
            mask = mask * MLXArray(-1e9, dtype: queries.dtype)

            let seqlens = cuSeqlens.asArray(Int.self)
            for idx in 1 ..< seqlens.count {
                let start = seqlens[idx - 1]
                let end = seqlens[idx]
                mask[0..., start ..< end, start ..< end] = MLXArray(0, dtype: queries.dtype)
            }

            let attended = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .array(mask),
            )
            .transposed(0, 2, 1, 3)
            .reshaped(sequenceLength, -1)

            return proj(attended)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "linear_fc1") var linear1: Linear
        @ModuleInfo(key: "linear_fc2") var linear2: Linear
        @ModuleInfo(key: "act") var activation: GELU

        init(dim: Int, hiddenDim: Int) {
            _linear1.wrappedValue = Linear(dim, hiddenDim, bias: true)
            _linear2.wrappedValue = Linear(hiddenDim, dim, bias: true)
            _activation.wrappedValue = GELU(approximation: .fast)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear2(activation(linear1(x)))
        }
    }

    final class VisionBlock: Module {
        @ModuleInfo(key: "norm1") var norm1: LayerNorm
        @ModuleInfo(key: "norm2") var norm2: LayerNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP

        init(_ config: Qwen3OmniMoEConfiguration.VisionConfiguration) {
            _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _attention.wrappedValue = Attention(dim: config.hiddenSize, numHeads: config.numHeads)
            _mlp.wrappedValue = MLP(dim: config.hiddenSize, hiddenDim: config.intermediateSize)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            cuSeqlens: MLXArray,
            rotaryPosEmb: MLXArray,
        )
            -> MLXArray
        {
            var states = hiddenStates
            states =
                states + attention(norm1(states), cuSeqlens: cuSeqlens, rotaryPosEmb: rotaryPosEmb)
            states = states + mlp(norm2(states))
            return states
        }
    }

    final class VisionModel: Module {
        let config: Qwen3OmniMoEConfiguration.VisionConfiguration
        let spatialMergeSize: Int
        let numGridPerSide: Int

        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        @ModuleInfo(key: "rotary_pos_emb") var rotaryEmbedding: VisionRotaryEmbedding
        @ModuleInfo(key: "pos_embed") var posEmbed: Embedding
        @ModuleInfo(key: "blocks") var blocks: [VisionBlock]
        @ModuleInfo(key: "merger") var merger: PatchMerger
        @ModuleInfo(key: "deepstack_merger_list") var deepstackMergers: [PatchMerger]
        let deepstackVisualIndexes: [Int]

        init(_ config: Qwen3OmniMoEConfiguration.VisionConfiguration) {
            self.config = config
            spatialMergeSize = config.spatialMergeSize
            numGridPerSide = Int(sqrt(Double(config.numPositionEmbeddings)))
            deepstackVisualIndexes = config.deepstackVisualIndexes

            _patchEmbed.wrappedValue = PatchEmbed(
                patchSize: config.patchSize,
                temporalPatchSize: config.temporalPatchSize,
                inChannels: config.inChannels,
                hiddenSize: config.hiddenSize,
            )

            let headDim = config.hiddenSize / config.numHeads
            _rotaryEmbedding.wrappedValue = VisionRotaryEmbedding(dimension: headDim / 2)

            _posEmbed.wrappedValue = Embedding(
                embeddingCount: config.numPositionEmbeddings,
                dimensions: config.hiddenSize,
            )

            _blocks.wrappedValue = (0 ..< config.depth).map { _ in VisionBlock(config) }
            _merger.wrappedValue = PatchMerger(config: config, usePostShuffleNorm: false)

            _deepstackMergers.wrappedValue = config.deepstackVisualIndexes.map { _ in
                PatchMerger(config: config, usePostShuffleNorm: true)
            }
        }

        private func rotaryPositionEmbedding(_ grids: [THW]) -> MLXArray {
            guard let maxHW = grids.map({ max($0.h, $0.w) }).max(), maxHW > 0 else {
                return MLXArray.zeros([1, 1], dtype: .float32)
            }

            let freqTable = rotaryEmbedding(sequenceLength: maxHW)
            let halfDim = freqTable.dim(-1)

            let merge = spatialMergeSize
            var allCoords: [MLXArray] = []
            let mergeScalar = MLXArray(Int32(merge))

            for grid in grids {
                let mergedH = grid.h / merge
                let mergedW = grid.w / merge

                guard mergedH > 0, mergedW > 0 else {
                    continue
                }

                // Generate block and intra-block indices fully in MLX
                var blockRows = MLXArray(0 ..< mergedH).asType(.int32)
                blockRows = blockRows.reshaped([mergedH, 1, 1, 1])

                var blockCols = MLXArray(0 ..< mergedW).asType(.int32)
                blockCols = blockCols.reshaped([1, mergedW, 1, 1])

                let intra = MLXArray(0 ..< merge).asType(.int32)
                let intraRow = intra.reshaped([1, 1, merge, 1])
                let intraCol = intra.reshaped([1, 1, 1, merge])

                // Broadcast arithmetic mirrors the Python implementation
                var hIndex = blockRows * mergeScalar + intraRow
                var wIndex = blockCols * mergeScalar + intraCol

                hIndex = broadcast(hIndex, to: [mergedH, mergedW, merge, merge])
                wIndex = broadcast(wIndex, to: [mergedH, mergedW, merge, merge])

                // Flatten and stack coordinate pairs
                let hFlattened = hIndex.flattened()
                let wFlattened = wIndex.flattened()
                var coords = stacked([hFlattened, wFlattened], axis: -1)

                // Repeat for temporal frames
                if grid.t > 1 {
                    coords = tiled(coords, repetitions: [grid.t, 1])
                }

                allCoords.append(coords)
            }

            guard !allCoords.isEmpty else {
                return MLXArray.zeros([0, halfDim * 2], dtype: freqTable.dtype)
            }

            // Concatenate all coordinate pairs
            let allCoordsConcat = concatenated(allCoords, axis: 0)  // (total_tokens, 2)

            // Extract h and w indices and lookup embeddings
            let hIndices = allCoordsConcat[0..., 0].asType(.int32)
            let wIndices = allCoordsConcat[0..., 1].asType(.int32)

            let hEmbeds = freqTable[hIndices, 0...]
            let wEmbeds = freqTable[wIndices, 0...]

            // Concatenate height and width embeddings
            return concatenated([hEmbeds, wEmbeds], axis: -1)
        }

        private func positionalEmbeddings(_ grids: [THW]) -> MLXArray {
            let hiddenSize = config.hiddenSize
            let maxIndex = numGridPerSide - 1

            // Step 1: Collect all indices and weights from all grids using MLX ops
            var cornerIndices: [[MLXArray]] = Array(repeating: [], count: 4)
            var cornerWeights: [[MLXArray]] = Array(repeating: [], count: 4)
            var gridSizes: [Int] = []

            for grid in grids {
                let h = grid.h
                let w = grid.w
                gridSizes.append(h * w)

                // Create linspace indices using broadcasting
                var hLinspace = MLXArray(0 ..< h).asType(.float32)
                hLinspace = hLinspace * MLXArray(Float(maxIndex)) / MLXArray(Float(max(1, h - 1)))

                var wLinspace = MLXArray(0 ..< w).asType(.float32)
                wLinspace = wLinspace * MLXArray(Float(maxIndex)) / MLXArray(Float(max(1, w - 1)))

                // Get floor/ceil and deltas
                let hFloor = hLinspace.asType(.int32)
                let hCeil = minimum(hFloor + 1, maxIndex)
                let dh = hLinspace - hFloor.asType(.float32)

                let wFloor = wLinspace.asType(.int32)
                let wCeil = minimum(wFloor + 1, maxIndex)
                let dw = wLinspace - wFloor.asType(.float32)

                // Broadcast to create meshgrid
                let hFloorExpanded = expandedDimensions(hFloor, axis: 1)  // (h, 1)
                let hCeilExpanded = expandedDimensions(hCeil, axis: 1)
                let wFloorExpanded = expandedDimensions(wFloor, axis: 0)  // (1, w)
                let wCeilExpanded = expandedDimensions(wCeil, axis: 0)

                let baseH = hFloorExpanded * numGridPerSide
                let baseHCeil = hCeilExpanded * numGridPerSide

                // Compute 4 corner indices
                cornerIndices[0].append((baseH + wFloorExpanded).flattened())
                cornerIndices[1].append((baseH + wCeilExpanded).flattened())
                cornerIndices[2].append((baseHCeil + wFloorExpanded).flattened())
                cornerIndices[3].append((baseHCeil + wCeilExpanded).flattened())

                // Compute bilinear weights
                let dhExpanded = expandedDimensions(dh, axis: 1)
                let dwExpanded = expandedDimensions(dw, axis: 0)

                cornerWeights[0].append(((1 - dhExpanded) * (1 - dwExpanded)).flattened())
                cornerWeights[1].append(((1 - dhExpanded) * dwExpanded).flattened())
                cornerWeights[2].append((dhExpanded * (1 - dwExpanded)).flattened())
                cornerWeights[3].append((dhExpanded * dwExpanded).flattened())
            }

            guard !cornerIndices[0].isEmpty else {
                return MLXArray.zeros([0, hiddenSize], dtype: posEmbed.weight.dtype)
            }

            // Step 2: Batch embedding lookup
            let indicesTensors = cornerIndices.map { concatenated($0, axis: 0).asType(.int32) }
            let weightsTensors = cornerWeights.map {
                concatenated($0, axis: 0).asType(posEmbed.weight.dtype)
            }

            let totalPatches = indicesTensors[0].dim(0)
            var patchPosEmbeds = MLXArray.zeros(
                [totalPatches, hiddenSize], dtype: posEmbed.weight.dtype,
            )

            for cornerIdx in 0 ..< 4 {
                let cornerEmbeds = posEmbed(indicesTensors[cornerIdx])
                let weighted =
                    cornerEmbeds * expandedDimensions(weightsTensors[cornerIdx], axis: -1)
                patchPosEmbeds = patchPosEmbeds + weighted
            }

            // Step 3: Split by grid (like Python lines 344-349)
            var patchPosEmbedsSplit: [MLXArray] = []
            var offset = 0

            for size in gridSizes {
                let slice = patchPosEmbeds[offset ..< (offset + size), 0...]
                patchPosEmbedsSplit.append(slice)
                offset += size
            }

            // Step 4: Process each grid (like Python lines 354-371)
            var resultEmbeds: [MLXArray] = []
            let merge = spatialMergeSize

            for (gridIdx, grid) in grids.enumerated() {
                let posEmbed = patchPosEmbedsSplit[gridIdx]
                let h = grid.h
                let w = grid.w
                let t = grid.t

                let featureDim = posEmbed.dim(-1)

                // Repeat for temporal dimension
                var temporalEmbeds = tiled(posEmbed, repetitions: [t, 1])

                // Reshape for merge pattern
                temporalEmbeds = temporalEmbeds.reshaped(t, h, w, featureDim)
                temporalEmbeds = temporalEmbeds.reshaped(
                    t,
                    h / merge,
                    merge,
                    w / merge,
                    merge,
                    featureDim,
                )
                temporalEmbeds = temporalEmbeds.transposed(0, 1, 3, 2, 4, 5)
                temporalEmbeds = temporalEmbeds.reshaped(-1, featureDim)

                resultEmbeds.append(temporalEmbeds)
            }

            return concatenated(resultEmbeds, axis: 0)
        }

        private func cumulativeSequenceLengths(_ grids: [THW]) -> MLXArray {
            var seqLengths: [MLXArray] = []

            for grid in grids {
                let perFrame = grid.h * grid.w
                let repeated = tiled(MLXArray(perFrame), repetitions: [grid.t])
                seqLengths.append(repeated)
            }

            guard !seqLengths.isEmpty else {
                return MLXArray(0, dtype: .int32)
            }

            let concatSeqLengths = concatenated(seqLengths).asType(.int32)

            let cumSum = concatSeqLengths.cumsum()

            return padded(
                cumSum, widths: [IntOrPair((1, 0))], mode: .constant,
                value: MLXArray(0, dtype: cumSum.dtype),
            )
        }

        func callAsFunction(_ pixelValues: MLXArray, gridTHW: [THW]) -> (MLXArray, [MLXArray]) {
            var hiddenStates = patchEmbed(pixelValues)

            let posEmbeds = positionalEmbeddings(gridTHW)
            hiddenStates = hiddenStates + posEmbeds

            let rotaryEmbeds = rotaryPositionEmbedding(gridTHW)
            let cuSeqlens = cumulativeSequenceLengths(gridTHW)

            var deepstackOutputs: [MLXArray] = []

            for (index, block) in blocks.enumerated() {
                hiddenStates = block(hiddenStates, cuSeqlens: cuSeqlens, rotaryPosEmb: rotaryEmbeds)
                if let dsIndex = deepstackVisualIndexes.firstIndex(of: index) {
                    let feature = deepstackMergers[dsIndex](hiddenStates)
                    deepstackOutputs.append(feature)
                }
            }

            hiddenStates = merger(hiddenStates)
            return (hiddenStates, deepstackOutputs)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized: [String: MLXArray] = [:]
            for (key, value) in weights {
                if key.contains("position_ids") {
                    continue
                } else if key.contains("patch_embed.proj.weight") {
                    if value.ndim == 5, value.dim(-1) == config.inChannels {
                        sanitized[key] = value
                    } else {
                        sanitized[key] = value.transposed(0, 2, 3, 4, 1)
                    }
                } else {
                    sanitized[key] = value
                }
            }
            return sanitized
        }
    }
}

// MARK: - Audio

enum Qwen3OmniMoEAudio {
    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: Qwen3OmniMoEConfiguration.AudioConfiguration) {
            heads = config.encoderAttentionHeads
            headDim = config.modelDimensions / config.encoderAttentionHeads
            scale = pow(Float(headDim), -0.5)
            _kProj.wrappedValue = Linear(config.modelDimensions, config.modelDimensions, bias: true)
            _vProj.wrappedValue = Linear(config.modelDimensions, config.modelDimensions, bias: true)
            _qProj.wrappedValue = Linear(config.modelDimensions, config.modelDimensions, bias: true)
            _outProj.wrappedValue = Linear(
                config.modelDimensions, config.modelDimensions, bias: true)
        }

        func callAsFunction(_ hiddenStates: MLXArray, cuSeqlens: MLXArray) -> MLXArray {
            let sequenceLength = hiddenStates.dim(0)
            guard sequenceLength > 0 else {
                return hiddenStates
            }

            var queries = qProj(hiddenStates)
                .reshaped(sequenceLength, heads, headDim)
                .transposed(1, 0, 2)
                .expandedDimensions(axis: 0)
            var keys = kProj(hiddenStates)
                .reshaped(sequenceLength, heads, headDim)
                .transposed(1, 0, 2)
                .expandedDimensions(axis: 0)
            let values = vProj(hiddenStates)
                .reshaped(sequenceLength, heads, headDim)
                .transposed(1, 0, 2)
                .expandedDimensions(axis: 0)

            queries = queries.asType(hiddenStates.dtype)
            keys = keys.asType(hiddenStates.dtype)

            var mask = MLXArray.zeros([1, sequenceLength, sequenceLength], dtype: queries.dtype)
            let seqlens = cuSeqlens.asArray(Int.self)
            if seqlens.count > 1 {
                mask =
                    ones([1, sequenceLength, sequenceLength], dtype: queries.dtype)
                    * MLXArray(-1e9, dtype: queries.dtype)
                for idx in 1 ..< seqlens.count {
                    let start = max(0, min(seqlens[idx - 1], sequenceLength))
                    let end = max(start, min(seqlens[idx], sequenceLength))
                    if start < end {
                        mask[0..., start ..< end, start ..< end] =
                            MLXArray(0, dtype: queries.dtype)
                    }
                }
            }

            let attended = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .array(mask),
            )
            .transposed(0, 2, 1, 3)
            .reshaped(sequenceLength, -1)

            return outProj(attended)
        }
    }

    final class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo(key: "self_attn_layer_norm") var selfAttentionLayerNorm: LayerNorm
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

        let activation = GELU(approximation: .precise)

        init(_ config: Qwen3OmniMoEConfiguration.AudioConfiguration) {
            _selfAttention.wrappedValue = Attention(config)
            _selfAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: config.modelDimensions)
            _fc1.wrappedValue = Linear(config.modelDimensions, config.encoderFFNDimensions)
            _fc2.wrappedValue = Linear(config.encoderFFNDimensions, config.modelDimensions)
            _finalLayerNorm.wrappedValue = LayerNorm(dimensions: config.modelDimensions)
        }

        func callAsFunction(_ hiddenStates: MLXArray, cuSeqlens: MLXArray) -> MLXArray {
            var hidden = hiddenStates
            var residual = hidden
            hidden = selfAttentionLayerNorm(hidden)
            hidden = selfAttention(hidden, cuSeqlens: cuSeqlens)
            hidden = residual + hidden

            residual = hidden
            hidden = finalLayerNorm(hidden)
            hidden = fc2(activation(fc1(hidden)))
            return residual + hidden
        }
    }

    struct SinusoidalPositionEmbedding {
        let embeddings: MLXArray

        init(length: Int, channels: Int, maxTimescale: Float = 10000) {
            let half = max(channels / 2, 1)
            let logIncrement = log(maxTimescale) / Float(max(half - 1, 1))
            let invTimescales = (0 ..< half).map { index in
                exp(-logIncrement * Float(index))
            }
            var values: [Float] = []
            values.reserveCapacity(length * channels)
            for position in 0 ..< length {
                for timescale in invTimescales {
                    values.append(sin(Float(position) * timescale))
                }
                for timescale in invTimescales {
                    values.append(cos(Float(position) * timescale))
                }
            }
            embeddings = MLXArray(values, [length, half * 2])
        }

        func callAsFunction(sequenceLength: Int) -> MLXArray {
            if sequenceLength <= embeddings.dim(0) {
                return embeddings[..<sequenceLength, 0...]
            }

            let extra = sequenceLength - embeddings.dim(0)
            return padded(embeddings, widths: [.init((0, extra)), 0])
        }
    }

    final class AudioModel: Module {
        let config: Qwen3OmniMoEConfiguration.AudioConfiguration
        let positionalEmbedding: SinusoidalPositionEmbedding

        @ModuleInfo(key: "conv2d1") var conv2d1: Conv2d
        @ModuleInfo(key: "conv2d2") var conv2d2: Conv2d
        @ModuleInfo(key: "conv2d3") var conv2d3: Conv2d
        @ModuleInfo(key: "conv_out") var convOut: Linear
        @ModuleInfo(key: "layers") var layers: [EncoderLayer]
        @ModuleInfo(key: "ln_post") var layerNormPost: LayerNorm
        @ModuleInfo var proj1: Linear
        @ModuleInfo var proj2: Linear

        let activation = GELU(approximation: .precise)

        init(_ config: Qwen3OmniMoEConfiguration.AudioConfiguration) {
            self.config = config
            positionalEmbedding = SinusoidalPositionEmbedding(
                length: config.maxSourcePositions,
                channels: config.modelDimensions,
            )

            _conv2d1.wrappedValue = Conv2d(
                inputChannels: 1,
                outputChannels: config.downsampleHiddenSize,
                kernelSize: 3,
                stride: 2,
                padding: 1,
            )
            _conv2d2.wrappedValue = Conv2d(
                inputChannels: config.downsampleHiddenSize,
                outputChannels: config.downsampleHiddenSize,
                kernelSize: 3,
                stride: 2,
                padding: 1,
            )
            _conv2d3.wrappedValue = Conv2d(
                inputChannels: config.downsampleHiddenSize,
                outputChannels: config.downsampleHiddenSize,
                kernelSize: 3,
                stride: 2,
                padding: 1,
            )
            let downsampledMelBins = Self.downsampledLength(config.melBins)
            _convOut.wrappedValue = Linear(
                config.downsampleHiddenSize * downsampledMelBins,
                config.modelDimensions,
                bias: false,
            )
            _layers.wrappedValue = (0 ..< config.encoderLayers).map { _ in EncoderLayer(config) }
            _layerNormPost.wrappedValue = LayerNorm(dimensions: config.modelDimensions)
            _proj1.wrappedValue = Linear(config.modelDimensions, config.modelDimensions)
            _proj2.wrappedValue = Linear(config.modelDimensions, config.outputDimensions)
        }

        func callAsFunction(_ audio: LMInput.ProcessedAudio) throws -> MLXArray {
            var inputFeatures = normalizedInputFeatures(audio.features)
            let batchSize = inputFeatures.dim(0)
            let frameCount = inputFeatures.dim(1)
            guard inputFeatures.dim(2) == config.melBins else {
                throw VLMError.processing(
                    "Qwen3-Omni audio features must have \(config.melBins) mel bins.",
                )
            }

            let featureLengths = featureLengths(
                mask: audio.mask, batchSize: batchSize, frameCount: frameCount)
            var sampleFeatures: [MLXArray] = []
            sampleFeatures.reserveCapacity(batchSize)
            for batchIndex in 0 ..< batchSize {
                let length = max(1, min(featureLengths[batchIndex], frameCount))
                sampleFeatures.append(inputFeatures[batchIndex, ..<length, 0...])
            }

            inputFeatures = concatenated(sampleFeatures, axis: 0)
            let chunkLength = max(config.windowSize * 2, 1)
            var chunkLengths: [Int] = []
            chunkLengths.reserveCapacity(sampleFeatures.count)
            for length in featureLengths {
                let safeLength = max(length, 1)
                var remaining = safeLength
                while remaining > 0 {
                    let current = min(chunkLength, remaining)
                    chunkLengths.append(current)
                    remaining -= current
                }
            }

            guard let maxChunkLength = chunkLengths.max() else {
                return MLXArray.zeros([0, config.outputDimensions], dtype: .float32)
            }

            var chunks: [MLXArray] = []
            chunks.reserveCapacity(chunkLengths.count)
            var offset = 0
            for length in chunkLengths {
                let end = min(offset + length, inputFeatures.dim(0))
                var chunk = inputFeatures[offset ..< end, 0...]
                if chunk.dim(0) < maxChunkLength {
                    chunk = padded(chunk, widths: [.init((0, maxChunkLength - chunk.dim(0))), 0])
                }
                chunks.append(chunk.transposed(1, 0))
                offset = end
            }

            var paddedFeatures = stacked(chunks, axis: 0).expandedDimensions(axis: -1)
            paddedFeatures = paddedFeatures.asType(conv2d1.weight.dtype)

            var paddedEmbed = activation(conv2d1(paddedFeatures))
            paddedEmbed = activation(conv2d2(paddedEmbed))
            paddedEmbed = activation(conv2d3(paddedEmbed))

            let chunkCount = paddedEmbed.dim(0)
            let melLength = paddedEmbed.dim(1)
            let sequenceLength = paddedEmbed.dim(2)
            let channels = paddedEmbed.dim(3)
            paddedEmbed =
                paddedEmbed
                .transposed(0, 2, 3, 1)
                .reshaped(chunkCount, sequenceLength, channels * melLength)
            paddedEmbed = convOut(paddedEmbed)
            paddedEmbed =
                paddedEmbed
                + positionalEmbedding(sequenceLength: sequenceLength)
                .asType(paddedEmbed.dtype)
                .expandedDimensions(axis: 0)

            let chunkOutputLengths = chunkLengths.map(Qwen3OmniSupport.audioFeatureOutputLength)
            var linearIndices: [Int] = []
            for (chunkIndex, outputLength) in chunkOutputLengths.enumerated() {
                let usable = min(outputLength, sequenceLength)
                if usable > 0 {
                    linearIndices.append(
                        contentsOf: (0 ..< usable).map { chunkIndex * sequenceLength + $0 })
                }
            }

            guard !linearIndices.isEmpty else {
                return MLXArray.zeros([0, config.outputDimensions], dtype: paddedEmbed.dtype)
            }

            var hiddenStates =
                paddedEmbed
                .reshaped(-1, config.modelDimensions)
                .take(MLXArray(linearIndices.map { Int32($0) }), axis: 0)

            let sampleOutputLengths = featureLengths.map(Qwen3OmniSupport.audioFeatureOutputLength)
            let cuSeqlens = cumulativeSequenceLengths(sampleOutputLengths)
            for layer in layers {
                hiddenStates = layer(hiddenStates, cuSeqlens: cuSeqlens)
            }

            hiddenStates = layerNormPost(hiddenStates)
            hiddenStates = proj1(hiddenStates)
            hiddenStates = activation(hiddenStates)
            return proj2(hiddenStates)
        }

        private func normalizedInputFeatures(_ features: MLXArray) -> MLXArray {
            var features = features
            if features.ndim == 2 {
                if features.dim(-1) == config.melBins {
                    features = features.expandedDimensions(axis: 0)
                } else {
                    features = features.transposed(1, 0).expandedDimensions(axis: 0)
                }
            } else if features.ndim == 3,
                features.dim(1) == config.melBins,
                features.dim(2) != config.melBins
            {
                features = features.transposed(0, 2, 1)
            }
            return features
        }

        private static func downsampledLength(_ length: Int) -> Int {
            func strideTwo(_ value: Int) -> Int {
                value <= 0 ? 0 : (value + 1) / 2
            }
            return strideTwo(strideTwo(strideTwo(length)))
        }

        private func featureLengths(mask: MLXArray?, batchSize: Int, frameCount: Int) -> [Int] {
            guard let mask else {
                return Array(repeating: frameCount, count: batchSize)
            }

            var currentMask = mask.asType(.bool)
            if currentMask.ndim == 1 {
                currentMask = currentMask.expandedDimensions(axis: 0)
            }
            let values = currentMask.asArray(Bool.self)
            var lengths: [Int] = []
            lengths.reserveCapacity(batchSize)
            for batchIndex in 0 ..< batchSize {
                let start = batchIndex * frameCount
                let end = min(start + frameCount, values.count)
                let count = values[start ..< end].count(where: { $0 })
                lengths.append(max(count, 1))
            }
            return lengths
        }

        private func cumulativeSequenceLengths(_ lengths: [Int]) -> MLXArray {
            var values = [Int32(0)]
            values.reserveCapacity(lengths.count + 1)
            var running = 0
            for length in lengths {
                running += max(length, 0)
                values.append(Int32(running))
            }
            return MLXArray(values)
        }
    }
}

// MARK: - Code2Wav

public enum Qwen3OmniMoECode2Wav {
    public static func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (key, value) in weights {
            if key.contains("position_ids") || key.contains("rotary_emb.inv_freq") {
                continue
            }

            if (key.contains("upsample") && key.contains("conv.weight") && !key.contains("dwconv"))
                || (key.contains("decoder") && key.contains("block")
                    && key.contains("conv.weight")
                    && !key.contains("conv1") && !key.contains("conv2"))
            {
                sanitized[key] = value.transposed(1, 2, 0)
            } else if key.contains("dwconv.conv.weight")
                || (key.contains("decoder") && key.contains("conv.weight")
                    && !key.contains("block"))
                || (key.contains("decoder") && key.contains("block")
                    && (key.contains("conv1.conv.weight") || key.contains("conv2.conv.weight")))
            {
                sanitized[key] = value.transposed(0, 2, 1)
            } else {
                sanitized[key] = value
            }
        }

        return sanitized
    }

    static func activate(_ x: MLXArray, _ name: String) -> MLXArray {
        switch name {
        case "gelu", "gelu_pytorch_tanh":
            gelu(x)
        case "silu", "swish":
            silu(x)
        default:
            silu(x)
        }
    }

    final class SnakeBeta: Module, UnaryLayer {
        @ModuleInfo var alpha: MLXArray
        @ModuleInfo var beta: MLXArray

        init(channels: Int) {
            _alpha.wrappedValue = MLXArray.zeros([channels])
            _beta.wrappedValue = MLXArray.zeros([channels])
        }

        func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            let alpha = exp(alpha)[.newAxis, 0..., .newAxis]
            let beta = exp(beta)[.newAxis, 0..., .newAxis]
            return hiddenStates
                + (1.0 / (beta + MLXArray(1e-9, dtype: hiddenStates.dtype)))
                * sin(hiddenStates * alpha).square()
        }
    }

    final class LayerScale: Module, UnaryLayer {
        @ModuleInfo var scale: MLXArray

        init(_ config: Qwen3OmniMoEConfiguration.Code2WavConfiguration) {
            _scale.wrappedValue = MLXArray(
                Array(repeating: config.layerScaleInitialScale, count: config.hiddenSize),
                [config.hiddenSize],
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            scale * x
        }
    }

    final class CausalConvNet: Module, UnaryLayer {
        @ModuleInfo var conv: Conv1d

        let stride: Int
        let kernelSize: Int
        let padding: Int

        init(
            inputChannels: Int,
            outputChannels: Int,
            kernelSize: Int,
            dilation: Int = 1,
            stride: Int = 1,
            groups: Int = 1,
        ) {
            self.stride = stride
            self.kernelSize = (kernelSize - 1) * dilation + 1
            padding = self.kernelSize - stride
            _conv.wrappedValue = Conv1d(
                inputChannels: inputChannels,
                outputChannels: outputChannels,
                kernelSize: kernelSize,
                stride: stride,
                dilation: dilation,
                groups: groups,
            )
        }

        private func extraPadding(length: Int) -> Int {
            let frames = (Float(length - kernelSize + padding) / Float(stride)) + 1
            let idealLength =
                (ceil(frames) - 1) * Float(stride) + Float(kernelSize - padding)
            return max(0, Int(idealLength) - length)
        }

        func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            let length = hiddenStates.dim(-1)
            let extra = extraPadding(length: length)
            var hidden = hiddenStates.transposed(0, 2, 1)
            hidden = padded(hidden, widths: [0, .init((padding, extra)), 0])
            return conv(hidden).transposed(0, 2, 1)
        }
    }

    final class CausalTransConvNet: Module, UnaryLayer {
        @ModuleInfo var conv: ConvTransposed1d

        let rightPad: Int

        init(inputChannels: Int, outputChannels: Int, kernelSize: Int, stride: Int = 1) {
            rightPad = max(0, kernelSize - stride)
            _conv.wrappedValue = ConvTransposed1d(
                inputChannels: inputChannels,
                outputChannels: outputChannels,
                kernelSize: kernelSize,
                stride: stride,
            )
        }

        func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            var hidden = hiddenStates.transposed(0, 2, 1)
            hidden = conv(hidden)
            let end = hidden.dim(1) - rightPad
            if rightPad > 0, end > 0 {
                hidden = hidden[0..., ..<end, 0...]
            }
            return hidden.transposed(0, 2, 1)
        }
    }

    final class ConvNeXtBlock: Module, UnaryLayer {
        @ModuleInfo var dwconv: CausalConvNet
        @ModuleInfo var norm: LayerNorm
        @ModuleInfo var pwconv1: Linear
        @ModuleInfo var pwconv2: Linear
        @ModuleInfo var gamma: MLXArray

        init(dimensions: Int) {
            _dwconv.wrappedValue = CausalConvNet(
                inputChannels: dimensions,
                outputChannels: dimensions,
                kernelSize: 7,
                groups: dimensions,
            )
            _norm.wrappedValue = LayerNorm(dimensions: dimensions, eps: 1e-6)
            _pwconv1.wrappedValue = Linear(dimensions, 4 * dimensions)
            _pwconv2.wrappedValue = Linear(4 * dimensions, dimensions)
            _gamma.wrappedValue = MLXArray(
                Array(repeating: Float(1e-6), count: dimensions),
                [dimensions],
            )
        }

        func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            let residual = hiddenStates
            var hidden = dwconv(hiddenStates)
            hidden = hidden.transposed(0, 2, 1)
            hidden = norm(hidden)
            hidden = pwconv1(hidden)
            hidden = gelu(hidden)
            hidden = pwconv2(hidden)
            hidden = gamma * hidden
            hidden = hidden.transposed(0, 2, 1)
            return residual + hidden
        }
    }

    final class DecoderResUnit: Module, UnaryLayer {
        @ModuleInfo var act1: SnakeBeta
        @ModuleInfo var conv1: CausalConvNet
        @ModuleInfo var act2: SnakeBeta
        @ModuleInfo var conv2: CausalConvNet

        init(dimensions: Int, dilation: Int) {
            _act1.wrappedValue = SnakeBeta(channels: dimensions)
            _conv1.wrappedValue = CausalConvNet(
                inputChannels: dimensions,
                outputChannels: dimensions,
                kernelSize: 7,
                dilation: dilation,
            )
            _act2.wrappedValue = SnakeBeta(channels: dimensions)
            _conv2.wrappedValue = CausalConvNet(
                inputChannels: dimensions,
                outputChannels: dimensions,
                kernelSize: 1,
            )
        }

        func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            let residual = hiddenStates
            var hidden = act1(hiddenStates)
            hidden = conv1(hidden)
            hidden = act2(hidden)
            hidden = conv2(hidden)
            return hidden + residual
        }
    }

    final class DecoderBlock: Module, UnaryLayer {
        @ModuleInfo var block: [UnaryLayer]

        init(_ config: Qwen3OmniMoEConfiguration.Code2WavConfiguration, index: Int) {
            let inputDimensions = config.decoderDimension / (1 << index)
            let outputDimensions = config.decoderDimension / (1 << (index + 1))
            let upsampleRate = config.upsampleRates[index]
            _block.wrappedValue = [
                SnakeBeta(channels: inputDimensions),
                CausalTransConvNet(
                    inputChannels: inputDimensions,
                    outputChannels: outputDimensions,
                    kernelSize: 2 * upsampleRate,
                    stride: upsampleRate,
                ),
                DecoderResUnit(dimensions: outputDimensions, dilation: 1),
                DecoderResUnit(dimensions: outputDimensions, dilation: 3),
                DecoderResUnit(dimensions: outputDimensions, dilation: 9),
            ]
        }

        func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            var hidden = hiddenStates
            for layer in block {
                hidden = layer(hidden)
            }
            return hidden
        }
    }

    final class UpsampleStage: Module, UnaryLayer {
        @ModuleInfo(key: "0") var transposedConv: CausalTransConvNet
        @ModuleInfo(key: "1") var convNeXt: ConvNeXtBlock

        init(hiddenSize: Int, factor: Int) {
            _transposedConv.wrappedValue = CausalTransConvNet(
                inputChannels: hiddenSize,
                outputChannels: hiddenSize,
                kernelSize: factor,
                stride: factor,
            )
            _convNeXt.wrappedValue = ConvNeXtBlock(dimensions: hiddenSize)
        }

        func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            convNeXt(transposedConv(hiddenStates))
        }
    }

    final class RotaryEmbedding {
        let invFreq: MLXArray

        init(_ config: Qwen3OmniMoEConfiguration.Code2WavConfiguration) {
            let headDimension = config.headDimension
            var frequencies = MLXArray(stride(from: 0, to: headDimension, by: 2)).asType(.float32)
            frequencies = frequencies / Float(max(headDimension, 1))
            invFreq = 1.0 / pow(MLXArray(config.ropeTheta), frequencies)
        }

        func callAsFunction(_ x: MLXArray, positionIDs: MLXArray) -> (MLXArray, MLXArray) {
            let positions = positionIDs.asType(.float32).expandedDimensions(axis: -1)
            let freqs = positions * invFreq.asType(.float32)[.newAxis, .newAxis, 0...]
            let embeddings = concatenated([freqs, freqs], axis: -1)
            return (cos(embeddings).asType(x.dtype), sin(embeddings).asType(x.dtype))
        }
    }

    static func applyRotaryPositionEmbedding(
        queries: MLXArray,
        keys: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
    )
        -> (MLXArray, MLXArray)
    {
        let cos = cos.expandedDimensions(axis: 1)
        let sin = sin.expandedDimensions(axis: 1)
        let qEmbedded = (queries * cos) + (RotaryEmbeddingOps.rotateHalf(queries) * sin)
        let kEmbedded = (keys * cos) + (RotaryEmbeddingOps.rotateHalf(keys) * sin)
        return (qEmbedded, kEmbedded)
    }

    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDimension: Int
        let kvGroups: Int
        let scale: Float
        let rotaryEmbedding: RotaryEmbedding

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        init(_ config: Qwen3OmniMoEConfiguration.Code2WavConfiguration) {
            heads = config.numAttentionHeads
            kvHeads = config.numKeyValueHeads
            headDimension = config.headDimension
            kvGroups = max(1, config.numAttentionHeads / max(config.numKeyValueHeads, 1))
            scale = pow(Float(max(headDimension, 1)), -0.5)
            rotaryEmbedding = RotaryEmbedding(config)

            _qProj.wrappedValue = Linear(
                config.hiddenSize,
                config.numAttentionHeads * headDimension,
                bias: config.attentionBias,
            )
            _kProj.wrappedValue = Linear(
                config.hiddenSize,
                config.numKeyValueHeads * headDimension,
                bias: config.attentionBias,
            )
            _vProj.wrappedValue = Linear(
                config.hiddenSize,
                config.numKeyValueHeads * headDimension,
                bias: config.attentionBias,
            )
            _oProj.wrappedValue = Linear(
                config.numAttentionHeads * headDimension,
                config.hiddenSize,
                bias: config.attentionBias,
            )
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            attentionMask: MLXArray? = nil,
            positionIDs: MLXArray? = nil,
            positionEmbeddings: (MLXArray, MLXArray)? = nil,
        )
            -> MLXArray
        {
            let batchSize = hiddenStates.dim(0)
            let sequenceLength = hiddenStates.dim(1)

            var queries = qProj(hiddenStates)
                .reshaped(batchSize, sequenceLength, heads, headDimension)
                .transposed(0, 2, 1, 3)
            var keys = kProj(hiddenStates)
                .reshaped(batchSize, sequenceLength, kvHeads, headDimension)
                .transposed(0, 2, 1, 3)
            var values = vProj(hiddenStates)
                .reshaped(batchSize, sequenceLength, kvHeads, headDimension)
                .transposed(0, 2, 1, 3)

            let embeddings: (MLXArray, MLXArray)
            if let positionEmbeddings {
                embeddings = positionEmbeddings
            } else {
                let ids =
                    positionIDs
                    ?? broadcast(
                        MLXArray(0 ..< sequenceLength).asType(.int32)[.newAxis, 0...],
                        to: [batchSize, sequenceLength],
                    )
                embeddings = rotaryEmbedding(hiddenStates, positionIDs: ids)
            }

            (queries, keys) = applyRotaryPositionEmbedding(
                queries: queries,
                keys: keys,
                cos: embeddings.0,
                sin: embeddings.1,
            )

            if kvGroups > 1 {
                keys = repeated(keys, count: kvGroups, axis: 1)
                values = repeated(values, count: kvGroups, axis: 1)
            }

            let mask =
                attentionMask ?? Self.causalMask(length: sequenceLength, dtype: queries.dtype)
            let output = MLXFast.scaledDotProductAttention(
                queries: queries,
                keys: keys,
                values: values,
                scale: scale,
                mask: .array(mask),
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, sequenceLength, -1)

            return oProj(output)
        }

        private static func causalMask(length: Int, dtype: MLX.DType) -> MLXArray {
            var values: [Float] = []
            values.reserveCapacity(length * length)
            for query in 0 ..< length {
                for key in 0 ..< length {
                    values.append(key <= query ? 0 : -1e9)
                }
            }
            return MLXArray(values, [1, 1, length, length]).asType(dtype)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        let activationName: String

        init(_ config: Qwen3OmniMoEConfiguration.Code2WavConfiguration) {
            activationName = config.hiddenActivation
            _gateProj.wrappedValue = Linear(
                config.hiddenSize,
                config.intermediateSize,
                bias: false,
            )
            _upProj.wrappedValue = Linear(
                config.hiddenSize,
                config.intermediateSize,
                bias: false,
            )
            _downProj.wrappedValue = Linear(
                config.intermediateSize,
                config.hiddenSize,
                bias: false,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(activate(gateProj(x), activationName) * upProj(x))
        }
    }

    final class TransformerLayer: Module {
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
        @ModuleInfo(key: "self_attn_layer_scale") var selfAttentionLayerScale: LayerScale
        @ModuleInfo(key: "mlp_layer_scale") var mlpLayerScale: LayerScale

        init(_ config: Qwen3OmniMoEConfiguration.Code2WavConfiguration) {
            _selfAttention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(config)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _selfAttentionLayerScale.wrappedValue = LayerScale(config)
            _mlpLayerScale.wrappedValue = LayerScale(config)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            attentionMask: MLXArray? = nil,
            positionIDs: MLXArray? = nil,
            positionEmbeddings: (MLXArray, MLXArray)? = nil,
        )
            -> MLXArray
        {
            var residual = hiddenStates
            var hidden = inputLayerNorm(hiddenStates)
            hidden = selfAttention(
                hidden,
                attentionMask: attentionMask,
                positionIDs: positionIDs,
                positionEmbeddings: positionEmbeddings,
            )
            hidden = residual + selfAttentionLayerScale(hidden)

            residual = hidden
            hidden = postAttentionLayerNorm(hidden)
            hidden = mlp(hidden)
            return residual + mlpLayerScale(hidden)
        }
    }

    final class TransformerModel: Module {
        @ModuleInfo var layers: [TransformerLayer]
        @ModuleInfo var norm: RMSNorm

        let rotaryEmbedding: RotaryEmbedding

        init(_ config: Qwen3OmniMoEConfiguration.Code2WavConfiguration) {
            rotaryEmbedding = RotaryEmbedding(config)
            _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
                TransformerLayer(config)
            }
            _norm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
        }

        func callAsFunction(
            inputEmbeddings: MLXArray,
            attentionMask: MLXArray? = nil,
            positionIDs: MLXArray? = nil,
        )
            -> MLXArray
        {
            var hidden = inputEmbeddings
            let batchSize = hidden.dim(0)
            let sequenceLength = hidden.dim(1)
            let ids =
                positionIDs
                ?? broadcast(
                    MLXArray(0 ..< sequenceLength).asType(.int32)[.newAxis, 0...],
                    to: [batchSize, sequenceLength],
                )
            let positionEmbeddings = rotaryEmbedding(hidden, positionIDs: ids)

            for layer in layers {
                hidden = layer(
                    hidden,
                    attentionMask: attentionMask,
                    positionIDs: ids,
                    positionEmbeddings: positionEmbeddings,
                )
            }
            return norm(hidden)
        }
    }

    public final class Code2WavModel: Module {
        public let config: Qwen3OmniMoEConfiguration.Code2WavConfiguration
        private let codeOffset: MLXArray

        @ModuleInfo(key: "pre_transformer") var preTransformer: TransformerModel
        @ModuleInfo(key: "code_embedding") var codeEmbedding: Embedding
        @ModuleInfo var upsample: [UpsampleStage]
        @ModuleInfo var decoder: [UnaryLayer]

        public init(_ config: Qwen3OmniMoEConfiguration.Code2WavConfiguration) {
            self.config = config
            codeOffset = MLXArray(
                (0 ..< config.numQuantizers).map { Int32($0 * config.codebookSize) },
                [1, config.numQuantizers, 1],
            )
            _preTransformer.wrappedValue = TransformerModel(config)
            _codeEmbedding.wrappedValue = Embedding(
                embeddingCount: config.codebookSize * config.numQuantizers,
                dimensions: config.hiddenSize,
            )
            _upsample.wrappedValue = config.upsamplingRatios.map { factor in
                UpsampleStage(hiddenSize: config.hiddenSize, factor: factor)
            }

            var decoderLayers: [UnaryLayer] = [
                CausalConvNet(
                    inputChannels: config.hiddenSize,
                    outputChannels: config.decoderDimension,
                    kernelSize: 7,
                )
            ]
            for index in config.upsampleRates.indices {
                decoderLayers.append(DecoderBlock(config, index: index))
            }
            let outputDimension =
                config.decoderDimension / max(1, 1 << config.upsampleRates.count)
            decoderLayers.append(SnakeBeta(channels: outputDimension))
            decoderLayers.append(
                CausalConvNet(
                    inputChannels: outputDimension,
                    outputChannels: 1,
                    kernelSize: 7,
                ),
            )
            _decoder.wrappedValue = decoderLayers
        }

        public func callAsFunction(codes: MLXArray) throws -> MLXArray {
            try callAsFunction(codes: codes, inputEmbeddings: nil)
        }

        public func callAsFunction(inputEmbeddings: MLXArray) throws -> MLXArray {
            try callAsFunction(codes: nil, inputEmbeddings: inputEmbeddings)
        }

        public func callAsFunction(
            codes: MLXArray? = nil,
            inputEmbeddings: MLXArray? = nil,
        )
            throws -> MLXArray
        {
            var hidden: MLXArray
            if let inputEmbeddings {
                hidden = inputEmbeddings
            } else if let codes {
                guard codes.dim(1) == config.numQuantizers else {
                    throw VLMError.processing(
                        "Expected \(config.numQuantizers) code layers, got \(codes.dim(1)).",
                    )
                }
                let indices = (codes + codeOffset.asType(codes.dtype)).asType(.int32)
                hidden = codeEmbedding(indices).mean(axis: 1)
            } else {
                throw VLMError.processing("Must provide codes or input embeddings.")
            }

            hidden = preTransformer(inputEmbeddings: hidden)
            hidden = hidden.transposed(0, 2, 1)
            for stage in upsample {
                hidden = stage(hidden)
            }

            var waveform = hidden
            for layer in decoder {
                waveform = layer(waveform)
            }
            return clip(waveform, min: Float32(-1), max: Float32(1))
        }

        public func chunkedDecode(
            codes: MLXArray,
            chunkSize: Int = 300,
            leftContextSize: Int = 25,
        )
            throws -> MLXArray
        {
            let (_, _, length) = (codes.dim(0), codes.dim(1), codes.dim(2))
            var chunks: [MLXArray] = []
            let factor = config.totalUpsampleFactor

            var start = 0
            while start < length {
                let end = min(start + chunkSize, length)
                let contextStart = max(0, start - leftContextSize)
                let chunkCodes = codes[0..., 0..., contextStart ..< end]
                let waveform = try self(codes: chunkCodes)
                let validStart = (start - contextStart) * factor
                let validLength = (end - start) * factor
                chunks.append(waveform[0..., 0..., validStart ..< (validStart + validLength)])
                start += chunkSize
            }

            return concatenated(chunks, axis: -1)
        }

        public func streamDecode(
            codesBuffer: MLXArray,
            chunkSize: Int = 300,
            leftContextSize: Int = 25,
            decodedLength: Int = 0,
        )
            throws -> (MLXArray?, Int)
        {
            let length = codesBuffer.dim(2)
            let start = decodedLength
            let contextStart = max(0, start - leftContextSize)
            let contextLength = start - contextStart
            let newTokens = chunkSize - contextLength
            guard length - start >= newTokens else {
                return (nil, decodedLength)
            }

            let end = start + newTokens
            let chunkCodes = codesBuffer[0..., 0..., contextStart ..< end]
            let waveform = try self(codes: chunkCodes)
            let factor = config.totalUpsampleFactor
            let validStart = contextLength * factor
            let validLength = (end - start) * factor
            return (
                waveform[0..., 0..., validStart ..< (validStart + validLength)],
                end,
            )
        }

        public func flushDecode(
            codesBuffer: MLXArray,
            leftContextSize: Int = 25,
            decodedLength: Int = 0,
        )
            throws -> MLXArray?
        {
            let length = codesBuffer.dim(2)
            guard decodedLength < length else {
                return nil
            }

            let contextStart = max(0, decodedLength - leftContextSize)
            let chunkCodes = codesBuffer[0..., 0..., contextStart...]
            let waveform = try self(codes: chunkCodes)
            let validStart = (decodedLength - contextStart) * config.totalUpsampleFactor
            return waveform[0..., 0..., validStart...]
        }
    }
}

// MARK: - Talker

public enum Qwen3OmniMoETalker {
    public static func sanitizeWeights(
        _ weights: [String: MLXArray],
        config: Qwen3OmniMoEConfiguration.TalkerConfiguration,
    )
        -> [String: MLXArray]
    {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)
        for (key, value) in weights {
            if key.contains("position_ids")
                || key.contains("rotary_emb.inv_freq")
                || key.hasPrefix("talker.model.embed_tokens.")
            {
                continue
            }
            sanitized[key] = value
        }
        for layer in 0 ..< config.textConfiguration.numHiddenLayers {
            let prefix = "talker.model.layers.\(layer).mlp"
            for projection in ["gate_proj", "down_proj", "up_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    var toStack: [MLXArray] = []
                    for expert in 0 ..< config.textConfiguration.numExperts {
                        let key = "\(prefix).experts.\(expert).\(projection).\(suffix)"
                        if let value = sanitized.removeValue(forKey: key) {
                            toStack.append(value)
                        }
                    }
                    if toStack.count == config.textConfiguration.numExperts {
                        sanitized["\(prefix).switch_mlp.\(projection).\(suffix)"] = stacked(toStack)
                    }
                }
            }
        }
        return sanitized
    }

    static func activate(_ x: MLXArray, _ name: String) -> MLXArray {
        switch name {
        case "gelu", "gelu_pytorch_tanh":
            gelu(x)
        case "silu", "swish":
            silu(x)
        default:
            silu(x)
        }
    }

    final class CodePredictorRotaryEmbedding {
        let invFreq: MLXArray

        init(_ config: Qwen3OmniMoEConfiguration.CodePredictorConfiguration) {
            var frequencies = MLXArray(stride(from: 0, to: config.headDim, by: 2)).asType(.float32)
            frequencies = frequencies / Float(max(config.headDim, 1))
            invFreq = 1.0 / pow(MLXArray(config.ropeTheta), frequencies)
        }

        func callAsFunction(_ x: MLXArray, positionIDs: MLXArray) -> (MLXArray, MLXArray) {
            let positions = positionIDs.asType(.float32).expandedDimensions(axis: -1)
            let freqs = positions * invFreq.asType(.float32)[.newAxis, .newAxis, 0...]
            let embeddings = concatenated([freqs, freqs], axis: -1)
            return (cos(embeddings).asType(x.dtype), sin(embeddings).asType(x.dtype))
        }
    }

    static func applyRotaryPositionEmbedding(
        queries: MLXArray,
        keys: MLXArray,
        cos: MLXArray,
        sin: MLXArray,
    )
        -> (MLXArray, MLXArray)
    {
        let cos = cos.expandedDimensions(axis: 1)
        let sin = sin.expandedDimensions(axis: 1)
        let qEmbedded = (queries * cos) + (RotaryEmbeddingOps.rotateHalf(queries) * sin)
        let kEmbedded = (keys * cos) + (RotaryEmbeddingOps.rotateHalf(keys) * sin)
        return (qEmbedded, kEmbedded)
    }

    final class CodePredictorMLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        let activationName: String

        init(_ config: Qwen3OmniMoEConfiguration.CodePredictorConfiguration) {
            activationName = config.hiddenActivation
            _gateProj.wrappedValue = Linear(
                config.hiddenSize,
                config.intermediateSize,
                bias: false,
            )
            _upProj.wrappedValue = Linear(
                config.hiddenSize,
                config.intermediateSize,
                bias: false,
            )
            _downProj.wrappedValue = Linear(
                config.intermediateSize,
                config.hiddenSize,
                bias: false,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(activate(gateProj(x), activationName) * upProj(x))
        }
    }

    final class CodePredictorAttention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let rotaryEmbedding: CodePredictorRotaryEmbedding

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        init(_ config: Qwen3OmniMoEConfiguration.CodePredictorConfiguration) {
            heads = config.numAttentionHeads
            kvHeads = config.numKeyValueHeads
            headDim = config.headDim
            scale = pow(Float(max(config.headDim, 1)), -0.5)
            rotaryEmbedding = CodePredictorRotaryEmbedding(config)

            _qProj.wrappedValue = Linear(
                config.hiddenSize,
                config.numAttentionHeads * config.headDim,
                bias: config.attentionBias,
            )
            _kProj.wrappedValue = Linear(
                config.hiddenSize,
                config.numKeyValueHeads * config.headDim,
                bias: config.attentionBias,
            )
            _vProj.wrappedValue = Linear(
                config.hiddenSize,
                config.numKeyValueHeads * config.headDim,
                bias: config.attentionBias,
            )
            _oProj.wrappedValue = Linear(
                config.numAttentionHeads * config.headDim,
                config.hiddenSize,
                bias: config.attentionBias,
            )
            _qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
            positionIDs: MLXArray?,
            positionEmbeddings: (MLXArray, MLXArray)?,
        )
            -> MLXArray
        {
            let batchSize = hiddenStates.dim(0)
            let sequenceLength = hiddenStates.dim(1)

            var queries = qProj(hiddenStates)
                .reshaped(batchSize, sequenceLength, heads, headDim)
            queries = qNorm(queries).transposed(0, 2, 1, 3)

            var keys = kProj(hiddenStates)
                .reshaped(batchSize, sequenceLength, kvHeads, headDim)
            keys = kNorm(keys).transposed(0, 2, 1, 3)

            let values = vProj(hiddenStates)
                .reshaped(batchSize, sequenceLength, kvHeads, headDim)
                .transposed(0, 2, 1, 3)

            let embeddings: (MLXArray, MLXArray)
            if let positionEmbeddings {
                embeddings = positionEmbeddings
            } else {
                let ids =
                    positionIDs
                    ?? broadcast(
                        MLXArray(0 ..< sequenceLength).asType(.int32)[.newAxis, 0...],
                        to: [batchSize, sequenceLength],
                    )
                embeddings = rotaryEmbedding(hiddenStates, positionIDs: ids)
            }

            (queries, keys) = applyRotaryPositionEmbedding(
                queries: queries,
                keys: keys,
                cos: embeddings.0,
                sin: embeddings.1,
            )

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batchSize, sequenceLength, -1)

            return oProj(output)
        }
    }

    final class CodePredictorDecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var selfAttention: CodePredictorAttention
        @ModuleInfo var mlp: CodePredictorMLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        let attentionType: String

        init(_ config: Qwen3OmniMoEConfiguration.CodePredictorConfiguration, index: Int) {
            attentionType =
                index >= 0 && index < config.layerTypes.count
                ? config.layerTypes[index]
                : "full_attention"
            _selfAttention.wrappedValue = CodePredictorAttention(config)
            _mlp.wrappedValue = CodePredictorMLP(config)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: config.rmsNormEps,
            )
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
            positionIDs: MLXArray?,
            positionEmbeddings: (MLXArray, MLXArray)?,
        )
            -> MLXArray
        {
            var residual = hiddenStates
            var hidden = inputLayerNorm(hiddenStates)
            hidden = selfAttention(
                hidden,
                mask: mask,
                cache: cache,
                positionIDs: positionIDs,
                positionEmbeddings: positionEmbeddings,
            )
            hidden = residual + hidden

            residual = hidden
            hidden = postAttentionLayerNorm(hidden)
            hidden = mlp(hidden)
            return residual + hidden
        }
    }

    public final class CodePredictorModel: Module {
        public let config: Qwen3OmniMoEConfiguration.CodePredictorConfiguration
        let rotaryEmbedding: CodePredictorRotaryEmbedding

        @ModuleInfo(key: "layers") var layers: [CodePredictorDecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm
        @ModuleInfo(key: "codec_embedding") var codecEmbedding: [Embedding]

        public init(_ config: Qwen3OmniMoEConfiguration.CodePredictorConfiguration) {
            self.config = config
            rotaryEmbedding = CodePredictorRotaryEmbedding(config)
            _layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
                CodePredictorDecoderLayer(config, index: $0)
            }
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _codecEmbedding.wrappedValue = (0 ..< max(config.numCodeGroups - 1, 0)).map { _ in
                Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
            }
        }

        func callAsFunction(
            inputEmbeddings: MLXArray,
            cache: [KVCache]? = nil,
            attentionMask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
            positionIDs providedPositionIDs: MLXArray? = nil,
        )
            -> MLXArray
        {
            var hidden = inputEmbeddings
            let batchSize = hidden.dim(0)
            let sequenceLength = hidden.dim(1)

            let positionIDs: MLXArray
            if let providedPositionIDs {
                positionIDs = providedPositionIDs
            } else {
                let offset = cache?.first?.offset ?? 0
                let base = MLXArray(offset ..< (offset + sequenceLength)).asType(.int32)
                positionIDs = broadcast(base[.newAxis, 0...], to: [batchSize, sequenceLength])
            }

            let mask = attentionMask ?? createAttentionMask(h: hidden, cache: cache?.first)
            let positionEmbeddings = rotaryEmbedding(hidden, positionIDs: positionIDs)

            for (index, layer) in layers.enumerated() {
                hidden = layer(
                    hidden,
                    mask: mask,
                    cache: cache?[index],
                    positionIDs: positionIDs,
                    positionEmbeddings: positionEmbeddings,
                )
            }
            return norm(hidden)
        }
    }

    public final class CodePredictor: Module {
        public let config: Qwen3OmniMoEConfiguration.CodePredictorConfiguration

        @ModuleInfo var model: CodePredictorModel
        @ModuleInfo(key: "lm_head") var lmHead: [Linear]

        public init(_ config: Qwen3OmniMoEConfiguration.CodePredictorConfiguration) {
            self.config = config
            _model.wrappedValue = CodePredictorModel(config)
            _lmHead.wrappedValue = (0 ..< max(config.numCodeGroups - 1, 0)).map { _ in
                Linear(config.hiddenSize, config.vocabSize, bias: false)
            }
        }

        public func callAsFunction(
            inputIDs: MLXArray? = nil,
            inputEmbeddings providedInputEmbeddings: MLXArray? = nil,
            cache: [KVCache]? = nil,
            attentionMask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
            positionIDs: MLXArray? = nil,
            generationStep providedGenerationStep: Int? = nil,
        )
            throws -> (logits: MLXArray, hiddenStates: MLXArray, inputEmbeddings: MLXArray)
        {
            var generationStep = providedGenerationStep
            var inputEmbeddings = providedInputEmbeddings

            if let inputEmbeddings, inputEmbeddings.dim(1) > 1, generationStep == nil {
                generationStep = inputEmbeddings.dim(1) - 2
            } else if let inputIDs, let generationStep {
                let embeddingIndex = max(0, min(generationStep - 1, model.codecEmbedding.count - 1))
                inputEmbeddings = model.codecEmbedding[embeddingIndex](inputIDs)
            }

            guard let inputEmbeddings else {
                throw VLMError.processing("Qwen3-Omni code predictor requires input embeddings.")
            }
            guard !lmHead.isEmpty else {
                throw VLMError.processing("Qwen3-Omni code predictor has no codec heads.")
            }

            let headIndex = max(0, min(generationStep ?? 0, lmHead.count - 1))
            let hiddenStates = model(
                inputEmbeddings: inputEmbeddings,
                cache: cache,
                attentionMask: attentionMask,
                positionIDs: positionIDs,
            )
            let logits = lmHead[headIndex](hiddenStates)
            return (logits, hiddenStates, inputEmbeddings)
        }
    }

    final class ResizeMLP: Module, UnaryLayer {
        @ModuleInfo(key: "linear_fc1") var linear1: Linear
        @ModuleInfo(key: "linear_fc2") var linear2: Linear

        init(_ config: Qwen3OmniMoEConfiguration.TalkerConfiguration) {
            _linear1.wrappedValue = Linear(
                config.thinkerHiddenSize,
                config.textConfiguration.intermediateSize,
                bias: true,
            )
            _linear2.wrappedValue = Linear(
                config.textConfiguration.intermediateSize,
                config.textConfiguration.hiddenSize,
                bias: true,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear2(silu(linear1(x)))
        }
    }

    final class TalkerTextMLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        let activationName: String

        init(_ config: Qwen3OmniMoEConfiguration.TextConfiguration, intermediateSize: Int) {
            activationName = config.hiddenAct
            _gateProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
            _upProj.wrappedValue = Linear(config.hiddenSize, intermediateSize, bias: false)
            _downProj.wrappedValue = Linear(intermediateSize, config.hiddenSize, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(activate(gateProj(x), activationName) * upProj(x))
        }
    }

    final class TalkerSparseMoeBlock: Module, UnaryLayer {
        let topK: Int
        let normTopKProb: Bool

        @ModuleInfo(key: "gate") var gate: Linear
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
        @ModuleInfo(key: "shared_expert") var sharedExpert: TalkerTextMLP
        @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

        init(_ config: Qwen3OmniMoEConfiguration.TextConfiguration) {
            topK = config.numExpertsPerToken
            normTopKProb = config.normTopKProb
            _gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.numExperts,
                activation: silu,
                bias: false,
            )
            _sharedExpert.wrappedValue = TalkerTextMLP(
                config,
                intermediateSize: config.sharedExpertIntermediateSize,
            )
            _sharedExpertGate.wrappedValue = Linear(config.hiddenSize, 1, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let logits = gate(x)
            let softGates = softmax(logits, axis: -1, precise: true)
            let indices = argPartition(-logits, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
            var scores = takeAlong(softGates, indices, axis: -1)
            if normTopKProb {
                scores = scores / scores.sum(axis: -1, keepDims: true)
            }
            let routed = switchMLP(x, indices)
            let sparse = (routed * scores[.ellipsis, .newAxis]).sum(axis: -2)
            return sparse + (sigmoid(sharedExpertGate(x)) * sharedExpert(x))
        }
    }

    final class TalkerDecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Qwen3OmniMoELanguage.Attention
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
        @ModuleInfo var mlp: TalkerSparseMoeBlock

        init(_ config: Qwen3OmniMoEConfiguration.TextConfiguration) {
            _attention.wrappedValue = Qwen3OmniMoELanguage.Attention(config)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: Float(config.rmsNormEps),
            )
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize,
                eps: Float(config.rmsNormEps),
            )
            _mlp.wrappedValue = TalkerSparseMoeBlock(config)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIDs: MLXArray?,
        )
            -> MLXArray
        {
            var residual = hiddenStates
            var hidden = inputLayerNorm(hiddenStates)
            hidden = attention(hidden, mask: mask, cache: cache, positionIDs: positionIDs)
            hidden = residual + hidden

            residual = hidden
            hidden = postAttentionLayerNorm(hidden)
            hidden = mlp(hidden)
            return residual + hidden
        }
    }

    public final class TalkerModel: Module {
        public let config: Qwen3OmniMoEConfiguration.TextConfiguration

        @ModuleInfo(key: "layers") var layers: [TalkerDecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm
        @ModuleInfo(key: "codec_embedding") var codecEmbedding: Embedding

        public init(_ config: Qwen3OmniMoEConfiguration.TextConfiguration) {
            self.config = config
            _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
                TalkerDecoderLayer(config)
            }
            _norm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps))
            _codecEmbedding.wrappedValue = Embedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize,
            )
        }

        public func callAsFunction(
            inputIDs: MLXArray? = nil,
            inputEmbeddings: MLXArray? = nil,
            cache: [KVCache]? = nil,
            attentionMask: MLXArray? = nil,
            positionIDs providedPositionIDs: MLXArray? = nil,
            visualMask: MLXArray? = nil,
            deepstackEmbeds: [MLXArray]? = nil,
        )
            -> MLXArray
        {
            var hidden: MLXArray
            if let inputEmbeddings {
                hidden = inputEmbeddings
            } else if let inputIDs {
                hidden = codecEmbedding(inputIDs)
            } else {
                fatalError("Qwen3-Omni talker requires input ids or embeddings.")
            }

            let positionIDs = normalizedPositionIDs(
                providedPositionIDs,
                batchSize: hidden.dim(0),
                sequenceLength: hidden.dim(1),
                cache: cache,
            )
            let mask = attentionMask ?? createAttentionMask(h: hidden, cache: cache)

            for (index, layer) in layers.enumerated() {
                hidden = layer(
                    hidden,
                    mask: mask,
                    cache: cache?[index],
                    positionIDs: positionIDs,
                )

                if let deepstackEmbeds, index < deepstackEmbeds.count, let visualMask {
                    hidden = applyDeepstack(
                        hiddenStates: hidden,
                        visualMask: visualMask,
                        visualEmbeds: deepstackEmbeds[index],
                    )
                }
            }

            return norm(hidden)
        }

        private func normalizedPositionIDs(
            _ positionIDs: MLXArray?,
            batchSize: Int,
            sequenceLength: Int,
            cache: [KVCache]?,
        )
            -> MLXArray
        {
            if let positionIDs {
                if positionIDs.ndim == 2 {
                    return broadcast(
                        positionIDs[.newAxis, 0..., 0...],
                        to: [3, positionIDs.dim(0), positionIDs.dim(1)],
                    )
                }
                return positionIDs
            }

            let offset = cache?.first?.offset ?? 0
            let base = MLXArray(offset ..< (offset + sequenceLength)).asType(.int32)
            let batchIDs = broadcast(base[.newAxis, 0...], to: [batchSize, sequenceLength])
            return broadcast(batchIDs[.newAxis, 0..., 0...], to: [3, batchSize, sequenceLength])
        }

        private func applyDeepstack(
            hiddenStates: MLXArray,
            visualMask: MLXArray,
            visualEmbeds: MLXArray,
        )
            -> MLXArray
        {
            let bools = visualMask.asType(.bool).asArray(Bool.self)
            var indices: [Int] = []
            indices.reserveCapacity(bools.count)
            for (idx, value) in bools.enumerated() where value {
                indices.append(idx)
            }
            guard !indices.isEmpty else {
                return hiddenStates
            }

            let indexArray = MLXArray(indices.map { UInt32($0) })
            let result = hiddenStates
            result[0..., indexArray, 0...] = result[0..., indexArray, 0...] + visualEmbeds
            return result
        }
    }

    public final class Talker: Module, KVCacheDimensionProvider {
        public struct CodeGenerator {
            private let talker: Talker
            private let trailingTextHidden: MLXArray
            private let ttsPadEmbedding: MLXArray
            private let sampler: any LogitSampler
            private let residualSampler: any LogitSampler
            private var cache: [KVCache]
            private var token: MLXArray
            private var pastHidden: MLXArray
            private var generationStep = 0
            private var remainingTokens: Int

            fileprivate init(
                talker: Talker,
                inputEmbeddings: MLXArray,
                trailingTextHidden: MLXArray,
                ttsPadEmbedding: MLXArray,
                maxNewTokens: Int,
                sampler: any LogitSampler,
                residualSampler: any LogitSampler,
            ) {
                self.talker = talker
                self.trailingTextHidden = trailingTextHidden
                self.ttsPadEmbedding = ttsPadEmbedding
                self.sampler = sampler
                self.residualSampler = residualSampler
                cache = talker.model.layers.map { _ in KVCacheSimple() }

                let output = talker(inputEmbeddings: inputEmbeddings, cache: cache)
                token = sampler.sample(logits: output.logits[0..., -1, 0...])
                pastHidden =
                    output.hiddenStates[
                        0..., (output.hiddenStates.dim(1) - 1)..., 0...,
                    ]
                remainingTokens = max(maxNewTokens, 0)
            }

            public mutating func next() throws -> MLXArray? {
                try Task.checkCancellation()
                guard remainingTokens > 0 else {
                    return nil
                }
                if token.asArray(Int.self).allSatisfy({ $0 == talker.config.codecEOSTokenID }) {
                    return nil
                }

                let prepared = try talker.predictResidualCodes(
                    firstCodeIDs: token[0..., .newAxis],
                    pastHidden: pastHidden,
                    trailingTextHidden: trailingTextHidden,
                    ttsPadEmbedding: ttsPadEmbedding,
                    generationStep: generationStep,
                    sampler: residualSampler,
                )
                let output = talker(inputEmbeddings: prepared.inputEmbeddings, cache: cache)
                pastHidden =
                    output.hiddenStates[
                        0..., (output.hiddenStates.dim(1) - 1)..., 0...,
                    ]
                token = sampler.sample(logits: output.logits[0..., -1, 0...])
                generationStep += 1
                remainingTokens -= 1
                return prepared.residualCodes
            }
        }

        public let config: Qwen3OmniMoEConfiguration.TalkerConfiguration
        public let kvHeads: [Int]

        @ModuleInfo var model: TalkerModel
        @ModuleInfo(key: "text_projection") var textProjection: ResizeMLP
        @ModuleInfo(key: "hidden_projection") var hiddenProjection: ResizeMLP
        @ModuleInfo(key: "code_predictor") var codePredictor: CodePredictor
        @ModuleInfo(key: "codec_head") var codecHead: Linear

        public init(_ config: Qwen3OmniMoEConfiguration.TalkerConfiguration) {
            self.config = config
            kvHeads = Array(
                repeating: config.textConfiguration.numKeyValueHeads,
                count: config.textConfiguration.numHiddenLayers,
            )
            _model.wrappedValue = TalkerModel(config.textConfiguration)
            _textProjection.wrappedValue = ResizeMLP(config)
            _hiddenProjection.wrappedValue = ResizeMLP(config)
            _codePredictor.wrappedValue = CodePredictor(config.codePredictorConfiguration)
            _codecHead.wrappedValue = Linear(
                config.textConfiguration.hiddenSize,
                config.textConfiguration.vocabSize,
                bias: false,
            )
        }

        public func projectText(_ hiddenStates: MLXArray) -> MLXArray {
            textProjection(hiddenStates)
        }

        public func projectHidden(_ hiddenStates: MLXArray) -> MLXArray {
            hiddenProjection(hiddenStates)
        }

        public func callAsFunction(
            inputIDs: MLXArray? = nil,
            inputEmbeddings: MLXArray? = nil,
            cache: [KVCache]? = nil,
            attentionMask: MLXArray? = nil,
            positionIDs: MLXArray? = nil,
            visualMask: MLXArray? = nil,
            deepstackEmbeds: [MLXArray]? = nil,
        )
            -> (logits: MLXArray, hiddenStates: MLXArray)
        {
            let hiddenStates = model(
                inputIDs: inputIDs,
                inputEmbeddings: inputEmbeddings,
                cache: cache,
                attentionMask: attentionMask,
                positionIDs: positionIDs,
                visualMask: visualMask,
                deepstackEmbeds: deepstackEmbeds,
            )
            return (codecHead(hiddenStates), hiddenStates)
        }

        public func predictResidualCodes(
            firstCodeIDs: MLXArray,
            pastHidden: MLXArray,
            trailingTextHidden: MLXArray?,
            ttsPadEmbedding: MLXArray?,
            generationStep: Int,
            sampler: any LogitSampler = ArgMaxSampler(),
        )
            throws -> (inputEmbeddings: MLXArray, residualCodes: MLXArray)
        {
            let lastIDHidden = model.codecEmbedding(firstCodeIDs)
            let predictorEmbeds = concatenated([pastHidden, lastIDHidden], axis: 1)
            let predictorCache: [KVCache] = codePredictor.model.layers.map { _ in
                KVCacheSimple()
            }

            var prediction = try codePredictor(
                inputEmbeddings: predictorEmbeds,
                cache: predictorCache,
            )
            var nextToken = sampler.sample(logits: prediction.logits[0..., -1, 0...])
            var currentCodes = [firstCodeIDs, nextToken[0..., .newAxis]]
            var residualEmbeds: [MLXArray] = []

            for codeStep in 1 ..< max(config.numCodeGroups - 1, 1) {
                prediction = try codePredictor(
                    inputIDs: nextToken[0..., .newAxis],
                    cache: predictorCache,
                    generationStep: codeStep,
                )
                residualEmbeds.append(prediction.inputEmbeddings)
                nextToken = sampler.sample(logits: prediction.logits[0..., -1, 0...])
                currentCodes.append(nextToken[0..., .newAxis])
            }

            guard let lastEmbedding = codePredictor.model.codecEmbedding.last else {
                throw VLMError.processing("Qwen3-Omni code predictor has no residual embeddings.")
            }
            let lastResidualHidden = lastEmbedding(nextToken[0..., .newAxis])
            let codecHiddens = [lastIDHidden] + residualEmbeds + [lastResidualHidden]
            var nextInputEmbeddings = concatenated(codecHiddens, axis: 1).sum(
                axis: 1, keepDims: true)

            if let trailingTextHidden, generationStep < trailingTextHidden.dim(1) {
                nextInputEmbeddings =
                    nextInputEmbeddings
                    + trailingTextHidden[0..., generationStep, 0...].expandedDimensions(axis: 1)
            } else if let ttsPadEmbedding {
                nextInputEmbeddings = nextInputEmbeddings + ttsPadEmbedding
            }

            return (nextInputEmbeddings, concatenated(currentCodes, axis: 1))
        }

        public func makeCodeGenerator(
            inputEmbeddings: MLXArray,
            trailingTextHidden: MLXArray,
            ttsPadEmbedding: MLXArray,
            maxNewTokens: Int = 2048,
            sampler: any LogitSampler = ArgMaxSampler(),
            residualSampler: (any LogitSampler)? = nil,
        )
            -> CodeGenerator
        {
            CodeGenerator(
                talker: self,
                inputEmbeddings: inputEmbeddings,
                trailingTextHidden: trailingTextHidden,
                ttsPadEmbedding: ttsPadEmbedding,
                maxNewTokens: maxNewTokens,
                sampler: sampler,
                residualSampler: residualSampler ?? sampler,
            )
        }

        public func generateCodeChunks(
            inputEmbeddings: MLXArray,
            trailingTextHidden: MLXArray,
            ttsPadEmbedding: MLXArray,
            maxNewTokens: Int = 2048,
            sampler: any LogitSampler = ArgMaxSampler(),
            residualSampler: (any LogitSampler)? = nil,
        )
            throws -> [MLXArray]
        {
            var generator = makeCodeGenerator(
                inputEmbeddings: inputEmbeddings,
                trailingTextHidden: trailingTextHidden,
                ttsPadEmbedding: ttsPadEmbedding,
                maxNewTokens: maxNewTokens,
                sampler: sampler,
                residualSampler: residualSampler,
            )
            var chunks: [MLXArray] = []
            chunks.reserveCapacity(max(maxNewTokens, 0))
            while let codes = try generator.next() {
                chunks.append(codes)
            }
            return chunks
        }

        public func generateCodes(
            inputEmbeddings: MLXArray,
            trailingTextHidden: MLXArray,
            ttsPadEmbedding: MLXArray,
            maxNewTokens: Int = 2048,
            sampler: any LogitSampler = ArgMaxSampler(),
            residualSampler: (any LogitSampler)? = nil,
        )
            throws -> MLXArray
        {
            let chunks = try generateCodeChunks(
                inputEmbeddings: inputEmbeddings,
                trailingTextHidden: trailingTextHidden,
                ttsPadEmbedding: ttsPadEmbedding,
                maxNewTokens: maxNewTokens,
                sampler: sampler,
                residualSampler: residualSampler,
            )
            guard !chunks.isEmpty else {
                return MLXArray.zeros(
                    [inputEmbeddings.dim(0), config.numCodeGroups, 0],
                    dtype: .int32,
                )
            }
            return stacked(chunks, axis: 1).transposed(0, 2, 1)
        }
    }
}

// MARK: - Language

enum Qwen3OmniMoELanguage {
    final class RotaryEmbedding {
        private let invFreq: MLXArray
        private let mropeSection: [Int]

        init(headDim: Int, base: Double, ropeScaling: Qwen3OmniMoEConfiguration.RoPEScaling?) {
            var freq = MLXArray(stride(from: 0, to: headDim, by: 2)).asType(.float32)
            freq = freq / Float(headDim)
            let baseArray = MLXArray(Float(base))
            invFreq = 1.0 / pow(baseArray, freq)
            mropeSection = ropeScaling?.mropeSection ?? [24, 20, 20]
        }

        private func applyInterleavedMRope(_ freqs: MLXArray) -> MLXArray {
            let freqs_t = freqs[0, 0..., 0..., 0...]  // (bs, seq_len, head_dim // 2)

            let dims = freqs_t.dim(-1)
            var slices: [MLXArray] = []

            for idx in 0 ..< dims {
                var slice = freqs_t[0..., 0..., idx]

                for (dimIndex, offset) in [(1, 1), (2, 2)] {
                    let end = min(mropeSection[dimIndex] * 3, dims)
                    if idx >= offset, idx < end, (idx - offset) % 3 == 0 {
                        slice = freqs[dimIndex, 0..., 0..., idx]
                        break
                    }
                }

                slices.append(slice)
            }

            return stacked(slices, axis: -1)
        }

        func callAsFunction(positionIDs: MLXArray, dtype: MLX.DType) -> (MLXArray, MLXArray) {
            var positionIDs = positionIDs
            if positionIDs.ndim == 2 {
                positionIDs = positionIDs[.newAxis, 0..., 0...]
                positionIDs = tiled(positionIDs, repetitions: [3, 1, 1])
            }

            let pos = positionIDs.asType(.float32)
            var invFreq = invFreq.asType(.float32)
            invFreq = invFreq[.newAxis, .newAxis, .newAxis, 0...]
            var freqs = pos[0..., 0..., 0..., .newAxis] * invFreq
            freqs = applyInterleavedMRope(freqs)

            let emb = concatenated([freqs, freqs], axis: -1)
            let cosValues = cos(emb).asType(dtype)
            let sinValues = sin(emb).asType(dtype)
            return (cosValues, sinValues)
        }
    }

    static func applyMultimodalRotary(
        q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray,
    )
        -> (MLXArray, MLXArray)
    {
        var cos = cos
        var sin = sin
        cos = expandedDimensions(cos, axis: 1)
        sin = expandedDimensions(sin, axis: 1)
        let qEmbedded = (q * cos) + (RotaryEmbeddingOps.rotateHalf(q) * sin)
        let kEmbedded = (k * cos) + (RotaryEmbeddingOps.rotateHalf(k) * sin)
        return (qEmbedded, kEmbedded)
    }

    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "o_proj") var wo: Linear

        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        let rotaryEmbedding: RotaryEmbedding

        init(_ config: Qwen3OmniMoEConfiguration.TextConfiguration) {
            let dim = config.hiddenSize
            heads = config.numAttentionHeads
            kvHeads = config.numKeyValueHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)

            _wq.wrappedValue = Linear(dim, heads * headDim, bias: false)
            _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: false)
            _wo.wrappedValue = Linear(heads * headDim, dim, bias: false)

            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: Float(config.rmsNormEps))

            rotaryEmbedding = RotaryEmbedding(
                headDim: headDim,
                base: config.ropeTheta,
                ropeScaling: config.ropeScaling,
            )
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIDs: MLXArray?,
        )
            -> MLXArray
        {
            let (batch, length) = (x.dim(0), x.dim(1))

            var queries = wq(x)
            var keys = wk(x)
            var values = wv(x)

            queries = queries.reshaped(batch, length, heads, headDim)
            queries = qNorm(queries).transposed(0, 2, 1, 3)

            keys = keys.reshaped(batch, length, kvHeads, headDim)
            keys = kNorm(keys).transposed(0, 2, 1, 3)

            values = values.reshaped(batch, length, kvHeads, headDim).transposed(0, 2, 1, 3)

            var kvSequenceLength = keys.dim(-2)
            var positionIDs = positionIDs

            if positionIDs == nil {
                let offset = cache?.offset ?? 0
                kvSequenceLength += offset + 1
                var base = MLXArray(stride(from: offset, to: offset + length, by: 1)).asType(.int32)
                base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
                positionIDs = base[.newAxis, 0..., 0...]
                positionIDs = tiled(positionIDs!, repetitions: [3, 1, 1])
            } else {
                if let cache {
                    kvSequenceLength += cache.offset + 1
                }
            }

            let (cosValues, sinValues) = rotaryEmbedding(positionIDs: positionIDs!, dtype: x.dtype)

            (queries, keys) = Qwen3OmniMoELanguage.applyMultimodalRotary(
                q: queries, k: keys, cos: cosValues, sin: sinValues,
            )

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                let slicedMask = mask[.ellipsis, 0 ..< kvSequenceLength]
                attentionMask = .array(slicedMask)
            } else {
                attentionMask = .none
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: attentionMask,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)

            return wo(output)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    final class SparseMoeBlock: Module, UnaryLayer {
        let topK: Int
        let normTopKProb: Bool

        @ModuleInfo(key: "gate") var gate: Linear
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

        init(_ config: Qwen3OmniMoEConfiguration.TextConfiguration) {
            topK = config.numExpertsPerToken
            normTopKProb = config.normTopKProb
            _gate.wrappedValue = Linear(config.hiddenSize, config.numExperts, bias: false)
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.moeIntermediateSize,
                numExperts: config.numExperts,
                activation: silu,
                bias: false,
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let logits = gate(x)
            let softGates = softmax(logits, axis: -1, precise: true)
            let indices = argPartition(-logits, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
            var scores = takeAlong(softGates, indices, axis: -1)
            if normTopKProb {
                scores = scores / scores.sum(axis: -1, keepDims: true)
            }
            let output = switchMLP(x, indices)
            return (output * scores[.ellipsis, .newAxis]).sum(axis: -2)
        }
    }

    final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        let mlp: UnaryLayer

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: Qwen3OmniMoEConfiguration.TextConfiguration, layerIndex: Int) {
            _attention.wrappedValue = Attention(config)
            if !config.mlpOnlyLayers.contains(layerIndex),
                config.numExperts > 0,
                (layerIndex + 1) % config.decoderSparseStep == 0
            {
                mlp = SparseMoeBlock(config)
            } else {
                mlp = MLP(
                    dimensions: config.hiddenSize,
                    hiddenDimensions: config.intermediateSize,
                )
            }
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps),
            )
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps),
            )
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIDs: MLXArray?,
        )
            -> MLXArray
        {
            var residual = attention(
                inputLayerNorm(x), mask: mask, cache: cache, positionIDs: positionIDs,
            )
            let hidden = x + residual
            residual = mlp(postAttentionLayerNorm(hidden))
            return hidden + residual
        }
    }

    final class Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm

        init(_ config: Qwen3OmniMoEConfiguration.TextConfiguration) {
            precondition(config.vocabSize > 0)
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabSize,
                dimensions: config.hiddenSize,
            )
            _layers.wrappedValue = (0 ..< config.numHiddenLayers).map {
                DecoderLayer(config, layerIndex: $0)
            }
            _norm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: Float(config.rmsNormEps),
            )
        }

        func callAsFunction(
            _ inputIDs: MLXArray?,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            positionIDs: MLXArray?,
            visualMask: MLXArray?,
            deepstackEmbeds: [MLXArray]?,
        )
            -> MLXArray
        {
            var hidden: MLXArray
            if let inputEmbeddings {
                hidden = inputEmbeddings
            } else if let inputIDs {
                hidden = embedTokens(inputIDs)
            } else {
                fatalError("Either input ids or embeddings must be provided")
            }

            var mask = mask
            if mask == nil {
                mask = createAttentionMask(h: hidden, cache: cache)
            }

            for (index, layer) in layers.enumerated() {
                let layerCache = cache?[index]
                hidden = layer(hidden, mask: mask, cache: layerCache, positionIDs: positionIDs)

                if let embeds = deepstackEmbeds, index < embeds.count,
                    let visualMask
                {
                    hidden = applyDeepstack(
                        hiddenStates: hidden,
                        visualMask: visualMask,
                        visualEmbeds: embeds[index],
                    )
                }
            }

            return norm(hidden)
        }

        func hiddenStates(
            _ inputIDs: MLXArray?,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            positionIDs: MLXArray?,
            visualMask: MLXArray?,
            deepstackEmbeds: [MLXArray]?,
            targetLayerIndex: Int,
        )
            throws -> MLXArray
        {
            guard targetLayerIndex >= 0, targetLayerIndex < layers.count else {
                throw VLMError.processing(
                    "Qwen3-Omni thinker hidden layer \(targetLayerIndex) is outside the configured layer range.",
                )
            }

            var hidden: MLXArray
            if let inputEmbeddings {
                hidden = inputEmbeddings
            } else if let inputIDs {
                hidden = embedTokens(inputIDs)
            } else {
                throw VLMError.processing("Either input ids or embeddings must be provided.")
            }

            var mask = mask
            if mask == nil {
                mask = createAttentionMask(h: hidden, cache: cache)
            }

            for (index, layer) in layers.enumerated() {
                let layerCache = cache?[index]
                hidden = layer(hidden, mask: mask, cache: layerCache, positionIDs: positionIDs)

                if let embeds = deepstackEmbeds, index < embeds.count,
                    let visualMask
                {
                    hidden = applyDeepstack(
                        hiddenStates: hidden,
                        visualMask: visualMask,
                        visualEmbeds: embeds[index],
                    )
                }

                if index == targetLayerIndex {
                    return hidden
                }
            }

            return hidden
        }

        private func applyDeepstack(
            hiddenStates: MLXArray,
            visualMask: MLXArray,
            visualEmbeds: MLXArray,
        )
            -> MLXArray
        {
            let indices = maskIndices(visualMask)
            guard !indices.isEmpty else {
                return hiddenStates
            }

            let indexArray = MLXArray(indices.map { UInt32($0) })

            let result = hiddenStates
            result[0..., indexArray, 0...] = result[0..., indexArray, 0...] + visualEmbeds

            return result
        }

        private func maskIndices(_ mask: MLXArray) -> [Int] {
            let bools = mask.asType(.bool).asArray(Bool.self)
            var indices: [Int] = []
            indices.reserveCapacity(bools.count)
            for (idx, value) in bools.enumerated() where value {
                indices.append(idx)
            }
            return indices
        }
    }

    final class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        let config: Qwen3OmniMoEConfiguration
        let textConfig: Qwen3OmniMoEConfiguration.TextConfiguration
        var kvHeads: [Int]

        private var ropeDeltas: MLXArray?

        init(_ config: Qwen3OmniMoEConfiguration) {
            self.config = config
            textConfig = config.textConfiguration
            model = Model(config.textConfiguration)
            kvHeads = Array(
                repeating: config.textConfiguration.numKeyValueHeads,
                count: config.textConfiguration.numHiddenLayers,
            )

            if !config.textConfiguration.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.textConfiguration.hiddenSize,
                    config.textConfiguration.vocabSize,
                    bias: false,
                )
            }
        }

        func callAsFunction(
            _ inputIDs: MLXArray?,
            cache: [KVCache]?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            positionIDs providedPositionIDs: MLXArray?,
            visualMask: MLXArray?,
            deepstackEmbeds: [MLXArray]?,
            pixelValues: MLXArray?,
            imageGridTHW: [THW]?,
            videoGridTHW: [THW]?,
        )
            -> LMOutput
        {
            if pixelValues != nil {
                ropeDeltas = nil
            }

            var positionIDs = providedPositionIDs

            if positionIDs == nil, mask == nil || mask?.ndim == 2 {
                if (cache?.first?.offset ?? 0) == 0 || ropeDeltas == nil || cache == nil {
                    if let inputIDs {
                        let (computed, deltas) = Qwen3OmniMoELanguage.getRopeIndex(
                            inputIDs: inputIDs,
                            imageGridTHW: imageGridTHW,
                            videoGridTHW: videoGridTHW,
                            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
                            imageTokenID: config.imageTokenIndex,
                            videoTokenID: config.videoTokenIndex,
                            visionStartTokenID: config.visionStartTokenID,
                            attentionMask: mask,
                        )

                        positionIDs = computed
                        ropeDeltas = deltas
                    } else if let cache, ropeDeltas == nil {
                        let batch = inputEmbeddings!.dim(0)
                        let seqLength = inputEmbeddings!.dim(1)
                        let currentOffset = cache.first?.offset ?? 0

                        var base = MLXArray(0 ..< seqLength).asType(.int32)
                        base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
                        let offsetValue = MLXArray(currentOffset).asType(.int32)
                        base = base + offsetValue

                        positionIDs = base[.newAxis, 0..., 0...]
                        positionIDs = tiled(positionIDs!, repetitions: [3, batch, seqLength])
                    }
                } else if let cache, let ropeDeltas {
                    let batch = (inputIDs ?? inputEmbeddings!).dim(0)
                    let seqLength = (inputIDs ?? inputEmbeddings!).dim(1)

                    let lastCacheOffset = cache.last?.offset ?? 0

                    var delta = MLXArray(lastCacheOffset).asType(.int32) + ropeDeltas.asType(.int32)

                    var base = MLXArray(0 ..< seqLength).asType(.int32)
                    base = base[.newAxis, 0...]
                    base = broadcast(base, to: [batch, seqLength])

                    if delta.dim(0) == 1, batch > 1 {
                        delta = repeated(delta, count: batch, axis: 0)
                    }

                    base = base + delta

                    positionIDs = base[.newAxis, 0..., 0...]
                    positionIDs = broadcast(positionIDs!, to: [3, batch, seqLength])
                }
            }

            var output = model(
                inputIDs,
                cache: cache,
                inputEmbeddings: inputEmbeddings,
                mask: nil,
                positionIDs: positionIDs,
                visualMask: visualMask,
                deepstackEmbeds: deepstackEmbeds,
            )

            if let lmHead {
                output = lmHead(output)
            } else {
                output = model.embedTokens.asLinear(output)
            }

            return LMOutput(logits: output)
        }
    }
}

extension Qwen3OmniMoELanguage {
    static func getRopeIndex(
        inputIDs: MLXArray,
        imageGridTHW: [THW]?,
        videoGridTHW: [THW]?,
        spatialMergeSize: Int,
        imageTokenID: Int,
        videoTokenID: Int,
        visionStartTokenID: Int,
        attentionMask: MLXArray? = nil,
    )
        -> (MLXArray, MLXArray)
    {
        let (batchSize, seqLength) = (inputIDs.dim(0), inputIDs.dim(1))

        var positionIDs = MLXArray(0 ..< seqLength).asType(.int32)
        positionIDs = broadcast(positionIDs[.newAxis, 0...], to: [batchSize, seqLength])

        guard inputIDs.ndim > 0, imageGridTHW != nil || videoGridTHW != nil else {
            let positionIds3D = broadcast(
                positionIDs[.newAxis, 0..., 0...], to: [3, batchSize, seqLength],
            )
            let zeros = MLXArray.zeros([batchSize], dtype: .int32)
            return (positionIds3D, zeros)
        }

        positionIDs = ones(like: inputIDs).asType(.int32)
        positionIDs = broadcast(positionIDs[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])

        var mropePositionDeltas: [Int] = []
        let mask = attentionMask ?? ones(like: inputIDs)

        // Process each batch item (assume batch=1 for now)
        for batchIdx in 0 ..< batchSize {
            var batchInputIDs = inputIDs[batchIdx, 0...]

            // Mask out padding - use where from MLX module
            batchInputIDs = `where`(
                mask[batchIdx, 0...] .== 1, batchInputIDs, zeros(like: batchInputIDs),
            )

            // Count images and videos in this sequence
            let visionStartMask = (batchInputIDs .== MLXArray(visionStartTokenID))
            let visionStartWeighted = `where`(
                visionStartMask, MLXArray(0 ..< seqLength), zeros(like: batchInputIDs),
            )
            let visionStartIdx = argMax(visionStartWeighted).item(Int.self)

            guard visionStartIdx < seqLength - 1 else {
                continue  // No vision tokens
            }

            let imageNums = ((batchInputIDs .== MLXArray(imageTokenID)).asType(.int32).sum()).item(
                Int.self,
            )
            let videoNums = ((batchInputIDs .== MLXArray(videoTokenID)).asType(.int32).sum()).item(
                Int.self,
            )

            let inputTokens = batchInputIDs.asArray(Int32.self).map { Int($0) }
            var llmPosIDsList: [MLXArray] = []

            var st = 0
            var remainImages = imageNums
            var remainVideos = videoNums
            var imageIndex = 0
            var videoIndex = 0

            // Process each image/video in sequence
            for _ in 0 ..< (imageNums + videoNums) {
                // Find next image/video token position
                let edImage: Int =
                    if remainImages > 0, let idx = inputTokens[st...].firstIndex(of: imageTokenID) {
                        idx
                    } else {
                        inputTokens.count + 1
                    }

                let edVideo: Int =
                    if remainVideos > 0, let idx = inputTokens[st...].firstIndex(of: videoTokenID) {
                        idx
                    } else {
                        inputTokens.count + 1
                    }

                let (t, h, w, ed): (Int, Int, Int, Int)
                if edImage < edVideo {
                    // Process image
                    guard let grid = imageGridTHW, imageIndex < grid.count else {
                        break
                    }
                    (t, h, w) = grid[imageIndex].values
                    imageIndex += 1
                    remainImages -= 1
                    ed = edImage
                } else {
                    // Process video
                    guard let grid = videoGridTHW, videoIndex < grid.count else {
                        break
                    }
                    (t, h, w) = grid[videoIndex].values
                    videoIndex += 1
                    remainVideos -= 1
                    ed = edVideo
                }

                let llmGridT = t
                let llmGridH = h / spatialMergeSize
                let llmGridW = w / spatialMergeSize

                // Calculate starting index
                let stIdx: Int
                if let lastArray = llmPosIDsList.last {
                    let maxVal = lastArray.max().item(Int.self)
                    stIdx = maxVal + 1
                } else {
                    stIdx = 0
                }

                // Add text tokens before this visual block
                let textLen = ed - st
                if textLen > 0 {
                    var index = MLXArray(0 ..< textLen).reshaped([1, textLen])
                    index = broadcast(index, to: [3, textLen])
                    index = index + MLXArray(stIdx)
                    llmPosIDsList.append(index)
                }

                // Add 3D position IDs for visual tokens
                // Python: mx.stack([t_index, h_index, w_index]) + text_len + st_idx
                // Adds offset to ALL three dimensions!
                var tIndex = MLXArray(0 ..< llmGridT).reshaped([llmGridT, 1])
                tIndex = broadcast(tIndex, to: [llmGridT, llmGridH * llmGridW])
                tIndex = tIndex.flattened()

                var hIndex = MLXArray(0 ..< llmGridH).reshaped([1, llmGridH, 1])
                hIndex = broadcast(hIndex, to: [llmGridT, llmGridH, llmGridW])
                hIndex = hIndex.flattened()

                var wIndex = MLXArray(0 ..< llmGridW).reshaped([1, 1, llmGridW])
                wIndex = broadcast(wIndex, to: [llmGridT, llmGridH, llmGridW])
                wIndex = wIndex.flattened()

                let visualPosIDs = stacked([tIndex, hIndex, wIndex]) + MLXArray(textLen + stIdx)
                llmPosIDsList.append(visualPosIDs)

                st = ed + llmGridT * llmGridH * llmGridW
            }

            // Add remaining text tokens after last visual block
            if st < inputTokens.count {
                let stIdx: Int
                if let lastArray = llmPosIDsList.last {
                    let maxVal = lastArray.max().item(Int.self)
                    stIdx = maxVal + 1
                } else {
                    stIdx = 0
                }

                let textLen = inputTokens.count - st
                var tIndex = MLXArray(0 ..< textLen).reshaped([1, textLen])
                tIndex = broadcast(tIndex, to: [3, textLen])
                llmPosIDsList.append(tIndex + MLXArray(stIdx))
            }

            // Concatenate all position IDs for this batch item
            if !llmPosIDsList.isEmpty {
                let llmPositions = concatenated(llmPosIDsList, axis: 1)  // [3, seq]

                // Update position_ids for this batch
                let expandedMask = broadcast(
                    mask[batchIdx, 0...][.newAxis, .newAxis, 0...], to: [3, 1, seqLength],
                )
                let expandedPositions = llmPositions[0..., .newAxis, 0...]
                let newPositions = `where`(
                    expandedMask, expandedPositions,
                    positionIDs[0..., batchIdx ..< batchIdx + 1, 0...],
                )

                // Replace this batch's position IDs (assumes batch size = 1)
                positionIDs = newPositions

                let maxPosID = llmPositions.max().item(Int.self)
                mropePositionDeltas.append(maxPosID + 1 - inputTokens.count)
            }
        }

        // Python always returns deltas array (zeros for text-only, computed values for multimodal)
        let deltas =
            if mropePositionDeltas.isEmpty {
                MLXArray.zeros([batchSize], dtype: .int32)
            } else {
                MLXArray(mropePositionDeltas.map { Int32($0) })
            }
        return (positionIDs, deltas)
    }
}

// MARK: - Model

private func qwen3OmniMaskedScatter(inputTensor: MLXArray, mask: MLXArray, source: MLXArray)
    -> MLXArray
{
    let mask = mask.asType(.bool)
    let inputShape = mask.shape
    let resultFlat = broadcast(inputTensor, to: inputShape).flattened()
    let maskFlat = mask.flattened()
    let sourceFlat = source.flattened()
    let selection = cumsum(maskFlat.asType(.int32), axis: 0) - 1
    let bounded = selection % sourceFlat.dim(0)
    let selected = sourceFlat.take(bounded, axis: 0)
    return MLX.where(maskFlat, selected, resultFlat).reshaped(inputShape)
}

public struct Qwen3OmniMoETalkerInput {
    public let inputEmbeddings: MLXArray
    public let inputIDs: MLXArray
    public let trailingTextHidden: MLXArray
    public let ttsPadEmbedding: MLXArray

    public init(
        inputEmbeddings: MLXArray,
        inputIDs: MLXArray,
        trailingTextHidden: MLXArray,
        ttsPadEmbedding: MLXArray,
    ) {
        self.inputEmbeddings = inputEmbeddings
        self.inputIDs = inputIDs
        self.trailingTextHidden = trailingTextHidden
        self.ttsPadEmbedding = ttsPadEmbedding
    }
}

public struct Qwen3OmniMoEThinkerTrace {
    public let sequences: MLXArray
    public let hiddenStates: MLXArray
    public let inputEmbeddings: MLXArray

    public init(
        sequences: MLXArray,
        hiddenStates: MLXArray,
        inputEmbeddings: MLXArray,
    ) {
        self.sequences = sequences
        self.hiddenStates = hiddenStates
        self.inputEmbeddings = inputEmbeddings
    }
}

public final class Qwen3OmniMoE: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionModel: Qwen3OmniMoEVision.VisionModel
    @ModuleInfo(key: "audio_tower") private var audioModel: Qwen3OmniMoEAudio.AudioModel?
    @ModuleInfo(key: "code2wav") private var code2WavModel: Qwen3OmniMoECode2Wav.Code2WavModel?
    @ModuleInfo(key: "talker") private var talkerModel: Qwen3OmniMoETalker.Talker?
    @ModuleInfo(key: "language_model") private var languageModel: Qwen3OmniMoELanguage.LanguageModel

    public let config: Qwen3OmniMoEConfiguration

    public init(_ config: Qwen3OmniMoEConfiguration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3OmniMoEVision.VisionModel(config.visionConfiguration)
        if let audioConfiguration = config.audioConfiguration {
            _audioModel.wrappedValue = Qwen3OmniMoEAudio.AudioModel(audioConfiguration)
        }
        if config.enableAudioOutput, let code2WavConfiguration = config.code2WavConfiguration {
            _code2WavModel.wrappedValue = Qwen3OmniMoECode2Wav.Code2WavModel(code2WavConfiguration)
        }
        if config.enableAudioOutput, let talkerConfiguration = config.talkerConfiguration {
            _talkerModel.wrappedValue = Qwen3OmniMoETalker.Talker(talkerConfiguration)
        }
        _languageModel.wrappedValue = Qwen3OmniMoELanguage.LanguageModel(config)
    }

    public var vocabularySize: Int {
        config.vocabSize
    }

    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    private func mergeInputIDsWithImageFeatures(
        imageFeatures: MLXArray,
        inputEmbeds: MLXArray,
        inputIDs: MLXArray,
        imageTokenIndex: Int,
        videoTokenIndex: Int,
    )
        throws -> (MLXArray, MLXArray)
    {
        let imageMask = (inputIDs .== MLXArray(imageTokenIndex))
        let videoMask = (inputIDs .== MLXArray(videoTokenIndex))
        var specialMask = (imageMask .|| videoMask)

        let nImageTokens = specialMask.sum().item(Int.self)

        specialMask = expandedDimensions(specialMask, axis: -1)
        let maskExpanded = broadcast(specialMask, to: inputEmbeds.shape)

        let nImageFeatures = imageFeatures.dim(0)
        let nImageMaskElements = maskExpanded.sum().item(Int.self)
        let imageFeatureSize = imageFeatures.size

        guard nImageMaskElements == imageFeatureSize else {
            throw Qwen3OmniMoEError.featureTokenMismatch(
                expected: nImageTokens, actual: nImageFeatures)
        }

        let originalShape = inputEmbeds.shape
        let flattenedEmbeds = inputEmbeds.flattened()
        let flattenedFeatures = imageFeatures.flattened()
        let flattenedMask = maskExpanded.flattened()

        let indices = nonZero(flattenedMask.asType(.bool))

        var result = flattenedEmbeds
        if !indices.isEmpty, indices.count == flattenedFeatures.size {
            let indexArray = MLXArray(indices.map { UInt32($0) })
            result[indexArray] = flattenedFeatures
        }

        result = result.reshaped(originalShape)

        let visualMask = specialMask.squeezed(axis: -1).asType(.bool)
        return (result, visualMask)
    }

    private func nonZero(_ mask: MLXArray) -> [Int] {
        let values = mask.asArray(Bool.self)
        var indices: [Int] = []
        indices.reserveCapacity(values.count)
        for (idx, value) in values.enumerated() where value {
            indices.append(idx)
        }
        return indices
    }

    private func mergeAudioFeatures(
        audio: LMInput.ProcessedAudio,
        inputEmbeds: MLXArray,
        inputIDs: MLXArray,
    )
        throws -> MLXArray
    {
        guard let audioModel else {
            throw Qwen3OmniMoEError.audioRequiresAudioTower
        }

        var audioFeatures = try audioModel(audio).asType(inputEmbeds.dtype)
        if audioFeatures.ndim == 3 {
            audioFeatures = audioFeatures.reshaped(-1, audioFeatures.dim(-1))
        }

        guard audioFeatures.dim(-1) == inputEmbeds.dim(-1) else {
            throw Qwen3OmniMoEError.audioHiddenSizeMismatch(
                expected: inputEmbeds.dim(-1),
                actual: audioFeatures.dim(-1),
            )
        }

        let audioMask = inputIDs .== MLXArray(config.audioTokenID)
        let audioTokenCount = audioMask.asType(.int32).sum().item(Int.self)
        guard audioTokenCount > 0 else {
            return inputEmbeds
        }

        if audioFeatures.dim(0) > audioTokenCount {
            audioFeatures = audioFeatures[..<audioTokenCount, 0...]
        } else if audioFeatures.dim(0) < audioTokenCount {
            let padding = MLXArray.zeros(
                [audioTokenCount - audioFeatures.dim(0), audioFeatures.dim(-1)],
                dtype: audioFeatures.dtype,
            )
            audioFeatures = concatenated([audioFeatures, padding], axis: 0)
        }

        let expandedMask = broadcast(expandedDimensions(audioMask, axis: -1), to: inputEmbeds.shape)
        return qwen3OmniMaskedScatter(
            inputTensor: inputEmbeds,
            mask: expandedMask,
            source: audioFeatures,
        )
    }

    private func combinedFrames(
        imageFrames: [THW]?,
        videoFrames: [THW]?,
    )
        -> [THW]
    {
        var frames: [THW] = []
        if let imageFrames {
            frames.append(contentsOf: imageFrames)
        }
        if let videoFrames {
            frames.append(contentsOf: videoFrames)
        }
        return frames
    }

    private func cumulativeSplitIndices(from sizes: [Int]) -> [Int] {
        var sum = 0
        return sizes.dropLast().map { size in
            sum += size
            return sum
        }
    }

    private struct ThinkerInputFeatures {
        var inputEmbeddings: MLXArray?
        var visualMask: MLXArray?
        var deepstackEmbeds: [MLXArray]?
        var pixelValues: MLXArray?
        var imageFrames: [THW]?
        var videoFrames: [THW]?
    }

    private func thinkerInputFeatures(
        inputIDs: MLXArray,
        input: LMInput,
        requireInputEmbeddings: Bool = false,
    )
        throws -> ThinkerInputFeatures
    {
        var pixelValues: MLXArray?
        var imageFrames: [THW]? = nil
        var videoFrames: [THW]? = nil

        let dtype = visionModel.patchEmbed.proj.weight.dtype

        var pixelParts: [MLXArray] = []

        if let image = input.image {
            pixelParts.append(image.pixels.asType(dtype))
            imageFrames = image.frames
        }

        if let video = input.video {
            pixelParts.append(video.pixels.asType(dtype))
            videoFrames = video.frames
        }

        if !pixelParts.isEmpty {
            pixelValues = concatenated(pixelParts)
        }

        var inputEmbeddings: MLXArray? = nil
        var visualMask: MLXArray?
        var deepstackEmbeds: [MLXArray]? = nil

        if requireInputEmbeddings || input.audio != nil || pixelValues != nil {
            inputEmbeddings = languageModel.model.embedTokens(inputIDs)
        }

        if let audio = input.audio {
            inputEmbeddings = try mergeAudioFeatures(
                audio: audio,
                inputEmbeds: inputEmbeddings!,
                inputIDs: inputIDs,
            )
        }

        if let pixelValues,
            let framesList = combinedFrames(imageFrames: imageFrames, videoFrames: videoFrames)
                .nilIfEmpty
        {
            let (visionHidden, deepstackOutputs) = visionModel(pixelValues, gridTHW: framesList)
            let mergeSize = config.visionConfiguration.spatialMergeSize
            let splits = framesList.map { $0.product / (mergeSize * mergeSize) }
            let splitIndices = cumulativeSplitIndices(from: splits)
            let featureSlices = visionHidden.split(indices: splitIndices)
            let flattenedFeatures = concatenated(featureSlices).asType(inputEmbeddings!.dtype)

            let (mergedEmbeds, mask) = try mergeInputIDsWithImageFeatures(
                imageFeatures: flattenedFeatures,
                inputEmbeds: inputEmbeddings!,
                inputIDs: inputIDs,
                imageTokenIndex: config.imageTokenIndex,
                videoTokenIndex: config.videoTokenIndex,
            )

            inputEmbeddings = mergedEmbeds
            visualMask = mask

            if !deepstackOutputs.isEmpty {
                deepstackEmbeds = deepstackOutputs.map { layerFeatures in
                    let splitIndices = cumulativeSplitIndices(from: splits)
                    let slices = layerFeatures.split(indices: splitIndices)
                    return concatenated(slices).asType(inputEmbeddings!.dtype)
                }
            }
        }

        return ThinkerInputFeatures(
            inputEmbeddings: inputEmbeddings,
            visualMask: visualMask,
            deepstackEmbeds: deepstackEmbeds,
            pixelValues: pixelValues,
            imageFrames: imageFrames,
            videoFrames: videoFrames,
        )
    }

    public func prepare(
        _ input: LMInput,
        cache: [KVCache],
        state _: LMOutput.State?, prefill: ModelPrefillOptions,
    )
        throws -> PrepareResult
    {
        let inputIDs = input.text.tokens
        let features = try thinkerInputFeatures(inputIDs: inputIDs, input: input)

        let typedCache = cache.isEmpty ? nil : cache

        let languageOutput = languageModel(
            inputIDs,
            cache: typedCache,
            inputEmbeddings: features.inputEmbeddings,
            mask: nil,
            positionIDs: nil,
            visualMask: features.visualMask,
            deepstackEmbeds: features.deepstackEmbeds,
            pixelValues: features.pixelValues,
            imageGridTHW: features.imageFrames,
            videoGridTHW: features.videoFrames,
        )

        let totalPositions = features.inputEmbeddings?.dim(1) ?? inputIDs.size
        prefill.progress?(totalPositions, totalPositions)
        return .logits(languageOutput)
    }

    public func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state _: LMOutput.State?,
    )
        -> LMOutput
    {
        languageModel(
            input.tokens,
            cache: cache,
            inputEmbeddings: nil,
            mask: nil,
            positionIDs: nil,
            visualMask: nil,
            deepstackEmbeds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil,
        )
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(
            inputs,
            cache: cache,
            inputEmbeddings: nil,
            mask: nil,
            positionIDs: nil,
            visualMask: nil,
            deepstackEmbeds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil,
        ).logits
    }

    public func extractThinkerTrace(
        input: LMInput,
        sequences: MLXArray,
        targetLayerIndex: Int? = nil,
    )
        throws -> Qwen3OmniMoEThinkerTrace
    {
        guard sequences.ndim == 2 else {
            throw VLMError.processing("Qwen3-Omni thinker replay requires a 2D sequence tensor.")
        }

        let replayInput = LMInput(
            text: .init(tokens: sequences),
            image: input.image,
            video: input.video,
            audio: input.audio,
        )
        let features = try thinkerInputFeatures(
            inputIDs: sequences,
            input: replayInput,
            requireInputEmbeddings: true,
        )
        guard let inputEmbeddings = features.inputEmbeddings else {
            throw VLMError.processing("Qwen3-Omni thinker replay failed to build input embeddings.")
        }

        let (positionIDs, _) = Qwen3OmniMoELanguage.getRopeIndex(
            inputIDs: sequences,
            imageGridTHW: features.imageFrames,
            videoGridTHW: features.videoFrames,
            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
            imageTokenID: config.imageTokenIndex,
            videoTokenID: config.videoTokenIndex,
            visionStartTokenID: config.visionStartTokenID,
            attentionMask: replayInput.text.mask,
        )
        let layerIndex =
            targetLayerIndex
            ?? config.talkerConfiguration?.acceptHiddenLayer
            ?? max(config.textConfiguration.numHiddenLayers - 1, 0)
        let hiddenStates = try languageModel.model.hiddenStates(
            sequences,
            cache: nil,
            inputEmbeddings: inputEmbeddings,
            mask: nil,
            positionIDs: positionIDs,
            visualMask: features.visualMask,
            deepstackEmbeds: features.deepstackEmbeds,
            targetLayerIndex: layerIndex,
        )

        return Qwen3OmniMoEThinkerTrace(
            sequences: sequences,
            hiddenStates: hiddenStates,
            inputEmbeddings: inputEmbeddings,
        )
    }

    public func makeTalkerInput(
        sequences: MLXArray,
        thinkerHiddenStates: MLXArray,
        thinkerInputEmbeddings: MLXArray,
        speaker: String = "ethan",
    )
        throws -> Qwen3OmniMoETalkerInput?
    {
        guard let talkerModel, let talkerConfiguration = config.talkerConfiguration else {
            throw VLMError.processing("Qwen3-Omni talker is not configured.")
        }
        guard sequences.ndim == 2, sequences.dim(0) == 1 else {
            throw VLMError.processing(
                "Qwen3-Omni audio output currently supports a single generated sequence.",
            )
        }
        guard thinkerHiddenStates.ndim == 3,
            thinkerInputEmbeddings.ndim == 3,
            thinkerHiddenStates.dim(0) == 1,
            thinkerInputEmbeddings.dim(0) == 1,
            thinkerHiddenStates.dim(1) == sequences.dim(1),
            thinkerInputEmbeddings.dim(1) == sequences.dim(1)
        else {
            throw VLMError.processing(
                "Qwen3-Omni thinker hidden states and embeddings must match generated sequence length.",
            )
        }

        let speakerID = talkerConfiguration.speakerID[speaker.lowercased()]
        guard let speakerID else {
            throw VLMError.processing("Qwen3-Omni speaker '\(speaker)' is not configured.")
        }

        let tokenIDs = sequences.asArray(Int.self)
        var segmentStarts: [Int] = []
        segmentStarts.reserveCapacity(tokenIDs.count)
        for (index, tokenID) in tokenIDs.enumerated() where tokenID == config.imStartTokenID {
            segmentStarts.append(index)
        }
        guard !segmentStarts.isEmpty else {
            return nil
        }
        segmentStarts.append(tokenIDs.count)

        let specialEmbeddings = try talkerSpecialEmbeddings()
        let talkerHiddenSize = talkerConfiguration.textConfiguration.hiddenSize

        var talkerInputEmbeddings: [MLXArray] = []
        var talkerInputIDs: [MLXArray] = []
        var trailingTextHidden: MLXArray?
        let lastSegmentIndex = segmentStarts.count - 2

        for index in 0 ..< lastSegmentIndex + 1 {
            let start = segmentStarts[index]
            let end = segmentStarts[index + 1]
            guard start + 1 < end else {
                continue
            }

            let roleTokenID = tokenIDs[start + 1]
            if roleTokenID == config.systemTokenID {
                continue
            }

            if roleTokenID == config.userTokenID {
                let userPart = userTalkerPart(
                    start: start,
                    end: end,
                    tokenIDs: tokenIDs,
                    thinkerHiddenStates: thinkerHiddenStates,
                    thinkerInputEmbeddings: thinkerInputEmbeddings,
                    talkerModel: talkerModel,
                )
                talkerInputEmbeddings.append(userPart)
                talkerInputIDs.append(sequences[0..., start ..< end])
            } else if roleTokenID == config.assistantTokenID, index == lastSegmentIndex {
                let assistantPart = try assistantTalkerPart(
                    start: start,
                    end: end,
                    speakerID: speakerID,
                    thinkerInputEmbeddings: thinkerInputEmbeddings,
                    ttsBOS: specialEmbeddings.bos,
                    ttsEOS: specialEmbeddings.eos,
                    ttsPad: specialEmbeddings.pad,
                    talkerModel: talkerModel,
                    talkerHiddenSize: talkerHiddenSize,
                )
                talkerInputEmbeddings.append(assistantPart.inputEmbeddings)
                talkerInputIDs.append(assistantPart.inputIDs)
                trailingTextHidden = assistantPart.trailingTextHidden
            } else if roleTokenID == config.assistantTokenID {
                continue
            } else {
                throw VLMError.processing(
                    "Qwen3-Omni expected a role token after <|im_start|>.",
                )
            }
        }

        guard !talkerInputEmbeddings.isEmpty, let trailingTextHidden else {
            return nil
        }

        return Qwen3OmniMoETalkerInput(
            inputEmbeddings: concatenated(talkerInputEmbeddings, axis: 1),
            inputIDs: concatenated(talkerInputIDs, axis: 1),
            trailingTextHidden: trailingTextHidden,
            ttsPadEmbedding: specialEmbeddings.pad,
        )
    }

    private func userTalkerPart(
        start: Int,
        end: Int,
        tokenIDs: [Int],
        thinkerHiddenStates: MLXArray,
        thinkerInputEmbeddings: MLXArray,
        talkerModel: Qwen3OmniMoETalker.Talker,
    )
        -> MLXArray
    {
        var pieces: [MLXArray] = []
        pieces.reserveCapacity(max(end - start, 0))

        for position in start ..< end {
            if tokenIDs[position] == config.audioTokenID
                || tokenIDs[position] == config.imageTokenID
                || tokenIDs[position] == config.videoTokenID
            {
                pieces.append(
                    talkerModel.projectHidden(
                        thinkerHiddenStates[0..., position ..< position + 1, 0...],
                    ),
                )
            } else {
                pieces.append(
                    talkerModel.projectText(
                        thinkerInputEmbeddings[0..., position ..< position + 1, 0...],
                    ),
                )
            }
        }

        return concatenated(pieces, axis: 1)
    }

    private func assistantTalkerPart(
        start: Int,
        end: Int,
        speakerID: Int,
        thinkerInputEmbeddings: MLXArray,
        ttsBOS: MLXArray,
        ttsEOS: MLXArray,
        ttsPad: MLXArray,
        talkerModel: Qwen3OmniMoETalker.Talker,
        talkerHiddenSize: Int,
    )
        throws -> (inputEmbeddings: MLXArray, inputIDs: MLXArray, trailingTextHidden: MLXArray)
    {
        guard end - start >= 4 else {
            throw VLMError.processing(
                "Qwen3-Omni assistant segment must include chat prefix and at least one text token.",
            )
        }

        let assistantHidden = talkerModel.projectText(
            thinkerInputEmbeddings[0..., start ..< end, 0...],
        )
        let assistantTextHidden = concatenated(
            [
                assistantHidden[0..., ..<3, 0...],
                broadcast(ttsPad, to: [1, 4, ttsPad.dim(-1)]),
                ttsBOS,
                assistantHidden[0..., 3 ..< 4, 0...],
            ],
            axis: 1,
        )

        let codecSpecialTokens = MLXArray(
            [
                Int32(config.talkerConfiguration?.codecNoThinkID ?? 2155),
                Int32(config.talkerConfiguration?.codecThinkBOSID ?? 2156),
                Int32(config.talkerConfiguration?.codecThinkEOSID ?? 2157),
                Int32(speakerID),
                Int32(config.talkerConfiguration?.codecPadID ?? 2148),
                Int32(config.talkerConfiguration?.codecBOSID ?? 2149),
            ],
            [1, 6],
        )
        let assistantCodecHidden = concatenated(
            [
                MLXArray.zeros([1, 3, talkerHiddenSize], dtype: assistantTextHidden.dtype),
                talkerModel.model.codecEmbedding(codecSpecialTokens),
            ],
            axis: 1,
        )
        let trailingTextHidden = concatenated(
            [
                assistantHidden[0..., 4..., 0...],
                ttsEOS,
            ],
            axis: 1,
        )
        let inputIDs = MLXArray(
            [Int32](
                repeating: Int32(config.ttsPadTokenID),
                count: assistantTextHidden.dim(1),
            ),
            [1, assistantTextHidden.dim(1)],
        )

        return (assistantTextHidden + assistantCodecHidden, inputIDs, trailingTextHidden)
    }

    public func generateAudioFromThinker(
        sequences: MLXArray,
        thinkerHiddenStates: MLXArray,
        thinkerInputEmbeddings: MLXArray,
        speaker: String = "ethan",
        maxNewTokens: Int = 2048,
        sampler: any LogitSampler = ArgMaxSampler(),
        residualSampler: (any LogitSampler)? = nil,
        chunkSize: Int = 300,
        leftContextSize: Int = 25,
    )
        throws -> MLXArray
    {
        guard
            let talkerInput = try makeTalkerInput(
                sequences: sequences,
                thinkerHiddenStates: thinkerHiddenStates,
                thinkerInputEmbeddings: thinkerInputEmbeddings,
                speaker: speaker,
            )
        else {
            return MLXArray.zeros([sequences.dim(0), 1, 0], dtype: .float32)
        }

        return try generateAudioFromTalker(
            inputEmbeddings: talkerInput.inputEmbeddings,
            trailingTextHidden: talkerInput.trailingTextHidden,
            ttsPadEmbedding: talkerInput.ttsPadEmbedding,
            maxNewTokens: maxNewTokens,
            sampler: sampler,
            residualSampler: residualSampler,
            chunkSize: chunkSize,
            leftContextSize: leftContextSize,
        )
    }

    public func decodeAudio(codes: MLXArray) throws -> MLXArray {
        guard let code2WavModel else {
            throw VLMError.processing("Qwen3-Omni code2wav is not configured.")
        }
        return try code2WavModel(codes: codes)
    }

    public func talker(
        inputIDs: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil,
        cache: [KVCache]? = nil,
        attentionMask: MLXArray? = nil,
        positionIDs: MLXArray? = nil,
        visualMask: MLXArray? = nil,
        deepstackEmbeds: [MLXArray]? = nil,
    )
        throws -> (logits: MLXArray, hiddenStates: MLXArray)
    {
        guard let talkerModel else {
            throw VLMError.processing("Qwen3-Omni talker is not configured.")
        }
        return talkerModel(
            inputIDs: inputIDs,
            inputEmbeddings: inputEmbeddings,
            cache: cache,
            attentionMask: attentionMask,
            positionIDs: positionIDs,
            visualMask: visualMask,
            deepstackEmbeds: deepstackEmbeds,
        )
    }

    public func talkerSpecialEmbeddings() throws -> (
        bos: MLXArray,
        eos: MLXArray,
        pad: MLXArray,
    ) {
        guard let talkerModel else {
            throw VLMError.processing("Qwen3-Omni talker is not configured.")
        }
        let specialTokens = MLXArray(
            [
                Int32(config.ttsBOSTokenID), Int32(config.ttsEOSTokenID),
                Int32(config.ttsPadTokenID),
            ],
            [1, 3],
        )
        let specialEmbeddings = languageModel.model.embedTokens(specialTokens)
        let projected = talkerModel.projectText(specialEmbeddings)
        return (
            projected[0..., 0 ..< 1, 0...],
            projected[0..., 1 ..< 2, 0...],
            projected[0..., 2 ..< 3, 0...],
        )
    }

    public func generateAudioCodes(
        inputEmbeddings: MLXArray,
        trailingTextHidden: MLXArray,
        ttsPadEmbedding: MLXArray,
        maxNewTokens: Int = 2048,
        sampler: any LogitSampler = ArgMaxSampler(),
        residualSampler: (any LogitSampler)? = nil,
    )
        throws -> MLXArray
    {
        guard let talkerModel else {
            throw VLMError.processing("Qwen3-Omni talker is not configured.")
        }
        return try talkerModel.generateCodes(
            inputEmbeddings: inputEmbeddings,
            trailingTextHidden: trailingTextHidden,
            ttsPadEmbedding: ttsPadEmbedding,
            maxNewTokens: maxNewTokens,
            sampler: sampler,
            residualSampler: residualSampler,
        )
    }

    public func generateAudioFromTalker(
        inputEmbeddings: MLXArray,
        trailingTextHidden: MLXArray,
        ttsPadEmbedding: MLXArray,
        maxNewTokens: Int = 2048,
        sampler: any LogitSampler = ArgMaxSampler(),
        residualSampler: (any LogitSampler)? = nil,
        chunkSize: Int = 300,
        leftContextSize: Int = 25,
    )
        throws -> MLXArray
    {
        guard let code2WavModel else {
            throw VLMError.processing("Qwen3-Omni code2wav is not configured.")
        }
        let codes = try generateAudioCodes(
            inputEmbeddings: inputEmbeddings,
            trailingTextHidden: trailingTextHidden,
            ttsPadEmbedding: ttsPadEmbedding,
            maxNewTokens: maxNewTokens,
            sampler: sampler,
            residualSampler: residualSampler,
        )
        guard codes.dim(2) > 0 else {
            return MLXArray.zeros([inputEmbeddings.dim(0), 1, 0], dtype: .float32)
        }
        return try code2WavModel.chunkedDecode(
            codes: codes,
            chunkSize: chunkSize,
            leftContextSize: leftContextSize,
        )
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        sanitize(weights: weights, code2WavIsMLX: false)
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        sanitize(weights: weights, code2WavIsMLX: metadata["format"] == "mlx")
    }

    private func sanitize(weights: [String: MLXArray], code2WavIsMLX: Bool) -> [String: MLXArray] {
        var adjusted: [String: MLXArray] = [:]
        adjusted.reserveCapacity(weights.count)
        var talkerWeights: [String: MLXArray] = [:]

        for (key, value) in weights {
            var newKey = key

            if newKey.hasPrefix("code2wav.") {
                guard code2WavModel != nil else {
                    continue
                }
                let sanitized =
                    code2WavIsMLX
                    ? [newKey: value] : Qwen3OmniMoECode2Wav.sanitizeWeights([newKey: value])
                adjusted.merge(sanitized) { _, new in new }
                continue
            }

            if newKey.hasPrefix("talker.") {
                guard config.talkerConfiguration != nil, talkerModel != nil else {
                    continue
                }
                talkerWeights[newKey] = value
                continue
            }

            if newKey.hasPrefix("thinker.audio_tokenizer.")
                || newKey.hasPrefix("thinker.audio_output.")
            {
                continue
            }

            if newKey.hasPrefix("thinker.vision_tower.") {
                newKey = String(newKey.dropFirst("thinker.".count))
            } else if newKey.hasPrefix("thinker.audio_tower.") {
                guard audioModel != nil else {
                    continue
                }
                newKey = String(newKey.dropFirst("thinker.".count))
            } else if newKey.hasPrefix("thinker.language_model.") {
                newKey = String(newKey.dropFirst("thinker.".count))
            } else if newKey.hasPrefix("thinker.model.") {
                newKey = newKey.replacingOccurrences(
                    of: "thinker.model.",
                    with: "language_model.model.",
                )
            } else if newKey.hasPrefix("thinker.visual.") {
                newKey = newKey.replacingOccurrences(
                    of: "thinker.visual.",
                    with: "vision_tower.",
                )
            } else if newKey.hasPrefix("thinker.lm_head.") {
                newKey = newKey.replacingOccurrences(
                    of: "thinker.lm_head.",
                    with: "language_model.lm_head.",
                )
            } else if newKey.contains("model.visual") {
                newKey = newKey.replacingOccurrences(of: "model.visual", with: "vision_tower")
            } else if newKey.contains("model.language_model") {
                newKey = newKey.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model",
                )
            } else if newKey.contains("lm_head") && !newKey.hasPrefix("language_model.") {
                newKey = newKey.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            }

            if newKey.contains("position_ids") || newKey.contains("rotary_emb.inv_freq") {
                continue
            }

            if config.textConfiguration.tieWordEmbeddings, newKey.contains(".lm_head.") {
                // if using tieWordEmbeddings omit these keys as they will not be consumed
                continue
            }

            var adjustedValue = value
            if newKey.hasPrefix("audio_tower.conv2d"),
                newKey.hasSuffix(".weight"),
                adjustedValue.ndim == 4,
                !(adjustedValue.dim(1) == 3 && adjustedValue.dim(2) == 3)
            {
                adjustedValue = adjustedValue.transposed(0, 2, 3, 1)
            }

            adjusted[newKey] = adjustedValue
        }

        if let talkerConfiguration = config.talkerConfiguration, !talkerWeights.isEmpty {
            let sanitizedTalker = Qwen3OmniMoETalker.sanitizeWeights(
                talkerWeights,
                config: talkerConfiguration,
            )
            adjusted.merge(sanitizedTalker) { _, new in new }
        }

        Qwen3OmniMoE.stackExperts(weights: &adjusted, config: config.textConfiguration)
        return visionModel.sanitize(weights: adjusted)
    }

    private static func stackExperts(
        weights: inout [String: MLXArray],
        config: Qwen3OmniMoEConfiguration.TextConfiguration,
    ) {
        guard config.numExperts > 0 else {
            return
        }

        for layer in 0 ..< config.numHiddenLayers {
            let prefix = "language_model.model.layers.\(layer).mlp"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    let firstKey = "\(prefix).experts.0.\(projection).\(suffix)"
                    guard weights[firstKey] != nil else {
                        continue
                    }

                    var toStack: [MLXArray] = []
                    for expert in 0 ..< config.numExperts {
                        if let value = weights.removeValue(
                            forKey: "\(prefix).experts.\(expert).\(projection).\(suffix)",
                        ) {
                            toStack.append(value)
                        }
                    }

                    if toStack.count == config.numExperts {
                        weights["\(prefix).switch_mlp.\(projection).\(suffix)"] = stacked(toStack)
                    }
                }
            }
        }
    }
}

extension [THW] {
    fileprivate var nilIfEmpty: [THW]? {
        isEmpty ? nil : self
    }
}

extension Array where Element: Equatable {
    fileprivate func ranges(of pattern: [Element]) -> [Range<Int>] {
        guard !pattern.isEmpty, pattern.count <= count else {
            return []
        }
        var ranges: [Range<Int>] = []
        var index = startIndex
        while index <= count - pattern.count {
            let end = index + pattern.count
            if Array(self[index ..< end]) == pattern {
                ranges.append(index ..< end)
                index = end
            } else {
                index += 1
            }
        }
        return ranges
    }
}

public struct Qwen3OmniMoEMessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> Message {
        let imageContent = message.images.map { _ in
            ["type": "image"]
        }
        let audioContent = message.audios.map { _ in
            ["type": "audio"]
        }
        let textContent = [["type": "text", "text": message.content]]
        let videoContent = message.videos.map { _ in
            ["type": "video"]
        }

        return [
            "role": message.role.rawValue,
            "content": imageContent + audioContent + videoContent + textContent,
        ]
    }
}
