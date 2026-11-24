import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct MiniCPMOConfiguration: Decodable, Sendable {
    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let rmsNormEps: Float
        public let vocabularySize: Int
        public let headDim: Int
        public let ropeTheta: Float
        public let maxPositionEmbeddings: Int
        public let ropeScaling: [String: StringOrNumber]?
        public let tieWordEmbeddings: Bool
        public let attentionBias: Bool
        public let hiddenActivation: String

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case headDim = "head_dim"
            case ropeTheta = "rope_theta"
            case maxPositionEmbeddings = "max_position_embeddings"
            case ropeScaling = "rope_scaling"
            case tieWordEmbeddings = "tie_word_embeddings"
            case attentionBias = "attention_bias"
            case hiddenActivation = "hidden_act"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "minicpmo"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 12288
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 36
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_748
            headDim =
                try c.decodeIfPresent(Int.self, forKey: .headDim)
                ?? (hiddenSize / attentionHeads)
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 40960
            ropeScaling = try c.decodeIfPresent(
                [String: StringOrNumber].self,
                forKey: .ropeScaling,
            )
            tieWordEmbeddings =
                try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
            attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            hiddenActivation =
                try c.decodeIfPresent(String.self, forKey: .hiddenActivation) ?? "silu"
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let channels: Int
        public let imageSize: Int
        public let patchSize: Int
        public let hiddenActivation: String
        public let layerNormEps: Float

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case channels = "num_channels"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case hiddenActivation = "hidden_act"
            case layerNormEps = "layer_norm_eps"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType)
                ?? "siglip_vision_model"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 27
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 980
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            hiddenActivation =
                try c.decodeIfPresent(String.self, forKey: .hiddenActivation)
                ?? "gelu_pytorch_tanh"
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        }
    }

    public struct AudioConfiguration: Decodable, Sendable {
        public let modelType: String
        public let modelDimensions: Int
        public let encoderLayers: Int
        public let encoderAttentionHeads: Int
        public let encoderFFNDimensions: Int
        public let melBins: Int
        public let maxSourcePositions: Int
        public let dropout: Float
        public let attentionDropout: Float
        public let activationDropout: Float
        public let activationFunction: String
        public let layerNormEps: Float

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case modelDimensions = "d_model"
            case encoderLayers = "encoder_layers"
            case encoderAttentionHeads = "encoder_attention_heads"
            case encoderFFNDimensions = "encoder_ffn_dim"
            case melBins = "num_mel_bins"
            case maxSourcePositions = "max_source_positions"
            case dropout
            case attentionDropout = "attention_dropout"
            case activationDropout = "activation_dropout"
            case activationFunction = "activation_function"
            case layerNormEps = "layer_norm_eps"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "whisper"
            modelDimensions = try c.decodeIfPresent(Int.self, forKey: .modelDimensions) ?? 1024
            encoderLayers = try c.decodeIfPresent(Int.self, forKey: .encoderLayers) ?? 24
            encoderAttentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .encoderAttentionHeads) ?? 16
            encoderFFNDimensions =
                try c.decodeIfPresent(Int.self, forKey: .encoderFFNDimensions) ?? 4096
            melBins = try c.decodeIfPresent(Int.self, forKey: .melBins) ?? 80
            maxSourcePositions =
                try c.decodeIfPresent(Int.self, forKey: .maxSourcePositions) ?? 1500
            dropout = try c.decodeIfPresent(Float.self, forKey: .dropout) ?? 0
            attentionDropout = try c.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0
            activationDropout = try c.decodeIfPresent(Float.self, forKey: .activationDropout) ?? 0
            activationFunction =
                try c.decodeIfPresent(String.self, forKey: .activationFunction) ?? "gelu"
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-5
        }
    }

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let audioConfiguration: AudioConfiguration?
    public let modelType: String
    public let queryCount: Int
    public let imageSize: Int
    public let patchSize: Int
    public let initVision: Bool
    public let initAudio: Bool
    public let initTTS: Bool
    public let audioPoolStep: Int
    public let audioChunkLength: Float
    private let _eosTokenID: MiniCPMOIntOrIntArray?
    public var eosTokenIDs: [Int]? {
        _eosTokenID?.values
    }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case audioConfiguration = "audio_config"
        case modelType = "model_type"
        case queryCount = "query_num"
        case imageSize = "image_size"
        case patchSize = "patch_size"
        case initVision = "init_vision"
        case initAudio = "init_audio"
        case initTTS = "init_tts"
        case audioPoolStep = "audio_pool_step"
        case audioChunkLength = "audio_chunk_length"
        case _eosTokenID = "eos_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if let text = try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration) {
            textConfiguration = text
        } else {
            textConfiguration = try TextConfiguration(from: decoder)
        }
        visionConfiguration =
            try c.decodeIfPresent(VisionConfiguration.self, forKey: .visionConfiguration)
            ?? VisionConfiguration(from: decoder)
        audioConfiguration = try c.decodeIfPresent(
            AudioConfiguration.self,
            forKey: .audioConfiguration,
        )
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "minicpmo"
        queryCount = try c.decodeIfPresent(Int.self, forKey: .queryCount) ?? 64
        imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 448
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        initVision = try c.decodeIfPresent(Bool.self, forKey: .initVision) ?? true
        initAudio = try c.decodeIfPresent(Bool.self, forKey: .initAudio) ?? true
        initTTS = try c.decodeIfPresent(Bool.self, forKey: .initTTS) ?? true
        audioPoolStep = try c.decodeIfPresent(Int.self, forKey: .audioPoolStep) ?? 5
        audioChunkLength = try c.decodeIfPresent(Float.self, forKey: .audioChunkLength) ?? 1.0
        _eosTokenID = try c.decodeIfPresent(MiniCPMOIntOrIntArray.self, forKey: ._eosTokenID)
    }
}

private enum MiniCPMOIntOrIntArray: Decodable {
    case int(Int)
    case ints([Int])

    var values: [Int] {
        switch self {
        case .int(let value): [value]
        case .ints(let values): values
        }
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if let value = try? container.decode(Int.self) {
            self = .int(value)
        } else {
            self = try .ints(container.decode([Int].self))
        }
    }
}

// MARK: - Processor

public struct MiniCPMOProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String?
    public let patchSize: Int
    public let scaleResolution: Int
    public let imageFeatureSize: Int
    public let imageStart: String
    public let imageEnd: String
    public let unknown: String
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let audioFeatureSize: Int
    public let audioSamplingRate: Int
    public let audioHopLength: Int
    public let audioFFTLength: Int
    public let audioChunkLength: Float
    public let audioPoolStep: Int
    public let audioStart: String
    public let audioEnd: String

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case patchSize = "patch_size"
        case scaleResolution = "scale_resolution"
        case imageFeatureSize = "image_feature_size"
        case imageStart = "im_start"
        case imageEnd = "im_end"
        case unknown = "unk"
        case imageMean = "norm_mean"
        case imageStd = "norm_std"
        case imageProcessor = "image_processor"
        case audioProcessor = "audio_processor"
        case audioFeatureSize = "feature_size"
        case audioSamplingRate = "sampling_rate"
        case audioHopLength = "hop_length"
        case audioFFTLength = "n_fft"
        case audioChunkLength = "chunk_length"
        case audioPoolStep = "audio_pool_step"
        case audioStart = "audio_start"
        case audioEnd = "audio_end"
    }

    public init(from decoder: Decoder) throws {
        let outer = try decoder.container(keyedBy: CodingKeys.self)
        let imageC =
            (try? outer.nestedContainer(keyedBy: CodingKeys.self, forKey: .imageProcessor))
            ?? outer
        let audioC = try? outer.nestedContainer(keyedBy: CodingKeys.self, forKey: .audioProcessor)
        processorClass = try outer.decodeIfPresent(String.self, forKey: .processorClass)
        patchSize = try imageC.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        scaleResolution =
            try imageC.decodeIfPresent(Int.self, forKey: .scaleResolution) ?? 448
        imageFeatureSize =
            try imageC.decodeIfPresent(Int.self, forKey: .imageFeatureSize) ?? 64
        imageStart = try imageC.decodeIfPresent(String.self, forKey: .imageStart) ?? "<image>"
        imageEnd = try imageC.decodeIfPresent(String.self, forKey: .imageEnd) ?? "</image>"
        unknown = try imageC.decodeIfPresent(String.self, forKey: .unknown) ?? "<unk>"
        imageMean =
            try imageC.decodeIfPresent([CGFloat].self, forKey: .imageMean)
            ?? [0.5, 0.5, 0.5]
        imageStd =
            try imageC.decodeIfPresent([CGFloat].self, forKey: .imageStd)
            ?? [0.5, 0.5, 0.5]
        let nestedAudioFeatureSize = try audioC?.decodeIfPresent(
            Int.self,
            forKey: .audioFeatureSize,
        )
        let nestedAudioSamplingRate = try audioC?.decodeIfPresent(
            Int.self,
            forKey: .audioSamplingRate,
        )
        let nestedAudioHopLength = try audioC?.decodeIfPresent(
            Int.self,
            forKey: .audioHopLength,
        )
        let nestedAudioFFTLength = try audioC?.decodeIfPresent(
            Int.self,
            forKey: .audioFFTLength,
        )
        let nestedAudioChunkLength = try audioC?.decodeIfPresent(
            Float.self,
            forKey: .audioChunkLength,
        )
        let rootAudioFeatureSize = try outer.decodeIfPresent(Int.self, forKey: .audioFeatureSize)
        let rootAudioSamplingRate = try outer.decodeIfPresent(Int.self, forKey: .audioSamplingRate)
        let rootAudioHopLength = try outer.decodeIfPresent(Int.self, forKey: .audioHopLength)
        let rootAudioFFTLength = try outer.decodeIfPresent(Int.self, forKey: .audioFFTLength)
        let rootAudioChunkLength = try outer.decodeIfPresent(Float.self, forKey: .audioChunkLength)
        audioFeatureSize = nestedAudioFeatureSize ?? rootAudioFeatureSize ?? 80
        audioSamplingRate = nestedAudioSamplingRate ?? rootAudioSamplingRate ?? 16000
        audioHopLength = nestedAudioHopLength ?? rootAudioHopLength ?? 160
        audioFFTLength = nestedAudioFFTLength ?? rootAudioFFTLength ?? 400
        audioChunkLength = nestedAudioChunkLength ?? rootAudioChunkLength ?? 30
        audioPoolStep = try outer.decodeIfPresent(Int.self, forKey: .audioPoolStep) ?? 5
        audioStart =
            try outer.decodeIfPresent(String.self, forKey: .audioStart)
            ?? "<|audio_start|>"
        audioEnd =
            try outer.decodeIfPresent(String.self, forKey: .audioEnd)
            ?? "<|audio_end|>"
    }
}

public struct MiniCPMOProcessor: UserInputProcessor {
    private let config: MiniCPMOProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageStartID: Int
    private let imageEndID: Int
    private let audioStartID: Int
    private let audioEndID: Int
    private let listenID: Int?
    private let unknownID: Int
    private let audioFeatureExtractor: MiniCPMOWHISPERFeatureExtractor

    public init(_ config: MiniCPMOProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        imageStartID = tokenizer.convertTokenToId(config.imageStart) ?? 151_646
        imageEndID = tokenizer.convertTokenToId(config.imageEnd) ?? 151_647
        audioStartID = tokenizer.convertTokenToId(config.audioStart) ?? 151_648
        audioEndID = tokenizer.convertTokenToId(config.audioEnd) ?? 151_649
        listenID = tokenizer.convertTokenToId("<|listen|>")
        unknownID = tokenizer.convertTokenToId(config.unknown) ?? 0
        audioFeatureExtractor = MiniCPMOWHISPERFeatureExtractor(
            featureSize: config.audioFeatureSize,
            samplingRate: config.audioSamplingRate,
            hopLength: config.audioHopLength,
            fftLength: config.audioFFTLength,
            chunkLength: config.audioChunkLength,
            audioPoolStep: config.audioPoolStep,
        )
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let prompt = Self.prompt(from: input)
        let images = MediaProcessing.images(from: input)
        let audios = MediaProcessing.audios(from: input)
        let promptText =
            if images.isEmpty {
                "<|im_start|>user\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
            } else {
                "<|im_start|>user\n\(config.imageStart)\n\(prompt)<|im_end|>\n<|im_start|>assistant\n"
            }

        var processed: [(pixels: MLXArray, target: (Int, Int))] = []
        processed.reserveCapacity(images.count)
        for image in images {
            try processed.append(preprocess(image: image.asCIImage(), processing: input.processing))
        }

        let audioData =
            audios.isEmpty
            ? nil
            : try await preprocess(audios: audios, processing: input.processing)
        let audioFeatureLengths = audioData?.featureLengths ?? []
        let audioTokenCounts = audioData?.tokenCounts ?? []
        let audioPrompt = injectAudioMarkers(
            into: promptText,
            audioCount: audioTokenCounts.count,
        )
        let (tokens, imageBounds, audioMetadata) = tokenize(
            audioPrompt,
            imageCount: processed.count,
            audioTokenCounts: audioTokenCounts,
            audioFeatureLengths: audioFeatureLengths,
        )

        if processed.isEmpty, audioData == nil {
            return LMInput(text: .init(tokens: MLXArray(tokens)[.newAxis]))
        }

        let imageInput: LMInput.ProcessedImage? =
            if processed.isEmpty {
                nil
            } else {
                .init(
                    pixels: concatenated(processed.map { $0.pixels.flattened() }, axis: 0),
                    frames: processed.map { THW(1, $0.target.0, $0.target.1) },
                )
            }
        let imageBoundsArray: MLXArray? =
            imageBounds.isEmpty
            ? nil
            : MLXArray(imageBounds.flatMap { [$0.0, $0.1] }, [imageBounds.count, 2])
        let audioInput: LMInput.ProcessedAudio? =
            if let audioData {
                .init(
                    features: audioData.features,
                    mask: MLXArray(
                        audioMetadata.flatMap { [$0.featureLength, $0.start, $0.end] },
                        [
                            audioMetadata.count, 3,
                        ]),
                )
            } else {
                nil
            }

        return LMInput(
            text: .init(tokens: MLXArray(tokens)[.newAxis]),
            image: imageInput,
            video: imageBoundsArray.map { .init(pixels: $0) },
            audio: audioInput,
        )
    }

    private func tokenize(
        _ text: String,
        imageCount: Int,
        audioTokenCounts: [Int],
        audioFeatureLengths: [Int],
    )
        -> ([Int], [(Int, Int)], [(featureLength: Int, start: Int, end: Int)])
    {
        guard imageCount > 0 || !audioTokenCounts.isEmpty else {
            return (
                tokenizer.encode(text: text, addSpecialTokens: false)
                    .filter { $0 != listenID },
                [],
                [],
            )
        }

        var tokens: [Int] = []
        var imageBounds: [(Int, Int)] = []
        var audioMetadata: [(featureLength: Int, start: Int, end: Int)] = []
        var cursor = text.startIndex
        var imageIndex = 0
        var audioIndex = 0

        while cursor < text.endIndex {
            guard let marker = nextMarker(in: text, from: cursor) else {
                tokens += encodeLiteral(String(text[cursor...]))
                break
            }

            if marker.range.lowerBound > cursor {
                tokens += encodeLiteral(String(text[cursor ..< marker.range.lowerBound]))
            }

            switch marker.kind {
            case .image:
                if imageIndex < imageCount {
                    appendImagePlaceholder(tokens: &tokens, bounds: &imageBounds)
                    imageIndex += 1
                } else {
                    tokens += encodeLiteral(String(text[marker.range]))
                }
            case .audio:
                if audioIndex < audioTokenCounts.count {
                    appendAudioPlaceholder(
                        tokens: &tokens,
                        metadata: &audioMetadata,
                        featureLength: audioFeatureLengths[audioIndex],
                        tokenCount: audioTokenCounts[audioIndex],
                    )
                    audioIndex += 1
                } else {
                    tokens += encodeLiteral(String(text[marker.range]))
                }
            }

            cursor = marker.range.upperBound
        }

        while imageIndex < imageCount {
            var prefix: [Int] = []
            var prefixBounds: [(Int, Int)] = []
            appendImagePlaceholder(tokens: &prefix, bounds: &prefixBounds)
            let delta = prefix.count
            tokens = prefix + tokens
            imageBounds = prefixBounds + imageBounds.map { ($0.0 + delta, $0.1 + delta) }
            audioMetadata = audioMetadata.map {
                ($0.featureLength, $0.start + delta, $0.end + delta)
            }
            imageIndex += 1
        }

        while audioIndex < audioTokenCounts.count {
            var prefix: [Int] = []
            var prefixMetadata: [(featureLength: Int, start: Int, end: Int)] = []
            appendAudioPlaceholder(
                tokens: &prefix,
                metadata: &prefixMetadata,
                featureLength: audioFeatureLengths[audioIndex],
                tokenCount: audioTokenCounts[audioIndex],
            )
            let delta = prefix.count
            tokens = prefix + tokens
            imageBounds = imageBounds.map { ($0.0 + delta, $0.1 + delta) }
            audioMetadata =
                prefixMetadata
                + audioMetadata.map { ($0.featureLength, $0.start + delta, $0.end + delta) }
            audioIndex += 1
        }

        return (tokens, imageBounds, audioMetadata)
    }

    private enum MarkerKind {
        case image
        case audio
    }

    private func nextMarker(in text: String, from cursor: String.Index) -> (
        range: Range<String.Index>,
        kind: MarkerKind,
    )? {
        var candidates: [(Range<String.Index>, MarkerKind)] = []
        if let imageRange = text[cursor...].range(of: config.imageStart) {
            candidates.append((imageRange, .image))
        }
        if let audioRange = text[cursor...].range(of: "<audio>./</audio>") {
            candidates.append((audioRange, .audio))
        }
        if let audioRange = text[cursor...].range(of: "<audio>") {
            candidates.append((audioRange, .audio))
        }
        return candidates.min { lhs, rhs in
            lhs.0.lowerBound < rhs.0.lowerBound
        }
    }

    private func encodeLiteral(_ text: String) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false).filter { $0 != listenID }
    }

    private func injectAudioMarkers(into text: String, audioCount: Int) -> String {
        guard audioCount > 0, !text.contains("<audio>") else {
            return text
        }
        let markers = String(repeating: "<audio>", count: audioCount)
        let userTag = "<|im_start|>user\n"
        if let range = text.range(of: userTag, options: .backwards) {
            return String(text[..<range.upperBound]) + markers + String(text[range.upperBound...])
        }
        return markers + text
    }

    private func appendImagePlaceholder(tokens: inout [Int], bounds: inout [(Int, Int)]) {
        tokens.append(imageStartID)
        let start = tokens.count
        tokens += Array(repeating: unknownID, count: config.imageFeatureSize)
        let end = tokens.count
        tokens.append(imageEndID)
        bounds.append((start, end))
    }

    private func appendAudioPlaceholder(
        tokens: inout [Int],
        metadata: inout [(featureLength: Int, start: Int, end: Int)],
        featureLength: Int,
        tokenCount: Int,
    ) {
        tokens.append(audioStartID)
        let start = tokens.count
        tokens += Array(repeating: unknownID, count: tokenCount)
        let end = tokens.count
        tokens.append(audioEndID)
        metadata.append((featureLength, start, end))
    }

    private func preprocess(
        audios: [UserInput.Audio],
        processing: UserInput.Processing,
    )
        async throws -> (
            features: MLXArray,
            featureLengths: [Int],
            tokenCounts: [Int],
        )
    {
        var audioProcessing = processing.audio
        audioProcessing.sampleRate = Double(config.audioSamplingRate)
        audioProcessing.channels = 1

        var samples: [MLXArray] = []
        samples.reserveCapacity(audios.count)
        for audio in audios {
            try await samples.append(audio.asMLXArray(processing: audioProcessing))
        }
        return try audioFeatureExtractor(samples)
    }

    private func preprocess(
        image source: CIImage,
        processing: UserInput.Processing?,
    )
        throws -> (pixels: MLXArray, target: (Int, Int))
    {
        let source = MediaProcessing.inSRGBToneCurveSpace(
            MediaProcessing.apply(source, processing: processing),
        )
        let size = source.extent.size
        let resized = bestResize(
            width: Int(size.width),
            height: Int(size.height),
            scaleResolution: config.scaleResolution,
            patchSize: config.patchSize,
        )
        let image = MediaProcessing.resampleBicubic(
            source,
            to: CGSize(width: resized.0, height: resized.1),
        )
        var array = MediaProcessing.renderRGBArray(image)
        let mean = MLXArray(config.imageMean.map { Float32($0) }).reshaped(1, 1, 3)
        let std = MLXArray(config.imageStd.map { Float32($0) }).reshaped(1, 1, 3)
        array = (array - mean) / std
        return (
            array.transposed(2, 0, 1).asType(.float32),
            (resized.1 / config.patchSize, resized.0 / config.patchSize),
        )
    }

    private func bestResize(
        width: Int,
        height: Int,
        scaleResolution: Int,
        patchSize: Int,
    )
        -> (Int, Int)
    {
        let ratio = Double(width) / Double(max(height, 1))
        let height = Int(Double(scaleResolution) / sqrt(max(ratio, 1e-6)))
        let width = Int(Double(height) * ratio)
        return (ensureDivide(width, by: patchSize), ensureDivide(height, by: patchSize))
    }

    private func ensureDivide(_ length: Int, by patchSize: Int) -> Int {
        max(Int((Double(length) / Double(patchSize)).rounded()) * patchSize, patchSize)
    }

    private static func prompt(from input: UserInput) -> String {
        switch input.prompt {
        case .text(let text):
            text
        case .messages:
            input.prompt.description
        case .chat(let messages):
            messages.last(where: { $0.role == .user })?.content
                ?? messages.map(\.content).joined(separator: "\n")
        }
    }

}

private struct MiniCPMOWHISPERFeatureExtractor: Sendable {
    let featureSize: Int
    let samplingRate: Int
    let hopLength: Int
    let fftLength: Int
    let chunkLength: Float
    let audioPoolStep: Int
    let window: MaterializedArray
    let melFilters: MaterializedArray

    init(
        featureSize: Int,
        samplingRate: Int,
        hopLength: Int,
        fftLength: Int,
        chunkLength: Float,
        audioPoolStep: Int,
    ) {
        self.featureSize = featureSize
        self.samplingRate = samplingRate
        self.hopLength = hopLength
        self.fftLength = fftLength
        self.chunkLength = chunkLength
        self.audioPoolStep = max(audioPoolStep, 1)
        let window = MLXArray(
            (0 ..< fftLength).map { index in
                Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double(index) / Double(fftLength)))
            },
            [fftLength],
        )
        let melFilters = Self.makeMelFilterBank(
            frequencyBins: fftLength / 2 + 1,
            melFilters: featureSize,
            minFrequency: 0,
            maxFrequency: Float(samplingRate) / 2,
            samplingRate: samplingRate,
        )
        let tables = materialize([window, melFilters])
        self.window = tables[0]
        self.melFilters = tables[1]
    }

    func callAsFunction(_ rawAudio: [MLXArray]) throws -> (
        features: MLXArray,
        featureLengths: [Int],
        tokenCounts: [Int],
    ) {
        guard !rawAudio.isEmpty else {
            throw VLMError.processing("No audio samples were provided.")
        }

        let maxSamples = max(1, Int(Float(samplingRate) * chunkLength))
        var spectrograms: [MLXArray] = []
        var featureLengths: [Int] = []
        var tokenCounts: [Int] = []
        spectrograms.reserveCapacity(rawAudio.count)
        featureLengths.reserveCapacity(rawAudio.count)
        tokenCounts.reserveCapacity(rawAudio.count)

        for audio in rawAudio {
            let samples = Gemma4AudioFeatureExtractor.monoFloatSamples(audio)
            let validSamples = min(max(samples.dim(0), 1), maxSamples)
            let waveform =
                samples.dim(0) > validSamples
                ? samples[..<validSamples]
                : samples
            let featureLength = min(
                max(1, Int(ceil(Double(validSamples) / Double(hopLength)))),
                max(1, maxSamples / hopLength),
            )
            featureLengths.append(featureLength)
            tokenCounts.append(
                Self.audioTokenCount(featureLength: featureLength, poolStep: audioPoolStep))
            spectrograms.append(extractLogMel(waveform, frameCount: featureLength))
        }

        let maxFrames = featureLengths.max() ?? 1
        let paddedSpectrograms = spectrograms.map { spectrogram in
            let pad = max(0, maxFrames - spectrogram.dim(1))
            guard pad > 0 else {
                return spectrogram
            }
            return padded(spectrogram, widths: [0, .init((0, pad))])
        }
        return (stacked(paddedSpectrograms, axis: 0), featureLengths, tokenCounts)
    }

    private func extractLogMel(_ waveform: MLXArray, frameCount: Int) -> MLXArray {
        var rows: [MLXArray] = []
        rows.reserveCapacity(frameCount)
        for frameIndex in 0 ..< frameCount {
            let start = frameIndex * hopLength
            let available = max(0, min(fftLength, waveform.dim(0) - start))
            var frame =
                available > 0
                ? waveform[start ..< (start + available)]
                : MLXArray.zeros([0], dtype: .float32)
            if available < fftLength {
                frame = padded(frame, widths: [.init((0, fftLength - available))])
            }
            rows.append(frame)
        }

        let frames = stacked(rows, axis: 0) * window.asType(waveform.dtype)
        let spectrum = MLXFFT.rfft(frames, n: fftLength, axis: -1)
        let power = spectrum.abs().square()
        var mel = power.matmul(melFilters.asType(power.dtype))
        mel = maximum(mel, MLXArray(Float32(1e-10), dtype: mel.dtype))
        var logMel = log10(mel)
        let floor = logMel.max() - MLXArray(Float32(8), dtype: logMel.dtype)
        logMel = maximum(logMel, floor)
        logMel =
            (logMel + MLXArray(Float32(4), dtype: logMel.dtype))
            / MLXArray(Float32(4), dtype: logMel.dtype)
        return logMel.transposed(1, 0).asType(.float32)
    }

    private static func audioTokenCount(featureLength: Int, poolStep: Int) -> Int {
        let convLength = ((featureLength - 1) / 2) + 1
        return max(1, ((convLength - poolStep) / poolStep) + 1)
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
        let fftFrequencies = (0 ..< frequencyBins).map { index in
            Float(index) * Float(samplingRate) / Float(2 * max(frequencyBins - 1, 1))
        }

        var values = Array(repeating: Float(0), count: frequencyBins * melFilters)
        for filterIndex in 0 ..< melFilters {
            let lower = frequencyPoints[filterIndex]
            let center = frequencyPoints[filterIndex + 1]
            let upper = frequencyPoints[filterIndex + 2]
            let enorm = 2.0 / max(upper - lower, 1e-10)
            for binIndex in 0 ..< frequencyBins {
                let frequency = fftFrequencies[binIndex]
                let rising = (frequency - lower) / max(center - lower, 1e-10)
                let falling = (upper - frequency) / max(upper - center, 1e-10)
                values[binIndex * melFilters + filterIndex] = max(0, min(rising, falling)) * enorm
            }
        }
        return MLXArray(values, [frequencyBins, melFilters])
    }
}

// MARK: - Language

private enum MiniCPMOLanguage {
    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        let rope: RoPELayer

        init(_ config: MiniCPMOConfiguration.TextConfiguration) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)

            _qProj.wrappedValue = Linear(
                config.hiddenSize, heads * headDim, bias: config.attentionBias,
            )
            _kProj.wrappedValue = Linear(
                config.hiddenSize, kvHeads * headDim, bias: config.attentionBias,
            )
            _vProj.wrappedValue = Linear(
                config.hiddenSize, kvHeads * headDim, bias: config.attentionBias,
            )
            _oProj.wrappedValue = Linear(
                heads * headDim, config.hiddenSize, bias: config.attentionBias,
            )
            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            rope = initializeRope(
                dims: headDim,
                base: config.ropeTheta,
                traditional: false,
                scalingConfig: config.ropeScaling,
                maxPositionEmbeddings: config.maxPositionEmbeddings,
            )
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let batch = x.dim(0)
            let length = x.dim(1)
            var queries = qProj(x)
            var keys = kProj(x)
            var values = vProj(x)

            queries = qNorm(queries.reshaped(batch, length, heads, headDim))
                .transposed(0, 2, 1, 3)
            keys = kNorm(keys.reshaped(batch, length, kvHeads, headDim))
                .transposed(0, 2, 1, 3)
            values = values.reshaped(batch, length, kvHeads, headDim)
                .transposed(0, 2, 1, 3)

            queries = applyRotaryPosition(rope, to: queries, offset: cache?.ropeOffset)
            keys = applyRotaryPosition(rope, to: keys, offset: cache?.ropeOffset)

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: mask,
            )
            .transposed(0, 2, 1, 3)
            .reshaped(batch, length, -1)

            return oProj(output)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
        @ModuleInfo var mlp: MLP

        init(_ config: MiniCPMOConfiguration.TextConfiguration) {
            _attention.wrappedValue = Attention(config)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps,
            )
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps,
            )
            _mlp.wrappedValue = MLP(
                dimensions: config.hiddenSize,
                hiddenDimensions: config.intermediateSize,
            )
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
            return h + mlp(postAttentionLayerNorm(h))
        }
    }

    final class Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm

        init(_ config: MiniCPMOConfiguration.TextConfiguration) {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in DecoderLayer(config) }
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache]? = nil,
        )
            -> MLXArray
        {
            var h = inputsEmbeds ?? embedTokens(inputs)
            let mask = createAttentionMask(h: h, cache: cache?.first)
            for (index, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[index])
            }
            return norm(h)
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?
        let config: MiniCPMOConfiguration.TextConfiguration
        let kvHeads: [Int]

        init(_ config: MiniCPMOConfiguration.TextConfiguration) {
            self.config = config
            _model.wrappedValue = Model(config)
            kvHeads = Array(repeating: config.kvHeads, count: config.hiddenLayers)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.hiddenSize,
                    config.vocabularySize,
                    bias: false,
                )
            }
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache]? = nil,
            state: LMOutput.State? = nil,
        )
            -> LMOutput
        {
            var out = model(inputs, inputsEmbeds: inputsEmbeds, cache: cache)
            if let lmHead {
                out = lmHead(out)
            } else {
                out = model.embedTokens.asLinear(out)
            }
            return LMOutput(logits: out, state: state)
        }
    }
}

// MARK: - Vision

private enum MiniCPMOVision {
    static func checkConvWeight(_ array: MLXArray) -> Bool {
        guard array.ndim == 4 else {
            return false
        }
        let shape = array.shape
        return shape[0] >= shape[1] && shape[0] >= shape[2] && shape[1] == shape[2]
    }

    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: MiniCPMOConfiguration.VisionConfiguration) {
            heads = config.attentionHeads
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _kProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _vProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _qProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let batch = x.dim(0)
            let length = x.dim(1)
            let q = qProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let k = kProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let v = vProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let output = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: nil,
            )
            return outProj(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let activation = GELU(approximation: .precise)

        init(_ config: MiniCPMOConfiguration.VisionConfiguration) {
            _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    final class EncoderLayer: Module {
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo var mlp: MLP

        init(_ config: MiniCPMOConfiguration.VisionConfiguration) {
            _layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
            _layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
            _selfAttention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(config)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = x + selfAttention(layerNorm1(x))
            return h + mlp(layerNorm2(h))
        }
    }

    final class Encoder: Module {
        @ModuleInfo(key: "layers") var layers: [EncoderLayer]

        init(_ config: MiniCPMOConfiguration.VisionConfiguration) {
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in EncoderLayer(config) }
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = x
            for layer in layers {
                h = layer(h)
            }
            return h
        }
    }

    final class Embeddings: Module {
        let config: MiniCPMOConfiguration.VisionConfiguration
        let patchSize: Int
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        init(_ config: MiniCPMOConfiguration.VisionConfiguration) {
            self.config = config
            patchSize = config.patchSize
            _patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.channels,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize),
                bias: true,
            )
            let side = config.imageSize / config.patchSize
            _positionEmbedding.wrappedValue = Embedding(
                embeddingCount: side * side,
                dimensions: config.hiddenSize,
            )
            super.init()
        }

        func callAsFunction(_ pixelValues: MLXArray, targetSizes: MLXArray) -> MLXArray {
            let embeddings = patchEmbedding(pixelValues).flattened(start: 1, end: 2)
            let positionIDs = buildPositionIDs(targetSizes: targetSizes, count: embeddings.dim(1))
            return embeddings + positionEmbedding(positionIDs)
        }

        private func buildPositionIDs(targetSizes: MLXArray, count: Int) -> MLXArray {
            let gridSide = Int(round(sqrt(Float(positionEmbedding.weight.dim(0)))))
            var all: [Int] = []
            all.reserveCapacity(targetSizes.dim(0) * count)
            let sizes = targetSizes.asArray(Int.self)
            for batchIndex in 0 ..< targetSizes.dim(0) {
                let targetH = max(sizes[batchIndex * 2], 1)
                let targetW = max(sizes[batchIndex * 2 + 1], 1)
                for index in 0 ..< count {
                    let h = index / targetW
                    let w = index % targetW
                    let bucketH = min(
                        Int(Float(h) / Float(targetH) * Float(gridSide)), gridSide - 1)
                    let bucketW = min(
                        Int(Float(w) / Float(targetW) * Float(gridSide)), gridSide - 1)
                    all.append(bucketH * gridSide + bucketW)
                }
            }
            return MLXArray(all, [targetSizes.dim(0), count])
        }
    }

    final class VisionModel: Module {
        @ModuleInfo var embeddings: Embeddings
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        init(_ config: MiniCPMOConfiguration.VisionConfiguration) {
            _embeddings.wrappedValue = Embeddings(config)
            _encoder.wrappedValue = Encoder(config)
            _postLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
            super.init()
        }

        func callAsFunction(_ pixelValues: MLXArray, targetSizes: MLXArray) -> MLXArray {
            var h = embeddings(pixelValues, targetSizes: targetSizes)
            h = h.asType(embeddings.patchEmbedding.weight.dtype)
            h = encoder(h)
            return postLayerNorm(h)
        }
    }
}

private final class MiniCPMOCrossAttention: Module {
    let heads: Int
    let headDim: Int
    let scale: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(dimensions: Int, heads: Int) {
        self.heads = heads
        headDim = dimensions / heads
        scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(dimensions, dimensions, bias: true)
        _kProj.wrappedValue = Linear(dimensions, dimensions, bias: true)
        _vProj.wrappedValue = Linear(dimensions, dimensions, bias: true)
        _outProj.wrappedValue = Linear(dimensions, dimensions, bias: true)
        super.init()
    }

    func callAsFunction(queries: MLXArray, keys: MLXArray, values: MLXArray) -> MLXArray {
        let batch = queries.dim(0)
        let qLength = queries.dim(1)
        let kvLength = keys.dim(1)
        let q = qProj(queries).reshaped(batch, qLength, heads, headDim).transposed(0, 2, 1, 3)
        let k = kProj(keys).reshaped(batch, kvLength, heads, headDim).transposed(0, 2, 1, 3)
        let v = vProj(values).reshaped(batch, kvLength, heads, headDim).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: scale, mask: nil,
        )
        return outProj(out.transposed(0, 2, 1, 3).reshaped(batch, qLength, -1))
    }
}

private final class MiniCPMOResampler: Module {
    let queryCount: Int
    let dimensions: Int
    @ParameterInfo(key: "query") var query: MLXArray
    @ModuleInfo(key: "kv_proj") var kvProj: Linear?
    @ModuleInfo(key: "attn") var attention: MiniCPMOCrossAttention
    @ModuleInfo(key: "ln_q") var lnQ: LayerNorm
    @ModuleInfo(key: "ln_kv") var lnKV: LayerNorm
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm
    @ParameterInfo(key: "proj") var proj: MLXArray

    init(queryCount: Int, dimensions: Int, heads: Int, kvDimensions: Int) {
        self.queryCount = queryCount
        self.dimensions = dimensions
        _query.wrappedValue = MLXArray.zeros([queryCount, dimensions])
        if kvDimensions != dimensions {
            _kvProj.wrappedValue = Linear(kvDimensions, dimensions, bias: false)
        }
        _attention.wrappedValue = MiniCPMOCrossAttention(dimensions: dimensions, heads: heads)
        _lnQ.wrappedValue = LayerNorm(dimensions: dimensions, eps: 1e-6)
        _lnKV.wrappedValue = LayerNorm(dimensions: dimensions, eps: 1e-6)
        _lnPost.wrappedValue = LayerNorm(dimensions: dimensions, eps: 1e-6)
        _proj.wrappedValue =
            MLXRandom.normal([dimensions, dimensions]) * pow(Float(dimensions), -0.5)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, targetSizes: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        var keys = kvProj.map { $0(x) } ?? x
        let pos = positionEmbedding(targetSizes: targetSizes, dtype: keys.dtype)
        keys = lnKV(keys + pos)
        var queries = lnQ(query)
        queries = broadcast(queries[.newAxis, 0..., 0...], to: [batch, queryCount, dimensions])
        let out = attention(queries: queries, keys: keys, values: keys)
        return lnPost(out).matmul(proj)
    }

    private func positionEmbedding(targetSizes: MLXArray, dtype: DType) -> MLXArray {
        let sizes = targetSizes.asArray(Int.self)
        let batch = targetSizes.dim(0)
        let patchCount = targetSizes.dim(0) > 0 ? max(1, sizes[0] * sizes[1]) : 1
        var values: [Float] = []
        values.reserveCapacity(batch * patchCount * dimensions)
        for batchIndex in 0 ..< batch {
            let h = max(sizes[batchIndex * 2], 1)
            let w = max(sizes[batchIndex * 2 + 1], 1)
            for row in 0 ..< h {
                for col in 0 ..< w {
                    appendSinCos(position: Float(col), dimensions: dimensions / 2, values: &values)
                    appendSinCos(position: Float(row), dimensions: dimensions / 2, values: &values)
                }
            }
        }
        return MLXArray(values, [batch, patchCount, dimensions]).asType(dtype)
    }

    private func appendSinCos(position: Float, dimensions: Int, values: inout [Float]) {
        let half = max(1, dimensions / 2)
        for index in 0 ..< half {
            let omega = pow(Float(10000), -Float(index) / Float(half))
            values.append(sin(position * omega))
        }
        for index in 0 ..< half {
            let omega = pow(Float(10000), -Float(index) / Float(half))
            values.append(cos(position * omega))
        }
    }
}

// MARK: - Audio

private enum MiniCPMOAudio {
    static func checkConv1dWeight(_ array: MLXArray, kernelSize: Int = 3) -> Bool {
        array.ndim == 3 && array.dim(1) == kernelSize
    }

    final class Attention: Module {
        let dimensions: Int
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: MiniCPMOConfiguration.AudioConfiguration) {
            dimensions = config.modelDimensions
            heads = config.encoderAttentionHeads
            headDim = dimensions / heads
            scale = pow(Float(headDim), -0.5)
            _kProj.wrappedValue = Linear(dimensions, dimensions, bias: false)
            _vProj.wrappedValue = Linear(dimensions, dimensions, bias: true)
            _qProj.wrappedValue = Linear(dimensions, dimensions, bias: true)
            _outProj.wrappedValue = Linear(dimensions, dimensions, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray, attentionMask: MLXArray?) -> MLXArray {
            let batch = x.dim(0)
            let length = x.dim(1)
            let q = qProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let k = kProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)
            let v = vProj(x).reshaped(batch, length, heads, headDim).transposed(0, 2, 1, 3)

            let mask: MLXArray? =
                if let attentionMask {
                    MLX.where(
                        (.!attentionMask)[0..., .newAxis, .newAxis, 0...],
                        MLXArray(-1e9, dtype: q.dtype),
                        MLXArray(0, dtype: q.dtype),
                    )
                } else {
                    nil
                }

            let out = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: mask,
            )
            return outProj(out.transposed(0, 2, 1, 3).reshaped(batch, length, dimensions))
        }
    }

    final class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn_layer_norm") var selfAttentionLayerNorm: LayerNorm
        @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let activation = GELU()

        init(_ config: MiniCPMOConfiguration.AudioConfiguration) {
            _selfAttentionLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.modelDimensions,
                eps: config.layerNormEps,
            )
            _finalLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.modelDimensions,
                eps: config.layerNormEps,
            )
            _selfAttention.wrappedValue = Attention(config)
            _fc1.wrappedValue = Linear(
                config.modelDimensions,
                config.encoderFFNDimensions,
                bias: true,
            )
            _fc2.wrappedValue = Linear(
                config.encoderFFNDimensions,
                config.modelDimensions,
                bias: true,
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray, attentionMask: MLXArray?) -> MLXArray {
            let attended =
                x + selfAttention(selfAttentionLayerNorm(x), attentionMask: attentionMask)
            return attended + fc2(activation(fc1(finalLayerNorm(attended))))
        }
    }

    final class Model: Module {
        let config: MiniCPMOConfiguration.AudioConfiguration

        @ModuleInfo var conv1: Conv1d
        @ModuleInfo var conv2: Conv1d
        @ModuleInfo(key: "embed_positions") var embedPositions: Embedding
        @ModuleInfo var layers: [EncoderLayer]
        @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

        init(_ config: MiniCPMOConfiguration.AudioConfiguration) {
            self.config = config
            _conv1.wrappedValue = Conv1d(
                inputChannels: config.melBins,
                outputChannels: config.modelDimensions,
                kernelSize: 3,
                stride: 1,
                padding: 1,
                bias: true,
            )
            _conv2.wrappedValue = Conv1d(
                inputChannels: config.modelDimensions,
                outputChannels: config.modelDimensions,
                kernelSize: 3,
                stride: 2,
                padding: 1,
                bias: true,
            )
            _embedPositions.wrappedValue = Embedding(
                embeddingCount: config.maxSourcePositions,
                dimensions: config.modelDimensions,
            )
            _layers.wrappedValue = (0 ..< config.encoderLayers).map { _ in EncoderLayer(config) }
            _layerNorm.wrappedValue = LayerNorm(
                dimensions: config.modelDimensions,
                eps: config.layerNormEps,
            )
            super.init()
        }

        func callAsFunction(_ inputFeatures: MLXArray, featureLengths: [Int]) -> MLXArray {
            var h = inputFeatures.transposed(0, 2, 1)
            h = gelu(conv1(h))
            h = gelu(conv2(h))

            let batch = h.dim(0)
            let length = h.dim(1)
            let positionIDs = broadcast(
                MLXArray(0 ..< length)[.newAxis, 0...],
                to: [batch, length],
            )
            h = h + embedPositions(positionIDs)

            let attentionMask = makeAttentionMask(
                featureLengths: featureLengths,
                sequenceLength: length,
            )
            for layer in layers {
                h = layer(h, attentionMask: attentionMask)
            }
            return layerNorm(h)
        }

        private func makeAttentionMask(featureLengths: [Int], sequenceLength: Int) -> MLXArray {
            var values: [Bool] = []
            values.reserveCapacity(featureLengths.count * sequenceLength)
            for featureLength in featureLengths {
                let convLength = min(max(((featureLength - 1) / 2) + 1, 1), sequenceLength)
                for index in 0 ..< sequenceLength {
                    values.append(index < convLength)
                }
            }
            return MLXArray(values, [featureLengths.count, sequenceLength])
        }
    }

    final class Projector: Module, UnaryLayer {
        @ModuleInfo var linear1: Linear
        @ModuleInfo var linear2: Linear

        init(inputDimensions: Int, outputDimensions: Int) {
            _linear1.wrappedValue = Linear(inputDimensions, outputDimensions, bias: true)
            _linear2.wrappedValue = Linear(outputDimensions, outputDimensions, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear2(relu(linear1(x)))
        }
    }
}

// MARK: - Model

public final class MiniCPMO: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "language_model") private var languageModel: MiniCPMOLanguage.LanguageModel
    @ModuleInfo(key: "vision_tower") private var visionTower: MiniCPMOVision.VisionModel
    @ModuleInfo private var resampler: MiniCPMOResampler
    @ModuleInfo(key: "audio_tower") private var audioTower: MiniCPMOAudio.Model?
    @ModuleInfo(key: "audio_projection_layer") private var audioProjectionLayer:
        MiniCPMOAudio.Projector?

    public let config: MiniCPMOConfiguration

    public init(_ config: MiniCPMOConfiguration) {
        self.config = config
        _languageModel.wrappedValue = MiniCPMOLanguage.LanguageModel(config.textConfiguration)
        _visionTower.wrappedValue = MiniCPMOVision.VisionModel(config.visionConfiguration)
        _resampler.wrappedValue = MiniCPMOResampler(
            queryCount: config.queryCount,
            dimensions: config.textConfiguration.hiddenSize,
            heads: max(1, config.textConfiguration.hiddenSize / 128),
            kvDimensions: config.visionConfiguration.hiddenSize,
        )
        if config.initAudio, let audioConfiguration = config.audioConfiguration {
            _audioTower.wrappedValue = MiniCPMOAudio.Model(audioConfiguration)
            _audioProjectionLayer.wrappedValue = MiniCPMOAudio.Projector(
                inputDimensions: audioConfiguration.encoderFFNDimensions / 4,
                outputDimensions: config.textConfiguration.hiddenSize,
            )
        }
        super.init()
    }

    public var vocabularySize: Int {
        config.textConfiguration.vocabularySize
    }

    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try languageModel.model.layers.map { _ in
            return try makeAttentionKVCache(configuration: configuration)
        }
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let inputIDs = input.text.tokens
        var inputEmbeddings: MLXArray?
        if input.image != nil || input.audio != nil {
            var textEmbeddings = languageModel.model.embedTokens(inputIDs)
            if let image = input.image {
                let features = try getVisionEmbeddings(
                    flatPixels: image.pixels,
                    targetFrames: image.frames ?? [],
                )
                .asType(textEmbeddings.dtype)
                textEmbeddings = try mergeVisionFeatures(
                    features: features,
                    inputEmbeddings: textEmbeddings,
                    bounds: input.video?.pixels,
                )
            }
            if let audio = input.audio {
                textEmbeddings = try mergeAudioFeatures(
                    audio: audio,
                    inputEmbeddings: textEmbeddings,
                )
            }
            inputEmbeddings = textEmbeddings
        }
        let output = languageModel(inputIDs, inputsEmbeds: inputEmbeddings, cache: cache)
        prefill.progress?(input.text.tokens.size, input.text.tokens.size)
        return .logits(output)
    }

    public func callAsFunction(
        _ input: LMInput.Text,
        cache: [KVCache]?,
        state: LMOutput.State?,
    )
        -> LMOutput
    {
        languageModel(input.tokens, cache: cache, state: state)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache).logits
    }

    private func getVisionEmbeddings(flatPixels: MLXArray, targetFrames: [THW]) throws -> MLXArray {
        var offset = 0
        var outputs: [MLXArray] = []
        let patch = config.visionConfiguration.patchSize
        let dtype = visionTower.embeddings.patchEmbedding.weight.dtype

        for frame in targetFrames {
            let height = frame.h * patch
            let width = frame.w * patch
            let count = 3 * height * width
            guard offset + count <= flatPixels.size else {
                throw VLMError.processing(
                    "MiniCPM-o pixel buffer is shorter than target frame metadata.")
            }
            let chunk = flatPixels[offset ..< (offset + count)]
                .reshaped(3, height, width)
                .transposed(1, 2, 0)
                .expandedDimensions(axis: 0)
                .asType(dtype)
            offset += count
            let target = MLXArray([frame.h, frame.w], [1, 2])
            let hidden = visionTower(chunk, targetSizes: target)
            outputs.append(resampler(hidden, targetSizes: target)[0, 0..., 0...])
        }

        guard !outputs.isEmpty else {
            return MLXArray.zeros([0, config.textConfiguration.hiddenSize])
        }
        return concatenated(outputs, axis: 0)
    }

    private func getAudioEmbeddings(audio: LMInput.ProcessedAudio) throws -> (
        embeddings: [MLXArray],
        bounds: [(Int, Int)],
    ) {
        guard let audioTower, let audioProjectionLayer else {
            throw VLMError.processing(
                "MiniCPM-o audio inputs require audio_config and audio weights.")
        }
        guard let metadata = audio.mask, metadata.ndim == 2, metadata.dim(1) == 3 else {
            throw VLMError.processing(
                "MiniCPM-o audio metadata is missing; use MiniCPMOProcessor for audio inputs.",
            )
        }

        var features = audio.features
        if features.ndim == 2 {
            features = features.expandedDimensions(axis: 0)
        }
        guard features.ndim == 3, features.dim(1) == audioTower.config.melBins else {
            throw VLMError.processing(
                "MiniCPM-o audio features must have shape [N, mel_bins, frames].")
        }

        let values = metadata.asArray(Int.self)
        let segmentCount = values.count / 3
        guard segmentCount == features.dim(0) else {
            throw VLMError.processing(
                "MiniCPM-o audio segment metadata does not match feature batch.",
            )
        }

        var featureLengths: [Int] = []
        var bounds: [(Int, Int)] = []
        featureLengths.reserveCapacity(segmentCount)
        bounds.reserveCapacity(segmentCount)
        for index in 0 ..< segmentCount {
            let featureLength = max(values[index * 3], 1)
            let start = values[index * 3 + 1]
            let end = values[index * 3 + 2]
            featureLengths.append(featureLength)
            bounds.append((start, end))
        }

        let audioStates = audioTower(features, featureLengths: featureLengths)
        let projected = audioProjectionLayer(audioStates)
        let poolStep = max(config.audioPoolStep, 1)

        var outputs: [MLXArray] = []
        outputs.reserveCapacity(segmentCount)
        for index in 0 ..< segmentCount {
            let validConvLength = min(
                max(((featureLengths[index] - 1) / 2) + 1, 1),
                projected.dim(1),
            )
            let pooledLength = max(1, ((validConvLength - poolStep) / poolStep) + 1)
            var rows: [MLXArray] = []
            rows.reserveCapacity(pooledLength)
            for pooledIndex in 0 ..< pooledLength {
                let start = pooledIndex * poolStep
                let end = min(start + poolStep, projected.dim(1))
                rows.append(mean(projected[index, start ..< end, 0...], axis: 0))
            }
            outputs.append(stacked(rows, axis: 0))
        }

        return (outputs, bounds)
    }

    private func mergeAudioFeatures(
        audio: LMInput.ProcessedAudio,
        inputEmbeddings: MLXArray,
    )
        throws -> MLXArray
    {
        let (segments, bounds) = try getAudioEmbeddings(audio: audio)
        guard !segments.isEmpty else {
            return inputEmbeddings
        }

        let result = inputEmbeddings
        for (index, segment) in segments.enumerated() {
            let bound = bounds[index]
            guard bound.1 > bound.0 else {
                continue
            }
            let indices = Array(bound.0 ..< bound.1)
            let usable = min(indices.count, segment.dim(0))
            guard usable > 0 else {
                continue
            }
            let tokenIndices = MLXArray(Array(indices.prefix(usable)))
            let segmentFeatures = segment[..<usable, 0...].asType(inputEmbeddings.dtype)
            result[0..., tokenIndices, 0...] = segmentFeatures.expandedDimensions(axis: 0)
        }
        return result
    }

    private func mergeVisionFeatures(
        features: MLXArray,
        inputEmbeddings: MLXArray,
        bounds: MLXArray?,
    )
        throws -> MLXArray
    {
        guard let bounds, bounds.size > 0 else {
            return inputEmbeddings
        }
        let boundValues = bounds.asArray(Int.self)
        var indices: [Int] = []
        for idx in stride(from: 0, to: boundValues.count, by: 2) {
            guard idx + 1 < boundValues.count else {
                continue
            }
            let start = boundValues[idx]
            let end = boundValues[idx + 1]
            if end > start {
                indices.append(contentsOf: start ..< end)
            }
        }
        guard indices.count == features.dim(0) else {
            throw VLMError.processing(
                "MiniCPM-o vision features (\(features.dim(0))) do not match placeholder count (\(indices.count)).",
            )
        }
        let result = inputEmbeddings
        result[0..., MLXArray(indices), 0...] = features.expandedDimensions(axis: 0)
        return result
    }

    public func sanitize(weights: [String: MLXArray], metadata _: [String: String]) -> [String:
        MLXArray]
    {
        sanitize(weights: weights)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)
        var inProjWeight: MLXArray?
        var inProjBias: MLXArray?

        for (originalKey, originalValue) in weights {
            var key = originalKey
            var value = originalValue

            if key.hasPrefix("tts.") || key.hasPrefix("audio_avg_pooler.") {
                continue
            }

            if key.hasPrefix("llm.") {
                key = key.replacingOccurrences(
                    of: "llm.", with: "language_model.", options: .anchored)
            } else if key.hasPrefix("vpm.") {
                key = key.replacingOccurrences(
                    of: "vpm.", with: "vision_tower.", options: .anchored)
            } else if key.hasPrefix("apm.") {
                key = key.replacingOccurrences(of: "apm.", with: "audio_tower.", options: .anchored)
            }

            guard
                key.hasPrefix("language_model.")
                    || key.hasPrefix("vision_tower.")
                    || key.hasPrefix("resampler.")
                    || (audioTower != nil && key.hasPrefix("audio_tower."))
                    || (audioProjectionLayer != nil && key.hasPrefix("audio_projection_layer."))
            else {
                continue
            }

            if key == "resampler.attn.in_proj_weight" {
                inProjWeight = value
                continue
            }
            if key == "resampler.attn.in_proj_bias" {
                inProjBias = value
                continue
            }
            if key.contains("position_ids") {
                continue
            }
            if key.hasSuffix("embeddings.patch_embedding.weight"),
                !MiniCPMOVision.checkConvWeight(value)
            {
                value = value.transposed(0, 2, 3, 1)
            }
            if key.hasSuffix("audio_tower.conv1.weight")
                || key.hasSuffix("audio_tower.conv2.weight"),
                !MiniCPMOAudio.checkConv1dWeight(value)
            {
                value = value.transposed(0, 2, 1)
            }
            if config.textConfiguration.tieWordEmbeddings, key == "language_model.lm_head.weight" {
                continue
            }

            sanitized[key] = value
        }

        if let inProjWeight {
            let parts = MLX.split(inProjWeight, parts: 3, axis: 0)
            sanitized["resampler.attn.q_proj.weight"] = parts[0]
            sanitized["resampler.attn.k_proj.weight"] = parts[1]
            sanitized["resampler.attn.v_proj.weight"] = parts[2]
        }
        if let inProjBias {
            let parts = MLX.split(inProjBias, parts: 3, axis: 0)
            sanitized["resampler.attn.q_proj.bias"] = parts[0]
            sanitized["resampler.attn.k_proj.bias"] = parts[1]
            sanitized["resampler.attn.v_proj.bias"] = parts[2]
        }

        return sanitized
    }
}
