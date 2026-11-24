import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct Phi4MultimodalConfiguration: Decodable, Sendable {
    public struct LoRAConfiguration: Decodable, Sendable {
        public let rank: Int
        public let alpha: Int

        enum CodingKeys: String, CodingKey {
            case rank = "r"
            case alpha = "lora_alpha"
        }

        public init(rank: Int, alpha: Int) {
            self.rank = rank
            self.alpha = alpha
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            rank = try c.decodeIfPresent(Int.self, forKey: .rank) ?? 1
            alpha = try c.decodeIfPresent(Int.self, forKey: .alpha) ?? rank
        }

        public var scale: Float {
            Float(alpha) / Float(max(rank, 1))
        }
    }

    public struct AudioProcessorConfiguration: Decodable, Sendable {
        public let config: AudioConfiguration?
    }

    public struct AudioConfiguration: Decodable, Sendable {
        public let attentionDimensions: Int
        public let attentionHeads: Int
        public let numBlocks: Int
        public let linearUnits: Int
        public let inputSize: Int
        public let timeReduction: Int
        public let kernelSize: Int
        public let activation: String
        public let convActivation: String
        public let convGLUType: String
        public let biasInGLU: Bool
        public let extPointwiseOutChannels: Int
        public let extPointwiseKernelSize: Int
        public let depthwiseSeparableOutChannels: Int
        public let depthwiseMultiplier: Int
        public let causal: Bool
        public let batchNorm: Bool
        public let cnnLayerNorm: Bool
        public let t5BiasMaxDistance: Int
        public let convChannels: Int
        public let chunkSize: Int
        public let leftChunk: Int

        enum CodingKeys: String, CodingKey {
            case attentionDimensions = "attention_dim"
            case attentionHeads = "attention_heads"
            case numBlocks = "num_blocks"
            case linearUnits = "linear_units"
            case inputSize = "input_size"
            case timeReduction = "time_reduction"
            case kernelSize = "kernel_size"
            case activation
            case convActivation = "conv_activation"
            case convGLUType = "conv_glu_type"
            case biasInGLU = "bias_in_glu"
            case extPointwiseOutChannels = "ext_pw_out_channel"
            case extPointwiseKernelSize = "ext_pw_kernel_size"
            case depthwiseSeparableOutChannels = "depthwise_seperable_out_channel"
            case depthwiseMultiplier = "depthwise_multiplier"
            case causal
            case batchNorm = "batch_norm"
            case cnnLayerNorm = "cnn_layer_norm"
            case t5BiasMaxDistance = "t5_bias_max_distance"
            case convChannels = "conv_channels"
            case chunkSize = "chunk_size"
            case leftChunk = "left_chunk"
        }

        public init(
            attentionDimensions: Int = 1024,
            attentionHeads: Int = 16,
            numBlocks: Int = 24,
            linearUnits: Int = 1536,
            inputSize: Int = 80,
            timeReduction: Int = 8,
            kernelSize: Int = 3,
            activation: String = "swish",
            convActivation: String = "swish",
            convGLUType: String = "swish",
            biasInGLU: Bool = true,
            extPointwiseOutChannels: Int = 1024,
            extPointwiseKernelSize: Int = 1,
            depthwiseSeparableOutChannels: Int = 1024,
            depthwiseMultiplier: Int = 1,
            causal: Bool = true,
            batchNorm: Bool = false,
            cnnLayerNorm: Bool = true,
            t5BiasMaxDistance: Int = 500,
            convChannels: Int = 1024,
            chunkSize: Int = -1,
            leftChunk: Int = 18,
        ) {
            self.attentionDimensions = attentionDimensions
            self.attentionHeads = attentionHeads
            self.numBlocks = numBlocks
            self.linearUnits = linearUnits
            self.inputSize = inputSize
            self.timeReduction = timeReduction
            self.kernelSize = kernelSize
            self.activation = activation
            self.convActivation = convActivation
            self.convGLUType = convGLUType
            self.biasInGLU = biasInGLU
            self.extPointwiseOutChannels = extPointwiseOutChannels
            self.extPointwiseKernelSize = extPointwiseKernelSize
            self.depthwiseSeparableOutChannels = depthwiseSeparableOutChannels
            self.depthwiseMultiplier = depthwiseMultiplier
            self.causal = causal
            self.batchNorm = batchNorm
            self.cnnLayerNorm = cnnLayerNorm
            self.t5BiasMaxDistance = t5BiasMaxDistance
            self.convChannels = convChannels
            self.chunkSize = chunkSize
            self.leftChunk = leftChunk
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            attentionDimensions =
                try c.decodeIfPresent(Int.self, forKey: .attentionDimensions) ?? 1024
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            numBlocks = try c.decodeIfPresent(Int.self, forKey: .numBlocks) ?? 24
            linearUnits = try c.decodeIfPresent(Int.self, forKey: .linearUnits) ?? 1536
            inputSize = try c.decodeIfPresent(Int.self, forKey: .inputSize) ?? 80
            timeReduction = try c.decodeIfPresent(Int.self, forKey: .timeReduction) ?? 8
            kernelSize = try c.decodeIfPresent(Int.self, forKey: .kernelSize) ?? 3
            activation = try c.decodeIfPresent(String.self, forKey: .activation) ?? "swish"
            convActivation =
                try c.decodeIfPresent(String.self, forKey: .convActivation) ?? "swish"
            convGLUType = try c.decodeIfPresent(String.self, forKey: .convGLUType) ?? "swish"
            biasInGLU = try c.decodeIfPresent(Bool.self, forKey: .biasInGLU) ?? true
            extPointwiseOutChannels =
                try c.decodeIfPresent(Int.self, forKey: .extPointwiseOutChannels) ?? 1024
            extPointwiseKernelSize =
                try c.decodeIfPresent(Int.self, forKey: .extPointwiseKernelSize) ?? 1
            depthwiseSeparableOutChannels =
                try c.decodeIfPresent(Int.self, forKey: .depthwiseSeparableOutChannels)
                ?? 1024
            depthwiseMultiplier =
                try c.decodeIfPresent(Int.self, forKey: .depthwiseMultiplier) ?? 1
            causal = try c.decodeIfPresent(Bool.self, forKey: .causal) ?? true
            batchNorm = try c.decodeIfPresent(Bool.self, forKey: .batchNorm) ?? false
            cnnLayerNorm = try c.decodeIfPresent(Bool.self, forKey: .cnnLayerNorm) ?? true
            t5BiasMaxDistance =
                try c.decodeIfPresent(Int.self, forKey: .t5BiasMaxDistance) ?? 500
            convChannels = try c.decodeIfPresent(Int.self, forKey: .convChannels) ?? 1024
            chunkSize = try c.decodeIfPresent(Int.self, forKey: .chunkSize) ?? -1
            leftChunk = try c.decodeIfPresent(Int.self, forKey: .leftChunk) ?? 18
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numAttentionHeads: Int
        public let numHiddenLayers: Int
        public let patchSize: Int
        public let imageSize: Int
        public let numChannels: Int
        public let layerNormEps: Float

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numAttentionHeads = "num_attention_heads"
            case numHiddenLayers = "num_hidden_layers"
            case patchSize = "patch_size"
            case imageSize = "image_size"
            case numChannels = "num_channels"
            case layerNormEps = "layer_norm_eps"
        }

        public init(
            modelType: String = "siglip2_vision_model",
            hiddenSize: Int = 1152,
            intermediateSize: Int = 4304,
            numAttentionHeads: Int = 16,
            numHiddenLayers: Int = 27,
            patchSize: Int = 14,
            imageSize: Int = 448,
            numChannels: Int = 3,
            layerNormEps: Float = 1e-6,
        ) {
            self.modelType = modelType
            self.hiddenSize = hiddenSize
            self.intermediateSize = intermediateSize
            self.numAttentionHeads = numAttentionHeads
            self.numHiddenLayers = numHiddenLayers
            self.patchSize = patchSize
            self.imageSize = imageSize
            self.numChannels = numChannels
            self.layerNormEps = layerNormEps
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType) ?? "siglip2_vision_model"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
            numAttentionHeads = try c.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
            numHiddenLayers = try c.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 27
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 448
            numChannels = try c.decodeIfPresent(Int.self, forKey: .numChannels) ?? 3
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
        }
    }

    public let modelType: String
    public let vocabularySize: Int
    public let hiddenSize: Int
    public let hiddenLayers: Int
    public let intermediateSize: Int
    public let attentionHeads: Int
    public let keyValueHeads: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let partialRotaryFactor: Float
    public let maxPositionEmbeddings: Int
    public let imageTokenIndex: Int
    public let audioTokenIndex: Int
    public let padTokenID: Int
    public let tieWordEmbeddings: Bool
    public let visionLoRA: LoRAConfiguration?
    public let speechLoRA: LoRAConfiguration?
    public let visionConfiguration: VisionConfiguration
    public let mmHiddenSize: Int
    public let audioConfiguration: AudioConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case keyValueHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case partialRotaryFactor = "partial_rotary_factor"
        case maxPositionEmbeddings = "max_position_embeddings"
        case imageTokenIndex = "image_token_index"
        case audioTokenIndex = "audio_token_index"
        case padTokenID = "pad_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
        case visionLoRA = "vision_lora"
        case speechLoRA = "speech_lora"
        case visionConfiguration = "vision_config"
        case mmHiddenSize = "mm_hidden_size"
        case audioProcessor = "audio_processor"
        case audioConfiguration = "audio_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "phi4mm"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 200_064
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 3072
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 8192
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 24
        keyValueHeads =
            try c.decodeIfPresent(Int.self, forKey: .keyValueHeads) ?? attentionHeads
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        partialRotaryFactor =
            try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.75
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? -200
        audioTokenIndex = try c.decodeIfPresent(Int.self, forKey: .audioTokenIndex) ?? 200_011
        padTokenID = try c.decodeIfPresent(Int.self, forKey: .padTokenID) ?? 199_999
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
        visionLoRA = try c.decodeIfPresent(LoRAConfiguration.self, forKey: .visionLoRA)
        speechLoRA = try c.decodeIfPresent(LoRAConfiguration.self, forKey: .speechLoRA)
        visionConfiguration =
            try c.decodeIfPresent(VisionConfiguration.self, forKey: .visionConfiguration)
            ?? .init()
        mmHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .mmHiddenSize)
            ?? visionConfiguration.hiddenSize
        let nested = try c.decodeIfPresent(
            AudioProcessorConfiguration.self,
            forKey: .audioProcessor,
        )?.config
        let direct = try c.decodeIfPresent(
            AudioConfiguration.self,
            forKey: .audioConfiguration,
        )
        audioConfiguration =
            nested
            ?? direct
            ?? .init()
    }
}

public struct Phi4MultimodalProcessorConfiguration: Decodable, Sendable {
    public let imageToken: String
    public let imageTokenID: Int
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let imagePatchSize: Int
    public let imageMinNumPatches: Int
    public let imageMaxNumPatches: Int
    public let audioToken: String
    public let audioTokenID: Int
    public let audioCompressionRate: Int
    public let audioDownsampleRate: Int
    public let audioFeatureStride: Int

    private struct AudioProcessorConfiguration: Decodable {
        let config: Phi4MultimodalConfiguration.AudioConfiguration?
    }

    enum CodingKeys: String, CodingKey {
        case imageToken = "image_token"
        case imageTokenID = "image_token_index"
        case imageProcessor = "image_processor"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case patchSize = "patch_size"
        case minNumPatches = "min_num_patches"
        case maxNumPatches = "max_num_patches"
        case audioToken = "audio_token"
        case audioTokenID = "audio_token_index"
        case audioProcessor = "audio_processor"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let imageProcessor =
            try? c.nestedContainer(keyedBy: CodingKeys.self, forKey: .imageProcessor)
        imageToken = try c.decodeIfPresent(String.self, forKey: .imageToken) ?? "<image>"
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? -200
        imageMean =
            try imageProcessor?.decodeIfPresent([CGFloat].self, forKey: .imageMean)
            ?? c.decodeIfPresent([CGFloat].self, forKey: .imageMean)
            ?? [0.5, 0.5, 0.5]
        imageStd =
            try imageProcessor?.decodeIfPresent([CGFloat].self, forKey: .imageStd)
            ?? c.decodeIfPresent([CGFloat].self, forKey: .imageStd)
            ?? [0.5, 0.5, 0.5]
        imagePatchSize =
            try imageProcessor?.decodeIfPresent(Int.self, forKey: .patchSize)
            ?? c.decodeIfPresent(Int.self, forKey: .patchSize)
            ?? 14
        imageMinNumPatches =
            try imageProcessor?.decodeIfPresent(Int.self, forKey: .minNumPatches)
            ?? c.decodeIfPresent(Int.self, forKey: .minNumPatches)
            ?? 256
        imageMaxNumPatches =
            try imageProcessor?.decodeIfPresent(Int.self, forKey: .maxNumPatches)
            ?? c.decodeIfPresent(Int.self, forKey: .maxNumPatches)
            ?? 3600
        audioToken = try c.decodeIfPresent(String.self, forKey: .audioToken) ?? "<|endoftext11|>"
        audioTokenID = try c.decodeIfPresent(Int.self, forKey: .audioTokenID) ?? 200_011
        let audioConfiguration =
            try c.decodeIfPresent(AudioProcessorConfiguration.self, forKey: .audioProcessor)?
            .config
        audioCompressionRate = audioConfiguration?.timeReduction ?? 8
        audioDownsampleRate = 1
        audioFeatureStride = 1
    }
}

public struct Phi4MultimodalProcessor: UserInputProcessor {
    private let config: Phi4MultimodalProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageTokenID: Int
    private let audioFeatureExtractor: Phi4MultimodalAudio.AudioFeatureExtractor

    public init(_ config: Phi4MultimodalProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        imageTokenID = tokenizer.convertTokenToId(config.imageToken) ?? config.imageTokenID
        audioFeatureExtractor = Phi4MultimodalAudio.AudioFeatureExtractor(
            audioCompressionRate: config.audioCompressionRate,
            audioDownsampleRate: config.audioDownsampleRate,
            audioFeatureStride: config.audioFeatureStride,
        )
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Phi4MultimodalMessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(
            messages: messages,
            tools: input.tools,
            additionalContext: input.additionalContext,
        )

        let images = MediaProcessing.images(from: input)
        let processedImage = try preprocess(images: images, processing: input.processing)

        let audios = MediaProcessing.audios(from: input)
        var processedAudio: LMInput.ProcessedAudio?
        var audioEmbedSizes: [Int32] = []
        if !audios.isEmpty {
            let audioOutput = try await preprocess(audios: audios, processing: input.processing)
            processedAudio = audioOutput.audio
            audioEmbedSizes = audioOutput.embedSizes
        }

        if !audioEmbedSizes.isEmpty {
            var audioIndex = 0
            var expandedTokens: [Int] = []
            expandedTokens.reserveCapacity(
                promptTokens.count + audioEmbedSizes.reduce(0) { $0 + Int($1) },
            )
            for token in promptTokens {
                if token == config.audioTokenID {
                    guard audioIndex < audioEmbedSizes.count else {
                        throw VLMError.processing(
                            "Phi-4 Multimodal has more audio tokens than audio inputs.")
                    }
                    expandedTokens.append(
                        contentsOf: Array(
                            repeating: config.audioTokenID,
                            count: Int(audioEmbedSizes[audioIndex]),
                        ),
                    )
                    audioIndex += 1
                } else {
                    expandedTokens.append(token)
                }
            }
            guard audioIndex == audioEmbedSizes.count else {
                throw VLMError.processing(
                    "Phi-4 Multimodal audio inputs require matching prompt audio tokens.")
            }
            promptTokens = expandedTokens
        }

        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: promptArray, mask: ones(like: promptArray).asType(.int8)),
            image: processedImage,
            audio: processedAudio,
        )
    }

    private func preprocess(images: [UserInput.Image], processing: UserInput.Processing?)
        throws -> LMInput.ProcessedImage?
    {
        guard !images.isEmpty else {
            return nil
        }

        var patches: [MLXArray] = []
        var metadata: [Int32] = []
        patches.reserveCapacity(images.count)
        metadata.reserveCapacity(images.count * (config.imageMaxNumPatches + 2))

        for image in images {
            let processed = try preprocess(image: image.asCIImage(), processing: processing)
            patches.append(processed.patches)
            metadata.append(Int32(processed.patchHeight))
            metadata.append(Int32(processed.patchWidth))
            metadata.append(contentsOf: processed.mask)
        }

        return LMInput.ProcessedImage(
            pixels: stacked(patches, axis: 0),
            positionIds: MLXArray(metadata, [images.count, config.imageMaxNumPatches + 2]),
        )
    }

    private struct ProcessedImage {
        let patches: MLXArray
        let mask: [Int32]
        let patchHeight: Int
        let patchWidth: Int
    }

    private func preprocess(image: CIImage, processing: UserInput.Processing?) throws
        -> ProcessedImage
    {
        let source = MediaProcessing.apply(image, processing: processing).toSRGB()
        let target = targetImageSize(for: source.extent.size)
        var processed = MediaProcessing.resampleBicubic(
            source,
            to: CGSize(width: target.width, height: target.height),
        )
        processed = MediaProcessing.normalize(processed, mean: imageMeanTuple, std: imageStdTuple)
        let pixels = MediaProcessing.asMLXArray(processed)
        let patchHeight = max(pixels.dim(-2) / config.imagePatchSize, 1)
        let patchWidth = max(pixels.dim(-1) / config.imagePatchSize, 1)
        let rawPatches = Self.imagePatches(pixels, patchSize: config.imagePatchSize)
        let validPatches = min(rawPatches.dim(0), config.imageMaxNumPatches)
        let paddedPatches: MLXArray =
            if rawPatches.dim(0) < config.imageMaxNumPatches {
                padded(
                    rawPatches,
                    widths: [.init((0, config.imageMaxNumPatches - rawPatches.dim(0))), 0],
                )
            } else {
                rawPatches[..<config.imageMaxNumPatches, 0...]
            }
        let mask = (0 ..< config.imageMaxNumPatches).map { index in
            Int32(index < validPatches ? 1 : 0)
        }
        return .init(
            patches: paddedPatches,
            mask: mask,
            patchHeight: patchHeight,
            patchWidth: patchWidth,
        )
    }

    private var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (
            config.imageMean.indices.contains(0) ? config.imageMean[0] : 0.5,
            config.imageMean.indices.contains(1) ? config.imageMean[1] : 0.5,
            config.imageMean.indices.contains(2) ? config.imageMean[2] : 0.5,
        )
    }

    private var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (
            config.imageStd.indices.contains(0) ? config.imageStd[0] : 0.5,
            config.imageStd.indices.contains(1) ? config.imageStd[1] : 0.5,
            config.imageStd.indices.contains(2) ? config.imageStd[2] : 0.5,
        )
    }

    private func targetImageSize(for size: CGSize) -> (height: Int, width: Int) {
        let patchSize = max(config.imagePatchSize, 1)
        let height = max(Int(size.height.rounded()), 1)
        let width = max(Int(size.width.rounded()), 1)
        let currentPatches = max((height / patchSize) * (width / patchSize), 1)
        let targetPatches: Int =
            if currentPatches < config.imageMinNumPatches {
                config.imageMinNumPatches
            } else if currentPatches > config.imageMaxNumPatches {
                config.imageMaxNumPatches
            } else {
                currentPatches
            }
        return Self.imageSizeForPatchBudget(
            height: height,
            width: width,
            patchSize: patchSize,
            maxPatches: targetPatches,
        )
    }

    private static func imageSizeForPatchBudget(
        height: Int,
        width: Int,
        patchSize: Int,
        maxPatches: Int,
    )
        -> (height: Int, width: Int)
    {
        let aspectRatio = Double(width) / Double(max(height, 1))
        var heightPatches = max(Int(sqrt(Double(maxPatches) / max(aspectRatio, 0.0001))), 1)
        var widthPatches = max(Int(Double(heightPatches) * aspectRatio), 1)
        while heightPatches * widthPatches > maxPatches {
            if heightPatches > widthPatches {
                heightPatches -= 1
            } else {
                widthPatches -= 1
            }
            heightPatches = max(heightPatches, 1)
            widthPatches = max(widthPatches, 1)
        }
        return (heightPatches * patchSize, widthPatches * patchSize)
    }

    private static func imagePatches(_ pixels: MLXArray, patchSize: Int) -> MLXArray {
        let height = pixels.dim(-2)
        let width = pixels.dim(-1)
        let patchHeight = max(height / patchSize, 1)
        let patchWidth = max(width / patchSize, 1)
        let hwc = pixels[0].transposed(1, 2, 0)
        return
            hwc
            .reshaped(patchHeight, patchSize, patchWidth, patchSize, 3)
            .transposed(0, 2, 1, 3, 4)
            .reshaped(patchHeight * patchWidth, patchSize * patchSize * 3)
    }

    private func preprocess(audios: [UserInput.Audio], processing: UserInput.Processing)
        async throws -> (audio: LMInput.ProcessedAudio, embedSizes: [Int32])
    {
        var audioProcessing = processing.audio
        audioProcessing.sampleRate = Double(audioFeatureExtractor.samplingRate)
        audioProcessing.channels = 1

        var samples: [(samples: MLXArray, sampleRate: Int)] = []
        samples.reserveCapacity(audios.count)
        for audio in audios {
            try await samples.append(
                (
                    samples: audio.asMLXArray(processing: audioProcessing),
                    sampleRate: audioFeatureExtractor.samplingRate,
                ))
        }

        let extracted = try audioFeatureExtractor(samples)
        let embedSizes = extracted.audioEmbedSizes.asArray(Int32.self)
        let metadata = Self.packAudioMetadata(
            embedSizes: embedSizes,
            attentionMask: extracted.audioAttentionMask,
            maxLength: extracted.inputAudioEmbeds.dim(1),
        )
        return (
            LMInput.ProcessedAudio(features: extracted.inputAudioEmbeds, mask: metadata),
            embedSizes,
        )
    }

    private static func packAudioMetadata(
        embedSizes: [Int32],
        attentionMask: MLXArray?,
        maxLength: Int,
    )
        -> MLXArray
    {
        if let attentionMask {
            let values = attentionMask.asArray(Int32.self)
            var packed: [Int32] = []
            packed.reserveCapacity(embedSizes.count * (maxLength + 1))
            for index in 0 ..< embedSizes.count {
                packed.append(embedSizes[index])
                let start = index * maxLength
                packed.append(contentsOf: values[start ..< (start + maxLength)])
            }
            return MLXArray(packed, [embedSizes.count, maxLength + 1])
        }
        return MLXArray(embedSizes, [embedSizes.count, 1])
    }

}

public struct Phi4MultimodalMessageGenerator: MessageGenerator {
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
                    "content": message.images.map { _ in ["type": "image"] }
                        + message.audios.map { _ in ["type": "audio"] }
                        + [["type": "text", "text": message.content]],
                ]
            }
        addToolMetadata(to: &dictionary, for: message)
        return dictionary
    }
}

// MARK: - Vision

private enum Phi4MultimodalVision {
    final class Attention: Module {
        let heads: Int
        let headDimensions: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: Phi4MultimodalConfiguration.VisionConfiguration) {
            heads = config.numAttentionHeads
            headDimensions = config.hiddenSize / config.numAttentionHeads
            scale = pow(Float(headDimensions), -0.5)
            _qProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _kProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _vProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
            let batch = x.dim(0)
            let length = x.dim(1)
            let queries = qProj(x)
                .reshaped(batch, length, heads, headDimensions)
                .transposed(0, 2, 1, 3)
            let keys = kProj(x)
                .reshaped(batch, length, heads, headDimensions)
                .transposed(0, 2, 1, 3)
            let values = vProj(x)
                .reshaped(batch, length, heads, headDimensions)
                .transposed(0, 2, 1, 3)

            var scores = matmul(queries * scale, keys.transposed(0, 1, 3, 2))
            if let mask {
                scores = scores + mask.asType(scores.dtype)
            }
            let attention = softmax(scores, axis: -1)
            let output = matmul(attention, values)
                .transposed(0, 2, 1, 3)
                .reshaped(batch, length, -1)
            return outProj(output)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        init(_ config: Phi4MultimodalConfiguration.VisionConfiguration) {
            _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(gelu(fc1(x)))
        }
    }

    final class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

        init(_ config: Phi4MultimodalConfiguration.VisionConfiguration) {
            _selfAttention.wrappedValue = Attention(config)
            _layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
            _mlp.wrappedValue = MLP(config)
            _layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
            let hidden = x + selfAttention(layerNorm1(x), mask: mask)
            return hidden + mlp(layerNorm2(hidden))
        }
    }

    final class Encoder: Module {
        @ModuleInfo var layers: [EncoderLayer]

        init(_ config: Phi4MultimodalConfiguration.VisionConfiguration) {
            _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
                EncoderLayer(config)
            }
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            outputHiddenStates: Bool = false,
            mask: MLXArray? = nil,
        )
            -> (MLXArray, [MLXArray]?)
        {
            var hidden = x
            var hiddenStates: [MLXArray]? = outputHiddenStates ? [hidden] : nil
            for layer in layers {
                hidden = layer(hidden, mask: mask)
                hiddenStates?.append(hidden)
            }
            return (hidden, hiddenStates)
        }
    }

    final class VisionEmbeddings: Module {
        let positionEmbeddingSize: Int
        let numPositions: Int

        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        init(_ config: Phi4MultimodalConfiguration.VisionConfiguration) {
            positionEmbeddingSize = max(config.imageSize / config.patchSize, 1)
            numPositions = positionEmbeddingSize * positionEmbeddingSize
            _patchEmbedding.wrappedValue = Linear(
                config.patchSize * config.patchSize * config.numChannels,
                config.hiddenSize,
                bias: true,
            )
            _positionEmbedding.wrappedValue = Embedding(
                embeddingCount: numPositions,
                dimensions: config.hiddenSize,
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray, spatialShapes: MLXArray? = nil) -> MLXArray {
            let patchEmbeddings = patchEmbedding(x.asType(patchEmbedding.weight.dtype))
            if let spatialShapes {
                let positional = resizePositionalEmbeddings(
                    spatialShapes: spatialShapes,
                    maxLength: x.dim(1),
                )
                return patchEmbeddings + positional.asType(patchEmbeddings.dtype)
            }

            let positionIDs = MLXArray(0 ..< numPositions).expandedDimensions(axis: 0)
            return patchEmbeddings + positionEmbedding(positionIDs)
        }

        private func resizePositionalEmbeddings(
            spatialShapes: MLXArray,
            maxLength: Int,
        )
            -> MLXArray
        {
            let batchSize = spatialShapes.dim(0)
            let embedDimensions = positionEmbedding.weight.dim(-1)
            let shapeValues = spatialShapes.asArray(Int32.self).map(Int.init)
            let source = positionEmbedding.weight
                .reshaped(positionEmbeddingSize, positionEmbeddingSize, embedDimensions)
                .transposed(2, 0, 1)
                .expandedDimensions(axis: 0)
            var rows: [MLXArray] = []
            rows.reserveCapacity(batchSize)

            for index in 0 ..< batchSize {
                let height = max(shapeValues[index * 2], 1)
                let width = max(shapeValues[index * 2 + 1], 1)
                var resized = bicubicInterpolate(
                    source,
                    size: (height, width),
                    alignCorners: false,
                    antialias: true,
                )
                resized = resized[0]
                    .transposed(1, 2, 0)
                    .reshaped(height * width, embedDimensions)
                    .asType(positionEmbedding.weight.dtype)
                if resized.dim(0) < maxLength {
                    let padding = broadcast(
                        resized[0 ..< 1, 0...],
                        to: [maxLength - resized.dim(0), embedDimensions],
                    )
                    resized = concatenated([resized, padding], axis: 0)
                }
                rows.append(resized[..<maxLength, 0...])
            }

            return stacked(rows, axis: 0)
        }
    }

    final class SigLip2VisionModel: Module {
        @ModuleInfo var embeddings: VisionEmbeddings
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        init(_ config: Phi4MultimodalConfiguration.VisionConfiguration) {
            _embeddings.wrappedValue = VisionEmbeddings(config)
            _encoder.wrappedValue = Encoder(config)
            _postLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize,
                eps: config.layerNormEps,
            )
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            spatialShapes: MLXArray? = nil,
            pixelAttentionMask: MLXArray? = nil,
            outputHiddenStates: Bool = false,
        )
            -> (MLXArray, MLXArray, [MLXArray]?)
        {
            var hidden = embeddings(x, spatialShapes: spatialShapes)
            hidden = hidden.asType(embeddings.patchEmbedding.weight.dtype)

            let mask: MLXArray? = pixelAttentionMask.map { pixelMask in
                MLX.where(
                    pixelMask.expandedDimensions(axis: 1).expandedDimensions(axis: 1).asType(.bool),
                    MLXArray(0.0, dtype: hidden.dtype),
                    MLXArray(-1e9, dtype: hidden.dtype),
                )
            }
            let (finalOutput, hiddenStates) = encoder(
                hidden,
                outputHiddenStates: outputHiddenStates,
                mask: mask,
            )
            return (postLayerNorm(finalOutput), hidden, hiddenStates)
        }
    }

    final class VisionTower: Module {
        @ModuleInfo(key: "vision_tower") var visionTower: SigLip2VisionModel

        init(_ config: Phi4MultimodalConfiguration.VisionConfiguration) {
            _visionTower.wrappedValue = SigLip2VisionModel(config)
            super.init()
        }

        func callAsFunction(
            pixelValues: MLXArray,
            pixelAttentionMask: MLXArray?,
            spatialShapes: MLXArray?,
        )
            -> [MLXArray]
        {
            let (_, _, hiddenStates) = visionTower(
                pixelValues,
                spatialShapes: spatialShapes,
                pixelAttentionMask: pixelAttentionMask,
                outputHiddenStates: true,
            )
            let allHiddenStates = hiddenStates ?? []
            let selected =
                allHiddenStates.count >= 2
                ? allHiddenStates[allHiddenStates.count - 2]
                : allHiddenStates.last ?? pixelValues

            guard let spatialShapes else {
                return (0 ..< selected.dim(0)).map { selected[$0] }
            }

            let shapeValues = spatialShapes.asArray(Int32.self).map(Int.init)
            var features: [MLXArray] = []
            features.reserveCapacity(selected.dim(0))
            for index in 0 ..< selected.dim(0) {
                let height = max(shapeValues[index * 2], 1)
                let width = max(shapeValues[index * 2 + 1], 1)
                features.append(selected[index, ..<min(height * width, selected.dim(1)), 0...])
            }
            return features
        }
    }
}

// MARK: - Audio

public enum Phi4MultimodalAudio {
    public static func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var audioWeights: [String: MLXArray] = [:]
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (key, value) in weights {
            if key.contains("position_ids") {
                continue
            }

            if key.contains("embed_tokens_extend.audio_embed.encoder.") {
                let newKey = key.replacingOccurrences(
                    of: "model.embed_tokens_extend.audio_embed.encoder.",
                    with: "audio_encoder.",
                )
                audioWeights[newKey] = value
                continue
            }

            if key.contains("embed_tokens_extend.audio_embed.audio_projection.") {
                var newKey = key.replacingOccurrences(
                    of: "model.embed_tokens_extend.audio_embed.audio_projection.",
                    with: "audio_projection.",
                )
                newKey = newKey.replacingOccurrences(of: ".speech.0.", with: ".speech.proj_0.")
                newKey = newKey.replacingOccurrences(of: ".speech.2.", with: ".speech.proj_2.")
                newKey = newKey.replacingOccurrences(of: ".vision.0.", with: ".vision.proj_0.")
                newKey = newKey.replacingOccurrences(of: ".vision.2.", with: ".vision.proj_2.")
                sanitized[newKey] = value
                continue
            }

            sanitized[key] = value
        }

        sanitized.merge(sanitizeAudioEncoderWeights(audioWeights)) { _, new in new }
        return sanitized
    }

    static func sanitizeAudioEncoderWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (key, value) in weights {
            var newKey = remapAudioEncoderKey(key)
            var adjusted = value

            if newKey.contains(".net.0.linear.") {
                newKey = newKey.replacingOccurrences(of: ".net.0.linear.", with: ".net_0.linear.")
            } else if newKey.contains(".net.2.") {
                newKey = newKey.replacingOccurrences(of: ".net.2.", with: ".net_2.")
            }
            if newKey.isEmpty {
                continue
            }

            if newKey.contains("glu.b1") || newKey.contains("glu.b2"), adjusted.ndim == 3 {
                sanitized[newKey] = adjusted.reshaped(-1)
                continue
            }

            if newKey.hasSuffix(".weight"), adjusted.ndim == 4 {
                sanitized[newKey] = adjusted.transposed(0, 2, 3, 1)
                continue
            }

            if newKey.hasSuffix(".weight"), adjusted.ndim == 3 {
                if newKey.contains("glu.ext_pw_conv_1d")
                    || newKey.contains("conv.ext_pw_conv_1d")
                {
                    adjusted = adjusted[0..., 0..., 0]
                } else {
                    adjusted = adjusted.transposed(0, 2, 1)
                }
                sanitized[newKey] = adjusted
                continue
            }

            sanitized[newKey] = adjusted
        }

        return sanitized
    }

    private static func remapAudioEncoderKey(_ key: String) -> String {
        guard let range = key.range(of: "embed.conv.") else {
            return key
        }

        let prefix = key[..<range.lowerBound]
        let suffix = key[range.upperBound...]
        let parts = suffix.split(separator: ".", maxSplits: 1, omittingEmptySubsequences: false)
        guard parts.count == 2, let index = Int(parts[0]) else {
            return key
        }

        let parameter = parts[1]
        switch index {
        case 0:
            return "\(prefix)embed.conv_0.\(parameter)"
        case 1, 4, 7:
            return ""
        case 2:
            return "\(prefix)embed.dw_pw_layers.0.dw.\(parameter)"
        case 3:
            return "\(prefix)embed.dw_pw_layers.0.pw.\(parameter)"
        case 5:
            return "\(prefix)embed.dw_pw_layers.1.dw.\(parameter)"
        case 6:
            return "\(prefix)embed.dw_pw_layers.1.pw.\(parameter)"
        default:
            return key
        }
    }

    public struct AudioFeatureExtractor: Sendable {
        public struct Output {
            public let inputAudioEmbeds: MLXArray
            public let audioEmbedSizes: MLXArray
            public let audioAttentionMask: MLXArray?
        }

        public let samplingRate: Int
        public let compressionRate: Int
        public let qformerCompressionRate: Int
        public let featureStride: Int

        private let fftLength = 512
        private let windowLength = 400
        private let hopLength = 160
        private let featureSize = 80
        private let hammingWindow: MaterializedArray
        private let melFilters: MaterializedArray

        public init(
            samplingRate: Int = 16000,
            audioCompressionRate: Int = 8,
            audioDownsampleRate: Int = 1,
            audioFeatureStride: Int = 1,
        ) {
            self.samplingRate = samplingRate
            compressionRate = max(audioCompressionRate, 1)
            qformerCompressionRate = max(audioDownsampleRate, 1)
            featureStride = max(audioFeatureStride, 1)
            let hammingWindow = Self.hammingWindow(length: windowLength)
            let melFilters = Self.makeSpeechLibMelFilterBank(
                sampleRate: samplingRate,
                fftLength: fftLength,
                melFilters: featureSize,
                minFrequency: 0,
                maxFrequency: 7690,
            )
            let tables = materialize([hammingWindow, melFilters])
            self.hammingWindow = tables[0]
            self.melFilters = tables[1]
        }

        public func callAsFunction(_ rawAudio: [MLXArray]) throws -> Output {
            try callAsFunction(rawAudio.map { (samples: $0, sampleRate: samplingRate) })
        }

        public func callAsFunction(_ rawAudio: [(samples: MLXArray, sampleRate: Int)]) throws
            -> Output
        {
            guard !rawAudio.isEmpty else {
                throw VLMError.processing("No audio samples were provided.")
            }

            var allEmbeds: [MLXArray] = []
            var embedSizes: [Int32] = []
            var frameCounts: [Int] = []
            allEmbeds.reserveCapacity(rawAudio.count)
            embedSizes.reserveCapacity(rawAudio.count)
            frameCounts.reserveCapacity(rawAudio.count)

            for clip in rawAudio {
                guard clip.sampleRate == samplingRate else {
                    throw VLMError.processing(
                        "Phi-4 Multimodal audio expects audio pre-resampled to \(samplingRate) Hz.",
                    )
                }
                let features = try extractFeatures(clip.samples)
                let audioFrames = features.dim(0) * featureStride
                allEmbeds.append(features)
                embedSizes.append(Int32(computeAudioEmbedSize(audioFrames: audioFrames)))
                frameCounts.append(audioFrames)
            }

            let maxLength = allEmbeds.map { $0.dim(0) }.max() ?? 0
            let paddedEmbeds = allEmbeds.map { embeds in
                let pad = max(0, maxLength - embeds.dim(0))
                guard pad > 0 else {
                    return embeds
                }
                return padded(embeds, widths: [.init((0, pad)), 0])
            }

            let attentionMask: MLXArray? =
                rawAudio.count > 1
                ? stacked(
                    frameCounts.map { frameCount in
                        MLXArray(
                            (0 ..< maxLength).map { index in
                                Int32(index < frameCount ? 1 : 0)
                            },
                            [maxLength],
                        )
                    },
                    axis: 0,
                )
                : nil

            return Output(
                inputAudioEmbeds: stacked(paddedEmbeds, axis: 0),
                audioEmbedSizes: MLXArray(embedSizes, [embedSizes.count]),
                audioAttentionMask: attentionMask,
            )
        }

        private func extractFeatures(_ input: MLXArray) throws -> MLXArray {
            let samples = Gemma4AudioFeatureExtractor.monoFloatSamples(input)
            guard samples.dim(0) > 0 else {
                throw VLMError.processing("Audio input is empty.")
            }

            let spectrogram = extractSpectrogram(samples)
            let power = spectrogram.square()
            let fbankPower = maximum(
                power.matmul(melFilters.asType(power.dtype)),
                MLXArray(1.0, dtype: power.dtype),
            )
            return log(fbankPower).asType(.float32)
        }

        private func extractSpectrogram(_ samples: MLXArray) -> MLXArray {
            let frameCount =
                samples.dim(0) >= windowLength
                ? ((samples.dim(0) - windowLength) / hopLength) + 1
                : 1
            var rows: [MLXArray] = []
            rows.reserveCapacity(frameCount)

            for frameIndex in 0 ..< frameCount {
                let start = frameIndex * hopLength
                let available = max(0, min(windowLength, samples.dim(0) - start))
                var frame =
                    available > 0
                    ? samples[start ..< (start + available)]
                    : MLXArray.zeros([0], dtype: .float32)
                if available < windowLength {
                    frame = padded(frame, widths: [.init((0, windowLength - available))])
                }
                rows.append(frame)
            }

            var frames = stacked(rows, axis: 0)
            if windowLength > 1 {
                let previous = concatenated(
                    [
                        frames[0..., 1 ..< 2],
                        frames[0..., ..<(windowLength - 1)],
                    ],
                    axis: 1,
                )
                frames =
                    (frames - MLXArray(0.97, dtype: frames.dtype) * previous)
                    * MLXArray(32768.0, dtype: frames.dtype)
            }

            let windowed = frames * hammingWindow.asType(frames.dtype)
            return MLXFFT.rfft(windowed, n: fftLength, axis: -1)
                .abs()
                .asType(.float32)
        }

        private func computeAudioEmbedSize(audioFrames: Int) -> Int {
            ceilDiv(ceilDiv(audioFrames, compressionRate), qformerCompressionRate)
        }

        private func ceilDiv(_ numerator: Int, _ denominator: Int) -> Int {
            max(0, (numerator + denominator - 1) / denominator)
        }

        private static func hammingWindow(length: Int) -> MLXArray {
            guard length > 1 else {
                return MLXArray([Float(1)], [1])
            }
            return MLXArray(
                (0 ..< length).map { index in
                    Float(
                        0.54 - 0.46
                            * cos(
                                2.0 * Double.pi * Double(index) / Double(length - 1),
                            ),
                    )
                },
                [length],
            )
        }

        private static func speechMel(_ frequency: Float) -> Float {
            1127.0 * log(1.0 + frequency / 700.0)
        }

        private static func makeSpeechLibMelFilterBank(
            sampleRate: Int,
            fftLength: Int,
            melFilters: Int,
            minFrequency: Float,
            maxFrequency: Float,
        )
            -> MLXArray
        {
            let bankWidth = fftLength / 2 + 1
            func binToMel(_ fftBin: Int) -> Float {
                speechMel(Float(fftBin * sampleRate) / Float(fftLength))
            }
            func frequencyToBin(_ frequency: Float) -> Int {
                Int(frequency * Float(fftLength) / Float(sampleRate) + 0.5)
            }

            let lowBin = min(max(frequencyToBin(minFrequency) + 1, 0), bankWidth)
            let highBin = min(max(frequencyToBin(maxFrequency), lowBin), bankWidth)
            let melMin = speechMel(minFrequency)
            let melMax = speechMel(maxFrequency)
            let melCenters = (0 ..< melFilters + 2).map { index in
                melMin + (melMax - melMin) * Float(index) / Float(melFilters + 1)
            }
            let melStep = (melMax - melMin) / Float(melFilters + 1)

            var values = Array(repeating: Float(0), count: bankWidth * melFilters)
            for filterIndex in 0 ..< melFilters {
                let left = melCenters[filterIndex]
                let center = melCenters[filterIndex + 1]
                let right = melCenters[filterIndex + 2]
                for fftBin in lowBin ..< highBin {
                    let melBin = binToMel(fftBin)
                    if left < melBin, melBin < right {
                        values[fftBin * melFilters + filterIndex] =
                            1.0 - abs(center - melBin) / melStep
                    }
                }
            }
            return MLXArray(values, [bankWidth, melFilters])
        }
    }

    public final class AudioProjection: Module {
        @ModuleInfo var speech: AudioProjectionBranch
        @ModuleInfo var vision: AudioProjectionBranch

        public init(audioDimensions: Int, hiddenSize: Int) {
            _speech.wrappedValue = AudioProjectionBranch(
                audioDimensions: audioDimensions,
                hiddenSize: hiddenSize,
            )
            _vision.wrappedValue = AudioProjectionBranch(
                audioDimensions: audioDimensions,
                hiddenSize: hiddenSize,
            )
            super.init()
        }

        public func callAsFunction(_ x: MLXArray, mode: String = "speech") -> MLXArray {
            switch mode {
            case "speech":
                speech(x)
            case "vision":
                vision(x)
            default:
                preconditionFailure("Unknown Phi-4 Multimodal audio projection mode: \(mode)")
            }
        }
    }

    public final class AudioProjectionBranch: Module, UnaryLayer {
        @ModuleInfo(key: "proj_0") var proj0: Linear
        @ModuleInfo(key: "proj_2") var proj2: Linear

        public init(audioDimensions: Int, hiddenSize: Int) {
            _proj0.wrappedValue = Linear(audioDimensions, hiddenSize, bias: true)
            _proj2.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
            super.init()
        }

        public func callAsFunction(_ x: MLXArray) -> MLXArray {
            proj2(gelu(proj0(x)))
        }
    }

    public final class ConformerEncoder: Module {
        let config: Phi4MultimodalConfiguration.AudioConfiguration
        let maxSequenceLength = 500

        @ModuleInfo(key: "encoder_embedding") var encoderEmbedding: MeanVarianceNormLayer
        @ModuleInfo var embed: NemoConvSubsampling
        @ModuleInfo(key: "relative_attention_bias_layer") var relativeAttentionBiasLayer:
            T5RelativeAttentionLogitBias
        @ModuleInfo var encoders: [ConformerEncoderLayer]

        public init(_ config: Phi4MultimodalConfiguration.AudioConfiguration) {
            self.config = config
            _encoderEmbedding.wrappedValue = MeanVarianceNormLayer(config.inputSize)
            _embed.wrappedValue = NemoConvSubsampling(config)
            _relativeAttentionBiasLayer.wrappedValue = T5RelativeAttentionLogitBias(
                heads: config.attentionHeads,
                maxDistance: config.t5BiasMaxDistance,
            )
            _encoders.wrappedValue = (0 ..< config.numBlocks).map { _ in
                ConformerEncoderLayer(config)
            }
            super.init()
        }

        public func callAsFunction(_ xsPad: MLXArray, masks: MLXArray? = nil)
            -> (MLXArray, MLXArray?)
        {
            var inputTensor = encoderEmbedding(xsPad)
            var outputMask: MLXArray?
            (inputTensor, outputMask) = embed(inputTensor, mask: masks)

            let originalBatchSize = inputTensor.dim(0)
            let sequenceLength = inputTensor.dim(1)
            var chunkPadding = 0
            var unfolded = false
            if sequenceLength > maxSequenceLength {
                unfolded = true
                chunkPadding =
                    sequenceLength % maxSequenceLength > 0
                    ? maxSequenceLength - (sequenceLength % maxSequenceLength)
                    : 0
                if chunkPadding > 0 {
                    let padding = MLXArray.zeros(
                        [inputTensor.dim(0), chunkPadding, inputTensor.dim(2)],
                        dtype: inputTensor.dtype,
                    )
                    inputTensor = concatenated([inputTensor, padding], axis: 1)
                }
                let paddedLength = inputTensor.dim(1)
                let chunks = paddedLength / maxSequenceLength
                inputTensor = inputTensor.reshaped(
                    originalBatchSize * chunks,
                    maxSequenceLength,
                    inputTensor.dim(2),
                )
            }

            let relativeAttentionBias = relativeAttentionBiasLayer(inputTensor)
            for layer in encoders {
                inputTensor = layer(
                    inputTensor,
                    mask: nil,
                    relativeAttentionBias: relativeAttentionBias,
                )
            }

            if unfolded {
                inputTensor = inputTensor.reshaped(originalBatchSize, -1, inputTensor.dim(-1))
                if chunkPadding > 0 {
                    inputTensor = inputTensor[0..., ..<(inputTensor.dim(1) - chunkPadding), 0...]
                }
            }

            return (inputTensor, outputMask)
        }
    }

    final class MeanVarianceNormLayer: Module, UnaryLayer {
        @ModuleInfo(key: "global_mean") var globalMean: MLXArray
        @ModuleInfo(key: "global_invstd") var globalInvStd: MLXArray

        init(_ inputSize: Int) {
            _globalMean.wrappedValue = MLXArray.zeros([inputSize])
            _globalInvStd.wrappedValue = MLXArray.ones([inputSize])
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            (x - globalMean) * globalInvStd
        }
    }

    final class NemoConvSubsampling: Module {
        let timeReduction: Int
        let samplingLayers: Int
        let kernelSize = 3
        let stride = 2
        let padding = 1

        @ModuleInfo(key: "conv_0") var conv0: Conv2d
        @ModuleInfo(key: "dw_pw_layers") var dwPwLayers: [DWPWConvPair]
        @ModuleInfo var out: Linear

        init(_ config: Phi4MultimodalConfiguration.AudioConfiguration) {
            timeReduction = config.timeReduction
            samplingLayers = max(1, Int(log2(Double(max(config.timeReduction, 1)))))
            let localKernelSize = 3
            let localStride = 2
            let localPadding = 1
            _conv0.wrappedValue = Conv2d(
                inputChannels: 1,
                outputChannels: config.convChannels,
                kernelSize: .init(localKernelSize),
                stride: .init(localStride),
                padding: .init(localPadding),
            )
            let additionalLayers = max(samplingLayers - 1, 0)
            _dwPwLayers.wrappedValue = (0 ..< additionalLayers).map { _ in
                DWPWConvPair(
                    channels: config.convChannels,
                    kernelSize: localKernelSize,
                    stride: localStride,
                    padding: localPadding,
                )
            }

            var frequencyOut = config.inputSize
            for _ in 0 ..< samplingLayers {
                frequencyOut =
                    ((frequencyOut + 2 * localPadding - localKernelSize) / localStride) + 1
            }
            _out.wrappedValue = Linear(
                config.convChannels * max(frequencyOut, 1),
                config.attentionDimensions,
                bias: true,
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> (MLXArray, MLXArray?) {
            var hidden = x.expandedDimensions(axis: -1)
            hidden = relu(conv0(hidden))
            for pair in dwPwLayers {
                hidden = pair(hidden)
            }

            let batchSize = hidden.dim(0)
            let time = hidden.dim(1)
            let frequency = hidden.dim(2)
            let channels = hidden.dim(3)
            hidden =
                hidden
                .transposed(0, 1, 3, 2)
                .reshaped(batchSize, time, channels * frequency)
            hidden = out(hidden)

            guard let mask else {
                return (hidden, nil)
            }

            let featureLengths = sum(mask, axis: 1)
            let paddingLength = ceil(featureLengths.asType(.float32) / Float(timeReduction))
                .asType(.int32)
            let outputMask =
                arange(0, time, dtype: .int32).expandedDimensions(axis: 0)
                .< paddingLength.expandedDimensions(axis: 1)
            return (hidden, outputMask.expandedDimensions(axis: 1))
        }
    }

    final class DWPWConvPair: Module, UnaryLayer {
        @ModuleInfo var dw: Conv2d
        @ModuleInfo var pw: Conv2d

        init(channels: Int, kernelSize: Int, stride: Int, padding: Int) {
            _dw.wrappedValue = Conv2d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: .init(kernelSize),
                stride: .init(stride),
                padding: .init(padding),
                groups: channels,
            )
            _pw.wrappedValue = Conv2d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: .init(1),
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            relu(pw(dw(x)))
        }
    }

    final class ConformerEncoderLayer: Module {
        @ModuleInfo(key: "feed_forward_in") var feedForwardIn: FeedForward
        @ModuleInfo(key: "self_attn") var selfAttention: MultiHeadedAttention
        @ModuleInfo var conv: ConvModule
        @ModuleInfo(key: "feed_forward_out") var feedForwardOut: FeedForward
        @ModuleInfo(key: "layer_norm_att") var layerNormAttention: LayerNorm
        @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

        init(_ config: Phi4MultimodalConfiguration.AudioConfiguration) {
            _feedForwardIn.wrappedValue = FeedForward(config)
            _selfAttention.wrappedValue = MultiHeadedAttention(config)
            _conv.wrappedValue = ConvModule(config)
            _feedForwardOut.wrappedValue = FeedForward(config)
            _layerNormAttention.wrappedValue = LayerNorm(dimensions: config.attentionDimensions)
            _layerNorm.wrappedValue = LayerNorm(dimensions: config.attentionDimensions)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray? = nil,
            relativeAttentionBias: MLXArray? = nil,
        )
            -> MLXArray
        {
            var hidden = x + 0.5 * feedForwardIn(x)
            let normalized = layerNormAttention(hidden)
            hidden =
                hidden
                + selfAttention(
                    query: normalized,
                    key: normalized,
                    value: normalized,
                    mask: mask,
                    relativeAttentionBias: relativeAttentionBias,
                )
            hidden = hidden + conv(hidden)
            hidden = hidden + 0.5 * feedForwardOut(hidden)
            return layerNorm(hidden)
        }
    }

    final class FeedForward: Module, UnaryLayer {
        @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
        @ModuleInfo(key: "net_0") var net0: GLULinear
        @ModuleInfo(key: "net_2") var net2: Linear

        init(_ config: Phi4MultimodalConfiguration.AudioConfiguration) {
            _layerNorm.wrappedValue = LayerNorm(dimensions: config.attentionDimensions)
            _net0.wrappedValue = GLULinear(
                inputDimensions: config.attentionDimensions,
                outputDimensions: config.linearUnits,
                activationName: config.activation,
                bias: config.biasInGLU,
            )
            _net2.wrappedValue = Linear(
                config.linearUnits,
                config.attentionDimensions,
                bias: true,
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            net2(net0(layerNorm(x)))
        }
    }

    final class GLULinear: Module, UnaryLayer {
        let outputDimensions: Int
        let activationName: String

        @ModuleInfo var linear: Linear

        init(inputDimensions: Int, outputDimensions: Int, activationName: String, bias: Bool) {
            self.outputDimensions = outputDimensions
            self.activationName = activationName
            _linear.wrappedValue = Linear(inputDimensions, outputDimensions * 2, bias: bias)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let projected = linear(x)
            let first = projected[0..., 0..., ..<outputDimensions]
            let gate = projected[0..., 0..., outputDimensions...]
            return first * Phi4MultimodalAudio.activate(gate, activationName)
        }
    }

    final class ConvModule: Module, UnaryLayer {
        let extPointwiseOutChannels: Int
        let causal: Bool
        let kernelSize: Int
        let activationName: String

        @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
        @ModuleInfo var glu: GLUPointwiseConv?
        @ModuleInfo(key: "dw_sep_conv_1d") var depthwiseSeparableConv1d: DepthwiseSeparableConv1d
        @ModuleInfo(key: "ext_pw_conv_1d") var extPointwiseConv1d: Linear?

        init(_ config: Phi4MultimodalConfiguration.AudioConfiguration) {
            extPointwiseOutChannels = config.extPointwiseOutChannels
            causal = config.causal
            kernelSize = config.kernelSize
            activationName = config.convActivation
            _layerNorm.wrappedValue = LayerNorm(dimensions: config.attentionDimensions)
            if config.extPointwiseOutChannels != 0 {
                _glu.wrappedValue = GLUPointwiseConv(config)
                _extPointwiseConv1d.wrappedValue = Linear(
                    config.attentionDimensions,
                    config.extPointwiseOutChannels,
                    bias: true,
                )
            }
            _depthwiseSeparableConv1d.wrappedValue = DepthwiseSeparableConv1d(config)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var hidden = layerNorm(x)
            if let glu {
                hidden = glu(hidden)
            }
            hidden = depthwiseSeparableConv1d(hidden)
            if causal, kernelSize > 1, hidden.dim(1) > kernelSize - 1 {
                hidden = hidden[0..., ..<(hidden.dim(1) - (kernelSize - 1)), 0...]
            }
            hidden = Phi4MultimodalAudio.activate(hidden, activationName)
            if let extPointwiseConv1d {
                hidden = extPointwiseConv1d(hidden)
            }
            return hidden
        }
    }

    final class GLUPointwiseConv: Module, UnaryLayer {
        let outputDimensions: Int
        let activationName: String
        let biasInGLU: Bool

        @ModuleInfo(key: "ext_pw_conv_1d") var extPointwiseConv1d: Linear
        @ModuleInfo var b1: MLXArray
        @ModuleInfo var b2: MLXArray

        init(_ config: Phi4MultimodalConfiguration.AudioConfiguration) {
            outputDimensions = config.extPointwiseOutChannels
            activationName = config.convGLUType
            biasInGLU = config.biasInGLU
            _extPointwiseConv1d.wrappedValue = Linear(
                config.attentionDimensions,
                config.extPointwiseOutChannels * 2,
                bias: true,
            )
            _b1.wrappedValue = MLXArray.zeros([config.extPointwiseOutChannels])
            _b2.wrappedValue = MLXArray.zeros([config.extPointwiseOutChannels])
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let projected = extPointwiseConv1d(x)
            var first = projected[0..., 0..., ..<outputDimensions]
            var gate = projected[0..., 0..., outputDimensions...]
            if biasInGLU {
                first = first + b1
                gate = gate + b2
            }
            return first * Phi4MultimodalAudio.activate(gate, activationName)
        }
    }

    final class DepthwiseSeparableConv1d: Module, UnaryLayer {
        let outChannels: Int

        @ModuleInfo(key: "dw_conv") var depthwiseConv: Conv1d
        @ModuleInfo(key: "pw_conv") var pointwiseConv: Conv1d?

        init(_ config: Phi4MultimodalConfiguration.AudioConfiguration) {
            let inputDimensions = config.attentionDimensions
            let multiplier = max(config.depthwiseMultiplier, 1)
            let padding = config.causal ? config.kernelSize - 1 : (config.kernelSize - 1) / 2
            outChannels = config.depthwiseSeparableOutChannels
            _depthwiseConv.wrappedValue = Conv1d(
                inputChannels: inputDimensions,
                outputChannels: inputDimensions * multiplier,
                kernelSize: config.kernelSize,
                padding: padding,
                groups: inputDimensions,
                bias: true,
            )
            if config.depthwiseSeparableOutChannels != 0 {
                _pointwiseConv.wrappedValue = Conv1d(
                    inputChannels: inputDimensions * multiplier,
                    outputChannels: config.depthwiseSeparableOutChannels,
                    kernelSize: 1,
                    bias: true,
                )
            }
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var hidden = depthwiseConv(x)
            if let pointwiseConv {
                hidden = pointwiseConv(hidden)
            }
            return hidden
        }
    }

    final class MultiHeadedAttention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "linear_q") var linearQ: Linear
        @ModuleInfo(key: "linear_k") var linearK: Linear
        @ModuleInfo(key: "linear_v") var linearV: Linear
        @ModuleInfo(key: "linear_out") var linearOut: Linear

        init(_ config: Phi4MultimodalConfiguration.AudioConfiguration) {
            heads = config.attentionHeads
            headDim = config.attentionDimensions / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _linearQ.wrappedValue = Linear(
                config.attentionDimensions,
                config.attentionDimensions,
                bias: true,
            )
            _linearK.wrappedValue = Linear(
                config.attentionDimensions,
                config.attentionDimensions,
                bias: true,
            )
            _linearV.wrappedValue = Linear(
                config.attentionDimensions,
                config.attentionDimensions,
                bias: true,
            )
            _linearOut.wrappedValue = Linear(
                config.attentionDimensions,
                config.attentionDimensions,
                bias: true,
            )
            super.init()
        }

        func callAsFunction(
            query: MLXArray,
            key: MLXArray,
            value: MLXArray,
            mask: MLXArray? = nil,
            relativeAttentionBias: MLXArray? = nil,
        )
            -> MLXArray
        {
            let batchSize = query.dim(0)
            let queryLength = query.dim(1)
            let queryStates = linearQ(query)
                .reshaped(batchSize, -1, heads, headDim)
                .transposed(0, 2, 1, 3)
            let keyStates = linearK(key)
                .reshaped(batchSize, -1, heads, headDim)
                .transposed(0, 2, 1, 3)
            let valueStates = linearV(value)
                .reshaped(batchSize, -1, heads, headDim)
                .transposed(0, 2, 1, 3)

            var scores = matmul(queryStates * scale, keyStates.transposed(0, 1, 3, 2))
            if let relativeAttentionBias {
                scores = scores + relativeAttentionBias.asType(scores.dtype)
            }
            if let mask {
                scores = MLX.where(mask, scores, MLXArray(-1e9, dtype: scores.dtype))
            }
            var attention = softmax(scores, axis: -1)
            if let mask {
                attention = MLX.where(mask, attention, MLXArray(0.0, dtype: attention.dtype))
            }
            let output = matmul(attention, valueStates)
                .transposed(0, 2, 1, 3)
                .reshaped(batchSize, queryLength, heads * headDim)
            return linearOut(output)
        }
    }

    final class T5RelativeAttentionLogitBias: Module {
        let heads: Int
        let maxDistance: Int
        let buckets: Int

        @ModuleInfo(key: "bias_values") var biasValues: Embedding

        init(heads: Int, maxDistance: Int) {
            self.heads = heads
            self.maxDistance = maxDistance
            buckets = maxDistance * 2
            _biasValues.wrappedValue = Embedding(embeddingCount: buckets, dimensions: heads)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let length = x.dim(1)
            let contextPosition = arange(0, length, dtype: .int32).expandedDimensions(axis: 1)
            let memoryPosition = arange(0, length, dtype: .int32).expandedDimensions(axis: 0)
            let relativePosition = clip(
                memoryPosition - contextPosition,
                min: -maxDistance,
                max: maxDistance - 1,
            )
            let biasIndex = relativePosition + Int32(buckets / 2)
            return biasValues(biasIndex).transposed(2, 0, 1).expandedDimensions(axis: 0)
        }
    }

    private static func activate(_ x: MLXArray, _ name: String) -> MLXArray {
        switch name.lowercased() {
        case "relu":
            relu(x)
        case "gelu":
            gelu(x)
        case "swish", "silu":
            silu(x)
        case "sigmoid":
            sigmoid(x)
        default:
            x
        }
    }
}

// MARK: - Language

private enum Phi4MultimodalLanguage {
    final class Attention: Module {
        let config: Phi4MultimodalConfiguration
        let headDim: Int
        let scale: Float
        let rope: RoPE

        @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        init(_ config: Phi4MultimodalConfiguration) {
            self.config = config
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            let qkvSize =
                config.attentionHeads * headDim + 2 * (config.keyValueHeads * headDim)
            _qkvProj.wrappedValue = Linear(config.hiddenSize, qkvSize, bias: false)
            _oProj.wrappedValue = Linear(
                config.attentionHeads * headDim,
                config.hiddenSize,
                bias: false,
            )
            rope = RoPE(
                dimensions: max(1, Int(Float(headDim) * config.partialRotaryFactor)),
                traditional: false,
                base: config.ropeTheta,
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
            let batchSize = x.dim(0)
            let sequenceLength = x.dim(1)
            let queryEnd = config.attentionHeads * headDim
            let keyEnd = queryEnd + config.keyValueHeads * headDim
            let qkv = qkvProj(x)
            let splits = split(qkv, indices: [queryEnd, keyEnd], axis: -1)

            var queries = splits[0]
                .reshaped(batchSize, sequenceLength, config.attentionHeads, headDim)
                .transposed(0, 2, 1, 3)
            var keys = splits[1]
                .reshaped(batchSize, sequenceLength, config.keyValueHeads, headDim)
                .transposed(0, 2, 1, 3)
            let values = splits[2]
                .reshaped(batchSize, sequenceLength, config.keyValueHeads, headDim)
                .transposed(0, 2, 1, 3)

            let ropeOffset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: ropeOffset)
            keys = applyRotaryPosition(rope, to: keys, offset: ropeOffset)

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

    final class MLP: Module, UnaryLayer {
        let intermediateSize: Int

        @ModuleInfo(key: "gate_up_proj") var gateUpProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        init(_ config: Phi4MultimodalConfiguration) {
            intermediateSize = config.intermediateSize
            _gateUpProj.wrappedValue = Linear(
                config.hiddenSize,
                2 * config.intermediateSize,
                bias: false,
            )
            _downProj.wrappedValue = Linear(
                config.intermediateSize,
                config.hiddenSize,
                bias: false,
            )
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let projected = gateUpProj(x)
            let splits = split(projected, indices: [intermediateSize], axis: -1)
            return downProj(silu(splits[0]) * splits[1])
        }
    }

    final class TransformerBlock: Module {
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: Phi4MultimodalConfiguration) {
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
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            cache: KVCache?,
        )
            -> MLXArray
        {
            var hidden = x + selfAttention(inputLayerNorm(x), mask: mask, cache: cache)
            hidden = hidden + mlp(postAttentionLayerNorm(hidden))
            return hidden
        }
    }

    final class TextModel: Module {
        let config: Phi4MultimodalConfiguration

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo var layers: [TransformerBlock]
        @ModuleInfo var norm: RMSNorm

        init(_ config: Phi4MultimodalConfiguration) {
            self.config = config
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
                TransformerBlock(config)
            }
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
            super.init()
        }

        func callAsFunction(
            _ inputIDs: MLXArray,
            inputEmbeddings: MLXArray? = nil,
            mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
            cache: [KVCache]? = nil,
        )
            -> MLXArray
        {
            var hidden = inputEmbeddings ?? embedTokens(inputIDs)
            let attentionMask = mask ?? createAttentionMask(h: hidden, cache: cache?.first)
            for (index, layer) in layers.enumerated() {
                hidden = layer(hidden, mask: attentionMask, cache: cache?[index])
            }
            return norm(hidden)
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo var model: TextModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        init(_ config: Phi4MultimodalConfiguration) {
            _model.wrappedValue = TextModel(config)
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
            _ inputIDs: MLXArray,
            cache: [KVCache]? = nil,
            inputEmbeddings: MLXArray? = nil,
        )
            -> MLXArray
        {
            let hidden = model(inputIDs, inputEmbeddings: inputEmbeddings, cache: cache)
            if let lmHead {
                return lmHead(hidden)
            }
            return model.embedTokens.asLinear(hidden)
        }
    }
}

// MARK: - Full Model

public enum Phi4MultimodalError: LocalizedError {
    case audioMetadataMissing
    case audioFeatureCountMismatch(expected: Int, actual: Int)
    case imageMetadataMissing
    case imageFeatureCountMismatch(expected: Int, actual: Int)

    public var errorDescription: String? {
        switch self {
        case .audioMetadataMissing:
            "Phi-4 Multimodal audio metadata is missing; use Phi4MultimodalProcessor for audio inputs."
        case .audioFeatureCountMismatch(let expected, let actual):
            "Phi-4 Multimodal audio features (\(expected)) do not match audio token count (\(actual))."
        case .imageMetadataMissing:
            "Phi-4 Multimodal image metadata is missing; use Phi4MultimodalProcessor for image inputs."
        case .imageFeatureCountMismatch(let expected, let actual):
            "Phi-4 Multimodal image features (\(expected)) do not match image token count (\(actual))."
        }
    }
}

private enum Phi4MultimodalActiveLoRA: Equatable {
    case base
    case vision
    case speech
    case visionAndSpeech
}

public final class Phi4Multimodal: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "language_model") private var languageModel:
        Phi4MultimodalLanguage.LanguageModel
    @ModuleInfo(key: "vision_tower") private var visionTower: Phi4MultimodalVision.VisionTower
    @ModuleInfo(key: "mm_projector") private var mmProjector: [Module]
    @ModuleInfo(key: "audio_encoder") private var audioEncoder: Phi4MultimodalAudio.ConformerEncoder
    @ModuleInfo(key: "audio_projection") private var audioProjection:
        Phi4MultimodalAudio.AudioProjection

    public let config: Phi4MultimodalConfiguration
    public let vocabularySize: Int
    public let kvHeads: [Int]
    private var baseLoRAWeights: [String: MLXArray] = [:]
    private var visionLoRAA: [String: MLXArray] = [:]
    private var visionLoRAB: [String: MLXArray] = [:]
    private var speechLoRAA: [String: MLXArray] = [:]
    private var speechLoRAB: [String: MLXArray] = [:]
    private var activeLoRA: Phi4MultimodalActiveLoRA = .base

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: Phi4MultimodalConfiguration) {
        self.config = config
        vocabularySize = config.vocabularySize
        kvHeads = (0 ..< config.hiddenLayers).map { _ in config.keyValueHeads }
        _languageModel.wrappedValue = Phi4MultimodalLanguage.LanguageModel(config)
        _visionTower.wrappedValue = Phi4MultimodalVision.VisionTower(config.visionConfiguration)
        _mmProjector.wrappedValue = [
            Linear(config.mmHiddenSize, config.hiddenSize, bias: true),
            GELU(),
            Linear(config.hiddenSize, config.hiddenSize, bias: true),
        ]
        _audioEncoder.wrappedValue = Phi4MultimodalAudio.ConformerEncoder(
            config.audioConfiguration,
        )
        _audioProjection.wrappedValue = Phi4MultimodalAudio.AudioProjection(
            audioDimensions: config.audioConfiguration.attentionDimensions,
            hiddenSize: config.hiddenSize,
        )
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let hasImage = input.image != nil
        let hasAudio = input.audio != nil
        try setModality(hasImage: hasImage, hasAudio: hasAudio)
        guard hasImage || hasAudio else {
            return .tokens(input.text)
        }

        let inputIDs = input.text.tokens
        let inputsEmbeds = try getInputEmbeddings(
            inputIDs: inputIDs,
            image: input.image,
            audio: input.audio,
        )
        let logits = languageModel(inputIDs, cache: cache, inputEmbeddings: inputsEmbeds)
        prefill.progress?(inputsEmbeds.dim(1), inputsEmbeds.dim(1))
        return .logits(.init(logits: logits))
    }

    private func getInputEmbeddings(
        inputIDs: MLXArray,
        image: LMInput.ProcessedImage?,
        audio: LMInput.ProcessedAudio?,
    )
        throws -> MLXArray
    {
        var safeInputIDs = inputIDs
        if image != nil {
            safeInputIDs = MLX.where(
                safeInputIDs .== config.imageTokenIndex,
                MLXArray(0, dtype: inputIDs.dtype),
                safeInputIDs,
            )
        }
        if audio != nil {
            safeInputIDs = MLX.where(
                safeInputIDs .== config.audioTokenIndex,
                MLXArray(0, dtype: inputIDs.dtype),
                safeInputIDs,
            )
        }
        let textEmbeds = languageModel.model.embedTokens(safeInputIDs)

        let imageFeatures: [MLXArray]? =
            if let image {
                try projectImageFeatures(image, dtype: textEmbeds.dtype)
            } else {
                nil
            }

        let audioOutput: (features: MLXArray, embedSizes: [Int])? =
            if let audio {
                try encodeAudioFeatures(audio, dtype: textEmbeds.dtype)
            } else {
                nil
            }

        return try mergeMultimodalFeatures(
            inputIDs: inputIDs,
            textEmbeds: textEmbeds,
            imageFeatures: imageFeatures,
            audioFeatures: audioOutput?.features,
            audioEmbedSizes: audioOutput?.embedSizes ?? [],
        )
    }

    private func projectImageFeatures(_ image: LMInput.ProcessedImage, dtype: DType) throws
        -> [MLXArray]
    {
        let (spatialShapes, pixelAttentionMask) = try unpackImageMetadata(image)
        let features = visionTower(
            pixelValues: image.pixels,
            pixelAttentionMask: pixelAttentionMask,
            spatialShapes: spatialShapes,
        )
        return features.map { feature in
            var hidden = feature
            for layer in mmProjector {
                hidden = (layer as! any UnaryLayer)(hidden)
            }
            return hidden.asType(dtype)
        }
    }

    private func unpackImageMetadata(_ image: LMInput.ProcessedImage) throws
        -> (spatialShapes: MLXArray, pixelAttentionMask: MLXArray)
    {
        guard let metadata = image.positionIds,
            metadata.ndim == 2,
            metadata.dim(0) == image.pixels.dim(0),
            metadata.dim(1) >= image.pixels.dim(1) + 2
        else {
            throw Phi4MultimodalError.imageMetadataMissing
        }

        let raw = metadata.asArray(Int32.self)
        let rowWidth = metadata.dim(1)
        let batchSize = metadata.dim(0)
        let maxPatches = image.pixels.dim(1)
        var spatialValues: [Int32] = []
        var maskValues: [Int32] = []
        spatialValues.reserveCapacity(batchSize * 2)
        maskValues.reserveCapacity(batchSize * maxPatches)
        for row in 0 ..< batchSize {
            let start = row * rowWidth
            spatialValues.append(raw[start])
            spatialValues.append(raw[start + 1])
            maskValues.append(contentsOf: raw[(start + 2) ..< (start + 2 + maxPatches)])
        }
        return (
            MLXArray(spatialValues, [batchSize, 2]),
            MLXArray(maskValues, [batchSize, maxPatches]),
        )
    }

    private func encodeAudioFeatures(_ audio: LMInput.ProcessedAudio, dtype: DType) throws
        -> (features: MLXArray, embedSizes: [Int])
    {
        let (audioAttentionMask, audioEmbedSizes) = try unpackAudioMetadata(audio)
        let (encodedAudio, _) = audioEncoder(audio.features, masks: audioAttentionMask)
        return (
            audioProjection(encodedAudio, mode: "speech").asType(dtype),
            audioEmbedSizes,
        )
    }

    private func unpackAudioMetadata(_ audio: LMInput.ProcessedAudio) throws
        -> (attentionMask: MLXArray?, embedSizes: [Int])
    {
        guard let metadata = audio.mask, metadata.ndim == 2,
            metadata.dim(0) == audio.features.dim(0)
        else {
            throw Phi4MultimodalError.audioMetadataMissing
        }

        let rawMetadata = metadata.asArray(Int32.self)
        let rowWidth = metadata.dim(1)
        var embedSizes: [Int] = []
        embedSizes.reserveCapacity(metadata.dim(0))
        for row in 0 ..< metadata.dim(0) {
            embedSizes.append(Int(rawMetadata[row * rowWidth]))
        }

        if rowWidth == 1 {
            return (nil, embedSizes)
        }

        let maskWidth = rowWidth - 1
        var values: [Int32] = []
        values.reserveCapacity(metadata.dim(0) * maskWidth)
        for row in 0 ..< metadata.dim(0) {
            let start = row * rowWidth + 1
            values.append(contentsOf: rawMetadata[start ..< (start + maskWidth)])
        }
        return (MLXArray(values, [metadata.dim(0), maskWidth]), embedSizes)
    }

    private func mergeMultimodalFeatures(
        inputIDs: MLXArray,
        textEmbeds: MLXArray,
        imageFeatures: [MLXArray]?,
        audioFeatures: MLXArray?,
        audioEmbedSizes: [Int],
    )
        throws -> MLXArray
    {
        let batchSize = inputIDs.dim(0)
        let sequenceLength = inputIDs.dim(1)
        let tokenIDs = inputIDs.asArray(Int.self)
        var mergedRows: [MLXArray] = []
        mergedRows.reserveCapacity(batchSize)
        var imageOffset = 0
        var audioOffset = 0

        for batch in 0 ..< batchSize {
            var parts: [MLXArray] = []
            var previous = 0
            var index = 0
            while index < sequenceLength {
                let tokenID = tokenIDs[batch * sequenceLength + index]
                if tokenID == config.imageTokenIndex, let imageFeatures {
                    if index > previous {
                        parts.append(textEmbeds[batch, previous ..< index, 0...])
                    }
                    guard imageOffset < imageFeatures.count else {
                        throw Phi4MultimodalError.imageFeatureCountMismatch(
                            expected: imageOffset + 1,
                            actual: imageFeatures.count,
                        )
                    }
                    parts.append(imageFeatures[imageOffset].asType(textEmbeds.dtype))
                    imageOffset += 1
                    previous = index + 1
                    index += 1
                } else if tokenID == config.audioTokenIndex, let audioFeatures {
                    if index > previous {
                        parts.append(textEmbeds[batch, previous ..< index, 0...])
                    }

                    var end = index
                    while end < sequenceLength,
                        tokenIDs[batch * sequenceLength + end] == config.audioTokenIndex
                    {
                        end += 1
                    }
                    let tokenCount = end - index
                    guard audioOffset < audioEmbedSizes.count else {
                        throw Phi4MultimodalError.audioFeatureCountMismatch(
                            expected: audioOffset + 1,
                            actual: audioEmbedSizes.count,
                        )
                    }
                    let featureCount = audioEmbedSizes[audioOffset]
                    if featureCount != tokenCount {
                        throw Phi4MultimodalError.audioFeatureCountMismatch(
                            expected: featureCount,
                            actual: tokenCount,
                        )
                    }
                    parts.append(audioFeatures[audioOffset, ..<featureCount, 0...])
                    audioOffset += 1
                    previous = end
                    index = end
                } else {
                    index += 1
                }
            }

            if previous < sequenceLength {
                parts.append(textEmbeds[batch, previous ..< sequenceLength, 0...])
            }
            mergedRows.append(parts.isEmpty ? textEmbeds[batch] : concatenated(parts, axis: 0))
        }

        if audioOffset != audioEmbedSizes.count {
            throw Phi4MultimodalError.audioFeatureCountMismatch(
                expected: audioEmbedSizes.count,
                actual: audioOffset,
            )
        }
        if let imageFeatures, imageOffset != imageFeatures.count {
            throw Phi4MultimodalError.imageFeatureCountMismatch(
                expected: imageFeatures.count,
                actual: imageOffset,
            )
        }

        guard batchSize > 1 else {
            return mergedRows[0].expandedDimensions(axis: 0)
        }

        let maxLength = mergedRows.map { $0.dim(0) }.max() ?? 0
        let paddedRows = mergedRows.map { row in
            let pad = max(0, maxLength - row.dim(0))
            guard pad > 0 else {
                return row
            }
            return padded(row, widths: [.init((0, pad)), 0])
        }
        return stacked(paddedRows, axis: 0)
    }

    private func setModality(hasImage: Bool, hasAudio: Bool) throws {
        let target: Phi4MultimodalActiveLoRA =
            if hasImage, hasAudio {
                .visionAndSpeech
            } else if hasImage {
                .vision
            } else if hasAudio {
                .speech
            } else {
                .base
            }
        guard target != activeLoRA else {
            return
        }
        guard !baseLoRAWeights.isEmpty else {
            activeLoRA = target
            return
        }
        let weights = mergedLoRAWeights(for: target)
        try update(parameters: ModuleParameters.unflattened(weights), verify: .noUnusedKeys)
        activeLoRA = target
    }

    private func mergedLoRAWeights(for target: Phi4MultimodalActiveLoRA) -> [String: MLXArray] {
        var weights: [String: MLXArray] = [:]
        weights.reserveCapacity(baseLoRAWeights.count)
        for (key, base) in baseLoRAWeights {
            var merged = base
            if target == .vision || target == .visionAndSpeech,
                let loraA = visionLoRAA[key],
                let loraB = visionLoRAB[key]
            {
                merged = merged + MLXArray(config.visionLoRA?.scale ?? 1) * matmul(loraB, loraA)
            }
            if target == .speech || target == .visionAndSpeech,
                let loraA = speechLoRAA[key],
                let loraB = speechLoRAB[key]
            {
                merged = merged + MLXArray(config.speechLoRA?.scale ?? 1) * matmul(loraB, loraA)
            }
            weights[key] = merged
        }
        return weights
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        var audioWeights: [String: MLXArray] = [:]
        var baseWeights: [String: MLXArray] = [:]
        var visionLoRAA: [String: MLXArray] = [:]
        var visionLoRAB: [String: MLXArray] = [:]
        var speechLoRAA: [String: MLXArray] = [:]
        var speechLoRAB: [String: MLXArray] = [:]

        for (key, value) in weights {
            if key.contains("position_ids")
                || key.contains("glb_GN")
                || key.contains("sub_GN")
                || key.contains("img_processor.head.")
            {
                continue
            }

            if key.contains("embed_tokens_extend.audio_embed.encoder.")
                || key.contains("embed_tokens_extend.audio_embed.audio_projection.")
            {
                audioWeights[key] = value
                continue
            }

            if key.contains("embed_tokens_extend.image_embed.img_processor.") {
                let newKey = key.replacingOccurrences(
                    of: "model.embed_tokens_extend.image_embed.img_processor.",
                    with: "vision_tower.vision_tower.",
                )
                sanitized[newKey] = Self.sanitizeVisionWeight(key: newKey, value: value)
                continue
            }

            if key.contains("embed_tokens_extend.image_embed.img_projection.") {
                let newKey = key.replacingOccurrences(
                    of: "model.embed_tokens_extend.image_embed.img_projection.",
                    with: "mm_projector.",
                )
                sanitized[newKey] = value
                continue
            }

            if key.contains(".lora_A.vision.") {
                visionLoRAA[
                    Self.remapLLMKey(key.replacingOccurrences(of: ".lora_A.vision.", with: "."))] =
                    value
                continue
            }
            if key.contains(".lora_B.vision.") {
                visionLoRAB[
                    Self.remapLLMKey(key.replacingOccurrences(of: ".lora_B.vision.", with: "."))] =
                    value
                continue
            }
            if key.contains(".lora_A.speech.") {
                speechLoRAA[
                    Self.remapLLMKey(key.replacingOccurrences(of: ".lora_A.speech.", with: "."))] =
                    value
                continue
            }
            if key.contains(".lora_B.speech.") {
                speechLoRAB[
                    Self.remapLLMKey(key.replacingOccurrences(of: ".lora_B.speech.", with: "."))] =
                    value
                continue
            }

            if key.contains(".base_layer.") {
                let baseKey = Self.remapLLMKey(
                    key.replacingOccurrences(of: ".base_layer.", with: "."),
                )
                baseWeights[baseKey] = value
                continue
            }

            let newKey = Self.remapLLMKey(key)
            sanitized[newKey] = Self.sanitizeVisionWeight(key: newKey, value: value)
        }

        baseLoRAWeights = baseWeights
        self.visionLoRAA = visionLoRAA
        self.visionLoRAB = visionLoRAB
        self.speechLoRAA = speechLoRAA
        self.speechLoRAB = speechLoRAB

        let defaultLoRA: Phi4MultimodalActiveLoRA =
            !baseWeights.isEmpty && !visionLoRAA.isEmpty && !visionLoRAB.isEmpty
            ? .vision
            : (!baseWeights.isEmpty && !speechLoRAA.isEmpty && !speechLoRAB.isEmpty
                ? .speech : .base)
        for (key, base) in baseWeights {
            if defaultLoRA == .vision,
                let visionA = visionLoRAA[key],
                let visionB = visionLoRAB[key]
            {
                sanitized[key] =
                    base + MLXArray(config.visionLoRA?.scale ?? 1) * matmul(visionB, visionA)
            } else if defaultLoRA == .speech,
                let speechA = speechLoRAA[key],
                let speechB = speechLoRAB[key]
            {
                sanitized[key] =
                    base + MLXArray(config.speechLoRA?.scale ?? 1) * matmul(speechB, speechA)
            } else {
                sanitized[key] = base
            }
        }
        activeLoRA = defaultLoRA

        if !audioWeights.isEmpty {
            sanitized.merge(Phi4MultimodalAudio.sanitizeWeights(audioWeights)) { _, new in new }
        }

        return sanitized
    }

    private static func remapLLMKey(_ key: String) -> String {
        if key.hasPrefix("model.") {
            return "language_model." + key
        }
        if key.hasPrefix("lm_head.") {
            return "language_model." + key
        }
        return key
    }

    private static func sanitizeVisionWeight(key: String, value: MLXArray) -> MLXArray {
        if key.contains("vision_tower.vision_tower.embeddings.patch_embedding.weight"),
            value.ndim == 4
        {
            return
                value
                .transposed(0, 2, 3, 1)
                .reshaped(value.dim(0), -1)
        }
        return value
    }
}
