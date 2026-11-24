import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct Gemma3nConfiguration: Decodable {
    public struct TextConfiguration: Decodable, Sendable {
        public let hiddenSize: Int
        public let hiddenLayers: Int
        public let kvHeads: Int

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case kvHeads = "num_key_value_heads"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 30
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let imageSize: Int
        public let vocabSize: Int
        public let vocabOffset: Int
        public let rmsNormEps: Float

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case imageSize = "image_size"
            case vocabSize = "vocab_size"
            case vocabOffset = "vocab_offset"
            case rmsNormEps = "rms_norm_eps"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType =
                try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma3n_vision"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? 768
            vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 128
            vocabOffset = try c.decodeIfPresent(Int.self, forKey: .vocabOffset) ?? 262_144
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        }
    }

    public struct AudioConfiguration: Decodable, Sendable {
        public let inputFeatureSize: Int
        public let hiddenSize: Int
        public let attentionChunkSize: Int
        public let attentionContextLeft: Int
        public let attentionContextRight: Int
        public let attentionInvalidLogitsValue: Float
        public let attentionLogitCap: Float
        public let attentionHeads: Int
        public let hiddenLayers: Int
        public let convKernelSize: Int
        public let reductionFactor: Int
        public let residualWeight: Float
        public let convChannelSize: [Int]
        public let convGroupNormEps: Float
        public let convKernelSizes: [[Int]]
        public let convStrideSizes: [[Int]]
        public let convEps: Float
        public let rmsNormEps: Float
        public let gradientClipping: Float
        public let vocabSize: Int
        public let vocabOffset: Int

        enum CodingKeys: String, CodingKey {
            case inputFeatureSize = "input_feat_size"
            case hiddenSize = "hidden_size"
            case attentionChunkSize = "conf_attention_chunk_size"
            case attentionContextLeft = "conf_attention_context_left"
            case attentionContextRight = "conf_attention_context_right"
            case attentionInvalidLogitsValue = "conf_attention_invalid_logits_value"
            case attentionLogitCap = "conf_attention_logit_cap"
            case attentionHeads = "conf_num_attention_heads"
            case hiddenLayers = "conf_num_hidden_layers"
            case convKernelSize = "conf_conv_kernel_size"
            case reductionFactor = "conf_reduction_factor"
            case residualWeight = "conf_residual_weight"
            case convChannelSize = "sscp_conv_channel_size"
            case convGroupNormEps = "sscp_conv_group_norm_eps"
            case convKernelSizes = "sscp_conv_kernel_size"
            case convStrideSizes = "sscp_conv_stride_size"
            case convEps = "sscp_conv_eps"
            case rmsNormEps = "rms_norm_eps"
            case gradientClipping = "gradient_clipping"
            case vocabSize = "vocab_size"
            case vocabOffset = "vocab_offset"
        }

        public init() {
            inputFeatureSize = 128
            hiddenSize = 1536
            attentionChunkSize = 12
            attentionContextLeft = 13
            attentionContextRight = 0
            attentionInvalidLogitsValue = -1e9
            attentionLogitCap = 50
            attentionHeads = 8
            hiddenLayers = 12
            convKernelSize = 5
            reductionFactor = 4
            residualWeight = 0.5
            convChannelSize = [128, 32]
            convGroupNormEps = 1e-3
            convKernelSizes = [[3, 3], [3, 3]]
            convStrideSizes = [[2, 2], [2, 2]]
            convEps = 1e-3
            rmsNormEps = 1e-6
            gradientClipping = 1e10
            vocabSize = 128
            vocabOffset = 262_272
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            inputFeatureSize = try c.decodeIfPresent(Int.self, forKey: .inputFeatureSize) ?? 128
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1536
            attentionChunkSize =
                try c.decodeIfPresent(Int.self, forKey: .attentionChunkSize) ?? 12
            attentionContextLeft =
                try c.decodeIfPresent(Int.self, forKey: .attentionContextLeft) ?? 13
            attentionContextRight =
                try c.decodeIfPresent(Int.self, forKey: .attentionContextRight) ?? 0
            attentionInvalidLogitsValue =
                try c.decodeIfPresent(Float.self, forKey: .attentionInvalidLogitsValue) ?? -1e9
            attentionLogitCap =
                try c.decodeIfPresent(Float.self, forKey: .attentionLogitCap) ?? 50
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 12
            convKernelSize = try c.decodeIfPresent(Int.self, forKey: .convKernelSize) ?? 5
            reductionFactor = try c.decodeIfPresent(Int.self, forKey: .reductionFactor) ?? 4
            residualWeight = try c.decodeIfPresent(Float.self, forKey: .residualWeight) ?? 0.5
            convChannelSize =
                try c.decodeIfPresent([Int].self, forKey: .convChannelSize) ?? [128, 32]
            convGroupNormEps =
                try c.decodeIfPresent(Float.self, forKey: .convGroupNormEps) ?? 1e-3
            convKernelSizes =
                try c.decodeIfPresent([[Int]].self, forKey: .convKernelSizes)
                ?? [[3, 3], [3, 3]]
            convStrideSizes =
                try c.decodeIfPresent([[Int]].self, forKey: .convStrideSizes)
                ?? [[2, 2], [2, 2]]
            convEps = try c.decodeIfPresent(Float.self, forKey: .convEps) ?? convGroupNormEps
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            gradientClipping =
                try c.decodeIfPresent(Float.self, forKey: .gradientClipping) ?? 1e10
            vocabSize = try c.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 128
            vocabOffset = try c.decodeIfPresent(Int.self, forKey: .vocabOffset) ?? 262_272
        }

        func kernelSize(at index: Int) -> (Int, Int) {
            let pair = index < convKernelSizes.count ? convKernelSizes[index] : [3, 3]
            return (pair.first ?? 3, pair.dropFirst().first ?? pair.first ?? 3)
        }

        func strideSize(at index: Int) -> (Int, Int) {
            let pair = index < convStrideSizes.count ? convStrideSizes[index] : [2, 2]
            return (pair.first ?? 2, pair.dropFirst().first ?? pair.first ?? 2)
        }
    }

    public let modelType: String
    public let textConfiguration: TextConfiguration
    public let textModelConfiguration: Gemma3nTextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let audioConfiguration: AudioConfiguration
    public let imageTokenID: Int
    public let audioTokenID: Int
    public let visionSoftTokensPerImage: Int
    public let audioSoftTokensPerImage: Int
    public let hiddenSize: Int
    public let eosTokenIDs: [Int]?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case audioConfiguration = "audio_config"
        case imageTokenID = "image_token_id"
        case audioTokenID = "audio_token_id"
        case visionSoftTokensPerImage = "vision_soft_tokens_per_image"
        case audioSoftTokensPerImage = "audio_soft_tokens_per_image"
        case hiddenSize = "hidden_size"
        case eosTokenID = "eos_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gemma3n"
        textConfiguration = try c.decode(TextConfiguration.self, forKey: .textConfiguration)
        textModelConfiguration = try Gemma3nTextConfiguration(from: decoder)
        visionConfiguration = try c.decode(VisionConfiguration.self, forKey: .visionConfiguration)
        audioConfiguration =
            try c.decodeIfPresent(AudioConfiguration.self, forKey: .audioConfiguration) ?? .init()
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? 262_145
        audioTokenID = try c.decodeIfPresent(Int.self, forKey: .audioTokenID) ?? 262_273
        visionSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: .visionSoftTokensPerImage) ?? 256
        audioSoftTokensPerImage =
            try c.decodeIfPresent(Int.self, forKey: .audioSoftTokensPerImage) ?? 188
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        if let ids = try c.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenID) {
            eosTokenIDs = ids.values
        } else {
            eosTokenIDs = nil
        }
    }
}

public enum Gemma3nIntOrSize: Decodable, Sendable {
    case int(Int)
    case size(width: Int, height: Int)

    var width: Int {
        switch self {
        case .int(let value): value
        case .size(let width, _): width
        }
    }

    var height: Int {
        switch self {
        case .int(let value): value
        case .size(_, let height): height
        }
    }

    enum CodingKeys: String, CodingKey {
        case width
        case height
        case shortestEdge = "shortest_edge"
    }

    public init(from decoder: Decoder) throws {
        if let value = try? decoder.singleValueContainer().decode(Int.self) {
            self = .int(value)
            return
        }
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let height =
            try c.decodeIfPresent(Int.self, forKey: .height)
            ?? c.decodeIfPresent(Int.self, forKey: .shortestEdge)
            ?? 768
        let width =
            try c.decodeIfPresent(Int.self, forKey: .width)
            ?? c.decodeIfPresent(Int.self, forKey: .shortestEdge)
            ?? height
        self = .size(width: width, height: height)
    }
}

public struct Gemma3nProcessorConfiguration: Decodable, Sendable {
    public let processorClass: String?
    public let imageSeqLength: Int
    public let audioSeqLength: Int
    public let audioTokenID: Int
    public let boaTokenID: Int
    public let eoaTokenID: Int
    public let size: Gemma3nIntOrSize
    public let doResize: Bool
    public let doRescale: Bool
    public let doNormalize: Bool
    public let rescaleFactor: Float
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let audioFeatureExtractor: Gemma4AudioFeatureExtractorConfiguration

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageSeqLength = "image_seq_length"
        case audioSeqLength = "audio_seq_length"
        case audioTokenID = "audio_token_id"
        case boaTokenID = "boa_token_id"
        case eoaTokenID = "eoa_token_id"
        case size
        case doResize = "do_resize"
        case doRescale = "do_rescale"
        case doNormalize = "do_normalize"
        case rescaleFactor = "rescale_factor"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case featureExtractor = "feature_extractor"
        case samplingRate = "sampling_rate"
        case featureSize = "feature_size"
        case paddingValue = "padding_value"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        processorClass = try c.decodeIfPresent(String.self, forKey: .processorClass)
        imageSeqLength = try c.decodeIfPresent(Int.self, forKey: .imageSeqLength) ?? 256
        audioSeqLength = try c.decodeIfPresent(Int.self, forKey: .audioSeqLength) ?? 188
        audioTokenID = try c.decodeIfPresent(Int.self, forKey: .audioTokenID) ?? 262_273
        boaTokenID = try c.decodeIfPresent(Int.self, forKey: .boaTokenID) ?? 256_000
        eoaTokenID = try c.decodeIfPresent(Int.self, forKey: .eoaTokenID) ?? 262_272
        size =
            try c.decodeIfPresent(Gemma3nIntOrSize.self, forKey: .size)
            ?? .size(
                width: 768, height: 768,
            )
        doResize = try c.decodeIfPresent(Bool.self, forKey: .doResize) ?? true
        doRescale = try c.decodeIfPresent(Bool.self, forKey: .doRescale) ?? true
        doNormalize = try c.decodeIfPresent(Bool.self, forKey: .doNormalize) ?? false
        rescaleFactor =
            try c.decodeIfPresent(Float.self, forKey: .rescaleFactor)
            ?? (1.0 / 255.0)
        imageMean = try c.decodeIfPresent([CGFloat].self, forKey: .imageMean) ?? [0.5, 0.5, 0.5]
        imageStd = try c.decodeIfPresent([CGFloat].self, forKey: .imageStd) ?? [0.5, 0.5, 0.5]
        let featureDecoder =
            if c.contains(.featureExtractor), try !c.decodeNil(forKey: .featureExtractor) {
                try c.superDecoder(forKey: .featureExtractor)
            } else {
                decoder
            }
        audioFeatureExtractor = try Gemma4AudioFeatureExtractorConfiguration(
            from: featureDecoder,
            defaults: .init(
                frameLengthMs: 32, minFrequency: 125, maxFrequency: 7600,
                preemphasis: 0.97, fftOverdrive: true, melFloor: 1e-5),
        )
    }
}

// MARK: - Processor

public struct Gemma3nProcessor: UserInputProcessor {
    private let config: Gemma3nProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let imageTokenID: Int
    private let audioTokenID: Int
    private let boaTokenID: Int
    private let eoaTokenID: Int
    private let featureExtractor: Gemma4AudioFeatureExtractor

    public init(_ config: Gemma3nProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
        imageTokenID = tokenizer.convertTokenToId("<image_soft_token>") ?? 262_145
        audioTokenID = tokenizer.convertTokenToId("<audio_soft_token>") ?? config.audioTokenID
        boaTokenID = tokenizer.convertTokenToId("<start_of_audio>") ?? config.boaTokenID
        eoaTokenID = tokenizer.convertTokenToId("<end_of_audio>") ?? config.eoaTokenID
        featureExtractor = Gemma4AudioFeatureExtractor(
            config.audioFeatureExtractor, variant: .gemma3n)
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let prompt = Self.prompt(from: input).trimmingCharacters(in: .whitespacesAndNewlines)
        let images = MediaProcessing.images(from: input)
        let audios = MediaProcessing.audios(from: input)

        let imageBlock =
            images.isEmpty
            ? ""
            : "\n\n<start_of_image>"
                + String(repeating: "<image_soft_token>", count: config.imageSeqLength)
                + "<end_of_image>\n\n"
        let audioBlock =
            audios.isEmpty
            ? ""
            : String(
                repeating: "\n\n<start_of_audio>"
                    + String(repeating: "<audio_soft_token>", count: config.audioSeqLength)
                    + "<end_of_audio>\n\n",
                count: audios.count,
            )
        let promptText =
            "<bos><start_of_turn>user\n\(imageBlock)\(prompt)\(audioBlock)<end_of_turn>\n<start_of_turn>model\n"
        let tokens = tokenizer.encode(text: promptText, addSpecialTokens: true)

        var processedImage: LMInput.ProcessedImage?
        if let image = images.first {
            let pixels = try preprocess(image: image.asCIImage(), processing: input.processing)
            processedImage = LMInput.ProcessedImage(
                pixels: pixels,
                frames: [THW(1, config.size.height, config.size.width)],
            )
        }

        var processedAudio: LMInput.ProcessedAudio?
        if !audios.isEmpty {
            processedAudio = try await preprocess(audios: audios, processing: input.processing)
        }

        return LMInput(
            text: .init(tokens: MLXArray(tokens)[.newAxis]),
            image: processedImage,
            audio: processedAudio,
        )
    }

    private func preprocess(audios: [UserInput.Audio], processing: UserInput.Processing)
        async throws
        -> LMInput.ProcessedAudio
    {
        var audioProcessing = processing.audio
        audioProcessing.sampleRate = Double(config.audioFeatureExtractor.samplingRate)
        audioProcessing.channels = 1

        var samples: [MLXArray] = []
        samples.reserveCapacity(audios.count)
        for audio in audios {
            try await samples.append(audio.asMLXArray(processing: audioProcessing))
        }
        let extracted = try featureExtractor(samples)
        // Finish thread-local FFT work before generation consumes the prepared input.
        eval(extracted.inputFeatures, extracted.inputFeaturesMask)
        return LMInput.ProcessedAudio(
            features: extracted.inputFeatures,
            mask: extracted.inputFeaturesMask,
        )
    }

    private func preprocess(image source: CIImage, processing: UserInput.Processing?) throws
        -> MLXArray
    {
        var image = MediaProcessing.apply(source, processing: processing)
        image = Self.zeroOrigin(image)
        let width = config.size.width
        let height = config.size.height
        if config.doResize {
            image = Self.zeroOrigin(
                image.resampled(to: CGSize(width: width, height: height), method: .bicubic),
            )
        }

        var raw = Self.renderRGB(image, width: width, height: height)
        if config.doRescale {
            raw = raw.map { $0 * config.rescaleFactor }
        }
        if config.doNormalize {
            for index in stride(from: 0, to: raw.count, by: 3) {
                raw[index] = (raw[index] - Float(config.imageMean[0])) / Float(config.imageStd[0])
                raw[index + 1] =
                    (raw[index + 1] - Float(config.imageMean[1])) / Float(config.imageStd[1])
                raw[index + 2] =
                    (raw[index + 2] - Float(config.imageMean[2])) / Float(config.imageStd[2])
            }
        }
        return MLXArray(raw, [height, width, 3]).transposed(2, 0, 1)[.newAxis]
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

    private static func zeroOrigin(_ image: CIImage) -> CIImage {
        image.transformed(
            by: CGAffineTransform(translationX: -image.extent.origin.x, y: -image.extent.origin.y))
    }

    private static func renderRGB(_ image: CIImage, width: Int, height: Int) -> [Float] {
        let format = CIFormat.RGBA8
        let componentsPerPixel = 4
        let bytesPerPixel = componentsPerPixel * MemoryLayout<UInt8>.stride
        let bytesPerRow = width * bytesPerPixel
        var data = Data(count: width * height * bytesPerPixel)
        let colorSpace = CGColorSpace(name: CGColorSpace.sRGB)
        let context = CIContext(options: [.cacheIntermediates: false])
        data.withUnsafeMutableBytes { ptr in
            context.render(
                image,
                toBitmap: ptr.baseAddress!,
                rowBytes: bytesPerRow,
                bounds: CGRect(x: 0, y: 0, width: width, height: height),
                format: format,
                colorSpace: colorSpace,
            )
        }
        let bytes = data.withUnsafeBytes { ptr in
            Array(ptr.bindMemory(to: UInt8.self))
        }
        var rgb = [Float]()
        rgb.reserveCapacity(width * height * 3)
        for index in stride(from: 0, to: bytes.count, by: 4) {
            let alpha = max(Float(bytes[index + 3]), 1)
            rgb.append(min(max(Float(bytes[index]) * 255 / alpha, 0), 255))
            rgb.append(min(max(Float(bytes[index + 1]) * 255 / alpha, 0), 255))
            rgb.append(min(max(Float(bytes[index + 2]) * 255 / alpha, 0), 255))
        }
        return rgb
    }
}

// MARK: - Vision

private final class Gemma3nIdentity: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x
    }
}

private final class Gemma3nRMSNoScale: Module {
    let eps: Float

    init(eps: Float = 1e-6) {
        self.eps = eps
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: MLXArray.mlxNone, eps: eps)
    }
}

private final class Gemma3nRMSNormAct2d: Module, UnaryLayer {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let applyAct: Bool

    init(channels: Int, eps: Float = 1e-6, applyAct: Bool = true) {
        self.eps = eps
        self.applyAct = applyAct
        _weight.wrappedValue = MLXArray.ones([channels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x * rsqrt(mean(x.square(), axis: -1, keepDims: true) + eps)
        y = y * weight.asType(y.dtype)
        return applyAct ? gelu(y) : y
    }
}

private final class Gemma3nLayerScale2d: Module, UnaryLayer {
    @ParameterInfo(key: "gamma") var gamma: MLXArray

    init(dim: Int, initValues: Float = 1e-5) {
        _gamma.wrappedValue = MLXArray.ones([dim]) * initValues
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * gamma.asType(x.dtype)
    }
}

private final class Gemma3nConv2dSame: Conv2d {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        let kernel = (weight.dim(1), weight.dim(2))
        let padH = Self.samePadding(
            input: x.dim(1), kernel: kernel.0, stride: stride.0, dilation: dilation.0)
        let padW = Self.samePadding(
            input: x.dim(2), kernel: kernel.1, stride: stride.1, dilation: dilation.1)
        let paddedInput = padded(
            x,
            widths: [0, [padH / 2, padH - padH / 2], [padW / 2, padW - padW / 2], 0],
            mode: .constant,
            value: MLXArray(0, dtype: x.dtype),
        )
        var y = conv2d(
            paddedInput,
            weight,
            stride: .init(stride),
            padding: 0,
            dilation: .init(dilation),
            groups: groups,
        )
        if let bias {
            y = y + bias
        }
        return y
    }

    private static func samePadding(input: Int, kernel: Int, stride: Int, dilation: Int) -> Int {
        let effective = dilation * (kernel - 1) + 1
        let output = (input + stride - 1) / stride
        return max(0, (output - 1) * stride + effective - input)
    }
}

private final class Gemma3nConvNormAct: Module, UnaryLayer {
    @ModuleInfo(key: "conv") var conv: Module
    @ModuleInfo(key: "bn") var norm: Gemma3nRMSNormAct2d

    init(
        same: Bool,
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        padding: Int = 0,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = false,
        applyAct: Bool = true,
        eps: Float = 1e-6,
    ) {
        if same {
            _conv.wrappedValue = Gemma3nConv2dSame(
                inputChannels: inChannels,
                outputChannels: outChannels,
                kernelSize: IntOrPair(kernelSize),
                stride: IntOrPair(stride),
                padding: 0,
                dilation: IntOrPair(dilation),
                groups: groups,
                bias: bias,
            )
        } else {
            _conv.wrappedValue = Conv2d(
                inputChannels: inChannels,
                outputChannels: outChannels,
                kernelSize: IntOrPair(kernelSize),
                stride: IntOrPair(stride),
                padding: IntOrPair(padding),
                dilation: IntOrPair(dilation),
                groups: groups,
                bias: bias,
            )
        }
        _norm.wrappedValue = Gemma3nRMSNormAct2d(
            channels: outChannels, eps: eps, applyAct: applyAct,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        norm((conv as! any UnaryLayer)(x))
    }
}

private func gemma3nMakeDivisible(_ value: Float, divisor: Int = 8, minValue: Int? = nil) -> Int {
    let minValue = minValue ?? divisor
    var newValue = max(minValue, Int(value + Float(divisor) / 2) / divisor * divisor)
    if Float(newValue) < 0.9 * value {
        newValue += divisor
    }
    return newValue
}

private func gemma3nNumGroups(groupSize: Int, channels: Int) -> Int {
    groupSize <= 0 ? 1 : channels / groupSize
}

private final class Gemma3nUniversalInvertedResidual: Module, UnaryLayer {
    let hasSkip: Bool
    @ModuleInfo(key: "dw_start") var dwStart: Module
    @ModuleInfo(key: "pw_exp") var pwExp: Gemma3nConvNormAct
    @ModuleInfo(key: "dw_mid") var dwMid: Module
    @ModuleInfo(key: "pw_proj") var pwProj: Gemma3nConvNormAct
    @ModuleInfo(key: "layer_scale") var layerScale: Module

    init(
        inChannels: Int,
        outChannels: Int,
        dwKernelSizeStart: Int = 0,
        dwKernelSizeMid: Int = 3,
        stride: Int = 1,
        dilation: Int = 1,
        groupSize: Int = 1,
        noSkip: Bool = false,
        expansionRatio: Float = 1.0,
        layerScaleInitValue: Float? = 1e-5,
    ) {
        hasSkip = inChannels == outChannels && stride == 1 && !noSkip
        if dwKernelSizeStart > 0 {
            _dwStart.wrappedValue = Gemma3nConvNormAct(
                same: true,
                inChannels: inChannels,
                outChannels: inChannels,
                kernelSize: dwKernelSizeStart,
                stride: 1,
                dilation: dilation,
                groups: gemma3nNumGroups(groupSize: groupSize, channels: inChannels),
                applyAct: false,
                eps: 1e-5,
            )
        } else {
            _dwStart.wrappedValue = Gemma3nIdentity()
        }
        let midChannels = gemma3nMakeDivisible(Float(inChannels) * expansionRatio)
        _pwExp.wrappedValue = Gemma3nConvNormAct(
            same: false,
            inChannels: inChannels,
            outChannels: midChannels,
            kernelSize: 1,
            eps: 1e-5,
        )
        if dwKernelSizeMid > 0 {
            _dwMid.wrappedValue = Gemma3nConvNormAct(
                same: true,
                inChannels: midChannels,
                outChannels: midChannels,
                kernelSize: dwKernelSizeMid,
                stride: stride,
                dilation: dilation,
                groups: gemma3nNumGroups(groupSize: groupSize, channels: midChannels),
                eps: 1e-5,
            )
        } else {
            _dwMid.wrappedValue = Gemma3nIdentity()
        }
        _pwProj.wrappedValue = Gemma3nConvNormAct(
            same: false,
            inChannels: midChannels,
            outChannels: outChannels,
            kernelSize: 1,
            applyAct: false,
            eps: 1e-5,
        )
        if let layerScaleInitValue {
            _layerScale.wrappedValue = Gemma3nLayerScale2d(
                dim: outChannels, initValues: layerScaleInitValue)
        } else {
            _layerScale.wrappedValue = Gemma3nIdentity()
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var y = (dwStart as! any UnaryLayer)(x)
        y = pwExp(y)
        y = (dwMid as! any UnaryLayer)(y)
        y = pwProj(y)
        y = (layerScale as! any UnaryLayer)(y)
        return hasSkip ? y + residual : y
    }
}

private final class Gemma3nEdgeResidual: Module, UnaryLayer {
    let hasSkip: Bool
    @ModuleInfo(key: "conv_exp") var convExp: Gemma3nConv2dSame
    @ModuleInfo(key: "bn1") var norm1: Gemma3nRMSNormAct2d
    @ModuleInfo(key: "conv_pwl") var convPwl: Conv2d
    @ModuleInfo(key: "bn2") var norm2: Gemma3nRMSNormAct2d

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        expansionRatio: Float = 1.0,
    ) {
        hasSkip = inChannels == outChannels && stride == 1
        let midChannels = gemma3nMakeDivisible(Float(inChannels) * expansionRatio)
        _convExp.wrappedValue = Gemma3nConv2dSame(
            inputChannels: inChannels,
            outputChannels: midChannels,
            kernelSize: IntOrPair(kernelSize),
            stride: IntOrPair(stride),
            padding: 0,
            groups: 1,
            bias: false,
        )
        _norm1.wrappedValue = Gemma3nRMSNormAct2d(channels: midChannels, eps: 1e-5)
        _convPwl.wrappedValue = Conv2d(
            inputChannels: midChannels,
            outputChannels: outChannels,
            kernelSize: 1,
            padding: 0,
            bias: false,
        )
        _norm2.wrappedValue = Gemma3nRMSNormAct2d(
            channels: outChannels, eps: 1e-5, applyAct: false,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = convExp(x)
        y = norm1(y)
        y = convPwl(y)
        y = norm2(y)
        return hasSkip ? y + x : y
    }
}

private final class Gemma3nAttentionProjection: Module, UnaryLayer {
    @ModuleInfo(key: "down_conv") var downConv: Module?
    @ModuleInfo(key: "norm") var norm: Gemma3nRMSNormAct2d?
    @ModuleInfo(key: "proj") var proj: Conv2d

    init(
        dim: Int,
        outputDim: Int,
        kernelSize: Int = 1,
        kvStride: Int = 1,
        depthwiseDownsample: Bool = false,
        bias: Bool = false,
    ) {
        if depthwiseDownsample, kvStride > 1 {
            _downConv.wrappedValue = Conv2d(
                inputChannels: dim,
                outputChannels: dim,
                kernelSize: IntOrPair(kernelSize),
                stride: IntOrPair(kvStride),
                padding: IntOrPair((kernelSize - 1) / 2),
                groups: dim,
                bias: false,
            )
            _norm.wrappedValue = Gemma3nRMSNormAct2d(channels: dim, eps: 1e-6, applyAct: false)
        } else {
            _downConv.wrappedValue = nil
            _norm.wrappedValue = nil
        }
        _proj.wrappedValue = Conv2d(
            inputChannels: dim,
            outputChannels: outputDim,
            kernelSize: 1,
            padding: 0,
            bias: bias,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        if let downConv {
            y = (downConv as! any UnaryLayer)(y)
            if let norm {
                y = norm(y)
            }
        }
        return proj(y)
    }
}

private final class Gemma3nMultiQueryAttention2d: Module, UnaryLayer {
    let numHeads: Int
    let keyDim: Int
    let valueDim: Int
    let scale: Float
    let queryStrides: (Int, Int)

    @ModuleInfo(key: "query") var query: Gemma3nAttentionProjection
    @ModuleInfo(key: "key") var key: Gemma3nAttentionProjection
    @ModuleInfo(key: "value") var value: Gemma3nAttentionProjection
    @ModuleInfo(key: "output") var output: Gemma3nAttentionProjection

    init(
        dim: Int,
        outputDim: Int? = nil,
        numHeads: Int,
        keyDim: Int,
        valueDim: Int,
        queryStrides: (Int, Int) = (1, 1),
        kvStride: Int = 1,
        dwKernelSize: Int = 3,
    ) {
        self.numHeads = numHeads
        self.keyDim = keyDim
        self.valueDim = valueDim
        scale = pow(Float(keyDim), -0.5)
        self.queryStrides = queryStrides
        _query.wrappedValue = Gemma3nAttentionProjection(
            dim: dim, outputDim: numHeads * keyDim, bias: false,
        )
        _key.wrappedValue = Gemma3nAttentionProjection(
            dim: dim,
            outputDim: keyDim,
            kernelSize: dwKernelSize,
            kvStride: kvStride,
            depthwiseDownsample: true,
            bias: false,
        )
        _value.wrappedValue = Gemma3nAttentionProjection(
            dim: dim,
            outputDim: valueDim,
            kernelSize: dwKernelSize,
            kvStride: kvStride,
            depthwiseDownsample: true,
            bias: false,
        )
        _output.wrappedValue = Gemma3nAttentionProjection(
            dim: valueDim * numHeads,
            outputDim: outputDim ?? dim,
            bias: false,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let height = x.dim(1)
        let width = x.dim(2)
        var q = query(x).reshaped(batch, height * width, numHeads, keyDim)
            .transposed(0, 2, 1, 3)
        let kRaw = key(x)
        let vRaw = value(x)
        let k = kRaw.reshaped(batch, -1, keyDim)[0..., .newAxis, 0..., 0...]
        let v = vRaw.reshaped(batch, -1, valueDim)[0..., .newAxis, 0..., 0...]
        q = q.asType(k.dtype)
        let out = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: nil,
        )
        let outHeight = height / queryStrides.0
        let outWidth = width / queryStrides.1
        let reshaped = out.transposed(0, 2, 1, 3)
            .reshaped(batch, outHeight, outWidth, numHeads * valueDim)
        return output(reshaped)
    }
}

private final class Gemma3nMobileAttention: Module, UnaryLayer {
    let hasSkip: Bool
    @ModuleInfo(key: "norm") var norm: Gemma3nRMSNormAct2d
    @ModuleInfo(key: "attn") var attention: Gemma3nMultiQueryAttention2d
    @ModuleInfo(key: "layer_scale") var layerScale: Module

    init(
        inChannels: Int,
        outChannels: Int,
        numHeads: Int,
        keyDim: Int,
        valueDim: Int,
        kvStride: Int,
        dwKernelSize: Int = 3,
        layerScaleInitValue: Float? = 1e-5,
    ) {
        hasSkip = inChannels == outChannels
        _norm.wrappedValue = Gemma3nRMSNormAct2d(
            channels: inChannels, eps: 1e-5, applyAct: false,
        )
        _attention.wrappedValue = Gemma3nMultiQueryAttention2d(
            dim: inChannels,
            outputDim: outChannels,
            numHeads: numHeads,
            keyDim: keyDim,
            valueDim: valueDim,
            kvStride: kvStride,
            dwKernelSize: dwKernelSize,
        )
        if let layerScaleInitValue {
            _layerScale.wrappedValue = Gemma3nLayerScale2d(
                dim: outChannels, initValues: layerScaleInitValue,
            )
        } else {
            _layerScale.wrappedValue = Gemma3nIdentity()
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = norm(x)
        y = attention(y)
        y = (layerScale as! any UnaryLayer)(y)
        return hasSkip ? y + x : y
    }
}

private final class Gemma3nMultiScaleFusionAdapter: Module, UnaryLayer {
    let outputResolution: (Int, Int)
    @ModuleInfo(key: "ffn") var ffn: Gemma3nUniversalInvertedResidual
    @ModuleInfo(key: "norm") var norm: Gemma3nRMSNormAct2d

    init(inChannels: Int, outChannels: Int, outputResolution: (Int, Int)) {
        self.outputResolution = outputResolution
        _ffn.wrappedValue = Gemma3nUniversalInvertedResidual(
            inChannels: inChannels,
            outChannels: outChannels,
            dwKernelSizeMid: 0,
            noSkip: true,
            expansionRatio: 2.0,
            layerScaleInitValue: nil,
        )
        _norm.wrappedValue = Gemma3nRMSNormAct2d(
            channels: outChannels, eps: 1e-6, applyAct: false,
        )
        super.init()
    }

    func callAsFunction(_ inputs: [MLXArray]) -> MLXArray {
        let targetH = inputs[0].dim(1)
        let targetW = inputs[0].dim(2)
        let resized = inputs.map { image in
            if image.dim(1) == targetH, image.dim(2) == targetW {
                image
            } else {
                nearestResize(image, height: targetH, width: targetW)
            }
        }
        var x = concatenated(resized, axis: -1)
        x = ffn(x)
        if x.dim(1) != outputResolution.0 || x.dim(2) != outputResolution.1 {
            if x.dim(1) % outputResolution.0 == 0, x.dim(2) % outputResolution.1 == 0 {
                let hStride = x.dim(1) / outputResolution.0
                let wStride = x.dim(2) / outputResolution.1
                x = AvgPool2d(kernelSize: [hStride, wStride], stride: [hStride, wStride])(x)
            } else {
                x = nearestResize(x, height: outputResolution.0, width: outputResolution.1)
            }
            x = norm(x)
        }
        return x
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x
    }

    private func nearestResize(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let sourceH = x.dim(1)
        let sourceW = x.dim(2)
        let hIdx = (MLXArray(0 ..< height).asType(.float32) * Float(sourceH) / Float(height))
            .asType(.int32)
        let wIdx = (MLXArray(0 ..< width).asType(.float32) * Float(sourceW) / Float(width))
            .asType(.int32)
        return x.take(hIdx, axis: 1).take(wIdx, axis: 2)
    }
}

private enum Gemma3nVisionBlockConfig {
    case edge(kernel: Int, filters: Int, stride: Int, expansion: Float)
    case inverted(start: Int, mid: Int, filters: Int, stride: Int, expansion: Float)
    case attention(heads: Int, kvDim: Int, kvStride: Int)
}

private func gemma3nVisionDefinition() -> [[Gemma3nVisionBlockConfig]] {
    [
        [.edge(kernel: 3, filters: 128, stride: 2, expansion: 4.0)]
            + Array(repeating: .edge(kernel: 3, filters: 128, stride: 1, expansion: 4.0), count: 2),
        [.inverted(start: 3, mid: 5, filters: 256, stride: 2, expansion: 6.0)]
            + [5, 3, 5, 3].map {
                .inverted(start: $0, mid: 0, filters: 256, stride: 1, expansion: 4.0)
            },
        [.inverted(start: 5, mid: 5, filters: 640, stride: 2, expansion: 6.0)]
            + Array(
                repeating: .inverted(start: 5, mid: 0, filters: 640, stride: 1, expansion: 4.0),
                count: 7,
            )
            + [.inverted(start: 0, mid: 0, filters: 640, stride: 1, expansion: 1.0)]
            + (0 ..< 14).flatMap { _ in
                [
                    .attention(heads: 12, kvDim: 64, kvStride: 2),
                    .inverted(start: 0, mid: 0, filters: 640, stride: 1, expansion: 2.0),
                ]
            },
        [.inverted(start: 5, mid: 5, filters: 1280, stride: 2, expansion: 6.0)]
            + (0 ..< 19).flatMap { _ in
                [
                    .attention(heads: 16, kvDim: 96, kvStride: 1),
                    .inverted(start: 0, mid: 0, filters: 1280, stride: 1, expansion: 2.0),
                ]
            },
    ]
}

private final class Gemma3nVisionTower: Module {
    @ModuleInfo(key: "conv_stem") var convStem: Gemma3nConvNormAct
    @ModuleInfo(key: "blocks") var blocks: [[Module]]
    @ModuleInfo(key: "msfa") var msfa: Gemma3nMultiScaleFusionAdapter

    override init() {
        _convStem.wrappedValue = Gemma3nConvNormAct(
            same: true,
            inChannels: 3,
            outChannels: 64,
            kernelSize: 3,
            stride: 2,
            bias: true,
            eps: 1e-5,
        )
        var inChannels = 64
        var stages: [[Module]] = []
        for stage in gemma3nVisionDefinition() {
            var modules: [Module] = []
            for block in stage {
                switch block {
                case .edge(let kernel, let filters, let stride, let expansion):
                    modules.append(
                        Gemma3nEdgeResidual(
                            inChannels: inChannels,
                            outChannels: filters,
                            kernelSize: kernel,
                            stride: stride,
                            expansionRatio: expansion,
                        ),
                    )
                    inChannels = filters
                case .inverted(let start, let mid, let filters, let stride, let expansion):
                    modules.append(
                        Gemma3nUniversalInvertedResidual(
                            inChannels: inChannels,
                            outChannels: filters,
                            dwKernelSizeStart: start,
                            dwKernelSizeMid: mid,
                            stride: stride,
                            expansionRatio: expansion,
                        ),
                    )
                    inChannels = filters
                case .attention(let heads, let kvDim, let kvStride):
                    modules.append(
                        Gemma3nMobileAttention(
                            inChannels: inChannels,
                            outChannels: inChannels,
                            numHeads: heads,
                            keyDim: kvDim,
                            valueDim: kvDim,
                            kvStride: kvStride,
                        ),
                    )
                }
            }
            stages.append(modules)
        }
        _blocks.wrappedValue = stages
        _msfa.wrappedValue = Gemma3nMultiScaleFusionAdapter(
            inChannels: 1920,
            outChannels: 2048,
            outputResolution: (16, 16),
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x.transposed(0, 2, 3, 1)
        x = convStem(x)
        var intermediates: [MLXArray] = []
        var previousOutput: MLXArray?
        for (stageIndex, stage) in blocks.enumerated() {
            for block in stage {
                x = (block as! any UnaryLayer)(x)
                // Bound inference intermediates while overlapping graph construction
                // with the preceding block. Training keeps the complete graph.
                if !training {
                    asyncEval(x)
                    if let previousOutput { eval(previousOutput) }
                    previousOutput = x
                }
            }
            let featureIndex = stageIndex + 1
            if featureIndex == 3 || featureIndex == 4 {
                intermediates.append(x)
            }
        }
        return msfa(intermediates)
    }
}

private final class Gemma3nVisionModel: Module {
    @ModuleInfo(key: "timm_model") var timmModel: Gemma3nVisionTower

    init(_: Gemma3nConfiguration.VisionConfiguration) {
        _timmModel.wrappedValue = Gemma3nVisionTower()
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        timmModel(x)
    }
}

private final class Gemma3nMultimodalEmbedder: Module {
    let vocabOffset: Int
    let vocabSize: Int
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "hard_embedding_norm") var hardEmbeddingNorm: RMSNorm
    @ModuleInfo(key: "soft_embedding_norm") var softEmbeddingNorm: RMSNorm
    @ModuleInfo(key: "embedding_projection") var embeddingProjection: Linear
    @ModuleInfo(key: "embedding_post_projection_norm") var embeddingPostProjectionNorm:
        Gemma3nRMSNoScale

    init(
        multimodalHiddenSize: Int, textHiddenSize: Int, vocabSize: Int, vocabOffset: Int, eps: Float
    ) {
        self.vocabOffset = vocabOffset
        self.vocabSize = vocabSize
        _embedding.wrappedValue = Embedding(
            embeddingCount: vocabSize, dimensions: multimodalHiddenSize,
        )
        _hardEmbeddingNorm.wrappedValue = RMSNorm(
            dimensions: multimodalHiddenSize, eps: eps,
        )
        _softEmbeddingNorm.wrappedValue = RMSNorm(
            dimensions: multimodalHiddenSize, eps: eps,
        )
        _embeddingProjection.wrappedValue = Linear(
            multimodalHiddenSize, textHiddenSize, bias: false,
        )
        _embeddingPostProjectionNorm.wrappedValue = Gemma3nRMSNoScale(eps: eps)
        super.init()
    }

    func callAsFunction(inputIDs: MLXArray) -> MLXArray {
        let hard = embedding(inputIDs - vocabOffset)
        return project(hardEmbeddingNorm(hard))
    }

    func callAsFunction(inputsEmbeds: MLXArray) -> MLXArray {
        project(softEmbeddingNorm(inputsEmbeds))
    }

    private func project(_ x: MLXArray) -> MLXArray {
        embeddingPostProjectionNorm(embeddingProjection(x))
    }
}

// MARK: - Audio

private final class Gemma3nAudioRMSNorm: Module, UnaryLayer {
    let eps: Float
    @ModuleInfo var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-6) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

private final class Gemma3nAudioCumulativeGroupNorm: Module, UnaryLayer {
    let channels: Int
    let eps: Float
    @ModuleInfo var weight: MLXArray

    init(channels: Int, eps: Float) {
        self.channels = channels
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([channels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let xFloat = x.asType(.float32)
        let sumAtTime = sum(xFloat, axes: [2, 3], keepDims: true)
        let countAtTime = MLXArray(Float32(x.dim(2) * x.dim(3)))
        let cumulativeCount = cumsum(ones(like: sumAtTime) * countAtTime, axis: 1)
        let cumulativeMean = cumsum(sumAtTime, axis: 1) / maximum(cumulativeCount, MLXArray(1.0))
        let squaredDiff = (xFloat - cumulativeMean).square()
        let cumulativeVariance =
            cumsum(sum(squaredDiff, axes: [2, 3], keepDims: true), axis: 1)
            / maximum(cumulativeCount, MLXArray(1.0))
        var normalized = (xFloat - cumulativeMean) * rsqrt(cumulativeVariance + eps)
        normalized = normalized * weight.asType(.float32).reshaped(1, 1, 1, channels)
        return normalized.asType(dtype)
    }
}

private final class Gemma3nAudioSSCPConvBlock: Module, UnaryLayer {
    let padding: (frequencyLeft: Int, frequencyRight: Int, timeTop: Int, timeBottom: Int)
    let stride: (time: Int, frequency: Int)

    @ModuleInfo var conv: Conv2d
    @ModuleInfo var norm: Gemma3nAudioCumulativeGroupNorm

    init(index: Int, inputFrequency: Int, config: Gemma3nConfiguration.AudioConfiguration) {
        let inputChannels = index == 0 ? 1 : config.convChannelSize[index - 1]
        let outputChannels = config.convChannelSize[index]
        let kernel = config.kernelSize(at: index)
        let stride = config.strideSize(at: index)
        self.stride = stride
        padding = (
            frequencyLeft: 1,
            frequencyRight: 1,
            timeTop: 0,
            timeBottom: kernel.0 - 1,
        )
        _conv.wrappedValue = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: .init(kernel),
            stride: .init(stride),
            padding: .init((0, 0)),
            bias: false,
        )
        let paddedFrequency = inputFrequency + padding.frequencyLeft + padding.frequencyRight
        let outputFrequency = ((paddedFrequency - kernel.1) / stride.1) + 1
        _norm.wrappedValue = Gemma3nAudioCumulativeGroupNorm(
            channels: outputChannels,
            eps: config.convEps,
        )
        _ = outputFrequency
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let paddedInput = padded(
            x,
            widths: [
                0,
                0,
                .init((padding.timeTop, padding.timeBottom)),
                .init((padding.frequencyLeft, padding.frequencyRight)),
            ],
        )
        let convolved = conv(paddedInput.transposed(0, 2, 3, 1))
        return relu(norm(convolved)).transposed(0, 3, 1, 2)
    }
}

private final class Gemma3nAudioSubSampleConvProjection: Module, UnaryLayer {
    @ModuleInfo(key: "conv_0") var conv0: Gemma3nAudioSSCPConvBlock
    @ModuleInfo(key: "conv_1") var conv1: Gemma3nAudioSSCPConvBlock
    @ModuleInfo(key: "input_proj_linear") var inputProjectionLinear: Linear

    init(config: Gemma3nConfiguration.AudioConfiguration) {
        var frequency = config.inputFeatureSize
        var frequencies: [Int] = []
        for index in 0 ..< 2 {
            let kernel = config.kernelSize(at: index)
            let stride = config.strideSize(at: index)
            frequency = ((frequency + 2 - kernel.1) / stride.1) + 1
            frequencies.append(frequency)
        }
        _conv0.wrappedValue = Gemma3nAudioSSCPConvBlock(
            index: 0,
            inputFrequency: config.inputFeatureSize,
            config: config,
        )
        _conv1.wrappedValue = Gemma3nAudioSSCPConvBlock(
            index: 1,
            inputFrequency: frequencies[0],
            config: config,
        )
        _inputProjectionLinear.wrappedValue = Linear(
            (config.convChannelSize.last ?? 32) * frequencies[1],
            config.hiddenSize,
            bias: false,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = expandedDimensions(x, axis: 1)
        hidden = conv0(hidden)
        hidden = conv1(hidden)
        let batch = hidden.dim(0)
        let channels = hidden.dim(1)
        let time = hidden.dim(2)
        let frequency = hidden.dim(3)
        hidden = hidden.transposed(0, 2, 3, 1).reshaped(batch, time, frequency * channels)
        return inputProjectionLinear(hidden)
    }
}

private final class Gemma3nAudioConformerFeedForward: Module, UnaryLayer {
    let gradientClipping: Float
    let residualWeight: Float

    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: Gemma3nAudioRMSNorm
    @ModuleInfo(key: "ffw_layer_1") var feedForwardLayer1: Linear
    @ModuleInfo(key: "ffw_layer_2") var feedForwardLayer2: Linear
    @ModuleInfo(key: "post_layer_norm") var postLayerNorm: Gemma3nAudioRMSNorm

    init(config: Gemma3nConfiguration.AudioConfiguration) {
        gradientClipping = config.gradientClipping
        residualWeight = config.residualWeight
        _preLayerNorm.wrappedValue = Gemma3nAudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        _feedForwardLayer1.wrappedValue = Linear(
            config.hiddenSize,
            config.hiddenSize * 4,
            bias: false,
        )
        _feedForwardLayer2.wrappedValue = Linear(
            config.hiddenSize * 4,
            config.hiddenSize,
            bias: false,
        )
        _postLayerNorm.wrappedValue = Gemma3nAudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var hidden = clip(x, min: -gradientClipping, max: gradientClipping)
        hidden = preLayerNorm(hidden)
        hidden = feedForwardLayer1(hidden)
        hidden = silu(hidden)
        hidden = feedForwardLayer2(hidden)
        hidden = clip(hidden, min: -gradientClipping, max: gradientClipping)
        hidden = postLayerNorm(hidden)
        return residual + hidden * residualWeight
    }
}

private final class Gemma3nAudioRelativePositionEmbedding: Module {
    let heads: Int
    let channels: Int
    let headDim: Int
    let maxBackward: Int
    let maxForward: Int
    let inverseTimescales: [Float]

    @ModuleInfo(key: "pos_proj") var positionProjection: Linear

    init(config: Gemma3nConfiguration.AudioConfiguration) {
        heads = config.attentionHeads
        channels = config.hiddenSize
        headDim = config.hiddenSize / config.attentionHeads
        maxBackward = max(0, config.attentionContextLeft - 1)
        maxForward = config.attentionContextRight
        let timescales = max(1, config.hiddenSize / 2)
        let increment = log(10000.0) / Double(max(timescales - 1, 1))
        inverseTimescales = (0 ..< timescales).map { index in
            Float(exp(Double(index) * -increment))
        }
        _positionProjection.wrappedValue = Linear(
            config.hiddenSize,
            config.attentionHeads * headDim,
            bias: false,
        )
        super.init()
    }

    private func timingSignal(_ positions: MLXArray, dtype: DType) -> MLXArray {
        let scaledTime =
            expandedDimensions(positions.asType(.float32), axis: -1)
            * MLXArray(inverseTimescales, [1, 1, inverseTimescales.count])
        return concatenated([sin(scaledTime), cos(scaledTime)], axis: -1).asType(dtype)
    }

    private func relativeShift(
        _ term: MLXArray,
        batch: Int,
        heads: Int,
        blocks: Int,
        blockSize: Int,
        contextSize: Int,
        maxSpanPlusOne: Int,
    )
        -> MLXArray
    {
        let padAmount = (contextSize + 1) - maxSpanPlusOne
        var shifted = padded(term, widths: [0, 0, 0, 0, .init((0, padAmount))])
        shifted = shifted.reshaped(batch, heads, blocks, blockSize * (contextSize + 1))
        shifted = shifted[0..., 0..., 0..., ..<(blockSize * contextSize)]
        return shifted.reshaped(batch, heads, blocks, blockSize, contextSize)
    }

    func callAsFunction(queries: MLXArray, keys: MLXArray) -> MLXArray {
        let batch = queries.dim(0)
        let blocks = queries.dim(1)
        let blockSize = queries.dim(2)
        let keyContext = keys.dim(2)

        let positions = MLXArray(
            Array(stride(from: maxBackward, through: -maxForward, by: -1)).map(Int32.init),
            [1, maxBackward + maxForward + 1],
        )
        let maxSpanPlusOne = positions.dim(1)
        var sinusoidal = timingSignal(positions, dtype: queries.dtype)
        sinusoidal = positionProjection(sinusoidal.asType(positionProjection.weight.dtype))
        sinusoidal = sinusoidal.reshaped(maxSpanPlusOne, heads, headDim).asType(queries.dtype)

        let queriesProjected = queries.transposed(0, 3, 1, 2, 4)
        let keysProjected = keys.transposed(0, 3, 1, 4, 2)
        let termAC = matmul(queriesProjected, keysProjected)

        let sinusoidalTransposed = sinusoidal.transposed(1, 2, 0)
        var termBD = matmul(
            queriesProjected.reshaped(batch, heads, blocks * blockSize, headDim),
            sinusoidalTransposed,
        )
        termBD = termBD.reshaped(batch, heads, blocks, blockSize, maxSpanPlusOne)
        termBD = relativeShift(
            termBD,
            batch: batch,
            heads: heads,
            blocks: blocks,
            blockSize: blockSize,
            contextSize: keyContext,
            maxSpanPlusOne: maxSpanPlusOne,
        )
        return termAC + termBD
    }
}

private final class Gemma3nAudioAttention: Module {
    let heads: Int
    let hiddenSize: Int
    let headDim: Int
    let chunkSize: Int
    let maxFutureHorizon: Int
    let maxPastHorizon: Int
    let contextSize: Int
    let invalidLogitsValue: Float
    let softcap: Float
    let queryScale: Float

    @ModuleInfo(key: "relative_position_embedding") var relativePositionEmbedding:
        Gemma3nAudioRelativePositionEmbedding
    @ModuleInfo(key: "per_dim_scale") var perDimensionScale: MLXArray
    @ModuleInfo(key: "q_proj") var queryProjection: Linear
    @ModuleInfo(key: "k_proj") var keyProjection: Linear
    @ModuleInfo(key: "v_proj") var valueProjection: Linear

    init(config: Gemma3nConfiguration.AudioConfiguration) {
        heads = config.attentionHeads
        hiddenSize = config.hiddenSize
        headDim = config.hiddenSize / config.attentionHeads
        chunkSize = config.attentionChunkSize
        maxFutureHorizon = config.attentionContextRight
        maxPastHorizon = max(0, config.attentionContextLeft - 1)
        contextSize = chunkSize + maxPastHorizon + maxFutureHorizon
        invalidLogitsValue = config.attentionInvalidLogitsValue
        softcap = config.attentionLogitCap
        queryScale = pow(Float(headDim), -0.5) / log(2.0)
        _relativePositionEmbedding.wrappedValue = Gemma3nAudioRelativePositionEmbedding(
            config: config,
        )
        _perDimensionScale.wrappedValue = MLXArray.zeros([headDim])
        _queryProjection.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _keyProjection.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        _valueProjection.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: false)
        super.init()
    }

    private func padTime(_ x: MLXArray, left: Int, right: Int) -> MLXArray {
        var widths = Array(repeating: IntOrPair(0), count: x.ndim)
        widths[1] = .init((left, right))
        return padded(x, widths: widths)
    }

    private func convertToBlock(_ x: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let time = x.dim(1)
        let rest = Array(x.shape.dropFirst(2))
        let blocks = (time + chunkSize - 1) / chunkSize
        let padLength = blocks * chunkSize - time
        let paddedX = padLength > 0 ? padTime(x, left: 0, right: padLength) : x
        return paddedX.reshaped([batch, blocks, chunkSize] + rest)
    }

    private func extractBlockContext(_ x: MLXArray) -> MLXArray {
        let paddedX = padTime(
            x,
            left: maxPastHorizon,
            right: maxFutureHorizon + chunkSize - 1,
        )
        let paddedTime = paddedX.dim(1)
        let blocks = (paddedTime - contextSize) / chunkSize + 1
        let starts = arange(0, blocks, dtype: .int32) * Int32(chunkSize)
        let offsets = arange(0, contextSize, dtype: .int32)
        let indices = expandedDimensions(starts, axis: -1) + expandedDimensions(offsets, axis: 0)
        return take(paddedX, indices, axis: 1)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, causalValidMask: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let time = x.dim(1)
        let qkvShape = [batch, time, heads, headDim]

        var query = queryProjection(x).asType(.float32).reshaped(qkvShape)
        let key = keyProjection(x).asType(.float32).reshaped(qkvShape)
        let value = valueProjection(x).asType(.float32).reshaped(qkvShape)

        query = query * (queryScale * softplus(perDimensionScale).asType(.float32))

        let queryBlocks = convertToBlock(query)
        let keyBlocks = extractBlockContext(key)
        let valueBlocks = extractBlockContext(value)
        let blocks = queryBlocks.dim(1)

        let extractedValid = extractBlockContext(.!mask)
        let condition =
            expandedDimensions(expandedDimensions(extractedValid, axis: 1), axis: 3)
            .&& causalValidMask[.newAxis, .newAxis, .newAxis, 0..., 0...]

        var logits = relativePositionEmbedding(queries: queryBlocks, keys: keyBlocks)
        logits = tanh(logits / softcap) * softcap
        logits = MLX.where(condition, logits, MLXArray(invalidLogitsValue))

        let probabilities = softmax(logits, axis: -1).asType(valueBlocks.dtype)
        var context = einsum("bnuwc,bucnh->buwnh", probabilities, valueBlocks)
        context = context.reshaped(batch, blocks * chunkSize, heads, headDim)
        context = context[0..., ..<time, 0..., 0...]
        return context
    }
}

private final class Gemma3nAudioConformerAttention: Module {
    let gradientClipping: Float

    @ModuleInfo(key: "pre_attn_norm") var preAttentionNorm: Gemma3nAudioRMSNorm
    @ModuleInfo var attn: Gemma3nAudioAttention
    @ModuleInfo var post: Linear
    @ModuleInfo(key: "post_norm") var postNorm: Gemma3nAudioRMSNorm

    init(config: Gemma3nConfiguration.AudioConfiguration) {
        gradientClipping = config.gradientClipping
        _preAttentionNorm.wrappedValue = Gemma3nAudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        _attn.wrappedValue = Gemma3nAudioAttention(config: config)
        _post.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        _postNorm.wrappedValue = Gemma3nAudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, causalValidMask: MLXArray) -> MLXArray {
        let residual = x
        var hidden = clip(x, min: -gradientClipping, max: gradientClipping)
        hidden = preAttentionNorm(hidden)
        hidden = attn(hidden, mask: mask, causalValidMask: causalValidMask)
        let batch = hidden.dim(0)
        let time = hidden.dim(1)
        hidden = hidden.reshaped(batch, time, -1)
        hidden = post(hidden)
        hidden = clip(hidden, min: -gradientClipping, max: gradientClipping)
        return residual + postNorm(hidden)
    }
}

private final class Gemma3nAudioConformerLightConv1d: Module, UnaryLayer {
    let gradientClipping: Float
    let causalPadding: Int

    @ModuleInfo(key: "pre_layer_norm") var preLayerNorm: Gemma3nAudioRMSNorm
    @ModuleInfo(key: "linear_start") var linearStart: Linear
    @ModuleInfo(key: "depthwise_conv1d") var depthwiseConv1d: Conv1d
    @ModuleInfo(key: "conv_norm") var convNorm: Gemma3nAudioRMSNorm
    @ModuleInfo(key: "linear_end") var linearEnd: Linear

    init(config: Gemma3nConfiguration.AudioConfiguration) {
        gradientClipping = config.gradientClipping
        causalPadding = config.convKernelSize - 1
        _preLayerNorm.wrappedValue = Gemma3nAudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        _linearStart.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * 2, bias: false)
        _depthwiseConv1d.wrappedValue = Conv1d(
            inputChannels: config.hiddenSize,
            outputChannels: config.hiddenSize,
            kernelSize: config.convKernelSize,
            padding: 0,
            groups: config.hiddenSize,
            bias: false,
        )
        _convNorm.wrappedValue = Gemma3nAudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        _linearEnd.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x
        var hidden = preLayerNorm(x)
        hidden = linearStart(hidden)
        let parts = split(hidden, parts: 2, axis: -1)
        hidden = parts[0] * sigmoid(parts[1])
        hidden = padded(hidden, widths: [0, .init((causalPadding, 0)), 0])
        hidden = depthwiseConv1d(hidden)
        hidden = clip(hidden, min: -gradientClipping, max: gradientClipping)
        hidden = convNorm(hidden)
        hidden = silu(hidden)
        hidden = linearEnd(hidden)
        return hidden + residual
    }
}

private final class Gemma3nAudioConformerBlock: Module {
    let gradientClipping: Float

    @ModuleInfo(key: "ffw_layer_start") var feedForwardStart: Gemma3nAudioConformerFeedForward
    @ModuleInfo var attention: Gemma3nAudioConformerAttention
    @ModuleInfo var lconv1d: Gemma3nAudioConformerLightConv1d
    @ModuleInfo(key: "ffw_layer_end") var feedForwardEnd: Gemma3nAudioConformerFeedForward
    @ModuleInfo var norm: Gemma3nAudioRMSNorm

    init(config: Gemma3nConfiguration.AudioConfiguration) {
        gradientClipping = config.gradientClipping
        _feedForwardStart.wrappedValue = Gemma3nAudioConformerFeedForward(config: config)
        _attention.wrappedValue = Gemma3nAudioConformerAttention(config: config)
        _lconv1d.wrappedValue = Gemma3nAudioConformerLightConv1d(config: config)
        _feedForwardEnd.wrappedValue = Gemma3nAudioConformerFeedForward(config: config)
        _norm.wrappedValue = Gemma3nAudioRMSNorm(
            dimensions: config.hiddenSize,
            eps: config.rmsNormEps,
        )
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, causalValidMask: MLXArray) -> MLXArray {
        var hidden = feedForwardStart(x)
        hidden = attention(hidden, mask: mask, causalValidMask: causalValidMask)
        hidden = hidden * expandedDimensions((.!mask).asType(hidden.dtype), axis: -1)
        hidden = lconv1d(hidden)
        hidden = feedForwardEnd(hidden)
        hidden = clip(hidden, min: -gradientClipping, max: gradientClipping)
        return norm(hidden)
    }
}

private final class Gemma3nAudioModel: Module {
    let config: Gemma3nConfiguration.AudioConfiguration

    @ModuleInfo(key: "subsample_conv_projection") var subsampleConvProjection:
        Gemma3nAudioSubSampleConvProjection
    @ModuleInfo var conformer: [Gemma3nAudioConformerBlock]

    init(config: Gemma3nConfiguration.AudioConfiguration) {
        self.config = config
        _subsampleConvProjection.wrappedValue = Gemma3nAudioSubSampleConvProjection(
            config: config,
        )
        _conformer.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
            Gemma3nAudioConformerBlock(config: config)
        }
        super.init()
    }

    private func buildCausalValidMask() -> MLXArray {
        let chunkSize = config.attentionChunkSize
        let maxFutureHorizon = config.attentionContextRight
        let maxPastHorizon = max(0, config.attentionContextLeft - 1)
        let upperDiagonal = maxPastHorizon + maxFutureHorizon
        let contextSize = chunkSize + maxPastHorizon + maxFutureHorizon

        let lowerCausal = tril(ones([contextSize, chunkSize], dtype: .float32)).T
        let upperCausal = tril(ones([chunkSize, contextSize], dtype: .float32), k: upperDiagonal)
        return (lowerCausal * upperCausal).asType(.bool)
    }

    private func subsampleMask(_ mask: MLXArray, targetLength: Int) -> MLXArray {
        var strideProduct = 1
        for index in 0 ..< 2 {
            strideProduct *= config.strideSize(at: index).0
        }
        let indices = minimum(
            arange(0, targetLength, dtype: .int32) * Int32(strideProduct),
            MLXArray(Int32(max(mask.dim(1) - 1, 0))),
        )
        let broadcastIndices = broadcast(
            expandedDimensions(indices, axis: 0),
            to: [mask.dim(0), targetLength],
        )
        return takeAlong(mask, broadcastIndices, axis: 1)
    }

    func callAsFunction(_ audioMel: MLXArray, mask: MLXArray) -> (MLXArray, MLXArray) {
        var audioEncodings = subsampleConvProjection(audioMel)
        var currentMask = subsampleMask(mask, targetLength: audioEncodings.dim(1))
        let causalValidMask = buildCausalValidMask()

        var previousOutput: MLXArray?
        for block in conformer {
            audioEncodings = block(
                audioEncodings, mask: currentMask, causalValidMask: causalValidMask)
            // Bound inference intermediates while overlapping graph construction
            // with the preceding block. Training keeps the complete graph.
            if !training {
                asyncEval(audioEncodings)
                if let previousOutput { eval(previousOutput) }
                previousOutput = audioEncodings
            }
        }

        if config.reductionFactor > 1 {
            audioEncodings = audioEncodings[0..., stride(by: config.reductionFactor), 0...]
            currentMask = currentMask[0..., stride(by: config.reductionFactor)]
        }
        if currentMask.dim(1) > audioEncodings.dim(1) {
            currentMask = currentMask[0..., ..<audioEncodings.dim(1)]
        } else if currentMask.dim(1) < audioEncodings.dim(1) {
            currentMask = padded(
                currentMask,
                widths: [0, .init((0, audioEncodings.dim(1) - currentMask.dim(1)))],
            )
        }
        audioEncodings = MLX.where(
            expandedDimensions(currentMask, axis: -1),
            MLXArray(0.0),
            audioEncodings,
        )
        return (audioEncodings, currentMask)
    }
}

// MARK: - Model

private enum Gemma3nError: Error, LocalizedError {
    case missingEmbeddingLayer
    case imageTokenCountMismatch(expected: Int, actual: Int)
    case audioTokenCountMismatch(expected: Int, actual: Int)

    var errorDescription: String? {
        switch self {
        case .missingEmbeddingLayer:
            "Gemma3n text embedding layer was not found."
        case .imageTokenCountMismatch(let expected, let actual):
            "Gemma3n image features (\(expected)) do not match image token count (\(actual))."
        case .audioTokenCountMismatch(let expected, let actual):
            "Gemma3n audio features (\(expected)) do not match audio token count (\(actual))."
        }
    }
}

private func gemma3nMaskedScatter(inputTensor: MLXArray, mask: MLXArray, source: MLXArray)
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

private final class Gemma3nRotaryOffsetCache: KVCache {
    private var base: KVCache
    private var deferredOffset: Int?

    init(_ base: KVCache) {
        self.base = base
    }

    var offset: Int {
        if let deferredOffset {
            self.deferredOffset = nil
            return deferredOffset
        }
        return base.offset
    }

    var maxSize: Int? {
        base.maxSize
    }

    var isTrimmable: Bool {
        base.isTrimmable
    }

    func innerState() -> [MLXArray] {
        deferredOffset = nil
        return base.innerState()
    }

    func update(keys: MLXArray, values: MLXArray) -> (MLXArray, MLXArray) {
        let offsetBeforeUpdate = base.offset
        let result = base.update(keys: keys, values: values)
        deferredOffset = offsetBeforeUpdate
        return result
    }

    var state: [MLXArray] {
        get {
            deferredOffset = nil
            return base.state
        }
        set {
            deferredOffset = nil
            base.state = newValue
        }
    }

    var metaState: [String] {
        get {
            deferredOffset = nil
            return base.metaState
        }
        set {
            deferredOffset = nil
            base.metaState = newValue
        }
    }

    @discardableResult
    func trim(_ n: Int) -> Int {
        deferredOffset = nil
        return base.trim(n)
    }

    func makeMask(n: Int, windowSize: Int?, returnArray: Bool)
        -> MLXFast.ScaledDotProductAttentionMaskMode
    {
        deferredOffset = nil
        return base.makeMask(n: n, windowSize: windowSize, returnArray: returnArray)
    }

    func copy() -> any KVCache {
        Gemma3nRotaryOffsetCache(base.copy())
    }
}

private func gemma3nTextCache(_ cache: [KVCache]?) -> [KVCache]? {
    cache?.map { Gemma3nRotaryOffsetCache($0) as KVCache }
}

private func gemma3nDebugEnabled() -> Bool {
    ProcessInfo.processInfo.environment["NATIVSDK_GEMMA3N_DEBUG"] != nil
}

private func gemma3nDebugPrint(_ message: String) {
    guard gemma3nDebugEnabled() else {
        return
    }
    FileHandle.standardError.write(Data("Gemma3n.debug \(message)\n".utf8))
}

private func gemma3nDebugStats(_ name: String, _ value: MLXArray) {
    guard gemma3nDebugEnabled() else {
        return
    }
    let x = value.asType(.float32)
    let minValue = x.min().item(Float.self)
    let maxValue = x.max().item(Float.self)
    let meanValue = mean(x).item(Float.self)
    gemma3nDebugPrint(
        "\(name) shape=\(value.shape) min=\(minValue) max=\(maxValue) mean=\(meanValue)")
}

public final class Gemma3n: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "model") private var textModel: Gemma3nTextModel
    @ModuleInfo(key: "vision_tower") private var visionTower: Gemma3nVisionModel
    @ModuleInfo(key: "embed_vision") private var embedVision: Gemma3nMultimodalEmbedder
    @ModuleInfo(key: "audio_tower") private var audioTower: Gemma3nAudioModel
    @ModuleInfo(key: "embed_audio") private var embedAudio: Gemma3nMultimodalEmbedder

    public let config: Gemma3nConfiguration

    public init(_ config: Gemma3nConfiguration) {
        self.config = config
        _textModel.wrappedValue = Gemma3nTextModel(config: config.textModelConfiguration)
        _visionTower.wrappedValue = Gemma3nVisionModel(config.visionConfiguration)
        _embedVision.wrappedValue = Gemma3nMultimodalEmbedder(
            multimodalHiddenSize: config.visionConfiguration.hiddenSize,
            textHiddenSize: config.textConfiguration.hiddenSize,
            vocabSize: config.visionConfiguration.vocabSize,
            vocabOffset: config.visionConfiguration.vocabOffset,
            eps: config.visionConfiguration.rmsNormEps,
        )
        _audioTower.wrappedValue = Gemma3nAudioModel(config: config.audioConfiguration)
        _embedAudio.wrappedValue = Gemma3nMultimodalEmbedder(
            multimodalHiddenSize: config.audioConfiguration.hiddenSize,
            textHiddenSize: config.textConfiguration.hiddenSize,
            vocabSize: config.audioConfiguration.vocabSize,
            vocabOffset: config.audioConfiguration.vocabOffset,
            eps: config.audioConfiguration.rmsNormEps,
        )
        super.init()
    }

    public var kvHeads: [Int] {
        Array(
            repeating: config.textConfiguration.kvHeads,
            count: config.textConfiguration.hiddenLayers)
    }

    public var loraLayers: [Module] {
        textModel.loraLayers
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try textModel.newCache(configuration: configuration)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        guard input.image != nil || input.audio != nil else {
            let logits = textModel(
                input.text.tokens, cache: gemma3nTextCache(cache), logitsToKeep: 1)
            prefill.progress?(input.text.tokens.size, input.text.tokens.size)
            return .logits(LMOutput(logits: logits))
        }
        let inputIDs = input.text.tokens
        gemma3nDebugStats("input_ids", inputIDs)
        var inputsEmbeds = try tokenEmbeddings(inputIDs)
        gemma3nDebugStats("token_embeds", inputsEmbeds)

        let visionMask = logicalAnd(
            inputIDs .>= config.visionConfiguration.vocabOffset,
            inputIDs .< config.audioConfiguration.vocabOffset,
        )
        let dummyVisionID =
            config.visionConfiguration.vocabOffset + config.visionConfiguration.vocabSize - 1
        let visionTokens = MLX.where(visionMask, inputIDs, MLXArray(dummyVisionID))
        let hardVisionEmbeds = embedVision(inputIDs: visionTokens)
        gemma3nDebugStats("hard_vision_embeds", hardVisionEmbeds)
        inputsEmbeds = MLX.where(
            expandedDimensions(visionMask, axis: -1),
            hardVisionEmbeds.asType(inputsEmbeds.dtype),
            inputsEmbeds,
        )
        gemma3nDebugStats("token_embeds_after_hard_vision", inputsEmbeds)

        let audioMask = inputIDs .>= config.audioConfiguration.vocabOffset
        let dummyAudioID =
            config.audioConfiguration.vocabOffset + config.audioConfiguration.vocabSize - 1
        let audioTokens = MLX.where(audioMask, inputIDs, MLXArray(dummyAudioID))
        let hardAudioEmbeds = embedAudio(inputIDs: audioTokens)
        inputsEmbeds = MLX.where(
            expandedDimensions(audioMask, axis: -1),
            hardAudioEmbeds.asType(inputsEmbeds.dtype),
            inputsEmbeds,
        )
        gemma3nDebugStats("token_embeds_after_hard_audio", inputsEmbeds)

        guard let pixels = input.image?.pixels else {
            if let audio = input.audio {
                inputsEmbeds = try mergeAudioFeatures(
                    audio: audio,
                    inputIDs: inputIDs,
                    inputsEmbeds: inputsEmbeds,
                )
            }
            let logits = textModel(
                inputIDs,
                inputsEmbeds: inputsEmbeds,
                cache: gemma3nTextCache(cache),
                logitsToKeep: 1,
            )
            prefill.progress?(inputsEmbeds.dim(1), inputsEmbeds.dim(1))
            return .logits(LMOutput(logits: logits))
        }

        var imageFeatures = visionTower(pixels)
        gemma3nDebugStats("vision_tower", imageFeatures)
        imageFeatures = imageFeatures.transposed(0, 3, 1, 2)
            .reshaped(
                imageFeatures.dim(0),
                config.visionConfiguration.hiddenSize,
                config.visionSoftTokensPerImage,
            )
            .transposed(0, 2, 1)
        gemma3nDebugStats("vision_tokens", imageFeatures)
        imageFeatures = imageFeatures * pow(Float(config.visionConfiguration.hiddenSize), 0.5)
        gemma3nDebugStats("vision_tokens_scaled", imageFeatures)
        imageFeatures = embedVision(inputsEmbeds: imageFeatures).asType(inputsEmbeds.dtype)
        gemma3nDebugStats("image_features", imageFeatures)

        let imageMask = inputIDs .== config.imageTokenID
        let imageTokens = imageMask.asType(.int32).sum().item(Int.self)
        gemma3nDebugPrint("image_tokens=\(imageTokens)")
        if imageTokens != imageFeatures.dim(1) {
            throw Gemma3nError.imageTokenCountMismatch(
                expected: imageFeatures.dim(1), actual: imageTokens,
            )
        }
        let expandedMask = broadcast(
            expandedDimensions(imageMask, axis: -1), to: inputsEmbeds.shape)
        inputsEmbeds = gemma3nMaskedScatter(
            inputTensor: inputsEmbeds,
            mask: expandedMask,
            source: imageFeatures,
        )
        gemma3nDebugStats("inputs_embeds_merged", inputsEmbeds)

        if let audio = input.audio {
            inputsEmbeds = try mergeAudioFeatures(
                audio: audio,
                inputIDs: inputIDs,
                inputsEmbeds: inputsEmbeds,
            )
        }

        let logits = textModel(
            inputIDs,
            inputsEmbeds: inputsEmbeds,
            cache: gemma3nTextCache(cache),
            logitsToKeep: 1,
        )
        gemma3nDebugStats("logits", logits)
        if gemma3nDebugEnabled() {
            let nextLogits = logits[0..., -1, 0...].asType(.float32)
            let nextToken = argMax(nextLogits, axis: -1).item(Int.self)
            let nextValue = nextLogits.max().item(Float.self)
            gemma3nDebugPrint("next_argmax=\(nextToken) next_logit=\(nextValue)")
        }
        prefill.progress?(inputsEmbeds.dim(1), inputsEmbeds.dim(1))
        return .logits(LMOutput(logits: logits))
    }

    private func mergeAudioFeatures(
        audio: LMInput.ProcessedAudio,
        inputIDs: MLXArray,
        inputsEmbeds: MLXArray,
    )
        throws -> MLXArray
    {
        let validMask =
            if let mask = audio.mask {
                mask.asType(.bool)
            } else {
                ones([audio.features.dim(0), audio.features.dim(1)], dtype: .bool)
            }
        let (audioOutputs, invalidAudioMask) = audioTower(audio.features, mask: .!validMask)
        var audioFeatures = embedAudio(inputsEmbeds: audioOutputs).asType(inputsEmbeds.dtype)

        let paddingID = MLXArray(
            [config.audioConfiguration.vocabOffset + config.audioConfiguration.vocabSize - 1],
            [1, 1],
        )
        let paddingEmbedding = embedAudio(inputIDs: paddingID).asType(inputsEmbeds.dtype)
        audioFeatures = MLX.where(
            expandedDimensions(invalidAudioMask, axis: -1),
            paddingEmbedding,
            audioFeatures,
        )

        let extraPadding = config.audioSoftTokensPerImage - audioFeatures.dim(1)
        if extraPadding > 0 {
            let padding = broadcast(
                paddingEmbedding,
                to: [audioFeatures.dim(0), extraPadding, audioFeatures.dim(2)],
            )
            audioFeatures = concatenated([audioFeatures, padding], axis: 1)
        } else if extraPadding < 0 {
            audioFeatures = audioFeatures[0..., ..<config.audioSoftTokensPerImage, 0...]
        }

        let mask = inputIDs .== config.audioTokenID
        let audioTokens = mask.asType(.int32).sum().item(Int.self)
        let expected = audioFeatures.dim(0) * audioFeatures.dim(1)
        if audioTokens != expected {
            throw Gemma3nError.audioTokenCountMismatch(expected: expected, actual: audioTokens)
        }
        let expandedMask = broadcast(expandedDimensions(mask, axis: -1), to: inputsEmbeds.shape)
        let merged = gemma3nMaskedScatter(
            inputTensor: inputsEmbeds,
            mask: expandedMask,
            source: audioFeatures,
        )
        gemma3nDebugStats("inputs_embeds_merged_audio", merged)
        return merged
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        textModel(inputs, cache: gemma3nTextCache(cache))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Self.sanitizeWeights(weights)
    }

    static func sanitizeWeights(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        var skipVisionTranspose = false
        if let probe =
            weights["model.vision_tower.timm_model.blocks.0.0.conv_exp.weight"]
            ?? weights["vision_tower.timm_model.blocks.0.0.conv_exp.weight"],
            probe.ndim == 4
        {
            skipVisionTranspose = probe.dim(3) > probe.dim(1)
        }

        for (rawKey, var value) in weights {
            var key = rawKey
            if key.hasPrefix("model.vision_tower.") {
                key = key.replacingOccurrences(of: "model.vision_tower.", with: "vision_tower.")
            } else if key.hasPrefix("model.embed_vision.") {
                key = key.replacingOccurrences(of: "model.embed_vision.", with: "embed_vision.")
            } else if key.hasPrefix("model.audio_tower.") {
                key = key.replacingOccurrences(of: "model.audio_tower.", with: "audio_tower.")
            } else if key.hasPrefix("model.embed_audio.") {
                key = key.replacingOccurrences(of: "model.embed_audio.", with: "embed_audio.")
            } else if key.hasPrefix("model.language_model.model.") {
                key = key.replacingOccurrences(
                    of: "model.language_model.model.",
                    with: "model.language_model.",
                )
            } else if key.hasPrefix("model.language_model.") {
                // The text model is mounted at key "model", so keep this prefix.
            } else if key.hasPrefix("language_model.model.") {
                key = key.replacingOccurrences(
                    of: "language_model.model.",
                    with: "model.language_model.",
                )
            } else if key.hasPrefix("language_model.") {
                key = key.replacingOccurrences(
                    of: "language_model.",
                    with: "model.language_model.",
                )
            } else if key.hasPrefix("model.") {
                continue
            }

            if key.hasPrefix("vision_tower."),
                value.ndim == 4,
                !skipVisionTranspose,
                key.contains("conv") && key.hasSuffix(".weight")
                    || key.hasSuffix("proj.weight")
            {
                value = value.transposed(0, 2, 3, 1)
            }
            if key.hasPrefix("audio_tower."),
                key.hasSuffix(".conv.weight"),
                value.ndim == 4,
                value.dim(1) != value.dim(2)
            {
                value = value.transposed(0, 2, 3, 1)
            }
            if key.hasPrefix("audio_tower."),
                key.hasSuffix("conv1d.weight"),
                value.ndim == 3,
                value.dim(-1) != 1
            {
                value = value.transposed(0, 2, 1)
            }
            sanitized[key] = value
        }

        return sanitized
    }

    private func tokenEmbeddings(_ inputIDs: MLXArray) throws -> MLXArray {
        let candidates = textModel.leafModules().flattened()
        guard
            let embedding = candidates.first(where: { $0.0 == "language_model.embed_tokens" })?.1
                as? any UnaryLayer
        else {
            throw Gemma3nError.missingEmbeddingLayer
        }
        return embedding(inputIDs)
    }
}
