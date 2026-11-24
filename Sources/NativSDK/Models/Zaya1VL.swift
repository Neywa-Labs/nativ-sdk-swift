import CoreImage
import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct Zaya1VLConfiguration: Decodable, Sendable {
    public struct TextConfiguration: Decodable, Sendable {
        public let modelType: String
        public let cca: Bool
        public let numQueryGroups: Int
        public let useCache: Bool
        public let attentionBias: Bool
        public let lmHeadBias: Bool
        public let vocabularySize: Int
        public let hiddenSize: Int
        public let ffnHiddenSize: Int
        public let hiddenLayers: Int
        public let numExperts: Int
        public let attentionHeads: Int
        public let headDim: Int
        public let activationFunction: String
        public let maxPositionEmbeddings: Int
        public let normEpsilon: Float
        public let tieWordEmbeddings: Bool
        public let ropeTheta: Float
        public let attentionDropout: Float
        public let moeRouterTopK: Int
        public let zayaMLPExpansion: Int
        public let zayaUseMod: Bool
        public let zayaUseEDA: Bool
        public let addBiasLinear: Bool
        public let gatedLinearUnit: Bool
        public let scaleResidualMerge: Bool
        public let partialRotaryFactor: Float
        public let kvHeads: Int
        public let ccaTime0: Int
        public let ccaTime1: Int
        public let visionLora: Bool
        public let visionLoraRankAttention: Int
        public let visionLoraRankMLP: Int

        public var ffnHiddenSizeOut: Int {
            gatedLinearUnit ? ffnHiddenSize / 2 : ffnHiddenSize
        }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case cca
            case numQueryGroups = "num_query_groups"
            case useCache = "use_cache"
            case attentionBias = "attention_bias"
            case lmHeadBias = "lm_head_bias"
            case vocabularySize = "vocab_size"
            case hiddenSize = "hidden_size"
            case ffnHiddenSize = "ffn_hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case numExperts = "num_experts"
            case attentionHeads = "num_attention_heads"
            case headDim = "head_dim"
            case activationFunction = "activation_func"
            case maxPositionEmbeddings = "max_position_embeddings"
            case normEpsilon = "norm_epsilon"
            case rmsNormEps = "rms_norm_eps"
            case tieWordEmbeddings = "tie_word_embeddings"
            case ropeTheta = "rope_theta"
            case rotaryBase = "rotary_base"
            case attentionDropout = "attention_dropout"
            case moeRouterTopK = "moe_router_topk"
            case zayaMLPExpansion = "zaya_mlp_expansion"
            case zayaUseMod = "zaya_use_mod"
            case zayaUseEDA = "zaya_use_eda"
            case addBiasLinear = "add_bias_linear"
            case gatedLinearUnit = "gated_linear_unit"
            case scaleResidualMerge = "scale_residual_merge"
            case partialRotaryFactor = "partial_rotary_factor"
            case ropePct = "rope_pct"
            case kvHeads = "num_key_value_heads"
            case ccaTime0 = "cca_time0"
            case ccaTime1 = "cca_time1"
            case visionLora = "vision_lora"
            case visionLoraRankAttention = "vision_lora_rank_attn"
            case visionLoraRankMLP = "vision_lora_rank_mlp"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "zaya1_vl"
            cca = try c.decodeIfPresent(Bool.self, forKey: .cca) ?? true
            numQueryGroups = try c.decodeIfPresent(Int.self, forKey: .numQueryGroups) ?? 2
            useCache = try c.decodeIfPresent(Bool.self, forKey: .useCache) ?? true
            attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            lmHeadBias = try c.decodeIfPresent(Bool.self, forKey: .lmHeadBias) ?? false
            vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262_272
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
            ffnHiddenSize = try c.decodeIfPresent(Int.self, forKey: .ffnHiddenSize) ?? 4096
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 40
            numExperts = try c.decodeIfPresent(Int.self, forKey: .numExperts) ?? 16
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 8
            headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 128
            activationFunction =
                try c.decodeIfPresent(String.self, forKey: .activationFunction) ?? "swiglu"
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 32768
            normEpsilon =
                try c.decodeIfPresent(Float.self, forKey: .normEpsilon)
                ?? c.decodeIfPresent(Float.self, forKey: .rmsNormEps)
                ?? 1e-5
            tieWordEmbeddings =
                try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
            ropeTheta =
                try c.decodeIfPresent(Float.self, forKey: .rotaryBase)
                ?? c.decodeIfPresent(Float.self, forKey: .ropeTheta)
                ?? 1_000_000
            attentionDropout =
                try c.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0
            moeRouterTopK = try c.decodeIfPresent(Int.self, forKey: .moeRouterTopK) ?? 1
            zayaMLPExpansion =
                try c.decodeIfPresent(Int.self, forKey: .zayaMLPExpansion) ?? 256
            zayaUseMod = try c.decodeIfPresent(Bool.self, forKey: .zayaUseMod) ?? true
            zayaUseEDA = try c.decodeIfPresent(Bool.self, forKey: .zayaUseEDA) ?? true
            addBiasLinear = try c.decodeIfPresent(Bool.self, forKey: .addBiasLinear) ?? false
            gatedLinearUnit = try c.decodeIfPresent(Bool.self, forKey: .gatedLinearUnit) ?? true
            scaleResidualMerge =
                try c.decodeIfPresent(Bool.self, forKey: .scaleResidualMerge) ?? true
            partialRotaryFactor =
                try c.decodeIfPresent(Float.self, forKey: .ropePct)
                ?? c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
                ?? 0.5
            kvHeads =
                try c.decodeIfPresent(Int.self, forKey: .kvHeads)
                ?? numQueryGroups
            ccaTime0 = try c.decodeIfPresent(Int.self, forKey: .ccaTime0) ?? 2
            ccaTime1 = try c.decodeIfPresent(Int.self, forKey: .ccaTime1) ?? 2
            visionLora = try c.decodeIfPresent(Bool.self, forKey: .visionLora) ?? true
            visionLoraRankAttention =
                try c.decodeIfPresent(Int.self, forKey: .visionLoraRankAttention) ?? 8
            visionLoraRankMLP =
                try c.decodeIfPresent(Int.self, forKey: .visionLoraRankMLP) ?? 32
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
        public let inChannels: Int
        public let layerNormEps: Float
        public let spatialMergeSize: Int
        public let temporalPatchSize: Int
        public let windowSize: Int
        public let fullAttentionBlockIndexes: [Int]
        public let tokensPerSecond: Int

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case depth
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case outHiddenSize = "out_hidden_size"
            case numHeads = "num_heads"
            case patchSize = "patch_size"
            case spatialPatchSize = "spatial_patch_size"
            case inChannels = "in_channels"
            case inChans = "in_chans"
            case layerNormEps = "layer_norm_eps"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
            case windowSize = "window_size"
            case fullAttentionBlockIndexes = "fullatt_block_indexes"
            case tokensPerSecond = "tokens_per_second"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen2_5_vl"
            depth = try c.decodeIfPresent(Int.self, forKey: .depth) ?? 32
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1280
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 3420
            outHiddenSize = try c.decodeIfPresent(Int.self, forKey: .outHiddenSize) ?? 1536
            numHeads = try c.decodeIfPresent(Int.self, forKey: .numHeads) ?? 16
            patchSize =
                try c.decodeIfPresent(Int.self, forKey: .patchSize)
                ?? c.decodeIfPresent(Int.self, forKey: .spatialPatchSize)
                ?? 14
            inChannels =
                try c.decodeIfPresent(Int.self, forKey: .inChannels)
                ?? c.decodeIfPresent(Int.self, forKey: .inChans)
                ?? 3
            layerNormEps = try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
            spatialMergeSize =
                try c.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? 2
            temporalPatchSize =
                try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 1
            windowSize = try c.decodeIfPresent(Int.self, forKey: .windowSize) ?? 112
            fullAttentionBlockIndexes =
                try c.decodeIfPresent([Int].self, forKey: .fullAttentionBlockIndexes)
                ?? [7, 15, 23, 31]
            tokensPerSecond = try c.decodeIfPresent(Int.self, forKey: .tokensPerSecond) ?? 2
        }
    }

    public let modelType: String
    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let imageTokenID: Int
    public let visionStartTokenID: Int
    public let visionEndTokenID: Int
    public let vocabularySize: Int
    public let eosTokenID: Int
    public let padTokenID: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case visionConfiguration = "vision_config"
        case imageTokenID = "image_token_id"
        case visionStartTokenID = "vision_start_token_id"
        case visionEndTokenID = "vision_end_token_id"
        case vocabularySize = "vocab_size"
        case eosTokenID = "eos_token_id"
        case padTokenID = "pad_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "zaya1_vl"
        textConfiguration = try TextConfiguration(from: decoder)
        visionConfiguration = try c.decode(VisionConfiguration.self, forKey: .visionConfiguration)
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? 262_147
        visionStartTokenID =
            try c.decodeIfPresent(Int.self, forKey: .visionStartTokenID) ?? 255_999
        visionEndTokenID =
            try c.decodeIfPresent(Int.self, forKey: .visionEndTokenID) ?? 256_000
        vocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .vocabularySize)
            ?? textConfiguration.vocabularySize
        eosTokenID = try c.decodeIfPresent(Int.self, forKey: .eosTokenID) ?? 262_143
        padTokenID = try c.decodeIfPresent(Int.self, forKey: .padTokenID) ?? 0
    }
}

public struct Zaya1VLProcessorConfiguration: Decodable, Sendable {
    public struct Size: Decodable, Sendable {
        public let longestEdge: Int?
        public let shortestEdge: Int?

        enum CodingKeys: String, CodingKey {
            case longestEdge = "longest_edge"
            case shortestEdge = "shortest_edge"
        }
    }

    public let processorClass: String
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let minPixels: Int
    public let maxPixels: Int
    public let mergeSize: Int
    public let patchSize: Int
    public let temporalPatchSize: Int
    public let rescaleFactor: CGFloat
    public let imageToken: String

    enum CodingKeys: String, CodingKey {
        case processorClass = "processor_class"
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case minPixels = "min_pixels"
        case maxPixels = "max_pixels"
        case mergeSize = "merge_size"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case rescaleFactor = "rescale_factor"
        case imageToken = "image_token"
        case size
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let size = try c.decodeIfPresent(Size.self, forKey: .size)
        processorClass =
            try c.decodeIfPresent(String.self, forKey: .processorClass) ?? "Zaya1VLProcessor"
        imageMean =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageMean)
            ?? [0.48145466, 0.4578275, 0.40821073]
        imageStd =
            try c.decodeIfPresent([CGFloat].self, forKey: .imageStd)
            ?? [0.26862954, 0.26130258, 0.27577711]
        minPixels =
            try c.decodeIfPresent(Int.self, forKey: .minPixels)
            ?? size?.shortestEdge
            ?? 3136
        maxPixels =
            try c.decodeIfPresent(Int.self, forKey: .maxPixels)
            ?? size?.longestEdge
            ?? 12_845_056
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
        temporalPatchSize = try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 1
        rescaleFactor =
            try c.decodeIfPresent(CGFloat.self, forKey: .rescaleFactor) ?? (1.0 / 255.0)
        imageToken = try c.decodeIfPresent(String.self, forKey: .imageToken) ?? "<image>"
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        (imageMean[0], imageMean[1], imageMean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        (imageStd[0], imageStd[1], imageStd[2])
    }
}

// MARK: - Processor

public struct Zaya1VLProcessor: UserInputProcessor {
    private let config: Zaya1VLProcessorConfiguration
    private let tokenizer: any Tokenizer

    public init(_ config: Zaya1VLProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func preprocess(images: [CIImage], processing: UserInput.Processing?) throws -> (
        MLXArray, THW,
    ) {
        let images = images.map { MediaProcessing.apply($0, processing: processing) }
        guard let first = images.first else {
            throw VLMError.imageProcessingFailure("No image provided")
        }

        let size = first.extent.size
        let factor = config.patchSize * config.mergeSize
        let (height, width) = try Zaya1VLImageProcessing.targetSize(
            height: Int(size.height),
            width: Int(size.width),
            factor: factor,
            minPixels: config.minPixels,
            maxPixels: config.maxPixels,
        )
        let target = CGSize(width: width, height: height)
        let processed =
            images
            .map { MediaProcessing.inSRGBToneCurveSpace($0) }
            .map { MediaProcessing.resampleBicubic($0, to: target) }
            .map {
                MediaProcessing.normalize(
                    $0,
                    mean: config.imageMeanTuple,
                    std: config.imageStdTuple,
                )
            }
            .map { MediaProcessing.asMLXArray($0) }

        return try QwenVL.patchify(
            images: processed,
            mergeSize: config.mergeSize,
            patchSize: config.patchSize,
            temporalPatchSize: config.temporalPatchSize,
        )
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        var prompt = Zaya1VLMessageGenerator().render(prompt: input.prompt)

        if input.images.isEmpty, input.videos.isEmpty {
            let tokens = tokenizer.encode(text: prompt)
            let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
            return LMInput(
                text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)))
        }

        var processedImage: LMInput.ProcessedImage?
        if !input.images.isEmpty {
            let imageFrames = try input.images.map {
                try preprocess(images: [$0.asCIImage()], processing: input.processing)
            }
            let pixels = concatenated(imageFrames.map(\.0))
            let frames = imageFrames.map(\.1)
            processedImage = .init(pixels: pixels, frames: frames)
            prompt = try expandImagePlaceholders(in: prompt, frames: frames)
        }

        if !input.videos.isEmpty {
            throw VLMError.processing("Zaya1-VL video preprocessing is not implemented")
        }

        let tokens = tokenizer.encode(text: prompt)
        let tokenArray = MLXArray(tokens).expandedDimensions(axis: 0)
        return LMInput(
            text: .init(tokens: tokenArray, mask: ones(like: tokenArray).asType(.int8)),
            image: processedImage,
        )
    }

    private func expandImagePlaceholders(in prompt: String, frames: [THW]) throws -> String {
        var result = prompt
        for frame in frames {
            guard let range = result.range(of: config.imageToken) else {
                throw VLMError.processing("Zaya1-VL image placeholder count does not match images")
            }
            let count = frame.product / (config.mergeSize * config.mergeSize)
            result.replaceSubrange(
                range,
                with: String(repeating: config.imageToken, count: count),
            )
        }
        return result
    }
}

public struct Zaya1VLMessageGenerator {
    public init() {}

    public func render(prompt: UserInput.Prompt) -> String {
        switch prompt {
        case .chat(let chat):
            return render(chat: chat)
        case .text(let text):
            return render(chat: [.user(text)])
        case .messages(let messages):
            let chat = messages.compactMap { message -> Chat.Message? in
                guard let role = message["role"] as? String,
                    let content = message["content"] as? String
                else {
                    return nil
                }
                return Chat.Message(
                    role: Chat.Message.Role(rawValue: role) ?? .user,
                    content: content,
                )
            }
            return chat.isEmpty ? render(chat: [.user(prompt.description)]) : render(chat: chat)
        }
    }

    public func render(chat: [Chat.Message]) -> String {
        var prompt = ""
        for message in chat {
            switch message.role {
            case .user:
                for _ in message.images {
                    prompt += "<|vision_start|><image><|vision_end|>\n"
                }
                prompt += "<|im_start|>user\n\(message.content)<|im_end|>\n"
            case .assistant:
                prompt += "<|im_start|>assistant\n\(message.content)<|im_end|>"
            case .system:
                prompt += "<|im_start|>system\n\(message.content)<|im_end|>\n"
            default:
                prompt += "<|im_start|>\(message.role.rawValue)\n\(message.content)<|im_end|>\n"
            }
        }
        prompt += "<|im_start|>assistant\n"
        return prompt
    }
}

private enum Zaya1VLImageProcessing {
    static func targetSize(
        height: Int,
        width: Int,
        factor: Int,
        minPixels: Int,
        maxPixels: Int,
    )
        throws -> (height: Int, width: Int)
    {
        if height < factor || width < factor {
            throw VLMError.imageProcessingFailure(
                "Image dimensions \(width)x\(height) must be at least \(factor)",
            )
        }
        if max(height, width) / min(height, width) > 200 {
            throw VLMError.imageProcessingFailure(
                "Absolute aspect ratio must be smaller than 200: \(width)x\(height)",
            )
        }

        var resizedHeight = max(factor, Int(round(Float(height) / Float(factor))) * factor)
        var resizedWidth = max(factor, Int(round(Float(width) / Float(factor))) * factor)
        if resizedHeight * resizedWidth > maxPixels {
            let beta = sqrt(Float(height * width) / Float(maxPixels))
            resizedHeight = Int(floor(Float(height) / beta / Float(factor))) * factor
            resizedWidth = Int(floor(Float(width) / beta / Float(factor))) * factor
        } else if resizedHeight * resizedWidth < minPixels {
            let beta = sqrt(Float(minPixels) / Float(height * width))
            resizedHeight = Int(ceil(Float(height) * beta / Float(factor))) * factor
            resizedWidth = Int(ceil(Float(width) * beta / Float(factor))) * factor
        }
        resizedHeight = max(factor, (resizedHeight / factor) * factor)
        resizedWidth = max(factor, (resizedWidth / factor) * factor)
        return (resizedHeight, resizedWidth)
    }

}

// MARK: - Language

private enum Zaya1VLLanguage {
    static func splitCache(_ cache: KVCache?) -> (KVCache?, ArraysCache?) {
        guard let cache else {
            return (nil, nil)
        }
        if let cacheList = cache as? CacheList {
            return (cacheList[0], cacheList[1] as? ArraysCache)
        }
        return (cache, nil)
    }

    static func repeatKV(_ x: MLXArray, repeats: Int) -> MLXArray {
        guard repeats > 1 else {
            return x
        }
        let expanded = expandedDimensions(x, axis: 2)
        let tiledValue = tiled(expanded, repetitions: [1, 1, repeats, 1, 1])
        return tiledValue.reshaped(x.dim(0), x.dim(1) * repeats, x.dim(2), x.dim(3))
    }

    final class ResidualScaling: Module {
        let isFirst: Bool
        @ParameterInfo(key: "hidden_states_scale") var hiddenStatesScale: MLXArray
        @ParameterInfo(key: "hidden_states_bias") var hiddenStatesBias: MLXArray
        @ParameterInfo(key: "residual_scale") var residualScale: MLXArray?
        @ParameterInfo(key: "residual_bias") var residualBias: MLXArray?

        init(config: Zaya1VLConfiguration.TextConfiguration, index: Int) {
            isFirst = index == 0
            _hiddenStatesScale.wrappedValue = MLXArray.ones([config.hiddenSize])
            _hiddenStatesBias.wrappedValue = MLXArray.zeros([config.hiddenSize])
            if !isFirst {
                _residualScale.wrappedValue = MLXArray.ones([config.hiddenSize])
                _residualBias.wrappedValue = MLXArray.zeros([config.hiddenSize])
            }
        }

        func callAsFunction(
            residual: MLXArray?,
            hiddenStates: MLXArray,
        )
            -> (MLXArray?, MLXArray)
        {
            let scaledHidden = (hiddenStates + hiddenStatesBias) * hiddenStatesScale
            guard !isFirst, let residual, let residualScale, let residualBias else {
                return (residual, scaledHidden)
            }
            return ((residual + residualBias) * residualScale, scaledHidden)
        }
    }

    final class CCA: Module {
        let config: Zaya1VLConfiguration.TextConfiguration
        let totalPadding: Int
        let numKVHeads: Int
        let numQHeads: Int
        let headDim: Int
        let latentKDim: Int
        let latentQDim: Int
        let gqaGroups: Int
        let sqrtHeadDim: Float

        @ModuleInfo(key: "linear_q") var linearQ: Linear
        @ModuleInfo(key: "linear_k") var linearK: Linear
        @ModuleInfo(key: "val_proj1") var valueProjection1: Linear
        @ModuleInfo(key: "val_proj2") var valueProjection2: Linear
        @ModuleInfo(key: "lora_linear_q") var loraLinearQ: LoRALinearPair?
        @ModuleInfo(key: "lora_linear_k") var loraLinearK: LoRALinearPair?
        @ModuleInfo(key: "lora_val_proj1") var loraValueProjection1: LoRALinearPair?
        @ModuleInfo(key: "lora_val_proj2") var loraValueProjection2: LoRALinearPair?
        @ModuleInfo(key: "conv_qk") var convQK: ConvStack
        @ParameterInfo(key: "temp") var temp: MLXArray

        init(_ config: Zaya1VLConfiguration.TextConfiguration) {
            self.config = config
            totalPadding = config.ccaTime0 + config.ccaTime1 - 2
            numKVHeads = config.kvHeads
            numQHeads = config.attentionHeads
            headDim = config.headDim
            latentKDim = numKVHeads * headDim
            latentQDim = numQHeads * headDim
            gqaGroups = numQHeads / numKVHeads
            sqrtHeadDim = sqrt(Float(headDim))
            let inOutChannels = latentKDim + latentQDim

            _linearQ.wrappedValue = Linear(
                config.hiddenSize,
                latentQDim,
                bias: config.attentionBias,
            )
            _linearK.wrappedValue = Linear(
                config.hiddenSize,
                latentKDim,
                bias: config.attentionBias,
            )
            _valueProjection1.wrappedValue = Linear(
                config.hiddenSize,
                latentKDim / 2,
                bias: config.attentionBias,
            )
            _valueProjection2.wrappedValue = Linear(
                config.hiddenSize,
                latentKDim / 2,
                bias: config.attentionBias,
            )
            if config.visionLora {
                _loraLinearQ.wrappedValue = LoRALinearPair(
                    input: config.hiddenSize,
                    rank: config.visionLoraRankAttention,
                    output: latentQDim,
                )
                _loraLinearK.wrappedValue = LoRALinearPair(
                    input: config.hiddenSize,
                    rank: config.visionLoraRankAttention,
                    output: latentKDim,
                )
                _loraValueProjection1.wrappedValue = LoRALinearPair(
                    input: config.hiddenSize,
                    rank: config.visionLoraRankAttention,
                    output: latentKDim / 2,
                )
                _loraValueProjection2.wrappedValue = LoRALinearPair(
                    input: config.hiddenSize,
                    rank: config.visionLoraRankAttention,
                    output: latentKDim / 2,
                )
            }
            _convQK.wrappedValue = ConvStack(
                channels: inOutChannels,
                firstKernel: config.ccaTime0,
                secondKernel: config.ccaTime1,
                groups: numKVHeads + numQHeads,
            )
            _temp.wrappedValue = MLXArray.zeros([numKVHeads])
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            cache: KVCache?,
            ccaMask: MLXArray?,
            imageMask: MLXArray?,
        )
            -> (MLXArray, MLXArray, MLXArray)
        {
            let (kvCache, auxCache) = Zaya1VLLanguage.splitCache(cache)
            var hiddenStates = hiddenStates
            if let ccaMask, hiddenStates.dim(1) > 1 {
                hiddenStates =
                    hiddenStates * expandedDimensions(ccaMask.asType(hiddenStates.dtype), axis: -1)
            }

            var q = linearQ(hiddenStates)
            var k = linearK(hiddenStates)
            let loraMask = imageMask.map { expandedDimensions($0.asType(q.dtype), axis: -1) }
            if let loraMask {
                if let loraLinearQ {
                    q = q + loraLinearQ(hiddenStates) * loraMask
                }
                if let loraLinearK {
                    k = k + loraLinearK(hiddenStates) * loraMask
                }
            }

            let queryPre = q.reshaped(
                q.dim(0), q.dim(1), numQHeads, headDim,
            )
            let keyPre = k.reshaped(
                k.dim(0), k.dim(1), numKVHeads, headDim,
            )
            let keyRepeated = Zaya1VLLanguage.repeatKV(
                keyPre.transposed(0, 2, 1, 3), repeats: gqaGroups
            )
            .transposed(0, 2, 1, 3)
            let queryKeyMeanQ = (queryPre + keyRepeated) / 2
            let queryKeyMeanK =
                queryKeyMeanQ
                .reshaped(q.dim(0), q.dim(1), numKVHeads, gqaGroups, headDim)
                .mean(axis: 3)

            let packed = concatenated([q, k], axis: -1)
            let (convolved, nextConvState) = convQK(
                packed,
                state: auxCache?[0],
                stateSize: totalPadding,
                useState: auxCache?[0] != nil && (kvCache?.offset ?? 0) > 0,
            )
            if let auxCache {
                auxCache[0] = nextConvState
            }

            let query =
                convolved[.ellipsis, ..<latentQDim]
                .reshaped(convolved.dim(0), convolved.dim(1), numQHeads, headDim)
                + queryKeyMeanQ
            let key =
                convolved[.ellipsis, latentQDim...]
                .reshaped(convolved.dim(0), convolved.dim(1), numKVHeads, headDim)
                + queryKeyMeanK

            var value1 = valueProjection1(hiddenStates)
            if let loraMask, let loraValueProjection1 {
                value1 = value1 + loraValueProjection1(hiddenStates) * loraMask
            }

            let shiftedHidden: MLXArray
            if (kvCache?.offset ?? 0) > 0, let previous = auxCache?[1] {
                let previous = expandedDimensions(previous, axis: 1)
                shiftedHidden = concatenated(
                    [previous, hiddenStates[0..., ..<max(0, hiddenStates.dim(1) - 1), 0...]],
                    axis: 1,
                )
            } else if hiddenStates.dim(1) > 1 {
                let zeros = MLXArray.zeros(
                    [hiddenStates.dim(0), 1, hiddenStates.dim(2)], dtype: hiddenStates.dtype)
                shiftedHidden = concatenated(
                    [zeros, hiddenStates[0..., ..<(hiddenStates.dim(1) - 1), 0...]], axis: 1)
            } else {
                shiftedHidden = MLXArray.zeros(hiddenStates.shape, dtype: hiddenStates.dtype)
            }

            var value2 = valueProjection2(shiftedHidden)
            if let loraMask, let loraValueProjection2 {
                value2 = value2 + loraValueProjection2(shiftedHidden) * loraMask
            }

            var normalizedQuery = query
            var normalizedKey = key
            let eps = MLXArray(1e-7, dtype: query.dtype)
            let queryNorm = maximum(sqrt((query * query).sum(axis: -1, keepDims: true)), eps)
            let keyNorm = maximum(sqrt((key * key).sum(axis: -1, keepDims: true)), eps)
            normalizedQuery = normalizedQuery * (sqrtHeadDim / queryNorm)
            normalizedKey =
                normalizedKey * (sqrtHeadDim / keyNorm) * temp[.newAxis, .newAxis, 0..., .newAxis]

            let value = concatenated([value1, value2], axis: -1)
                .reshaped(hiddenStates.dim(0), hiddenStates.dim(1), numKVHeads, headDim)
            if let auxCache {
                auxCache[1] = hiddenStates[0..., -1, 0...]
            }
            return (
                normalizedQuery.reshaped(
                    hiddenStates.dim(0), hiddenStates.dim(1), numQHeads * headDim),
                normalizedKey.reshaped(
                    hiddenStates.dim(0), hiddenStates.dim(1), numKVHeads * headDim),
                value.reshaped(hiddenStates.dim(0), hiddenStates.dim(1), numKVHeads * headDim),
            )
        }
    }

    final class LoRALinearPair: Module, UnaryLayer {
        @ModuleInfo(key: "0") var down: Linear
        @ModuleInfo(key: "1") var up: Linear

        init(input: Int, rank: Int, output: Int) {
            _down.wrappedValue = Linear(input, rank, bias: false)
            _up.wrappedValue = Linear(rank, output, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            up(down(x))
        }
    }

    final class ConvStack: Module {
        @ModuleInfo(key: "0") var first: Conv1d
        @ModuleInfo(key: "1") var second: Conv1d

        init(channels: Int, firstKernel: Int, secondKernel: Int, groups: Int) {
            _first.wrappedValue = Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: firstKernel,
                groups: channels,
                bias: true,
            )
            _second.wrappedValue = Conv1d(
                inputChannels: channels,
                outputChannels: channels,
                kernelSize: secondKernel,
                groups: groups,
                bias: true,
            )
        }

        func callAsFunction(
            _ x: MLXArray,
            state: MLXArray?,
            stateSize: Int,
            useState: Bool,
        )
            -> (MLXArray, MLXArray)
        {
            let convInput: MLXArray
            let stateSource: MLXArray
            if useState, let state {
                convInput = concatenated([state, x], axis: 1)
                stateSource = convInput
            } else {
                let pad = MLXArray.zeros([x.dim(0), stateSize, x.dim(2)], dtype: x.dtype)
                convInput = concatenated([pad, x], axis: 1)
                stateSource = x
            }
            let output = second(first(convInput))
            let nextState: MLXArray
            if stateSize == 0 {
                nextState = MLXArray.zeros([x.dim(0), 0, x.dim(2)], dtype: x.dtype)
            } else if stateSource.dim(1) < stateSize {
                let pad = MLXArray.zeros(
                    [x.dim(0), stateSize - stateSource.dim(1), x.dim(2)],
                    dtype: x.dtype,
                )
                nextState = concatenated([pad, stateSource], axis: 1)
            } else {
                nextState = contiguous(stateSource[0..., (stateSource.dim(1) - stateSize)..., 0...])
            }
            return (output, nextState)
        }
    }

    final class Attention: Module {
        let config: Zaya1VLConfiguration.TextConfiguration
        let numAttentionHeads: Int
        let numKeyValueHeads: Int
        let headDim: Int
        let scale: Float
        @ModuleInfo(key: "o_proj") var outputProjection: Linear
        @ModuleInfo(key: "qkv") var qkv: CCA
        @ModuleInfo(key: "lora_linear_o") var loraLinearOutput: LoRALinearPair?
        @ModuleInfo(key: "rotary_emb") var rope: RoPE

        init(_ config: Zaya1VLConfiguration.TextConfiguration) {
            self.config = config
            numAttentionHeads = config.attentionHeads
            numKeyValueHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(headDim), -0.5)
            _outputProjection.wrappedValue = Linear(
                config.attentionHeads * config.headDim,
                config.hiddenSize,
                bias: config.attentionBias,
            )
            _qkv.wrappedValue = CCA(config)
            if config.visionLora {
                _loraLinearOutput.wrappedValue = LoRALinearPair(
                    input: config.attentionHeads * config.headDim,
                    rank: config.visionLoraRankAttention,
                    output: config.hiddenSize,
                )
            }
            _rope.wrappedValue = RoPE(
                dimensions: Int(Float(config.headDim) * config.partialRotaryFactor),
                traditional: false,
                base: config.ropeTheta,
            )
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            ccaMask: MLXArray?,
            imageMask: MLXArray?,
            cache: KVCache?,
        )
            -> MLXArray
        {
            let batch = hiddenStates.dim(0)
            let length = hiddenStates.dim(1)
            let (kvCache, _) = Zaya1VLLanguage.splitCache(cache)
            var (q, k, v) = qkv(
                hiddenStates,
                cache: cache,
                ccaMask: ccaMask,
                imageMask: imageMask,
            )

            q = q.reshaped(batch, length, numAttentionHeads, headDim).transposed(0, 2, 1, 3)
            k = k.reshaped(batch, length, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)
            v = v.reshaped(batch, length, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)

            let offset = kvCache?.offset ?? 0
            q = rope(q, offset: offset)
            k = rope(k, offset: offset)

            let output =
                attentionWithCacheUpdate(
                    queries: q,
                    keys: k,
                    values: v,
                    cache: kvCache,
                    scale: scale,
                    mask: mask,
                )
                .transposed(0, 2, 1, 3)
                .reshaped(batch, length, -1)

            var projected = outputProjection(output)
            if let imageMask, let loraLinearOutput {
                projected =
                    projected
                    + loraLinearOutput(output)
                    * expandedDimensions(imageMask.asType(projected.dtype), axis: -1)
            }
            return projected
        }
    }

    final class RouterMLP: Module, UnaryLayer {
        @ModuleInfo(key: "0") var linear0: Linear
        @ModuleInfo(key: "2") var linear2: Linear
        @ModuleInfo(key: "4") var linear4: Linear

        init(config: Zaya1VLConfiguration.TextConfiguration, output: Int) {
            _linear0.wrappedValue = Linear(
                config.zayaMLPExpansion,
                config.zayaMLPExpansion,
                bias: true,
            )
            _linear2.wrappedValue = Linear(
                config.zayaMLPExpansion,
                config.zayaMLPExpansion,
                bias: true,
            )
            _linear4.wrappedValue = Linear(config.zayaMLPExpansion, output, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear4(gelu(linear2(gelu(linear0(x)))))
        }
    }

    final class Router: Module {
        let config: Zaya1VLConfiguration.TextConfiguration
        let numLocalExperts: Int
        let numExperts: Int
        let topK: Int
        let useEDA: Bool
        @ModuleInfo(key: "down_proj") var downProjection: Linear
        @ModuleInfo(key: "rmsnorm_eda") var rmsNormEDA: RMSNorm
        @ParameterInfo(key: "router_states_scale") var routerStatesScale: MLXArray?
        @ModuleInfo(key: "router_mlp") var routerMLP: RouterMLP
        @ParameterInfo(key: "balancing_biases") var balancingBiases: MLXArray

        init(_ config: Zaya1VLConfiguration.TextConfiguration, layerIndex: Int) {
            self.config = config
            numLocalExperts = config.numExperts
            numExperts = config.numExperts + (config.zayaUseMod ? 1 : 0)
            topK = config.moeRouterTopK
            useEDA = config.zayaUseEDA && layerIndex != 0
            _downProjection.wrappedValue = Linear(
                config.hiddenSize,
                config.zayaMLPExpansion,
                bias: true,
            )
            _rmsNormEDA.wrappedValue = RMSNorm(
                dimensions: config.zayaMLPExpansion,
                eps: config.normEpsilon,
            )
            if useEDA {
                _routerStatesScale.wrappedValue = MLXArray.ones([config.zayaMLPExpansion])
            }
            _routerMLP.wrappedValue = RouterMLP(config: config, output: numExperts)
            let biases = MLXArray.zeros([numExperts], dtype: .float32)
            if config.zayaUseMod {
                biases[numExperts - 1] = MLXArray(-1, dtype: .float32)
            }
            _balancingBiases.wrappedValue = biases
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            routerStates: MLXArray?,
        )
            -> (MLXArray, MLXArray, MLXArray)
        {
            var states = downProjection(hiddenStates)
            if useEDA, let routerStates, let routerStatesScale {
                states = states + routerStates * routerStatesScale
            }
            let nextRouterStates = states
            states = routerMLP(rmsNormEDA(states))
            let probabilities = softmax(states.asType(.float32), axis: -1).asType(
                hiddenStates.dtype)
            let biased = probabilities.asType(.float32) + balancingBiases
            let choices: MLXArray =
                if topK == 1 {
                    argMax(biased, axis: -1).expandedDimensions(axis: -1)
                } else {
                    argPartition(-biased, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
                }
            let routeProbabilities = takeAlong(probabilities, choices, axis: -1)
            return (routeProbabilities, choices, nextRouterStates)
        }
    }

    final class SwitchMLP: Module {
        let config: Zaya1VLConfiguration.TextConfiguration
        let numExperts: Int
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
        @ModuleInfo(key: "lora_fc1") var loraFC1: [SwitchLinear]?
        @ModuleInfo(key: "lora_fc2") var loraFC2: [SwitchLinear]?

        init(_ config: Zaya1VLConfiguration.TextConfiguration) {
            self.config = config
            numExperts = config.numExperts
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: config.hiddenSize,
                hiddenDims: config.ffnHiddenSizeOut,
                numExperts: config.numExperts,
                bias: config.addBiasLinear,
            )
            if config.visionLora {
                _loraFC1.wrappedValue = [
                    SwitchLinear(
                        inputDims: config.hiddenSize,
                        outputDims: config.visionLoraRankMLP,
                        numExperts: config.numExperts,
                        bias: false,
                    ),
                    SwitchLinear(
                        inputDims: config.visionLoraRankMLP,
                        outputDims: config.ffnHiddenSize,
                        numExperts: config.numExperts,
                        bias: false,
                    ),
                ]
                _loraFC2.wrappedValue = [
                    SwitchLinear(
                        inputDims: config.ffnHiddenSizeOut,
                        outputDims: config.visionLoraRankMLP,
                        numExperts: config.numExperts,
                        bias: false,
                    ),
                    SwitchLinear(
                        inputDims: config.visionLoraRankMLP,
                        outputDims: config.hiddenSize,
                        numExperts: config.numExperts,
                        bias: false,
                    ),
                ]
            }
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            expertChoice: MLXArray,
            routeProbabilities: MLXArray,
            imageMask: MLXArray?,
        )
            -> MLXArray
        {
            let skipMask = expertChoice .== MLXArray(numExperts)
            let expertIndices = minimum(expertChoice, MLXArray(numExperts - 1))
            var output = switchMLP(hiddenStates, expertIndices)

            if let imageMask, let loraFC1, let loraFC2 {
                var x = expandedDimensions(hiddenStates, axes: [-2, -3])
                var addon = loraFC1[0](x, expertIndices, sortedIndices: false).squeezed(axis: -2)
                addon = loraFC1[1](
                    expandedDimensions(addon, axis: -2), expertIndices, sortedIndices: false
                )
                .squeezed(axis: -2)
                let parts = addon.split(parts: 2, axis: -1)
                x = silu(parts[0]) * parts[1]
                addon = loraFC2[0](
                    expandedDimensions(x, axis: -2), expertIndices, sortedIndices: false
                )
                .squeezed(axis: -2)
                addon = loraFC2[1](
                    expandedDimensions(addon, axis: -2), expertIndices, sortedIndices: false
                )
                .squeezed(axis: -2)
                output =
                    output + addon
                    * expandedDimensions(imageMask.asType(output.dtype), axes: [-1, -2])
            }

            if config.zayaUseMod {
                output = MLX.where(
                    expandedDimensions(skipMask, axis: -1),
                    expandedDimensions(hiddenStates, axis: -2),
                    output,
                )
            }
            return (output * routeProbabilities[.ellipsis, .newAxis]).sum(axis: -2)
        }
    }

    final class ZayaBlock: Module {
        @ModuleInfo(key: "router") var router: Router
        @ModuleInfo(key: "experts") var experts: SwitchMLP

        init(_ config: Zaya1VLConfiguration.TextConfiguration, layerIndex: Int) {
            _router.wrappedValue = Router(config, layerIndex: layerIndex)
            _experts.wrappedValue = SwitchMLP(config)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            previousRouterStates: MLXArray?,
            imageMask: MLXArray?,
        )
            -> (MLXArray, MLXArray)
        {
            let (routeProbabilities, expertChoice, nextRouterStates) =
                router(hiddenStates, routerStates: previousRouterStates)
            let output = experts(
                hiddenStates,
                expertChoice: expertChoice,
                routeProbabilities: routeProbabilities,
                imageMask: imageMask,
            )
            return (output, nextRouterStates)
        }
    }

    final class AttentionLayer: Module {
        let config: Zaya1VLConfiguration.TextConfiguration
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "input_norm") var inputNorm: RMSNorm
        @ModuleInfo(key: "res_scale") var residualScale: ResidualScaling?

        init(_ config: Zaya1VLConfiguration.TextConfiguration, layerIndex: Int) {
            self.config = config
            _attention.wrappedValue = Attention(config)
            _inputNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEpsilon)
            if config.scaleResidualMerge {
                _residualScale.wrappedValue = ResidualScaling(config: config, index: 2 * layerIndex)
            }
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            residual: MLXArray?,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            imageMask: MLXArray?,
            cache: KVCache?,
            ccaMask: MLXArray?,
        )
            -> (MLXArray, MLXArray)
        {
            var hiddenStates = hiddenStates
            var residual = residual
            if let residualScale {
                (residual, hiddenStates) = residualScale(
                    residual: residual,
                    hiddenStates: hiddenStates,
                )
            }
            let mergedResidual = residual == nil ? hiddenStates : hiddenStates + residual!
            let output = attention(
                inputNorm(mergedResidual),
                mask: mask,
                ccaMask: ccaMask,
                imageMask: imageMask,
                cache: cache,
            )
            return (output, mergedResidual)
        }
    }

    final class MLPLayer: Module {
        let config: Zaya1VLConfiguration.TextConfiguration
        @ModuleInfo(key: "zaya_block") var zayaBlock: ZayaBlock
        @ModuleInfo(key: "input_norm") var inputNorm: RMSNorm
        @ModuleInfo(key: "res_scale") var residualScale: ResidualScaling?

        init(_ config: Zaya1VLConfiguration.TextConfiguration, layerIndex: Int) {
            self.config = config
            _zayaBlock.wrappedValue = ZayaBlock(config, layerIndex: layerIndex)
            _inputNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEpsilon)
            if config.scaleResidualMerge {
                _residualScale.wrappedValue = ResidualScaling(
                    config: config, index: 2 * layerIndex + 1)
            }
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            residual: MLXArray?,
            imageMask: MLXArray?,
            previousRouterStates: MLXArray?,
        )
            -> (MLXArray, MLXArray, MLXArray)
        {
            var hiddenStates = hiddenStates
            var residual = residual
            if let residualScale {
                (residual, hiddenStates) = residualScale(
                    residual: residual,
                    hiddenStates: hiddenStates,
                )
            }
            let mergedResidual = residual == nil ? hiddenStates : hiddenStates + residual!
            let (output, nextRouterStates) = zayaBlock(
                inputNorm(mergedResidual),
                previousRouterStates: previousRouterStates,
                imageMask: imageMask,
            )
            return (output, mergedResidual, nextRouterStates)
        }
    }

    final class DecoderLayer: Module {
        @ModuleInfo(key: "attn") var attention: AttentionLayer
        @ModuleInfo(key: "mlp") var mlp: MLPLayer

        init(_ config: Zaya1VLConfiguration.TextConfiguration, layerIndex: Int) {
            _attention.wrappedValue = AttentionLayer(config, layerIndex: layerIndex)
            _mlp.wrappedValue = MLPLayer(config, layerIndex: layerIndex)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            residual: MLXArray?,
            mask: MLXFast.ScaledDotProductAttentionMaskMode,
            imageMask: MLXArray?,
            cache: KVCache?,
            previousRouterStates: MLXArray?,
            ccaMask: MLXArray?,
        )
            -> (MLXArray, MLXArray, MLXArray)
        {
            let (attnOutput, attnResidual) = attention(
                hiddenStates,
                residual: residual,
                mask: mask,
                imageMask: imageMask,
                cache: cache,
                ccaMask: ccaMask,
            )
            let (mlpOutput, mlpResidual, nextRouterStates) = mlp(
                attnOutput,
                residual: attnResidual,
                imageMask: imageMask,
                previousRouterStates: previousRouterStates,
            )
            return (mlpOutput, mlpResidual, nextRouterStates)
        }
    }

    final class ZayaModel: Module {
        let config: Zaya1VLConfiguration.TextConfiguration
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") var layers: [DecoderLayer]
        @ModuleInfo(key: "res_scale") var residualScale: ResidualScaling?
        @ModuleInfo(key: "final_norm") var finalNorm: RMSNorm

        init(_ config: Zaya1VLConfiguration.TextConfiguration) {
            self.config = config
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize,
                dimensions: config.hiddenSize,
            )
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
                DecoderLayer(config, layerIndex: $0)
            }
            if config.scaleResidualMerge {
                _residualScale.wrappedValue = ResidualScaling(
                    config: config,
                    index: config.hiddenLayers,
                )
            }
            _finalNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.normEpsilon)
        }

        func callAsFunction(
            inputIDs: MLXArray?,
            inputEmbeddings: MLXArray?,
            mask: MLXArray?,
            imageMask: MLXArray?,
            cache: [KVCache]?,
        )
            -> MLXArray
        {
            var hidden =
                inputEmbeddings
                ?? embedTokens(inputIDs ?? MLXArray.zeros([1, 0], dtype: .int32))
            let firstCache = cache?.first
            let attnMask = createAttentionMask(h: hidden, cache: firstCache)
            let maskMode = attnMask
            let ccaMask = (mask?.ndim == 2) ? mask : nil
            var residual: MLXArray?
            var routerStates: MLXArray?

            for (index, layer) in layers.enumerated() {
                let (nextHidden, nextResidual, nextRouterStates) = layer(
                    hidden,
                    residual: residual,
                    mask: maskMode,
                    imageMask: imageMask,
                    cache: cache?[index],
                    previousRouterStates: routerStates,
                    ccaMask: ccaMask,
                )
                hidden = nextHidden
                residual = nextResidual
                routerStates = nextRouterStates
            }

            if let residualScale {
                (residual, hidden) = residualScale(residual: residual, hiddenStates: hidden)
            }
            let merged = residual == nil ? hidden : hidden + residual!
            return finalNorm(merged)
        }
    }

    final class LanguageModel: Module {
        let args: Zaya1VLConfiguration.TextConfiguration
        @ModuleInfo(key: "model") var model: ZayaModel
        @ModuleInfo(key: "lm_head") var lmHead: Linear?
        let kvHeads: [Int]

        init(_ args: Zaya1VLConfiguration.TextConfiguration) {
            self.args = args
            _model.wrappedValue = ZayaModel(args)
            if !args.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    args.hiddenSize,
                    args.vocabularySize,
                    bias: args.lmHeadBias,
                )
            }
            kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        }

        func callAsFunction(
            _ inputIDs: MLXArray?,
            cache: [KVCache]?,
            state: LMOutput.State?,
            inputEmbeddings: MLXArray? = nil,
            mask: MLXArray? = nil,
            imageMask: MLXArray? = nil,
        )
            -> LMOutput
        {
            let hidden = model(
                inputIDs: inputIDs,
                inputEmbeddings: inputEmbeddings,
                mask: mask,
                imageMask: imageMask,
                cache: cache,
            )
            let logits = lmHead.map { $0(hidden) } ?? model.embedTokens.asLinear(hidden)
            return LMOutput(logits: logits, state: state)
        }
    }
}

// MARK: - Vision

private enum Zaya1VLVision {
    static func applyRotary(_ tensor: MLXArray, freqs: MLXArray) -> MLXArray {
        var cos = cos(freqs)
        var sin = sin(freqs)
        cos = expandedDimensions(cos, axis: 1)
        cos = tiled(cos, repetitions: [1, 1, 2])
        cos = expandedDimensions(cos, axis: 0)
        sin = expandedDimensions(sin, axis: 1)
        sin = tiled(sin, repetitions: [1, 1, 2])
        sin = expandedDimensions(sin, axis: 0)
        return ((tensor * cos) + (RotaryEmbeddingOps.rotateHalf(tensor) * sin)).asType(tensor.dtype)
    }

    final class VisionRotaryEmbedding {
        let dimensions: Int
        let theta: Float
        let inverseFreq: MLXArray

        init(dimensions: Int, theta: Float = 10000) {
            self.dimensions = dimensions
            self.theta = theta
            let positions = MLXArray(stride(from: 0, to: dimensions, by: 2)).asType(.float32)
            inverseFreq = 1.0 / pow(theta, positions / Float(dimensions))
        }

        func callAsFunction(sequenceLength: Int) -> MLXArray {
            outer(MLXArray(0 ..< sequenceLength).asType(inverseFreq.dtype), inverseFreq)
        }
    }

    final class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo(key: "proj") var proj: Conv3d
        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let hiddenSize: Int

        init(_ config: Zaya1VLConfiguration.VisionConfiguration) {
            patchSize = config.patchSize
            temporalPatchSize = config.temporalPatchSize
            inChannels = config.inChannels
            hiddenSize = config.hiddenSize
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
            var hidden = x.reshaped(
                -1,
                inChannels,
                temporalPatchSize,
                patchSize,
                patchSize,
            )
            .movedAxis(source: 1, destination: 4)
            hidden = proj(hidden)
            return hidden.reshaped(-1, hiddenSize)
        }
    }

    final class PatchMerger: Module, UnaryLayer {
        let hiddenSize: Int
        @ModuleInfo(key: "ln_q") var norm: RMSNorm
        @ModuleInfo(key: "mlp") var mlp: (Linear, GELU, Linear)

        init(_ config: Zaya1VLConfiguration.VisionConfiguration) {
            hiddenSize = config.hiddenSize * config.spatialMergeSize * config.spatialMergeSize
            _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _mlp.wrappedValue = (
                Linear(hiddenSize, hiddenSize),
                GELU(),
                Linear(hiddenSize, config.outHiddenSize),
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let hidden = norm(x).reshaped(-1, hiddenSize)
            return mlp.2(gelu(mlp.0(hidden)))
        }
    }

    final class Attention: Module {
        let numHeads: Int
        let scale: Float
        @ModuleInfo(key: "qkv") var qkv: Linear
        @ModuleInfo(key: "proj") var projection: Linear

        init(dimensions: Int, numHeads: Int) {
            self.numHeads = numHeads
            scale = pow(Float(dimensions / numHeads), -0.5)
            _qkv.wrappedValue = Linear(dimensions, 3 * dimensions, bias: true)
            _projection.wrappedValue = Linear(dimensions, dimensions, bias: true)
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionMask: MLXArray,
            rotaryPositionEmbedding: MLXArray,
        )
            -> MLXArray
        {
            let sequenceLength = x.dim(0)
            let parts = qkv(x).split(parts: 3, axis: -1)
            var q = parts[0].reshaped(sequenceLength, numHeads, -1)
            var k = parts[1].reshaped(sequenceLength, numHeads, -1)
            var v = parts[2].reshaped(sequenceLength, numHeads, -1)
            q = applyRotary(q, freqs: rotaryPositionEmbedding)
            k = applyRotary(k, freqs: rotaryPositionEmbedding)
            q = q.reshaped(1, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)
            k = k.reshaped(1, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)
            v = v.reshaped(1, sequenceLength, numHeads, -1).transposed(0, 2, 1, 3)
            let output = MLXFast.scaledDotProductAttention(
                queries: q,
                keys: k,
                values: v,
                scale: scale,
                mask: .array(attentionMask),
            )
            .transposed(0, 2, 1, 3)
            .reshaped(sequenceLength, -1)
            return projection(output)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(_ config: Zaya1VLConfiguration.VisionConfiguration) {
            _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: true)
            _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    final class Block: Module {
        @ModuleInfo(key: "norm1") var norm1: RMSNorm
        @ModuleInfo(key: "norm2") var norm2: RMSNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP

        init(_ config: Zaya1VLConfiguration.VisionConfiguration) {
            _norm1.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _norm2.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _attention.wrappedValue = Attention(
                dimensions: config.hiddenSize, numHeads: config.numHeads)
            _mlp.wrappedValue = MLP(config)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            attentionMask: MLXArray,
            rotaryPositionEmbedding: MLXArray,
        )
            -> MLXArray
        {
            let hidden =
                hiddenStates
                + attention(
                    norm1(hiddenStates),
                    attentionMask: attentionMask,
                    rotaryPositionEmbedding: rotaryPositionEmbedding,
                )
            return hidden + mlp(norm2(hidden))
        }
    }

    final class VisionModel: Module {
        let config: Zaya1VLConfiguration.VisionConfiguration
        let spatialMergeSize: Int
        let windowSize: Int
        let patchSize: Int
        let spatialMergeUnit: Int
        let fullAttentionBlockIndexes: [Int]
        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        let rotaryPositionEmbedding: VisionRotaryEmbedding
        @ModuleInfo(key: "blocks") var blocks: [Block]
        @ModuleInfo(key: "merger") var merger: PatchMerger

        init(_ config: Zaya1VLConfiguration.VisionConfiguration) {
            self.config = config
            spatialMergeSize = config.spatialMergeSize
            windowSize = config.windowSize
            patchSize = config.patchSize
            spatialMergeUnit = config.spatialMergeSize * config.spatialMergeSize
            fullAttentionBlockIndexes = config.fullAttentionBlockIndexes
            _patchEmbed.wrappedValue = PatchEmbed(config)
            let headDimensions = config.hiddenSize / config.numHeads
            rotaryPositionEmbedding = VisionRotaryEmbedding(dimensions: headDimensions / 2)
            _blocks.wrappedValue = (0 ..< config.depth).map { _ in Block(config) }
            _merger.wrappedValue = PatchMerger(config)
        }

        func callAsFunction(_ pixelValues: MLXArray, frames: [THW]) -> MLXArray {
            var hiddenStates = patchEmbed(pixelValues)
            let rotaryEmbedding = rotaryPositionEmbeddings(frames)
            let (windowIndex, cuWindowSeqlens) = getWindowIndex(frames)
            let sequenceLength = hiddenStates.dim(0)
            var cuSeqlens = [0]
            for frame in frames {
                let length = frame.h * frame.w
                cuSeqlens.append(
                    contentsOf: Array(repeating: length, count: frame.t).map {
                        cuSeqlens.last! + $0
                    },
                )
            }
            let fullAttentionMask = attentionMask(
                sequenceLength: sequenceLength,
                cuSeqlens: MLXArray(cuSeqlens),
                dtype: hiddenStates.dtype,
            )
            let windowAttentionMask = attentionMask(
                sequenceLength: sequenceLength,
                cuSeqlens: cuWindowSeqlens,
                dtype: hiddenStates.dtype,
            )

            hiddenStates = hiddenStates.reshaped(
                sequenceLength / spatialMergeUnit,
                spatialMergeUnit,
                -1,
            )
            hiddenStates = hiddenStates[windowIndex, 0..., 0...].reshaped(sequenceLength, -1)
            var rotary = rotaryEmbedding.reshaped(
                sequenceLength / spatialMergeUnit,
                spatialMergeUnit,
                -1,
            )
            rotary = rotary[windowIndex, 0..., 0...].reshaped(sequenceLength, -1)

            for (index, block) in blocks.enumerated() {
                let mask =
                    fullAttentionBlockIndexes.contains(index)
                    ? fullAttentionMask
                    : windowAttentionMask
                hiddenStates = block(
                    hiddenStates,
                    attentionMask: mask,
                    rotaryPositionEmbedding: rotary,
                )
            }
            hiddenStates = merger(hiddenStates)
            let reverseIndex = argSort(windowIndex, axis: 0)
            return hiddenStates[reverseIndex, 0...]
        }

        private func rotaryPositionEmbeddings(_ frames: [THW]) -> MLXArray {
            var positionIDs = [MLXArray]()
            for frame in frames {
                let (t, h, w) = frame.values
                var hIDs = expandedDimensions(MLXArray(0 ..< h), axis: 1)
                hIDs = repeated(hIDs, count: w, axis: 1)
                hIDs =
                    hIDs
                    .reshaped(
                        h / spatialMergeSize, spatialMergeSize, w / spatialMergeSize,
                        spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()
                var wIDs = expandedDimensions(MLXArray(0 ..< w), axis: 0)
                wIDs = repeated(wIDs, count: h, axis: 0)
                wIDs =
                    wIDs
                    .reshaped(
                        h / spatialMergeSize, spatialMergeSize, w / spatialMergeSize,
                        spatialMergeSize
                    )
                    .transposed(0, 2, 1, 3)
                    .flattened()
                positionIDs.append(tiled(stacked([hIDs, wIDs], axis: -1), repetitions: [t, 1]))
            }
            let indices = concatenated(positionIDs, axis: 0)
            let maxFrameSize = frames.map { max($0.h, $0.w) }.max() ?? 0
            return rotaryPositionEmbedding(sequenceLength: maxFrameSize)[indices].reshaped(
                indices.dim(0), -1)
        }

        private func getWindowIndex(_ frames: [THW]) -> (MLXArray, MLXArray) {
            var windowIndex = [MLXArray]()
            var cuWindowSeqlens = [0]
            var windowIndexID = 0
            let vitMergerWindowSize = windowSize / spatialMergeSize / patchSize

            for frame in frames {
                let (gridT, gridH, gridW) = frame.values
                let llmGridH = gridH / spatialMergeSize
                let llmGridW = gridW / spatialMergeSize
                let index = MLXArray(0 ..< (gridT * llmGridH * llmGridW))
                    .reshaped(gridT, llmGridH, llmGridW)
                let padH = vitMergerWindowSize - llmGridH % vitMergerWindowSize
                let padW = vitMergerWindowSize - llmGridW % vitMergerWindowSize
                let numWindowsH = (llmGridH + padH) / vitMergerWindowSize
                let numWindowsW = (llmGridW + padW) / vitMergerWindowSize
                let paddedIndex = padded(
                    index,
                    widths: [[0, 0], [0, padH], [0, padW]],
                    mode: .constant,
                    value: MLXArray(-100),
                )
                let transposedIndex =
                    paddedIndex
                    .reshaped(
                        gridT, numWindowsH, vitMergerWindowSize, numWindowsW, vitMergerWindowSize
                    )
                    .transposed(0, 1, 3, 2, 4)
                    .reshaped(
                        gridT, numWindowsH * numWindowsW, vitMergerWindowSize, vitMergerWindowSize)
                let sequenceLengths = sum(transposedIndex .!= -100, axes: [2, 3]).reshaped(-1)
                let flattened = transposedIndex.flattened()
                let validIndices =
                    flattened.asArray(Int.self).enumerated()
                    .filter { $0.element != -100 }
                    .map(\.offset)
                windowIndex.append(flattened[MLXArray(validIndices)] + windowIndexID)
                let temporary =
                    cumsum(sequenceLengths, axis: 0) * spatialMergeUnit + cuWindowSeqlens.last!
                cuWindowSeqlens.append(contentsOf: temporary.asArray(Int.self))
                windowIndexID += gridT * llmGridH * llmGridW
            }

            let combined = concatenated(windowIndex, axis: 0)
            let seqlensArray = MLXArray(cuWindowSeqlens)
            var seen = Set<Int>()
            var unique = [Int]()
            for (index, value) in cuWindowSeqlens.enumerated() where !seen.contains(value) {
                seen.insert(value)
                unique.append(index)
            }
            return (combined, seqlensArray[MLXArray(unique)])
        }

        private func attentionMask(sequenceLength: Int, cuSeqlens: MLXArray, dtype: DType)
            -> MLXArray
        {
            let mask = full([1, sequenceLength, sequenceLength], values: false)
            let seqlens = cuSeqlens.asArray(Int.self)
            for index in 1 ..< seqlens.count {
                mask[
                    0..., seqlens[index - 1] ..< seqlens[index],
                    seqlens[index - 1] ..< seqlens[index]] =
                    MLXArray(true)
            }
            return MLX.where(
                mask[.newAxis, 0..., 0..., 0...],
                MLXArray(0, dtype: dtype),
                MLXArray(-10000, dtype: dtype),
            )
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var result = [String: MLXArray]()
            for (key, value) in weights {
                if key.contains("position_ids") {
                    continue
                }
                if key.hasSuffix("patch_embed.proj.weight"), value.ndim == 5,
                    value.dim(-1) != config.inChannels
                {
                    result[key] = value.transposed(0, 2, 3, 4, 1)
                } else if key.hasSuffix("patch_embed.proj.weight"), value.ndim == 4 {
                    result[key] = value.transposed(0, 2, 3, 1)
                } else {
                    result[key] = value
                }
            }
            return result
        }
    }
}

// MARK: - Model

public final class Zaya1VL: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: Zaya1VLVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Zaya1VLLanguage.LanguageModel

    public let config: Zaya1VLConfiguration
    public var vocabularySize: Int {
        config.vocabularySize
    }

    public var kvHeads: [Int] {
        languageModel.kvHeads
    }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public init(_ config: Zaya1VLConfiguration) {
        self.config = config
        _visionTower.wrappedValue = Zaya1VLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = Zaya1VLLanguage.LanguageModel(config.textConfiguration)
    }

    private func inputEmbeddings(
        inputIDs: MLXArray,
        pixelValues: MLXArray?,
        frames: [THW]?,
    )
        throws -> (MLXArray, MLXArray?)
    {
        let embeddings = languageModel.model.embedTokens(inputIDs)
        guard let pixelValues, let frames, !frames.isEmpty else {
            return (embeddings, nil)
        }
        let features = visionTower(
            pixelValues.asType(visionTower.patchEmbed.proj.weight.dtype), frames: frames
        )
        .asType(embeddings.dtype)
        let imageMask = inputIDs .== MLXArray(config.imageTokenID)
        let positions = imageMask.flattened().asArray(Bool.self).enumerated()
            .filter(\.element)
            .map(\.offset)
        precondition(
            positions.count == features.dim(0),
            "Zaya1-VL image feature/token count mismatch: tokens \(positions.count), features \(features.dim(0)).",
        )
        embeddings[0..., MLXArray(positions), 0...] = features.expandedDimensions(axis: 0)
        return (embeddings, imageMask)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let (embeddings, imageMask) = try inputEmbeddings(
            inputIDs: input.text.tokens,
            pixelValues: input.image?.pixels,
            frames: input.image?.frames,
        )
        let result = languageModel(
            nil,
            cache: cache,
            state: nil,
            inputEmbeddings: embeddings,
            mask: input.text.mask,
            imageMask: imageMask,
        )
        prefill.progress?(input.text.tokens.size, input.text.tokens.size)
        return .logits(result)
    }

    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        languageModel(input.tokens, cache: cache, state: state, mask: input.mask)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache, state: nil).logits
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try (0 ..< kvHeads.count).map { _ in
            CacheList(try makeAttentionKVCache(configuration: configuration), ArraysCache(size: 2))
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Zaya1VLSanitizer.sanitize(weights: weights, config: config, visionTower: visionTower)
    }
}

private enum Zaya1VLSanitizer {
    static func sanitize(
        weights: [String: MLXArray],
        config: Zaya1VLConfiguration,
        visionTower: Zaya1VLVision.VisionModel,
    )
        -> [String: MLXArray]
    {
        var sanitized = [String: MLXArray]()
        for (key, value) in weights {
            if key.contains("position_ids") || key.contains("rotary_emb.inv_freq") {
                continue
            }
            if key == "lm_head.weight", config.textConfiguration.tieWordEmbeddings {
                continue
            }
            var newKey = key
            if newKey.hasPrefix("model.layers."),
                newKey.contains(".zaya_block.experts.switch_mlp.")
            {
                newKey = newKey.replacingOccurrences(
                    of: #"^model\.layers\.(\d+)\.zaya_block\.experts\.switch_mlp\."#,
                    with: "language_model.model.layers.$1.mlp.zaya_block.experts.switch_mlp.",
                    options: .regularExpression,
                )
            } else if newKey.hasPrefix("model.") {
                newKey = newKey.replacingOccurrences(
                    of: "model.",
                    with: "language_model.model.",
                    options: [],
                    range: newKey.startIndex ..< newKey.index(newKey.startIndex, offsetBy: 6),
                )
            } else if newKey.hasPrefix("lm_head.") {
                newKey = newKey.replacingOccurrences(
                    of: "lm_head.",
                    with: "language_model.lm_head.",
                    options: [],
                    range: newKey.startIndex ..< newKey.index(newKey.startIndex, offsetBy: 8),
                )
            }
            sanitized[newKey] = value
        }

        stackLocalExperts(weights: &sanitized, config: config.textConfiguration)
        sanitized = visionTower.sanitize(weights: sanitized)
        transposeConvQKIfNeeded(weights: &sanitized)
        return sanitized
    }

    private static func stackLocalExperts(
        weights: inout [String: MLXArray],
        config: Zaya1VLConfiguration.TextConfiguration,
    ) {
        guard config.visionLora else {
            return
        }
        for layer in 0 ..< config.hiddenLayers {
            let prefix = "language_model.model.layers.\(layer).mlp.zaya_block.experts"
            for loraName in ["lora_fc1", "lora_fc2"] {
                for subIndex in 0 ... 1 {
                    var tensors = [MLXArray]()
                    for expert in 0 ..< config.numExperts {
                        let key =
                            "\(prefix).local_experts.\(expert).\(loraName).\(subIndex).weight"
                        guard let tensor = weights.removeValue(forKey: key) else {
                            tensors.removeAll()
                            break
                        }
                        tensors.append(tensor)
                    }
                    if tensors.count == config.numExperts {
                        weights["\(prefix).\(loraName).\(subIndex).weight"] = stacked(
                            tensors, axis: 0)
                    }
                }
            }
        }
    }

    private static func transposeConvQKIfNeeded(weights: inout [String: MLXArray]) {
        for (key, value) in weights where key.contains("conv_qk") && key.hasSuffix("weight") {
            if value.ndim == 3, value.shape[1] != 2, value.shape[2] == 2 {
                weights[key] = value.transposed(0, 2, 1)
            }
        }
    }
}
