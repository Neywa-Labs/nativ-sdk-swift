// Copyright 2026 The HuggingFace Team. All rights reserved.
// Adapted from the Apache-2.0 mlx-vlm LFM3 implementation.
import Foundation

public struct LFM3RotaryConfiguration: Decodable, Sendable {
    let theta: Float
    let partialFactor: Float
    let type: String
    enum CodingKeys: String, CodingKey {
        case theta = "rope_theta"
        case partialFactor = "partial_rotary_factor"
        case type = "rope_type"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        theta = try c.decodeIfPresent(Float.self, forKey: .theta) ?? 1_000_000
        partialFactor = try c.decodeIfPresent(Float.self, forKey: .partialFactor) ?? 1
        type = try c.decodeIfPresent(String.self, forKey: .type) ?? "default"
    }
}

public struct LFM3TextConfiguration: Decodable, Sendable {
    let hiddenSize: Int
    let intermediateSize: Int
    let hiddenLayers: Int
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let vocabularySize: Int
    let normEps: Float
    let convLength: Int
    let convBias: Bool
    let layerTypes: [String]
    let rotary: [String: LFM3RotaryConfiguration]
    let slidingWindow: Int
    let sandwichNorm: Bool
    let tieEmbeddings: Bool
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case normEps = "norm_eps"
        case convLength = "conv_L_cache"
        case convBias = "conv_bias"
        case layerTypes = "layer_types"
        case rotary = "rope_parameters"
        case slidingWindow = "sliding_window"
        case sandwichNorm = "use_sandwich_norm"
        case tieEmbeddings = "tie_word_embeddings"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 7680
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 24
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 4
        headDim =
            try c.decodeIfPresent(Int.self, forKey: .headDim) ?? hiddenSize / max(1, attentionHeads)
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 128000
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        convLength = try c.decodeIfPresent(Int.self, forKey: .convLength) ?? 7
        convBias = try c.decodeIfPresent(Bool.self, forKey: .convBias) ?? false
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
        rotary =
            try c.decodeIfPresent([String: LFM3RotaryConfiguration].self, forKey: .rotary) ?? [:]
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 256
        sandwichNorm = try c.decodeIfPresent(Bool.self, forKey: .sandwichNorm) ?? true
        tieEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieEmbeddings) ?? true
        guard hiddenSize > 0, intermediateSize > 0, attentionHeads > 0, kvHeads > 0,
            attentionHeads % kvHeads == 0, headDim > 0, headDim % 2 == 0,
            vocabularySize > 0, convLength > 0, hiddenLayers > 0,
            layerTypes.count == hiddenLayers,
            layerTypes.allSatisfy({ ["conv", "full_attention", "sliding_attention"].contains($0) }),
            !layerTypes.contains("sliding_attention") || slidingWindow > 0
        else {
            throw ModelFactoryError.invalidConfiguration("Invalid LFM3 text dimensions or layers")
        }
        for type in Set(layerTypes).subtracting(["conv"]) {
            if let rope = rotary[type] {
                guard rope.type == "default", rope.theta > 0, rope.partialFactor > 0,
                    rope.partialFactor <= 1, Int(Float(headDim) * rope.partialFactor) >= 2
                else {
                    throw ModelFactoryError.invalidConfiguration(
                        "Unsupported LFM3 rotary configuration")
                }
            }
        }
    }
}

public struct LFM3VisionConfiguration: Decodable, Sendable {
    let hiddenSize: Int
    let intermediateSize: Int
    let hiddenLayers: Int
    let attentionHeads: Int
    let kvHeads: Int
    let channels: Int
    let patchSize: Int
    let temporalPatchSize: Int
    let mergeSize: Int
    let kernelSize: Int
    let normEps: Float
    let layerTypes: [String]
    let rotary: LFM3RotaryConfiguration?
    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case channels = "in_channels"
        case patchSize = "patch_size"
        case temporalPatchSize = "temporal_patch_size"
        case mergeSize = "spatial_merge_size"
        case kernelSize = "conv_kernel_size"
        case normEps = "norm_eps"
        case layerTypes = "layer_types"
        case rotary = "rope_parameters"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2560
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 14
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
        channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? 3
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 16
        temporalPatchSize = try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize) ?? 2
        mergeSize = try c.decodeIfPresent(Int.self, forKey: .mergeSize) ?? 2
        kernelSize = try c.decodeIfPresent(Int.self, forKey: .kernelSize) ?? 3
        normEps = try c.decodeIfPresent(Float.self, forKey: .normEps) ?? 1e-5
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
        rotary = try c.decodeIfPresent(LFM3RotaryConfiguration.self, forKey: .rotary)
        guard hiddenSize > 0, attentionHeads > 0, kvHeads > 0,
            hiddenSize % (attentionHeads * 4) == 0, attentionHeads % kvHeads == 0,
            patchSize > 0, mergeSize > 0, temporalPatchSize > 0, kernelSize > 0,
            layerTypes.count == hiddenLayers,
            layerTypes.allSatisfy({ ["conv", "full_attention"].contains($0) }),
            rotary == nil || rotary?.type == "axial"
        else { throw ModelFactoryError.invalidConfiguration("Invalid LFM3 vision configuration") }
    }
}

public struct LFM3AudioEncoderConfiguration: Decodable, Sendable {
    let text: LFM2Configuration
    let channels: Int
    let hiddenSize: Int
    let melBins: Int
    let window: Int
    enum CodingKeys: String, CodingKey {
        case text = "text_config"
        case legacyText = "lfm"
        case channels = "conv_channels"
        case legacyChannels = "dim_conv"
        case hiddenSize = "hidden_size"
        case legacyHiddenSize = "dim_out"
        case melBins = "num_mel_bins"
        case legacyMelBins = "dim_in"
        case window = "sliding_window"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        text =
            try c.decodeIfPresent(LFM2Configuration.self, forKey: .text)
            ?? c.decode(LFM2Configuration.self, forKey: .legacyText)
        channels =
            try c.decodeIfPresent(Int.self, forKey: .channels)
            ?? c.decodeIfPresent(Int.self, forKey: .legacyChannels) ?? 256
        hiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .hiddenSize)
            ?? c.decodeIfPresent(Int.self, forKey: .legacyHiddenSize) ?? 768
        melBins =
            try c.decodeIfPresent(Int.self, forKey: .melBins)
            ?? c.decodeIfPresent(Int.self, forKey: .legacyMelBins) ?? 128
        window = try c.decodeIfPresent(Int.self, forKey: .window) ?? 32
        guard channels > 0, hiddenSize == text.hiddenSize, melBins > 0,
            melBins % 8 == 0, window > 0
        else { throw ModelFactoryError.invalidConfiguration("Invalid LFM3 audio encoder") }
    }
}

public struct LFM3Configuration: Decodable, Sendable {
    let modelType: String
    let textConfig: LFM3TextConfiguration?
    let legacyText: LFM3TextConfiguration?
    let audioConfig: LFM3AudioEncoderConfiguration?
    let legacyAudio: LFM3AudioEncoderConfiguration?
    let depthConfig: LFM3TextConfiguration?
    let legacyDepth: LFM3TextConfiguration?
    let vision: LFM3VisionConfiguration?
    let numCodebooks: Int
    let legacyCodebooks: Int?
    let codebookSize: Int
    let legacyVocabulary: Int?
    let audioEOS: Int
    let maskgitSteps: Int
    let maskgitSchedule: [Int]?
    let textSlots: Int
    let audioSlots: Int
    let audioToken: Int
    let audioOutToken: Int
    let audioStart: Int
    let textEnd: Int
    let mixedStart: Int
    let imageToken: Int
    let videoToken: Int
    let downsample: Int
    let projectorSize: Int
    let projectorBias: Bool
    let projectorEps: Float
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfig = "text_config"
        case legacyText = "lfm"
        case audioConfig = "audio_encoder_config"
        case legacyAudio = "encoder"
        case depthConfig = "depth_config"
        case legacyDepth = "depthformer"
        case vision = "vision_config"
        case numCodebooks = "num_codebooks"
        case legacyCodebooks = "codebooks"
        case codebookSize = "codebook_size"
        case legacyVocabulary = "audio_vocab_size"
        case audioEOS = "audio_eos_code"
        case maskgitSteps = "maskgit_steps"
        case maskgitSchedule = "maskgit_schedule"
        case textSlots = "interleaved_n_text"
        case audioSlots = "interleaved_n_audio"
        case audioToken = "audio_token_id"
        case audioOutToken = "audio_out_token_id"
        case audioStart = "audio_start_token_id"
        case textEnd = "text_end_token_id"
        case mixedStart = "mixed_start_token_id"
        case imageToken = "image_token_id"
        case videoToken = "video_token_id"
        case downsample = "downsample_factor"
        case projectorSize = "projector_hidden_size"
        case projectorBias = "projector_bias"
        case projectorEps = "projector_layer_norm_eps"
    }
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decode(String.self, forKey: .modelType)
        textConfig = try c.decodeIfPresent(LFM3TextConfiguration.self, forKey: .textConfig)
        legacyText = try c.decodeIfPresent(LFM3TextConfiguration.self, forKey: .legacyText)
        audioConfig = try c.decodeIfPresent(
            LFM3AudioEncoderConfiguration.self, forKey: .audioConfig)
        legacyAudio = try c.decodeIfPresent(
            LFM3AudioEncoderConfiguration.self, forKey: .legacyAudio)
        depthConfig = try c.decodeIfPresent(LFM3TextConfiguration.self, forKey: .depthConfig)
        legacyDepth = try c.decodeIfPresent(LFM3TextConfiguration.self, forKey: .legacyDepth)
        vision =
            modelType == "lfm3_omni"
            ? try c.decode(LFM3VisionConfiguration.self, forKey: .vision) : nil
        numCodebooks =
            try c.decodeIfPresent(Int.self, forKey: .numCodebooks) ?? c.decodeIfPresent(
                Int.self, forKey: .legacyCodebooks) ?? 12
        legacyCodebooks = try c.decodeIfPresent(Int.self, forKey: .legacyCodebooks)
        codebookSize =
            try c.decodeIfPresent(Int.self, forKey: .codebookSize) ?? c.decodeIfPresent(
                Int.self, forKey: .legacyVocabulary) ?? 2017
        legacyVocabulary = try c.decodeIfPresent(Int.self, forKey: .legacyVocabulary)
        audioEOS = try c.decodeIfPresent(Int.self, forKey: .audioEOS) ?? codebookSize - 1
        maskgitSteps = try c.decodeIfPresent(Int.self, forKey: .maskgitSteps) ?? 3
        maskgitSchedule = try c.decodeIfPresent([Int].self, forKey: .maskgitSchedule)
        textSlots = try c.decodeIfPresent(Int.self, forKey: .textSlots) ?? 6
        audioSlots = try c.decodeIfPresent(Int.self, forKey: .audioSlots) ?? 12
        audioToken = try c.decodeIfPresent(Int.self, forKey: .audioToken) ?? 125031
        audioOutToken = try c.decodeIfPresent(Int.self, forKey: .audioOutToken) ?? 125032
        audioStart = try c.decodeIfPresent(Int.self, forKey: .audioStart) ?? 125012
        textEnd = try c.decodeIfPresent(Int.self, forKey: .textEnd) ?? 125014
        mixedStart = try c.decodeIfPresent(Int.self, forKey: .mixedStart) ?? 125015
        imageToken = try c.decodeIfPresent(Int.self, forKey: .imageToken) ?? 125030
        videoToken = try c.decodeIfPresent(Int.self, forKey: .videoToken) ?? 125029
        downsample = try c.decodeIfPresent(Int.self, forKey: .downsample) ?? 2
        projectorSize = try c.decodeIfPresent(Int.self, forKey: .projectorSize) ?? 4096
        projectorBias = try c.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? true
        projectorEps = try c.decodeIfPresent(Float.self, forKey: .projectorEps) ?? 1e-6
        guard let text = textConfig ?? legacyText, audioConfig ?? legacyAudio != nil,
            let depth = depthConfig ?? legacyDepth,
            ["lfm3_audio", "lfm3_omni"].contains(modelType),
            numCodebooks > 0, codebookSize > 1, maskgitSteps > 0,
            maskgitSteps <= numCodebooks, textSlots > 0, audioSlots > 0,
            audioEOS == codebookSize - 1,
            [audioToken, audioOutToken, audioStart, textEnd, mixedStart].allSatisfy({
                $0 >= 0 && $0 < text.vocabularySize
            }), !depth.layerTypes.contains("conv")
        else { throw ModelFactoryError.invalidConfiguration("Invalid LFM3 speech configuration") }
        if let schedule = maskgitSchedule {
            guard schedule.allSatisfy({ $0 > 0 }), schedule.reduce(0, +) == numCodebooks
            else {
                throw ModelFactoryError.invalidConfiguration("MaskGIT must reveal every codebook")
            }
        }
        if modelType == "lfm3_omni" {
            guard let vision, vision.mergeSize == downsample,
                imageToken >= 0, videoToken >= 0,
                imageToken < text.vocabularySize, videoToken < text.vocabularySize
            else {
                throw ModelFactoryError.invalidConfiguration(
                    "LFM3-Omni requires matching vision geometry")
            }
        }
    }
}

extension LFM3Configuration {
    var text: LFM3TextConfiguration { (textConfig ?? legacyText)! }
    var audio: LFM3AudioEncoderConfiguration { (audioConfig ?? legacyAudio)! }
    var depth: LFM3TextConfiguration { (depthConfig ?? legacyDepth)! }
    var schedule: [Int] {
        if let maskgitSchedule { return maskgitSchedule }
        var result = Array(repeating: numCodebooks / maskgitSteps, count: maskgitSteps)
        result[result.count - 1] += numCodebooks % maskgitSteps
        return result
    }
}
