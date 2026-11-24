import Foundation
import MLX
import MLXNN

struct EmbeddingGemma2TextConfiguration: Decodable, Sendable {
    let vocabularySize: Int
    let hiddenSize: Int
    let intermediateSize: Int
    let hiddenLayers: Int
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let perLayerInputSize: Int
    let embeddingDimension: Int
    let rmsNormEps: Float
    let slidingWindow: Int
    let maximumSequenceLength: Int
    let attentionBias: Bool
    let padTokenId: Int
    let layerTypes: [String]
    let perLayerConfiguration: [String: [String: Int]]
    let ropeParameters: [String: [String: StringOrNumber]]

    enum CodingKeys: String, CodingKey {
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case perLayerInputSize = "hidden_size_per_layer_input"
        case embeddingDimension = "embedding_dim"
        case rmsNormEps = "rms_norm_eps"
        case slidingWindow = "sliding_window"
        case maximumSequenceLength = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case padTokenId = "pad_token_id"
        case layerTypes = "layer_types"
        case perLayerConfiguration = "per_layer_config"
        case ropeParameters = "rope_parameters"
        case hiddenActivation = "hidden_activation"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 262144
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 512
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 2048
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 24
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 4
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 2
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 256
        perLayerInputSize = try c.decodeIfPresent(Int.self, forKey: .perLayerInputSize) ?? 512
        embeddingDimension = try c.decodeIfPresent(Int.self, forKey: .embeddingDimension) ?? 768
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 512
        maximumSequenceLength =
            try c.decodeIfPresent(Int.self, forKey: .maximumSequenceLength) ?? 262144
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        padTokenId = try c.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 0
        let activation =
            try c.decodeIfPresent(String.self, forKey: .hiddenActivation) ?? "gelu_pytorch_tanh"
        guard hiddenLayers > 0, hiddenSize > 0, intermediateSize > 0, vocabularySize > 0,
            perLayerInputSize >= 0, embeddingDimension > 0, maximumSequenceLength > 0,
            slidingWindow >= 0, rmsNormEps > 0, rmsNormEps.isFinite,
            (0 ..< vocabularySize).contains(padTokenId), activation == "gelu_pytorch_tanh"
        else {
            throw ModelExecutionError.invalidOptions("Invalid EmbeddingGemma 2 text configuration")
        }
        var types =
            try c.decodeIfPresent([String].self, forKey: .layerTypes)
            ?? (0 ..< hiddenLayers).map {
                ($0 + 1) % 6 == 0 ? "full_attention" : "sliding_attention"
            }
        guard types.count == hiddenLayers,
            types.allSatisfy({ ["full_attention", "sliding_attention"].contains($0) })
        else {
            throw ModelExecutionError.invalidOptions("Invalid EmbeddingGemma 2 layer types")
        }
        types[hiddenLayers - 1] = "full_attention"
        layerTypes = types
        let overrides =
            try c.decodeIfPresent([String: [String: Int]].self, forKey: .perLayerConfiguration)
            ?? Dictionary(
                uniqueKeysWithValues: types.enumerated().compactMap { index, kind in
                    kind == "full_attention"
                        ? (String(index), ["head_dim": 512, "num_key_value_heads": 1]) : nil
                })
        var mapped: [String: [String: Int]] = [:]
        for (key, value) in overrides {
            guard let index = Int(key), (0 ..< hiddenLayers).contains(index),
                mapped[String(index)] == nil
            else {
                throw ModelExecutionError.invalidOptions(
                    "Invalid EmbeddingGemma 2 per-layer configuration")
            }
            mapped[String(index)] = value
        }
        perLayerConfiguration = mapped
        ropeParameters =
            try c.decodeIfPresent([String: [String: StringOrNumber]].self, forKey: .ropeParameters)
            ?? [
                "full_attention": [
                    "rope_theta": .float(1_000_000), "rope_type": .string("default"),
                ],
                "sliding_attention": [
                    "rope_theta": .float(10_000), "rope_type": .string("default"),
                ],
            ]
        for index in 0 ..< hiddenLayers {
            let layer = mapped[String(index)] ?? [:]
            let heads = layer["num_attention_heads"] ?? attentionHeads
            let kv = layer["num_key_value_heads"] ?? kvHeads
            let dim = layer["head_dim"] ?? headDim
            let rope = ropeParameters[types[index]]
            let theta = rope?["rope_theta"]?.asFloat() ?? 0
            guard heads > 0, kv > 0, heads % kv == 0, dim > 0, dim % 2 == 0,
                theta > 0, theta.isFinite, rope?["rope_type"] == .string("default")
            else {
                throw ModelExecutionError.invalidOptions(
                    "Invalid EmbeddingGemma 2 attention configuration")
            }
        }
    }
}

struct EmbeddingGemma2Configuration: Decodable, Sendable {
    let text: EmbeddingGemma2TextConfiguration
    let vision: Gemma4VisionConfiguration?
    let audio: Gemma4AudioEncoderConfiguration?
    let imageTokenId: Int
    let videoTokenId: Int
    let audioTokenId: Int
    let beginImageTokenId: Int
    let endImageTokenId: Int
    let beginAudioTokenId: Int
    let endAudioTokenId: Int
    let visionSoftTokens: Int

    enum CodingKeys: String, CodingKey {
        case text = "text_config"
        case vision = "vision_config"
        case audio = "audio_config"
        case imageTokenId = "image_token_id"
        case videoTokenId = "video_token_id"
        case audioTokenId = "audio_token_id"
        case beginImageTokenId = "boi_token_id"
        case endImageTokenId = "eoi_token_id"
        case beginAudioTokenId = "boa_token_id"
        case endAudioTokenId = "eoa_token_index"
        case visionSoftTokens = "vision_soft_tokens_per_image"
    }

    init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        if c.contains(.text) {
            text = try c.decode(EmbeddingGemma2TextConfiguration.self, forKey: .text)
        } else {
            text = try EmbeddingGemma2TextConfiguration(from: decoder)
        }
        vision = try c.decodeIfPresent(Gemma4VisionConfiguration.self, forKey: .vision)
        audio = try c.decodeIfPresent(Gemma4AudioEncoderConfiguration.self, forKey: .audio)
        imageTokenId = try c.decodeIfPresent(Int.self, forKey: .imageTokenId) ?? 258880
        videoTokenId = try c.decodeIfPresent(Int.self, forKey: .videoTokenId) ?? 258884
        audioTokenId = try c.decodeIfPresent(Int.self, forKey: .audioTokenId) ?? 258881
        beginImageTokenId = try c.decodeIfPresent(Int.self, forKey: .beginImageTokenId) ?? 255999
        endImageTokenId = try c.decodeIfPresent(Int.self, forKey: .endImageTokenId) ?? 258882
        beginAudioTokenId = try c.decodeIfPresent(Int.self, forKey: .beginAudioTokenId) ?? 256000
        endAudioTokenId = try c.decodeIfPresent(Int.self, forKey: .endAudioTokenId) ?? 258883
        visionSoftTokens = try c.decodeIfPresent(Int.self, forKey: .visionSoftTokens) ?? 280
    }
}

final class EmbeddingGemma2Attention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let theta: Float
    let eps: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: EmbeddingGemma2TextConfiguration, index: Int) {
        let overrides = config.perLayerConfiguration[String(index)] ?? [:]
        heads = overrides["num_attention_heads"] ?? config.attentionHeads
        kvHeads = overrides["num_key_value_heads"] ?? config.kvHeads
        headDim = overrides["head_dim"] ?? config.headDim
        theta = config.ropeParameters[config.layerTypes[index]]!["rope_theta"]!.asFloat()!
        eps = config.rmsNormEps
        _qProj.wrappedValue = Linear(config.hiddenSize, heads * headDim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(heads * headDim, config.hiddenSize, bias: config.attentionBias)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
    }

    func rotary(_ x: MLXArray, positions: MLXArray) -> MLXArray {
        // Preserve FP32 angles and the reference's activation-dtype cosine/sine rounding.
        let frequencies =
            1 / pow(MLXArray(theta), arange(0, headDim, step: 2).asType(.float32) / Float(headDim))
        let angles = positions.asType(.float32).expandedDimensions(axis: -1) * frequencies
        let doubled = concatenated([angles, angles], axis: -1).expandedDimensions(axis: 1)
        let half = headDim / 2
        let rotated = concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
        return x * cos(doubled).asType(x.dtype) + rotated * sin(doubled).asType(x.dtype)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, positions: MLXArray) -> MLXArray {
        let b = x.dim(0)
        let n = x.dim(1)
        let q = rotary(
            qNorm(qProj(x).reshaped(b, n, heads, headDim)).transposed(0, 2, 1, 3),
            positions: positions)
        let k = rotary(
            kNorm(kProj(x).reshaped(b, n, kvHeads, headDim)).transposed(0, 2, 1, 3),
            positions: positions)
        let v = MLXFast.rmsNorm(
            vProj(x).reshaped(b, n, kvHeads, headDim), weight: .mlxNone, eps: eps
        ).transposed(0, 2, 1, 3)
        let output = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1, mask: .array(mask))
        return oProj(output.transposed(0, 2, 1, 3).reshaped(b, n, -1))
    }
}

final class EmbeddingGemma2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ config: EmbeddingGemma2TextConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(geluApproximate(gate(x)) * up(x)) }
}

final class EmbeddingGemma2PerLayerInputs: Module, UnaryLayer {
    let layers: Int
    let width: Int
    let scale: Float
    @ModuleInfo(key: "per_layer_model_projection") var projection: Linear
    @ModuleInfo(key: "per_layer_projection_norm") var norm: RMSNorm
    init(_ config: EmbeddingGemma2TextConfiguration) {
        layers = config.hiddenLayers
        width = config.perLayerInputSize
        scale = pow(Float(config.hiddenSize), -0.5)
        _projection.wrappedValue = Linear(config.hiddenSize, layers * width, bias: false)
        _norm.wrappedValue = RMSNorm(dimensions: width, eps: config.rmsNormEps)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        norm((projection(x) * scale).reshaped(x.dim(0), x.dim(1), layers, width))
    }
}

final class EmbeddingGemma2PerLayerBlock: Module {
    @ModuleInfo(key: "per_layer_input_gate") var gate: Linear
    @ModuleInfo(key: "per_layer_projection") var projection: Linear
    @ModuleInfo(key: "post_per_layer_input_norm") var norm: RMSNorm
    init(_ config: EmbeddingGemma2TextConfiguration) {
        _gate.wrappedValue = Linear(config.hiddenSize, config.perLayerInputSize, bias: false)
        _projection.wrappedValue = Linear(config.perLayerInputSize, config.hiddenSize, bias: false)
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }
    func callAsFunction(_ x: MLXArray, input: MLXArray) -> MLXArray {
        x + norm(projection(geluApproximate(gate(x)) * input))
    }
}

final class EmbeddingGemma2EncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: EmbeddingGemma2Attention
    let mlp: EmbeddingGemma2MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var feedforwardNorm: RMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var outputNorm: RMSNorm
    @ModuleInfo(key: "ple_block") var perLayer: EmbeddingGemma2PerLayerBlock?
    @ParameterInfo(key: "layer_scalar") var scalar: MLXArray
    init(_ config: EmbeddingGemma2TextConfiguration, index: Int) {
        _attention.wrappedValue = EmbeddingGemma2Attention(config, index: index)
        mlp = EmbeddingGemma2MLP(config)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _attentionNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _feedforwardNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _outputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        if config.perLayerInputSize > 0 {
            _perLayer.wrappedValue = EmbeddingGemma2PerLayerBlock(config)
        }
        _scalar.wrappedValue = MLXArray.ones([1])
    }
    func callAsFunction(
        _ x: MLXArray, mask: MLXArray, positions: MLXArray, perLayerInput: MLXArray?
    ) -> MLXArray {
        var h = x + attentionNorm(attention(inputNorm(x), mask: mask, positions: positions))
        h = h + outputNorm(mlp(feedforwardNorm(h)))
        if let perLayer, let perLayerInput { h = perLayer(h, input: perLayerInput) }
        return h * scalar
    }
}

final class EmbeddingGemma2TextModel: Module {
    let config: EmbeddingGemma2TextConfiguration
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let ple: EmbeddingGemma2PerLayerInputs?
    let layers: [EmbeddingGemma2EncoderLayer]
    let norm: RMSNorm
    @ModuleInfo(key: "embedding_projection") var projection: Linear
    init(_ config: EmbeddingGemma2TextConfiguration) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        ple = config.perLayerInputSize > 0 ? EmbeddingGemma2PerLayerInputs(config) : nil
        layers = (0 ..< config.hiddenLayers).map { EmbeddingGemma2EncoderLayer(config, index: $0) }
        norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _projection.wrappedValue = Linear(config.hiddenSize, config.embeddingDimension, bias: false)
    }
    func callAsFunction(_ embeddings: MLXArray, valid: MLXArray, positions: MLXArray? = nil) throws
        -> MLXArray
    {
        let length = embeddings.dim(1)
        let indices = arange(length)
        let positions = positions ?? indices.expandedDimensions(axis: 0)
        guard positions.ndim == 2, positions.dim(1) == length,
            positions.dim(0) == 1 || positions.dim(0) == embeddings.dim(0)
        else { throw ModelExecutionError.invalidOptions("Invalid EmbeddingGemma 2 positions") }
        let full = embeddingAttentionMask(valid)
        let sliding =
            full .&& (abs(indices.expandedDimensions(axis: 1) - indices) .<= config.slidingWindow)
        let inputs = ple?(embeddings)
        var h = embeddings
        for (index, layer) in layers.enumerated() {
            try Task.checkCancellation()
            h = layer(
                h, mask: config.layerTypes[index] == "sliding_attention" ? sliding : full,
                positions: positions, perLayerInput: inputs?[0..., 0..., index, 0...])
        }
        return projection(norm(h))
    }
}

/// EmbeddingGemma 2 projects bidirectional text and Gemma 4 media features into one retrieval space.
final class EmbeddingGemma2Model: Module, EmbeddingModel {
    let config: EmbeddingGemma2Configuration
    @ModuleInfo(key: "language_model") var languageModel: EmbeddingGemma2TextModel
    @ModuleInfo(key: "vision_tower") var visionTower: Gemma4VisionModel?
    @ModuleInfo(key: "embed_vision") var embedVision: Gemma4MultimodalEmbedder?
    @ModuleInfo(key: "audio_tower") var audioTower: Gemma4AudioEncoder?
    @ModuleInfo(key: "embed_audio") var embedAudio: Gemma4MultimodalEmbedder?
    var vocabularySize: Int { config.text.vocabularySize }
    var maximumSequenceLength: Int { config.text.maximumSequenceLength }

    init(_ config: EmbeddingGemma2Configuration) throws {
        self.config = config
        _languageModel.wrappedValue = EmbeddingGemma2TextModel(config.text)
        if let vision = config.vision {
            guard vision.hiddenSize > 0, vision.hiddenLayers > 0, vision.intermediateSize > 0,
                vision.attentionHeads > 0, vision.keyValueHeads > 0,
                vision.attentionHeads % vision.keyValueHeads == 0, vision.headDim > 0,
                vision.headDim % 4 == 0, vision.patchSize > 0, vision.poolingKernelSize > 0,
                vision.positionEmbeddingSize > 0
            else {
                throw ModelExecutionError.invalidOptions(
                    "Invalid EmbeddingGemma 2 vision configuration")
            }
            _visionTower.wrappedValue = Gemma4VisionModel(config: vision)
            _embedVision.wrappedValue = Gemma4MultimodalEmbedder(
                embeddingDim: vision.hiddenSize, textHiddenSize: config.text.hiddenSize,
                eps: vision.rmsNormEps)
        }
        if let audio = config.audio {
            guard audio.hiddenSize > 0, audio.hiddenLayers > 0, audio.attentionHeads > 0,
                audio.hiddenSize % audio.attentionHeads == 0,
                audio.subsamplingConvChannels.count == 2,
                audio.subsamplingConvChannels.allSatisfy({ $0 > 0 }), audio.inputFeatureSize > 0,
                audio.attentionChunkSize > 0, audio.convKernelSize > 0,
                audio.attentionContextLeft >= 0, audio.attentionContextRight >= 0,
                audio.rmsNormEps > 0, audio.rmsNormEps.isFinite,
                audio.attentionLogitCap > 0, audio.attentionLogitCap.isFinite,
                audio.gradientClipping > 0, audio.gradientClipping.isFinite,
                audio.residualWeight.isFinite, audio.hiddenActivation == "silu",
                (audio.outputProjectionDimensions ?? audio.hiddenSize) > 0
            else {
                throw ModelExecutionError.invalidOptions(
                    "Invalid EmbeddingGemma 2 audio configuration")
            }
            _audioTower.wrappedValue = Gemma4AudioEncoder(config: audio)
            _embedAudio.wrappedValue = Gemma4MultimodalEmbedder(
                embeddingDim: audio.outputProjectionDimensions ?? audio.hiddenSize,
                textHiddenSize: config.text.hiddenSize, eps: audio.rmsNormEps)
        }
    }

    func imageFeatures(pixels: MLXArray, frames: [THW]?) throws -> MLXArray {
        guard let visionTower, let embedVision, pixels.ndim == 4, pixels.dim(1) == 3 else {
            throw ModelExecutionError.invalidOptions(
                "EmbeddingGemma 2 images and video require a vision tower and NCHW pixels")
        }
        let sizes =
            frames ?? Array(repeating: THW(1, pixels.dim(2), pixels.dim(3)), count: pixels.dim(0))
        guard sizes.count == pixels.dim(0), !sizes.isEmpty else {
            throw ModelExecutionError.invalidOptions("Invalid EmbeddingGemma 2 image sizes")
        }
        var features: [MLXArray] = []
        for (index, size) in sizes.enumerated() {
            let multiple = visionTower.patchSize * visionTower.poolingKernelSize
            guard size.h > 0, size.w > 0, size.h <= pixels.dim(2), size.w <= pixels.dim(3),
                size.h % multiple == 0, size.w % multiple == 0,
                size.h / visionTower.patchSize <= visionTower.config.positionEmbeddingSize,
                size.w / visionTower.patchSize <= visionTower.config.positionEmbeddingSize
            else { throw ModelExecutionError.invalidOptions("Invalid EmbeddingGemma 2 patch grid") }
            try Task.checkCancellation()
            let encoded = visionTower(pixels[index ..< (index + 1), 0..., ..<size.h, ..<size.w])
            let projected = embedVision(encoded).squeezed(axis: 0)
            features.append(projected)
        }
        return concatenated(features, axis: 0)
    }

    func audioFeatures(_ audio: LMInput.ProcessedAudio) throws -> MLXArray {
        guard let audioTower, let embedAudio, audio.features.ndim == 3,
            audio.features.dim(0) > 0, audio.features.dim(1) > 0,
            audio.features.dim(2) == audioTower.inputFeatureSize
        else {
            throw ModelExecutionError.invalidOptions(
                "EmbeddingGemma 2 audio requires an audio tower and mel features")
        }
        let shape = Array(audio.features.shape.prefix(2))
        let valid = audio.mask ?? MLXArray.ones(shape, dtype: .bool)
        guard valid.shape == shape, all((valid .== 0) .|| (valid .== 1)).item(Bool.self) else {
            throw ModelExecutionError.invalidOptions("Invalid EmbeddingGemma 2 audio mask")
        }
        // This encoder uses the upstream tower's strict context horizon boundaries.
        let chunk = audioTower.config.attentionChunkSize
        let past = max(0, audioTower.config.attentionContextLeft - 1)
        let future = audioTower.config.attentionContextRight
        let distance =
            arange(chunk).expandedDimensions(axis: 1) + past - arange(chunk + past + future)
        let contextMask =
            ((distance .>= 0) .&& (distance .< past))
            .|| ((distance .< 0) .&& (-distance .< future))
        let (encoded, invalid) = audioTower(
            audio.features, mask: .!valid.asType(.bool), causalValidMask: contextMask)
        let features = embedAudio(encoded)
        var rows: [MLXArray] = []
        for index in 0 ..< features.dim(0) {
            let count = sum((.!invalid[index]).asType(.int32)).item(Int.self)
            if count > 0 { rows.append(features[index, ..<count, 0...]) }
        }
        guard !rows.isEmpty else {
            throw ModelExecutionError.invalidOptions("EmbeddingGemma 2 audio has no valid features")
        }
        return concatenated(rows, axis: 0)
    }

    func scatter(_ embeddings: MLXArray, ids: MLXArray, token: Int, features: MLXArray?) throws
        -> MLXArray
    {
        let mask = ids .== token
        let count = sum(mask.asType(.int32)).item(Int.self)
        guard count == (features?.dim(0) ?? 0) else {
            throw ModelExecutionError.invalidOptions(
                "EmbeddingGemma 2 media token count does not match feature count")
        }
        guard let features, count > 0 else { return embeddings }
        let indices = maximum(cumsum(mask.flattened().asType(.int32)) - 1, 0)
        let gathered = take(features, indices, axis: 0)
        let aligned = gathered.reshaped(embeddings.shape).asType(embeddings.dtype)
        return which(mask.expandedDimensions(axis: -1), aligned, embeddings)
    }

    func embed(_ input: LMInput) throws -> EmbeddingOutput {
        // The shared validator deliberately rejects audio for other encoder families.
        let textInput = LMInput(text: input.text)
        let (ids, valid) = try embeddingTokens(
            textInput, vocabularySize: vocabularySize, maximumLength: maximumSequenceLength)
        let media =
            (ids .== config.imageTokenId) .|| (ids .== config.videoTokenId)
            .|| (ids .== config.audioTokenId)
        var embeddings = languageModel.embedTokens(
            which(media, MLXArray(config.text.padTokenId), ids))
        embeddings =
            embeddings * MLXArray(sqrt(Float(config.text.hiddenSize)), dtype: embeddings.dtype)
        let image = try input.image.map { try imageFeatures(pixels: $0.pixels, frames: $0.frames) }
        let video = try input.video.map { try imageFeatures(pixels: $0.pixels, frames: $0.frames) }
        let audio = try input.audio.map { try audioFeatures($0) }
        embeddings = try scatter(embeddings, ids: ids, token: config.imageTokenId, features: image)
        embeddings = try scatter(embeddings, ids: ids, token: config.videoTokenId, features: video)
        embeddings = try scatter(embeddings, ids: ids, token: config.audioTokenId, features: audio)
        let hidden = try languageModel(embeddings, valid: valid)
        return .pooled(normalizeEmbeddings(EmbeddingPooling.mean.apply(hidden, mask: valid)))
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (original, value) in weights {
            let key = original.hasPrefix("model.") ? String(original.dropFirst(6)) : original
            if key.contains("rotary_emb.inv_freq") { continue }
            if key.hasPrefix("audio_tower."), config.audio?.useClippedLinears == false,
                ["input_min", "input_max", "output_min", "output_max"].contains(where: {
                    key.hasSuffix($0)
                })
            {
                continue
            }
            if visionTower == nil
                && (key.hasPrefix("vision_tower.") || key.hasPrefix("embed_vision."))
            {
                continue
            }
            if audioTower == nil && (key.hasPrefix("audio_tower.") || key.hasPrefix("embed_audio."))
            {
                continue
            }
            var tensor = value
            if key.contains("subsample_conv_projection"), key.hasSuffix("conv.weight"),
                tensor.ndim == 4
            {
                let channels =
                    key.contains(".layer0.") ? 1 : config.audio!.subsamplingConvChannels[0]
                if tensor.dim(-1) != channels { tensor = tensor.transposed(0, 2, 3, 1) }
            }
            if key.hasSuffix("depthwise_conv1d.weight"), tensor.ndim == 3, tensor.dim(-1) != 1 {
                tensor = tensor.transposed(0, 2, 1)
            }
            result[key] = tensor
        }
        return result
    }
}
