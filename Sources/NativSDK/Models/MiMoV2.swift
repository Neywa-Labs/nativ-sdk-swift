import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct MiMoV2TextConfiguration: Decodable, Sendable {
    private let _modelType: String?
    private let _vocabularySize: Int?
    private let _hiddenSize: Int?
    private let _intermediateSize: Int?
    private let _moeIntermediateSize: Int?
    private let _hiddenLayers: Int?
    private let _attentionHeads: Int?
    private let _kvHeads: Int?
    private let _headDim: Int?
    private let _vHeadDim: Int?
    private let _swaHeadDim: Int?
    private let _swaVHeadDim: Int?
    private let _swaAttentionHeads: Int?
    private let _swaKvHeads: Int?
    private let _ropeTheta: Float?
    private let _swaRopeTheta: Float?
    private let _partialRotaryFactor: Float?
    private let _maxPositionEmbeddings: Int?
    private let _layernormEpsilon: Float?
    private let _slidingWindowSize: Int?
    private let _hybridLayerPattern: [Int]?
    private let _moeLayerFreq: [Int]?
    private let _addSwaAttentionSinkBias: Bool?
    private let _addFullAttentionSinkBias: Bool?
    private let _numExpertsPerTok: Int?
    private let _topkMethod: String?
    private let _scoringFunc: String?
    private let _normTopkProb: Bool?
    private let _nGroup: Int?
    private let _topkGroup: Int?
    private let _attentionValueScale: Float?
    public let nRoutedExperts: Int?
    public let nSharedExperts: Int?
    public let routedScalingFactor: Float?

    public var modelType: String { _modelType ?? "mimo_v2" }
    public var vocabularySize: Int { _vocabularySize ?? 152_576 }
    public var hiddenSize: Int { _hiddenSize ?? 4096 }
    public var hiddenLayers: Int { _hiddenLayers ?? 48 }
    public var attentionValueScale: Float { _attentionValueScale ?? 0.707 }
    public var hybridLayerPattern: [Int] { _hybridLayerPattern ?? [] }
    public var moeLayerFreq: [Int] { _moeLayerFreq ?? [] }

    enum CodingKeys: String, CodingKey {
        case _modelType = "model_type"
        case _vocabularySize = "vocab_size"
        case _hiddenSize = "hidden_size"
        case _intermediateSize = "intermediate_size"
        case _moeIntermediateSize = "moe_intermediate_size"
        case _hiddenLayers = "num_hidden_layers"
        case _attentionHeads = "num_attention_heads"
        case _kvHeads = "num_key_value_heads"
        case _headDim = "head_dim"
        case _vHeadDim = "v_head_dim"
        case _swaHeadDim = "swa_head_dim"
        case _swaVHeadDim = "swa_v_head_dim"
        case _swaAttentionHeads = "swa_num_attention_heads"
        case _swaKvHeads = "swa_num_key_value_heads"
        case _ropeTheta = "rope_theta"
        case _swaRopeTheta = "swa_rope_theta"
        case _partialRotaryFactor = "partial_rotary_factor"
        case _maxPositionEmbeddings = "max_position_embeddings"
        case _layernormEpsilon = "layernorm_epsilon"
        case _slidingWindowSize = "sliding_window_size"
        case _hybridLayerPattern = "hybrid_layer_pattern"
        case _moeLayerFreq = "moe_layer_freq"
        case _addSwaAttentionSinkBias = "add_swa_attention_sink_bias"
        case _addFullAttentionSinkBias = "add_full_attention_sink_bias"
        case _numExpertsPerTok = "num_experts_per_tok"
        case _topkMethod = "topk_method"
        case _scoringFunc = "scoring_func"
        case _normTopkProb = "norm_topk_prob"
        case _nGroup = "n_group"
        case _topkGroup = "topk_group"
        case _attentionValueScale = "attention_value_scale"
        case nRoutedExperts = "n_routed_experts"
        case nSharedExperts = "n_shared_experts"
        case routedScalingFactor = "routed_scaling_factor"
    }

    /// The same hyperparameters in the shape MiMo-V2-Flash's decoder stack takes. V2.6
    /// reuses that stack unchanged and differs only in how its checkpoint is stored.
    public var flashConfiguration: MiMoV2FlashConfiguration {
        MiMoV2FlashConfiguration(
            modelType: modelType,
            numExpertsPerTok: _numExpertsPerTok ?? 8,
            hybridLayerPattern: hybridLayerPattern,
            moeLayerFreq: moeLayerFreq,
            addSwaAttentionSinkBias: _addSwaAttentionSinkBias ?? true,
            addFullAttentionSinkBias: _addFullAttentionSinkBias ?? false,
            slidingWindowSize: _slidingWindowSize ?? 128,
            vocabularySize: vocabularySize,
            hiddenSize: hiddenSize,
            intermediateSize: _intermediateSize ?? 16384,
            moeIntermediateSize: _moeIntermediateSize ?? 2048,
            hiddenLayers: hiddenLayers,
            attentionHeads: _attentionHeads ?? 64,
            kvHeads: _kvHeads ?? 4,
            nSharedExperts: nSharedExperts,
            nRoutedExperts: nRoutedExperts ?? 256,
            routedScalingFactor: routedScalingFactor,
            topkMethod: _topkMethod ?? "noaux_tc",
            scoringFunc: _scoringFunc ?? "sigmoid",
            normTopkProb: _normTopkProb ?? true,
            nGroup: _nGroup ?? 1,
            topkGroup: _topkGroup ?? 1,
            maxPositionEmbeddings: _maxPositionEmbeddings ?? 1_048_576,
            layernormEpsilon: _layernormEpsilon ?? 1e-6,
            ropeTheta: _ropeTheta ?? 10_000_000,
            swaRopeTheta: _swaRopeTheta ?? 10000,
            swaAttentionHeads: _swaAttentionHeads ?? 64,
            swaKvHeads: _swaKvHeads ?? 8,
            headDim: _headDim ?? 192,
            vHeadDim: _vHeadDim ?? 128,
            swaHeadDim: _swaHeadDim ?? 192,
            swaVHeadDim: _swaVHeadDim ?? 128,
            partialRotaryFactor: _partialRotaryFactor ?? 0.334,
            attentionValueScale: attentionValueScale)
    }
}

public struct MiMoV2VisionConfiguration: Decodable, Sendable {
    private let _modelType: String?
    private let _depth: Int?
    private let _hiddenSize: Int?
    private let _intermediateSize: Int?
    private let _outHiddenSize: Int?
    private let _numHeads: Int?
    private let _kvHeads: Int?
    private let _inChannels: Int?
    private let _inChans: Int?
    private let _patchSize: Int?
    private let _spatialMergeSize: Int?
    private let _temporalPatchSize: Int?
    private let _qkChannels: Int?
    private let _rmsNormEps: Float?
    private let _useSink: Bool?
    private let _visualTokenWindowSize: Int?
    private let _fullAttentionBlockIndexes: [Int]?
    private let _windowAttentionTypes: [Int]?

    public var modelType: String { _modelType ?? "mimovl" }
    public var depth: Int { _depth ?? 28 }
    public var hiddenSize: Int { _hiddenSize ?? 1280 }
    public var intermediateSize: Int { _intermediateSize ?? 4608 }
    public var outHiddenSize: Int { _outHiddenSize ?? 4096 }
    public var numHeads: Int { _numHeads ?? 32 }
    public var kvHeads: Int { _kvHeads ?? 8 }
    /// Released configs spell this `in_chans`; upstream accepts either.
    public var inChannels: Int { _inChannels ?? _inChans ?? 3 }
    public var patchSize: Int { _patchSize ?? 16 }
    public var spatialMergeSize: Int { _spatialMergeSize ?? 2 }
    public var temporalPatchSize: Int { _temporalPatchSize ?? 2 }
    /// Released configs omit this; qkv is 3072 = 48 * 64, not 1280 / 32.
    public var qkChannels: Int { _qkChannels ?? 64 }
    public var rmsNormEps: Float { _rmsNormEps ?? 1e-6 }
    public var useSink: Bool { _useSink ?? true }
    public var visualTokenWindowSize: Int { _visualTokenWindowSize ?? 64 }
    public var fullAttentionBlockIndexes: [Int] { _fullAttentionBlockIndexes ?? [0, 9, 18, 27] }

    /// One entry per block: 1 selects the column-major window ordering, anything else
    /// leaves the row-major one. Absent, no block reorders.
    public var windowAttentionTypes: [Int] {
        if let declared = _windowAttentionTypes, !declared.isEmpty { return declared }
        return Array(repeating: -1, count: depth)
    }

    enum CodingKeys: String, CodingKey {
        case _modelType = "model_type"
        case _depth = "depth"
        case _hiddenSize = "hidden_size"
        case _intermediateSize = "intermediate_size"
        case _outHiddenSize = "out_hidden_size"
        case _numHeads = "num_heads"
        case _kvHeads = "num_key_value_heads"
        case _inChannels = "in_channels"
        case _inChans = "in_chans"
        case _patchSize = "patch_size"
        case _spatialMergeSize = "spatial_merge_size"
        case _temporalPatchSize = "temporal_patch_size"
        case _qkChannels = "qk_channels"
        case _rmsNormEps = "rms_norm_eps"
        case _useSink = "use_sink"
        case _visualTokenWindowSize = "visual_token_window_size"
        case _fullAttentionBlockIndexes = "fullatt_block_indexes"
        case _windowAttentionTypes = "vit_window_attn_types"
    }
}

public struct MiMoV2AudioConfiguration: Decodable, Sendable {
    private let _outHiddenSize: Int?
    private let _audioChannels: Int?
    private let _groupSize: Int?
    private let _inputFullAttention: Bool?
    private let _localAttentionHeads: Int?
    private let _localDimensions: Int?
    private let _localIntermediateSize: Int?
    private let _localLayers: Int?
    private let _addPostNorm: Bool?
    private let _projectionLayers: Int?
    private let _ropeTheta: Float?
    private let _speechVocabularySize: StringOrNumber?
    private let _speechZeroEmbeddingIndex: StringOrNumber?

    public var outHiddenSize: Int { _outHiddenSize ?? 4096 }
    public var audioChannels: Int { _audioChannels ?? 20 }
    public var groupSize: Int { _groupSize ?? 4 }
    public var inputFullAttention: Bool { _inputFullAttention ?? true }
    public var localAttentionHeads: Int { _localAttentionHeads ?? 16 }
    public var localDimensions: Int { _localDimensions ?? 1024 }
    public var localIntermediateSize: Int { _localIntermediateSize ?? 4096 }
    public var localLayers: Int { _localLayers ?? 6 }
    public var addPostNorm: Bool { _addPostNorm ?? true }
    public var projectionLayers: Int { _projectionLayers ?? 2 }
    public var ropeTheta: Float { _ropeTheta ?? 640_000 }

    /// Released configs ship these two as strings, so a plain `asInt()` would read
    /// neither and leave the default standing. An undersized speech table then indexes
    /// out of bounds, which MLX answers with zeros rather than an error.
    public var speechVocabularySize: Int {
        MiMoV2AudioConfiguration.integer(_speechVocabularySize) ?? 1280
    }
    public var speechZeroEmbeddingIndex: Int {
        MiMoV2AudioConfiguration.integer(_speechZeroEmbeddingIndex) ?? 1024
    }

    static func integer(_ value: StringOrNumber?) -> Int? {
        guard let value else { return nil }
        if let direct = value.asInt() { return direct }
        if case .string(let text) = value { return Int(text) }
        return nil
    }

    enum CodingKeys: String, CodingKey {
        case _outHiddenSize = "out_hidden_size"
        case _audioChannels = "audio_channels"
        case _groupSize = "group_size"
        case _inputFullAttention = "input_full_attention"
        case _localAttentionHeads = "input_local_attn_heads"
        case _localDimensions = "input_local_dim"
        case _localIntermediateSize = "input_local_intermediate_size"
        case _localLayers = "input_local_layers"
        case _addPostNorm = "add_post_norm"
        case _projectionLayers = "projection_layers"
        case _ropeTheta = "rope_theta"
        case _speechVocabularySize = "speech_vocab_size"
        case _speechZeroEmbeddingIndex = "speech_zeroemb_idx"
    }
}

public struct MiMoV2Configuration: Decodable, Sendable, ModelConfigurationValidating {
    public let textConfiguration: MiMoV2TextConfiguration
    public let visionConfiguration: MiMoV2VisionConfiguration
    public let audioConfiguration: MiMoV2AudioConfiguration
    private let _modelType: String?
    private let _imageTokenId: Int?
    private let _videoTokenId: Int?
    private let _audioTokenId: Int?
    private let _skipVision: Bool?

    public var modelType: String { _modelType ?? "mimo_v2" }
    public var imageTokenId: Int { _imageTokenId ?? 151_655 }
    public var videoTokenId: Int { _videoTokenId ?? 151_656 }
    public var audioTokenId: Int { _audioTokenId ?? 151_669 }
    public var skipVision: Bool { _skipVision ?? false }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case audioConfiguration = "audio_config"
        case _modelType = "model_type"
        case _imageTokenId = "image_token_id"
        case _videoTokenId = "video_token_id"
        case _audioTokenId = "audio_token_id"
        case _skipVision = "skip_vision"
    }

    public init(from decoder: Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration =
            try container.decodeIfPresent(
                MiMoV2TextConfiguration.self, forKey: .textConfiguration)
            ?? MiMoV2TextConfiguration(from: decoder)
        visionConfiguration =
            try container.decodeIfPresent(
                MiMoV2VisionConfiguration.self, forKey: .visionConfiguration)
            ?? JSONDecoder().decode(MiMoV2VisionConfiguration.self, from: Data("{}".utf8))
        audioConfiguration =
            try container.decodeIfPresent(
                MiMoV2AudioConfiguration.self, forKey: .audioConfiguration)
            ?? JSONDecoder().decode(MiMoV2AudioConfiguration.self, from: Data("{}".utf8))
        _modelType = try container.decodeIfPresent(String.self, forKey: ._modelType)
        _imageTokenId = try container.decodeIfPresent(Int.self, forKey: ._imageTokenId)
        _videoTokenId = try container.decodeIfPresent(Int.self, forKey: ._videoTokenId)
        _audioTokenId = try container.decodeIfPresent(Int.self, forKey: ._audioTokenId)
        _skipVision = try container.decodeIfPresent(Bool.self, forKey: ._skipVision)
    }

    /// The decoder stack indexes both per-layer patterns, and MiMo-V2's text config
    /// defaults them to empty rather than deriving them.
    public func validateModelConfiguration() throws {
        let layers = textConfiguration.hiddenLayers
        for (name, pattern) in [
            ("hybrid_layer_pattern", textConfiguration.hybridLayerPattern),
            ("moe_layer_freq", textConfiguration.moeLayerFreq),
        ] where pattern.count != layers {
            throw MiMoV2ConfigurationError.layerPatternLength(name, pattern.count, layers)
        }
        if textConfiguration.attentionValueScale <= 0 {
            throw MiMoV2ConfigurationError.nonPositiveValueScale(
                textConfiguration.attentionValueScale)
        }
        let projectionLayers = audioConfiguration.projectionLayers
        guard projectionLayers == 1 || projectionLayers == 2 else {
            throw MiMoV2ConfigurationError.unsupportedProjectionLayers(projectionLayers)
        }
    }
}

public enum MiMoV2ConfigurationError: LocalizedError {
    case layerPatternLength(String, Int, Int)
    case nonPositiveValueScale(Float)
    case unsupportedProjectionLayers(Int)

    public var errorDescription: String? {
        switch self {
        case .layerPatternLength(let name, let given, let layers):
            "`\(name)` must have one entry per hidden layer, got \(given) for \(layers)"
        case .nonPositiveValueScale(let scale):
            "`attention_value_scale` must be positive, got \(scale)"
        case .unsupportedProjectionLayers(let count):
            "unsupported `projection_layers`=\(count)"
        }
    }
}

// MARK: - Vision

public enum MiMoV2Vision {
    /// Kernel and stride are equal, so the patch convolution is a linear map over
    /// flattened `in_channels x temporal_patch_size x patch_size^2` patches;
    /// ``VisionModel/sanitize(weights:)`` reshapes the checkpoint's 5D weight to match.
    class PatchEmbed: Module, UnaryLayer {
        let patchDimensions: Int
        @ModuleInfo(key: "proj") var proj: Linear

        init(_ config: MiMoV2VisionConfiguration) {
            self.patchDimensions =
                config.inChannels * config.temporalPatchSize * config.patchSize
                * config.patchSize
            _proj.wrappedValue = Linear(patchDimensions, config.hiddenSize, bias: false)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            proj(x.reshaped(-1, patchDimensions))
        }
    }

    class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gate: Linear
        @ModuleInfo(key: "up_proj") var up: Linear
        @ModuleInfo(key: "down_proj") var down: Linear

        init(_ config: MiMoV2VisionConfiguration) {
            _gate.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
            _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
            _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(silu(gate(x)) * up(x))
        }
    }

    class Attention: Module {
        let numHeads: Int
        let numKVHeads: Int
        let headDimensions: Int
        let scale: Float
        let windowSize: Int

        @ModuleInfo(key: "qkv") var qkv: Linear
        @ModuleInfo(key: "proj") var proj: Linear
        @ParameterInfo(key: "sinks") var sinks: MLXArray?

        init(_ config: MiMoV2VisionConfiguration, useSinks: Bool, windowSize: Int) {
            self.numHeads = config.numHeads
            self.numKVHeads = config.kvHeads
            self.headDimensions = config.qkChannels
            self.scale = pow(Float(config.qkChannels), -0.5)
            self.windowSize = windowSize

            let qkvDimensions = (numHeads + 2 * numKVHeads) * headDimensions
            _qkv.wrappedValue = Linear(config.hiddenSize, qkvDimensions)
            _proj.wrappedValue = Linear(numHeads * headDimensions, config.hiddenSize)
            _sinks.wrappedValue = useSinks ? MLXArray.zeros([numHeads]) : nil
        }

        func mask(_ n: Int, fullAttention: Bool, dtype: DType) -> MLXArray? {
            var mask: MLXArray?
            if !fullAttention, windowSize > 0 {
                let index = MLXArray(0 ..< n)
                let outside = abs(index[0..., .newAxis] - index[.newAxis, 0...]) .> windowSize
                mask =
                    MLX.where(
                        outside, MLXArray(-Float.infinity).asType(dtype),
                        MLXArray(Float(0)).asType(dtype))[.newAxis, .newAxis]
            }
            if let sinks {
                let sink = MLXArray.zeros([1, numHeads, 1, n], dtype: dtype)
                sink[.ellipsis, 0] = sinks.reshaped(1, numHeads, 1).asType(dtype)
                mask = mask.map { $0 + sink } ?? sink
            }
            return mask
        }

        func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, fullAttention: Bool)
            -> MLXArray
        {
            let (batch, n) = (q.dim(0), q.dim(1))
            let q = q.transposed(0, 2, 1, 3)
            var k = k.transposed(0, 2, 1, 3)
            var v = v.transposed(0, 2, 1, 3)
            let repeats = numHeads / numKVHeads
            if repeats > 1 {
                k = repeated(k, count: repeats, axis: 1)
                v = repeated(v, count: repeats, axis: 1)
            }
            let maskMode: MLXFast.ScaledDotProductAttentionMaskMode =
                mask(n, fullAttention: fullAttention, dtype: q.dtype)
                .map { .array($0) } ?? .none
            let out = MLXFast.scaledDotProductAttention(
                queries: q, keys: k, values: v, scale: scale, mask: maskMode)
            return out.transposed(0, 2, 1, 3).reshaped(batch, n, -1)
        }

        func callAsFunction(
            _ x: MLXArray, cos: MLXArray, sin: MLXArray, fullAttention: Bool,
            spans: [(Int, Int)]
        ) -> MLXArray {
            let n = x.dim(0)
            let fused = qkv(x)
            let qDimensions = numHeads * headDimensions
            let kvDimensions = numKVHeads * headDimensions

            var q = fused[0..., ..<qDimensions].reshaped(n, numHeads, headDimensions)
            var k = fused[0..., qDimensions ..< (qDimensions + kvDimensions)]
                .reshaped(n, numKVHeads, headDimensions)
            let v = fused[0..., (qDimensions + kvDimensions)...]
                .reshaped(n, numKVHeads, headDimensions)
            (q, k) = applyRotary(q, k, cos: cos, sin: sin)

            var groups: [Int: [(index: Int, start: Int, stop: Int)]] = [:]
            for (index, span) in spans.enumerated() {
                groups[span.1 - span.0, default: []].append((index, span.0, span.1))
            }
            var chunks = [MLXArray?](repeating: nil, count: spans.count)
            for items in groups.values {
                let outputs = attend(
                    stacked(items.map { q[$0.start ..< $0.stop] }),
                    stacked(items.map { k[$0.start ..< $0.stop] }),
                    stacked(items.map { v[$0.start ..< $0.stop] }),
                    fullAttention: fullAttention)
                for (position, item) in items.enumerated() {
                    chunks[item.index] = outputs[position]
                }
            }
            return proj(concatenated(chunks.map { $0! }, axis: 0))
        }
    }

    class Block: Module {
        @ModuleInfo(key: "norm1") var norm1: RMSNorm
        @ModuleInfo(key: "norm2") var norm2: RMSNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP

        init(_ config: MiMoV2VisionConfiguration, useSinks: Bool, windowSize: Int) {
            _norm1.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _norm2.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _attention.wrappedValue = Attention(
                config, useSinks: useSinks, windowSize: windowSize)
            _mlp.wrappedValue = MLP(config)
        }

        func callAsFunction(
            _ x: MLXArray, cos: MLXArray, sin: MLXArray, fullAttention: Bool,
            spans: [(Int, Int)]
        ) -> MLXArray {
            var x =
                x
                + attention(
                    norm1(x), cos: cos, sin: sin, fullAttention: fullAttention, spans: spans)
            x = x + mlp(norm2(x))
            return x
        }
    }

    class PatchMerger: Module, UnaryLayer {
        let hiddenSize: Int
        @ModuleInfo(key: "ln_q") var layerNormQ: LayerNorm
        @ModuleInfo var mlp: (Linear, GELU, Linear)

        init(_ config: MiMoV2VisionConfiguration) {
            self.hiddenSize =
                config.hiddenSize * config.spatialMergeSize * config.spatialMergeSize
            _layerNormQ.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: 1e-6, affine: true, bias: false)
            self.mlp = (
                Linear(hiddenSize, hiddenSize, bias: false),
                GELU(),
                Linear(hiddenSize, config.outHiddenSize, bias: false)
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var x = layerNormQ(x).reshaped(-1, hiddenSize)
            x = mlp.0(x)
            x = mlp.1(x)
            x = mlp.2(x)
            return x
        }
    }

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    }

    static func applyRotary(_ q: MLXArray, _ k: MLXArray, cos: MLXArray, sin: MLXArray)
        -> (MLXArray, MLXArray)
    {
        let dtype = q.dtype
        let cos = cos.expandedDimensions(axis: -2).asType(.float32)
        let sin = sin.expandedDimensions(axis: -2).asType(.float32)
        let q = q.asType(.float32)
        let k = k.asType(.float32)
        return (
            (q * cos + rotateHalf(q) * sin).asType(dtype),
            (k * cos + rotateHalf(k) * sin).asType(dtype)
        )
    }

    /// MiMo-V2.6 vision tower (`vision_model_type: mimovl`).
    public class VisionModel: Module {
        let config: MiMoV2VisionConfiguration
        let spatialMergeSize: Int
        let spatialMergeUnit: Int
        let fullAttentionBlockIndexes: Set<Int>
        let windowAttentionTypes: [Int]

        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        let rotaryPositionEmbedding: QwenVL.VisionRotaryEmbedding
        let blocks: [Block]
        @ModuleInfo(key: "merger") var merger: PatchMerger

        public init(_ config: MiMoV2VisionConfiguration) {
            self.config = config
            self.spatialMergeSize = config.spatialMergeSize
            self.spatialMergeUnit = config.spatialMergeSize * config.spatialMergeSize
            self.fullAttentionBlockIndexes = Set(config.fullAttentionBlockIndexes)
            self.windowAttentionTypes = config.windowAttentionTypes

            _patchEmbed.wrappedValue = PatchEmbed(config)
            self.rotaryPositionEmbedding = QwenVL.VisionRotaryEmbedding(
                dimensions: config.qkChannels / 2, theta: 10000)
            let full = Set(config.fullAttentionBlockIndexes)
            self.blocks = (0 ..< config.depth).map { index in
                Block(
                    config,
                    useSinks: config.useSink && !full.contains(index),
                    windowSize: config.visualTokenWindowSize)
            }
            _merger.wrappedValue = PatchMerger(config)
        }

        public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized: [String: MLXArray] = [:]
            for (key, value) in weights {
                if key.hasSuffix("patch_embed.proj.weight"), value.ndim == 5 {
                    sanitized[key] = value.reshaped(value.dim(0), -1)
                } else {
                    sanitized[key] = value
                }
            }
            return sanitized
        }

        func applyIndex(_ x: MLXArray, _ index: MLXArray) -> MLXArray {
            let tail = Array(x.shape.dropFirst())
            let grouped = x.reshaped([-1, spatialMergeUnit] + tail)
            return take(grouped, index, axis: 0).reshaped([-1] + tail)
        }

        /// Merge-unit blocks reordered column-major within each frame.
        func windowIndex(_ grid: [(Int, Int, Int)]) -> MLXArray {
            var parts: [MLXArray] = []
            var offset = 0
            let m = spatialMergeSize
            for (t, h, w) in grid {
                let (gh, gw) = (h / m, w / m)
                let index = MLXArray(0 ..< (t * gh * gw)).reshaped(t, gh, gw)
                parts.append(index.transposed(0, 2, 1).reshaped(-1) + offset)
                offset += t * gh * gw
            }
            return concatenated(parts, axis: 0)
        }

        func rotaryPositions(_ grid: [(Int, Int, Int)]) -> MLXArray {
            let m = spatialMergeSize
            var parts: [MLXArray] = []
            for (t, h, w) in grid {
                var hpos = broadcast(MLXArray(0 ..< h)[0..., .newAxis], to: [h, w])
                hpos = hpos.reshaped(h / m, m, w / m, m).transposed(0, 2, 1, 3).reshaped(-1)
                var wpos = broadcast(MLXArray(0 ..< w)[.newAxis, 0...], to: [h, w])
                wpos = wpos.reshaped(h / m, m, w / m, m).transposed(0, 2, 1, 3).reshaped(-1)
                parts.append(
                    tiled(stacked([hpos, wpos], axis: -1), repetitions: [t, 1]))
            }
            let positions = concatenated(parts, axis: 0)
            let maxGrid = grid.map { max($0.1, $0.2) }.max() ?? 1
            let table = rotaryPositionEmbedding(sequenceLength: maxGrid)
            return take(table, positions.reshaped(-1), axis: 0)
                .reshaped(positions.dim(0), -1)
        }

        public func callAsFunction(_ pixelValues: MLXArray, gridTHW: [(Int, Int, Int)])
            -> MLXArray
        {
            var x = patchEmbed(pixelValues)

            let rotary = rotaryPositions(gridTHW)
            let embedded = concatenated([rotary, rotary], axis: -1)
            let (rowCos, rowSin) = (cos(embedded), sin(embedded))
            let columnIndex = windowIndex(gridTHW)
            let reverseIndex = argSort(columnIndex, axis: -1)
            let columnEmbedded = applyIndex(embedded, columnIndex)
            let (columnCos, columnSin) = (cos(columnEmbedded), sin(columnEmbedded))

            var spans: [(Int, Int)] = []
            var total = 0
            for (t, h, w) in gridTHW {
                for _ in 0 ..< t {
                    spans.append((total, total + h * w))
                    total += h * w
                }
            }

            let types = windowAttentionTypes
            for (index, block) in blocks.enumerated() {
                if types[index] == 1, index == 0 || types[index - 1] != 1 {
                    x = applyIndex(x, columnIndex)
                } else if index > 0, types[index] != 1, types[index - 1] == 1 {
                    x = applyIndex(x, reverseIndex)
                }
                let useColumns = types[index] == 1
                x = block(
                    x, cos: useColumns ? columnCos : rowCos,
                    sin: useColumns ? columnSin : rowSin,
                    fullAttention: fullAttentionBlockIndexes.contains(index), spans: spans)
            }
            return merger(x)
        }
    }
}

// MARK: - Audio

public enum MiMoV2Audio {
    /// The local transformer is a Qwen2 stack, which is how upstream builds it; its
    /// hyperparameters are the audio config's `input_local_*` fields.
    class Transformer: Module {
        let layers: [Qwen2TransformerBlock]
        @ModuleInfo(key: "norm") var norm: RMSNorm?
        let fullAttention: Bool

        init(_ config: MiMoV2AudioConfiguration) {
            let args = MiMoV2Audio.qwen2Configuration(config)
            self.layers = (0 ..< config.localLayers).map { _ in Qwen2TransformerBlock(args) }
            _norm.wrappedValue =
                config.addPostNorm
                ? RMSNorm(dimensions: config.localDimensions, eps: 1e-6) : nil
            self.fullAttention = config.inputFullAttention
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let mask: MLXFast.ScaledDotProductAttentionMaskMode =
                fullAttention
                ? .none
                : .array(MultiHeadAttention.createAdditiveCausalMask(x.dim(1)))
            var x = x
            for layer in layers {
                x = layer(x, mask: mask, cache: nil)
            }
            return norm.map { $0(x) } ?? x
        }
    }

    class Projection: Module, UnaryLayer {
        @ModuleInfo var mlp: (Linear, GELU, Linear)

        init(inputSize: Int, outputSize: Int) {
            self.mlp = (
                Linear(inputSize, inputSize * 4, bias: false),
                GELU(),
                Linear(inputSize * 4, outputSize, bias: false)
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            mlp.2(mlp.1(mlp.0(x)))
        }
    }

    public class Encoder: Module {
        let config: MiMoV2AudioConfiguration
        @ModuleInfo(key: "input_local_transformer") var localTransformer: Transformer
        @ModuleInfo(key: "projection") var projection: UnaryLayer

        public init(_ config: MiMoV2AudioConfiguration) {
            self.config = config
            _localTransformer.wrappedValue = Transformer(config)
            let projectionInput = config.localDimensions * config.groupSize
            _projection.wrappedValue =
                config.projectionLayers == 1
                ? Linear(projectionInput, config.outHiddenSize, bias: false)
                : Projection(inputSize: projectionInput, outputSize: config.outHiddenSize)
        }

        public func callAsFunction(_ audioCodes: MLXArray, speechEmbeddings: [Embedding])
            -> MLXArray
        {
            precondition(
                audioCodes.ndim == 2,
                "audio_codes must be 2D, received shape \(audioCodes.shape)")
            precondition(audioCodes.dim(0) > 0, "audio_codes must contain at least one frame")
            precondition(
                audioCodes.dim(1) >= config.audioChannels,
                "audio_codes must contain at least \(config.audioChannels) channels")

            var codes = audioCodes[0..., ..<config.audioChannels]
            let length = codes.dim(0)
            let group = config.groupSize
            let padded = ((length + group - 1) / group) * group
            if padded > length {
                codes = concatenated(
                    [codes, repeated(codes[(length - 1)...], count: padded - length, axis: 0)],
                    axis: 0)
            }
            codes = codes.reshaped(-1, group, config.audioChannels)

            var embeddings = speechEmbeddings[0](codes[0..., 0..., 0])
            for channel in 1 ..< config.audioChannels {
                embeddings = embeddings + speechEmbeddings[channel](codes[0..., 0..., channel])
            }
            let hidden = localTransformer(embeddings)
            return projection(hidden.reshaped(hidden.dim(0), -1))
        }
    }

    static func speechEmbeddings(_ config: MiMoV2AudioConfiguration) -> [Embedding] {
        (0 ..< config.audioChannels).map { _ in
            Embedding(
                embeddingCount: config.speechVocabularySize,
                dimensions: config.localDimensions)
        }
    }

    /// The audio stack's Qwen2 hyperparameters. Qwen2's configuration is decode-only, so
    /// it is built through JSON rather than a memberwise initializer.
    static func qwen2Configuration(_ config: MiMoV2AudioConfiguration) -> Qwen2Configuration {
        let json = """
            {
              "model_type": "qwen2",
              "hidden_size": \(config.localDimensions),
              "num_hidden_layers": \(config.localLayers),
              "intermediate_size": \(config.localIntermediateSize),
              "num_attention_heads": \(config.localAttentionHeads),
              "num_key_value_heads": \(config.localAttentionHeads),
              "rms_norm_eps": 1e-6,
              "vocab_size": 1,
              "rope_theta": \(config.ropeTheta)
            }
            """
        do {
            return try JSONDecoder().decode(Qwen2Configuration.self, from: Data(json.utf8))
        } catch {
            preconditionFailure("audio transformer configuration is not decodable: \(error)")
        }
    }
}

// MARK: - Model

public final class MiMoV2: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "language_model") private var languageModel: MiMoV2FlashModel
    @ModuleInfo(key: "vision_tower") private var visionTower: MiMoV2Vision.VisionModel?
    @ModuleInfo(key: "audio_encoder") private var audioEncoder: MiMoV2Audio.Encoder
    @ModuleInfo(key: "speech_embeddings") private var speechEmbeddings: [Embedding]

    public let config: MiMoV2Configuration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: MiMoV2Configuration) {
        self.config = config
        _languageModel.wrappedValue = MiMoV2FlashModel(
            config.textConfiguration.flashConfiguration)
        _visionTower.wrappedValue =
            config.skipVision ? nil : MiMoV2Vision.VisionModel(config.visionConfiguration)
        _audioEncoder.wrappedValue = MiMoV2Audio.Encoder(config.audioConfiguration)
        _speechEmbeddings.wrappedValue = MiMoV2Audio.speechEmbeddings(
            config.audioConfiguration)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }

        let embeddings = try inputEmbeddings(
            inputIds: tokens, pixelValues: input.image?.pixels,
            gridTHW: MiMoV2.grid(from: input.image?.frames))
        let total = embeddings.dim(1)
        let processed = try prefill.forEachChunk(total: total) { range in
            _ = languageModel(
                tokens[0..., range], cache: cache, inputEmbedding: embeddings[0..., range, 0...])
            asyncEval(cache)
        }
        if processed > 0 { eval(cache) }
        prefill.progress?(total, total)
        return .logits(
            .init(
                logits: languageModel(
                    tokens[0..., processed...], cache: cache,
                    inputEmbedding: embeddings[0..., processed..., 0...])))
    }

    static func grid(from rows: [THW]?) -> [(Int, Int, Int)]? {
        rows.map { $0.map { ($0.t, $0.h, $0.w) } }
    }

    func inputEmbeddings(
        inputIds: MLXArray, pixelValues: MLXArray?, gridTHW: [(Int, Int, Int)]?
    ) throws -> MLXArray {
        let inputEmbedding = languageModel.model.embedTokens(inputIds)
        guard let pixelValues else { return inputEmbedding }
        guard let visionTower else { throw MiMoV2Error.visionDisabled }
        guard let gridTHW, !gridTHW.isEmpty else { throw MiMoV2Error.missingGrid }

        let dtype = visionTower.patchEmbed.proj.weight.dtype
        let features = visionTower(pixelValues.asType(dtype), gridTHW: gridTHW)
        return try MiMoV2.replaceModalEmbeddings(
            inputIds: inputIds, inputEmbedding: inputEmbedding, tokenId: config.imageTokenId,
            features: features)
    }

    public func encodeAudio(_ audioCodes: MLXArray) -> MLXArray {
        audioEncoder(audioCodes, speechEmbeddings: speechEmbeddings)
    }

    static func replaceModalEmbeddings(
        inputIds: MLXArray, inputEmbedding: MLXArray, tokenId: Int, features: MLXArray
    ) throws -> MLXArray {
        guard features.ndim == 2 else {
            throw MiMoV2Error.featureRank(features.shape)
        }
        guard features.dim(-1) == inputEmbedding.dim(-1) else {
            throw MiMoV2Error.featureWidth(features.dim(-1), inputEmbedding.dim(-1))
        }
        let ids: [Int32] = inputIds.flattened().asArray(Int32.self)
        let positions = ids.enumerated().compactMap { $1 == Int32(tokenId) ? $0 : nil }
        guard positions.count == features.dim(0) else {
            throw MiMoV2Error.placeholderCount(positions.count, features.dim(0))
        }
        if positions.isEmpty { return inputEmbedding }
        let flat = inputEmbedding.reshaped(-1, inputEmbedding.dim(-1))
        flat[MLXArray(positions.map(Int32.init))] = features.asType(inputEmbedding.dtype)
        return flat.reshaped(inputEmbedding.shape)
    }

    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        try languageModel.newCache(configuration: cacheConfiguration)
    }

    /// A converted checkpoint is already prefixed; a released one keeps the vision tower
    /// under `visual.` and the audio encoder at the top level.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        if weights.keys.contains(where: { $0.hasPrefix("language_model.") }) { return weights }

        var vision: [String: MLXArray] = [:]
        var audio: [String: MLXArray] = [:]
        var speech: [String: MLXArray] = [:]
        var language: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("audio_encoder.") {
                audio[String(key.dropFirst("audio_encoder.".count))] = value
            } else if key.hasPrefix("speech_embeddings.") {
                speech[key] = value
            } else if key.hasPrefix("visual.") {
                vision[String(key.dropFirst("visual.".count))] = value
            } else {
                language[key] = value
            }
        }

        var sanitized: [String: MLXArray] = [:]
        for (key, value) in sanitizeLanguage(language) {
            sanitized["language_model.\(key)"] = value
        }
        if let visionTower {
            for (key, value) in visionTower.sanitize(weights: vision) {
                sanitized["vision_tower.\(key)"] = value
            }
        }
        for (key, value) in audio {
            sanitized["audio_encoder.\(key)"] = value
        }
        sanitized.merge(speech) { first, _ in first }
        return sanitized
    }

    /// V2.6 stores the text half differently from V2-Flash: attention projections are
    /// fused into one `qkv_proj` and the routed experts ship as MXFP4. Both are undone
    /// here so the inherited modules load as they are.
    func sanitizeLanguage(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var weights = weights
        if weights.keys.contains(where: { $0.hasSuffix("weight_scale") }) {
            do {
                try quantizeByParent(model: self) { path, _ in
                    path.contains("switch_mlp") ? (groupSize: 32, bits: 4, mode: .mxfp4) : nil
                }
            } catch {
                preconditionFailure("the MXFP4 experts could not be quantized: \(error)")
            }
            weights = MiMoV2.convertMXFP4Experts(weights)
        }
        if weights.keys.contains(where: { $0.hasSuffix("self_attn.qkv_proj.weight") }) {
            do {
                weights = try unfuseQKV(weights)
            } catch {
                preconditionFailure("the fused qkv projections could not be split: \(error)")
            }
        }
        return languageModel.sanitize(weights: weights)
    }

    static func convertMXFP4Experts(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasSuffix(".weight_scale") { continue }
            guard let scale = weights["\(key)_scale"] else {
                out[key] = value
                continue
            }
            out[key] = value.view(dtype: .uint32)
            out[String(key.dropLast(".weight".count)) + ".scales"] = scale
        }
        return out
    }
}

public enum MiMoV2Error: LocalizedError {
    case visionDisabled
    case missingGrid
    case featureRank([Int])
    case featureWidth(Int, Int)
    case placeholderCount(Int, Int)
    case noTensorParallelDegree

    public var errorDescription: String? {
        switch self {
        case .visionDisabled:
            "cannot encode an image when vision is disabled"
        case .missingGrid:
            "a grid is required with pixel values"
        case .featureRank(let shape):
            "modal features must be 2D, received shape \(shape)"
        case .featureWidth(let features, let text):
            "modal feature width \(features) does not match text embedding width \(text)"
        case .placeholderCount(let positions, let features):
            "found \(positions) placeholder tokens for \(features) features"
        case .noTensorParallelDegree:
            "no tensor-parallel degree explains the fused qkv shapes"
        }
    }
}

extension MiMoV2: LoRAModel {
    public var loraLayers: [Module] { languageModel.model.layers }
}

// MARK: - Fused qkv

extension MiMoV2 {
    static let fp8BlockSize = 128

    static func layerIndex(of key: String) -> Int? {
        let parts = key.split(separator: ".")
        guard let position = parts.firstIndex(of: "layers"), position + 1 < parts.count else {
            return nil
        }
        return Int(parts[position + 1])
    }

    func qkvHeads(layer: Int) -> (heads: Int, kvHeads: Int, headDim: Int, vHeadDim: Int) {
        let text = config.textConfiguration.flashConfiguration
        if text.hybridLayerPattern[layer] != 0 {
            return (text.swaAttentionHeads, text.swaKvHeads, text.swaHeadDim, text.swaVHeadDim)
        }
        return (text.attentionHeads, text.kvHeads, text.headDim, text.vHeadDim)
    }

    /// Tensor-parallel degrees consistent with one fused tensor's shapes.
    static func candidateDegrees(
        rows: Int, heads: Int, kvHeads: Int, headDim: Int, vHeadDim: Int, gridRows: Int
    ) -> Set<Int> {
        var found: Set<Int> = []
        var degree = 1
        while degree <= kvHeads {
            if heads % degree == 0, kvHeads % degree == 0 {
                let shard =
                    (heads / degree) * headDim + (kvHeads / degree) * (headDim + vHeadDim)
                let blocks = (shard + fp8BlockSize - 1) / fp8BlockSize
                if shard * degree == rows, blocks * degree == gridRows {
                    found.insert(degree)
                }
            }
            degree *= 2
        }
        return found
    }

    /// Recover the degree the checkpoint was sharded with.
    ///
    /// A layer whose fused rows happen to be a whole number of 128-row blocks admits
    /// several degrees, so the degree is intersected across every fused tensor. Full
    /// attention layers, where the per-shard padding shows up in the grid, are what pin
    /// it down.
    func tensorParallelDegree(_ weights: [String: MLXArray]) throws -> Int {
        var common: Set<Int>?
        for (key, value) in weights {
            guard key.hasSuffix("self_attn.qkv_proj.weight"), !key.contains(".mtp."),
                let scale = weights["\(key)_scale_inv"],
                let layer = MiMoV2.layerIndex(of: key)
            else { continue }
            let heads = qkvHeads(layer: layer)
            let found = MiMoV2.candidateDegrees(
                rows: value.dim(0), heads: heads.heads, kvHeads: heads.kvHeads,
                headDim: heads.headDim, vHeadDim: heads.vHeadDim, gridRows: scale.dim(0))
            common = common.map { $0.intersection(found) } ?? found
        }
        guard let common, let degree = common.max() else {
            throw MiMoV2Error.noTensorParallelDegree
        }
        return degree
    }

    /// Split `qkv_proj` back into q/k/v.
    ///
    /// The fused weight is not one tensor followed by a single scale grid: it is the
    /// tensor-parallel shards concatenated, each holding its own q heads then k then v,
    /// and each carrying its own FP8 scale grid padded up to a whole number of 128-row
    /// blocks. Read as a single grid neither divides, and walking it contiguously
    /// misaligns every shard after the first.
    func unfuseQKV(_ weights: [String: MLXArray]) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        let sharded = weights.keys.contains {
            $0.hasSuffix("self_attn.qkv_proj.weight_scale_inv") && !$0.contains(".mtp.")
        }
        let degree = sharded ? try tensorParallelDegree(weights) : 1

        for (key, value) in weights {
            let fused = key.hasSuffix("self_attn.qkv_proj.weight") && !key.contains(".mtp.")
            guard fused, let layer = MiMoV2.layerIndex(of: key) else {
                let isScale =
                    key.hasSuffix("self_attn.qkv_proj.weight_scale_inv")
                    && !key.contains(".mtp.")
                if !isScale { out[key] = value }
                continue
            }

            let scale = weights["\(key)_scale_inv"]
            let prefix = String(key.dropLast("qkv_proj.weight".count))
            let heads = qkvHeads(layer: layer)
            let qRows = heads.heads / degree * heads.headDim
            let kRows = heads.kvHeads / degree * heads.headDim
            let vRows = heads.kvHeads / degree * heads.vHeadDim
            let shard = qRows + kRows + vRows
            let blocks = (shard + MiMoV2.fp8BlockSize - 1) / MiMoV2.fp8BlockSize

            var parts: [String: [MLXArray]] = ["q_proj": [], "k_proj": [], "v_proj": []]
            for rank in 0 ..< degree {
                let rows = value[(rank * shard) ..< ((rank + 1) * shard)]
                let dense =
                    scale.map {
                        MiMoV2.dequantizeFP8Block(
                            rows, scale: $0[(rank * blocks) ..< ((rank + 1) * blocks)])
                    } ?? rows
                var row = 0
                for (name, size) in [("q_proj", qRows), ("k_proj", kRows), ("v_proj", vRows)] {
                    parts[name]?.append(dense[row ..< (row + size)])
                    row += size
                }
            }
            for (name, chunks) in parts {
                out["\(prefix)\(name).weight"] = concatenated(chunks, axis: 0)
            }
        }
        return out
    }

    static func dequantizeFP8Block(_ weight: MLXArray, scale: MLXArray) -> MLXArray {
        let block = fp8BlockSize
        let decoded = fromFP8(weight, dtype: .float32)
        let (rows, columns) = (decoded.dim(0), decoded.dim(1))
        let padRows = block * scale.dim(0) - rows
        let padColumns = block * scale.dim(1) - columns

        var padded = MLX.padded(
            decoded, widths: [.init((0, padRows)), .init((0, padColumns))])
        padded = padded.reshaped(
            (rows + padRows) / block, block, (columns + padColumns) / block, block)
        let scaled = (padded * scale[0..., .newAxis, 0..., .newAxis])
            .reshaped(rows + padRows, columns + padColumns)
        return scaled[0 ..< rows, 0 ..< columns].asType(.bfloat16)
    }
}
