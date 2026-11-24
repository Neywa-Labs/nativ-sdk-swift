import Foundation
import MLX
import MLXNN

let nemotronParseEncoderStatesKey = LMOutput.Key<MLXArray>(
    "nativsdk.nemotronParse.encoderHiddenStates",
)

func nemotronParseVisionWeightKey(_ key: String) -> String {
    let patch = "encoder.model_encoder.radio_model.model.patch_generator."
    let blocks = "encoder.model_encoder.radio_model.model.blocks."
    if key.hasPrefix(patch) {
        let suffix = String(key.dropFirst(patch.count))
        if suffix.contains("embedder.weight") { return "vision_tower.patch_embed.weight" }
        if suffix.contains("pos_embed") { return "vision_tower.pos_embed" }
        if suffix.contains("cls_token.token") { return "vision_tower.cls_token" }
        return "vision_tower." + suffix
    }
    if key.hasPrefix(blocks) { return "vision_tower.blocks." + key.dropFirst(blocks.count) }
    if key.hasPrefix("encoder.") { return "vision_tower.neck." + key.dropFirst(8) }
    return key
}

func nemotronParseVisionLayouts(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var result: [String: MLXArray] = [:]
    for (key, value) in weights {
        if key.contains("summary_idxs") || key.contains("input_conditioner") { continue }
        if key.contains("neck.conv1.weight"), value.dim(-1) == 1, value.dim(1) != 1 {
            result[key] = value.transposed(0, 2, 1)
        } else if key.contains("neck.conv2.weight"), value.ndim == 4,
            value.dim(1) == value.dim(0), value.dim(-1) != value.dim(0)
        {
            result[key] = value.transposed(0, 2, 3, 1)
        } else {
            result[key] = value
        }
    }
    return result
}

// MARK: - Configuration

public struct NemotronParseConfiguration: Decodable, Sendable {

    public struct VisionConfiguration: Decodable, Sendable {
        public var modelType: String = "nemotron_parse_vision"
        public var hiddenSize: Int = 1280
        public var heads: Int = 16
        public var mlpRatio: Float = 4.0
        public var layers: Int = 32
        public var patchSize: Int = 16
        public var imageSize: [Int] = [2048, 1664]
        public var clsTokens: Int = 4
        public var registerTokens: Int = 4
        public var summaryIndices: [Int] = [0, 1, 2]
        public var neckDimensions: Int = 1024

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case heads = "num_heads"
            case mlpRatio = "mlp_ratio"
            case layers = "num_layers"
            case patchSize = "patch_size"
            case imageSize = "image_size"
            case clsTokens = "num_cls_tokens"
            case registerTokens = "num_register_tokens"
            case summaryIndices = "summary_idxs"
            case neckDimensions = "neck_dim"
        }

        init() {}

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
            heads = try c.decodeIfPresent(Int.self, forKey: .heads) ?? heads
            mlpRatio = try c.decodeIfPresent(Float.self, forKey: .mlpRatio) ?? mlpRatio
            layers = try c.decodeIfPresent(Int.self, forKey: .layers) ?? layers
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? patchSize
            imageSize = try c.decodeIfPresent([Int].self, forKey: .imageSize) ?? imageSize
            clsTokens = try c.decodeIfPresent(Int.self, forKey: .clsTokens) ?? clsTokens
            registerTokens =
                try c.decodeIfPresent(Int.self, forKey: .registerTokens) ?? registerTokens
            summaryIndices =
                try c.decodeIfPresent([Int].self, forKey: .summaryIndices) ?? summaryIndices
            neckDimensions =
                try c.decodeIfPresent(Int.self, forKey: .neckDimensions) ?? neckDimensions
        }

        public var skippedTokens: Int { clsTokens + registerTokens }
    }

    public struct TextConfiguration: Decodable, Sendable {
        public var modelType: String = "nemotron_parse"
        public var modelDimension: Int = 1024
        public var attentionHeads: Int = 16
        public var feedForwardDimension: Int = 4096
        public var layers: Int = 10
        public var scaleEmbedding: Bool = true
        public var maxPositionEmbeddings: Int = 9000
        public var vocabularySize: Int = 72256
        public var padTokenID: Int = 1
        public var bosTokenID: Int = 0
        public var eosTokenID: Int = 2
        public var decoderStartTokenID: Int = 2

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case modelDimension = "d_model"
            case attentionHeads = "decoder_attention_heads"
            case feedForwardDimension = "decoder_ffn_dim"
            case layers = "decoder_layers"
            case scaleEmbedding = "scale_embedding"
            case maxPositionEmbeddings = "max_position_embeddings"
            case vocabularySize = "vocab_size"
            case padTokenID = "pad_token_id"
            case bosTokenID = "bos_token_id"
            case eosTokenID = "eos_token_id"
            case decoderStartTokenID = "decoder_start_token_id"
        }

        init() {}

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
            modelDimension =
                try c.decodeIfPresent(Int.self, forKey: .modelDimension) ?? modelDimension
            attentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? attentionHeads
            feedForwardDimension =
                try c.decodeIfPresent(Int.self, forKey: .feedForwardDimension)
                ?? feedForwardDimension
            layers = try c.decodeIfPresent(Int.self, forKey: .layers) ?? layers
            scaleEmbedding =
                try c.decodeIfPresent(Bool.self, forKey: .scaleEmbedding) ?? scaleEmbedding
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
                ?? maxPositionEmbeddings
            vocabularySize =
                try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? vocabularySize
            padTokenID = try c.decodeIfPresent(Int.self, forKey: .padTokenID) ?? padTokenID
            bosTokenID = try c.decodeIfPresent(Int.self, forKey: .bosTokenID) ?? bosTokenID
            eosTokenID =
                try c.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenID)?.values.first
                ?? eosTokenID
            // A checkpoint's decoder sub-config writes null here; the top level owns it.
            decoderStartTokenID =
                try c.decodeIfPresent(Int.self, forKey: .decoderStartTokenID)
                ?? decoderStartTokenID
        }

        public var headDimensions: Int { modelDimension / attentionHeads }
    }

    public var modelType: String
    public var visionConfiguration: VisionConfiguration
    public var textConfiguration: TextConfiguration
    public var tieWordEmbeddings: Bool

    /// The halves live under `encoder` and `decoder`, not `vision_config` and
    /// `text_config`; the top-level `decoder_start_token_id` and `image_size` win over
    /// whatever the halves say.
    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case encoder
        case decoder
        case visionConfiguration = "vision_config"
        case textConfiguration = "text_config"
        case decoderStartTokenID = "decoder_start_token_id"
        case imageSize = "image_size"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case padTokenID = "pad_token_id"
        case bosTokenID = "bos_token_id"
        case eosTokenID = "eos_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "nemotron_parse"
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true

        var vision =
            try c.decodeIfPresent(VisionConfiguration.self, forKey: .encoder)
            ?? c.decodeIfPresent(VisionConfiguration.self, forKey: .visionConfiguration)
            ?? VisionConfiguration()
        var text =
            try c.decodeIfPresent(TextConfiguration.self, forKey: .decoder)
            ?? c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? TextConfiguration()

        if let imageSize = try c.decodeIfPresent([Int].self, forKey: .imageSize) {
            vision.imageSize = imageSize
        }
        if let start = try c.decodeIfPresent(Int.self, forKey: .decoderStartTokenID) {
            text.decoderStartTokenID = start
        }
        if let vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) {
            text.vocabularySize = vocabularySize
        }
        if let maxPositions = try c.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        {
            text.maxPositionEmbeddings = maxPositions
        }
        if let pad = try c.decodeIfPresent(Int.self, forKey: .padTokenID) {
            text.padTokenID = pad
        }
        if let bos = try c.decodeIfPresent(Int.self, forKey: .bosTokenID) {
            text.bosTokenID = bos
        }
        if let eos = try c.decodeIfPresent(IntOrIntArray.self, forKey: .eosTokenID)?.values.first {
            text.eosTokenID = eos
        }

        visionConfiguration = vision
        textConfiguration = text
    }
}

// MARK: - Vision

enum NemotronParseVision {

    final class Attention: Module {
        let heads: Int
        let scale: Float

        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear

        init(_ config: NemotronParseConfiguration.VisionConfiguration) {
            heads = config.heads
            scale = pow(Float(config.hiddenSize / config.heads), -0.5)
            _qkv.wrappedValue = Linear(config.hiddenSize, config.hiddenSize * 3, bias: true)
            _proj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (B, N, D) = (x.dim(0), x.dim(1), x.dim(2))
            let fused = qkv(x)
                .reshaped(B, N, 3, heads, D / heads)
                .transposed(2, 0, 3, 1, 4)
            let output = MLXFast.scaledDotProductAttention(
                queries: fused[0], keys: fused[1], values: fused[2],
                scale: scale, mask: .none)
            return proj(output.transposed(0, 2, 1, 3).reshaped(B, N, D))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear

        let gelu = GELU()

        init(_ config: NemotronParseConfiguration.VisionConfiguration) {
            let hidden = Int(Float(config.hiddenSize) * config.mlpRatio)
            _fc1.wrappedValue = Linear(config.hiddenSize, hidden, bias: true)
            _fc2.wrappedValue = Linear(hidden, config.hiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(gelu(fc1(x)))
        }
    }

    final class ViTBlock: Module {
        @ModuleInfo var norm1: LayerNorm
        @ModuleInfo var attn: Attention
        @ModuleInfo var norm2: LayerNorm
        @ModuleInfo var mlp: MLP

        init(_ config: NemotronParseConfiguration.VisionConfiguration) {
            _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _attn.wrappedValue = Attention(config)
            _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _mlp.wrappedValue = MLP(config)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = x + attn(norm1(x))
            return h + mlp(norm2(h))
        }
    }

    /// Projects the patch grid down and folds the summary tokens onto the end.
    ///
    /// `conv1` is a width-one convolution over the feature axis -- a linear projection
    /// written as a convolution because that is what the checkpoint stores. `conv2`
    /// compresses four patches horizontally into one.
    final class Neck: Module {
        let neckDimensions: Int
        let patchSize: Int

        @ModuleInfo var conv1: Conv1d
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo var conv2: Conv2d
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm
        @ModuleInfo(key: "sum_proj") var summaryProjection: Linear
        @ModuleInfo(key: "layer_norm3") var layerNorm3: LayerNorm

        init(_ config: NemotronParseConfiguration.VisionConfiguration) {
            neckDimensions = config.neckDimensions
            patchSize = config.patchSize
            _conv1.wrappedValue = Conv1d(
                inputChannels: config.hiddenSize, outputChannels: config.neckDimensions,
                kernelSize: 1, bias: true)
            _layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.neckDimensions, eps: 1e-6)
            _conv2.wrappedValue = Conv2d(
                inputChannels: config.neckDimensions,
                outputChannels: config.neckDimensions,
                kernelSize: .init((1, 4)), stride: .init((1, 4)), bias: false)
            _layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.neckDimensions, eps: 1e-6)
            _summaryProjection.wrappedValue = Linear(
                config.summaryIndices.count * config.hiddenSize, config.neckDimensions,
                bias: true)
            _layerNorm3.wrappedValue = LayerNorm(
                dimensions: config.neckDimensions, eps: 1e-6)
            super.init()
        }

        func callAsFunction(
            features: MLXArray, summary: MLXArray, patchGrid: (Int, Int)
        ) -> MLXArray {
            let batch = features.dim(0)
            var x = layerNorm1(conv1(features))
            x = x.reshaped(batch, patchGrid.0, patchGrid.1, neckDimensions)
            x = layerNorm2(conv2(x).reshaped(batch, -1, neckDimensions))
            let pooled = layerNorm3(summaryProjection(summary))
            return concatenated([x, pooled[0..., .newAxis, 0...]], axis: 1)
        }
    }

    final class VisionModel: Module {
        let config: NemotronParseConfiguration.VisionConfiguration

        @ModuleInfo(key: "patch_embed") var patchEmbed: Linear
        @ParameterInfo(key: "pos_embed") var positionEmbedding: MLXArray
        @ParameterInfo(key: "cls_token") var clsToken: MLXArray

        @ModuleInfo var blocks: [ViTBlock]
        @ModuleInfo var neck: Neck

        /// The learned positional grid is always 128 by 128 in the checkpoint.
        static let positionGridSide = 128

        init(_ config: NemotronParseConfiguration.VisionConfiguration) {
            self.config = config
            _patchEmbed.wrappedValue = Linear(
                3 * config.patchSize * config.patchSize, config.hiddenSize, bias: false)
            _positionEmbedding.wrappedValue = MLXArray.zeros([
                1, Self.positionGridSide * Self.positionGridSide, config.hiddenSize,
            ])
            _clsToken.wrappedValue = MLXArray.zeros([
                config.clsTokens + config.registerTokens, config.hiddenSize,
            ])
            _blocks.wrappedValue = (0 ..< config.layers).map { _ in ViTBlock(config) }
            _neck.wrappedValue = Neck(config)
            super.init()
        }

        /// Channel-first pixels to a flat sequence of flattened patches.
        func patchify(_ x: MLXArray) -> (MLXArray, (Int, Int)) {
            let batch = x.dim(0)
            let patch = config.patchSize
            let rows = x.dim(2) / patch
            let columns = x.dim(3) / patch
            let patches =
                x
                .reshaped(batch, 3, rows, patch, columns, patch)
                .transposed(0, 2, 4, 1, 3, 5)
                .reshaped(batch, rows * columns, -1)
            return (patches, (rows, columns))
        }

        /// The grid is cropped when it is already large enough and resampled otherwise,
        /// to the square of the larger side and only then cropped.
        func positions(rows: Int, columns: Int) -> MLXArray {
            let side = Self.positionGridSide
            var grid = positionEmbedding.reshaped(side, side, config.hiddenSize)
            if rows == side, columns == side {
                return grid.reshaped(-1, config.hiddenSize)
            }
            let target = Swift.max(rows, columns)
            if target != side {
                grid = bilinearInterpolate(
                    grid, newHeight: target, newWidth: target, alignCorners: true)
            }
            return grid[..<rows, ..<columns, 0...].reshaped(-1, config.hiddenSize)
        }

        func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
            let (patches, grid) = patchify(pixelValues)
            var x = patchEmbed(patches.asType(patchEmbed.weight.dtype))
            x = x + positions(rows: grid.0, columns: grid.1)[.newAxis, 0..., 0...]

            let prefix = broadcast(
                clsToken[.newAxis, 0..., 0...],
                to: [x.dim(0), clsToken.dim(0), clsToken.dim(1)])
            x = concatenated([prefix.asType(x.dtype), x], axis: 1)

            for block in blocks {
                x = block(x)
            }

            let summary = x[0..., ..<config.clsTokens, 0...][
                0..., MLXArray(config.summaryIndices.map(Int32.init)), 0...
            ].reshaped(x.dim(0), -1)
            let features = x[0..., config.skippedTokens..., 0...]

            return neck(features: features, summary: summary, patchGrid: grid)
        }

        /// Drops the buffers the module hardcodes and moves the neck's convolutions from
        /// the torch layout. Both moves are idempotent, because the loader runs this over
        /// already-converted checkpoints too.
        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            nemotronParseVisionLayouts(weights)
        }
    }
}

// MARK: - Decoder

enum NemotronParseText {

    final class Attention: Module {
        let heads: Int
        let headDimensions: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: NemotronParseConfiguration.TextConfiguration) {
            heads = config.attentionHeads
            headDimensions = config.headDimensions
            scale = pow(Float(headDimensions), -0.5)
            let dimensions = config.modelDimension
            _qProj.wrappedValue = Linear(dimensions, dimensions, bias: true)
            _kProj.wrappedValue = Linear(dimensions, dimensions, bias: true)
            _vProj.wrappedValue = Linear(dimensions, dimensions, bias: true)
            _outProj.wrappedValue = Linear(dimensions, dimensions, bias: true)
            super.init()
        }

        private func heads(_ x: MLXArray) -> MLXArray {
            x.reshaped(x.dim(0), x.dim(1), heads, headDimensions).transposed(0, 2, 1, 3)
        }

        func selfAttention(
            _ x: MLXArray, cache: KVCache?, mask: MLXFast.ScaledDotProductAttentionMaskMode
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))
            let output = attentionWithCacheUpdate(
                queries: heads(qProj(x)), keys: heads(kProj(x)), values: heads(vProj(x)),
                cache: cache, scale: scale, mask: mask)
            return outProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        }

        /// The encoder's keys and values never change, so they are projected on the first
        /// step and read from the cache thereafter. Recomputing them would run two
        /// projections over every encoder position on every decoded token -- thousands of
        /// positions for a page of text.
        func crossAttention(
            _ x: MLXArray, encoderStates: MLXArray, slots: ArraysCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))
            let keys: MLXArray
            let values: MLXArray
            if let cachedKeys = slots?[0], let cachedValues = slots?[1] {
                keys = cachedKeys
                values = cachedValues
            } else {
                keys = heads(kProj(encoderStates))
                values = heads(vProj(encoderStates))
                slots?[0] = keys
                slots?[1] = values
            }

            let output = MLXFast.scaledDotProductAttention(
                queries: heads(qProj(x)), keys: keys, values: values, scale: scale,
                mask: .none)
            return outProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        }
    }

    /// An mBART decoder layer: each branch normalizes its input and the residual is the
    /// un-normalized state.
    final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo(key: "self_attn_layer_norm") var selfAttentionLayerNorm: LayerNorm
        @ModuleInfo(key: "encoder_attn") var encoderAttention: Attention
        @ModuleInfo(key: "encoder_attn_layer_norm") var encoderAttentionLayerNorm: LayerNorm
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

        let activation = GELU()

        init(_ config: NemotronParseConfiguration.TextConfiguration) {
            let dimensions = config.modelDimension
            _selfAttention.wrappedValue = Attention(config)
            _selfAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: dimensions)
            _encoderAttention.wrappedValue = Attention(config)
            _encoderAttentionLayerNorm.wrappedValue = LayerNorm(dimensions: dimensions)
            _fc1.wrappedValue = Linear(dimensions, config.feedForwardDimension, bias: true)
            _fc2.wrappedValue = Linear(config.feedForwardDimension, dimensions, bias: true)
            _finalLayerNorm.wrappedValue = LayerNorm(dimensions: dimensions)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray, encoderStates: MLXArray?, cache: KVCache?
        ) -> MLXArray {
            let pair = cache as? CacheList
            let selfCache = pair?.children.first ?? cache
            let crossSlots = pair?.children.last as? ArraysCache

            var h =
                x
                + selfAttention.selfAttention(
                    selfAttentionLayerNorm(x),
                    cache: selfCache,
                    mask: createAttentionMask(h: x, cache: selfCache))

            if let encoderStates {
                h =
                    h
                    + encoderAttention.crossAttention(
                        encoderAttentionLayerNorm(h),
                        encoderStates: encoderStates, slots: crossSlots)
            }

            return h + fc2(activation(fc1(finalLayerNorm(h))))
        }
    }

    final class Decoder: Module {
        let embedScale: Float

        @ModuleInfo var layers: [DecoderLayer]
        @ModuleInfo(key: "layernorm_embedding") var layerNormEmbedding: LayerNorm
        @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

        init(_ config: NemotronParseConfiguration.TextConfiguration) {
            embedScale = config.scaleEmbedding ? sqrt(Float(config.modelDimension)) : 1
            _layers.wrappedValue = (0 ..< config.layers).map { _ in DecoderLayer(config) }
            _layerNormEmbedding.wrappedValue = LayerNorm(dimensions: config.modelDimension)
            _layerNorm.wrappedValue = LayerNorm(dimensions: config.modelDimension)
            super.init()
        }

        func callAsFunction(
            inputsEmbeds: MLXArray, encoderStates: MLXArray?, cache: [KVCache]?
        ) -> MLXArray {
            var h = layerNormEmbedding(inputsEmbeds * embedScale)
            for (i, layer) in layers.enumerated() {
                h = layer(h, encoderStates: encoderStates, cache: cache?[i])
            }
            return layerNorm(h)
        }
    }

    final class Model: Module {
        @ModuleInfo var shared: Embedding
        @ModuleInfo var decoder: Decoder

        init(_ config: NemotronParseConfiguration.TextConfiguration) {
            _shared.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.modelDimension)
            _decoder.wrappedValue = Decoder(config)
            super.init()
        }
    }

    final class LanguageModel: Module, KVCacheDimensionProvider {
        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear

        let config: NemotronParseConfiguration.TextConfiguration

        init(_ config: NemotronParseConfiguration.TextConfiguration) {
            self.config = config
            _model.wrappedValue = Model(config)
            _lmHead.wrappedValue = Linear(
                config.modelDimension, config.vocabularySize, bias: false)
            super.init()
        }

        var kvHeads: [Int] {
            Array(repeating: config.attentionHeads, count: config.layers)
        }

        func callAsFunction(
            inputIDs: MLXArray, encoderStates: MLXArray?, cache: [KVCache]?
        ) -> MLXArray {
            let embedded = model.shared(Self.stepIDs(inputIDs))
            return lmHead(
                model.decoder(
                    inputsEmbeds: embedded, encoderStates: encoderStates, cache: cache))
        }

        /// The generation loop hands back whatever shape it has; the decoder wants
        /// `[batch, steps]`.
        static func stepIDs(_ inputs: MLXArray) -> MLXArray {
            switch inputs.ndim {
            case 0: return inputs[.newAxis, .newAxis]
            case 1: return inputs.expandedDimensions(axis: 0)
            case 2: return inputs
            default: return inputs.reshaped(inputs.dim(0), -1)
            }
        }
    }
}

// MARK: - Model

public final class NemotronParse: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") var visionTower: NemotronParseVision.VisionModel
    @ModuleInfo(key: "language_model") var languageModel: NemotronParseText.LanguageModel

    public let config: NemotronParseConfiguration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: NemotronParseConfiguration) {
        self.config = config
        _visionTower.wrappedValue = NemotronParseVision.VisionModel(
            config.visionConfiguration)
        _languageModel.wrappedValue = NemotronParseText.LanguageModel(
            config.textConfiguration)
    }

    /// One self-attention cache and two slots for the encoder's keys and values, per
    /// layer.
    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        (0 ..< config.textConfiguration.layers).map { _ in
            CacheList(KVCacheSimple(), ArraysCache(size: 2))
        }
    }

    /// Image to text: the prompt's tokens are not read, and generation starts from the
    /// decoder's own start token.
    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        let encoderStates: MLXArray
        if let pixels = input.image?.pixels {
            encoderStates = visionTower(pixels)
        } else if let carried = state?[nemotronParseEncoderStatesKey] {
            encoderStates = carried
        } else {
            throw VLMError.imageRequired
        }
        var outputState = LMOutput.State()
        outputState[nemotronParseEncoderStatesKey] = encoderStates
        let output = callAsFunction(
            .init(tokens: MLXArray([config.textConfiguration.decoderStartTokenID]), mask: nil),
            cache: cache, state: outputState)
        prefill.progress?(1, 1)
        return .logits(output)
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let encoderStates: MLXArray
        if let carried = state?[nemotronParseEncoderStatesKey] {
            encoderStates = carried
        } else {
            preconditionFailure("the Nemotron Parse decoder requires encoder states")
        }

        let logits = languageModel(
            inputIDs: input.tokens, encoderStates: encoderStates, cache: cache)

        var outputState = LMOutput.State()
        outputState[nemotronParseEncoderStatesKey] = encoderStates
        return LMOutput(logits: logits, state: outputState)
    }

    public func callAsFunction(_: MLXArray, cache _: [KVCache]?) -> MLXArray {
        preconditionFailure(
            "Nemotron Parse needs encoder states; use callAsFunction(_:cache:state:)")
    }

    /// Moves the checkpoint's names onto the module tree. The encoder's ViT lives under a
    /// patch-generator prefix, everything else under `encoder.` is the neck, and the tied
    /// embedding is stored on the decoder rather than shared.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let patchGenerator = "encoder.model_encoder.radio_model.model.patch_generator."
        let visionBlocks = "encoder.model_encoder.radio_model.model.blocks."
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count + 1)

        for (rawKey, value) in weights {
            var key = rawKey
            if key.hasPrefix(patchGenerator) {
                key = "vision_tower." + String(key.dropFirst(patchGenerator.count))
                if key.contains("embedder.weight") {
                    key = key.replacingOccurrences(
                        of: "embedder.weight", with: "patch_embed.weight")
                } else if key.contains("pos_embed") {
                    key = "vision_tower.pos_embed"
                } else if key.contains("cls_token.token") {
                    key = "vision_tower.cls_token"
                }
            } else if key.hasPrefix(visionBlocks) {
                key = "vision_tower.blocks." + String(key.dropFirst(visionBlocks.count))
            } else if key.hasPrefix("encoder.") {
                key = "vision_tower.neck." + String(key.dropFirst("encoder.".count))
            } else if key.hasPrefix("decoder.") {
                key =
                    "language_model.model.decoder."
                    + String(key.dropFirst("decoder.".count))
            } else if key.hasPrefix("lm_head.") {
                key = "language_model." + key
            }
            sanitized[key] = value
        }

        if let embedding = sanitized.removeValue(
            forKey: "language_model.model.decoder.embed_tokens.weight")
        {
            sanitized["language_model.model.shared.weight"] = embedding
        }
        if sanitized["language_model.lm_head.weight"] == nil,
            let shared = sanitized["language_model.model.shared.weight"]
        {
            sanitized["language_model.lm_head.weight"] = shared
        }

        return visionTower.sanitize(weights: sanitized)
    }
}

extension NemotronParse: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.decoder.layers
    }
}

extension NemotronParse: ContinuationStateCopying {
    public func copyContinuationState(_ state: LMOutput.State) throws -> LMOutput.State {
        var result = state
        result[nemotronParseEncoderStatesKey] = state[nemotronParseEncoderStatesKey].map {
            $0[.ellipsis]
        }
        return result
    }
}
