import Foundation
import MLX
import MLXNN

public enum InternVLNextVision {

    static func normalization(_ config: InternVLNextConfiguration.VisionConfiguration)
        -> UnaryLayer
    {
        config.normType == "rms_norm"
            ? RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            : LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    class PatchEmbeddings: Module {
        @ModuleInfo(key: "projection") var projection: Conv2d

        init(_ config: InternVLNextConfiguration.VisionConfiguration) {
            _projection.wrappedValue = Conv2d(
                inputChannels: config.numChannels,
                outputChannels: config.hiddenSize,
                kernelSize: IntOrPair(config.patchSize),
                stride: IntOrPair(config.patchSize))
        }

        func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
            projection(pixelValues)
                .reshaped(pixelValues.dim(0), -1, projection.weight.dim(0))
        }
    }

    class Embeddings: Module {
        @ModuleInfo(key: "patch_embeddings") var patchEmbeddings: PatchEmbeddings
        @ParameterInfo(key: "cls_token") var clsToken: MLXArray
        @ParameterInfo(key: "position_embeddings") var positionEmbeddings: MLXArray

        let imageSize: Int
        let patchSize: Int

        init(_ config: InternVLNextConfiguration.VisionConfiguration) {
            _patchEmbeddings.wrappedValue = PatchEmbeddings(config)
            self.imageSize = config.imageSize
            self.patchSize = config.patchSize
            _clsToken.wrappedValue = MLXArray.zeros([1, 1, config.hiddenSize])
            let perSide = config.imageSize / config.patchSize
            _positionEmbeddings.wrappedValue = MLXArray.zeros(
                [1, perSide * perSide + 1, config.hiddenSize])
        }

        func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
            let patches = patchEmbeddings(pixelValues)
            let clsToken = broadcast(
                self.clsToken, to: [pixelValues.dim(0), 1, self.clsToken.dim(-1)])

            var positions = positionEmbeddings
            if patches.dim(1) + 1 != positions.dim(1) {
                // A checkpoint's table is sized for the configured image; a different
                // input resolution needs it resampled, with the class position kept aside
                // because it belongs to no grid cell.
                let perSide = imageSize / patchSize
                let grid = positions[0..., 1...].reshaped(1, perSide, perSide, -1)
                let resampled = resamplePositions(
                    grid.transposed(0, 3, 1, 2),
                    height: pixelValues.dim(1) / patchSize,
                    width: pixelValues.dim(2) / patchSize
                )
                .transposed(0, 2, 3, 1)
                positions = concatenated(
                    [
                        positions[0..., ..<1],
                        resampled.reshaped(1, -1, resampled.dim(-1)),
                    ], axis: 1)
            }
            return concatenated([clsToken, patches], axis: 1) + positions
        }
    }

    class Attention: Module {
        let heads: Int
        let headDimensions: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var wq: Linear
        @ModuleInfo(key: "k_proj") var wk: Linear
        @ModuleInfo(key: "v_proj") var wv: Linear
        @ModuleInfo(key: "projection_layer") var projectionLayer: Linear
        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?

        init(_ config: InternVLNextConfiguration.VisionConfiguration) {
            self.heads = config.numAttentionHeads
            self.headDimensions = config.hiddenSize / config.numAttentionHeads
            self.scale = pow(Float(headDimensions), -0.5)

            let size = config.hiddenSize
            _wq.wrappedValue = Linear(size, size, bias: config.attentionBias)
            _wk.wrappedValue = Linear(size, size, bias: config.attentionBias)
            _wv.wrappedValue = Linear(size, size, bias: config.attentionBias)
            _projectionLayer.wrappedValue = Linear(size, size)
            if config.useQKNorm {
                _qNorm.wrappedValue = RMSNorm(dimensions: size, eps: config.layerNormEps)
                _kNorm.wrappedValue = RMSNorm(dimensions: size, eps: config.layerNormEps)
            }
        }

        func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            let (B, L, D) = (hiddenStates.dim(0), hiddenStates.dim(1), hiddenStates.dim(2))
            var queries = wq(hiddenStates)
            var keys = wk(hiddenStates)
            if let qNorm { queries = qNorm(queries) }
            if let kNorm { keys = kNorm(keys) }
            let values = wv(hiddenStates)

            queries = queries.reshaped(B, L, heads, headDimensions).transposed(0, 2, 1, 3)
            keys = keys.reshaped(B, L, heads, headDimensions).transposed(0, 2, 1, 3)
            let reshapedValues = values.reshaped(B, L, heads, headDimensions)
                .transposed(0, 2, 1, 3)

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: reshapedValues, scale: scale,
                mask: .none
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, D)
            return projectionLayer(output)
        }
    }

    class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "fc1") var fc1: Linear
        @ModuleInfo(key: "fc2") var fc2: Linear
        let activation = GELU(approximation: .precise)

        init(_ config: InternVLNextConfiguration.VisionConfiguration) {
            _fc1.wrappedValue = Linear(config.hiddenSize, config.intermediateSize)
            _fc2.wrappedValue = Linear(config.intermediateSize, config.hiddenSize)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(activation(fc1(x))) }
    }

    class EncoderLayer: Module {
        @ModuleInfo(key: "attention") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP
        @ModuleInfo(key: "layernorm_before") var normBefore: UnaryLayer
        @ModuleInfo(key: "layernorm_after") var normAfter: UnaryLayer
        @ParameterInfo(key: "lambda_1") var lambda1: MLXArray
        @ParameterInfo(key: "lambda_2") var lambda2: MLXArray

        init(_ config: InternVLNextConfiguration.VisionConfiguration) {
            _attention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(config)
            _normBefore.wrappedValue = normalization(config)
            _normAfter.wrappedValue = normalization(config)
            _lambda1.wrappedValue = MLXArray.full(
                [config.hiddenSize], values: MLXArray(config.layerScaleInitValue))
            _lambda2.wrappedValue = MLXArray.full(
                [config.hiddenSize], values: MLXArray(config.layerScaleInitValue))
        }

        func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            let attended = hiddenStates + attention(normBefore(hiddenStates)) * lambda1
            return attended + mlp(normAfter(attended)) * lambda2
        }
    }

    public class VisionModel: Module {
        @ModuleInfo(key: "embeddings") var embeddings: Embeddings
        @ModuleInfo(key: "encoder") var encoder: Encoder

        class Encoder: Module {
            let layer: [EncoderLayer]

            init(_ config: InternVLNextConfiguration.VisionConfiguration) {
                self.layer = (0 ..< config.numHiddenLayers).map { _ in EncoderLayer(config) }
            }

            func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
                var h = hiddenStates
                for layer in layer { h = layer(h) }
                return h
            }
        }

        init(_ config: InternVLNextConfiguration.VisionConfiguration) {
            _embeddings.wrappedValue = Embeddings(config)
            _encoder.wrappedValue = Encoder(config)
        }

        func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
            encoder(embeddings(pixelValues))
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized = weights
            for (key, value) in weights
            where key.hasSuffix("patch_embeddings.projection.weight") {
                if !Self.isChannelsLast(value) {
                    sanitized[key] = value.transposed(0, 2, 3, 1)
                }
            }
            return sanitized
        }

        /// Whether a 4-D convolution weight is already in MLX's `(out, kH, kW, in)`
        /// layout rather than PyTorch's `(out, in, kH, kW)`.
        ///
        /// The test is upstream's: the output channels are at least as many as either
        /// spatial extent, and the two spatial extents agree. A square kernel read as
        /// NCHW puts the input channels where a spatial extent belongs, which breaks the
        /// second condition for every kernel wider than its channel count.
        static func isChannelsLast(_ weight: MLXArray) -> Bool {
            guard weight.ndim == 4 else { return false }
            let (out, kH, kW) = (weight.dim(0), weight.dim(1), weight.dim(2))
            return out >= kH && out >= kW && kH == kW
        }
    }

    /// Nearest-neighbour resample over the two spatial axes of an NCHW table.
    ///
    /// Only reached when the input resolution differs from the configured one, and only
    /// for the position table, which is smooth; the alternative is refusing the image.
    static func resamplePositions(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        bicubicInterpolate(x, size: (height, width))
    }
}

public struct InternVLNextConfiguration: Codable, Sendable {
    public struct VisionConfiguration: Codable, Sendable {
        public let modelType: String
        private let _hiddenSize: Int?
        public var hiddenSize: Int { _hiddenSize ?? 1024 }
        private let _intermediateSize: Int?
        public var intermediateSize: Int { _intermediateSize ?? 4096 }
        private let _numHiddenLayers: Int?
        public var numHiddenLayers: Int { _numHiddenLayers ?? 24 }
        private let _numAttentionHeads: Int?
        public var numAttentionHeads: Int { _numAttentionHeads ?? 16 }
        /// Written either as a scalar or as a `[height, width]` pair; upstream takes the
        /// first entry of a pair, so a non-square declaration is read on its first axis.
        private let _imageSize: IntOrIntArray?
        public var imageSize: Int { _imageSize?.values.first ?? 448 }
        private let _patchSize: IntOrIntArray?
        public var patchSize: Int { _patchSize?.values.first ?? 14 }
        private let _numChannels: Int?
        public var numChannels: Int { _numChannels ?? 3 }
        private let _layerNormEps: Float?
        public var layerNormEps: Float { _layerNormEps ?? 1e-6 }
        private let _layerScaleInitValue: Float?
        public var layerScaleInitValue: Float { _layerScaleInitValue ?? 0.1 }
        private let _normType: String?
        public var normType: String { _normType ?? "layer_norm" }
        private let _attentionBias: Bool?
        public var attentionBias: Bool { _attentionBias ?? true }
        private let _useQKNorm: Bool?
        public var useQKNorm: Bool { _useQKNorm ?? false }

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case _hiddenSize = "hidden_size"
            case _intermediateSize = "intermediate_size"
            case _numHiddenLayers = "num_hidden_layers"
            case _numAttentionHeads = "num_attention_heads"
            case _imageSize = "image_size"
            case _patchSize = "patch_size"
            case _numChannels = "num_channels"
            case _layerNormEps = "layer_norm_eps"
            case _layerScaleInitValue = "layer_scale_init_value"
            case _normType = "norm_type"
            case _attentionBias = "attention_bias"
            case _useQKNorm = "use_qk_norm"
        }
    }

    public let textConfiguration: Qwen3Configuration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _downsampleRatio: Float?
    public var downsampleRatio: Float { _downsampleRatio ?? 0.5 }
    private let _imageTokenId: Int?
    public var imageTokenId: Int { _imageTokenId ?? 151_671 }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _downsampleRatio = "downsample_ratio"
        case _imageTokenId = "image_token_id"
    }
}

class InternVLNextProjector: Module, UnaryLayer {
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    let activation = GELU(approximation: .precise)

    init(_ config: InternVLNextConfiguration) {
        let factor = Int((1 / config.downsampleRatio).rounded())
        let visionSize = config.visionConfiguration.hiddenSize * factor * factor
        let textSize = config.textConfiguration.hiddenSize
        _layerNorm.wrappedValue = LayerNorm(dimensions: visionSize)
        _linear1.wrappedValue = Linear(visionSize, textSize)
        _linear2.wrappedValue = Linear(textSize, textSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear2(activation(linear1(layerNorm(x))))
    }
}

public final class InternVLNext: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: InternVLNextVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: Qwen3Model
    @ModuleInfo(key: "multi_modal_projector") private var projector: InternVLNextProjector

    public let config: InternVLNextConfiguration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public init(_ config: InternVLNextConfiguration) {
        self.config = config
        _visionTower.wrappedValue = InternVLNextVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = Qwen3Model(config.textConfiguration)
        _projector.wrappedValue = InternVLNextProjector(config)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 { tokens = tokens.expandedDimensions(axis: 0) }

        let embeddings = try inputEmbeddings(inputIds: tokens, pixelValues: input.image?.pixels)
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

    func inputEmbeddings(inputIds: MLXArray, pixelValues: MLXArray?) throws -> MLXArray {
        let inputsEmbeds = languageModel.model.embedTokens(inputIds)
        guard var pixelValues else { return inputsEmbeds }

        if pixelValues.ndim == 5 { pixelValues = pixelValues[0] }
        var features = visionTower(pixelValues.transposed(0, 2, 3, 1))
        // The class token carries no patch, so it is dropped before the shuffle folds
        // neighbouring patches into the channel axis.
        features = features[0..., 1...]
        features = Self.pixelShuffle(features, downsampleRatio: config.downsampleRatio)
        features = projector(features).reshaped(-1, inputsEmbeds.dim(-1))

        let ids: [Int32] = inputIds[0].asArray(Int32.self)
        let positions = ids.enumerated().compactMap {
            $1 == Int32(config.imageTokenId) ? $0 : nil
        }
        guard positions.count == features.dim(0) else {
            throw VLMError.processing(
                "the prompt holds \(positions.count) image tokens and the tower produced "
                    + "\(features.dim(0)) features; the processor must emit one token each")
        }
        let merged = inputsEmbeds
        merged[0..., MLXArray(positions.map(Int32.init)), 0...] = features
        return merged
    }

    /// Folds each `factor x factor` block of patches into the channel axis, which is how
    /// the projector sees a whole neighbourhood at once.
    static func pixelShuffle(_ input: MLXArray, downsampleRatio: Float) -> MLXArray {
        let batch = input.dim(0)
        let channels = input.dim(2)
        let side = Int(Double(input.dim(1)).squareRoot().rounded())
        let factor = Int((1 / downsampleRatio).rounded())

        var x = input.reshaped(batch, side, side, channels)
        x = x.reshaped(batch, side, side / factor, channels * factor)
        x = x.transposed(0, 2, 1, 3)
        x = x.reshaped(batch, side / factor, side / factor, channels * factor * factor)
        return x.reshaped(batch, -1, channels * factor * factor)
    }

    /// Some conversions nest the backbone one level too deep.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            var key = key
            if key.hasPrefix("language_model.model.model.") {
                key = "language_model.model." + key.dropFirst("language_model.model.model.".count)
            } else if key.hasPrefix("language_model.model.lm_head.") {
                key =
                    "language_model.lm_head."
                    + key.dropFirst("language_model.model.lm_head.".count)
            }
            sanitized[key] = value
        }
        return visionTower.sanitize(weights: sanitized)
    }
}

extension InternVLNext: LoRAModel {
    public var loraLayers: [Module] { languageModel.loraLayers }
}
