import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct CohereCompassConfiguration: Decodable, Sendable {

    public struct TextConfiguration: Decodable, Sendable {
        public var modelType: String = "cohere_compass_text"
        public var vocabularySize: Int = 131_072
        public var hiddenSize: Int = 4096
        public var intermediateSize: Int = 11008
        public var hiddenLayers: Int = 32
        public var attentionHeads: Int = 32
        public var maxPositionEmbeddings: Int = 8192
        public var layerNormEps: Float = 1e-5
        public var normType: String = "layer_norm"
        public var transformerBlockType: String = "parallel"
        public var tieWordEmbeddings: Bool = false
        public var ropeTheta: Float = 10000
        public var ropeStyle: String = "split"
        public var ropeOnAllLayers: Bool = true
        public var attentionBias: Bool = false
        public var mlpBias: Bool = false

        private var declaredKVHeads: Int?
        private var declaredHeadDim: Int?
        private var declaredRMSNormEps: Float?
        private var declaredSWARopeTheta: Float?
        private var declaredLayerTypes: [String]?
        public var slidingWindow: Int?
        public var logitScale: Float?
        public var ropeParameters: [String: StringOrNumber]?
        public var perLayerRopeParameters: [String: [String: StringOrNumber]]?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case declaredKVHeads = "num_key_value_heads"
            case declaredHeadDim = "head_dim"
            case maxPositionEmbeddings = "max_position_embeddings"
            case declaredRMSNormEps = "rms_norm_eps"
            case layerNormEps = "layer_norm_eps"
            case normType = "norm_type"
            case transformerBlockType = "transformer_block_type"
            case tieWordEmbeddings = "tie_word_embeddings"
            case ropeTheta = "rope_theta"
            case declaredSWARopeTheta = "swa_rope_theta"
            case ropeParameters = "rope_parameters"
            case ropeStyle = "rope_style"
            case ropeOnAllLayers = "rope_on_all_layers"
            case attentionBias = "attention_bias"
            case mlpBias = "mlp_bias"
            case slidingWindow = "sliding_window"
            case declaredLayerTypes = "layer_types"
            case logitScale = "logit_scale"
        }

        init() {}

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
            vocabularySize =
                try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? vocabularySize
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
            hiddenLayers =
                try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? hiddenLayers
            attentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? attentionHeads
            declaredKVHeads = try c.decodeIfPresent(Int.self, forKey: .declaredKVHeads)
            declaredHeadDim = try c.decodeIfPresent(Int.self, forKey: .declaredHeadDim)
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
                ?? maxPositionEmbeddings
            declaredRMSNormEps = try c.decodeIfPresent(Float.self, forKey: .declaredRMSNormEps)
            layerNormEps =
                try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? layerNormEps
            normType = try c.decodeIfPresent(String.self, forKey: .normType) ?? normType
            transformerBlockType =
                try c.decodeIfPresent(String.self, forKey: .transformerBlockType)
                ?? transformerBlockType
            tieWordEmbeddings =
                try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
                ?? tieWordEmbeddings
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? ropeTheta
            declaredSWARopeTheta = try c.decodeIfPresent(
                Float.self, forKey: .declaredSWARopeTheta)
            ropeStyle = try c.decodeIfPresent(String.self, forKey: .ropeStyle) ?? ropeStyle
            ropeOnAllLayers =
                try c.decodeIfPresent(Bool.self, forKey: .ropeOnAllLayers) ?? ropeOnAllLayers
            attentionBias =
                try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? attentionBias
            mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? mlpBias
            slidingWindow = try c.decodeIfPresent(Int.self, forKey: .slidingWindow)
            declaredLayerTypes = try c.decodeIfPresent(
                [String].self, forKey: .declaredLayerTypes)
            logitScale = try c.decodeIfPresent(Float.self, forKey: .logitScale)

            // `rope_parameters` is either one parameter set or a set per layer type.
            perLayerRopeParameters = try? c.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: .ropeParameters)
            if perLayerRopeParameters == nil {
                ropeParameters = try? c.decodeIfPresent(
                    [String: StringOrNumber].self, forKey: .ropeParameters)
            }
        }

        public var kvHeads: Int { declaredKVHeads ?? attentionHeads }
        public var headDim: Int { declaredHeadDim ?? hiddenSize / attentionHeads }
        public var rmsNormEps: Float? { declaredRMSNormEps }
        public var swaRopeTheta: Float { declaredSWARopeTheta ?? ropeTheta }
        public var layerTypes: [String] {
            declaredLayerTypes ?? Array(repeating: "full_attention", count: hiddenLayers)
        }
        public var isParallelBlock: Bool { transformerBlockType == "parallel" }

        /// The rope parameters that apply to a layer, and whether the layer rotates at all.
        func ropeParameters(layerType: String) -> (
            parameters: [String: StringOrNumber]?, enabled: Bool
        ) {
            if let perLayer = perLayerRopeParameters,
                layerTypes.contains(where: { perLayer[$0] != nil })
            {
                let parameters = perLayer[layerType]
                return (parameters, parameters != nil)
            }
            let enabled = ropeOnAllLayers || layerType == "sliding_attention"
            return (ropeParameters, enabled)
        }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public var modelType: String = "cohere_compass_vision"
        public var depth: Int = 27
        public var hiddenSize: Int = 1152
        public var intermediateSize: Int = 4304
        public var heads: Int = 16
        public var channels: Int = 3
        public var patchSize: Int = 16
        public var spatialMergeSize: Int = 2
        public var temporalPatchSize: Int = 2
        public var outHiddenSize: Int = 3584
        public var positionEmbeddings: Int = 2304
        public var deepstackVisualIndexes: [Int] = [8, 16, 24]
        public var useRope: Bool = true

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case depth
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case heads = "num_heads"
            case channels = "in_channels"
            case patchSize = "patch_size"
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
            case outHiddenSize = "out_hidden_size"
            case positionEmbeddings = "num_position_embeddings"
            case deepstackVisualIndexes = "deepstack_visual_indexes"
            case useRope = "use_rope"
        }

        init() {}

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
            depth = try c.decodeIfPresent(Int.self, forKey: .depth) ?? depth
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
            heads = try c.decodeIfPresent(Int.self, forKey: .heads) ?? heads
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? channels
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? patchSize
            spatialMergeSize =
                try c.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? spatialMergeSize
            temporalPatchSize =
                try c.decodeIfPresent(Int.self, forKey: .temporalPatchSize)
                ?? temporalPatchSize
            outHiddenSize =
                try c.decodeIfPresent(Int.self, forKey: .outHiddenSize) ?? outHiddenSize
            positionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .positionEmbeddings)
                ?? positionEmbeddings
            deepstackVisualIndexes =
                try c.decodeIfPresent([Int].self, forKey: .deepstackVisualIndexes)
                ?? deepstackVisualIndexes
            useRope = try c.decodeIfPresent(Bool.self, forKey: .useRope) ?? useRope
        }

        public var mergeUnit: Int { spatialMergeSize * spatialMergeSize }
        public var gridPerSide: Int {
            Int(Double(positionEmbeddings).squareRoot().rounded())
        }
    }

    public var modelType: String
    public var textConfiguration: TextConfiguration
    public var visionConfiguration: VisionConfiguration?
    public var imageTokenID: Int?
    public var visionStartTokenID: Int?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case imageTokenID = "image_token_id"
        case visionStartTokenID = "vision_start_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "cohere_compass"
        textConfiguration =
            try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? TextConfiguration()
        visionConfiguration = try c.decodeIfPresent(
            VisionConfiguration.self, forKey: .visionConfiguration)
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID)
        visionStartTokenID = try c.decodeIfPresent(Int.self, forKey: .visionStartTokenID)
    }
}

// MARK: - Text

enum CohereCompassText {

    /// Weight-only RMS normalization in float32. The weight is the scale itself, not an
    /// offset from one.
    final class RMSNorm: Module, UnaryLayer {
        @ParameterInfo(key: "weight") var weight: MLXArray
        let eps: Float

        init(dimensions: Int, eps: Float) {
            self.eps = eps
            _weight.wrappedValue = MLXArray.ones([dimensions])
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let float = x.asType(.float32)
            let normalized = float * rsqrt((float * float).mean(axis: -1, keepDims: true) + eps)
            return (normalized * weight.asType(.float32)).asType(x.dtype)
        }
    }

    /// Mean-subtracting normalization with a weight and no bias.
    final class LayerNorm: Module, UnaryLayer {
        @ParameterInfo(key: "weight") var weight: MLXArray
        let eps: Float

        init(dimensions: Int, eps: Float) {
            self.eps = eps
            _weight.wrappedValue = MLXArray.ones([dimensions])
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let float = x.asType(.float32)
            let mean = float.mean(axis: -1, keepDims: true)
            let centered = float - mean
            let variance = (centered * centered).mean(axis: -1, keepDims: true)
            return (centered * rsqrt(variance + eps) * weight.asType(.float32))
                .asType(x.dtype)
        }
    }

    static func makeNorm(_ config: CohereCompassConfiguration.TextConfiguration)
        -> UnaryLayer
    {
        config.normType == "rms_norm"
            ? RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps ?? 1e-6)
            : LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    /// Runs a module one row at a time above batch one.
    ///
    /// Upstream does this so a row's output does not depend on how many rows share the
    /// call. Matching it is what keeps a batched run's logits equal to the single-row ones.
    static func rowwise(_ module: UnaryLayer, _ x: MLXArray) -> MLXArray {
        guard x.ndim >= 3, x.dim(0) > 1 else { return module(x) }
        return concatenated((0 ..< x.dim(0)).map { module(x[$0 ..< ($0 + 1)]) })
    }

    static func rotateHalfSplit(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    }

    static func rotateHalfInterleaved(_ x: MLXArray) -> MLXArray {
        let even = x[.ellipsis, .stride(by: 2)]
        let odd = x[.ellipsis, .stride(from: 1, by: 2)]
        return MLX.stacked([-odd, even], axis: -1).flattened(start: -2)
    }

    static func applyRoPE(
        _ q: MLXArray, _ k: MLXArray, cos: MLXArray, sin: MLXArray, style: String
    ) -> (MLXArray, MLXArray) {
        let cos = cos.expandedDimensions(axis: 1)
        let sin = sin.expandedDimensions(axis: 1)
        let rotate = style == "interleave" ? rotateHalfInterleaved : rotateHalfSplit
        return (
            (q * cos + rotate(q) * sin).asType(q.dtype),
            (k * cos + rotate(k) * sin).asType(k.dtype)
        )
    }

    /// The rotary table, including the repeated-frequency M-RoPE layout a multimodal
    /// checkpoint asks for.
    final class RotaryEmbedding {
        let dimensions: Int
        let ropeStyle: String
        let enabled: Bool
        let inverseFrequencies: MLXArray
        let positionSelector: MLXArray?

        init(_ config: CohereCompassConfiguration.TextConfiguration, layerType: String?) {
            dimensions = config.headDim
            ropeStyle = config.ropeStyle

            let (parameters, enabled) = config.ropeParameters(
                layerType: layerType ?? "full_attention")
            self.enabled = layerType == nil ? true : enabled

            let theta =
                parameters?["rope_theta"]?.asFloat()
                ?? (layerType == "sliding_attention" ? config.swaRopeTheta : config.ropeTheta)
            inverseFrequencies =
                1
                / pow(
                    MLXArray(theta),
                    MLXArray(stride(from: 0, to: dimensions, by: 2).map(Float.init))
                        / Float(dimensions))

            // Three sections that sum to the frequency count select the temporal, height
            // and width position rows, interleaved every third frequency.
            if let section = parameters?["mrope_section"]?.asInts(),
                section.count == 3,
                section.allSatisfy({ $0 >= 0 }),
                section.reduce(0, +) == dimensions / 2
            {
                var selector = [Int32](repeating: 0, count: dimensions / 2)
                for (axis, offset) in [(1, 1), (2, 2)] {
                    var index = offset
                    while index < Swift.min(section[axis] * 3, dimensions / 2) {
                        selector[index] = Int32(axis)
                        index += 3
                    }
                }
                positionSelector = MLXArray(selector)
            } else {
                positionSelector = nil
            }
        }

        func callAsFunction(_ x: MLXArray, positionIDs: MLXArray) -> (MLXArray, MLXArray) {
            let frequencies: MLXArray
            if positionIDs.ndim == 3, let positionSelector {
                let positions = take(positionIDs, positionSelector, axis: 0)
                    .transposed(1, 2, 0)
                frequencies = positions.asType(.float32) * inverseFrequencies
            } else {
                frequencies =
                    positionIDs.asType(.float32)[.ellipsis, .newAxis] * inverseFrequencies
            }
            let embedding =
                ropeStyle == "interleave"
                ? repeated(frequencies, count: 2, axis: -1)
                : concatenated([frequencies, frequencies], axis: -1)
            return (cos(embedding).asType(x.dtype), sin(embedding).asType(x.dtype))
        }
    }
}

extension CohereCompassText {

    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let layerType: String
        let slidingWindow: Int?
        let ropeStyle: String
        let rotary: RotaryEmbedding

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        init(_ config: CohereCompassConfiguration.TextConfiguration, layerIndex: Int) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(config.headDim), -0.5)
            layerType = config.layerTypes[layerIndex]
            slidingWindow = layerType == "sliding_attention" ? config.slidingWindow : nil
            ropeStyle = config.ropeStyle
            rotary = RotaryEmbedding(config, layerType: layerType)

            _qProj.wrappedValue = Linear(
                config.hiddenSize, heads * headDim, bias: config.attentionBias)
            _kProj.wrappedValue = Linear(
                config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
            _vProj.wrappedValue = Linear(
                config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
            _oProj.wrappedValue = Linear(
                heads * headDim, config.hiddenSize, bias: config.attentionBias)
            super.init()
        }

        func callAsFunction(
            _ hiddenStates: MLXArray, mask: MLXArray?, cache: KVCache?,
            positionEmbeddings: (cos: MLXArray, sin: MLXArray)?
        ) -> MLXArray {
            let (B, L) = (hiddenStates.dim(0), hiddenStates.dim(1))
            var queries = rowwise(qProj, hiddenStates)
                .reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            var keys = rowwise(kProj, hiddenStates)
                .reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            let values = rowwise(vProj, hiddenStates)
                .reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            if rotary.enabled, let positionEmbeddings {
                (queries, keys) = applyRoPE(
                    queries, keys, cos: positionEmbeddings.cos,
                    sin: positionEmbeddings.sin, style: ropeStyle)
            }

            var mask = mask
            let output: MLXArray
            if let cache {
                let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
                if let value = mask, value.dim(-1) != cachedKeys.dim(2) {
                    mask = value[.ellipsis, (-cachedKeys.dim(2))...]
                }
                output = MLXFast.scaledDotProductAttention(
                    queries: queries, keys: cachedKeys, values: cachedValues,
                    scale: scale, mask: mask.map { .array($0) } ?? .none)
            } else {
                output = MLXFast.scaledDotProductAttention(
                    queries: queries, keys: keys, values: values, scale: scale,
                    mask: mask.map { .array($0) } ?? .none)
            }

            return rowwise(
                oProj, output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        }
    }

    /// A parallel block reads one normalization for both branches and sums them into the
    /// residual; a sequential one normalizes again between them.
    final class DecoderLayer: Module {
        let attentionType: String
        let isParallel: Bool

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: UnaryLayer
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo var mlp: SwiGLUMLP
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: UnaryLayer?

        init(_ config: CohereCompassConfiguration.TextConfiguration, layerIndex: Int) {
            attentionType = config.layerTypes[layerIndex]
            isParallel = config.isParallelBlock
            _inputLayerNorm.wrappedValue = makeNorm(config)
            _attention.wrappedValue = Attention(config, layerIndex: layerIndex)
            _mlp.wrappedValue = SwiGLUMLP(
                inputDimensions: config.hiddenSize,
                hiddenDimensions: config.intermediateSize,
                bias: config.mlpBias)
            if !isParallel {
                _postAttentionLayerNorm.wrappedValue = makeNorm(config)
            }
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXArray?, cache: KVCache?,
            positionEmbeddings: (cos: MLXArray, sin: MLXArray)?
        ) -> MLXArray {
            let normalized = inputLayerNorm(x)
            let attended = attention(
                normalized, mask: mask, cache: cache,
                positionEmbeddings: positionEmbeddings)
            if let postAttentionLayerNorm {
                let h = x + attended
                return h + rowwise(mlp, postAttentionLayerNorm(h))
            }
            return x + attended + rowwise(mlp, normalized)
        }
    }

    final class Backbone: Module {
        let config: CohereCompassConfiguration.TextConfiguration

        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo var layers: [DecoderLayer]
        @ModuleInfo var norm: UnaryLayer

        /// One table per distinct layer type, in first-appearance order.
        let rotaryByLayerType: [(type: String, rotary: RotaryEmbedding)]
        let primaryRotary: RotaryEmbedding

        init(_ config: CohereCompassConfiguration.TextConfiguration) {
            self.config = config
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
                DecoderLayer(config, layerIndex: $0)
            }
            _norm.wrappedValue =
                config.rmsNormEps != nil
                ? RMSNorm(
                    dimensions: config.hiddenSize,
                    eps: config.rmsNormEps ?? config.layerNormEps)
                : LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)

            var seen: Set<String> = []
            var tables: [(type: String, rotary: RotaryEmbedding)] = []
            for layerType in config.layerTypes where seen.insert(layerType).inserted {
                tables.append((layerType, RotaryEmbedding(config, layerType: layerType)))
            }
            rotaryByLayerType = tables
            primaryRotary =
                tables.first(where: { $0.rotary.enabled })?.rotary
                ?? tables.first!.rotary
            super.init()
        }

        func callAsFunction(
            inputs: MLXArray?, inputEmbedding: MLXArray?, mask: MLXArray?,
            cache: [KVCache]?, positionIDs: MLXArray?,
            visualPositionMask: MLXArray? = nil,
            deepstackEmbeddings: [MLXArray]? = nil
        ) -> MLXArray {
            guard var h = inputEmbedding ?? inputs.map({ embedTokens($0) }) else {
                fatalError("either inputs or inputEmbedding must be provided")
            }

            let types = config.layerTypes
            let globalIndex = types.firstIndex(of: "full_attention")
            let slidingIndex = types.firstIndex(of: "sliding_attention")

            var globalMask = mask
            var slidingMask = mask
            if mask == nil {
                globalMask = causalMask(
                    h: h, cache: globalIndex.flatMap { cache?[$0] }, window: nil)
                slidingMask = causalMask(
                    h: h, cache: slidingIndex.flatMap { cache?[$0] },
                    window: config.slidingWindow)
            }

            var positions = positionIDs
            if positions == nil {
                // Positions continue from what the cache already holds. Restarting at
                // zero on a warm cache rotates every step as if it were the first, which
                // is silent on a cold single forward and wrong for every decode step.
                let offset = cache?.first?.offset ?? 0
                var linear =
                    (MLXArray(0 ..< h.dim(1)) + MLXArray(Int32(offset)))
                    .expandedDimensions(axis: 0)
                if primaryRotary.positionSelector != nil {
                    linear = broadcast(
                        linear.expandedDimensions(axis: 0),
                        to: [3, h.dim(0), h.dim(1)])
                }
                positions = linear
            }

            var embeddingsByType: [String: (cos: MLXArray, sin: MLXArray)] = [:]
            for entry in rotaryByLayerType where entry.rotary.enabled {
                embeddingsByType[entry.type] = entry.rotary(h, positionIDs: positions!)
            }

            for (index, layer) in layers.enumerated() {
                h = layer(
                    h,
                    mask: layer.attentionType == "full_attention" ? globalMask : slidingMask,
                    cache: cache?[index],
                    positionEmbeddings: embeddingsByType[layer.attentionType])
                if let deepstackEmbeddings, index < deepstackEmbeddings.count {
                    h = Self.addDeepstack(
                        h, visualPositionMask: visualPositionMask,
                        embeddings: deepstackEmbeddings[index])
                }
            }
            return norm(h)
        }

        /// Adds each layer's visual features back at the image positions, in order.
        static func addDeepstack(
            _ hiddenStates: MLXArray, visualPositionMask: MLXArray?, embeddings: MLXArray
        ) -> MLXArray {
            guard let visualPositionMask else { return hiddenStates }
            let selected = visualPositionMask.reshaped(-1)
            eval(selected)
            let count = selected.asType(.int32).sum().item(Int.self)
            guard count > 0, embeddings.dim(0) > 0 else { return hiddenStates }

            let flat = hiddenStates.reshaped(-1, hiddenStates.dim(-1))
            let rank = clip(
                cumsum(selected.asType(.int32), axis: 0) - 1,
                min: 0, max: embeddings.dim(0) - 1)
            let gathered = takeAlong(
                embeddings,
                broadcast(rank[0..., .newAxis], to: [flat.dim(0), embeddings.dim(-1)]),
                axis: 0)
            let keep = broadcast(selected[0..., .newAxis], to: flat.shape)
            return MLX.where(keep, flat + gathered.asType(flat.dtype), flat)
                .reshaped(hiddenStates.shape)
        }

        /// The cache decides: a rotating cache's mask has to match how it rolled its own
        /// keys, which is not something a caller can reconstruct from the offset.
        private func causalMask(h: MLXArray, cache: KVCache?, window: Int?) -> MLXArray? {
            guard h.dim(1) > 1 else { return nil }
            let mode = createAttentionMask(
                h: h, cache: cache, windowSize: window, returnArray: true)
            if case .array(let mask) = mode {
                return mask
            }
            return nil
        }
    }
}

// MARK: - Vision

enum CohereCompassVision {

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    }

    static func applyRoPE(_ q: MLXArray, _ k: MLXArray, frequencies: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        let cosine = concatenated([cos(frequencies), cos(frequencies)], axis: -1)
        let sine = concatenated([sin(frequencies), sin(frequencies)], axis: -1)
        let (qDtype, kDtype) = (q.dtype, k.dtype)
        let q32 = q.asType(.float32)
        let k32 = k.asType(.float32)
        let c = cosine.asType(.float32).expandedDimensions(axis: -2)
        let s = sine.asType(.float32).expandedDimensions(axis: -2)
        return (
            (q32 * c + rotateHalf(q32) * s).asType(qDtype),
            (k32 * c + rotateHalf(k32) * s).asType(kDtype)
        )
    }

    final class PatchEmbed: Module, UnaryLayer {
        let channels: Int
        let temporalPatchSize: Int
        let patchSize: Int
        let hiddenSize: Int

        @ModuleInfo var proj: Conv3d

        init(_ config: CohereCompassConfiguration.VisionConfiguration) {
            channels = config.channels
            temporalPatchSize = config.temporalPatchSize
            patchSize = config.patchSize
            hiddenSize = config.hiddenSize
            _proj.wrappedValue = Conv3d(
                inputChannels: config.channels, outputChannels: config.hiddenSize,
                kernelSize: .init(
                    (config.temporalPatchSize, config.patchSize, config.patchSize)),
                stride: .init(
                    (config.temporalPatchSize, config.patchSize, config.patchSize)),
                bias: true)
            super.init()
        }

        func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            let reshaped =
                hiddenStates
                .reshaped(-1, channels, temporalPatchSize, patchSize, patchSize)
                .transposed(0, 2, 3, 4, 1)
            return proj(reshaped).reshaped(-1, hiddenSize)
        }
    }

    /// Folds each merge block into one row.
    ///
    /// The plain merger normalizes before the fold and the deep-stack mergers after it,
    /// over the folded width -- a different normalization over different numbers.
    final class PatchMerger: Module, UnaryLayer {
        let mergedSize: Int
        let normalizeAfterShuffle: Bool

        @ModuleInfo var norm: LayerNorm
        @ModuleInfo(key: "linear_fc1") var linear1: Linear
        @ModuleInfo(key: "linear_fc2") var linear2: Linear

        let activation = GELU()

        init(
            _ config: CohereCompassConfiguration.VisionConfiguration,
            normalizeAfterShuffle: Bool = false
        ) {
            mergedSize = config.hiddenSize * config.mergeUnit
            self.normalizeAfterShuffle = normalizeAfterShuffle
            _norm.wrappedValue = LayerNorm(
                dimensions: normalizeAfterShuffle ? mergedSize : config.hiddenSize,
                eps: 1e-6)
            _linear1.wrappedValue = Linear(mergedSize, mergedSize, bias: true)
            _linear2.wrappedValue = Linear(mergedSize, config.outHiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = normalizeAfterShuffle ? norm(x.reshaped(-1, mergedSize)) : norm(x)
            h = h.reshaped(-1, mergedSize)
            return linear2(activation(linear1(h)))
        }
    }

    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo var qkv: Linear
        @ModuleInfo var proj: Linear

        init(_ config: CohereCompassConfiguration.VisionConfiguration) {
            heads = config.heads
            headDim = config.hiddenSize / config.heads
            scale = pow(Float(headDim), -0.5)
            _qkv.wrappedValue = Linear(
                config.hiddenSize, config.hiddenSize * 3, bias: true)
            _proj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray, boundaries: [Int], frequencies: MLXArray?
        ) -> MLXArray {
            let length = x.dim(0)
            let fused = qkv(x).reshaped(length, 3, heads, headDim)
            var q = fused[0..., 0]
            var k = fused[0..., 1]
            let v = fused[0..., 2]

            if let frequencies {
                (q, k) = applyRoPE(q, k, frequencies: frequencies)
            }

            let queries = q.transposed(1, 0, 2)[.newAxis, .ellipsis]
            let keys = k.transposed(1, 0, 2)[.newAxis, .ellipsis]
            let values = v.transposed(1, 0, 2)[.newAxis, .ellipsis]

            var outputs: [MLXArray] = []
            outputs.reserveCapacity(Swift.max(boundaries.count - 1, 1))
            for index in 0 ..< Swift.max(boundaries.count - 1, 0) {
                let (from, to) = (boundaries[index], boundaries[index + 1])
                guard to > from else { continue }
                outputs.append(
                    MLXFast.scaledDotProductAttention(
                        queries: queries[0..., 0..., from ..< to, 0...],
                        keys: keys[0..., 0..., from ..< to, 0...],
                        values: values[0..., 0..., from ..< to, 0...],
                        scale: scale, mask: .none))
            }
            let attended = outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 2)
            return proj(attended.transposed(0, 2, 1, 3).reshaped(length, -1))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "linear_fc1") var linear1: Linear
        @ModuleInfo(key: "linear_fc2") var linear2: Linear

        let activation = GELU(approximation: .precise)

        init(_ config: CohereCompassConfiguration.VisionConfiguration) {
            _linear1.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize, bias: true)
            _linear2.wrappedValue = Linear(
                config.intermediateSize, config.hiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear2(activation(linear1(x)))
        }
    }

    final class Block: Module {
        @ModuleInfo var norm1: LayerNorm
        @ModuleInfo var norm2: LayerNorm
        @ModuleInfo var attn: Attention
        @ModuleInfo var mlp: MLP

        init(_ config: CohereCompassConfiguration.VisionConfiguration) {
            _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _attn.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(config)
            super.init()
        }

        func callAsFunction(
            _ hiddenStates: MLXArray, boundaries: [Int], frequencies: MLXArray?
        ) -> MLXArray {
            let h =
                hiddenStates
                + attn(norm1(hiddenStates), boundaries: boundaries, frequencies: frequencies)
            return h + mlp(norm2(h))
        }
    }

    final class VisionModel: Module {
        let config: CohereCompassConfiguration.VisionConfiguration
        let rotaryDimensions: Int

        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        @ModuleInfo(key: "pos_embed") var positionEmbed: Embedding
        @ModuleInfo var blocks: [Block]
        @ModuleInfo var merger: PatchMerger
        @ModuleInfo(key: "deepstack_merger_list") var deepstackMergers: [PatchMerger]

        init(_ config: CohereCompassConfiguration.VisionConfiguration) {
            precondition(
                config.modelType == "cohere_compass_vision",
                "Unsupported vision modelType: \(config.modelType)")
            self.config = config
            rotaryDimensions = (config.hiddenSize / config.heads) / 2
            _patchEmbed.wrappedValue = PatchEmbed(config)
            _positionEmbed.wrappedValue = Embedding(
                embeddingCount: config.positionEmbeddings, dimensions: config.hiddenSize)
            _blocks.wrappedValue = (0 ..< config.depth).map { _ in Block(config) }
            _merger.wrappedValue = PatchMerger(config)
            _deepstackMergers.wrappedValue = config.deepstackVisualIndexes.map { _ in
                PatchMerger(config, normalizeAfterShuffle: true)
            }
            super.init()
        }

        /// Bilinearly resamples the learned position grid onto each image's patch grid,
        /// then reorders it into merge blocks so it lines up with the folded patches.
        func interpolatedPositions(grid: [[Int]]) -> MLXArray {
            let side = config.gridPerSide
            let merge = config.spatialMergeSize
            var corners: [[Int32]] = Array(repeating: [], count: 4)
            var weights: [[Float]] = Array(repeating: [], count: 4)

            for entry in grid {
                let (h, w) = (entry[1], entry[2])
                let rowPositions = (0 ..< h).map { index -> Float in
                    h == 1 ? 0 : Float(index) * Float(side - 1) / Float(h - 1)
                }
                let columnPositions = (0 ..< w).map { index -> Float in
                    w == 1 ? 0 : Float(index) * Float(side - 1) / Float(w - 1)
                }
                let rowFloor = rowPositions.map { Int($0) }
                let columnFloor = columnPositions.map { Int($0) }
                for (rowIndex, row) in rowFloor.enumerated() {
                    let rowCeil = Swift.min(row + 1, side - 1)
                    let dh = rowPositions[rowIndex] - Float(row)
                    for (columnIndex, column) in columnFloor.enumerated() {
                        let columnCeil = Swift.min(column + 1, side - 1)
                        let dw = columnPositions[columnIndex] - Float(column)
                        corners[0].append(Int32(row * side + column))
                        corners[1].append(Int32(row * side + columnCeil))
                        corners[2].append(Int32(rowCeil * side + column))
                        corners[3].append(Int32(rowCeil * side + columnCeil))
                        weights[0].append((1 - dh) * (1 - dw))
                        weights[1].append((1 - dh) * dw)
                        weights[2].append(dh * (1 - dw))
                        weights[3].append(dh * dw)
                    }
                }
            }

            let indices = MLXArray(corners.flatMap { $0 }, [4, corners[0].count])
            let weightArray = MLXArray(weights.flatMap { $0 }, [4, weights[0].count])
            let embedded =
                positionEmbed(indices) * weightArray[.ellipsis, .newAxis]
            let flatPositions = embedded.sum(axis: 0)

            var rows: [MLXArray] = []
            var offset = 0
            for entry in grid {
                let (t, h, w) = (entry[0], entry[1], entry[2])
                let count = h * w
                var block = flatPositions[offset ..< (offset + count)]
                offset += count
                block = tiled(block, repetitions: [t, 1]).reshaped(t, h, w, -1)
                block =
                    block
                    .reshaped(t, h / merge, merge, w / merge, merge, block.dim(-1))
                    .transposed(0, 1, 3, 2, 4, 5)
                    .reshaped(-1, block.dim(-1))
                rows.append(block)
            }
            return concatenated(rows, axis: 0)
        }

        /// Row and column frequencies, indexed by each patch's position *within* its merge
        /// block rather than in raster order.
        func rotaryFrequencies(grid: [[Int]]) -> MLXArray {
            let merge = config.spatialMergeSize
            let maxSide = grid.flatMap { [$0[1], $0[2]] }.max() ?? 1
            let inverse =
                1
                / pow(
                    MLXArray(Float(10000)),
                    MLXArray(stride(from: 0, to: rotaryDimensions, by: 2).map(Float.init))
                        / Float(rotaryDimensions))
            let table = (MLXArray(0 ..< maxSide).asType(.float32)[0..., .newAxis] * inverse)
                .reshaped(maxSide, -1)

            var rowIDs: [Int32] = []
            var columnIDs: [Int32] = []
            for entry in grid {
                let (t, h, w) = (entry[0], entry[1], entry[2])
                var rows: [Int32] = []
                var columns: [Int32] = []
                for blockRow in 0 ..< (h / merge) {
                    for blockColumn in 0 ..< (w / merge) {
                        for innerRow in 0 ..< merge {
                            for innerColumn in 0 ..< merge {
                                rows.append(Int32(blockRow * merge + innerRow))
                                columns.append(Int32(blockColumn * merge + innerColumn))
                            }
                        }
                    }
                }
                for _ in 0 ..< t {
                    rowIDs.append(contentsOf: rows)
                    columnIDs.append(contentsOf: columns)
                }
            }

            return concatenated(
                [table[MLXArray(rowIDs)], table[MLXArray(columnIDs)]], axis: -1)
        }

        private func forwardSingle(_ hiddenStates: MLXArray, grid: [[Int]]) -> (
            features: MLXArray, deepstack: [MLXArray]
        ) {
            var h = patchEmbed(hiddenStates)
            h = h + interpolatedPositions(grid: grid).asType(h.dtype)
            let frequencies = config.useRope ? rotaryFrequencies(grid: grid) : nil

            var boundaries = [0]
            for entry in grid {
                let frame = entry[1] * entry[2]
                for _ in 0 ..< entry[0] {
                    boundaries.append(boundaries[boundaries.count - 1] + frame)
                }
            }

            var deepstack: [MLXArray] = []
            for (index, block) in blocks.enumerated() {
                h = block(h, boundaries: boundaries, frequencies: frequencies)
                if let position = config.deepstackVisualIndexes.firstIndex(of: index) {
                    deepstack.append(deepstackMergers[position](h))
                }
            }
            return (merger(h), deepstack)
        }

        /// One image at a time. Upstream keeps each image's attention graph independent
        /// because the flattened path's accumulation order changes its output enough to
        /// alter decoding through a 27-layer tower.
        func callAsFunction(_ hiddenStates: MLXArray, grid: [[Int]]) -> (
            features: MLXArray, deepstack: [MLXArray]
        ) {
            guard grid.count > 1 else {
                return forwardSingle(hiddenStates, grid: grid)
            }

            var features: [MLXArray] = []
            var deepstack: [[MLXArray]] = Array(
                repeating: [], count: config.deepstackVisualIndexes.count)
            var offset = 0
            for entry in grid {
                let length = entry[0] * entry[1] * entry[2]
                let (imageFeatures, imageDeepstack) = forwardSingle(
                    hiddenStates[offset ..< (offset + length)], grid: [entry])
                offset += length
                features.append(imageFeatures)
                for (index, value) in imageDeepstack.enumerated() {
                    deepstack[index].append(value)
                }
            }
            return (
                concatenated(features, axis: 0),
                deepstack.map { concatenated($0, axis: 0) }
            )
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized: [String: MLXArray] = [:]
            for (key, value) in weights {
                if key.contains("position_ids") { continue }
                if key.hasSuffix("patch_embed.proj.weight"), value.ndim == 5,
                    value.dim(1) == config.channels
                {
                    sanitized[key] = value.transposed(0, 2, 3, 4, 1)
                } else {
                    sanitized[key] = value
                }
            }
            return sanitized
        }
    }
}

// MARK: - Model

public final class CohereCompass: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "language_model") private var languageModel: LanguageHalf
    @ModuleInfo(key: "vision_tower") private var visionTower: CohereCompassVision.VisionModel?

    /// Matches the checkpoint's `language_model.model.*` and `language_model.lm_head`.
    fileprivate final class LanguageHalf: Module {
        @ModuleInfo var model: CohereCompassText.Backbone
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        init(_ config: CohereCompassConfiguration.TextConfiguration) {
            _model.wrappedValue = CohereCompassText.Backbone(config)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.hiddenSize, config.vocabularySize, bias: false)
            }
            super.init()
        }
    }

    public let config: CohereCompassConfiguration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] {
        Array(
            repeating: config.textConfiguration.kvHeads,
            count: config.textConfiguration.hiddenLayers)
    }

    public init(_ config: CohereCompassConfiguration) {
        self.config = config
        _languageModel.wrappedValue = LanguageHalf(config.textConfiguration)
        _visionTower.wrappedValue = config.visionConfiguration.map {
            CohereCompassVision.VisionModel($0)
        }
    }

    /// Sliding layers keep a rotating cache at the architecture's window; the rest keep
    /// everything.
    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        let window = config.textConfiguration.slidingWindow ?? 4096
        return try config.textConfiguration.layerTypes.map { layerType in
            layerType == "sliding_attention"
                ? try makeSlidingWindowKVCache(configuration: cacheConfiguration, window: window)
                : try makeAttentionKVCache(configuration: cacheConfiguration)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(inputs: inputs, inputEmbedding: nil, cache: cache, positionIDs: nil)
    }

    private func logits(
        inputs: MLXArray?, inputEmbedding: MLXArray?, cache: [KVCache]?,
        positionIDs: MLXArray?, visualPositionMask: MLXArray? = nil,
        deepstackEmbeddings: [MLXArray]? = nil
    ) -> MLXArray {
        let hidden = languageModel.model(
            inputs: inputs, inputEmbedding: inputEmbedding, mask: nil, cache: cache,
            positionIDs: positionIDs, visualPositionMask: visualPositionMask,
            deepstackEmbeddings: deepstackEmbeddings)
        var out =
            languageModel.lmHead.map { CohereCompassText.rowwise($0, hidden) }
            ?? CohereCompassText.rowwise(
                LinearFromEmbedding(languageModel.model.embedTokens), hidden)
        if let scale = config.textConfiguration.logitScale {
            out = out * scale
        }
        return out
    }

    /// The tied head, as a layer so it can go through the row-wise wrapper unchanged.
    private final class LinearFromEmbedding: Module, UnaryLayer {
        @ModuleInfo var embedding: Embedding

        init(_ embedding: Embedding) {
            _embedding.wrappedValue = embedding
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            embedding.asLinear(x)
        }
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }

        guard let image = input.image, let visionTower, let imageTokenID = config.imageTokenID
        else {
            let total = tokens.dim(1)
            let processed = try prefill.forEachChunk(total: total) { range in
                _ = logits(
                    inputs: tokens[0..., range], inputEmbedding: nil, cache: cache,
                    positionIDs: nil)
                asyncEval(cache)
            }
            if processed > 0 { eval(cache) }
            prefill.progress?(total, total)
            return .logits(
                LMOutput(
                    logits: logits(
                        inputs: tokens[0..., processed...], inputEmbedding: nil,
                        cache: cache, positionIDs: nil)))
        }

        guard let frames = image.frames, !frames.isEmpty else {
            throw VLMError.imageRequired
        }
        let grid = frames.map { [$0.t, $0.h, $0.w] }
        let (features, deepstack) = visionTower(
            image.pixels.asType(visionTower.patchEmbed.proj.weight.dtype), grid: grid)

        let embedded = languageModel.model.embedTokens(tokens)
        let selected = tokens .== MLXArray(Int32(imageTokenID))
        let merged = Self.scatter(
            embedded: embedded, features: features.asType(embedded.dtype),
            selected: selected)

        // A one-shot forward: the deep-stack features are indexed by their position in the
        // whole prompt, so a chunk would have to carry its own window of them.
        let total = merged.dim(1)
        prefill.progress?(total, total)
        return .logits(
            LMOutput(
                logits: logits(
                    inputs: nil, inputEmbedding: merged, cache: cache, positionIDs: nil,
                    visualPositionMask: selected, deepstackEmbeddings: deepstack)))
    }

    private static func scatter(
        embedded: MLXArray, features: MLXArray, selected: MLXArray
    ) -> MLXArray {
        let flat = embedded.reshaped(-1, embedded.dim(-1))
        let flatSelected = selected.reshaped(-1)
        let rank = clip(
            cumsum(flatSelected.asType(.int32), axis: 0) - 1,
            min: 0, max: features.dim(0) - 1)
        let source = takeAlong(
            features,
            broadcast(rank[0..., .newAxis], to: [flat.dim(0), features.dim(-1)]),
            axis: 0)
        return MLX.where(
            broadcast(flatSelected[0..., .newAxis], to: flat.shape), source, flat
        ).reshaped(embedded.shape)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var moved: [String: MLXArray] = [:]
        let tied = config.textConfiguration.tieWordEmbeddings

        for (rawKey, value) in weights {
            if tied, rawKey == "lm_head.weight" { continue }
            var key = rawKey
            if key.hasPrefix("model.language_model.") {
                key =
                    "language_model.model."
                    + String(key.dropFirst("model.language_model.".count))
            } else if key.hasPrefix("model.visual.") {
                key = "vision_tower." + String(key.dropFirst("model.visual.".count))
            } else if key.hasPrefix("lm_head.") {
                key = "language_model." + key
            }
            moved[key] = value
        }

        guard let visionTower else { return moved }
        var vision: [String: MLXArray] = [:]
        var rest: [String: MLXArray] = [:]
        for (key, value) in moved {
            if key.hasPrefix("vision_tower.") {
                vision[String(key.dropFirst("vision_tower.".count))] = value
            } else {
                rest[key] = value
            }
        }
        for (key, value) in visionTower.sanitize(weights: vision) {
            rest["vision_tower." + key] = value
        }
        return rest
    }
}

extension CohereCompass: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
