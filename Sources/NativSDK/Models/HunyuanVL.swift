import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct HunyuanVLConfiguration: Decodable, Sendable {

    public struct VisionConfiguration: Decodable, Sendable {
        public var modelType: String = "hunyuan_vl"
        public var hiddenSize: Int = 1152
        public var outHiddenSize: Int = 1024
        public var hiddenLayers: Int = 27
        public var attentionHeads: Int = 16
        public var intermediateSize: Int = 4304
        public var patchSize: Int = 16
        public var channels: Int = 3
        public var spatialMergeSize: Int = 2
        public var rmsNormEps: Float = 1e-5
        public var maxImageSize: Int = 2048

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case outHiddenSize = "out_hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case intermediateSize = "intermediate_size"
            case patchSize = "patch_size"
            case channels = "num_channels"
            case spatialMergeSize = "spatial_merge_size"
            case rmsNormEps = "rms_norm_eps"
            case maxImageSize = "max_image_size"
        }

        init() {}

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
            outHiddenSize =
                try c.decodeIfPresent(Int.self, forKey: .outHiddenSize) ?? outHiddenSize
            hiddenLayers =
                try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? hiddenLayers
            attentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? attentionHeads
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? patchSize
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? channels
            spatialMergeSize =
                try c.decodeIfPresent(Int.self, forKey: .spatialMergeSize) ?? spatialMergeSize
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? rmsNormEps
            maxImageSize =
                try c.decodeIfPresent(Int.self, forKey: .maxImageSize) ?? maxImageSize
        }

        public var maxPatches: Int {
            let side = maxImageSize / patchSize
            return side * side
        }
        public var positions: Int { maxPatches + 1 }
        public var positionEdge: Int { Int(Double(positions).squareRoot()) }
    }

    public struct TextConfiguration: Decodable, Sendable {
        public var modelType: String = "hunyuan_vl"
        public var vocabularySize: Int = 120_818
        public var hiddenSize: Int = 1024
        public var hiddenLayers: Int = 24
        public var attentionHeads: Int = 16
        public var intermediateSize: Int = 3584
        public var attentionBias: Bool = false
        public var mlpBias: Bool = false
        public var useQKNorm: Bool = true
        public var ropeTheta: Float = 10000
        public var maxPositionEmbeddings: Int = 32768
        public var rmsNormEps: Float = 1e-5
        public var tieWordEmbeddings: Bool = true

        private var declaredKVHeads: Int?
        private var declaredHeadDim: Int?
        public var ropeScaling: [String: StringOrNumber]?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case declaredKVHeads = "num_key_value_heads"
            case declaredHeadDim = "head_dim"
            case intermediateSize = "intermediate_size"
            case attentionBias = "attention_bias"
            case mlpBias = "mlp_bias"
            case useQKNorm = "use_qk_norm"
            case ropeTheta = "rope_theta"
            case ropeScaling = "rope_scaling"
            case maxPositionEmbeddings = "max_position_embeddings"
            case rmsNormEps = "rms_norm_eps"
            case tieWordEmbeddings = "tie_word_embeddings"
        }

        init() {}

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
            vocabularySize =
                try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? vocabularySize
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
            hiddenLayers =
                try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? hiddenLayers
            attentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? attentionHeads
            declaredKVHeads = try c.decodeIfPresent(Int.self, forKey: .declaredKVHeads)
            declaredHeadDim = try c.decodeIfPresent(Int.self, forKey: .declaredHeadDim)
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
            attentionBias =
                try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? attentionBias
            mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? mlpBias
            useQKNorm = try c.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? useQKNorm
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? ropeTheta
            maxPositionEmbeddings =
                try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
                ?? maxPositionEmbeddings
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? rmsNormEps
            tieWordEmbeddings =
                try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings)
                ?? tieWordEmbeddings

            if c.contains(.ropeScaling) {
                ropeScaling = try c.decodeIfPresent(
                    [String: StringOrNumber].self, forKey: .ropeScaling)
            } else {
                ropeScaling = Self.defaultRopeScaling
            }
        }

        /// The dataclass default, which a checkpoint that omits the key still gets.
        static let defaultRopeScaling: [String: StringOrNumber] = [
            "alpha": .float(1000),
            "beta_fast": .int(32),
            "beta_slow": .int(1),
            "factor": .float(1),
            "mscale": .float(1),
            "mscale_all_dim": .float(1),
            "type": .string("xdrope"),
            "xdrope_section": .ints([16, 16, 16, 16]),
        ]

        public var kvHeads: Int { declaredKVHeads ?? attentionHeads }
        public var headDim: Int { declaredHeadDim ?? hiddenSize / attentionHeads }
        public var xdropeSection: [Int]? {
            ropeScaling?["xdrope_section"]?.asInts()
        }
        public var ropeType: String? {
            if case .string(let value)? = ropeScaling?["type"] { return value }
            return nil
        }

        /// The base the rotary table is built on. An `alpha` under an xdrope or dynamic
        /// scaling stretches it by `alpha ** (dim / (dim - 2))`.
        public var effectiveRopeBase: Float {
            guard let type = ropeType, ["xdrope", "dynamic"].contains(type),
                let alpha = ropeScaling?["alpha"]?.asFloat(), alpha != 0
            else {
                return ropeTheta
            }
            return ropeTheta * pow(alpha, Float(headDim) / Float(headDim - 2))
        }
    }

    public var modelType: String
    public var textConfiguration: TextConfiguration
    public var visionConfiguration: VisionConfiguration
    public var imageTokenID: Int
    public var imageStartTokenID: Int
    public var imageEndTokenID: Int
    public var imageNewlineTokenID: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case imageTokenID = "image_token_id"
        case imageStartTokenID = "image_start_token_id"
        case imageEndTokenID = "image_end_token_id"
        case imageNewlineTokenID = "image_newline_token_id"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "hunyuan_vl"
        // Both halves also read the top level, so a flat configuration works.
        textConfiguration =
            try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? TextConfiguration(from: decoder)
        visionConfiguration =
            try c.decodeIfPresent(VisionConfiguration.self, forKey: .visionConfiguration)
            ?? VisionConfiguration(from: decoder)
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? 120_120
        imageStartTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imageStartTokenID) ?? 120_118
        imageEndTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imageEndTokenID) ?? 120_119
        imageNewlineTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imageNewlineTokenID) ?? 120_121
    }
}

// MARK: - Text

enum HunyuanVLText {

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    }

    /// The rotary table, built once and extended as the sequence grows.
    final class RotaryTable {
        let inverseFrequencies: MLXArray
        private var cosine: MLXArray?
        private var sine: MLXArray?
        private var cachedLength = 0

        init(_ config: HunyuanVLConfiguration.TextConfiguration) {
            let dimensions = config.headDim
            inverseFrequencies =
                1
                / pow(
                    MLXArray(config.effectiveRopeBase),
                    MLXArray(stride(from: 0, to: dimensions, by: 2).map(Float.init))
                        / Float(dimensions))
        }

        func callAsFunction(length: Int, dtype: DType) -> (cos: MLXArray, sin: MLXArray) {
            if length > cachedLength {
                cachedLength = length
                let positions = MLXArray(0 ..< length).asType(.float32)
                let frequencies = outer(positions, inverseFrequencies)
                let embedding = concatenated([frequencies, frequencies], axis: -1)
                cosine = cos(embedding).asType(dtype)
                sine = sin(embedding).asType(dtype)
            }
            return (cosine![..<length], sine![..<length])
        }
    }

    /// Rotates with one position row per section of the head dimension.
    ///
    /// The section list is doubled to cover both halves of the rotation, and section `i`
    /// takes its positions from axis `i % axes` -- so the halves repeat the axis order
    /// rather than continuing it.
    static func applyXDRoPE(
        _ q: MLXArray, _ k: MLXArray, cos: MLXArray, sin: MLXArray,
        positionIDs: MLXArray, section: [Int]
    ) -> (MLXArray, MLXArray) {
        let axes = section.count
        let (batch, length) = (positionIDs.dim(0), positionIDs.dim(2))

        func arrange(_ table: MLXArray) -> MLXArray {
            var gathered = table[positionIDs]
            gathered = gathered.transposed(0, 2, 1, 3).reshaped(batch, length, axes, -1)
            let doubled = section + section
            var bounds: [Int] = []
            var running = 0
            for width in doubled.dropLast() {
                running += width
                bounds.append(running)
            }
            let parts = split(gathered, indices: bounds, axis: -1)
            let selected = parts.enumerated().map { index, part in
                part[0..., 0..., index % axes, 0...]
            }
            return concatenated(selected, axis: -1).reshaped(batch, 1, length, -1)
        }

        let cosine = arrange(cos).asType(.float32)
        let sine = arrange(sin).asType(.float32)
        let (qDtype, kDtype) = (q.dtype, k.dtype)
        let q32 = q.asType(.float32)
        let k32 = k.asType(.float32)
        return (
            (q32 * cosine + rotateHalf(q32) * sine).asType(qDtype),
            (k32 * cosine + rotateHalf(k32) * sine).asType(kDtype)
        )
    }

    static func applyRoPE(_ q: MLXArray, _ k: MLXArray, cos: MLXArray, sin: MLXArray) -> (
        MLXArray, MLXArray
    ) {
        let cosine = cos[.newAxis, .newAxis, 0..., 0...].asType(.float32)
        let sine = sin[.newAxis, .newAxis, 0..., 0...].asType(.float32)
        let (qDtype, kDtype) = (q.dtype, k.dtype)
        let q32 = q.asType(.float32)
        let k32 = k.asType(.float32)
        return (
            (q32 * cosine + rotateHalf(q32) * sine).asType(qDtype),
            (k32 * cosine + rotateHalf(k32) * sine).asType(kDtype)
        )
    }

    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let useQKNorm: Bool
        let section: [Int]?
        let rotary: RotaryTable

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "query_layernorm") var queryNorm: RMSNorm?
        @ModuleInfo(key: "key_layernorm") var keyNorm: RMSNorm?

        init(_ config: HunyuanVLConfiguration.TextConfiguration) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(config.headDim), -0.5)
            useQKNorm = config.useQKNorm
            section = config.xdropeSection
            rotary = RotaryTable(config)

            _qProj.wrappedValue = Linear(
                config.hiddenSize, heads * headDim, bias: config.attentionBias)
            _kProj.wrappedValue = Linear(
                config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
            _vProj.wrappedValue = Linear(
                config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
            _oProj.wrappedValue = Linear(
                heads * headDim, config.hiddenSize, bias: config.attentionBias)
            if config.useQKNorm {
                _queryNorm.wrappedValue = RMSNorm(
                    dimensions: headDim, eps: config.rmsNormEps)
                _keyNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
            }
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXArray?, cache: KVCache?, positionIDs: MLXArray?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))
            var queries = qProj(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            var keys = kProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
            let values = vProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

            let offset = cache?.offset ?? 0
            let keyLength = L + offset
            let (cosine, sine) = rotary(length: keyLength, dtype: values.dtype)

            if let section, let positionIDs, offset == 0 {
                (queries, keys) = applyXDRoPE(
                    queries, keys, cos: cosine, sin: sine, positionIDs: positionIDs,
                    section: section)
            } else {
                let slicedCos = offset > 0 ? cosine[(-L)...] : cosine
                let slicedSin = offset > 0 ? sine[(-L)...] : sine
                (queries, keys) = applyRoPE(
                    queries, keys, cos: slicedCos, sin: slicedSin)
            }

            // The head norms come after the rotation, not before it.
            if useQKNorm, let queryNorm, let keyNorm {
                queries = queryNorm(queries)
                keys = keyNorm(keys)
            }

            var mask = mask
            let output: MLXArray
            if let cache {
                let (cachedKeys, cachedValues) = cache.update(keys: keys, values: values)
                if let value = mask, value.dim(-1) != cachedKeys.dim(2) {
                    mask = value[.ellipsis, ..<cachedKeys.dim(2)]
                }
                output = MLXFast.scaledDotProductAttention(
                    queries: queries, keys: cachedKeys, values: cachedValues,
                    scale: scale, mask: mask.map { .array($0) } ?? .none)
            } else {
                output = MLXFast.scaledDotProductAttention(
                    queries: queries, keys: keys, values: values, scale: scale,
                    mask: mask.map { .array($0) } ?? .none)
            }

            return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        }
    }

    final class DecoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo var mlp: SwiGLUMLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: HunyuanVLConfiguration.TextConfiguration) {
            _attention.wrappedValue = Attention(config)
            _mlp.wrappedValue = SwiGLUMLP(
                inputDimensions: config.hiddenSize,
                hiddenDimensions: config.intermediateSize,
                bias: config.mlpBias)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXArray?, cache: KVCache?, positionIDs: MLXArray?
        ) -> MLXArray {
            let h =
                x
                + attention(
                    inputLayerNorm(x), mask: mask, cache: cache, positionIDs: positionIDs)
            return h + mlp(postAttentionLayerNorm(h))
        }
    }

    final class Backbone: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo var layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm

        init(_ config: HunyuanVLConfiguration.TextConfiguration) {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
                DecoderLayer(config)
            }
            _norm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            super.init()
        }

        func callAsFunction(
            inputs: MLXArray?, inputEmbedding: MLXArray?, cache: [KVCache]?,
            positionIDs: MLXArray?
        ) -> MLXArray {
            guard var h = inputEmbedding ?? inputs.map({ embedTokens($0) }) else {
                fatalError("either inputs or inputEmbedding must be provided")
            }

            var mask: MLXArray?
            if h.dim(1) > 1 {
                let mode = createAttentionMask(
                    h: h, cache: cache?.first, returnArray: true)
                if case .array(let value) = mode { mask = value }
            }

            for (index, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[index], positionIDs: positionIDs)
            }
            return norm(h)
        }
    }

    /// The four position rows XD-RoPE rotates against: the sequential index, the image
    /// index, and the row and column inside each image's merged grid.
    ///
    /// An image's span is `(gridW + 1) * gridH` positions -- the extra column per row is
    /// the newline the merger appends.
    static func xdropePositions(
        tokens: [Int32], imageTokenID: Int, grid: [[Int]], spatialMergeSize: Int,
        axes: Int
    ) -> MLXArray {
        let length = tokens.count
        let sequential = (0 ..< length).map(Int32.init)
        var temporal = sequential
        var rows = sequential
        var columns = sequential

        let starts = tokens.indices.filter { tokens[$0] == Int32(imageTokenID) }
        if !grid.isEmpty, !starts.isEmpty {
            for (imageIndex, start) in starts.enumerated() {
                guard let entry = grid.first else { break }
                let gridH = entry[1] / spatialMergeSize
                let gridW = entry[2] / spatialMergeSize
                let position = start + 1
                let end = Swift.min(position + (gridW + 1) * gridH, length)
                guard end > position else { continue }

                var columnPattern: [Int32] = []
                var rowPattern: [Int32] = []
                for row in 0 ..< gridH {
                    for column in 0 ..< (gridW + 1) {
                        columnPattern.append(Int32(column))
                        rowPattern.append(Int32(row))
                    }
                }
                for (offset, index) in (position ..< end).enumerated() {
                    columns[index] = columnPattern[offset]
                    rows[index] = rowPattern[offset]
                    temporal[index] = Int32(imageIndex)
                }
            }
        }

        switch axes {
        case 4:
            return MLX.stacked([
                MLXArray(sequential), MLXArray(temporal), MLXArray(rows),
                MLXArray(columns),
            ])
        case 3:
            return MLX.stacked([MLXArray(temporal), MLXArray(rows), MLXArray(columns)])
        default:
            return MLX.stacked(Array(repeating: MLXArray(sequential), count: axes))
        }
    }
}

// MARK: - Vision

enum HunyuanVLVision {

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "dense_h_to_4h") var up: Linear
        @ModuleInfo(key: "dense_4h_to_h") var down: Linear

        init(_ config: HunyuanVLConfiguration.VisionConfiguration) {
            _up.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize, bias: true)
            _down.wrappedValue = Linear(
                config.intermediateSize, config.hiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            down(gelu(up(x)))
        }
    }

    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        /// Queries are scored in blocks so the scores never materialize for a whole page
        /// of patches at once.
        static let chunk = 1024

        init(_ config: HunyuanVLConfiguration.VisionConfiguration) {
            heads = config.attentionHeads
            headDim = config.hiddenSize / config.attentionHeads
            scale = pow(Float(headDim), -0.5)
            _qProj.wrappedValue = Linear(
                config.hiddenSize, heads * headDim, bias: true)
            _kProj.wrappedValue = Linear(
                config.hiddenSize, heads * headDim, bias: true)
            _vProj.wrappedValue = Linear(
                config.hiddenSize, heads * headDim, bias: true)
            _oProj.wrappedValue = Linear(
                heads * headDim, config.hiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))
            let queries = qProj(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            let keys = kProj(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
            let values = vProj(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)

            var outputs: [MLXArray] = []
            var start = 0
            while start < L {
                let end = Swift.min(start + Self.chunk, L)
                outputs.append(
                    MLXFast.scaledDotProductAttention(
                        queries: queries[0..., 0..., start ..< end, 0...],
                        keys: keys, values: values, scale: scale, mask: .none))
                start = end
            }
            let attended = outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 2)
            return oProj(attended.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        }
    }

    final class Block: Module {
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: LayerNorm
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo var mlp: MLP

        init(_ config: HunyuanVLConfiguration.VisionConfiguration) {
            _inputLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _postAttentionLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _attention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(config)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let h = x + attention(inputLayerNorm(x))
            return h + mlp(postAttentionLayerNorm(h))
        }
    }

    /// A per-patch convolution plus a position table resampled to each image's grid.
    ///
    /// The table's first row is a class position the tower never uses, and the resampling
    /// scale carries an odd tenth -- `source / (target + 0.1)` -- which is the
    /// checkpoint's own convention rather than a rounding choice.
    final class PatchEmbed: Module {
        let config: HunyuanVLConfiguration.VisionConfiguration

        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Conv2d
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        init(_ config: HunyuanVLConfiguration.VisionConfiguration) {
            self.config = config
            _patchEmbedding.wrappedValue = Conv2d(
                inputChannels: config.channels, outputChannels: config.hiddenSize,
                kernelSize: .init(config.patchSize), stride: .init(config.patchSize),
                bias: true)
            _positionEmbedding.wrappedValue = Embedding(
                embeddingCount: config.positions, dimensions: config.hiddenSize)
            super.init()
        }

        func callAsFunction(_ pixelValues: MLXArray, grid: [[Int]]) -> MLXArray {
            let patches = pixelValues.dim(0)
            let reshaped =
                pixelValues
                .reshaped(patches, config.channels, config.patchSize, config.patchSize)
                .transposed(0, 2, 3, 1)
            let embedded = patchEmbedding(reshaped)
                .reshaped(1, patches, config.hiddenSize)

            let edge = config.positionEdge
            let table = positionEmbedding.weight[1...]
                .reshaped(1, edge, edge, config.hiddenSize)
            let positions = grid.map { entry in
                interpolate(table, height: entry[1], width: entry[2])
                    .reshaped(1, -1, config.hiddenSize)
            }
            return embedded + concatenated(positions, axis: 1)
        }

        func interpolate(_ table: MLXArray, height: Int, width: Int) -> MLXArray {
            let (sourceHeight, sourceWidth) = (table.dim(1), table.dim(2))
            if sourceHeight == height, sourceWidth == width {
                return table
            }

            let heightScale = Float(sourceHeight) / (Float(height) + 0.1)
            let widthScale = Float(sourceWidth) / (Float(width) + 0.1)
            let rowCoordinates =
                (MLXArray(0 ..< height).asType(.float32) + 0.5) * heightScale - 0.5
            let columnCoordinates =
                (MLXArray(0 ..< width).asType(.float32) + 0.5) * widthScale - 0.5

            let rowFloor = rowCoordinates.asType(.int32)
            let columnFloor = columnCoordinates.asType(.int32)
            let rowCeil = minimum(rowFloor + 1, sourceHeight - 1)
            let columnCeil = minimum(columnFloor + 1, sourceWidth - 1)
            let dRow = (rowCoordinates - rowFloor.asType(.float32))[
                0..., .newAxis, .newAxis]
            let dColumn = (columnCoordinates - columnFloor.asType(.float32))[
                .newAxis, 0..., .newAxis]

            let plane = table[0]
            let topLeft = plane[rowFloor][0..., columnFloor]
            let topRight = plane[rowFloor][0..., columnCeil]
            let bottomLeft = plane[rowCeil][0..., columnFloor]
            let bottomRight = plane[rowCeil][0..., columnCeil]

            let result =
                (1 - dRow) * (1 - dColumn) * topLeft
                + (1 - dRow) * dColumn * topRight
                + dRow * (1 - dColumn) * bottomLeft
                + dRow * dColumn * bottomRight
            return result[.newAxis].asType(table.dtype)
        }
    }

    /// Folds each merge block into one row, appends a newline column per row, and wraps
    /// the whole image in learned begin and end rows.
    final class PatchMerger: Module {
        let config: HunyuanVLConfiguration.VisionConfiguration

        @ModuleInfo(key: "before_rms") var beforeNorm: RMSNorm
        @ModuleInfo(key: "after_rms") var afterNorm: RMSNorm
        @ModuleInfo var proj: [Module]
        @ModuleInfo var mlp: Linear

        @ParameterInfo(key: "image_newline") var imageNewline: MLXArray
        @ParameterInfo(key: "image_begin") var imageBegin: MLXArray
        @ParameterInfo(key: "image_end") var imageEnd: MLXArray
        @ParameterInfo(key: "image_sep") var imageSeparator: MLXArray

        init(_ config: HunyuanVLConfiguration.VisionConfiguration) {
            self.config = config
            let merged = config.hiddenSize * 2
            let final = config.hiddenSize * 4

            _beforeNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _afterNorm.wrappedValue = RMSNorm(
                dimensions: config.outHiddenSize, eps: config.rmsNormEps)
            // The activation keeps index 1 so the convolutions stay at proj.0 and proj.2.
            _proj.wrappedValue = [
                Conv2d(
                    inputChannels: config.hiddenSize, outputChannels: merged,
                    kernelSize: .init(config.spatialMergeSize),
                    stride: .init(config.spatialMergeSize), bias: true),
                GELU(),
                Conv2d(
                    inputChannels: merged, outputChannels: final, kernelSize: .init(1),
                    bias: true),
            ]
            _mlp.wrappedValue = Linear(final, config.outHiddenSize, bias: true)

            _imageNewline.wrappedValue = MLXArray.zeros([final])
            _imageBegin.wrappedValue = MLXArray.zeros([config.outHiddenSize])
            _imageEnd.wrappedValue = MLXArray.zeros([config.outHiddenSize])
            _imageSeparator.wrappedValue = MLXArray.zeros([config.outHiddenSize])
            super.init()
        }

        func callAsFunction(_ hiddenStates: MLXArray, gridHeight: Int, gridWidth: Int)
            -> MLXArray
        {
            let B = hiddenStates.dim(0)
            let final = config.hiddenSize * 4
            var x = beforeNorm(hiddenStates)
                .reshaped(B, gridHeight, gridWidth, config.hiddenSize)

            for layer in proj {
                if let convolution = layer as? Conv2d {
                    x = convolution(x)
                } else if let activation = layer as? GELU {
                    x = activation(x)
                }
            }

            let mergedHeight = gridHeight / config.spatialMergeSize
            let mergedWidth = gridWidth / config.spatialMergeSize
            x = x.reshaped(B, mergedHeight, mergedWidth, final)

            let newlines = broadcast(
                imageNewline[.newAxis, .newAxis, .newAxis, 0...],
                to: [B, mergedHeight, 1, final])
            x = concatenated([x, newlines], axis: 2)
                .reshaped(B, mergedHeight * (mergedWidth + 1), final)
            x = mlp(x)

            let begin = broadcast(
                imageBegin[.newAxis, .newAxis, 0...], to: [B, 1, config.outHiddenSize])
            let end = broadcast(
                imageEnd[.newAxis, .newAxis, 0...], to: [B, 1, config.outHiddenSize])
            return afterNorm(concatenated([begin, x, end], axis: 1))
        }
    }

    final class VisionModel: Module {
        let config: HunyuanVLConfiguration.VisionConfiguration

        @ModuleInfo var embeddings: PatchEmbed
        @ModuleInfo var layers: [Block]
        @ModuleInfo var perceive: PatchMerger

        init(_ config: HunyuanVLConfiguration.VisionConfiguration) {
            precondition(
                config.modelType == "hunyuan_vl",
                "Unsupported vision modelType: \(config.modelType)")
            self.config = config
            _embeddings.wrappedValue = PatchEmbed(config)
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in Block(config) }
            _perceive.wrappedValue = PatchMerger(config)
            super.init()
        }

        func callAsFunction(_ pixelValues: MLXArray, grid: [[Int]]) -> MLXArray {
            var h = embeddings(pixelValues, grid: grid)
            for layer in layers {
                h = layer(h)
            }

            var merged: [MLXArray] = []
            var offset = 0
            for entry in grid {
                let length = entry[1] * entry[2]
                merged.append(
                    perceive(
                        h[0..., offset ..< (offset + length), 0...],
                        gridHeight: entry[1], gridWidth: entry[2]))
                offset += length
            }
            return concatenated(merged, axis: 1)
        }
    }
}

// MARK: - Model

public final class HunyuanVL: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionTower: HunyuanVLVision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: LanguageHalf

    fileprivate final class LanguageHalf: Module {
        @ModuleInfo var model: HunyuanVLText.Backbone
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        init(_ config: HunyuanVLConfiguration.TextConfiguration) {
            _model.wrappedValue = HunyuanVLText.Backbone(config)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.hiddenSize, config.vocabularySize, bias: false)
            }
            super.init()
        }
    }

    public let config: HunyuanVLConfiguration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] {
        Array(
            repeating: config.textConfiguration.kvHeads,
            count: config.textConfiguration.hiddenLayers)
    }

    public init(_ config: HunyuanVLConfiguration) {
        self.config = config
        _visionTower.wrappedValue = HunyuanVLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = LanguageHalf(config.textConfiguration)
    }

    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        try (0 ..< config.textConfiguration.hiddenLayers).map { _ in
            try makeAttentionKVCache(configuration: cacheConfiguration)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(inputs: inputs, inputEmbedding: nil, cache: cache, positionIDs: nil)
    }

    private func logits(
        inputs: MLXArray?, inputEmbedding: MLXArray?, cache: [KVCache]?,
        positionIDs: MLXArray?
    ) -> MLXArray {
        var positions = positionIDs
        let offset = cache?.first?.offset ?? 0

        if positions == nil, offset == 0, let inputs {
            positions = Self.positions(
                tokens: inputs, config: config)
        }

        let hidden = languageModel.model(
            inputs: inputs, inputEmbedding: inputEmbedding, cache: cache,
            positionIDs: positions)
        if let head = languageModel.lmHead {
            return head(hidden)
        }
        return languageModel.model.embedTokens.asLinear(hidden)
    }

    private static func positions(
        tokens: MLXArray, config: HunyuanVLConfiguration, grid: [[Int]] = []
    ) -> MLXArray? {
        guard let section = config.textConfiguration.xdropeSection else { return nil }
        var ids = tokens
        if ids.ndim == 1 {
            ids = ids.expandedDimensions(axis: 0)
        }
        let rows = (0 ..< ids.dim(0)).map { row in
            HunyuanVLText.xdropePositions(
                tokens: ids[row].asArray(Int32.self),
                imageTokenID: config.imageTokenID,
                grid: grid,
                spatialMergeSize: config.visionConfiguration.spatialMergeSize,
                axes: section.count)
        }
        return MLX.stacked(rows, axis: 0)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }

        guard let image = input.image else {
            let total = tokens.dim(1)
            let promptPositions = Self.positions(tokens: tokens, config: config)
            let processed = try prefill.forEachChunk(total: total) { range in
                _ = logits(
                    inputs: tokens[0..., range], inputEmbedding: nil, cache: cache,
                    positionIDs: promptPositions.map { $0[0..., 0..., range] })
                asyncEval(cache)
            }
            if processed > 0 { eval(cache) }
            prefill.progress?(total, total)
            let result = logits(
                inputs: tokens[0..., processed...], inputEmbedding: nil, cache: cache,
                positionIDs: promptPositions.map { $0[0..., 0..., processed...] })
            return .logits(LMOutput(logits: result))
        }

        guard let frames = image.frames, !frames.isEmpty else {
            throw VLMError.imageRequired
        }
        let grid = frames.map { [$0.t, $0.h, $0.w] }
        let features = visionTower(
            image.pixels.asType(
                visionTower.embeddings.patchEmbedding.weight.dtype), grid: grid)

        let embedded = languageModel.model.embedTokens(tokens)
        let selected = tokens .== MLXArray(Int32(config.imageTokenID))
        let merged = Self.scatter(
            embedded: embedded,
            features: features.reshaped(-1, features.dim(-1))
                .asType(embedded.dtype),
            selected: selected)

        let promptPositions = Self.positions(tokens: tokens, config: config, grid: grid)

        let total = merged.dim(1)
        let processed = try prefill.forEachChunk(total: total) { range in
            _ = logits(
                inputs: nil, inputEmbedding: merged[0..., range], cache: cache,
                positionIDs: promptPositions.map { $0[0..., 0..., range] })
            asyncEval(cache)
        }
        if processed > 0 { eval(cache) }
        prefill.progress?(total, total)

        let result = logits(
            inputs: nil, inputEmbedding: merged[0..., processed...], cache: cache,
            positionIDs: promptPositions.map { $0[0..., 0..., processed...] })
        return .logits(LMOutput(logits: result))
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

    /// The checkpoint keeps the text stack at `model.` and the tower at `vit.`, and stores
    /// both convolution layouts the torch way.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (rawKey, value) in weights {
            var key = rawKey
            if key.hasPrefix("model.") {
                key = "language_model." + key
            } else if key.hasPrefix("vit.") {
                key = "vision_tower." + String(key.dropFirst("vit.".count))
            } else if key.hasPrefix("lm_head.") {
                key = "language_model." + key
            }

            if key.contains("patch_embedding.weight") || key.contains("proj.0.weight")
                || key.contains("proj.2.weight"),
                !Self.isMLXConvolution(value)
            {
                sanitized[key] = value.transposed(0, 2, 3, 1)
            } else {
                sanitized[key] = value
            }
        }
        return sanitized
    }

    private static func isMLXConvolution(_ array: MLXArray) -> Bool {
        guard array.ndim == 4 else { return false }
        let (out, kH, kW) = (array.dim(0), array.dim(1), array.dim(2))
        return out >= kH && out >= kW && kH == kW
    }
}

extension HunyuanVL: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
