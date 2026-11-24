import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct Phi4SigLipConfiguration: Decodable, Sendable {

    /// Phi-3's text tower. The checkpoint repeats these keys at the top level, where they
    /// take precedence over anything nested under `text_config`.
    public struct TextConfiguration: Decodable, Sendable {
        public var modelType: String = "phi4-siglip"
        public var vocabularySize: Int = 100_352
        public var hiddenLayers: Int = 40
        public var intermediateSize: Int = 17920
        public var attentionHeads: Int = 40
        public var rmsNormEps: Float = 1e-5
        public var hiddenSize: Int = 5120
        public var kvHeads: Int = 10
        public var ropeTheta: Float = 500_000
        public var ropeTraditional: Bool = false
        public var partialRotaryFactor: Float = 1.0
        public var ropeScaling: [String: StringOrNumber]?

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case vocabularySize = "vocab_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case rmsNormEps = "rms_norm_eps"
            case hiddenSize = "hidden_size"
            case kvHeads = "num_key_value_heads"
            case ropeTheta = "rope_theta"
            case ropeTraditional = "rope_traditional"
            case partialRotaryFactor = "partial_rotary_factor"
            case ropeScaling = "rope_scaling"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
            vocabularySize =
                try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? vocabularySize
            hiddenLayers =
                try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? hiddenLayers
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
            attentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? attentionHeads
            rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? rmsNormEps
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? kvHeads
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? ropeTheta
            ropeTraditional =
                try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? ropeTraditional
            partialRotaryFactor =
                try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
                ?? partialRotaryFactor
            ropeScaling = try c.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeScaling)
        }

        /// Only `linear` scales the rotation. Upstream reads no other kind here, so a
        /// long-rope checkpoint would rotate unscaled rather than take a different path.
        public var ropeScale: Float {
            guard let ropeScaling,
                case .string("linear")? = ropeScaling["type"] ?? ropeScaling["rope_type"],
                let factor = ropeScaling["factor"]?.asFloat()
            else {
                return 1
            }
            return 1 / factor
        }

        public var headDimensions: Int { hiddenSize / attentionHeads }
    }

    public struct VisionConfiguration: Decodable, Sendable {
        public var modelType: String = "siglip2_vision_model"
        public var hiddenSize: Int = 1152
        public var intermediateSize: Int = 4304
        public var hiddenLayers: Int = 27
        public var attentionHeads: Int = 16
        public var channels: Int = 3
        public var imageSize: Int = 512
        public var patchSize: Int = 16
        public var patches: Int = 256
        public var layerNormEps: Float = 1e-6

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case channels = "num_channels"
            case imageSize = "image_size"
            case patchSize = "patch_size"
            case patches = "num_patches"
            case layerNormEps = "layer_norm_eps"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
            hiddenLayers =
                try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? hiddenLayers
            attentionHeads =
                try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? attentionHeads
            channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? channels
            imageSize = try c.decodeIfPresent(Int.self, forKey: .imageSize) ?? imageSize
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? patchSize
            patches = try c.decodeIfPresent(Int.self, forKey: .patches) ?? patches
            layerNormEps =
                try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? layerNormEps
        }
    }

    public var modelType: String
    public var textConfiguration: TextConfiguration
    public var visionConfiguration: VisionConfiguration
    public var multiModalHiddenSize: Int
    public var visionSelectLayer: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case multiModalHiddenSize = "mm_hidden_size"
        case visionSelectLayer = "mm_vision_select_layer"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "phi4-siglip"
        visionConfiguration =
            try c.decodeIfPresent(VisionConfiguration.self, forKey: .visionConfiguration)
            ?? VisionConfiguration(from: decoder)
        multiModalHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .multiModalHiddenSize) ?? 1152
        visionSelectLayer =
            try c.decodeIfPresent(Int.self, forKey: .visionSelectLayer) ?? -2

        // The top level repeats the text fields and wins over `text_config`, so the
        // nested object is decoded first and then overwritten from the root.
        var text =
            try c.decodeIfPresent(TextConfiguration.self, forKey: .textConfiguration)
            ?? TextConfiguration(from: decoder)
        let root = try TextConfiguration(from: decoder)
        let rootKeys = try decoder.container(keyedBy: TextConfiguration.CodingKeys.self)
        if rootKeys.contains(.hiddenSize) { text.hiddenSize = root.hiddenSize }
        if rootKeys.contains(.hiddenLayers) { text.hiddenLayers = root.hiddenLayers }
        if rootKeys.contains(.intermediateSize) {
            text.intermediateSize = root.intermediateSize
        }
        if rootKeys.contains(.attentionHeads) { text.attentionHeads = root.attentionHeads }
        if rootKeys.contains(.kvHeads) { text.kvHeads = root.kvHeads }
        if rootKeys.contains(.vocabularySize) { text.vocabularySize = root.vocabularySize }
        if rootKeys.contains(.rmsNormEps) { text.rmsNormEps = root.rmsNormEps }
        if rootKeys.contains(.ropeTheta) { text.ropeTheta = root.ropeTheta }
        if rootKeys.contains(.ropeTraditional) {
            text.ropeTraditional = root.ropeTraditional
        }
        if rootKeys.contains(.partialRotaryFactor) {
            text.partialRotaryFactor = root.partialRotaryFactor
        }
        if rootKeys.contains(.ropeScaling) { text.ropeScaling = root.ropeScaling }
        textConfiguration = text
    }
}

// MARK: - Language

enum Phi4SigLipLanguage {

    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDimensions: Int
        let scale: Float

        @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        let rope: RoPE
        let ropeDimensions: Int

        init(_ config: Phi4SigLipConfiguration.TextConfiguration) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDimensions = config.headDimensions
            scale = pow(Float(headDimensions), -0.5)
            ropeDimensions = Int(Float(headDimensions) * config.partialRotaryFactor)

            _qkvProj.wrappedValue = Linear(
                config.hiddenSize,
                (heads + 2 * kvHeads) * headDimensions,
                bias: false)
            _oProj.wrappedValue = Linear(
                heads * headDimensions, config.hiddenSize, bias: false)
            rope = RoPE(
                dimensions: ropeDimensions,
                traditional: config.ropeTraditional,
                base: config.ropeTheta,
                scale: config.ropeScale)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))
            let queryWidth = heads * headDimensions
            let keyWidth = kvHeads * headDimensions
            let parts = split(
                qkvProj(x), indices: [queryWidth, queryWidth + keyWidth], axis: -1)

            var queries = parts[0].reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
            var keys = parts[1].reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
            let values = parts[2].reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

            let offset = cache?.offset ?? 0
            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)

            let output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values, cache: cache,
                scale: scale, mask: mask)
            return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_up_proj") var gateUpProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gateUpProj.wrappedValue = Linear(dimensions, 2 * hiddenDimensions, bias: false)
            _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let halves = split(gateUpProj(x), parts: 2, axis: -1)
            return downProj(compiledSiluProduct(halves[0], halves[1]))
        }
    }

    final class TransformerBlock: Module {
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        init(_ config: Phi4SigLipConfiguration.TextConfiguration) {
            _selfAttention.wrappedValue = Attention(config)
            _mlp.wrappedValue = MLP(
                dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
        ) -> MLXArray {
            let h = x + selfAttention(inputLayerNorm(x), mask: mask, cache: cache)
            return h + mlp(postAttentionLayerNorm(h))
        }
    }

    final class Backbone: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo var layers: [TransformerBlock]
        @ModuleInfo var norm: RMSNorm

        init(_ config: Phi4SigLipConfiguration.TextConfiguration) {
            precondition(config.vocabularySize > 0)
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
                TransformerBlock(config)
            }
            _norm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps)
            super.init()
        }

        func callAsFunction(
            inputs: MLXArray?, inputEmbedding: MLXArray?, cache: [KVCache]?
        ) -> MLXArray {
            guard var h = inputEmbedding ?? inputs.map({ embedTokens($0) }) else {
                fatalError("either inputs or inputEmbedding must be provided")
            }
            let mask = createAttentionMask(h: h, cache: cache?.first)
            for (i, layer) in layers.enumerated() {
                h = layer(h, mask: mask, cache: cache?[i])
            }
            return norm(h)
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo var model: Backbone
        @ModuleInfo(key: "lm_head") var lmHead: Linear

        init(_ config: Phi4SigLipConfiguration.TextConfiguration) {
            _model.wrappedValue = Backbone(config)
            _lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
            super.init()
        }

        func callAsFunction(
            inputs: MLXArray?, inputEmbedding: MLXArray? = nil, cache: [KVCache]?
        ) -> MLXArray {
            lmHead(model(inputs: inputs, inputEmbedding: inputEmbedding, cache: cache))
        }
    }
}

// MARK: - Vision

enum Phi4SigLipVision {

    final class Attention: Module {
        let heads: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: Phi4SigLipConfiguration.VisionConfiguration) {
            precondition(config.hiddenSize % config.attentionHeads == 0)
            heads = config.attentionHeads
            scale = pow(Float(config.hiddenSize / config.attentionHeads), -0.5)
            _qProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _kProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _vProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))
            let queries = qProj(x).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
            let keys = kProj(x).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
            let values = vProj(x).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale,
                mask: mask.map { .array($0) } ?? .none)
            return outProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "fc1") var fc1: Linear
        @ModuleInfo(key: "fc2") var fc2: Linear

        let activation = GELU(approximation: .precise)

        init(_ config: Phi4SigLipConfiguration.VisionConfiguration) {
            _fc1.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize, bias: true)
            _fc2.wrappedValue = Linear(
                config.intermediateSize, config.hiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x)))
        }
    }

    final class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var selfAttention: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

        init(_ config: Phi4SigLipConfiguration.VisionConfiguration) {
            _selfAttention.wrappedValue = Attention(config)
            _layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            _mlp.wrappedValue = MLP(config)
            _layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            super.init()
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> MLXArray {
            let h = x + selfAttention(layerNorm1(x), mask: mask)
            return h + mlp(layerNorm2(h))
        }
    }

    final class Encoder: Module {
        @ModuleInfo var layers: [EncoderLayer]

        init(_ config: Phi4SigLipConfiguration.VisionConfiguration) {
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
                EncoderLayer(config)
            }
            super.init()
        }

        /// Seeded with the patch embedding, so index `i` is layer `i - 1`'s output and a
        /// select layer of -2 names the second-to-last layer.
        func callAsFunction(_ x: MLXArray, mask: MLXArray?) -> [MLXArray] {
            var states = [x]
            var h = x
            for layer in layers {
                h = layer(h, mask: mask)
                states.append(h)
            }
            return states
        }
    }

    /// A patch embedding over already-flattened patches, with the positional table
    /// resampled to each image's own patch grid.
    final class VisionEmbeddings: Module {
        let side: Int
        let positions: Int

        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
        @ModuleInfo(key: "position_embedding") var positionEmbedding: Embedding

        init(_ config: Phi4SigLipConfiguration.VisionConfiguration) {
            positions = config.patches
            side = Int(Double(config.patches).squareRoot().rounded())
            _patchEmbedding.wrappedValue = Linear(
                config.channels * config.patchSize * config.patchSize,
                config.hiddenSize, bias: true)
            _positionEmbedding.wrappedValue = Embedding(
                embeddingCount: config.patches, dimensions: config.hiddenSize)
            super.init()
        }

        func callAsFunction(_ pixelValues: MLXArray, spatialShapes: MLXArray?) -> MLXArray {
            let patches = patchEmbedding(pixelValues.asType(patchEmbedding.weight.dtype))
            guard let spatialShapes else {
                return patches
                    + positionEmbedding(MLXArray(0 ..< positions).expandedDimensions(axis: 0))
            }
            return patches
                + resized(spatialShapes: spatialShapes, maxLength: pixelValues.dim(1))
                .asType(patches.dtype)
        }

        private func resized(spatialShapes: MLXArray, maxLength: Int) -> MLXArray {
            let dimensions = positionEmbedding.weight.dim(-1)
            let shapes = spatialShapes.asArray(Int32.self).map(Int.init)
            let source = positionEmbedding.weight
                .reshaped(side, side, dimensions)
                .transposed(2, 0, 1)
                .expandedDimensions(axis: 0)

            var rows: [MLXArray] = []
            rows.reserveCapacity(spatialShapes.dim(0))
            for index in 0 ..< spatialShapes.dim(0) {
                let height = Swift.max(shapes[index * 2], 1)
                let width = Swift.max(shapes[index * 2 + 1], 1)
                var row = bicubicInterpolate(
                    source, size: (height, width), alignCorners: false, antialias: false)[0]
                    .transposed(1, 2, 0)
                    .reshaped(height * width, dimensions)
                if row.dim(0) < maxLength {
                    row = concatenated(
                        [
                            row,
                            broadcast(
                                row[0 ..< 1, 0...],
                                to: [maxLength - row.dim(0), dimensions]),
                        ], axis: 0)
                }
                rows.append(row[..<maxLength, 0...])
            }
            return stacked(rows, axis: 0)
        }
    }

    final class SigLip2VisionModel: Module {
        @ModuleInfo var embeddings: VisionEmbeddings
        @ModuleInfo var encoder: Encoder
        @ModuleInfo(key: "post_layernorm") var postLayerNorm: LayerNorm

        init(_ config: Phi4SigLipConfiguration.VisionConfiguration) {
            _embeddings.wrappedValue = VisionEmbeddings(config)
            _encoder.wrappedValue = Encoder(config)
            _postLayerNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize)
            super.init()
        }

        func callAsFunction(_ x: MLXArray, spatialShapes: MLXArray?) -> (
            [MLXArray], MLXArray, MLXArray
        ) {
            let embedded = embeddings(x, spatialShapes: spatialShapes)
                .asType(embeddings.patchEmbedding.weight.dtype)
            let states = encoder(embedded, mask: nil)
            return (states, embedded, postLayerNorm(states[states.count - 1]))
        }
    }

    final class VisionModel: Module {
        @ModuleInfo(key: "vision_model") var visionModel: SigLip2VisionModel

        init(_ config: Phi4SigLipConfiguration.VisionConfiguration) {
            precondition(
                ["phi4-siglip", "siglip2_vision_model"].contains(config.modelType),
                "Unsupported vision modelType: \(config.modelType)")
            _visionModel.wrappedValue = SigLip2VisionModel(config)
            super.init()
        }

        func callAsFunction(_ x: MLXArray, spatialShapes: MLXArray?) -> (
            [MLXArray], MLXArray, MLXArray
        ) {
            visionModel(x, spatialShapes: spatialShapes)
        }
    }

    /// `mlp2x_gelu`: two biased linears around an exact GELU.
    final class MultiModalProjector: Module, UnaryLayer {
        @ModuleInfo(key: "linear_1") var linear1: Linear
        @ModuleInfo(key: "linear_2") var linear2: Linear

        let activation = GELU()

        init(_ config: Phi4SigLipConfiguration) {
            let hiddenSize = config.textConfiguration.hiddenSize
            _linear1.wrappedValue = Linear(
                config.multiModalHiddenSize, hiddenSize, bias: true)
            _linear2.wrappedValue = Linear(hiddenSize, hiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear2(activation(linear1(x)))
        }
    }
}

// MARK: - Model

public final class Phi4SigLip: Module, VLMModel, KVCacheDimensionProvider {

    /// The prompt marks an image with a negative id no vocabulary can hold, so it never
    /// reaches the embedding table.
    public static let imageTokenIndex = -200

    @ModuleInfo(key: "language_model") private var languageModel: Phi4SigLipLanguage.LanguageModel
    @ModuleInfo(key: "vision_tower") var visionTower: VisionTower
    @ModuleInfo(key: "mm_projector") private var projector: Phi4SigLipVision.MultiModalProjector

    /// The checkpoint nests the tower one level deeper than the module needs.
    final class VisionTower: Module {
        @ModuleInfo(key: "vision_tower") var visionTower: Phi4SigLipVision.VisionModel

        init(_ config: Phi4SigLipConfiguration.VisionConfiguration) {
            _visionTower.wrappedValue = Phi4SigLipVision.VisionModel(config)
            super.init()
        }

        func callAsFunction(_ x: MLXArray, spatialShapes: MLXArray?) -> (
            [MLXArray], MLXArray, MLXArray
        ) {
            visionTower(x, spatialShapes: spatialShapes)
        }
    }

    public let config: Phi4SigLipConfiguration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] {
        Array(
            repeating: config.textConfiguration.kvHeads,
            count: config.textConfiguration.hiddenLayers)
    }

    public init(_ config: Phi4SigLipConfiguration) {
        self.config = config
        _languageModel.wrappedValue = Phi4SigLipLanguage.LanguageModel(
            config.textConfiguration)
        _visionTower.wrappedValue = VisionTower(config.visionConfiguration)
        _projector.wrappedValue = Phi4SigLipVision.MultiModalProjector(config)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs: inputs, cache: cache)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }

        let embeddings = try input.image.map {
            try inputEmbeddings(inputIds: tokens, image: $0)
        }
        let total = embeddings?.dim(1) ?? tokens.dim(1)

        func forward(_ range: Range<Int>) -> MLXArray {
            if let embeddings {
                return languageModel(
                    inputs: nil, inputEmbedding: embeddings[0..., range], cache: cache)
            }
            return languageModel(inputs: tokens[0..., range], cache: cache)
        }

        let processed = try prefill.forEachChunk(total: total) { range in
            _ = forward(range)
            asyncEval(cache)
        }
        if processed > 0 {
            eval(cache)
        }

        prefill.progress?(total, total)
        return .logits(LMOutput(logits: forward(processed ..< total)))
    }

    /// Splices the projected patch features in at the image positions.
    ///
    /// The image id is negative, so the positions are replaced rather than scattered
    /// into: each one contributes no embedding of its own and every image expands to as
    /// many positions as it has valid patches.
    private func inputEmbeddings(inputIds: MLXArray, image: LMInput.ProcessedImage) throws
        -> MLXArray
    {
        let ids = inputIds.asArray(Int32.self)
        let dtype = languageModel.model.embedTokens.weight.dtype
        let (states, _, _) = visionTower(
            image.pixels.asType(dtype), spatialShapes: Self.spatialShapes(of: image))
        let selected = states[Self.index(config.visionSelectLayer, count: states.count)]

        let lengths = Self.validPatchCounts(of: image, fallback: selected.dim(1))
        var features: [MLXArray] = []
        features.reserveCapacity(selected.dim(0))
        for row in 0 ..< selected.dim(0) {
            features.append(projector(selected[row, ..<lengths[row], 0...]))
        }

        var segments: [MLXArray] = []
        var start = 0
        var imageIndex = 0
        for (position, id) in ids.enumerated() where Int(id) == Self.imageTokenIndex {
            if position > start {
                segments.append(
                    languageModel.model.embedTokens(
                        inputIds.reshaped(-1)[start ..< position]))
            }
            guard imageIndex < features.count else {
                throw VLMError.imageRequired
            }
            segments.append(features[imageIndex].asType(dtype))
            imageIndex += 1
            start = position + 1
        }
        if start < ids.count {
            segments.append(
                languageModel.model.embedTokens(inputIds.reshaped(-1)[start...]))
        }

        guard !segments.isEmpty else {
            return languageModel.model.embedTokens(inputIds)
        }
        return concatenated(segments, axis: 0).expandedDimensions(axis: 0)
    }

    static func index(_ selectLayer: Int, count: Int) -> Int {
        let resolved = selectLayer < 0 ? count + selectLayer : selectLayer
        return Swift.min(Swift.max(resolved, 0), count - 1)
    }

    private static func spatialShapes(of image: LMInput.ProcessedImage) -> MLXArray? {
        guard let frames = image.frames, !frames.isEmpty else { return nil }
        return MLXArray(
            frames.flatMap { [Int32($0.h), Int32($0.w)] }, [frames.count, 2])
    }

    private static func validPatchCounts(of image: LMInput.ProcessedImage, fallback: Int)
        -> [Int]
    {
        guard let frames = image.frames, !frames.isEmpty else {
            return Array(repeating: fallback, count: image.pixels.dim(0))
        }
        return frames.map { Swift.min($0.h * $0.w, fallback) }
    }

    /// Moves the checkpoint's flat layout onto the module tree: the language model gains
    /// its prefix, the projector's numbered members gain names, and the attention-pooling
    /// head the model never reads is dropped.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        for (rawKey, value) in weights {
            if rawKey.contains("position_ids") || rawKey.contains("vision_model.head.") {
                continue
            }

            var key =
                rawKey
                .replacingOccurrences(of: "mm_projector.0.", with: "mm_projector.linear_1.")
                .replacingOccurrences(of: "mm_projector.2.", with: "mm_projector.linear_2.")

            if key.hasPrefix("model.vision_tower.") || key.hasPrefix("model.mm_projector.") {
                key = String(key.dropFirst("model.".count))
            } else if key.hasPrefix("model.") || key.hasPrefix("lm_head.") {
                key = "language_model." + key
            }

            sanitized[key] = value
        }
        return sanitized
    }
}

extension Phi4SigLip: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
