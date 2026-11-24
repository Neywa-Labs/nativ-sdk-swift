import Foundation
import MLX
import MLXNN

// MARK: - Configuration

public struct MolmoPointConfiguration: Decodable, Sendable {
    public struct VitConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let hiddenLayers: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let patchSize: Int
        public let numPositions: Int
        public let inputSize: [Int]
        public let layerNormEps: Float
        public let float32Attention: Bool

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case hiddenLayers = "num_hidden_layers"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case patchSize = "image_patch_size"
            case numPositions = "image_num_pos"
            case inputSize = "image_default_input_size"
            case layerNormEps = "layer_norm_eps"
            case float32Attention = "float32_attention"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "molmo2"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 4304
            hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 27
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 16
            headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 72
            patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? 14
            numPositions = try c.decodeIfPresent(Int.self, forKey: .numPositions) ?? 729
            inputSize = try c.decodeIfPresent([Int].self, forKey: .inputSize) ?? [378, 378]
            layerNormEps =
                try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? 1e-6
            float32Attention =
                try c.decodeIfPresent(Bool.self, forKey: .float32Attention) ?? true
        }

        var molmo2Configuration: Molmo2Configuration.VitConfiguration {
            Molmo2Configuration.VitConfiguration(
                modelType: modelType,
                hiddenSize: hiddenSize,
                intermediateSize: intermediateSize,
                hiddenLayers: hiddenLayers,
                attentionHeads: attentionHeads,
                kvHeads: kvHeads,
                headDim: headDim,
                patchSize: patchSize,
                numPositions: numPositions,
                inputSize: inputSize,
                layerNormEps: layerNormEps,
                float32Attention: float32Attention,
            )
        }
    }

    public struct AdapterConfiguration: Decodable, Sendable {
        public let modelType: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let textHiddenSize: Int
        public let attentionHeads: Int
        public let kvHeads: Int
        public let headDim: Int
        public let vitLayers: [Int]
        public let poolingAttentionMask: Bool
        public let float32Attention: Bool

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case textHiddenSize = "text_hidden_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case headDim = "head_dim"
            case vitLayers = "vit_layers"
            case poolingAttentionMask = "pooling_attention_mask"
            case float32Attention = "float32_attention"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "molmo_point"
            hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1152
            intermediateSize =
                try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 12288
            textHiddenSize =
                try c.decodeIfPresent(Int.self, forKey: .textHiddenSize) ?? 4096
            attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
            kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 16
            headDim = try c.decodeIfPresent(Int.self, forKey: .headDim) ?? 72
            vitLayers = try c.decodeIfPresent([Int].self, forKey: .vitLayers) ?? [-3, -9]
            poolingAttentionMask =
                try c.decodeIfPresent(Bool.self, forKey: .poolingAttentionMask) ?? true
            float32Attention =
                try c.decodeIfPresent(Bool.self, forKey: .float32Attention) ?? true
        }
    }

    public let modelType: String
    public let textConfiguration: Molmo2Configuration.TextConfiguration
    public let vitConfiguration: VitConfiguration
    public let adapterConfiguration: AdapterConfiguration
    public let imageStartTokenID: Int
    public let imageEndTokenID: Int
    public let imagePatchTokenID: Int
    public let imageNonIndexablePatchTokenID: Int
    public let imageColumnTokenID: Int
    public let patchTokenID: Int
    public let subpatchTokenID: Int
    public let locationTokenID: Int
    public let patchEmbedDim: Int
    public let noMorePointsClass: Bool
    public let layerNormX: Bool

    public var totalVocabularySize: Int {
        textConfiguration.vocabularySize + textConfiguration.additionalVocabularySize
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case vitConfiguration = "vit_config"
        case adapterConfiguration = "adapter_config"
        case imageStartTokenID = "image_start_token_id"
        case imageEndTokenID = "image_end_token_id"
        case imagePatchTokenID = "image_patch_id"
        case imageNonIndexablePatchTokenID = "image_non_indexable_patch_id"
        case imageLowResTokenID = "image_low_res_id"
        case imageColumnTokenID = "image_col_id"
        case patchTokenID = "patch_token_id"
        case subpatchTokenID = "subpatch_token_id"
        case locationTokenID = "location_token_id"
        case patchEmbedDim = "patch_embed_dim"
        case noMorePointsClass = "no_more_points_class"
        case layerNormX = "layer_norm_x"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "molmo_point"
        textConfiguration =
            try c.decodeIfPresent(
                Molmo2Configuration.TextConfiguration.self,
                forKey: .textConfiguration,
            )
            ?? JSONDecoder.molmoPointDefault(Molmo2Configuration.TextConfiguration.self)
        vitConfiguration =
            try c.decodeIfPresent(VitConfiguration.self, forKey: .vitConfiguration)
            ?? JSONDecoder.molmoPointDefault(VitConfiguration.self)
        adapterConfiguration =
            try c.decodeIfPresent(AdapterConfiguration.self, forKey: .adapterConfiguration)
            ?? JSONDecoder.molmoPointDefault(AdapterConfiguration.self)
        imageStartTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imageStartTokenID) ?? 151_936
        imageEndTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imageEndTokenID) ?? 151_937
        imagePatchTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imagePatchTokenID) ?? 151_938
        imageNonIndexablePatchTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imageNonIndexablePatchTokenID)
            ?? c.decodeIfPresent(Int.self, forKey: .imageLowResTokenID) ?? 151_942
        imageColumnTokenID =
            try c.decodeIfPresent(Int.self, forKey: .imageColumnTokenID) ?? 151_939
        patchTokenID =
            try c.decodeIfPresent(Int.self, forKey: .patchTokenID) ?? 151_947
        subpatchTokenID =
            try c.decodeIfPresent(Int.self, forKey: .subpatchTokenID) ?? 151_948
        locationTokenID =
            try c.decodeIfPresent(Int.self, forKey: .locationTokenID) ?? 151_949
        patchEmbedDim =
            try c.decodeIfPresent(Int.self, forKey: .patchEmbedDim) ?? 512
        noMorePointsClass =
            try c.decodeIfPresent(Bool.self, forKey: .noMorePointsClass) ?? true
        layerNormX =
            try c.decodeIfPresent(Bool.self, forKey: .layerNormX) ?? true
    }
}

extension JSONDecoder {
    fileprivate static func molmoPointDefault<T: Decodable>(_: T.Type) -> T {
        try! JSONDecoder.json5().decode(T.self, from: "{}".data(using: .utf8)!)
    }
}

// MARK: - Language

private enum MolmoPointLanguage {
    final class ExtendedHead: Module, UnaryLayer {
        @ModuleInfo(key: "output_embeddings") var outputEmbeddings: MLXArray
        @ModuleInfo(key: "new_output_embeddings") var newOutputEmbeddings: MLXArray

        init(vocabularySize: Int, additionalVocabularySize: Int, hiddenSize: Int) {
            _outputEmbeddings.wrappedValue = MLXArray.zeros([vocabularySize, hiddenSize])
            _newOutputEmbeddings.wrappedValue =
                MLXArray.zeros([additionalVocabularySize, hiddenSize])
        }

        func callAsFunction(_ hiddenStates: MLXArray) -> MLXArray {
            let base = matmul(hiddenStates, outputEmbeddings.T)
            let extra = matmul(hiddenStates, newOutputEmbeddings.T)
            return concatenated([base, extra], axis: -1)
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo var model: Molmo2Language.Transformer
        @ModuleInfo(key: "lm_head") var lmHead: ExtendedHead
        let kvHeads: [Int]

        init(_ config: Molmo2Configuration.TextConfiguration) {
            _model.wrappedValue = Molmo2Language.Transformer(config)
            _lmHead.wrappedValue = ExtendedHead(
                vocabularySize: config.vocabularySize,
                additionalVocabularySize: config.additionalVocabularySize,
                hiddenSize: config.hiddenSize,
            )
            kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        }

        func callAsFunction(
            _ inputs: MLXArray?,
            cache: [KVCache]? = nil,
            inputEmbedding: MLXArray? = nil,
        )
            -> LMOutput
        {
            let hidden = model(inputs, cache: cache, inputEmbedding: inputEmbedding)
            return LMOutput(logits: lmHead(hidden))
        }
    }
}

// MARK: - Vision

private enum MolmoPointVision {
    final class MLP: Module, UnaryLayer {
        @ModuleInfo var w1: Linear
        @ModuleInfo var w2: Linear
        let activation = GELU(approximation: .precise)

        init(hidden: Int, intermediate: Int) {
            _w1.wrappedValue = Linear(hidden, intermediate, bias: true)
            _w2.wrappedValue = Linear(intermediate, hidden, bias: true)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            w2(activation(w1(x)))
        }
    }

    final class Attention: Module {
        @ModuleInfo var wq: Linear
        @ModuleInfo var wk: Linear
        @ModuleInfo var wv: Linear
        @ModuleInfo var wo: Linear

        let hidden: Int
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float

        init(hidden: Int, heads: Int, kvHeads: Int, headDim: Int) {
            self.hidden = hidden
            self.heads = heads
            self.kvHeads = kvHeads
            self.headDim = headDim
            scale = pow(Float(headDim), -0.5)
            _wq.wrappedValue = Linear(hidden, heads * headDim, bias: true)
            _wk.wrappedValue = Linear(hidden, kvHeads * headDim, bias: true)
            _wv.wrappedValue = Linear(hidden, kvHeads * headDim, bias: true)
            _wo.wrappedValue = Linear(heads * headDim, hidden, bias: true)
        }

        func callAsFunction(_ input: MLXArray) -> MLXArray {
            let batch = input.dim(0)
            let length = input.dim(1)
            var q = wq(input).reshaped(batch, length, heads, headDim)
            var k = wk(input).reshaped(batch, length, kvHeads, headDim)
            var v = wv(input).reshaped(batch, length, kvHeads, headDim)
            if heads != kvHeads {
                let groups = heads / kvHeads
                k = repeated(k, count: groups, axis: 2)
                v = repeated(v, count: groups, axis: 2)
            }
            q = q.transposed(0, 2, 1, 3)
            k = k.transposed(0, 2, 1, 3)
            v = v.transposed(0, 2, 1, 3)
            let scores = matmul(q * scale, k.transposed(0, 1, 3, 2))
            let weights = softmax(scores.asType(.float32), axis: -1).asType(q.dtype)
            return wo(
                matmul(weights, v)
                    .transposed(0, 2, 1, 3)
                    .reshaped(batch, length, -1),
            )
        }
    }

    final class Block: Module {
        @ModuleInfo var attention: Attention
        @ModuleInfo(key: "feed_forward") var feedForward: MLP
        @ModuleInfo(key: "attention_norm") var attentionNorm: LayerNorm
        @ModuleInfo(key: "ffn_norm") var ffnNorm: LayerNorm

        init(_ config: MolmoPointConfiguration.VitConfiguration) {
            _attention.wrappedValue = Attention(
                hidden: config.hiddenSize,
                heads: config.attentionHeads,
                kvHeads: config.kvHeads,
                headDim: config.headDim,
            )
            _feedForward.wrappedValue = MLP(
                hidden: config.hiddenSize,
                intermediate: config.intermediateSize,
            )
            _attentionNorm.wrappedValue =
                LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
            _ffnNorm.wrappedValue =
                LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var h = x + attention(attentionNorm(x))
            h = h + feedForward(ffnNorm(h))
            return h
        }
    }

    final class VisionEncoder: Module {
        @ModuleInfo(key: "positional_embedding") var positionalEmbedding: MLXArray
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
        let resblocks: [Block]

        init(_ config: MolmoPointConfiguration) {
            let vitConfig = config.vitConfiguration
            let lastLayerNeeded =
                (config.adapterConfiguration.vitLayers.map {
                    $0 >= 0 ? $0 : vitConfig.hiddenLayers + $0
                }.max() ?? (vitConfig.hiddenLayers - 1)) + 1
            _positionalEmbedding.wrappedValue =
                MLXArray.zeros([vitConfig.numPositions, vitConfig.hiddenSize])
            _patchEmbedding.wrappedValue = Linear(
                vitConfig.patchSize * vitConfig.patchSize * 3,
                vitConfig.hiddenSize,
                bias: true,
            )
            resblocks = (0 ..< lastLayerNeeded).map { _ in Block(vitConfig) }
        }

        func callAsFunction(_ images: MLXArray) -> [MLXArray] {
            var hidden = patchEmbedding(images)
            hidden = hidden + positionalEmbedding.expandedDimensions(axis: 0).asType(hidden.dtype)
            var states = [MLXArray]()
            for block in resblocks {
                hidden = block(hidden)
                states.append(hidden)
            }
            return states
        }
    }

    final class PoolingAttention: Module {
        @ModuleInfo var wq: Linear
        @ModuleInfo var wk: Linear
        @ModuleInfo var wv: Linear

        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let float32Attention: Bool

        init(
            hidden: Int,
            heads: Int,
            kvHeads: Int,
            headDim: Int,
            inputDim: Int,
            float32Attention: Bool,
        ) {
            self.heads = heads
            self.kvHeads = kvHeads
            self.headDim = headDim
            scale = pow(Float(headDim), -0.5)
            self.float32Attention = float32Attention
            _wq.wrappedValue = Linear(inputDim, heads * headDim, bias: true)
            _wk.wrappedValue = Linear(inputDim, kvHeads * headDim, bias: true)
            _wv.wrappedValue = Linear(inputDim, kvHeads * headDim, bias: true)
            precondition(hidden == heads * headDim)
        }

        func callAsFunction(_ qInput: MLXArray, kvInput: MLXArray, mask: MLXArray?) -> MLXArray {
            let batch = qInput.dim(0)
            let qLength = qInput.dim(1)
            let kvLength = kvInput.dim(1)
            var q = wq(qInput).reshaped(batch, qLength, heads, headDim)
            var k = wk(kvInput).reshaped(batch, kvLength, kvHeads, headDim)
            var v = wv(kvInput).reshaped(batch, kvLength, kvHeads, headDim)
            if heads != kvHeads {
                let groups = heads / kvHeads
                k = repeated(k, count: groups, axis: 2)
                v = repeated(v, count: groups, axis: 2)
            }
            q = q.transposed(0, 2, 1, 3)
            k = k.transposed(0, 2, 1, 3)
            v = v.transposed(0, 2, 1, 3)
            let dtype = q.dtype
            if float32Attention {
                q = q.asType(.float32)
                k = k.asType(.float32)
                v = v.asType(.float32)
            }
            var scores = matmul(q, k.transposed(0, 1, 3, 2)) * scale
            if let mask {
                scores = MLX.where(
                    mask,
                    scores,
                    MLXArray.full(
                        scores.shape,
                        values: MLXArray(Float32(-1e9)),
                        dtype: scores.dtype,
                    ),
                )
            }
            let weights = softmax(scores, axis: -1)
            return matmul(weights, v).asType(dtype)
                .transposed(0, 2, 1, 3)
                .reshaped(batch, qLength, -1)
        }
    }

    final class Connector: Module {
        @ModuleInfo(key: "image_pooling_2d") var imagePooling: PoolingAttention
        @ModuleInfo(key: "image_projector") var imageProjector: Molmo2Vision.Projector

        let adapterConfig: MolmoPointConfiguration.AdapterConfiguration

        init(_ config: MolmoPointConfiguration) {
            adapterConfig = config.adapterConfiguration
            let poolDim =
                config.vitConfiguration.hiddenSize * config.adapterConfiguration.vitLayers.count
            _imagePooling.wrappedValue = PoolingAttention(
                hidden: config.adapterConfiguration.hiddenSize,
                heads: config.adapterConfiguration.attentionHeads,
                kvHeads: config.adapterConfiguration.kvHeads,
                headDim: config.adapterConfiguration.headDim,
                inputDim: poolDim,
                float32Attention: config.adapterConfiguration.float32Attention,
            )
            _imageProjector.wrappedValue = Molmo2Vision.Projector(
                input: config.adapterConfiguration.hiddenSize,
                hidden: config.adapterConfiguration.intermediateSize,
                output: config.adapterConfiguration.textHiddenSize,
            )
        }

        func callAsFunction(_ features: MLXArray, tokenPooling: MLXArray) -> MLXArray {
            let dim = features.dim(-1)
            let pooling = tokenPooling.asArray(Int32.self).map(Int.init)
            let poolingWidth = tokenPooling.dim(1)
            var clipped = [Int]()
            var validValues = [Float]()
            var validTokenIndices = [Int]()
            for row in 0 ..< tokenPooling.dim(0) {
                var rowHasValid = false
                for col in 0 ..< poolingWidth {
                    let value = pooling[row * poolingWidth + col]
                    if value >= 0 {
                        clipped.append(value)
                        validValues.append(1)
                        rowHasValid = true
                    } else {
                        clipped.append(0)
                        validValues.append(0)
                    }
                }
                if rowHasValid {
                    validTokenIndices.append(row)
                }
            }

            let gathered = features[MLXArray(clipped, [clipped.count])]
                .reshaped(tokenPooling.dim(0), poolingWidth, dim)
            let valid = MLXArray(validValues, [tokenPooling.dim(0), poolingWidth])
            let toPool = gathered * valid.expandedDimensions(axis: -1).asType(gathered.dtype)
            let denom = maximum(sum(valid, axis: -1, keepDims: true), MLXArray(Float32(1)))
            let query =
                sum(toPool, axis: -2, keepDims: true)
                / denom.expandedDimensions(axis: -1).asType(toPool.dtype)
            let mask: MLXArray? =
                adapterConfig.poolingAttentionMask
                ? valid.reshaped(-1, 1, 1, poolingWidth).asType(.bool)
                : nil
            var pooled = imagePooling(query, kvInput: toPool, mask: mask)
                .reshaped(tokenPooling.dim(0), -1)
            pooled = imageProjector(pooled)
            return pooled[MLXArray(validTokenIndices, [validTokenIndices.count])]
        }
    }
}

// MARK: - Point Parameter Containers

private final class MolmoPointPadWithLearnedVector: Module {
    @ModuleInfo var vector: MLXArray

    init(dimensions: Int) {
        _vector.wrappedValue = MLXArray.zeros([dimensions])
    }
}

private final class MolmoPointPredictor: Module {
    @ModuleInfo(key: "x_norm") var xNorm: RMSNorm
    @ModuleInfo(key: "patch_q") var patchQ: Linear
    @ModuleInfo(key: "patch_k") var patchK: Linear
    @ModuleInfo(key: "subpatch_q") var subpatchQ: Linear
    @ModuleInfo(key: "subpatch_k") var subpatchK: Linear
    @ModuleInfo(key: "subpatch_loc_k") var subpatchLocationK: Linear
    @ModuleInfo(key: "add_no_point_class_embed") var addNoPointClassEmbed:
        MolmoPointPadWithLearnedVector

    init(_ config: MolmoPointConfiguration) {
        let llmDim = config.textConfiguration.hiddenSize
        let vitDim =
            config.vitConfiguration.hiddenSize * config.adapterConfiguration.vitLayers.count
        _xNorm.wrappedValue = RMSNorm(
            dimensions: llmDim,
            eps: config.textConfiguration.layerNormEps,
        )
        _patchQ.wrappedValue = Linear(llmDim, config.patchEmbedDim, bias: true)
        _patchK.wrappedValue = Linear(llmDim, config.patchEmbedDim, bias: true)
        _subpatchQ.wrappedValue = Linear(llmDim, config.patchEmbedDim, bias: true)
        _subpatchK.wrappedValue = Linear(vitDim, config.patchEmbedDim, bias: true)
        _subpatchLocationK.wrappedValue = Linear(llmDim, 9, bias: true)
        _addNoPointClassEmbed.wrappedValue =
            MolmoPointPadWithLearnedVector(dimensions: config.patchEmbedDim)
    }
}

// MARK: - Model

public final class MolmoPoint: Module, LanguageModel, KVCacheDimensionProvider {
    @ModuleInfo private var lm: MolmoPointLanguage.LanguageModel
    @ModuleInfo(key: "vision_model") private var visionModel: MolmoPointVision.VisionEncoder
    @ModuleInfo private var connector: MolmoPointVision.Connector
    @ModuleInfo(key: "build_vit_embedding") private var buildVitEmbedding: Linear
    @ModuleInfo(key: "point_predictor") private var pointPredictor: MolmoPointPredictor

    public let config: MolmoPointConfiguration
    public var kvHeads: [Int] {
        lm.kvHeads
    }

    public var loraLayers: [Module] {
        lm.model.blocks
    }

    public init(_ config: MolmoPointConfiguration) {
        self.config = config
        _lm.wrappedValue = MolmoPointLanguage.LanguageModel(config.textConfiguration)
        _visionModel.wrappedValue = MolmoPointVision.VisionEncoder(config)
        _connector.wrappedValue = MolmoPointVision.Connector(config)
        let vitDim =
            config.vitConfiguration.hiddenSize * config.adapterConfiguration.vitLayers.count
        _buildVitEmbedding.wrappedValue = Linear(
            vitDim,
            config.textConfiguration.hiddenSize,
            bias: true,
        )
        _pointPredictor.wrappedValue = MolmoPointPredictor(config)
    }

    private var vitLayerIndices: [Int] {
        config.adapterConfiguration.vitLayers.map {
            $0 >= 0 ? $0 : config.vitConfiguration.hiddenLayers + $0
        }
    }

    private func inputEmbeddings(
        inputIDs: MLXArray,
        pixelValues: MLXArray?,
        tokenPooling: MLXArray?,
    )
        -> MLXArray
    {
        let safeIDs = inputIDs * (inputIDs .!= -1).asType(inputIDs.dtype)
        var embeddings = lm.model.wte(safeIDs)
        guard let pixelValues, let tokenPooling else {
            return embeddings
        }

        let dtype = visionModel.patchEmbedding.weight.dtype
        let imageStates = visionModel(pixelValues.asType(dtype))
        let selected = vitLayerIndices.map { imageStates[$0] }
        let features = concatenated(selected, axis: -1)
            .reshaped(pixelValues.dim(0) * pixelValues.dim(1), -1)
        let imageFeatures = Self.clipImageFeatures(
            connector(features, tokenPooling: tokenPooling),
        ).asType(.float32)

        let ids = inputIDs.flattened().asArray(Int32.self).map(Int.init)
        let positions = ids.indices.filter {
            ids[$0] == config.imagePatchTokenID
                || ids[$0] == config.imageNonIndexablePatchTokenID
        }
        precondition(
            positions.count == imageFeatures.dim(0),
            "MolmoPoint image feature/token count mismatch",
        )
        let flat = embeddings.asType(.float32).reshaped(-1, embeddings.dim(-1))
        let positionArray = MLXArray(positions, [positions.count])
        flat[positionArray] = flat[positionArray] + imageFeatures
        embeddings = flat.reshaped(embeddings.shape)
        return embeddings
    }

    private static func clipImageFeatures(_ imageFeatures: MLXArray) -> MLXArray {
        let dtypeMax = Float(
            imageFeatures.dtype.finfo?.max ?? Double(Float.greatestFiniteMagnitude))
        let clipLimit = imageFeatures.dtype == .float16 ? min(Float(65000), dtypeMax) : dtypeMax
        let cleaned = MLX.where(
            isNaN(imageFeatures),
            zeros(like: imageFeatures),
            imageFeatures,
        )
        return clip(cleaned, min: -clipLimit, max: clipLimit)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?, prefill: ModelPrefillOptions
    ) throws
        -> PrepareResult
    {
        let embeddings = inputEmbeddings(
            inputIDs: input.text.tokens,
            pixelValues: input.image?.pixels,
            tokenPooling: input.video?.pixels,
        )
        defer { prefill.progress?(embeddings.dim(1), embeddings.dim(1)) }
        return .logits(lm(nil, cache: cache, inputEmbedding: embeddings))
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lm(inputs, cache: cache).logits
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { key, _ in
            !key.contains("rotary_emb.inv_freq")
        }
    }
}
