import Foundation
import MLX
import MLXNN

// MARK: - Configuration

/// MiniMax M3's text stack.
///
/// The sparse-attention settings arrive either as a nested `sparse_attention_config` or as
/// flat `index_*` fields, and either the layer schedule or a length-derived default decides
/// which layers use them. Both spellings are normalized here so the model reads one shape.
public struct MiniMaxM3TextConfiguration: Decodable, Sendable {

    public var modelType: String = "minimax_m3"
    public var hiddenSize: Int = 6144
    public var intermediateSize: Int = 3072
    public var denseIntermediateSize: Int = 12288
    public var sharedIntermediateSize: Int = 3072
    public var attentionHeads: Int = 64
    public var hiddenLayers: Int = 60
    public var rmsNormEps: Float = 1e-6
    public var ropeTheta: Float = 5_000_000
    public var partialRotaryFactor: Float = 0.5
    public var ropeScaling: [String: StringOrNumber]?
    public var maxPositionEmbeddings: Int = 1_048_576
    public var vocabularySize: Int = 200_064
    public var tieWordEmbeddings: Bool = false
    public var swiGLUAlpha: Float = 1.702
    public var swiGLUBeta: Float = 1.0
    public var swiGLULimit: Float = 7.0
    public var useQKNorm: Bool = true
    public var qkNormType: String = "per_head"
    public var useGemmaNorm: Bool = true
    public var localExperts: Int = 128
    public var expertsPerToken: Int = 4
    public var sharedExperts: Int = 1
    public var useRoutingBias: Bool = true
    public var routedScalingFactor: Float = 2.0
    public var attentionOutputGate: Bool = false

    private var declaredKVHeads: Int?
    private var declaredHeadDim: Int?
    private var declaredRotaryDim: Int?
    private var declaredPackSharedExpert: Bool?
    private var declaredMoELayerFrequency: [Int]?
    private var mlpLayerTypes: [String]?
    private var layerTypes: [String]?
    private var declaredIndexHeads: Int?
    private var declaredIndexHeadDim: Int?
    private var declaredIndexBlockSize: Int?
    private var declaredIndexTopKBlocks: Int?
    private var declaredIndexLocalBlocks: Int?
    private var declaredSparse: [String: StringOrNumber]?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case denseIntermediateSize = "dense_intermediate_size"
        case sharedIntermediateSize = "shared_intermediate_size"
        case attentionHeads = "num_attention_heads"
        case declaredKVHeads = "num_key_value_heads"
        case declaredHeadDim = "head_dim"
        case hiddenLayers = "num_hidden_layers"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case declaredRotaryDim = "rotary_dim"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case vocabularySize = "vocab_size"
        case tieWordEmbeddings = "tie_word_embeddings"
        case hiddenAct = "hidden_act"
        case swiGLUAlpha = "swiglu_alpha"
        case swiGLUBeta = "swiglu_beta"
        case swiGLULimit = "swiglu_limit"
        case useQKNorm = "use_qk_norm"
        case qkNormType = "qk_norm_type"
        case useGemmaNorm = "use_gemma_norm"
        case localExperts = "num_local_experts"
        case expertsPerToken = "num_experts_per_tok"
        case sharedExperts = "n_shared_experts"
        case scoringFunction = "scoring_func"
        case useRoutingBias = "use_routing_bias"
        case routedScalingFactor = "routed_scaling_factor"
        case declaredMoELayerFrequency = "moe_layer_freq"
        case mlpLayerTypes = "mlp_layer_types"
        case declaredSparse = "sparse_attention_config"
        case layerTypes = "layer_types"
        case declaredIndexHeads = "index_n_heads"
        case declaredIndexHeadDim = "index_head_dim"
        case declaredIndexBlockSize = "index_block_size"
        case declaredIndexTopKBlocks = "index_topk_blocks"
        case declaredIndexLocalBlocks = "index_local_blocks"
        case attentionOutputGate = "attention_output_gate"
        case declaredPackSharedExpert = "pack_shared_expert"
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)

        if let activation = try c.decodeIfPresent(String.self, forKey: .hiddenAct),
            activation != "swigluoai"
        {
            throw DecodingError.dataCorruptedError(
                forKey: .hiddenAct, in: c,
                debugDescription: "MiniMax M3 implements the swigluoai activation only")
        }
        if let scoring = try c.decodeIfPresent(String.self, forKey: .scoringFunction),
            scoring != "sigmoid"
        {
            throw DecodingError.dataCorruptedError(
                forKey: .scoringFunction, in: c,
                debugDescription: "MiniMax M3 implements sigmoid routing only")
        }

        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
        intermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
        denseIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .denseIntermediateSize)
            ?? denseIntermediateSize
        sharedIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .sharedIntermediateSize)
            ?? sharedIntermediateSize
        attentionHeads =
            try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? attentionHeads
        declaredKVHeads = try c.decodeIfPresent(Int.self, forKey: .declaredKVHeads)
        declaredHeadDim = try c.decodeIfPresent(Int.self, forKey: .declaredHeadDim)
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? hiddenLayers
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? rmsNormEps
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? ropeTheta
        declaredRotaryDim = try c.decodeIfPresent(Int.self, forKey: .declaredRotaryDim)
        partialRotaryFactor =
            try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor)
            ?? partialRotaryFactor
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings)
            ?? maxPositionEmbeddings
        vocabularySize =
            try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? vocabularySize
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? tieWordEmbeddings
        swiGLUAlpha = try c.decodeIfPresent(Float.self, forKey: .swiGLUAlpha) ?? swiGLUAlpha
        swiGLUBeta = try c.decodeIfPresent(Float.self, forKey: .swiGLUBeta) ?? swiGLUBeta
        swiGLULimit = try c.decodeIfPresent(Float.self, forKey: .swiGLULimit) ?? swiGLULimit
        useQKNorm = try c.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? useQKNorm
        qkNormType = try c.decodeIfPresent(String.self, forKey: .qkNormType) ?? qkNormType
        useGemmaNorm =
            try c.decodeIfPresent(Bool.self, forKey: .useGemmaNorm) ?? useGemmaNorm
        localExperts = try c.decodeIfPresent(Int.self, forKey: .localExperts) ?? localExperts
        expertsPerToken =
            try c.decodeIfPresent(Int.self, forKey: .expertsPerToken) ?? expertsPerToken
        sharedExperts =
            try c.decodeIfPresent(Int.self, forKey: .sharedExperts) ?? sharedExperts
        useRoutingBias =
            try c.decodeIfPresent(Bool.self, forKey: .useRoutingBias) ?? useRoutingBias
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor)
            ?? routedScalingFactor
        attentionOutputGate =
            try c.decodeIfPresent(Bool.self, forKey: .attentionOutputGate)
            ?? attentionOutputGate
        declaredPackSharedExpert = try c.decodeIfPresent(
            Bool.self, forKey: .declaredPackSharedExpert)
        declaredMoELayerFrequency = try c.decodeIfPresent(
            [Int].self, forKey: .declaredMoELayerFrequency)
        mlpLayerTypes = try c.decodeIfPresent([String].self, forKey: .mlpLayerTypes)
        layerTypes = try c.decodeIfPresent([String].self, forKey: .layerTypes)
        declaredIndexHeads = try c.decodeIfPresent(Int.self, forKey: .declaredIndexHeads)
        declaredIndexHeadDim = try c.decodeIfPresent(Int.self, forKey: .declaredIndexHeadDim)
        declaredIndexBlockSize = try c.decodeIfPresent(
            Int.self, forKey: .declaredIndexBlockSize)
        declaredIndexTopKBlocks = try c.decodeIfPresent(
            Int.self, forKey: .declaredIndexTopKBlocks)
        declaredIndexLocalBlocks = try c.decodeIfPresent(
            Int.self, forKey: .declaredIndexLocalBlocks)
        declaredSparse = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .declaredSparse)
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
    }

    public var kvHeads: Int { declaredKVHeads ?? attentionHeads }
    public var headDim: Int { declaredHeadDim ?? hiddenSize / attentionHeads }
    public var rotaryDim: Int {
        declaredRotaryDim ?? Int(Float(headDim) * partialRotaryFactor)
    }

    /// The first three layers are dense and the rest sparse, unless a schedule says
    /// otherwise.
    private var defaultLayerFrequency: [Int] {
        let dense = Swift.min(3, hiddenLayers)
        return Array(repeating: 0, count: dense)
            + Array(repeating: 1, count: hiddenLayers - dense)
    }

    public var moeLayerFrequency: [Int] {
        if let declared = declaredMoELayerFrequency, !declared.isEmpty {
            return declared
        }
        if let types = mlpLayerTypes {
            return types.map { $0 == "sparse" ? 1 : 0 }
        }
        return defaultLayerFrequency
    }

    /// Which layers run the block-sparse indexer. `layer_types` names them; otherwise the
    /// same dense-prefix default applies.
    public var sparseAttentionFrequency: [Int] {
        if let types = layerTypes {
            return types.map { $0 == "minimax_m3_sparse" ? 1 : 0 }
        }
        if let declared = declaredSparse?["sparse_attention_freq"]?.asInts() {
            return declared
        }
        if let disable = declaredSparse?["sparse_disable_index_value"]?.asInts() {
            return disable
        }
        return defaultLayerFrequency
    }

    /// A flat `index_*` field fills the nested slot when the nested one is absent.
    private func sparseValue(_ key: String, alias: Int?, fallback: Int) -> Int {
        if let nested = declaredSparse?[key]?.asInt() {
            return nested
        }
        return alias ?? fallback
    }

    public var indexHeadDim: Int {
        sparseValue("sparse_index_dim", alias: declaredIndexHeadDim, fallback: 128)
    }
    public var indexHeads: Int {
        sparseValue("sparse_num_index_heads", alias: declaredIndexHeads, fallback: 4)
    }
    public var indexTopKBlocks: Int {
        sparseValue("sparse_topk_blocks", alias: declaredIndexTopKBlocks, fallback: 16)
    }
    public var indexBlockSize: Int {
        sparseValue("sparse_block_size", alias: declaredIndexBlockSize, fallback: 128)
    }
    public var indexLocalBlocks: Int {
        sparseValue("sparse_local_block", alias: declaredIndexLocalBlocks, fallback: 1)
    }
    public var indexInitialBlocks: Int {
        declaredSparse?["sparse_init_block"]?.asInt() ?? 0
    }

    public func isMoELayer(_ layerIndex: Int) -> Bool {
        let frequency = moeLayerFrequency
        guard layerIndex < frequency.count else { return false }
        return frequency[layerIndex] != 0 && localExperts > 0
    }

    public func isSparseAttentionLayer(_ layerIndex: Int) -> Bool {
        let frequency = sparseAttentionFrequency
        guard layerIndex < frequency.count else { return false }
        return frequency[layerIndex] != 0
    }

    /// The shared expert rides in the routed bank's packed projection only when the two
    /// widths agree, which is what lets a checkpoint quantize it separately.
    public var packsSharedExpert: Bool {
        declaredPackSharedExpert ?? (sharedIntermediateSize == intermediateSize)
    }
}

// MARK: - Primitives

enum MiniMaxM3Text {

    /// The clamped SwiGLU: the gate is bounded above only, the linear half on both sides,
    /// and the product carries a `+ beta` offset the ordinary form has no room for.
    static func swiGLUOAI(
        linear: MLXArray, gate: MLXArray, alpha: Float, limit: Float, beta: Float
    ) -> MLXArray {
        let boundedGate = clip(gate, max: limit)
        let boundedLinear = clip(linear, min: -limit, max: limit)
        return boundedGate * sigmoid(alpha * boundedGate) * (boundedLinear + beta)
    }

    /// RMS normalization whose weight is an offset from one under the Gemma convention and
    /// the scale itself otherwise -- and whose initial value differs to match.
    final class RMSNorm: Module, UnaryLayer {
        @ParameterInfo(key: "weight") var weight: MLXArray
        let eps: Float
        let gemma: Bool

        init(dimensions: Int, eps: Float, gemma: Bool) {
            self.eps = eps
            self.gemma = gemma
            _weight.wrappedValue =
                gemma ? MLXArray.zeros([dimensions]) : MLXArray.ones([dimensions])
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            MLXFast.rmsNorm(x, weight: gemma ? weight + 1 : weight, eps: eps)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear

        let alpha: Float
        let limit: Float
        let beta: Float

        init(
            hiddenSize: Int, intermediateSize: Int, alpha: Float, limit: Float, beta: Float,
            bias: Bool = false
        ) {
            _gateProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
            _upProj.wrappedValue = Linear(hiddenSize, intermediateSize, bias: bias)
            _downProj.wrappedValue = Linear(intermediateSize, hiddenSize, bias: bias)
            self.alpha = alpha
            self.limit = limit
            self.beta = beta
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(
                swiGLUOAI(
                    linear: upProj(x), gate: gateProj(x), alpha: alpha, limit: limit,
                    beta: beta))
        }
    }

    /// Sigmoid-scored routing with an optional bias that steers selection only.
    static func selectExperts(
        gates: MLXArray, bias: MLXArray?, topK: Int, routedScalingFactor: Float
    ) -> (indices: MLXArray, weights: MLXArray) {
        let scores = sigmoid(gates.asType(.float32))
        let ranked = bias.map { scores + $0.asType(scores.dtype) } ?? scores
        let indices = argPartition(-ranked, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
        var weights = takeAlong(scores, indices, axis: -1)
        weights = weights / (weights.sum(axis: -1, keepDims: true) + 1e-20)
        return (indices, (weights * routedScalingFactor).asType(gates.dtype))
    }
}

// MARK: - Sparse selection

extension MiniMaxM3Text {

    /// One cache per sparse-attention layer: the ordinary keys and values, plus the
    /// indexer's own keys, which are a different width and live on their own timeline.
    final class IndexKVCache: KVCacheSimple {
        private var indexKeys: MLXArray?
        private(set) var indexOffset = 0

        override func innerState() -> [MLXArray] {
            super.innerState() + [indexKeys].compactMap { $0 }
        }

        override func copy() -> any KVCache {
            let result = IndexKVCache()
            result.keys = keys.map { $0[.ellipsis] }
            result.values = values.map { $0[.ellipsis] }
            result.offset = offset
            result.step = step
            result.indexKeys = indexKeys.map { $0[.ellipsis] }
            result.indexOffset = indexOffset
            return result
        }

        @discardableResult
        override func trim(_ n: Int) -> Int {
            let removed = super.trim(n)
            indexOffset = max(0, indexOffset - removed)
            return removed
        }

        /// Appends the indexer's keys and returns everything cached so far.
        func updateIndexAndFetch(_ keys: MLXArray) -> MLXArray {
            let previous = indexOffset
            let incoming = keys.dim(2)
            if indexKeys == nil || previous + incoming > indexKeys!.dim(2) {
                let steps = (step + incoming - 1) / step
                let fresh = MLXArray.zeros(
                    [keys.dim(0), keys.dim(1), steps * step, keys.dim(3)],
                    dtype: keys.dtype)
                if var existing = indexKeys {
                    if previous % step != 0 {
                        existing = existing[.ellipsis, ..<previous, 0...]
                    }
                    indexKeys = concatenated([existing, fresh], axis: 2)
                } else {
                    indexKeys = fresh
                }
            }
            indexOffset += incoming
            indexKeys![.ellipsis, previous ..< indexOffset, 0...] = keys
            return indexKeys![.ellipsis, ..<indexOffset, 0...]
        }

    }

    /// Chooses which key blocks a query may read.
    ///
    /// Blocks are scored by the best index-head score inside them, the first
    /// `init_blocks` and the last `local_blocks` are forced in with sentinel scores, and
    /// the selection is returned sorted with invalid slots as -1.
    struct BlockSelector {
        let blockSize: Int
        let topKBlocks: Int
        let initialBlocks: Int
        let localBlocks: Int
        let scale: Float

        func select(
            queries: MLXArray, keys: MLXArray, queryPositions: MLXArray
        ) -> MLXArray {
            let (B, _, L, _) = (
                queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3)
            )
            let totalLength = keys.dim(2)
            let negative = MLXArray(-Float.infinity, dtype: .float32)

            var scores =
                queries.asType(.float32).matmul(keys.asType(.float32).swappedAxes(-1, -2))
                * scale

            let keyPositions = MLXArray(0 ..< totalLength)
            let causal =
                keyPositions[.newAxis, .newAxis, 0...]
                .<= queryPositions[0..., 0..., .newAxis]
            scores = MLX.where(causal.expandedDimensions(axis: 1), scores, negative)

            let blockCount = (totalLength + blockSize - 1) / blockSize
            let padding = blockCount * blockSize - totalLength
            if padding > 0 {
                scores = concatenated(
                    [
                        scores,
                        MLXArray.full(
                            Array(scores.shape.dropLast()) + [padding],
                            values: negative.asType(scores.dtype)),
                    ], axis: -1)
            }

            let blocks = MLXArray(0 ..< blockCount)
            let currentBlock = floorDivide(queryPositions, blockSize)
            let causalBlock =
                blocks[.newAxis, .newAxis, 0...] .<= currentBlock[0..., 0..., .newAxis]

            var blockScores =
                scores
                .reshaped(B, queries.dim(1), L, blockCount, blockSize)
                .max(axis: -1)
                .max(axis: 1)
            // A block whose every score is -inf comes back as NaN from the reduction.
            blockScores = MLX.where(isNaN(blockScores), negative, blockScores)

            let validBlocks = causalBlock
            var selected = MLX.where(validBlocks, blockScores, negative)

            if initialBlocks > 0 {
                let initial =
                    (blocks[.newAxis, .newAxis, 0...] .< MLXArray(Int32(initialBlocks)))
                    .&& validBlocks
                selected = MLX.where(initial, MLXArray(Float(1e30)), selected)
            }
            if localBlocks > 0 {
                let start = maximum(currentBlock - localBlocks + 1, 0)
                let local =
                    (blocks[.newAxis, .newAxis, 0...] .>= start[0..., 0..., .newAxis])
                    .&& (blocks[.newAxis, .newAxis, 0...]
                        .<= currentBlock[0..., 0..., .newAxis])
                selected = MLX.where(local .&& validBlocks, MLXArray(Float(1e29)), selected)
            }

            let width = Swift.min(topKBlocks, blockCount)
            let indices = argPartition(-selected, kth: width - 1, axis: -1)[
                .ellipsis, ..<width]
            let indicesValid = takeAlong(validBlocks, indices, axis: -1)
            return Self.pack(indices: indices, valid: indicesValid, blockCount: blockCount)
        }

        /// Invalid slots sort to the end and come back as -1, so a consumer can stop at
        /// the first negative rather than checking a separate mask.
        static func pack(indices: MLXArray, valid: MLXArray, blockCount: Int) -> MLXArray {
            let sentinel = MLXArray.full(indices.shape, values: MLXArray(Int32(blockCount)))
            var packed = MLX.where(valid, indices.asType(.int32), sentinel)
            let order = argSort(packed, axis: -1)
            packed = takeAlong(packed, order, axis: -1)
            return MLX.where(
                packed .== MLXArray(Int32(blockCount)), MLXArray(Int32(-1)), packed)
        }

        /// Expands the selected blocks back into a per-key boolean mask.
        func blockMask(
            indices: MLXArray, keyLength: Int, queryPositions: MLXArray
        ) -> MLXArray {
            let (B, L) = (indices.dim(0), indices.dim(1))
            let blockCount = (keyLength + blockSize - 1) / blockSize
            let blocks = MLXArray(0 ..< blockCount).asType(.int32)
            let keep = any(
                indices[.ellipsis, .newAxis] .== blocks, axis: -2)

            let keyPositions = MLXArray(0 ..< keyLength)
            let keyBlocks = broadcast(
                floorDivide(keyPositions, blockSize).asType(.int32)[
                    .newAxis, .newAxis, 0...],
                to: [B, L, keyLength])
            let keyKeep = takeAlong(keep, keyBlocks, axis: -1)
            let causal =
                keyPositions[.newAxis, .newAxis, 0...]
                .<= queryPositions[0..., 0..., .newAxis]
            return (keyKeep .&& causal).expandedDimensions(axis: 1)
        }
    }
}

// MARK: - Attention

extension MiniMaxM3Text {

    /// Grouped-query attention with partial rotary, per-head QK norms, and -- on the
    /// layers the schedule names -- a block-sparse index that decides which keys a query
    /// may read.
    ///
    /// The index projections live on this module rather than inside the selector so the
    /// checkpoint keys stay where a converted checkpoint puts them.
    final class Attention: Module {
        let heads: Int
        let kvHeads: Int
        let headDim: Int
        let scale: Float
        let useQKNorm: Bool
        let hasSparseIndex: Bool
        let indexHeads: Int
        let indexDim: Int
        let selector: BlockSelector?

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear
        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?
        @ModuleInfo(key: "index_q_proj") var indexQProj: Linear?
        @ModuleInfo(key: "index_k_proj") var indexKProj: Linear?
        @ModuleInfo(key: "index_q_norm") var indexQNorm: RMSNorm?
        @ModuleInfo(key: "index_k_norm") var indexKNorm: RMSNorm?

        let rope: RoPELayer

        init(_ config: MiniMaxM3TextConfiguration, layerIndex: Int) {
            heads = config.attentionHeads
            kvHeads = config.kvHeads
            headDim = config.headDim
            scale = pow(Float(config.headDim), -0.5)
            useQKNorm = config.useQKNorm
            hasSparseIndex = config.isSparseAttentionLayer(layerIndex)
            indexHeads = config.indexHeads
            indexDim = config.indexHeadDim

            _qProj.wrappedValue = Linear(
                config.hiddenSize, heads * headDim, bias: false)
            _kProj.wrappedValue = Linear(
                config.hiddenSize, kvHeads * headDim, bias: false)
            _vProj.wrappedValue = Linear(
                config.hiddenSize, kvHeads * headDim, bias: false)
            _oProj.wrappedValue = Linear(
                heads * headDim, config.hiddenSize, bias: false)

            if config.useQKNorm {
                _qNorm.wrappedValue = RMSNorm(
                    dimensions: headDim, eps: config.rmsNormEps, gemma: config.useGemmaNorm)
                _kNorm.wrappedValue = RMSNorm(
                    dimensions: headDim, eps: config.rmsNormEps, gemma: config.useGemmaNorm)
            }

            if hasSparseIndex {
                _indexQProj.wrappedValue = Linear(
                    config.hiddenSize, indexHeads * indexDim, bias: false)
                _indexKProj.wrappedValue = Linear(config.hiddenSize, indexDim, bias: false)
                _indexQNorm.wrappedValue = RMSNorm(
                    dimensions: indexDim, eps: config.rmsNormEps, gemma: config.useGemmaNorm)
                _indexKNorm.wrappedValue = RMSNorm(
                    dimensions: indexDim, eps: config.rmsNormEps, gemma: config.useGemmaNorm)
                selector = BlockSelector(
                    blockSize: config.indexBlockSize,
                    topKBlocks: config.indexTopKBlocks,
                    initialBlocks: config.indexInitialBlocks,
                    localBlocks: config.indexLocalBlocks,
                    scale: pow(Float(config.headDim), -0.5))
                precondition(
                    config.attentionHeads % indexHeads == 0,
                    "MiniMax M3's index heads have to divide its attention heads")
            } else {
                selector = nil
            }

            rope = initializeRope(
                dims: config.rotaryDim,
                base: config.ropeTheta,
                traditional: false,
                scalingConfig: config.ropeScaling,
                maxPositionEmbeddings: config.maxPositionEmbeddings)
            super.init()
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
            let (B, L) = (x.dim(0), x.dim(1))
            let offset = cache?.offset ?? 0

            var queries = qProj(x).reshaped(B, L, heads, headDim)
            var keys = kProj(x).reshaped(B, L, kvHeads, headDim)
            var values = vProj(x).reshaped(B, L, kvHeads, headDim)

            // The head norms come before the transpose, so they act on the head axis's
            // last dimension rather than the sequence's.
            if useQKNorm, let qNorm, let kNorm {
                queries = qNorm(queries)
                keys = kNorm(keys)
            }

            queries = queries.transposed(0, 2, 1, 3)
            keys = keys.transposed(0, 2, 1, 3)
            values = values.transposed(0, 2, 1, 3)

            queries = rope(queries, offset: offset)
            keys = rope(keys, offset: offset)
            if let cache {
                (keys, values) = cache.update(keys: keys, values: values)
            }

            var attentionMask = mask
            // A missing cache still selects, over the keys this forward just built --
            // which is what keeps a one-shot forward comparable with a cached one.
            if hasSparseIndex, let selector, cache == nil || cache is IndexKVCache {
                let positions =
                    MLXArray(Int32(offset) ..< Int32(offset + L))[.newAxis, 0...]
                let queryPositions = broadcast(positions, to: [B, L])

                var indexQueries = indexQProj!(x).reshaped(B, L, indexHeads, indexDim)
                var indexKeys = indexKProj!(x).reshaped(B, L, 1, indexDim)
                indexQueries = indexQNorm!(indexQueries).transposed(0, 2, 1, 3)
                indexKeys = indexKNorm!(indexKeys).transposed(0, 2, 1, 3)
                indexQueries = rope(indexQueries, offset: offset)
                indexKeys = rope(indexKeys, offset: offset)
                if let indexCache = cache as? IndexKVCache {
                    indexKeys = indexCache.updateIndexAndFetch(indexKeys)
                }

                // Everything cached still fits in the budget, so there is nothing to
                // select and the attention stays dense.
                if indexKeys.dim(2) > selector.blockSize * selector.topKBlocks {
                    let indices = selector.select(
                        queries: indexQueries, keys: indexKeys,
                        queryPositions: queryPositions)
                    var sparse = selector.blockMask(
                        indices: indices, keyLength: keys.dim(2),
                        queryPositions: queryPositions)
                    if let existing = attentionMask, existing.dtype == .bool {
                        sparse = sparse .&& existing
                    }
                    attentionMask = sparse
                }
            }

            let output = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: values, scale: scale,
                mask: attentionMask.map { .array($0) } ?? .none)
            return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
        }
    }

    /// The routed feed-forward. When the shared expert's width matches the bank's it
    /// rides along as one more expert with a weight of exactly one, which is what lets a
    /// checkpoint quantize the two together.
    final class SparseMoEBlock: Module, UnaryLayer {
        let expertsPerToken: Int
        let routedScalingFactor: Float
        let sharedExpertIndex: Int
        let packsSharedExpert: Bool

        @ModuleInfo var gate: Linear
        @ModuleInfo(key: "switch_mlp") var switchMLP: Module
        @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray?
        @ModuleInfo(key: "shared_experts") var sharedExperts: MLP?

        init(_ config: MiniMaxM3TextConfiguration) {
            expertsPerToken = config.expertsPerToken
            routedScalingFactor = config.routedScalingFactor
            sharedExpertIndex = config.localExperts
            packsSharedExpert = config.packsSharedExpert

            let alpha = config.swiGLUAlpha
            let limit = config.swiGLULimit
            let beta = config.swiGLUBeta
            // The fused projection hands its combiner (gate, up); the activation wants
            // (linear, gate), so the pair is swapped here.
            let activationProduct: @Sendable (MLXArray, MLXArray) -> MLXArray = { gate, up in
                swiGLUOAI(linear: up, gate: gate, alpha: alpha, limit: limit, beta: beta)
            }

            _gate.wrappedValue = Linear(config.hiddenSize, config.localExperts, bias: false)
            if packsSharedExpert {
                _switchMLP.wrappedValue = FusedGateUpSwitchGLU(
                    inputDims: config.hiddenSize,
                    hiddenDims: config.intermediateSize,
                    numExperts: config.localExperts + 1,
                    activationProduct: activationProduct)
            } else {
                _switchMLP.wrappedValue = SwitchGLU(
                    inputDims: config.hiddenSize,
                    hiddenDims: config.intermediateSize,
                    numExperts: config.localExperts,
                    activationProduct: activationProduct)
                if config.sharedExperts > 0 {
                    _sharedExperts.wrappedValue = MLP(
                        hiddenSize: config.hiddenSize,
                        intermediateSize: config.sharedIntermediateSize,
                        alpha: alpha, limit: limit, beta: beta)
                }
            }
            if config.useRoutingBias {
                _correctionBias.wrappedValue = MLXArray.zeros([config.localExperts])
            }
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let gates = gate(x.asType(.float32))
            var (indices, weights) = selectExperts(
                gates: gates, bias: correctionBias, topK: expertsPerToken,
                routedScalingFactor: routedScalingFactor)
            weights = weights.asType(x.dtype)

            if packsSharedExpert {
                let sharedIndices = MLXArray.full(
                    Array(indices.shape.dropLast()) + [1],
                    values: MLXArray(Int32(sharedExpertIndex)))
                let sharedWeights = MLXArray.ones(
                    Array(weights.shape.dropLast()) + [1], dtype: weights.dtype)
                indices = concatenated([indices.asType(.int32), sharedIndices], axis: -1)
                weights = concatenated([weights, sharedWeights], axis: -1)
            }

            var y: MLXArray
            if let fused = switchMLP as? FusedGateUpSwitchGLU {
                y = fused(x, indices)
            } else if let plain = switchMLP as? SwitchGLU {
                y = plain(x, indices)
            } else {
                fatalError("unexpected switched projection")
            }
            y = (y * weights[.ellipsis, .newAxis]).sum(axis: -2)

            if let sharedExperts {
                y = y + sharedExperts(x)
            }
            return y
        }
    }

    final class DecoderLayer: Module {
        let isMoELayer: Bool

        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
        @ModuleInfo(key: "block_sparse_moe") var moe: SparseMoEBlock?
        @ModuleInfo var mlp: MLP?

        init(_ config: MiniMaxM3TextConfiguration, layerIndex: Int) {
            isMoELayer = config.isMoELayer(layerIndex)
            _attention.wrappedValue = Attention(config, layerIndex: layerIndex)
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps,
                gemma: config.useGemmaNorm)
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps,
                gemma: config.useGemmaNorm)
            if isMoELayer {
                _moe.wrappedValue = SparseMoEBlock(config)
            } else {
                // A dense layer is wider than a routed expert.
                _mlp.wrappedValue = MLP(
                    hiddenSize: config.hiddenSize,
                    intermediateSize: config.denseIntermediateSize,
                    alpha: config.swiGLUAlpha, limit: config.swiGLULimit,
                    beta: config.swiGLUBeta)
            }
            super.init()
        }

        func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
            let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
            let normalized = postAttentionLayerNorm(h)
            return h + (isMoELayer ? moe!(normalized) : mlp!(normalized))
        }
    }

    final class Backbone: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo var layers: [DecoderLayer]
        @ModuleInfo var norm: RMSNorm

        init(_ config: MiniMaxM3TextConfiguration) {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
                DecoderLayer(config, layerIndex: $0)
            }
            _norm.wrappedValue = RMSNorm(
                dimensions: config.hiddenSize, eps: config.rmsNormEps,
                gemma: config.useGemmaNorm)
            super.init()
        }

        func callAsFunction(
            inputs: MLXArray?, inputEmbedding: MLXArray?, cache: [KVCache]?
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
                h = layer(h, mask: mask, cache: cache?[index])
            }
            return norm(h)
        }
    }
}

// MARK: - Vision

/// MiniMax M3's vision half.
public struct MiniMaxM3VisionConfiguration: Decodable, Sendable {
    public var modelType: String = "clip_vision_model"
    public var hiddenSize: Int = 1280
    public var intermediateSize: Int = 5120
    public var attentionHeads: Int = 16
    public var hiddenLayers: Int = 32
    public var patchSize: Int = 14
    public var channels: Int = 3
    public var layerNormEps: Float = 1e-5
    public var hiddenActivation: String = "gelu"
    public var ropeTheta: Float = 10000
    public var segmentMaxFrames: Int = 4
    public var spatialMergeSize: Int = 2
    public var temporalPatchSize: Int = 2

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case hiddenLayers = "num_hidden_layers"
        case patchSize = "patch_size"
        case channels = "num_channels"
        case layerNormEps = "layer_norm_eps"
        case hiddenActivation = "hidden_act"
        case ropeTheta = "rope_theta"
        case segmentMaxFrames = "vision_segment_max_frames"
        case compression = "img_token_compression_config"
    }

    private struct Compression: Decodable {
        var spatialMergeSize: Int?
        var temporalPatchSize: Int?

        enum CodingKeys: String, CodingKey {
            case spatialMergeSize = "spatial_merge_size"
            case temporalPatchSize = "temporal_patch_size"
        }
    }

    public init() {}

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? modelType
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? hiddenSize
        intermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? intermediateSize
        attentionHeads =
            try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? attentionHeads
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? hiddenLayers
        patchSize = try c.decodeIfPresent(Int.self, forKey: .patchSize) ?? patchSize
        channels = try c.decodeIfPresent(Int.self, forKey: .channels) ?? channels
        layerNormEps =
            try c.decodeIfPresent(Float.self, forKey: .layerNormEps) ?? layerNormEps
        hiddenActivation =
            try c.decodeIfPresent(String.self, forKey: .hiddenActivation)
            ?? hiddenActivation
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? ropeTheta
        segmentMaxFrames =
            try c.decodeIfPresent(Int.self, forKey: .segmentMaxFrames) ?? segmentMaxFrames
        if let compression = try c.decodeIfPresent(Compression.self, forKey: .compression) {
            spatialMergeSize = compression.spatialMergeSize ?? spatialMergeSize
            temporalPatchSize = compression.temporalPatchSize ?? temporalPatchSize
        }
    }

    public var headDim: Int { hiddenSize / attentionHeads }
    public var patchDim: Int { channels * temporalPatchSize * patchSize * patchSize }
}

enum MiniMaxM3Vision {

    static func activation(_ x: MLXArray, kind: String) -> MLXArray {
        switch kind {
        case "quick_gelu": return x * sigmoid(1.702 * x)
        case "silu": return silu(x)
        default: return gelu(x)
        }
    }

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        return concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
    }

    /// Rotates only the leading `cos.dim(-1)` channels and passes the rest through, which
    /// is how a three-axis table that does not divide the head dimension is applied.
    static func applyRoPE(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let c = cos[.newAxis, 0..., .newAxis, 0...]
        let s = sin[.newAxis, 0..., .newAxis, 0...]
        let width = c.dim(-1)
        let rotated = x[.ellipsis, ..<width]
        let passthrough = x[.ellipsis, width...]
        let out = rotated * c + rotateHalf(rotated) * s
        return concatenated([out, passthrough], axis: -1).asType(dtype)
    }

    /// The patch embedding is stored as a five-dimensional convolution weight but applied
    /// as one matmul over flattened patches.
    final class PatchEmbedding: Module {
        let hiddenSize: Int
        let patchDim: Int

        @ParameterInfo(key: "weight") var weight: MLXArray

        init(_ config: MiniMaxM3VisionConfiguration) {
            hiddenSize = config.hiddenSize
            patchDim = config.patchDim
            _weight.wrappedValue = MLXArray.zeros([
                config.hiddenSize, config.channels, config.temporalPatchSize,
                config.patchSize, config.patchSize,
            ])
            super.init()
        }

        func callAsFunction(_ pixelValues: MLXArray) -> MLXArray {
            let shape = pixelValues.shape
            let flat = pixelValues.reshaped(-1, patchDim)
            let matrix = weight.reshaped(hiddenSize, patchDim)
            return flat.matmul(matrix.transposed())
                .reshaped(Array(shape.dropLast()) + [hiddenSize])
        }
    }

    final class Embeddings: Module {
        @ModuleInfo(key: "patch_embedding") var patchEmbedding: PatchEmbedding

        init(_ config: MiniMaxM3VisionConfiguration) {
            _patchEmbedding.wrappedValue = PatchEmbedding(config)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            patchEmbedding(x)
        }
    }

    final class Attention: Module {
        let heads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ config: MiniMaxM3VisionConfiguration) {
            heads = config.attentionHeads
            headDim = config.headDim
            scale = pow(Float(config.headDim), -0.5)
            _qProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _kProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _vProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            _outProj.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)
            super.init()
        }

        func callAsFunction(
            _ hiddenStates: MLXArray, boundaries: [Int],
            positionEmbeddings: (cos: MLXArray, sin: MLXArray)?
        ) -> MLXArray {
            let length = hiddenStates.dim(0)
            var q = qProj(hiddenStates).reshaped(length, heads, -1)
            var k = kProj(hiddenStates).reshaped(length, heads, -1)
            let v = vProj(hiddenStates).reshaped(length, heads, -1)

            if let positionEmbeddings {
                q =
                    applyRoPE(
                        q.expandedDimensions(axis: 0), cos: positionEmbeddings.cos,
                        sin: positionEmbeddings.sin)[0]
                k =
                    applyRoPE(
                        k.expandedDimensions(axis: 0), cos: positionEmbeddings.cos,
                        sin: positionEmbeddings.sin)[0]
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
            return outProj(attended[0].transposed(1, 0, 2).reshaped(length, -1))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo var fc1: Linear
        @ModuleInfo var fc2: Linear
        let kind: String

        init(_ config: MiniMaxM3VisionConfiguration) {
            _fc1.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize, bias: true)
            _fc2.wrappedValue = Linear(
                config.intermediateSize, config.hiddenSize, bias: true)
            kind = config.hiddenActivation
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            fc2(activation(fc1(x), kind: kind))
        }
    }

    final class EncoderLayer: Module {
        @ModuleInfo(key: "self_attn") var attention: Attention
        @ModuleInfo(key: "layer_norm1") var layerNorm1: LayerNorm
        @ModuleInfo var mlp: MLP
        @ModuleInfo(key: "layer_norm2") var layerNorm2: LayerNorm

        init(_ config: MiniMaxM3VisionConfiguration) {
            _attention.wrappedValue = Attention(config)
            _layerNorm1.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            _mlp.wrappedValue = MLP(config)
            _layerNorm2.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray, boundaries: [Int],
            positionEmbeddings: (cos: MLXArray, sin: MLXArray)?
        ) -> MLXArray {
            let h =
                x
                + attention(
                    layerNorm1(x), boundaries: boundaries,
                    positionEmbeddings: positionEmbeddings)
            return h + mlp(layerNorm2(h))
        }
    }

    final class Encoder: Module {
        @ModuleInfo var layers: [EncoderLayer]

        init(_ config: MiniMaxM3VisionConfiguration) {
            _layers.wrappedValue = (0 ..< config.hiddenLayers).map { _ in
                EncoderLayer(config)
            }
            super.init()
        }

        /// Returns the final state and, when asked, every intermediate one -- seeded with
        /// the input, so index `i` is layer `i - 1`'s output.
        func callAsFunction(
            _ x: MLXArray, boundaries: [Int],
            positionEmbeddings: (cos: MLXArray, sin: MLXArray)?,
            outputHiddenStates: Bool
        ) -> (MLXArray, [MLXArray]?) {
            var states: [MLXArray]? = outputHiddenStates ? [x] : nil
            var h = x
            for layer in layers {
                h = layer(h, boundaries: boundaries, positionEmbeddings: positionEmbeddings)
                states?.append(h)
            }
            return (h, states)
        }
    }

    /// The tower. `pre_layrnorm` is spelled the way the checkpoint spells it.
    final class Transformer: Module {
        let config: MiniMaxM3VisionConfiguration

        @ModuleInfo var embeddings: Embeddings
        @ModuleInfo(key: "pre_layrnorm") var preLayerNorm: LayerNorm
        @ModuleInfo var encoder: Encoder

        init(_ config: MiniMaxM3VisionConfiguration) {
            self.config = config
            _embeddings.wrappedValue = Embeddings(config)
            _preLayerNorm.wrappedValue = LayerNorm(
                dimensions: config.hiddenSize, eps: config.layerNormEps)
            _encoder.wrappedValue = Encoder(config)
            super.init()
        }

        /// A long video is cut into segments of at most `vision_segment_max_frames`, and
        /// each segment attends on its own.
        func segments(_ grid: [[Int]]) -> [[Int]] {
            var out: [[Int]] = []
            for entry in grid {
                let (t, h, w) = (entry[0], entry[1], entry[2])
                if config.segmentMaxFrames <= 0 || t <= config.segmentMaxFrames {
                    out.append([t, h, w])
                    continue
                }
                var start = 0
                while start < t {
                    out.append([Swift.min(config.segmentMaxFrames, t - start), h, w])
                    start += config.segmentMaxFrames
                }
            }
            return out
        }

        /// Three axes -- frame, row, column -- each taking a slice of the rotary width,
        /// and the whole thing doubled so the rotation covers both halves.
        func rotaryAngles(_ grid: [[Int]]) -> MLXArray {
            let merge = config.spatialMergeSize
            let ropeDims = 2 * (config.headDim / 2)
            let axisDim = 2 * ((ropeDims / 3) / 2)

            func axisFrequencies(_ coordinates: [Int32]) -> MLXArray {
                let inverse =
                    1
                    / pow(
                        MLXArray(config.ropeTheta),
                        MLXArray(stride(from: 0, to: axisDim, by: 2).map(Float.init))
                            / Float(axisDim))
                return outer(MLXArray(coordinates).asType(.float32), inverse)
            }

            var rows: [MLXArray] = []
            for entry in segments(grid) {
                let (t, h, w) = (entry[0], entry[1], entry[2])
                let mergedH = h / merge
                let mergedW = w / merge
                var frames: [Int32] = []
                var heights: [Int32] = []
                var widths: [Int32] = []
                for frame in 0 ..< t {
                    for blockRow in 0 ..< mergedH {
                        for blockColumn in 0 ..< mergedW {
                            for innerRow in 0 ..< merge {
                                for innerColumn in 0 ..< merge {
                                    frames.append(Int32(frame))
                                    heights.append(Int32(blockRow * merge + innerRow))
                                    widths.append(Int32(blockColumn * merge + innerColumn))
                                }
                            }
                        }
                    }
                }
                let combined = concatenated(
                    [
                        axisFrequencies(frames), axisFrequencies(heights),
                        axisFrequencies(widths),
                    ], axis: -1)
                rows.append(concatenated([combined, combined], axis: -1))
            }
            return concatenated(rows, axis: 0)
        }

        func callAsFunction(
            _ pixelValues: MLXArray, grid: [[Int]], outputHiddenStates: Bool = false
        ) -> (MLXArray, [MLXArray]?) {
            var h = embeddings(pixelValues).reshaped(-1, config.hiddenSize)
            h = preLayerNorm(h)
            let angles = rotaryAngles(grid)
            let positionEmbeddings = (cos: cos(angles), sin: sin(angles))

            var boundaries = [0]
            for entry in segments(grid) {
                boundaries.append(
                    boundaries[boundaries.count - 1] + entry[0] * entry[1] * entry[2])
            }

            return encoder(
                h, boundaries: boundaries, positionEmbeddings: positionEmbeddings,
                outputHiddenStates: outputHiddenStates)
        }
    }

    final class VisionModel: Module {
        @ModuleInfo(key: "vision_model") var visionModel: Transformer

        init(_ config: MiniMaxM3VisionConfiguration) {
            _visionModel.wrappedValue = Transformer(config)
            super.init()
        }

        func callAsFunction(
            _ pixelValues: MLXArray, grid: [[Int]], outputHiddenStates: Bool = false
        ) -> (MLXArray, [MLXArray]?) {
            visionModel(pixelValues, grid: grid, outputHiddenStates: outputHiddenStates)
        }
    }

    /// The projector, whose activation is named rather than fixed.
    final class Projector: Module {
        @ModuleInfo(key: "linear_1") var linear1: Linear
        @ModuleInfo(key: "linear_2") var linear2: Linear
        let kind: String

        init(
            inputDimensions: Int, hiddenDimensions: Int, outputDimensions: Int,
            bias: Bool, kind: String
        ) {
            _linear1.wrappedValue = Linear(inputDimensions, hiddenDimensions, bias: bias)
            _linear2.wrappedValue = Linear(hiddenDimensions, outputDimensions, bias: bias)
            self.kind = kind
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear2(activation(linear1(x), kind: kind))
        }
    }
}

// MARK: - Model

/// The whole family's configuration: the text half, the vision half, and the projector
/// settings that sit between them.
public struct MiniMaxM3Configuration: Decodable, Sendable {
    public var modelType: String
    public var textConfiguration: MiniMaxM3TextConfiguration
    public var visionConfiguration: MiniMaxM3VisionConfiguration
    public var imageTokenID: Int
    public var videoTokenID: Int
    public var projectorHiddenSize: Int
    public var projectorActivation: String
    public var projectorBias: Bool
    public var patchMergeBias: Bool
    public var visionFeatureLayer: [Int]
    public var visionFeatureSelectStrategy: String

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case imageTokenIndex = "image_token_index"
        case imageTokenID = "image_token_id"
        case videoTokenIndex = "video_token_index"
        case videoTokenID = "video_token_id"
        case projectorHiddenSize = "projector_hidden_size"
        case projectorActivation = "projector_hidden_act"
        case projectorBias = "multimodal_projector_bias"
        case patchMergeBias = "patch_merge_bias"
        case visionFeatureLayer = "vision_feature_layer"
        case visionFeatureSelectStrategy = "vision_feature_select_strategy"
        case compression = "img_token_compression_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "minimax_m3_vl"
        textConfiguration =
            try c.decodeIfPresent(
                MiniMaxM3TextConfiguration.self, forKey: .textConfiguration)
            ?? MiniMaxM3TextConfiguration()

        var vision =
            try c.decodeIfPresent(
                MiniMaxM3VisionConfiguration.self, forKey: .visionConfiguration)
            ?? MiniMaxM3VisionConfiguration()
        // The compression settings can sit at the top level instead of inside the vision
        // half, in which case the vision half inherits them.
        if let top = try? c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .compression)
        {
            if let merge = top["spatial_merge_size"]?.asInt() {
                vision.spatialMergeSize = merge
            }
            if let temporal = top["temporal_patch_size"]?.asInt() {
                vision.temporalPatchSize = temporal
            }
        }
        visionConfiguration = vision

        let imageIndex = try c.decodeIfPresent(Int.self, forKey: .imageTokenIndex) ?? 200_025
        let videoIndex = try c.decodeIfPresent(Int.self, forKey: .videoTokenIndex) ?? 200_026
        imageTokenID = try c.decodeIfPresent(Int.self, forKey: .imageTokenID) ?? imageIndex
        videoTokenID = try c.decodeIfPresent(Int.self, forKey: .videoTokenID) ?? videoIndex

        projectorHiddenSize =
            try c.decodeIfPresent(Int.self, forKey: .projectorHiddenSize) ?? 6144
        projectorActivation =
            try c.decodeIfPresent(String.self, forKey: .projectorActivation) ?? "gelu"
        projectorBias = try c.decodeIfPresent(Bool.self, forKey: .projectorBias) ?? true
        patchMergeBias = try c.decodeIfPresent(Bool.self, forKey: .patchMergeBias) ?? true
        if let single = try? c.decodeIfPresent(Int.self, forKey: .visionFeatureLayer) {
            visionFeatureLayer = [single]
        } else {
            visionFeatureLayer =
                try c.decodeIfPresent([Int].self, forKey: .visionFeatureLayer) ?? [-1]
        }
        visionFeatureSelectStrategy =
            try c.decodeIfPresent(String.self, forKey: .visionFeatureSelectStrategy)
            ?? "full"
    }

    /// Whether the tower has to return every hidden state, which it only does when the
    /// selection asks for something other than the last one in full.
    public var needsHiddenStates: Bool {
        visionFeatureLayer != [-1] || visionFeatureSelectStrategy != "full"
    }
}

public final class MiniMaxM3: Module, VLMModel, KVCacheDimensionProvider {

    @ModuleInfo(key: "vision_tower") private var visionTower: MiniMaxM3Vision.VisionModel
    @ModuleInfo(key: "language_model") private var languageModel: LanguageHalf
    @ModuleInfo(key: "multi_modal_projector") private var projector: MiniMaxM3Vision.Projector
    @ModuleInfo(key: "patch_merge_mlp") private var patchMerge: MiniMaxM3Vision.Projector

    fileprivate final class LanguageHalf: Module {
        @ModuleInfo var model: MiniMaxM3Text.Backbone
        @ModuleInfo(key: "lm_head") var lmHead: Linear?

        init(_ config: MiniMaxM3TextConfiguration) {
            _model.wrappedValue = MiniMaxM3Text.Backbone(config)
            if !config.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.hiddenSize, config.vocabularySize, bias: false)
            }
            super.init()
        }
    }

    public let config: MiniMaxM3Configuration

    public var vocabularySize: Int { config.textConfiguration.vocabularySize }
    public var kvHeads: [Int] {
        Array(
            repeating: config.textConfiguration.kvHeads,
            count: config.textConfiguration.hiddenLayers)
    }

    public init(_ config: MiniMaxM3Configuration) {
        self.config = config
        _visionTower.wrappedValue = MiniMaxM3Vision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = LanguageHalf(config.textConfiguration)

        let vision = config.visionConfiguration
        _projector.wrappedValue = MiniMaxM3Vision.Projector(
            inputDimensions: vision.hiddenSize * config.visionFeatureLayer.count,
            hiddenDimensions: config.projectorHiddenSize,
            outputDimensions: config.textConfiguration.hiddenSize,
            bias: config.projectorBias, kind: config.projectorActivation)
        _patchMerge.wrappedValue = MiniMaxM3Vision.Projector(
            inputDimensions: config.textConfiguration.hiddenSize
                * vision.spatialMergeSize * vision.spatialMergeSize,
            hiddenDimensions: config.textConfiguration.hiddenSize,
            outputDimensions: config.textConfiguration.hiddenSize,
            bias: config.patchMergeBias, kind: config.projectorActivation)
    }

    /// The layers with an index carry the indexer's keys alongside the ordinary ones.
    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        let text = config.textConfiguration
        return try (0 ..< text.hiddenLayers).map { index in
            text.isSparseAttentionLayer(index)
                ? MiniMaxM3Text.IndexKVCache()
                : try makeAttentionKVCache(configuration: cacheConfiguration)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(inputs: inputs, inputEmbedding: nil, cache: cache)
    }

    private func logits(inputs: MLXArray?, inputEmbedding: MLXArray?, cache: [KVCache]?)
        -> MLXArray
    {
        let hidden = languageModel.model(
            inputs: inputs, inputEmbedding: inputEmbedding, cache: cache)
        if let head = languageModel.lmHead {
            return head(hidden)
        }
        return languageModel.model.embedTokens.asLinear(hidden)
    }

    /// The tower's output, projected and then folded down by the spatial merge.
    func visualFeatures(_ pixelValues: MLXArray, grid: [[Int]]) -> MLXArray {
        let dtype = visionTower.visionModel.embeddings.patchEmbedding.weight.dtype
        let (last, hiddenStates) = visionTower(
            pixelValues.asType(dtype), grid: grid,
            outputHiddenStates: config.needsHiddenStates)

        var features: MLXArray
        if let hiddenStates, config.needsHiddenStates {
            let selected = config.visionFeatureLayer.map { layer -> MLXArray in
                let index = layer < 0 ? hiddenStates.count + layer : layer
                return Self.applySelectStrategy(
                    hiddenStates[Swift.min(Swift.max(index, 0), hiddenStates.count - 1)],
                    strategy: config.visionFeatureSelectStrategy)
            }
            features = selected.count == 1 ? selected[0] : concatenated(selected, axis: -1)
        } else {
            features = last
        }

        features = projector(features)
        return mergeVisualTokens(features, grid: grid)
    }

    private static func applySelectStrategy(_ features: MLXArray, strategy: String)
        -> MLXArray
    {
        switch strategy {
        case "full": return features
        case "default":
            return features.ndim >= 3 ? features[0..., 1...] : features[1...]
        default:
            preconditionFailure("Unexpected feature selection strategy: \(strategy)")
        }
    }

    /// Each merge block's rows are flattened into one and passed through the merge head,
    /// so an image's token count drops by the square of the merge size.
    private func mergeVisualTokens(_ features: MLXArray, grid: [[Int]]) -> MLXArray {
        let merge = config.visionConfiguration.spatialMergeSize
        let dimensions = features.dim(-1)
        var outputs: [MLXArray] = []
        var offset = 0
        for entry in grid {
            let (t, h, w) = (entry[0], entry[1], entry[2])
            let length = t * h * w
            let slice = features[offset ..< (offset + length)]
            offset += length
            outputs.append(
                patchMerge(
                    slice
                        .reshaped(t, h / merge, w / merge, merge, merge, dimensions)
                        .reshaped(-1, merge * merge * dimensions)))
        }
        return concatenated(outputs, axis: 0)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state _: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }

        let embeddings = try input.image.map { image -> MLXArray in
            guard let frames = image.frames, !frames.isEmpty else {
                throw VLMError.imageRequired
            }
            let grid = frames.map { [$0.t, $0.h, $0.w] }
            let features = visualFeatures(image.pixels, grid: grid)
            let embedded = languageModel.model.embedTokens(tokens)
            let selected =
                (tokens .== MLXArray(Int32(config.imageTokenID)))
                .|| (tokens .== MLXArray(Int32(config.videoTokenID)))
            return Self.scatter(
                embedded: embedded, features: features.asType(embedded.dtype),
                selected: selected)
        }

        let total = embeddings?.dim(1) ?? tokens.dim(1)

        func forward(_ range: Range<Int>) -> MLXArray {
            if let embeddings {
                return logits(
                    inputs: nil, inputEmbedding: embeddings[0..., range], cache: cache)
            }
            return logits(inputs: tokens[0..., range], inputEmbedding: nil, cache: cache)
        }

        let processed = try prefill.forEachChunk(total: total) { range in
            _ = forward(range)
            asyncEval(cache)
        }
        if processed > 0 { eval(cache) }
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: forward(processed ..< total)))
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

    /// Moves the checkpoint's names onto the module tree.
    ///
    /// The container nests both halves one level deeper than the modules do, block-scaled
    /// weights arrive as `weight_scale_inv` beside a uint8 payload that has to be packed
    /// four values to a word, and the routed experts arrive one per key -- fused with the
    /// shared expert when the two widths agree.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Self.sanitize(weights: weights, text: config.textConfiguration)
    }

    static func sanitize(weights: [String: MLXArray], text: MiniMaxM3TextConfiguration) -> [String:
        MLXArray]
    {
        var moved: [String: MLXArray] = [:]
        moved.reserveCapacity(weights.count)

        for (rawKey, value) in weights {
            var key = rawKey
            for (from, to) in [
                ("model.language_model.", "language_model."),
                ("model.vision_tower.", "vision_tower."),
                ("model.multi_modal_projector.", "multi_modal_projector."),
                ("model.patch_merge_mlp.", "patch_merge_mlp."),
            ] where key.hasPrefix(from) {
                key = to + String(key.dropFirst(from.count))
                break
            }
            moved[key] = value
        }

        // A `weight_scale_inv` marks its sibling as a packed uint8 payload.
        let scaled = moved.keys.filter { $0.hasSuffix(".weight_scale_inv") }
        for scaleKey in scaled {
            let weightKey = scaleKey.replacingOccurrences(
                of: ".weight_scale_inv", with: ".weight")
            if let weight = moved[weightKey] {
                moved[weightKey] = Self.packUInt8(weight)
            }
            moved[scaleKey.replacingOccurrences(of: ".weight_scale_inv", with: ".scales")] =
                moved.removeValue(forKey: scaleKey)
        }

        for index in 0 ..< text.hiddenLayers where text.isMoELayer(index) {
            Self.moveExperts(
                &moved,
                prefix: "language_model.model.layers.\(index).block_sparse_moe",
                experts: text.localExperts, packShared: text.packsSharedExpert)
        }
        return moved
    }

    /// Four uint8 values per uint32 word, little end first.
    static func packUInt8(_ weight: MLXArray) -> MLXArray {
        guard weight.dtype == .uint8, weight.dim(-1) % 4 == 0 else { return weight }
        let shape = Array(weight.shape.dropLast()) + [weight.dim(-1) / 4, 4]
        let shifts = MLXArray([UInt32(0), 8, 16, 24])
        return (weight.reshaped(shape).asType(.uint32) << shifts).sum(axis: -1)
    }

    private static func moveExperts(
        _ weights: inout [String: MLXArray], prefix: String, experts: Int,
        packShared: Bool
    ) {
        func keys(_ name: String, _ suffix: String) -> [String] {
            (0 ..< experts).map { "\(prefix).experts.\($0).\(name).\(suffix)" }
        }

        for suffix in ["weight", "scales", "biases", "bias"] {
            if packShared {
                let gateKeys = keys("w1", suffix)
                let upKeys = keys("w3", suffix)
                let sharedGate = "\(prefix).shared_experts.gate_proj.\(suffix)"
                let sharedUp = "\(prefix).shared_experts.up_proj.\(suffix)"
                if (gateKeys + upKeys + [sharedGate, sharedUp]).allSatisfy({
                    weights[$0] != nil
                }) {
                    let gate = MLX.stacked(gateKeys.map { weights.removeValue(forKey: $0)! })
                    let up = MLX.stacked(upKeys.map { weights.removeValue(forKey: $0)! })
                    let routed = concatenated([gate, up], axis: 1)
                    let shared = concatenated(
                        [
                            weights.removeValue(forKey: sharedGate)!,
                            weights.removeValue(forKey: sharedUp)!,
                        ], axis: 0
                    ).expandedDimensions(axis: 0)
                    weights["\(prefix).switch_mlp.gate_up_proj.\(suffix)"] = concatenated(
                        [routed, shared], axis: 0)
                }

                let downKeys = keys("w2", suffix)
                let sharedDown = "\(prefix).shared_experts.down_proj.\(suffix)"
                if (downKeys + [sharedDown]).allSatisfy({ weights[$0] != nil }) {
                    let down = MLX.stacked(downKeys.map { weights.removeValue(forKey: $0)! })
                    let shared = weights.removeValue(forKey: sharedDown)!
                        .expandedDimensions(axis: 0)
                    weights["\(prefix).switch_mlp.down_proj.\(suffix)"] = concatenated(
                        [down, shared], axis: 0)
                }
                continue
            }

            for (from, to) in [("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj")] {
                let names = keys(from, suffix)
                if names.allSatisfy({ weights[$0] != nil }) {
                    weights["\(prefix).switch_mlp.\(to).\(suffix)"] = MLX.stacked(
                        names.map { weights.removeValue(forKey: $0)! })
                }
            }
        }
    }
}

extension MiniMaxM3: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}

/// MiniMax M3's text-only checkpoint, without vision modules or image preprocessing.
public final class MiniMaxM3Model: Module, LLMModel, KVCacheDimensionProvider, LoRAModel {
    @ModuleInfo(key: "language_model") private var languageModel: MiniMaxM3.LanguageHalf
    public let configuration: MiniMaxM3TextConfiguration
    public var vocabularySize: Int { configuration.vocabularySize }
    public var kvHeads: [Int] {
        Array(repeating: configuration.kvHeads, count: configuration.hiddenLayers)
    }
    public var loraLayers: [Module] { languageModel.model.layers }

    public init(_ configuration: MiniMaxM3TextConfiguration) {
        self.configuration = configuration
        _languageModel.wrappedValue = MiniMaxM3.LanguageHalf(configuration)
        super.init()
    }
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = languageModel.model(inputs: inputs, inputEmbedding: nil, cache: cache)
        if let head = languageModel.lmHead { return head(hidden) }
        return languageModel.model.embedTokens.asLinear(hidden)
    }
    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try (0 ..< self.configuration.hiddenLayers).map { index in
            self.configuration.isSparseAttentionLayer(index)
                ? MiniMaxM3Text.IndexKVCache()
                : try makeAttentionKVCache(configuration: configuration)
        }
    }
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var named: [String: MLXArray] = [:]
        for (key, value) in weights where !key.hasPrefix("mtp.") {
            named[
                key.hasPrefix("model.") || key.hasPrefix("lm_head.") ? "language_model." + key : key
            ] = value
        }
        return MiniMaxM3.sanitize(weights: named, text: configuration)
    }
}
