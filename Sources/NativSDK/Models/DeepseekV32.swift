import Foundation
import MLX
import MLXNN

public struct DeepseekV32Configuration: Decodable, Sendable {
    public var modelType: String = "deepseek_v32"
    public var vocabularySize: Int = 102_400
    public var hiddenSize: Int = 4096
    public var indexHeadDim: Int = 128
    public var indexHeads: Int = 64
    public var indexTopK: Int = 2048
    public var intermediateSize: Int = 11008
    public var moeIntermediateSize: Int = 1407
    public var hiddenLayers: Int = 30
    public var attentionHeads: Int = 32
    public var kvHeads: Int = 32
    public var sharedExperts: Int?
    public var routedExperts: Int?
    public var routedScalingFactor: Float = 1.0
    public var kvLoraRank: Int = 512
    public var qLoraRank: Int = 1536
    public var qkRopeHeadDim: Int = 64
    public var vHeadDim: Int = 128
    public var qkNopeHeadDim: Int = 128
    public var topkMethod: String = "noaux_tc"
    public var scoringFunction: String = "sigmoid"
    public var normTopkProb: Bool = true
    public var groups: Int = 1
    public var topkGroups: Int = 1
    public var expertsPerToken: Int = 1
    public var moeLayerFrequency: Int = 1
    public var firstKDenseReplace: Int = 0
    public var maxPositionEmbeddings: Int = 2048
    public var rmsNormEps: Float = 1e-6
    public var ropeTheta: Float = 10000
    public var ropeScaling: [String: StringOrNumber]?
    public var attentionBias: Bool = false

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case indexHeadDim = "index_head_dim"
        case indexHeads = "index_n_heads"
        case indexTopK = "index_topk"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case sharedExperts = "n_shared_experts"
        case routedExperts = "n_routed_experts"
        case routedScalingFactor = "routed_scaling_factor"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case topkMethod = "topk_method"
        case scoringFunction = "scoring_func"
        case normTopkProb = "norm_topk_prob"
        case groups = "n_group"
        case topkGroups = "topk_group"
        case expertsPerToken = "num_experts_per_tok"
        case moeLayerFrequency = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case ropeParameters = "rope_parameters"
        case attentionBias = "attention_bias"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func take<T: Decodable>(_ key: CodingKeys, _ fallback: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: key) ?? fallback
        }
        modelType = try take(.modelType, modelType)
        vocabularySize = try take(.vocabularySize, vocabularySize)
        hiddenSize = try take(.hiddenSize, hiddenSize)
        indexHeadDim = try take(.indexHeadDim, indexHeadDim)
        indexHeads = try take(.indexHeads, indexHeads)
        indexTopK = try take(.indexTopK, indexTopK)
        intermediateSize = try take(.intermediateSize, intermediateSize)
        moeIntermediateSize = try take(.moeIntermediateSize, moeIntermediateSize)
        hiddenLayers = try take(.hiddenLayers, hiddenLayers)
        attentionHeads = try take(.attentionHeads, attentionHeads)
        kvHeads = try take(.kvHeads, kvHeads)
        sharedExperts = try c.decodeIfPresent(Int.self, forKey: .sharedExperts)
        routedExperts = try c.decodeIfPresent(Int.self, forKey: .routedExperts)
        routedScalingFactor = try take(.routedScalingFactor, routedScalingFactor)
        kvLoraRank = try take(.kvLoraRank, kvLoraRank)
        qLoraRank = try take(.qLoraRank, qLoraRank)
        qkRopeHeadDim = try take(.qkRopeHeadDim, qkRopeHeadDim)
        vHeadDim = try take(.vHeadDim, vHeadDim)
        qkNopeHeadDim = try take(.qkNopeHeadDim, qkNopeHeadDim)
        topkMethod = try take(.topkMethod, topkMethod)
        scoringFunction = try take(.scoringFunction, scoringFunction)
        normTopkProb = try take(.normTopkProb, normTopkProb)
        groups = try take(.groups, groups)
        topkGroups = try take(.topkGroups, topkGroups)
        expertsPerToken = try take(.expertsPerToken, expertsPerToken)
        moeLayerFrequency = try take(.moeLayerFrequency, moeLayerFrequency)
        firstKDenseReplace = try take(.firstKDenseReplace, firstKDenseReplace)
        maxPositionEmbeddings = try take(.maxPositionEmbeddings, maxPositionEmbeddings)
        rmsNormEps = try take(.rmsNormEps, rmsNormEps)
        attentionBias = try take(.attentionBias, attentionBias)

        // transformers 5.x nests rope under `rope_parameters`, which also carries theta.
        let nested = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)
        let flat = try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling)
        ropeScaling = nested ?? flat
        ropeTheta =
            try c.decodeIfPresent(Float.self, forKey: .ropeTheta)
            ?? ropeScaling?["rope_theta"]?.asFloat() ?? ropeTheta
    }

    /// Whether a layer routes to experts rather than a dense MLP.
    public func isMoELayer(_ index: Int) -> Bool {
        routedExperts != nil && index >= firstKDenseReplace
            && index % moeLayerFrequency == 0
    }
}

// MARK: - Sparse indexer

/// Picks the `index_topk` cache positions each query attends to.
///
/// A cheap single-head-scored pass over the whole cache, on its own tiny K cache. Returns
/// nil while the cache still fits inside the budget, which is when every position is
/// attended anyway and the selection would be wasted work.
class DeepseekV32Indexer: Module {
    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo var wk: Linear
    @ModuleInfo(key: "k_norm") var kNorm: LayerNorm
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear

    let heads: Int
    let headDim: Int
    let indexTopK: Int
    let softmaxScale: Float
    let rope: RoPELayer

    init(_ config: DeepseekV32Configuration) {
        heads = config.indexHeads
        headDim = config.indexHeadDim
        indexTopK = config.indexTopK
        softmaxScale = pow(Float(config.indexHeadDim), -0.5)

        _wqB.wrappedValue = Linear(config.qLoraRank, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(config.hiddenSize, headDim, bias: false)
        _kNorm.wrappedValue = LayerNorm(dimensions: headDim)
        _weightsProj.wrappedValue = Linear(config.hiddenSize, heads, bias: false)
        rope = initializeRope(
            dims: config.qkRopeHeadDim,
            base: config.ropeTheta,
            traditional: true,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, queryRank: MLXArray, mask: MLXArray?, cache: KVCache?
    ) -> MLXArray? {
        let batch = x.dim(0)
        let length = x.dim(1)

        var q = wqB(queryRank).reshaped(batch, length, heads, headDim).swappedAxes(1, 2)
        var k = kNorm(wk(x)).reshaped(batch, 1, length, headDim)

        let offset = cache?.offset ?? 0
        q = rope(q, offset: offset)
        k = rope(k, offset: offset)

        if let cache {
            (k, _) = cache.update(keys: k, values: MLXArray.zeros([batch, 1, length, 0]))
        }
        // Everything fits in the budget, so there is nothing to select.
        if k.dim(2) <= indexTopK { return nil }

        var scores = maximum(q.matmul(k.swappedAxes(-1, -2)), 0)
        let weights = weightsProj(x) * (pow(Float(heads), -0.5) * softmaxScale)
        scores = scores * weights.swappedAxes(-1, -2)[.ellipsis, .newAxis]
        scores = scores.sum(axis: 1, keepDims: true)
        if let mask {
            scores = MLX.where(mask, scores, MLXArray(-Float.infinity, dtype: scores.dtype))
        }
        return argPartition(scores, kth: scores.dim(-1) - indexTopK, axis: -1)[
            .ellipsis, (-indexTopK)...]
    }
}

// MARK: - Attention

/// Multi-head latent attention with decoupled RoPE and a sparse indexer.
///
/// The positional half is scored separately and handed to the attention as its mask, which
/// is what lets the content half run against the compressed latent. Two caches per layer:
/// the latent and its positional key, plus the indexer's own.
class DeepseekV32Attention: Module {
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProjWithMQA: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: Module
    @ModuleInfo(key: "unembed_out") var unembedOut: Module
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo var indexer: DeepseekV32Indexer?

    let heads: Int
    let qkNopeHeadDim: Int
    let qkRopeHeadDim: Int
    let kvLoraRank: Int
    let vHeadDim: Int
    let qHeadDim: Int
    var scale: Float
    let rope: RoPELayer

    init(_ config: DeepseekV32Configuration, forceFullIndexer: Bool = true) {
        heads = config.attentionHeads
        qkNopeHeadDim = config.qkNopeHeadDim
        qkRopeHeadDim = config.qkRopeHeadDim
        kvLoraRank = config.kvLoraRank
        vHeadDim = config.vHeadDim
        qHeadDim = config.qkNopeHeadDim + config.qkRopeHeadDim
        scale = pow(Float(qHeadDim), -0.5)

        _qAProj.wrappedValue = Linear(
            config.hiddenSize, config.qLoraRank, bias: config.attentionBias)
        _qALayerNorm.wrappedValue = RMSNorm(dimensions: config.qLoraRank, eps: 1e-6)
        _qBProj.wrappedValue = Linear(config.qLoraRank, heads * qHeadDim, bias: false)
        _kvAProjWithMQA.wrappedValue = Linear(
            config.hiddenSize, kvLoraRank + qkRopeHeadDim, bias: config.attentionBias)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: kvLoraRank, eps: 1e-6)
        _embedQ.wrappedValue = MultiLinear(
            inputDimensions: qkNopeHeadDim, outputDimensions: kvLoraRank, numHeads: heads)
        _unembedOut.wrappedValue = MultiLinear(
            inputDimensions: kvLoraRank, outputDimensions: vHeadDim, numHeads: heads)
        _oProj.wrappedValue = Linear(
            heads * vHeadDim, config.hiddenSize, bias: config.attentionBias)

        // A long-rope checkpoint folds its own attention scaling into the softmax scale.
        if let ropeScaling = config.ropeScaling,
            let mscaleAllDim = ropeScaling["mscale_all_dim"]?.asFloat(), mscaleAllDim != 0,
            let factor = ropeScaling["factor"]?.asFloat(), factor > 1
        {
            let s = 0.1 * mscaleAllDim * log(factor) + 1.0
            scale = scale * s * s
        }

        if forceFullIndexer {
            _indexer.wrappedValue = DeepseekV32Indexer(config)
        }
        rope = initializeRope(
            dims: qkRopeHeadDim,
            base: config.ropeTheta,
            traditional: true,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
        super.init()
    }

    private func multiLinear(
        _ module: Module, _ x: MLXArray, transpose: Bool = true
    ) -> MLXArray {
        if let linear = module as? MultiLinear {
            return linear(x, transpose: transpose)
        }
        fatalError("embed_q and unembed_out must be MultiLinear or QuantizedMultiLinear")
    }

    /// `previousTopKIndices` carries the selection forward for a layer that shares another's
    /// indexer rather than running its own.
    func callAsFunction(
        _ x: MLXArray,
        mask: MLXArray?,
        cache: CacheList?,
        previousTopKIndices: MLXArray? = nil
    ) -> (output: MLXArray, topKIndices: MLXArray?) {
        let batch = x.dim(0)
        let length = x.dim(1)

        let queryRank = qALayerNorm(qAProj(x))
        let q = qBProj(queryRank).reshaped(batch, length, heads, qHeadDim)
            .transposed(0, 2, 1, 3)
        let splitQ = split(q, indices: [qkNopeHeadDim], axis: -1)
        var qNope = splitQ[0]
        var qPE = splitQ[1]

        let splitKV = split(kvAProjWithMQA(x), indices: [kvLoraRank], axis: -1)
        var kPE = splitKV[1].reshaped(batch, length, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)
        var kvLatent = expandedDimensions(kvALayerNorm(splitKV[0]), axis: 1)

        let offset = cache?[0].offset ?? 0
        qPE = rope(qPE, offset: offset)
        kPE = rope(kPE, offset: offset)

        if let cache {
            (kvLatent, kPE) = cache[0].update(keys: kvLatent, values: kPE)
        }

        var effectiveMask = mask
        let topKIndices =
            indexer.flatMap {
                $0(x, queryRank: queryRank, mask: mask, cache: cache?[1])
            } ?? previousTopKIndices

        if let topKIndices {
            if length == 1 {
                // One query: gather only the selected positions rather than masking.
                let index = topKIndices[0..., 0..., 0, 0..., .newAxis]
                kvLatent = takeAlong(
                    kvLatent,
                    broadcast(
                        index, to: Array(index.shape.dropLast()) + [kvLatent.dim(-1)]),
                    axis: 2)
                kPE = takeAlong(
                    kPE,
                    broadcast(index, to: Array(index.shape.dropLast()) + [kPE.dim(-1)]),
                    axis: 2)
                effectiveMask = mask.map { takeAlong($0, topKIndices, axis: -1) }
            } else {
                var shape = topKIndices.shape
                shape[shape.count - 1] = kvLatent.dim(2)
                var sparse = MLXArray.zeros(shape, dtype: .bool)
                sparse = putAlong(sparse, topKIndices, values: MLXArray(true), axis: -1)
                effectiveMask = mask.map { sparse .&& $0 } ?? sparse
            }
        }

        // The latent cache must not be read before the indexer's own writes land; without
        // the dependency the two are unordered and the selection can be read stale.
        // A shared-indexer layer keeps only the latent cache, so there is nothing to order
        // against: it reuses a selection another layer already made.
        if let cache, cache.children.count > 1,
            let latent = (cache[0] as? MaterializedKVCache)?.storage as? KVCacheSimple,
            let index = (cache[1] as? MaterializedKVCache)?.storage as? KVCacheSimple
        {
            latent.orderKeys(after: index)
        }

        var peScores = (qPE * scale).matmul(kPE.swappedAxes(-1, -2))
        if let effectiveMask {
            peScores = MLX.where(
                effectiveMask, peScores,
                MLXArray(-Float.greatestFiniteMagnitude, dtype: peScores.dtype))
        }

        // Below the crossover, folding into the query beats materializing K and V.
        let absorbed =
            length == 1
            || (allowsAbsorbedPrefill
                && length
                    <= maxAbsorbedQueries(
                        kvLoraRank: kvLoraRank, qkNopeHeadDim: qkNopeHeadDim,
                        vHeadDim: vHeadDim, cacheLength: latentLength(kvLatent)))

        let keys: MLXArray
        let values: MLXArray
        if absorbed {
            qNope = multiLinear(embedQ, qNope)
            keys = kvLatent
            values = kvLatent
        } else {
            keys = multiLinear(embedQ, kvLatent, transpose: false)
            values = multiLinear(unembedOut, kvLatent)
        }

        var output = MLXFast.scaledDotProductAttention(
            queries: qNope, keys: keys, values: values, scale: scale,
            mask: .array(peScores), sinks: attentionSinks(dtype: qNope.dtype))
        if absorbed {
            output = multiLinear(unembedOut, output)
        }
        output = gateAttentionOutput(output, input: x)
        output = output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        return (oProj(output), topKIndices)
    }

    /// A per-head logit for an always-attended slot, or nil for a stack without one.
    ///
    /// Overridable because hy_v4 shares this whole sparse path and adds a learned sink;
    /// duplicating the path to add one argument would leave two copies of the selection,
    /// the cache ordering and the absorbed/materialized crossover to keep in step.
    var allowsAbsorbedPrefill: Bool { true }

    func attentionSinks(dtype _: DType) -> MLXArray? { nil }

    /// A gate on the attention output, before the heads are folded back together. The
    /// identity here; hy_v4 draws a per-head sigmoid from the block input.
    func gateAttentionOutput(_ output: MLXArray, input _: MLXArray) -> MLXArray { output }
}

// MARK: - Mixture of experts

/// `noaux_tc` routing: sigmoid gates, a bias used only for selection, and optional
/// group pruning before the top-k.
func deepseekV32GroupExpertSelect(
    gates: MLXArray,
    correctionBias: MLXArray,
    topK: Int,
    groups: Int,
    topkGroups: Int,
    routedScalingFactor: Float,
    normTopkProb: Bool
) -> (indices: MLXArray, scores: MLXArray) {
    let original = sigmoid(gates.asType(.float32))
    var scores = original + correctionBias

    if groups > 1 {
        scores = unflatten(scores, axis: -1, shape: [groups, -1])
        let groupScores = MLX.top(scores, k: 2, axis: -1).sum(axis: -1, keepDims: true)
        let pruned = groups - topkGroups
        let groupIndices = argPartition(groupScores, kth: pruned - 1, axis: -2)[
            .ellipsis, ..<pruned, 0...]
        scores = putAlong(
            scores, stopGradient(groupIndices), values: MLXArray(Float(0)), axis: -2)
        scores = flattened(scores, start: -2, end: -1)
    }

    let indices = argPartition(-scores, kth: topK - 1, axis: -1)[.ellipsis, ..<topK]
    var selected = takeAlong(original, indices, axis: -1)
    if topK > 1, normTopkProb {
        selected = selected / selected.sum(axis: -1, keepDims: true)
    }
    return (indices, selected * routedScalingFactor)
}

class DeepseekV32MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ config: DeepseekV32Configuration, intermediateSize: Int? = nil) {
        let hidden = intermediateSize ?? config.intermediateSize
        _gateProj.wrappedValue = Linear(config.hiddenSize, hidden, bias: false)
        _upProj.wrappedValue = Linear(config.hiddenSize, hidden, bias: false)
        _downProj.wrappedValue = Linear(hidden, config.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

class DeepseekV32MoEGate: Module {
    @ModuleInfo var weight: MLXArray
    @ModuleInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    let topK: Int
    let groups: Int
    let topkGroups: Int
    let routedScalingFactor: Float
    let normTopkProb: Bool

    init(_ config: DeepseekV32Configuration) {
        topK = config.expertsPerToken
        groups = config.groups
        topkGroups = config.topkGroups
        routedScalingFactor = config.routedScalingFactor
        normTopkProb = config.normTopkProb
        let experts = config.routedExperts ?? 0
        _weight.wrappedValue = MLXArray.zeros([experts, config.hiddenSize])
        _correctionBias.wrappedValue = MLXArray.zeros([experts])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        deepseekV32GroupExpertSelect(
            gates: x.matmul(weight.transposed()),
            correctionBias: correctionBias,
            topK: topK,
            groups: groups,
            topkGroups: topkGroups,
            routedScalingFactor: routedScalingFactor,
            normTopkProb: normTopkProb)
    }
}

class DeepseekV32MoE: Module, UnaryLayer {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo var gate: DeepseekV32MoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: DeepseekV32MLP?

    init(_ config: DeepseekV32Configuration) {
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize,
            hiddenDims: config.moeIntermediateSize,
            numExperts: config.routedExperts ?? 0)
        _gate.wrappedValue = DeepseekV32MoEGate(config)
        if let shared = config.sharedExperts {
            _sharedExperts.wrappedValue = DeepseekV32MLP(
                config, intermediateSize: config.moeIntermediateSize * shared)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = switchMLP(x, indices)
        y = (y * scores[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

// MARK: - Stack

class DeepseekV32DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: DeepseekV32Attention
    @ModuleInfo var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(
        _ config: DeepseekV32Configuration, layerIndex: Int,
        makeAttention: (DeepseekV32Configuration, Int) -> DeepseekV32Attention = { config, _ in
            DeepseekV32Attention(config)
        }
    ) {
        _attention.wrappedValue = makeAttention(config, layerIndex)
        _mlp.wrappedValue =
            config.isMoELayer(layerIndex) ? DeepseekV32MoE(config) : DeepseekV32MLP(config)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: CacheList?,
        previousTopKIndices: MLXArray? = nil
    ) -> (hidden: MLXArray, topKIndices: MLXArray?) {
        let (attended, topKIndices) = attention(
            inputLayerNorm(x), mask: mask, cache: cache,
            previousTopKIndices: previousTopKIndices)
        let h = x + attended
        return (h + mlp(postAttentionLayerNorm(h)), topKIndices)
    }
}

class DeepseekV32Inner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [DeepseekV32DecoderLayer]
    @ModuleInfo var norm: RMSNorm

    let config: DeepseekV32Configuration

    init(
        _ config: DeepseekV32Configuration,
        makeLayer: (DeepseekV32Configuration, Int) -> DeepseekV32DecoderLayer = {
            DeepseekV32DecoderLayer($0, layerIndex: $1)
        }
    ) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map { makeLayer(config, $0) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        var h = embedTokens(inputs)
        let first = (cache?.first as? CacheList)?[0]
        // The array form: the positional scores are combined with this mask arithmetically
        // before attention, so a mask mode would not compose.
        let mask: MLXArray? = createAttentionMask(h: h, cache: first.map { [$0] })

        // A layer with no indexer of its own reuses the last selection made.
        var topKIndices: MLXArray?
        for (index, layer) in layers.enumerated() {
            let (hidden, selected) = layer(
                h, mask: mask, cache: cache?[index] as? CacheList,
                previousTopKIndices: topKIndices)
            h = hidden
            topKIndices = selected
        }
        return norm(h)
    }
}

/// DeepSeek V3.2's language model.
public final class DeepseekV32Model: Module, LLMModel, KVCacheDimensionProvider {
    @ModuleInfo var model: DeepseekV32Inner
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public let config: DeepseekV32Configuration
    public var vocabularySize: Int { config.vocabularySize }
    /// One entry per layer; the pair inside each is built by ``newCache(configuration:)``.
    public var kvHeads: [Int] { Array(repeating: 1, count: config.hiddenLayers) }

    public init(_ config: DeepseekV32Configuration) {
        self.config = config
        _model.wrappedValue = DeepseekV32Inner(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    public var loraLayers: [Module] {
        model.layers
    }

    /// Two caches per layer: the compressed latent with its positional key, and the
    /// indexer's own keys.
    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        (0 ..< config.hiddenLayers).map { _ in
            CacheList(MaterializedKVCache(KVCacheSimple()), MaterializedKVCache(KVCacheSimple()))
        }
    }

    /// Normalizes a published checkpoint onto this layout.
    ///
    /// Three transforms, in order: block-scaled fp8 weights are dequantized, per-expert
    /// tensors are stacked into the switched MLP, and the fused `kv_b_proj` is split into
    /// the absorbed `embed_q` and `unembed_out` -- requantized when the source was
    /// quantized, since the split cannot be done in packed form.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            // Some exports carry the MTP block alongside; it belongs to the drafter.
            let parts = key.split(separator: ".")
            if parts.count >= 3, parts[1] == "layers", let layer = Int(parts[2]),
                layer >= config.hiddenLayers
            {
                continue
            }
            sanitized[key] = value
        }

        var dequantized: [String: MLXArray] = [:]
        for (key, value) in sanitized {
            guard key.contains("weight_scale_inv") else {
                if dequantized[key] == nil { dequantized[key] = value }
                continue
            }
            let weightKey = key.replacingOccurrences(of: "_scale_inv", with: "")
            guard let weight = sanitized[weightKey] else {
                dequantized[key] = value
                continue
            }
            dequantized[weightKey] = Self.dequantizeBlockScaled(weight, scaleInverse: value)
        }
        sanitized = dequantized

        stackExperts(weights: &sanitized)
        splitFusedKV(weights: &sanitized)
        return sanitized
    }

    /// Undoes DeepSeek's 128x128 block-scaled fp8 storage.
    static func dequantizeBlockScaled(
        _ weight: MLXArray, scaleInverse: MLXArray
    ) -> MLXArray {
        let block = 128
        var value = fromFP8(weight.asType(.uint8), dtype: .bfloat16)
        let rows = value.dim(0)
        let columns = value.dim(1)
        let padBottom = (block - rows % block) % block
        let padSide = (block - columns % block) % block
        if padBottom > 0 || padSide > 0 {
            value = padded(value, widths: [.init((0, padBottom)), .init((0, padSide))])
        }
        let paddedRows = rows + padBottom
        let paddedColumns = columns + padSide
        value = value.reshaped(paddedRows / block, block, paddedColumns / block, block)
        value =
            (value * scaleInverse[0..., .newAxis, 0..., .newAxis])
            .reshaped(paddedRows, paddedColumns)
        return value[..<rows, ..<columns].asType(.bfloat16)
    }

    func stackExperts(weights: inout [String: MLXArray]) {
        Self.stackExperts(weights: &weights, config: config)
    }

    static func stackExperts(
        weights: inout [String: MLXArray], config: DeepseekV32Configuration
    ) {
        guard let experts = config.routedExperts else { return }
        for layer in 0 ..< config.hiddenLayers {
            let prefix = "model.layers.\(layer).mlp"
            for projection in ["gate_proj", "down_proj", "up_proj"] {
                for suffix in ["weight", "scales", "biases"] {
                    guard weights["\(prefix).experts.0.\(projection).\(suffix)"] != nil
                    else { continue }
                    let tensors = (0 ..< experts).map { expert in
                        weights.removeValue(
                            forKey: "\(prefix).experts.\(expert).\(projection).\(suffix)")!
                    }
                    weights["\(prefix).switch_mlp.\(projection).\(suffix)"] = MLX.stacked(
                        tensors)
                }
            }
        }
    }

    func splitFusedKV(weights: inout [String: MLXArray]) {
        Self.splitFusedKV(weights: &weights, config: config)
    }

    static func splitFusedKV(
        weights: inout [String: MLXArray], config: DeepseekV32Configuration
    ) {
        let headDim = config.qkNopeHeadDim + config.vHeadDim
        for layer in 0 ..< config.hiddenLayers {
            let prefix = "model.layers.\(layer).self_attn"
            guard var fused = weights.removeValue(forKey: "\(prefix).kv_b_proj.weight")
            else { continue }

            var bits: Int?
            var groupSize: Int?
            if let scales = weights.removeValue(forKey: "\(prefix).kv_b_proj.scales") {
                let biases = weights.removeValue(forKey: "\(prefix).kv_b_proj.biases")
                bits = (fused.dim(-1) * 32) / config.kvLoraRank
                groupSize = config.kvLoraRank / scales.dim(-1)
                fused = dequantized(
                    fused, scales: scales, biases: biases,
                    groupSize: groupSize!, bits: bits!)
            }

            let reshaped = fused.reshaped(config.attentionHeads, headDim, -1)
            var embedQ = contiguous(
                reshaped[0..., ..<config.qkNopeHeadDim, 0...].swappedAxes(-1, -2))
            var unembedOut = contiguous(reshaped[0..., config.qkNopeHeadDim..., 0...])

            if let bits, let groupSize {
                let (qWeight, qScales, qBiases) = MLX.quantized(
                    embedQ, groupSize: groupSize, bits: bits)
                let (vWeight, vScales, vBiases) = MLX.quantized(
                    unembedOut, groupSize: groupSize, bits: bits)
                weights["\(prefix).embed_q.scales"] = qScales
                weights["\(prefix).embed_q.biases"] = qBiases
                weights["\(prefix).unembed_out.scales"] = vScales
                weights["\(prefix).unembed_out.biases"] = vBiases
                embedQ = qWeight
                unembedOut = vWeight
            }
            weights["\(prefix).embed_q.weight"] = embedQ
            weights["\(prefix).unembed_out.weight"] = unembedOut
        }
    }
}
