import Foundation
import MLX
import MLXNN

/// Chooses which cached positions each query attends to.
class LongcatSparseIndexer: Module {

    let heads: Int
    let headDim: Int
    let topK: Int
    let initialTokens: Int
    let localTokens: Int
    let softmaxScale: Float

    @ModuleInfo(key: "wq_b") var wqB: Linear
    @ModuleInfo var wk: Linear
    @ModuleInfo(key: "k_norm") var kNorm: UnaryLayer
    @ModuleInfo(key: "weights_proj") var weightsProj: Linear

    let rope: RoPELayer

    init(_ args: LongcatFlashSparseConfiguration) {
        self.heads = args.indexHeads
        self.headDim = args.indexHeadDim
        self.topK = args.indexTopK
        self.initialTokens = args.indexInitialTokens
        self.localTokens = args.indexLocalTokens
        self.softmaxScale = pow(Float(args.indexHeadDim), -0.5)

        _wqB.wrappedValue = Linear(
            args.base.qLoraRank ?? args.base.hiddenSize, heads * headDim, bias: false)
        _wk.wrappedValue = Linear(args.base.hiddenSize, headDim, bias: false)
        _kNorm.wrappedValue =
            args.indexKeyNormType == "rms"
            ? RMSNorm(dimensions: headDim, eps: args.base.rmsNormEps)
            : LayerNorm(dimensions: headDim)
        _weightsProj.wrappedValue = Linear(args.base.hiddenSize, heads, bias: false)

        self.rope = initializeRope(
            dims: args.base.qkRopeHeadDim, base: args.base.ropeTheta,
            traditional: args.indexerRopeInterleave,
            scalingConfig: args.base.ropeScaling,
            maxPositionEmbeddings: args.base.maxPositionEmbeddings)
        super.init()
    }

    /// The selected positions, or `nil` when everything already fits in the budget.
    func callAsFunction(
        _ x: MLXArray, queryRank: MLXArray, mask: MLXArray?, cache: KVCache?
    ) -> MLXArray? {
        let (B, L) = (x.dim(0), x.dim(1))

        var q = wqB(queryRank).reshaped(B, L, heads, headDim).swappedAxes(1, 2)
        var k = kNorm(wk(x)).reshaped(B, 1, L, headDim)

        let offset = cache?.offset ?? 0
        q = rope(q, offset: offset)
        k = rope(k, offset: offset)
        if let cache {
            (k, _) = cache.update(keys: k, values: MLXArray.zeros([B, 1, L, 0]))
        }

        let cached = k.dim(2)
        if cached <= topK { return nil }

        var scores = maximum(matmul(q, k.swappedAxes(-1, -2)), 0)
        let weights = weightsProj(x) * (pow(Float(heads), -0.5) * softmaxScale)
        scores = (scores * weights.swappedAxes(-1, -2)[.ellipsis, .newAxis])
            .sum(axis: 1, keepDims: true)

        if initialTokens > 0 || localTokens > 0 {
            let columns = MLXArray(0 ..< cached)[.newAxis]
            var forced = columns .< initialTokens
            if localTokens > 0 {
                let rows = (MLXArray(0 ..< L) + offset)[0..., .newAxis]
                forced =
                    forced
                    .|| ((columns .<= rows) .&& (columns .> rows - localTokens))
            } else {
                forced = broadcast(forced, to: [L, cached])
            }
            scores = MLX.where(
                forced[.newAxis, .newAxis], MLXArray(Float.infinity, dtype: scores.dtype),
                scores)
        }

        if let mask {
            scores = MLX.where(
                mask, scores, MLXArray(-Float.infinity, dtype: scores.dtype))
        }

        return argPartition(scores, kth: scores.dim(-1) - topK, axis: -1)[
            .ellipsis, (-topK)...]
    }
}

class LongcatSparseMLA: Module {

    let heads: Int
    let qkRopeHeadDim: Int
    let qkNopeHeadDim: Int
    let kvLoraRank: Int
    let qLoraRank: Int?
    let vHeadDim: Int
    let qkHeadDim: Int
    let scale: Float
    let usesSparseAttention: Bool

    let queryLoraScale: Float?
    let keyValueLoraScale: Float?

    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?

    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProj: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm

    @ModuleInfo(key: "embed_q") var embedQ: Module
    @ModuleInfo(key: "unembed_out") var unembedOut: Module

    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "indexer") var indexer: LongcatSparseIndexer?

    let rope: RoPELayer

    init(_ args: LongcatFlashSparseConfiguration, ownsIndexer: Bool) {
        let base = args.base
        self.heads = base.attentionHeads
        self.qkRopeHeadDim = base.qkRopeHeadDim
        self.qkNopeHeadDim = base.qkNopeHeadDim
        self.kvLoraRank = base.kvLoraRank
        self.qLoraRank = base.qLoraRank
        self.vHeadDim = base.vHeadDim
        self.qkHeadDim = base.qkNopeHeadDim + base.qkRopeHeadDim
        self.usesSparseAttention = args.usesSparseAttention

        var scale = pow(Float(qkHeadDim), -0.5)
        if let scaling = base.ropeScaling,
            let mscaleAllDim = scaling["mscale_all_dim"]?.asFloat(), mscaleAllDim != 0,
            let factor = scaling["factor"]?.asFloat(), factor > 1
        {
            let adjustment = 0.1 * mscaleAllDim * log(factor) + 1
            scale = scale * adjustment * adjustment
        }
        self.scale = scale

        if let qLoraRank = base.qLoraRank {
            _qAProj.wrappedValue = Linear(
                base.hiddenSize, qLoraRank, bias: base.attentionBias)
            _qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank)
            _qBProj.wrappedValue = Linear(qLoraRank, heads * qkHeadDim, bias: false)
        } else {
            _qProj.wrappedValue = Linear(base.hiddenSize, heads * qkHeadDim, bias: false)
        }

        _kvAProj.wrappedValue = Linear(
            base.hiddenSize, base.kvLoraRank + base.qkRopeHeadDim,
            bias: base.attentionBias)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: base.kvLoraRank)

        _embedQ.wrappedValue = MultiLinear(
            inputDimensions: base.qkNopeHeadDim, outputDimensions: base.kvLoraRank,
            numHeads: heads)
        _unembedOut.wrappedValue = MultiLinear(
            inputDimensions: base.kvLoraRank, outputDimensions: base.vHeadDim,
            numHeads: heads)

        _oProj.wrappedValue = Linear(
            heads * base.vHeadDim, base.hiddenSize, bias: base.attentionBias)

        self.queryLoraScale =
            base.scaleQueryLora && base.qLoraRank != nil
            ? sqrt(Float(base.hiddenSize) / Float(base.qLoraRank!)) : nil
        self.keyValueLoraScale =
            base.scaleKeyValueLora
            ? sqrt(Float(base.hiddenSize) / Float(base.kvLoraRank)) : nil

        if args.usesSparseAttention && ownsIndexer {
            _indexer.wrappedValue = LongcatSparseIndexer(args)
        }

        self.rope = initializeRope(
            dims: base.qkRopeHeadDim, base: base.ropeTheta, traditional: true,
            scalingConfig: base.ropeScaling,
            maxPositionEmbeddings: base.maxPositionEmbeddings)
        super.init()
    }

    private func multiLinear(_ module: Module, _ x: MLXArray, transpose: Bool = true)
        -> MLXArray
    {
        if let linear = module as? MultiLinear {
            return linear(x, transpose: transpose)
        }
        if let quantized = module as? QuantizedMultiLinear {
            return quantized(x, transpose: transpose)
        }
        fatalError("embed_q and unembed_out must be MultiLinear or QuantizedMultiLinear")
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, latentCache: KVCache?, indexCache: KVCache?,
        selectedIndices: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        let (B, L) = (x.dim(0), x.dim(1))

        var queryRank: MLXArray? = nil
        var q: MLXArray
        if let qBProj, let qALayerNorm, let qAProj {
            let rank = qALayerNorm(qAProj(x))
            queryRank = rank
            q = qBProj(rank)
        } else {
            q = qProj!(x)
        }
        q = q.reshaped(B, L, heads, qkHeadDim).transposed(0, 2, 1, 3)
        if let queryLoraScale {
            q = q * queryLoraScale
        }
        let qSplit = split(q, indices: [qkNopeHeadDim], axis: -1)
        var qNope = qSplit[0]
        var qPE = qSplit[1]

        let compressed = kvAProj(x)
        let kvSplit = split(compressed, indices: [kvLoraRank], axis: -1)
        var kPE = kvSplit[1].reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)
        var kvLatent = kvALayerNorm(kvSplit[0])
        if let keyValueLoraScale {
            kvLatent = kvLatent * keyValueLoraScale
        }

        let offset = latentCache?.offset ?? 0
        qPE = rope(qPE, offset: offset)
        kPE = rope(kPE, offset: offset)

        kvLatent = kvLatent.expandedDimensions(axis: 1)
        if let latentCache {
            (kvLatent, kPE) = latentCache.update(keys: kvLatent, values: kPE)
        }

        var indices = selectedIndices
        if let indexer, let queryRank {
            indices = indexer(x, queryRank: queryRank, mask: mask, cache: indexCache)
        }

        var effectiveMask = mask
        if usesSparseAttention, let indices {
            if L == 1 {
                let rows = indices[0..., 0..., 0, 0..., .newAxis]
                kvLatent = takeAlong(
                    kvLatent,
                    broadcast(rows, to: rows.shape.dropLast() + [kvLatent.dim(-1)]),
                    axis: 2)
                kPE = takeAlong(
                    kPE, broadcast(rows, to: rows.shape.dropLast() + [kPE.dim(-1)]),
                    axis: 2)
                if let mask {
                    effectiveMask = takeAlong(mask, indices, axis: -1)
                }
            } else {
                var shape = indices.shape
                shape[shape.count - 1] = kvLatent.dim(2)
                var sparse = MLXArray.zeros(shape, dtype: .bool)
                sparse = putAlong(sparse, indices, values: MLXArray(true), axis: -1)
                effectiveMask = mask.map { sparse .&& $0 } ?? sparse
            }

            if let latent = (latentCache as? MaterializedKVCache)?.storage as? KVCacheSimple,
                let index = (indexCache as? MaterializedKVCache)?.storage as? KVCacheSimple
            {
                latent.orderKeys(after: index)
            }
        }

        var peScores = (qPE * scale).matmul(kPE.swappedAxes(-1, -2))
        if let effectiveMask {
            peScores = MLX.where(
                effectiveMask, peScores,
                MLXArray(-Float.greatestFiniteMagnitude, dtype: peScores.dtype))
        }

        let absorbed =
            L == 1
            || L
                <= maxAbsorbedQueries(
                    kvLoraRank: kvLoraRank, qkNopeHeadDim: qkNopeHeadDim,
                    vHeadDim: vHeadDim, cacheLength: latentLength(kvLatent))

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
            mask: .array(peScores))
        if absorbed {
            output = multiLinear(unembedOut, output)
        }

        return (oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1)), indices)
    }
}

class LongcatSparseBlock: Module {

    @ModuleInfo(key: "mlp") var mixture: LongcatFlashMoE
    @ModuleInfo(key: "self_attn") var attentions: [LongcatSparseMLA]
    @ModuleInfo(key: "mlps") var feedForwards: [SwiGLUMLP]
    @ModuleInfo(key: "input_layernorm") var inputLayerNorms: [RMSNorm]
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorms: [RMSNorm]

    let usesSparseAttention: Bool

    init(_ args: LongcatFlashSparseConfiguration) {
        self.usesSparseAttention = args.usesSparseAttention
        _mixture.wrappedValue = LongcatFlashMoE(args.base)
        _attentions.wrappedValue = (0 ..< 2).map {
            LongcatSparseMLA(args, ownsIndexer: $0 == 0)
        }
        _feedForwards.wrappedValue = (0 ..< 2).map { _ in
            SwiGLUMLP(
                inputDimensions: args.base.hiddenSize,
                hiddenDimensions: args.base.feedForwardSize)
        }
        _inputLayerNorms.wrappedValue = (0 ..< 2).map { _ in
            RMSNorm(dimensions: args.base.hiddenSize, eps: args.base.rmsNormEps)
        }
        _postAttentionLayerNorms.wrappedValue = (0 ..< 2).map { _ in
            RMSNorm(dimensions: args.base.hiddenSize, eps: args.base.rmsNormEps)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: CacheList?) -> MLXArray {
        var h = x
        var shortcut: MLXArray? = nil
        var indices: MLXArray? = nil

        for i in 0 ..< 2 {
            var residual = h
            let normalized = inputLayerNorms[i](h)

            let latentCache: KVCache?
            let indexCache: KVCache?
            if usesSparseAttention {
                latentCache = i == 0 ? cache?[0] : cache?[2]
                indexCache = i == 0 ? cache?[1] : nil
            } else {
                latentCache = cache?[i]
                indexCache = nil
            }

            let (attended, produced) = attentions[i](
                normalized, mask: mask, latentCache: latentCache, indexCache: indexCache,
                selectedIndices: i == 0 ? nil : indices)
            if i == 0 {
                indices = produced
            }
            h = residual + attended

            residual = h
            let post = postAttentionLayerNorms[i](h)
            if i == 0 {
                shortcut = mixture(post)
            }
            h = residual + feedForwards[i](post)
            if i == 1, let shortcut {
                h = h + shortcut
            }
        }

        return h
    }
}

public class LongcatFlashSparseModelInner: Module {

    @ModuleInfo(key: "ngram_embeddings") var ngramEmbeddings: LongcatNgramEmbedding?
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding?

    let layers: [LongcatSparseBlock]
    let norm: RMSNorm

    let usesNgram: Bool

    public init(_ args: LongcatFlashSparseConfiguration) {
        precondition(args.base.vocabularySize > 0)
        self.usesNgram = args.usesNgramEmbedding

        if usesNgram {
            _ngramEmbeddings.wrappedValue = LongcatNgramEmbedding(
                vocabularySize: args.base.vocabularySize,
                hiddenSize: args.base.hiddenSize,
                vocabularyRatio: args.ngramVocabularyRatio,
                splits: args.embeddingSplits, neighbours: args.embeddingNeighbours)
        } else {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: args.base.vocabularySize,
                dimensions: args.base.hiddenSize)
        }

        self.layers = (0 ..< args.base.layerCount).map { _ in LongcatSparseBlock(args) }
        self.norm = RMSNorm(dimensions: args.base.hiddenSize, eps: args.base.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let layerOffset = usesNgram ? 1 : 0
        var h: MLXArray
        if let ngramEmbeddings {
            h = ngramEmbeddings(inputs, cache: cache?.first as? ArraysCache)
        } else {
            h = embedTokens!(inputs)
        }

        let first =
            (cache?.count ?? 0) > layerOffset
            ? (cache?[layerOffset] as? CacheList)?[0] : nil
        let mask: MLXArray? = createAttentionMask(h: h, cache: first.map { [$0] })

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i + layerOffset] as? CacheList)
        }

        return norm(h)
    }
}

public class LongcatFlashSparseModel: Module, LLMModel {

    public let vocabularySize: Int

    public let model: LongcatFlashSparseModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let configuration: LongcatFlashSparseConfiguration

    public init(_ args: LongcatFlashSparseConfiguration) {
        self.configuration = args
        self.vocabularySize = args.base.vocabularySize
        self.model = LongcatFlashSparseModelInner(args)
        _lmHead.wrappedValue = Linear(
            args.base.hiddenSize, args.base.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    /// Three caches per layer under the indexer -- latent, index, latent -- and two without,
    /// preceded by one array slot when the n-gram embedding is on.
    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        let perLayer: [KVCache] = model.layers.flatMap { _ -> [KVCache] in
            configuration.usesSparseAttention
                ? [
                    CacheList(
                        MaterializedKVCache(KVCacheSimple()), MaterializedKVCache(KVCacheSimple()),
                        MaterializedKVCache(KVCacheSimple()))
                ]
                : [
                    CacheList(
                        MaterializedKVCache(KVCacheSimple()), MaterializedKVCache(KVCacheSimple()))
                ]
        }
        return configuration.usesNgramEmbedding
            ? [ArraysCache(size: 1)] + perLayer : perLayer
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = LongcatFlashModel.sanitize(
            weights: weights, configuration: configuration.base)
        if configuration.usesNgramEmbedding,
            let table = sanitized.removeValue(forKey: "model.embed_tokens.weight")
        {
            sanitized["model.ngram_embeddings.word_embeddings.weight"] = table
        }
        return sanitized
    }
}

public struct LongcatFlashSparseConfiguration: Decodable, Sendable {

    var base: LongcatFlashConfiguration

    var indexHeads: Int
    var indexHeadDim: Int
    var indexTopK: Int
    var indexInitialTokens: Int
    var indexLocalTokens: Int
    var attentionSubBlocksPerIndex: Int
    var indexKeyNormType: String
    var indexerRopeInterleave: Bool
    var ngramVocabularyRatio: Int
    var embeddingNeighbours: Int
    var embeddingSplits: Int

    /// `LSA` selects the sparse indexer; `MLA` leaves attention dense.
    var usesSparseAttention: Bool { base.attentionMethod == "LSA" }

    /// A ratio of zero means no n-gram embedding, and a plain `embed_tokens` instead.
    var usesNgramEmbedding: Bool { ngramVocabularyRatio > 0 }

    enum CodingKeys: String, CodingKey {
        case indexHeads = "index_n_heads"
        case indexHeadDim = "index_head_dim"
        case indexTopK = "index_topk"
        case indexInitialTokens = "index_init_tokens"
        case indexLocalTokens = "index_local_tokens"
        case attentionSubBlocksPerIndex = "cli_factor"
        case indexKeyNormType = "index_k_norm_type"
        case indexerRopeInterleave = "indexer_rope_interleave"
        case ngramVocabularyRatio = "oe_vocab_size_ratio"
        case embeddingNeighbours = "oe_neighbor_num"
        case embeddingSplits = "oe_split_num"
    }

    public init(from decoder: Decoder) throws {
        base = try LongcatFlashConfiguration(from: decoder)
        let c = try decoder.container(keyedBy: CodingKeys.self)
        indexHeads = try c.decodeIfPresent(Int.self, forKey: .indexHeads) ?? 0
        indexHeadDim = try c.decodeIfPresent(Int.self, forKey: .indexHeadDim) ?? 128
        indexTopK = try c.decodeIfPresent(Int.self, forKey: .indexTopK) ?? 0
        indexInitialTokens =
            try c.decodeIfPresent(Int.self, forKey: .indexInitialTokens) ?? 0
        indexLocalTokens = try c.decodeIfPresent(Int.self, forKey: .indexLocalTokens) ?? 0
        attentionSubBlocksPerIndex =
            try c.decodeIfPresent(Int.self, forKey: .attentionSubBlocksPerIndex) ?? 1
        indexKeyNormType =
            try c.decodeIfPresent(String.self, forKey: .indexKeyNormType) ?? "rms"
        indexerRopeInterleave =
            try c.decodeIfPresent(Bool.self, forKey: .indexerRopeInterleave) ?? true
        ngramVocabularyRatio =
            try c.decodeIfPresent(Int.self, forKey: .ngramVocabularyRatio) ?? 0
        embeddingNeighbours =
            try c.decodeIfPresent(Int.self, forKey: .embeddingNeighbours) ?? 4
        embeddingSplits = try c.decodeIfPresent(Int.self, forKey: .embeddingSplits) ?? 4
    }
}

extension LongcatFlashSparseModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
