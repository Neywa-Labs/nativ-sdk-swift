import Foundation
import MLX
import MLXNN

/// Qwen4-Exp's text stack: sparse attention, gated-delta linear attention, a routed MoE,
/// hyper-connections, and per-layer n-gram embeddings on the layers that name them.
public struct Qwen4ExpTextConfiguration: Decodable, Sendable {
    public var modelType: String = "qwen4_exp"
    public var hiddenSize: Int = 2048
    public var hiddenLayers: Int = 24
    public var attentionHeads: Int = 16
    public var kvHeads: Int = 2
    public var headDim: Int?
    public var vocabularySize: Int = 151_936
    public var maxPositionEmbeddings: Int = 262_144
    public var rmsNormEps: Float = 1e-6
    /// How many residual streams a hyper-connection block carries.
    public var hcCount: Int = 4
    public var hcLowRank: Int = 320
    public var layerTypes: [String]?
    public var fullAttentionInterval: Int = 4
    public var pleLayerIDs: [Int] = []
    public var experts: Int = 128
    public var expertsPerToken: Int = 8
    public var moeIntermediateSize: Int = 768
    public var sharedExpertIntermediateSize: Int = 768
    public var normTopkProb: Bool = true
    public var indexerHeads: Int = 4
    public var indexerKVHeads: Int = 1
    public var indexerHeadDim: Int = 128
    public var indexerBudget: Int = 2048
    public var indexerCompressRatio: Int = 4
    public var outputGateType: String = "sigmoid"
    public var attentionBias: Bool = false
    public var tieWordEmbeddings: Bool = false
    public var mtpHiddenLayers: Int = 1
    public var ropeParameters: [String: StringOrNumber]?

    public var linearValueHeads: Int = 32
    public var linearKeyHeads: Int = 16
    public var linearKeyHeadDim: Int = 128
    public var linearValueHeadDim: Int = 128
    public var linearConvKernelDim: Int = 4

    public var pleEmbedDim: Int?
    public var pleConvKernelSize: Int = 4
    public var ngramSize: Int = 3
    public var headsPerNGram: Int = 8
    public var ngramVocabSizeBase: Int = 20_000_000
    public var ngramVocabDivisor: Int = 128
    public var ngramShards: Int = 128
    public var hashSeed: Int = 1234
    public var eosTokenID: Int = 0
    public var usesExternalPLEStorage: Bool = false

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case hcCount = "hc_count"
        case hcLowRank = "hc_lowrank"
        case layerTypes = "layer_types"
        case fullAttentionInterval = "full_attention_interval"
        case pleLayerIDs = "ple_layer_ids"
        case experts = "num_experts"
        case expertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
        case normTopkProb = "norm_topk_prob"
        case indexerHeads = "indexer_n_heads"
        case indexerKVHeads = "indexer_kv_heads"
        case indexerHeadDim = "indexer_head_dim"
        case indexerBudget = "indexer_budget"
        case indexerCompressRatio = "indexer_compress_ratio"
        case outputGateType = "output_gate_type"
        case attentionBias = "attention_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
        case mtpHiddenLayers = "mtp_num_hidden_layers"
        case ropeParameters = "rope_parameters"
        case linearValueHeads = "linear_num_value_heads"
        case linearKeyHeads = "linear_num_key_heads"
        case linearKeyHeadDim = "linear_key_head_dim"
        case linearValueHeadDim = "linear_value_head_dim"
        case linearConvKernelDim = "linear_conv_kernel_dim"
        case pleEmbedDim = "ple_embed_dim"
        case pleConvKernelSize = "ple_conv_kernel_size"
        case ngramSize = "ngram_size"
        case headsPerNGram = "heads_per_ngram"
        case ngramVocabSizeBase = "ngram_vocab_size_base"
        case ngramVocabDivisor = "make_ngram_vocab_size_divisible_by"
        case ngramShards = "split_ngram_parts"
        case hashSeed = "seed"
        case eosTokenID = "eos_token_id"
        case pleStorage = "ple_storage"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        func take<T: Decodable>(_ key: CodingKeys, _ fallback: T) throws -> T {
            try c.decodeIfPresent(T.self, forKey: key) ?? fallback
        }
        modelType = try take(.modelType, modelType)
        hiddenSize = try take(.hiddenSize, hiddenSize)
        hiddenLayers = try take(.hiddenLayers, hiddenLayers)
        attentionHeads = try take(.attentionHeads, attentionHeads)
        kvHeads = try take(.kvHeads, kvHeads)
        headDim = try c.decodeIfPresent(Int.self, forKey: .headDim)
        vocabularySize = try take(.vocabularySize, vocabularySize)
        maxPositionEmbeddings = try take(.maxPositionEmbeddings, maxPositionEmbeddings)
        rmsNormEps = try take(.rmsNormEps, rmsNormEps)
        hcCount = try take(.hcCount, hcCount)
        hcLowRank = try take(.hcLowRank, hcLowRank)
        fullAttentionInterval = try take(.fullAttentionInterval, fullAttentionInterval)
        pleLayerIDs = try take(.pleLayerIDs, pleLayerIDs)
        experts = try take(.experts, experts)
        expertsPerToken = try take(.expertsPerToken, expertsPerToken)
        moeIntermediateSize = try take(.moeIntermediateSize, moeIntermediateSize)
        sharedExpertIntermediateSize = try take(
            .sharedExpertIntermediateSize, sharedExpertIntermediateSize)
        normTopkProb = try take(.normTopkProb, normTopkProb)
        indexerHeads = try take(.indexerHeads, indexerHeads)
        indexerKVHeads = try take(.indexerKVHeads, indexerKVHeads)
        indexerHeadDim = try take(.indexerHeadDim, indexerHeadDim)
        indexerBudget = try take(.indexerBudget, indexerBudget)
        indexerCompressRatio = try take(.indexerCompressRatio, indexerCompressRatio)
        outputGateType = try take(.outputGateType, outputGateType)
        attentionBias = try take(.attentionBias, attentionBias)
        tieWordEmbeddings = try take(.tieWordEmbeddings, tieWordEmbeddings)
        mtpHiddenLayers = try take(.mtpHiddenLayers, mtpHiddenLayers)
        ropeParameters = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeParameters)

        linearValueHeads = try take(.linearValueHeads, linearValueHeads)
        linearKeyHeads = try take(.linearKeyHeads, linearKeyHeads)
        linearKeyHeadDim = try take(.linearKeyHeadDim, linearKeyHeadDim)
        linearValueHeadDim = try take(.linearValueHeadDim, linearValueHeadDim)
        linearConvKernelDim = try take(.linearConvKernelDim, linearConvKernelDim)

        pleEmbedDim = try c.decodeIfPresent(Int.self, forKey: .pleEmbedDim)
        pleConvKernelSize = try take(.pleConvKernelSize, pleConvKernelSize)
        ngramSize = try take(.ngramSize, ngramSize)
        headsPerNGram = try take(.headsPerNGram, headsPerNGram)
        ngramVocabSizeBase = try take(.ngramVocabSizeBase, ngramVocabSizeBase)
        ngramVocabDivisor = try take(.ngramVocabDivisor, ngramVocabDivisor)
        ngramShards = try take(.ngramShards, ngramShards)
        hashSeed = try take(.hashSeed, hashSeed)
        // Either a single id or a list; the first is the one the n-gram shift pads with.
        if let single = try? c.decode(Int.self, forKey: .eosTokenID) {
            eosTokenID = single
        } else if let list = try? c.decode([Int].self, forKey: .eosTokenID),
            let first = list.first
        {
            eosTokenID = first
        }
        if let storage = try? c.decode(
            [String: StringOrNumber].self, forKey: .pleStorage), !storage.isEmpty
        {
            usesExternalPLEStorage = true
        }

        pleLayerIDs = Array(Set(pleLayerIDs)).sorted()
        // A checkpoint may name the sparse kind `full_attention`; both mean the same layer.
        if let declared = try c.decodeIfPresent([String].self, forKey: .layerTypes) {
            layerTypes = declared.map {
                $0 == "full_attention" ? "qwen_sparse_attention" : $0
            }
        } else {
            layerTypes = (0 ..< hiddenLayers).map {
                ($0 + 1) % fullAttentionInterval != 0
                    ? "linear_attention" : "qwen_sparse_attention"
            }
        }

        func refuse(_ reason: String) -> DecodingError {
            DecodingError.dataCorrupted(
                .init(codingPath: c.codingPath, debugDescription: reason))
        }
        let kinds = layerTypes ?? []
        guard kinds.count == hiddenLayers else {
            throw refuse(
                "layer_types must contain one entry per decoder layer: "
                    + "\(kinds.count) != \(hiddenLayers)")
        }
        let unsupported = Set(kinds).subtracting(["linear_attention", "qwen_sparse_attention"])
        guard unsupported.isEmpty else {
            throw refuse("Unsupported Qwen4-Exp layer types: \(unsupported.sorted())")
        }
        guard hcCount > 1 else { throw refuse("Qwen4-Exp requires hc_count > 1") }
        guard outputGateType == "sigmoid" || outputGateType == "silu" else {
            throw refuse("Qwen4-Exp output_gate_type must be 'sigmoid' or 'silu'")
        }
        guard indexerKVHeads == 1 else {
            throw refuse("Qwen4-Exp QSA requires indexer_kv_heads=1")
        }
        guard indexerCompressRatio != 0, indexerBudget % indexerCompressRatio == 0 else {
            throw refuse("indexer_budget must be divisible by indexer_compress_ratio")
        }
        guard ngramHeads > 0, effectivePLEEmbedDim % ngramHeads == 0 else {
            throw refuse(
                "ple_embed_dim must be divisible by the total number of n-gram heads")
        }
        let outOfRange = pleLayerIDs.filter { $0 < 1 || $0 > hiddenLayers }
        guard outOfRange.isEmpty else {
            throw refuse("Invalid one-indexed PLE layers: \(outOfRange)")
        }
        let notLinear = pleLayerIDs.filter { kinds[$0 - 1] != "linear_attention" }
        guard notLinear.isEmpty else {
            throw refuse(
                "PLE is only supported on linear-attention layers, got \(notLinear)")
        }
        guard pleLayerIDs.isEmpty || c.contains(.eosTokenID) else {
            throw refuse("eos_token_id is required when PLE is enabled")
        }
    }

    public var resolvedHeadDim: Int { headDim ?? (hiddenSize / attentionHeads) }
    public var hyperConnectionWidth: Int { hcCount * hiddenSize }

    public var effectivePLEEmbedDim: Int { pleEmbedDim ?? hiddenSize }
    public var ngramContextLength: Int { ngramSize - 1 }
    public var ngramHeads: Int { (ngramSize - 1) * headsPerNGram }

    /// `ple_layer_ids` is one-based, and its position in the sorted list is what seeds
    /// that layer's hash multipliers.
    public func pleIndex(forLayer index: Int) -> Int? {
        pleLayerIDs.firstIndex(of: index + 1)
    }

    public func layerIsSparse(_ index: Int) -> Bool {
        (layerTypes?[index] ?? "qwen_sparse_attention") == "qwen_sparse_attention"
    }
}

/// Qwen4's RMS norm.
///
/// Two things differ from the usual one. Its checkpoint weights are centred at zero and
/// applied as `1 + weight`, so a freshly built norm is the identity rather than a no-op
/// scaling. And with a group size it normalizes within groups along the last axis, which is
/// how one norm covers several hyper-connection streams without mixing them.
class Qwen4ExpRMSNorm: Module {
    @ParameterInfo var weight: MLXArray

    let eps: Float
    let groupSize: Int?

    init(dimensions: Int, groupSize: Int? = nil, eps: Float = 1e-6) {
        self.eps = eps
        self.groupSize = groupSize
        if let groupSize, dimensions % groupSize != 0 {
            fatalError("Qwen4-Exp RMSNorm needs \(dimensions) divisible by \(groupSize)")
        }
        _weight.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let dtype = x.dtype
        var y = x.asType(.float32)
        var w = weight
        if let groupSize {
            y = y.reshaped(Array(y.shape.dropLast()) + [-1, groupSize])
            w = weight.reshaped(-1, groupSize)
        }
        y = y * rsqrt(y.square().mean(axis: -1, keepDims: true) + eps)
        y = y * (1.0 + w.asType(.float32))
        return y.reshaped(x.shape).asType(dtype)
    }
}

/// Mixes the hyper-connection streams into one block input.
///
/// The streams are normed together but per stream, then gated by a low-rank projection and
/// averaged. With `useCombine` the block also returns the untouched streams and the weights
/// its output is injected back with; the MTP head takes only the mixed input.
class Qwen4ExpGatedResidual: Module {
    @ModuleInfo(key: "hc_norm") var hcNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "input_mix_weight_down") var mixDown: Linear
    @ModuleInfo(key: "input_mix_weight_up") var mixUp: Linear
    @ModuleInfo(key: "block_inject_weight") var blockInject: Linear?

    let hcCount: Int
    let hiddenSize: Int

    init(_ config: Qwen4ExpTextConfiguration, useCombine: Bool = true) {
        hcCount = config.hcCount
        hiddenSize = config.hiddenSize
        let width = config.hyperConnectionWidth

        _hcNorm.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: width, groupSize: hiddenSize, eps: config.rmsNormEps)
        _mixDown.wrappedValue = Linear(width, config.hcLowRank, bias: false)
        _mixUp.wrappedValue = Linear(config.hcLowRank, width, bias: false)
        if useCombine {
            _blockInject.wrappedValue = Linear(width, hcCount, bias: false)
        }
        super.init()
    }

    /// - Returns: the mixed block input, and — when this residual combines — the normed
    ///   streams and the per-stream weights the block's output is injected with.
    func callAsFunction(
        _ hyperInput: MLXArray
    ) -> (mixed: MLXArray, streams: MLXArray?, injection: MLXArray?) {
        let normed = hcNorm(hyperInput)
        // Dividing by the stream count keeps the gate's scale independent of how many
        // streams there are.
        var mix = silu(mixDown(normed) / Float(hcCount))
        mix = sigmoid(mixUp(mix))
        mix = mix.reshaped(Array(mix.shape.dropLast()) + [hcCount, hiddenSize])
        let streams = normed.reshaped(Array(normed.shape.dropLast()) + [hcCount, hiddenSize])
        let mixed = (mix * streams).mean(axis: -2)

        guard let blockInject else { return (mixed, nil, nil) }
        let injection = 2 * sigmoid(blockInject(normed) / Float(hcCount))
        return (mixed, hyperInput, injection)
    }
}

// MARK: - Sparse-attention cache

/// Concatenates indexer positions, reconciling plain text positions with multimodal ones.
///
/// Text positions are one row per batch; MRoPE positions carry an extra leading axis. A
/// sequence can start as one and continue as the other, so whichever is narrower is
/// broadcast up rather than the mismatch being an error.
func appendQSAIndexerPositions(
    cached: MLXArray?, positionIDs: MLXArray
) throws -> MLXArray {
    guard var cached else { return positionIDs }
    var positions = positionIDs
    if cached.ndim == 3, positions.ndim == 2 {
        positions = broadcast(
            positions[.newAxis], to: [cached.dim(0)] + positions.shape)
    } else if cached.ndim == 2, positions.ndim == 3 {
        cached = broadcast(cached[.newAxis], to: [positions.dim(0)] + cached.shape)
    } else if cached.ndim != positions.ndim {
        throw Qwen4ExpError.positionRank(cached: cached.shape, current: positions.shape)
    }
    return concatenated([cached, positions], axis: -1)
}

public enum Qwen4ExpError: Error, CustomStringConvertible {
    case positionRank(cached: [Int], current: [Int])

    public var description: String {
        switch self {
        case .positionRank(let cached, let current):
            "QSA position ids must be 2-D text positions or 3-D MRoPE positions, got "
                + "cached=\(cached) and current=\(current)."
        }
    }
}

/// The KV cache a Qwen sparse-attention layer keeps.
///
/// Alongside the projected keys and values it carries the indexer's *unprojected* keys and
/// the positions they were seen at, because the indexer scores against those rather than the
/// attention's own keys. They have to move together: a trim that took the KV back but left
/// the indexer's keys would select against positions that no longer exist, which is exactly
/// what a rejected speculative block asks for.
public class QSAKVCache: KVCacheSimple {
    public var indexKeys: MLXArray?
    public var indexPositionIDs: MLXArray?
    /// Block-compressed indexer keys, rebuilt on demand rather than trimmed.
    public var indexBlockKeys: MLXArray?
    public var indexBlockRatio: Int?

    public override func innerState() -> [MLXArray] {
        super.innerState() + [indexKeys, indexPositionIDs, indexBlockKeys].compactMap { $0 }
    }

    public override func copy() -> any KVCache {
        let result = QSAKVCache()
        result.keys = keys.map { $0[.ellipsis] }
        result.values = values.map { $0[.ellipsis] }
        result.offset = offset
        result.step = step
        result.indexKeys = indexKeys.map { $0[.ellipsis] }
        result.indexPositionIDs = indexPositionIDs.map { $0[.ellipsis] }
        result.indexBlockKeys = indexBlockKeys.map { $0[.ellipsis] }
        result.indexBlockRatio = indexBlockRatio
        return result
    }

    @discardableResult
    public func updateIndexer(
        keys: MLXArray, positionIDs: MLXArray
    ) throws -> (keys: MLXArray, positionIDs: MLXArray) {
        if indexKeys == nil {
            indexKeys = keys
            indexPositionIDs = positionIDs
        } else {
            indexKeys = concatenated([indexKeys!, keys], axis: 1)
            indexPositionIDs = try appendQSAIndexerPositions(
                cached: indexPositionIDs, positionIDs: positionIDs)
        }
        return (indexKeys!, indexPositionIDs!)
    }

    /// Dropped rather than trimmed: they are a compression of the keys and are cheaper to
    /// rebuild than to keep consistent.
    public func clearIndexBlocks() {
        indexBlockKeys = nil
        indexBlockRatio = nil
    }

    @discardableResult
    public override func trim(_ n: Int) -> Int {
        let trimmed = super.trim(n)
        if indexKeys != nil {
            indexKeys = indexKeys![0..., ..<offset]
            indexPositionIDs = indexPositionIDs![.ellipsis, ..<offset]
        }
        clearIndexBlocks()
        return trimmed
    }
}

// MARK: - Sparse-attention indexer

/// Which key blocks a sparse-attention layer will attend to, before the mask is built.
struct QSASelection {
    let selectedBlocks: MLXArray
    let queryEnds: MLXArray
    let completeCounts: MLXArray
    let keyLength: Int
    /// True once every query has more complete blocks than the budget, so no row falls back
    /// to dense attention.
    let allSparse: Bool
}

/// Picks compressed key blocks by Qwen sparse-attention score.
///
/// Keys are pooled into blocks of `compressRatio`, scored against the query, and the
/// best `budget / compressRatio` blocks are attended. Below that many complete blocks
/// there is nothing to choose between, so the layer stays dense and this returns nothing.
///
/// Only the unpadded single-stream path is ported. Upstream additionally gathers per row
/// when a batch is left-padded, which needs the batched caches this package does not have.
class Qwen4ExpQSAIndexer: Module {
    @ModuleInfo(key: "index_qk_proj") var indexQKProj: Linear
    @ModuleInfo(key: "q_layernorm") var queryNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "k_layernorm") var keyNorm: Qwen4ExpRMSNorm

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let tokenBudget: Int
    let compressRatio: Int
    let blockTopK: Int
    let rotary: Qwen35Language.RotaryEmbedding

    init(_ config: Qwen4ExpTextConfiguration, rotary: Qwen35Language.RotaryEmbedding) {
        heads = config.indexerHeads
        kvHeads = config.indexerKVHeads
        headDim = config.indexerHeadDim
        tokenBudget = config.indexerBudget
        compressRatio = config.indexerCompressRatio
        blockTopK = config.indexerBudget / config.indexerCompressRatio
        self.rotary = rotary

        _indexQKProj.wrappedValue = Linear(
            config.hiddenSize, (heads + kvHeads) * headDim, bias: false)
        _queryNorm.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: headDim, eps: config.rmsNormEps)
        _keyNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        super.init()
    }

    private func defaultPositionIDs(batch: Int, start: Int, length: Int) -> MLXArray {
        let steps = MLXArray(Int32(0) ..< Int32(length))
        return broadcast((Int32(start) + steps)[.newAxis], to: [batch, length])
    }

    /// MRoPE's helper rotates two operands at once, so the second is a throwaway here.
    private func applyRope(_ x: MLXArray, positionIDs: MLXArray) -> MLXArray {
        let (cosValues, sinValues) = rotary(x: x, positionIds: positionIDs)
        return Qwen35Language.applyMultimodalRotaryPosEmb(
            q: x, k: x, cos: cosValues, sin: sinValues
        ).0
    }

    func callAsFunction(
        _ hiddenStates: MLXArray, cache: QSAKVCache?, positionIDs: MLXArray?
    ) throws -> MLXArray? {
        guard
            let selection = try select(
                projected: indexQKProj(hiddenStates), cache: cache,
                positionIDs: positionIDs)
        else { return nil }
        return buildMask(selection)
    }

    func select(
        projected qk: MLXArray, cache: QSAKVCache?, positionIDs: MLXArray?
    ) throws -> QSASelection? {
        let batch = qk.dim(0)
        let sequence = qk.dim(1)
        let pastLength = cache?.offset ?? 0
        let positions =
            positionIDs
            ?? defaultPositionIDs(batch: batch, start: pastLength, length: sequence)

        let reshaped = qk.reshaped(batch, sequence, heads + kvHeads, headDim)
        var query = reshaped[0..., 0..., ..<heads]
        let newKeys = MLX.squeezed(reshaped[0..., 0..., heads...], axis: 2)
        query = queryNorm(query).transposed(0, 2, 1, 3)

        var rawKeys = newKeys
        var fullPositionIDs = positions
        if let cache {
            (rawKeys, fullPositionIDs) = try cache.updateIndexer(
                keys: newKeys, positionIDs: positions)
        }

        let keyLength = rawKeys.dim(1)
        let completeBlocks = keyLength / compressRatio
        // Fewer complete blocks than the budget means every one would be selected.
        guard completeBlocks > blockTopK else { return nil }

        query = applyRope(query, positionIDs: positions)

        // Blocks already pooled, normed and rotated are kept on the cache, so a later
        // prefill chunk only summarizes the blocks it completed.
        let cachedBlocks = cache?.indexBlockKeys
        let canReuse =
            cachedBlocks != nil && cachedBlocks!.ndim == 4 && cachedBlocks!.dim(0) == batch
            && cachedBlocks!.dim(1) == 1 && cachedBlocks!.dim(-1) == headDim
            && cachedBlocks!.dim(2) <= completeBlocks
            && cache?.indexBlockRatio == compressRatio
        let firstNewBlock = canReuse ? cachedBlocks!.dim(2) : 0

        var pooledKeys: MLXArray
        if completeBlocks > firstNewBlock {
            let newBlockIDs = MLXArray(Int32(firstNewBlock) ..< Int32(completeBlocks))
            let tokenStart = firstNewBlock * compressRatio
            let completeLength = completeBlocks * compressRatio
            var pooled = rawKeys[0..., tokenStart ..< completeLength]
                .reshaped(batch, completeBlocks - firstNewBlock, compressRatio, headDim)
            pooled = expandedDimensions(
                keyNorm(pooled.asType(.float32).mean(axis: 2).asType(rawKeys.dtype)),
                axis: 1)
            let blockPositions = fullPositionIDs[
                .ellipsis, newBlockIDs * Int32(compressRatio)]
            pooled = applyRope(pooled, positionIDs: blockPositions)
            pooledKeys =
                (canReuse && firstNewBlock > 0)
                ? concatenated([cachedBlocks!, pooled], axis: 2) : pooled
        } else {
            pooledKeys = cachedBlocks!
        }
        cache?.indexBlockKeys = pooledKeys
        cache?.indexBlockRatio = compressRatio

        // Scored in float32 as the reference does: which blocks win is a discrete choice,
        // and rounding the products flips the ones near the cut-off.
        var scores = query.asType(.float32)
            .matmul(pooledKeys.asType(.float32).transposed(0, 1, 3, 2))
        scores = maximum(scores, 0).sum(axis: 1) / Float(headDim).squareRoot()

        let queryEnds = Int32(pastLength) + MLXArray(Int32(0) ..< Int32(sequence)) + 1
        // Floor division: a partially filled block is not a complete one, and true
        // division would round a row up into the sparse path with an empty tail.
        let completeCounts = floorDivide(queryEnds[.newAxis], Int32(compressRatio))
        let blockIDs = MLXArray(Int32(0) ..< Int32(completeBlocks))
        let valid = blockIDs[.newAxis, .newAxis] .< completeCounts[.ellipsis, .newAxis]
        scores = MLX.where(valid, scores, MLXArray(-Float.infinity))

        let selectedBlocks = argPartition(scores, kth: completeBlocks - blockTopK, axis: -1)[
            .ellipsis, (-blockTopK)...]

        return QSASelection(
            selectedBlocks: selectedBlocks,
            queryEnds: broadcast(queryEnds[.newAxis], to: [batch, sequence]),
            completeCounts: broadcast(completeCounts, to: [batch, sequence]),
            keyLength: keyLength,
            allSparse: pastLength + 1 >= (blockTopK + 1) * compressRatio)
    }

    /// Expands a block selection into a per-token attention mask.
    ///
    /// The winning blocks are scattered straight onto the token axis. Comparing every token
    /// against every pick instead would cost `sequence * keyLength * blockTopK` bytes per
    /// prefill step.
    func buildMask(_ selection: QSASelection) -> MLXArray {
        let selectedBlocks = selection.selectedBlocks
        let batch = selectedBlocks.dim(0)
        let sequence = selectedBlocks.dim(1)
        let keyLength = selection.keyLength

        var tokenIndices =
            (selectedBlocks[.ellipsis, .newAxis] * Int32(compressRatio)
            + MLXArray(Int32(0) ..< Int32(compressRatio))[.newAxis, .newAxis, .newAxis])
            .reshaped(batch, sequence, -1)
        let validTokens = tokenIndices .< Int32(keyLength)
        // Out-of-range picks are parked in a scratch column that is sliced away.
        tokenIndices = MLX.where(validTokens, tokenIndices, MLXArray(Int32(keyLength)))
        var selectedTokens = putAlong(
            MLXArray.zeros([batch, sequence, keyLength + 1], dtype: .bool),
            tokenIndices, values: validTokens, axis: -1
        )[.ellipsis, ..<keyLength]

        // The tokens after the last complete block are always visible: they have not been
        // pooled into a block yet, so no selection could have reached them.
        let positions = MLXArray(Int32(0) ..< Int32(keyLength))
        let tailStarts = selection.completeCounts * Int32(compressRatio)
        let tail =
            (positions[.newAxis, .newAxis] .>= tailStarts[.ellipsis, .newAxis])
            .&& (positions[.newAxis, .newAxis] .< selection.queryEnds[.ellipsis, .newAxis])
        let causal = positions[.newAxis, .newAxis] .< selection.queryEnds[.ellipsis, .newAxis]

        // A row with no more complete blocks than the budget attends densely instead.
        let useSparse = selection.completeCounts .> Int32(blockTopK)
        selectedTokens = MLX.where(
            useSparse[.ellipsis, .newAxis], selectedTokens .|| tail, causal)
        return expandedDimensions(selectedTokens, axis: 1)
    }
}

// MARK: - Rope parameters

extension Qwen4ExpTextConfiguration {
    public var ropeTheta: Float {
        ropeParameters?["rope_theta"]?.asFloat() ?? 10_000_000
    }

    public var partialRotaryFactor: Float {
        ropeParameters?["partial_rotary_factor"]?.asFloat() ?? 0.25
    }

    public var mropeSection: [Int] {
        ropeParameters?["mrope_section"]?.asInts() ?? [11, 11, 10]
    }
}

// MARK: - Sparse attention

/// Qwen4-Exp's sparse attention.
///
/// The query projection is twice as wide as the head count needs, because the second half
/// is an output gate applied after attention. Position enters through MRoPE over a partial
/// rotary slice, and the indexer decides which key blocks are visible -- returning nothing
/// while the cache is short enough that every block would be selected, in which case the
/// layer attends densely.
class Qwen4ExpAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: Qwen4ExpRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: Qwen4ExpRMSNorm
    @ModuleInfo var indexer: Qwen4ExpQSAIndexer

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float
    let rotary: Qwen35Language.RotaryEmbedding

    init(_ config: Qwen4ExpTextConfiguration) {
        heads = config.attentionHeads
        kvHeads = config.kvHeads
        headDim = config.resolvedHeadDim
        scale = pow(Float(headDim), -0.5)

        // Twice the width: the second half is the output gate.
        _qProj.wrappedValue = Linear(
            config.hiddenSize, heads * headDim * 2, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(
            heads * headDim, config.hiddenSize, bias: config.attentionBias)
        _qNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = Qwen4ExpRMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        let rotary = Qwen35Language.RotaryEmbedding(
            dim: Int(Float(headDim) * config.partialRotaryFactor),
            base: config.ropeTheta,
            mropeSection: config.mropeSection)
        self.rotary = rotary
        _indexer.wrappedValue = Qwen4ExpQSAIndexer(config, rotary: rotary)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cache: QSAKVCache?, positionIDs: MLXArray?
    ) throws -> MLXArray {
        let batch = x.dim(0)
        let length = x.dim(1)

        // The indexer runs first: it reads the cache offset as its past length, and its
        // own keys have to be recorded whether or not a selection comes back.
        let selection = try indexer.select(
            projected: indexer.indexQKProj(x), cache: cache, positionIDs: positionIDs)

        let projected = qProj(x).reshaped(batch, length, heads, -1)
        let halves = projected.split(parts: 2, axis: -1)
        var queries = qNorm(halves[0]).transposed(0, 2, 1, 3)
        let gate = halves[1].reshaped(batch, length, -1)

        var keys = kNorm(kProj(x).reshaped(batch, length, kvHeads, -1))
            .transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(batch, length, kvHeads, -1).transposed(0, 2, 1, 3)

        var positions = positionIDs
        if positions == nil {
            let offset = cache?.offset ?? 0
            var base = MLXArray(Int32(offset) ..< Int32(offset + length))
            base = tiled(base[.newAxis, 0...], repetitions: [batch, 1])
            positions = tiled(base[.newAxis, 0..., 0...], repetitions: [3, 1, 1])
        }
        let (cosValues, sinValues) = rotary(x: values, positionIds: positions!)
        (queries, keys) = Qwen35Language.applyMultimodalRotaryPosEmb(
            q: queries, k: keys, cos: cosValues, sin: sinValues)

        // A selection replaces the causal mask; without one the layer is dense.
        var attentionMask = mask
        if let selection {
            attentionMask = indexer.buildMask(selection)
        }

        var output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache, scale: scale,
            mask: attentionMask.map { .array($0) } ?? .none)
        output = output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
        return oProj(output * sigmoid(gate))
    }
}

// MARK: - Mixture of experts

/// The same sparse block Qwen3.5 uses, over Qwen4-Exp's config.
///
/// Parameter names match the checkpoint's, so this is a retyping rather than a variant.
class Qwen4ExpSparseMoeBlock: Module, UnaryLayer {
    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_expert") var sharedExpert: SwiGLUMLP
    @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

    let normTopkProb: Bool
    let topK: Int

    init(_ config: Qwen4ExpTextConfiguration) {
        normTopkProb = config.normTopkProb
        topK = config.expertsPerToken

        _gate.wrappedValue = Linear(config.hiddenSize, config.experts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize, hiddenDims: config.moeIntermediateSize,
            numExperts: config.experts)
        _sharedExpert.wrappedValue = SwiGLUMLP(
            inputDimensions: config.hiddenSize,
            hiddenDimensions: config.sharedExpertIntermediateSize)
        _sharedExpertGate.wrappedValue = Linear(config.hiddenSize, 1, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = MLX.softmax(gate(x), axis: -1, precise: true)
        let (indices, scores) = moeRouterTopK(gates, k: topK, normalize: normTopkProb)
        let routed = weightedExpertSum(switchMLP(x, indices), scores)
        return routed + sigmoid(sharedExpertGate(x)) * sharedExpert(x)
    }
}

// MARK: - Decoder layer

/// One Qwen4-Exp block.
///
/// The residual is a hyper-connection rather than an addition: the streams are mixed into
/// one block input, the branch runs on that, and its output is injected back into every
/// stream with its own weight. The branch is either sparse attention or gated-delta linear
/// attention, and a layer named in `ple_layer_ids` adds its per-layer embedding to the
/// streams before the block runs.
class Qwen4ExpDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: Qwen4ExpAttention?
    @ModuleInfo(key: "linear_attn") var linearAttention: Qwen4ExpGatedDeltaNet?
    @ModuleInfo var mlp: Qwen4ExpSparseMoeBlock
    @ModuleInfo var ple: Qwen4ExpPLELayer?
    @ModuleInfo(key: "attn_hyper_connection") var attentionHyperConnection: Qwen4ExpGatedResidual
    @ModuleInfo(key: "mlp_hyper_connection") var mlpHyperConnection: Qwen4ExpGatedResidual

    let isLinear: Bool

    init(_ config: Qwen4ExpTextConfiguration, layerIndex: Int) {
        isLinear = !config.layerIsSparse(layerIndex)
        if isLinear {
            _linearAttention.wrappedValue = Qwen4ExpGatedDeltaNet(config)
        } else {
            _attention.wrappedValue = Qwen4ExpAttention(config)
        }
        _mlp.wrappedValue = Qwen4ExpSparseMoeBlock(config)
        if let pleIndex = config.pleIndex(forLayer: layerIndex) {
            _ple.wrappedValue = Qwen4ExpPLELayer(config, pleIndex: pleIndex)
        }
        _attentionHyperConnection.wrappedValue = Qwen4ExpGatedResidual(config)
        _mlpHyperConnection.wrappedValue = Qwen4ExpGatedResidual(config)
        super.init()
    }

    private func inject(
        branch: MLXArray, into streams: MLXArray, weights: MLXArray
    ) -> MLXArray {
        let injection = branch[.ellipsis, .newAxis, 0...] * weights[.ellipsis, .newAxis]
        return streams + injection.reshaped(streams.shape)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        inputIDs: MLXArray?,
        mask: MLXArray?,
        cache: KVCache?,
        positionIDs: MLXArray?
    ) throws -> MLXArray {
        var hidden = hiddenStates
        if let ple, let inputIDs {
            hidden =
                hidden
                + ple(hidden, inputIDs: inputIDs, cache: cache as? ArraysCache, mask: mask)
        }

        var (mixed, streams, weights) = attentionHyperConnection(hidden)
        var branch: MLXArray
        if let linearAttention {
            branch = linearAttention(mixed, mask: mask, cache: cache as? ArraysCache)
        } else {
            branch = try attention!(
                mixed, mask: mask, cache: cache as? QSAKVCache, positionIDs: positionIDs)
        }
        var h = inject(branch: branch, into: streams!, weights: weights!)

        (mixed, streams, weights) = mlpHyperConnection(h)
        branch = mlp(mixed)
        h = inject(branch: branch, into: streams!, weights: weights!)
        return h
    }
}

// MARK: - Linear attention

/// RMS normalization with a gate whose nonlinearity the configuration names.
class Qwen4ExpRMSNormGated: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    let usesSigmoid: Bool

    init(dimensions: Int, eps: Float, activation: String) {
        self.eps = eps
        usesSigmoid = activation == "sigmoid"
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let dtype = x.dtype
        let normalized = MLXFast.rmsNorm(x, weight: weight, eps: eps).asType(.float32)
        let gate32 = gate.asType(.float32)
        let activated = usesSigmoid ? sigmoid(gate32) : silu(gate32)
        return (normalized * activated).asType(dtype)
    }
}

/// Gated delta attention.
///
/// The queries and keys are **L2**-normalized -- the epsilon lands after the sum, not
/// after the mean -- and only then scaled by the inverse square root of the head
/// dimension. An RMS normalization here would divide by another square root of the width
/// and quietly change every score.
class Qwen4ExpGatedDeltaNet: Module {
    let valueHeads: Int
    let keyHeads: Int
    let keyHeadDim: Int
    let valueHeadDim: Int
    let keyDim: Int
    let valueDim: Int
    let convKernelSize: Int
    let convDim: Int

    @ModuleInfo(key: "conv1d") var conv: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear
    @ModuleInfo(key: "norm") var norm: Qwen4ExpRMSNormGated
    @ModuleInfo(key: "out_proj") var outProj: Linear

    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray

    init(_ config: Qwen4ExpTextConfiguration) {
        valueHeads = config.linearValueHeads
        keyHeads = config.linearKeyHeads
        keyHeadDim = config.linearKeyHeadDim
        valueHeadDim = config.linearValueHeadDim
        keyDim = keyHeads * keyHeadDim
        valueDim = valueHeads * valueHeadDim
        convKernelSize = config.linearConvKernelDim
        convDim = 2 * keyDim + valueDim

        _conv.wrappedValue = Conv1d(
            inputChannels: convDim, outputChannels: convDim,
            kernelSize: convKernelSize, padding: 0, groups: convDim, bias: false)
        _inProjQKV.wrappedValue = Linear(config.hiddenSize, convDim, bias: false)
        _inProjZ.wrappedValue = Linear(config.hiddenSize, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(config.hiddenSize, valueHeads, bias: false)
        _inProjA.wrappedValue = Linear(config.hiddenSize, valueHeads, bias: false)
        _norm.wrappedValue = Qwen4ExpRMSNormGated(
            dimensions: valueHeadDim, eps: config.rmsNormEps,
            activation: config.outputGateType)
        _outProj.wrappedValue = Linear(valueDim, config.hiddenSize, bias: false)
        _aLog.wrappedValue = MLXArray.zeros([valueHeads])
        _dtBias.wrappedValue = MLXArray.zeros([valueHeads])
        super.init()
    }

    private func l2Normalized(_ x: MLXArray) -> MLXArray {
        x * rsqrt((x * x).sum(axis: -1, keepDims: true) + 1e-6)
    }

    func callAsFunction(_ inputs: MLXArray, mask: MLXArray?, cache: ArraysCache?)
        -> MLXArray
    {
        let (B, S) = (inputs.dim(0), inputs.dim(1))
        var mixed = inProjQKV(inputs)
        let z = inProjZ(inputs).reshaped(B, S, -1, valueHeadDim)
        let b = inProjB(inputs)
        let a = inProjA(inputs)

        if let mask {
            mixed = MLX.where(mask[.ellipsis, .newAxis], mixed, MLXArray(0))
        }

        let tail =
            cache?[0]
            ?? MLXArray.zeros([B, convKernelSize - 1, convDim], dtype: inputs.dtype)
        let padded = concatenated([tail, mixed], axis: 1)
        if let cache {
            let keep = convKernelSize - 1
            if let lengths = cache.currentLengths {
                let ends = clip(lengths, min: 0, max: S)
                let positions = (ends[0..., .newAxis] + MLXArray(0 ..< keep))[
                    .ellipsis, .newAxis]
                cache[0] = takeAlong(
                    padded, broadcast(positions, to: [B, keep, convDim]).asType(.int32),
                    axis: 1)
            } else {
                cache[0] = padded[0..., (-keep)...]
            }
        }
        let convolved = silu(conv(padded))

        let parts = split(convolved, indices: [keyDim, 2 * keyDim], axis: -1)
        var q = parts[0].reshaped(B, S, keyHeads, keyHeadDim)
        var k = parts[1].reshaped(B, S, keyHeads, keyHeadDim)
        let v = parts[2].reshaped(B, S, valueHeads, valueHeadDim)

        let scale = pow(Float(keyHeadDim), -0.5)
        q = l2Normalized(q) * scale
        k = l2Normalized(k)

        let (out, state) = gatedDeltaUpdate(
            q: q, k: k, v: v, a: a, b: b, aLog: aLog, dtBias: dtBias,
            state: cache?[1], mask: mask)
        if let cache {
            cache[1] = state
            cache.advance(S)
        }

        return outProj(norm(out, gate: z).reshaped(B, S, -1))
    }
}

// MARK: - Per-layer embeddings

/// The deterministic tables a per-layer embedding derives from its configuration: hash
/// multipliers, and each n-gram head's prime vocabulary size and offset into the joined
/// table. None of them are in the checkpoint.
///
/// Every multiplier is odd and bounded so `token * multiplier` cannot overflow a signed
/// 64-bit product against the unigram vocabulary -- which is why the bound comes from the
/// vocabulary size rather than being fixed.
enum Qwen4ExpHash {
    static let gamma: UInt64 = 0x9E37_79B9_7F4A_7C15
    static let mix1: UInt64 = 0xBF58_476D_1CE4_E5B9
    static let mix2: UInt64 = 0x94D0_49BB_1331_11EB
    static let prime: UInt64 = 10007

    static func splitmix64(_ value: UInt64) -> UInt64 {
        var x = value &+ gamma
        x = (x ^ (x >> 30)) &* mix1
        x = (x ^ (x >> 27)) &* mix2
        return x ^ (x >> 31)
    }

    static func layerMultipliers(
        unigramVocabularySize: Int, ngramSize: Int, pleIndex: Int, seed: Int
    ) -> [Int64] {
        let multiplierMax = UInt64(Int64.max) / UInt64(Swift.max(unigramVocabularySize, 1))
        let halfBound = Swift.max(UInt64(1), multiplierMax / 2)
        let baseSeed = UInt64(bitPattern: Int64(seed)) &+ prime &* UInt64(pleIndex)
        return (0 ..< ngramSize).map { index in
            let value = baseSeed &+ gamma &* UInt64(index + 1)
            return Int64(2 &* (splitmix64(value) % halfBound) &+ 1)
        }
    }

    static func isPrime(_ value: Int) -> Bool {
        if value < 2 { return false }
        if value % 2 == 0 { return value == 2 }
        var divisor = 3
        while divisor * divisor <= value {
            if value % divisor == 0 { return false }
            divisor += 2
        }
        return true
    }

    /// The `count`-th prime strictly greater than `start`.
    static func nthPrimeAfter(_ start: Int, count: Int) -> Int {
        var prime = start
        for _ in 0 ..< count {
            prime += 1
            while !isPrime(prime) { prime += 1 }
        }
        return prime
    }

    /// Per-head `(vocabularySize, offset)` for one PLE layer's n-gram heads.
    static func headTables(
        ngramHeads: Int, vocabularySizeBase: Int, pleIndex: Int
    ) -> (sizes: [Int64], offsets: [Int64], total: Int) {
        var sizes: [Int64] = []
        var offsets: [Int64] = []
        var total = 0
        for head in 0 ..< ngramHeads {
            let size = nthPrimeAfter(
                vocabularySizeBase - 1, count: pleIndex * ngramHeads + head + 1)
            sizes.append(Int64(size))
            offsets.append(Int64(total))
            total += size
        }
        return (sizes, offsets, total)
    }
}

/// An embedding table held as checkpoint-sized row shards.
///
/// The PLE table is far too large to join into one tensor, so the shards stay separate and
/// a lookup routes each row to the shard that owns it. Rows are split evenly, with the
/// first `rows % shardCount` shards taking one extra -- the layout the checkpoint's shard
/// shapes assume.
class Qwen4ExpShardedEmbedding: Module {
    @ModuleInfo var shards: [Embedding]

    let shardOffsets: [Int]
    let dimensions: Int

    init(rows: Int, dimensions: Int, shardCount: Int) {
        precondition(
            shardCount > 0 && shardCount <= rows,
            "Qwen4-Exp PLE needs a shard count in [1, \(rows)], got \(shardCount)")
        self.dimensions = dimensions
        let base = rows / shardCount
        let remainder = rows % shardCount
        let sizes = (0 ..< shardCount).map { base + ($0 < remainder ? 1 : 0) }
        var offsets = [0]
        for size in sizes { offsets.append(offsets[offsets.count - 1] + size) }
        shardOffsets = offsets
        _shards.wrappedValue = sizes.map {
            Embedding(embeddingCount: $0, dimensions: dimensions)
        }
        super.init()
    }

    private func shard(owning row: Int) -> Int {
        var low = 0
        var high = shardOffsets.count - 1
        while low + 1 < high {
            let mid = (low + high) / 2
            if shardOffsets[mid] <= row { low = mid } else { high = mid }
        }
        return low
    }

    /// Costs one host synchronization per lookup, which is the cheaper trade: scheduling a
    /// gather against every one of the shards for each token dominates reading the ids back.
    func callAsFunction(_ indices: MLXArray) -> MLXArray {
        let flat = indices.reshaped(-1)
        guard shards.count > 1 else {
            return shards[0](flat).reshaped(indices.shape + [dimensions])
        }
        let rows = flat.asArray(Int64.self)
        let limit = Int64(shardOffsets[shardOffsets.count - 1])
        precondition(
            rows.allSatisfy { $0 >= 0 && $0 < limit },
            "Qwen4-Exp PLE index is outside the sharded vocabulary")

        var buckets = [[Int]](repeating: [], count: shards.count)
        for (position, row) in rows.enumerated() {
            buckets[shard(owning: Int(row))].append(position)
        }
        var order: [Int32] = []
        var gathered: [MLXArray] = []
        for (index, positions) in buckets.enumerated() where !positions.isEmpty {
            let low = Int64(shardOffsets[index])
            gathered.append(shards[index](MLXArray(positions.map { Int32(rows[$0] - low) })))
            order.append(contentsOf: positions.map(Int32.init))
        }
        var values = concatenated(gathered, axis: 0)
        if gathered.count > 1 {
            var inverse = [Int32](repeating: 0, count: rows.count)
            for (slot, position) in order.enumerated() { inverse[Int(position)] = Int32(slot) }
            values = values[MLXArray(inverse)]
        }
        return values.reshaped(indices.shape + [dimensions])
    }
}

/// Hashed n-gram embeddings.
///
/// Each head owns its own prime-sized slice of one padded table, and a token's id within
/// that head is the exclusive-or of the multiplied ids of the `n` tokens ending at it,
/// reduced modulo the head's prime. The shift that builds those histories stops at an
/// end-of-sequence token, so an n-gram never reaches across a document boundary.
class Qwen4ExpNGramEmbedding: Module {
    let ngramSize: Int
    let contextLength: Int
    let headsPerNGram: Int
    let ngramHeads: Int
    let eosTokenID: Int64

    @ModuleInfo(key: "ngram_embedding") var table: Qwen4ExpShardedEmbedding

    /// Held as host values, not `MLXArray`s: a stored array would be published as a
    /// parameter and then looked for in the checkpoint, which does not carry it.
    let multipliers: [Int64]
    let headVocabularySizes: [Int64]
    let headOffsets: [Int64]

    init(_ config: Qwen4ExpTextConfiguration, embeddingDim: Int, pleIndex: Int) {
        ngramSize = config.ngramSize
        contextLength = config.ngramContextLength
        headsPerNGram = config.headsPerNGram
        ngramHeads = config.ngramHeads
        eosTokenID = Int64(config.eosTokenID)

        let tables = Qwen4ExpHash.headTables(
            ngramHeads: config.ngramHeads,
            vocabularySizeBase: config.ngramVocabSizeBase, pleIndex: pleIndex)
        headVocabularySizes = tables.sizes
        headOffsets = tables.offsets
        multipliers = Qwen4ExpHash.layerMultipliers(
            unigramVocabularySize: config.vocabularySize, ngramSize: config.ngramSize,
            pleIndex: pleIndex, seed: config.hashSeed)

        let divisor = Swift.max(config.ngramVocabDivisor, 1)
        let padded = ((tables.total + divisor - 1) / divisor) * divisor
        _table.wrappedValue = Qwen4ExpShardedEmbedding(
            rows: padded, dimensions: embeddingDim / config.ngramHeads,
            shardCount: Swift.min(config.ngramShards, padded))
        super.init()
    }

    /// Shifts right by `shift`, refusing to cross an end-of-sequence token: a position
    /// fewer than `shift` tokens into its own segment takes the end-of-sequence id instead.
    func shiftRightIgnoringEOS(_ tokens: MLXArray, shift: Int) -> MLXArray {
        guard shift != 0 else { return tokens }
        let (batch, length) = (tokens.dim(0), tokens.dim(1))
        let positions = MLXArray(0 ..< length).asType(.int64)
        let eosPositions = MLX.where(
            tokens .== MLXArray(eosTokenID), positions, MLXArray(Int64(-1)))
        let inclusive = cummax(eosPositions, axis: 1)
        let previous = concatenated(
            [
                MLXArray.full([batch, 1], values: MLXArray(Int64(-1))),
                inclusive[0..., ..<(length - 1)],
            ], axis: 1)
        let positionInSegment = positions.expandedDimensions(axis: 0) - (previous + 1)
        let source = positions - MLXArray(Int64(shift))
        let gather = broadcast(
            maximum(source, MLXArray(Int64(0))).expandedDimensions(axis: 0),
            to: [batch, length])
        let shifted = takeAlong(tokens, gather.asType(.int32), axis: 1)
        let valid =
            (positionInSegment .>= MLXArray(Int64(shift)))
            .&& (source.expandedDimensions(axis: 0) .>= MLXArray(Int64(0)))
        return MLX.where(valid, shifted, MLXArray(eosTokenID))
    }

    func callAsFunction(_ inputIDs: MLXArray, cache: ArraysCache?) -> MLXArray {
        let tokens = inputIDs.asType(.int64)
        let batch = tokens.dim(0)
        let length = tokens.dim(1)

        let previousContext =
            cache?[3]
            ?? MLXArray.full([batch, contextLength], values: MLXArray(eosTokenID))
        let history = concatenated([previousContext, tokens], axis: -1)
        cache?[3] = contiguous(history[0..., (-contextLength)...])

        let shifted = (0 ..< ngramSize).map { shiftRightIgnoringEOS(history, shift: $0) }
        let multiplier = multipliers.map { MLXArray($0) }

        var blocks: [MLXArray] = []
        for ngram in 2 ... ngramSize {
            let start = (ngram - 2) * headsPerNGram
            let end = start + headsPerNGram
            var mixed = shifted[0] * multiplier[0]
            for position in 1 ..< ngram {
                mixed = mixed ^ (shifted[position] * multiplier[position])
            }
            let sizes = MLXArray(Array(headVocabularySizes[start ..< end])).reshaped(1, 1, -1)
            let offsets = MLXArray(Array(headOffsets[start ..< end])).reshaped(1, 1, -1)
            blocks.append(mixed.expandedDimensions(axis: -1) % sizes + offsets)
        }

        let ids = concatenated(blocks, axis: -1)[0..., (-length)...]
        let embeddings = table(ids)
        return embeddings.reshaped(Array(embeddings.shape.dropLast(2)) + [-1])
    }
}

/// A per-layer embedding block: hashed n-grams gated against the hidden state, plus a
/// dilated depthwise convolution over the gated values.
///
/// The gate is a signed square root of the key/query dot product, which keeps its scale
/// bounded where a plain dot product would not. The convolution is dilated by the n-gram
/// size, so its state window is `(kernel - 1) * ngramSize` positions rather than
/// `kernel - 1`.
class Qwen4ExpPLELayer: Module {
    let hiddenSize: Int
    let hcCount: Int
    let convDilation: Int
    let convStateLength: Int

    @ModuleInfo(key: "ple_embedding") var embedding: Qwen4ExpNGramEmbedding
    @ModuleInfo(key: "key_proj") var keyProj: Linear
    @ModuleInfo(key: "value_proj") var valueProj: Linear
    @ModuleInfo(key: "norm_key") var normKey: Qwen4ExpRMSNorm
    @ModuleInfo(key: "norm_query") var normQuery: Qwen4ExpRMSNorm
    @ModuleInfo(key: "norm_conv") var normConv: Qwen4ExpRMSNorm
    @ModuleInfo(key: "conv1d") var conv: Conv1d

    init(_ config: Qwen4ExpTextConfiguration, pleIndex: Int) {
        hiddenSize = config.hiddenSize
        hcCount = config.hcCount
        convDilation = config.ngramSize
        convStateLength = (config.pleConvKernelSize - 1) * config.ngramSize
        let width = config.hyperConnectionWidth

        _embedding.wrappedValue = Qwen4ExpNGramEmbedding(
            config, embeddingDim: config.effectivePLEEmbedDim, pleIndex: pleIndex)
        _keyProj.wrappedValue = Linear(config.effectivePLEEmbedDim, width, bias: false)
        _valueProj.wrappedValue = Linear(
            config.effectivePLEEmbedDim, config.hiddenSize, bias: false)
        _normKey.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: width, groupSize: config.hiddenSize, eps: config.rmsNormEps)
        _normQuery.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: width, groupSize: config.hiddenSize, eps: config.rmsNormEps)
        _normConv.wrappedValue = Qwen4ExpRMSNorm(
            dimensions: width, groupSize: config.hiddenSize, eps: config.rmsNormEps)
        _conv.wrappedValue = Conv1d(
            inputChannels: width, outputChannels: width,
            kernelSize: config.pleConvKernelSize, padding: 0, dilation: config.ngramSize,
            groups: width, bias: false)
        super.init()
    }

    private func shortConvolution(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let state =
            cache?[2]
            ?? MLXArray.zeros([x.dim(0), convStateLength, x.dim(-1)], dtype: x.dtype)
        let padded = concatenated([state, x], axis: 1)
        cache?[2] = contiguous(padded[0..., (-convStateLength)...])
        return silu(conv(padded))
    }

    func callAsFunction(
        _ hiddenStates: MLXArray, inputIDs: MLXArray, cache: ArraysCache?, mask: MLXArray?
    ) -> MLXArray {
        let embeddings = embedding(inputIDs, cache: cache)
        let leading = Array(hiddenStates.shape.dropLast())

        let keys = normKey(keyProj(embeddings)).reshaped(leading + [hcCount, hiddenSize])
        let values = valueProj(embeddings)
        let queries = normQuery(hiddenStates).reshaped(leading + [hcCount, hiddenSize])

        var gate =
            (keys * queries).sum(axis: -1, keepDims: true) / Float(hiddenSize).squareRoot()
        gate = sign(gate) * sqrt(maximum(abs(gate), 1e-6))

        var gatedValues =
            (sigmoid(gate) * values.expandedDimensions(axis: -2))
            .reshaped(hiddenStates.shape)
        var normalized = normConv(gatedValues)
        if let mask, mask.ndim == 2 {
            let keep = mask.expandedDimensions(axis: -1)
            gatedValues = MLX.where(keep, gatedValues, MLXArray(0).asType(gatedValues.dtype))
            normalized = MLX.where(keep, normalized, MLXArray(0).asType(normalized.dtype))
        }
        return gatedValues + shortConvolution(normalized, cache: cache)
    }
}

// MARK: - Backbone

/// The decoder stack over `hcCount` residual streams.
///
/// The embedding is tiled across the streams on the way in and the streams are collapsed
/// by one last hyper-connection on the way out, so everything between carries
/// `hcCount * hiddenSize` per position.
class Qwen4ExpModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Qwen4ExpDecoderLayer]
    @ModuleInfo(key: "hyper_connection_mixer") var mixer: Qwen4ExpGatedResidual

    let hcCount: Int
    /// The first layer of each kind, whose cache the shared masks are built from.
    let ssmIndex: Int
    let attentionIndex: Int

    init(_ config: Qwen4ExpTextConfiguration) {
        hcCount = config.hcCount
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.hiddenLayers).map {
            Qwen4ExpDecoderLayer(config, layerIndex: $0)
        }
        _mixer.wrappedValue = Qwen4ExpGatedResidual(config, useCombine: false)
        ssmIndex = (0 ..< config.hiddenLayers).first { !config.layerIsSparse($0) } ?? 0
        attentionIndex = (0 ..< config.hiddenLayers).first { config.layerIsSparse($0) } ?? 0
        super.init()
    }

    func callAsFunction(
        inputs: MLXArray?,
        inputEmbedding: MLXArray?,
        mask: MLXArray?,
        cache: [KVCache]?,
        positionIDs: MLXArray?
    ) throws -> MLXArray {
        var tokens = inputs
        if tokens?.ndim == 1 {
            tokens = tokens!.expandedDimensions(axis: 0)
        }
        let embedded = inputEmbedding ?? embedTokens(tokens!)
        let caches: [KVCache?] = cache ?? Array(repeating: nil, count: layers.count)

        var hidden = tiled(embedded, repetitions: [1, 1, hcCount])
        let attentionMask: MLXArray? = createAttentionMask(
            h: hidden, cache: caches[attentionIndex].map { [$0] })
        var ssmMask = (caches[ssmIndex] as? ArraysCache)?.makeMask(N: embedded.dim(1))
        if let mask, mask.ndim == 2 {
            ssmMask = mask
        }

        for (index, layer) in layers.enumerated() {
            hidden = try layer(
                hidden, inputIDs: tokens,
                mask: layer.isLinear ? ssmMask : attentionMask,
                cache: caches[index], positionIDs: positionIDs)
        }
        return mixer(hidden).mixed
    }
}

/// Qwen4-Exp's text model.
public class Qwen4ExpLanguageModel: Module, LLMModel, KVCacheDimensionProvider {
    @ModuleInfo var model: Qwen4ExpModelInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public let config: Qwen4ExpTextConfiguration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] {
        (0 ..< config.hiddenLayers).map { config.layerIsSparse($0) ? config.kvHeads : 0 }
    }
    public var loraLayers: [Module] { model.layers }

    public init(_ config: Qwen4ExpTextConfiguration) {
        precondition(
            !config.usesExternalPLEStorage,
            "Qwen4-Exp checkpoints with an external ple_storage manifest are not supported")
        self.config = config
        _model.wrappedValue = Qwen4ExpModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
        }
        super.init()
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        logits(inputs: inputs, inputEmbedding: nil, cache: cache)
    }

    public func logits(
        inputs: MLXArray?, inputEmbedding: MLXArray?, cache: [KVCache]?,
        mask: MLXArray? = nil, positionIDs: MLXArray? = nil
    ) -> MLXArray {
        let out = try! model(
            inputs: inputs, inputEmbedding: inputEmbedding, mask: mask, cache: cache,
            positionIDs: positionIDs)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    /// A sparse layer keeps keys, values and the indexer's own keys; a linear layer keeps
    /// its convolution window and recurrent state, plus -- where the layer carries a
    /// per-layer embedding -- that block's own convolution window and token history.
    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        model.layers.map { layer in
            layer.isLinear ? ArraysCache(size: layer.ple != nil ? 4 : 2) : QSAKVCache()
        }
    }

    /// Moves the checkpoint onto the module tree.
    ///
    /// The drafter's tensors ride along in the same directory and belong to a different
    /// module; the n-gram shards are stored one key per shard; the fused expert projection
    /// holds the gate and up halves stacked; and `conv1d` is stored channels-last.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights.filter { !$0.key.hasPrefix("mtp.") }
        if config.tieWordEmbeddings {
            sanitized["lm_head.weight"] = nil
        }

        for layer in 0 ..< config.hiddenLayers {
            let prefix = "model.layers.\(layer).mlp"
            guard let gateUp = sanitized.removeValue(forKey: "\(prefix).experts.gate_up_proj")
            else { continue }
            let midpoint = gateUp.dim(-2) / 2
            sanitized["\(prefix).switch_mlp.gate_proj.weight"] =
                gateUp[.ellipsis, ..<midpoint, 0...]
            sanitized["\(prefix).switch_mlp.up_proj.weight"] =
                gateUp[.ellipsis, midpoint..., 0...]
            sanitized["\(prefix).switch_mlp.down_proj.weight"] = sanitized.removeValue(
                forKey: "\(prefix).experts.down_proj")
        }

        var result: [String: MLXArray] = [:]
        result.reserveCapacity(sanitized.count)
        for (rawKey, rawValue) in sanitized {
            var key = Qwen4ExpLanguageModel.renameNGramShard(rawKey)
            if key.hasPrefix("language_model.") {
                key = String(key.dropFirst("language_model.".count))
            }
            var value = rawValue
            if key.contains("conv1d.weight") && value.dim(-1) != 1 {
                value = value.movedAxis(source: 2, destination: 1)
            }
            result[key] = value
        }
        return result
    }

    /// `...ngram_embedding.shard_7.weight` names the shard in the key; the module holds
    /// the shards in an array.
    static func renameNGramShard(_ key: String) -> String {
        guard let range = key.range(of: ".ngram_embedding.shard_") else { return key }
        let tail = key[range.upperBound...]
        guard let dot = tail.firstIndex(of: "."), Int(tail[..<dot]) != nil else { return key }
        return key[..<range.lowerBound] + ".ngram_embedding.shards." + tail
    }
}

// MARK: - Vision-language model

/// Qwen4-Exp's top-level configuration.
public struct Qwen4ExpConfiguration: Decodable, Sendable {
    public var textConfiguration: Qwen4ExpTextConfiguration
    public var visionConfiguration: Qwen3VLConfiguration.VisionConfiguration?
    public var modelType: String = "qwen4_exp"

    private var _imageTokenID: Int?
    private var _videoTokenID: Int?
    private var _imageTokenIndex: Int?
    private var _videoTokenIndex: Int?
    private var _visionStartTokenID: Int?
    private var _visionEndTokenID: Int?
    private var _vocabularySize: Int?

    public var imageTokenID: Int { _imageTokenID ?? 248_056 }
    public var videoTokenID: Int { _videoTokenID ?? 248_057 }
    public var imageTokenIndex: Int { _imageTokenIndex ?? imageTokenID }
    public var videoTokenIndex: Int { _videoTokenIndex ?? videoTokenID }
    public var visionStartTokenID: Int { _visionStartTokenID ?? 248_053 }
    public var visionEndTokenID: Int { _visionEndTokenID ?? 248_054 }
    public var vocabularySize: Int { _vocabularySize ?? textConfiguration.vocabularySize }
    public var spatialMergeSize: Int { visionConfiguration?.spatialMergeSize ?? 2 }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _imageTokenID = "image_token_id"
        case _videoTokenID = "video_token_id"
        case _imageTokenIndex = "image_token_index"
        case _videoTokenIndex = "video_token_index"
        case _visionStartTokenID = "vision_start_token_id"
        case _visionEndTokenID = "vision_end_token_id"
        case _vocabularySize = "vocab_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        textConfiguration = try c.decode(
            Qwen4ExpTextConfiguration.self, forKey: .textConfiguration)
        visionConfiguration = try c.decodeIfPresent(
            Qwen3VLConfiguration.VisionConfiguration.self, forKey: .visionConfiguration)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen4_exp"
        _imageTokenID = try c.decodeIfPresent(Int.self, forKey: ._imageTokenID)
        _videoTokenID = try c.decodeIfPresent(Int.self, forKey: ._videoTokenID)
        _imageTokenIndex = try c.decodeIfPresent(Int.self, forKey: ._imageTokenIndex)
        _videoTokenIndex = try c.decodeIfPresent(Int.self, forKey: ._videoTokenIndex)
        _visionStartTokenID = try c.decodeIfPresent(Int.self, forKey: ._visionStartTokenID)
        _visionEndTokenID = try c.decodeIfPresent(Int.self, forKey: ._visionEndTokenID)
        _vocabularySize = try c.decodeIfPresent(Int.self, forKey: ._vocabularySize)

        let deepstack = visionConfiguration?.deepstackVisualIndexes ?? []
        guard deepstack.isEmpty else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: c.codingPath,
                    debugDescription:
                        "Qwen4-Exp does not use deepstack visual features, but "
                        + "deepstack_visual_indexes=\(deepstack) was set."))
        }
    }
}

/// Qwen4-Exp: a Qwen3-VL vision tower in front of the Qwen4-Exp text stack.
public class Qwen4Exp: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") var visionModel: Qwen3VLVision.VisionModel?
    @ModuleInfo(key: "language_model") var languageModel: Qwen4ExpLanguageModel

    public let config: Qwen4ExpConfiguration

    public var vocabularySize: Int { config.vocabularySize }
    public var kvHeads: [Int] { languageModel.kvHeads }
    public var loraLayers: [Module] { languageModel.loraLayers }

    public init(_ config: Qwen4ExpConfiguration) {
        self.config = config
        _visionModel.wrappedValue = config.visionConfiguration.map {
            Qwen3VLVision.VisionModel($0)
        }
        _languageModel.wrappedValue = Qwen4ExpLanguageModel(config.textConfiguration)
        super.init()
    }

    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        try languageModel.newCache(configuration: cacheConfiguration)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel.logits(inputs: inputs, inputEmbedding: nil, cache: cache)
    }

    private static let ropeDeltaKey = LMOutput.Key<MLXArray>("qwen4exp.ropeDelta")

    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let tokens = input.tokens.ndim == 1 ? input.tokens[.newAxis, 0...] : input.tokens
        let offset = cache?.first(where: { $0 is QSAKVCache })?.offset ?? 0
        let delta = state?[Self.ropeDeltaKey] ?? MLXArray(Int32(0))
        let positions = broadcast(
            (MLXArray(offset ..< offset + tokens.dim(1)) + delta).reshaped(1, 1, -1),
            to: [3, tokens.dim(0), tokens.dim(1)])
        return LMOutput(
            logits: languageModel.logits(
                inputs: tokens, inputEmbedding: nil, cache: cache, positionIDs: positions),
            state: state)
    }

    public func prepare(
        _ input: LMInput, cache: [KVCache], state: LMOutput.State?,
        prefill: ModelPrefillOptions
    ) throws -> PrepareResult {
        var tokens = input.text.tokens
        if tokens.ndim == 1 {
            tokens = tokens.expandedDimensions(axis: 0)
        }

        let offset = cache.first(where: { $0 is QSAKVCache })?.offset ?? 0
        let anchor = try QwenVL.continuationAnchor(
            model: "Qwen4Exp", key: Self.ropeDeltaKey, cacheOffset: offset,
            batchSize: tokens.dim(0), state: state)
        let (embeddings, localPositions, localDelta) = try visionEmbeddings(input, tokens: tokens)
        let positionIDs =
            localPositions.map { $0 + anchor }
            ?? broadcast(
                MLXArray(anchor ..< anchor + tokens.dim(1)).reshaped(1, 1, -1),
                to: [3, tokens.dim(0), tokens.dim(1)])
        var resumeState = LMOutput.State()
        resumeState[Self.ropeDeltaKey] = (localDelta ?? MLXArray(Int32(0))) + anchor - offset
        let total = embeddings?.dim(1) ?? tokens.dim(1)

        func forward(_ range: Range<Int>) -> MLXArray {
            let positions = positionIDs[0..., 0..., range]
            if let embeddings {
                return languageModel.logits(
                    inputs: tokens[0..., range], inputEmbedding: embeddings[0..., range],
                    cache: cache,
                    positionIDs: positions)
            }
            return languageModel.logits(
                inputs: tokens[0..., range], inputEmbedding: nil, cache: cache,
                positionIDs: positions)
        }

        let processed = try prefill.forEachChunk(total: total) { range in
            _ = forward(range)
            asyncEval(cache)
        }
        if processed > 0 {
            eval(cache)
        }
        prefill.progress?(total, total)
        return .logits(LMOutput(logits: forward(processed ..< total), state: resumeState))
    }

    /// The merged prompt embedding and its three-axis M-RoPE positions, or `nil` for both
    /// when the prompt carries no media -- in which case the text stack lays the positions
    /// out from the cache offset itself.
    private func visionEmbeddings(_ input: LMInput, tokens: MLXArray) throws -> (
        MLXArray?, MLXArray?, MLXArray?
    ) {
        guard let visionModel else {
            guard input.image == nil, input.video == nil else {
                throw VLMError.imageRequired
            }
            return (nil, nil, nil)
        }

        let dtype = visionModel.patchEmbed.proj.weight.dtype
        var pixelParts: [MLXArray] = []
        var imageFrames: [THW]?
        var videoFrames: [THW]?
        if let image = input.image {
            pixelParts.append(image.pixels.asType(dtype))
            imageFrames = image.frames
        }
        if let video = input.video {
            pixelParts.append(video.pixels.asType(dtype))
            videoFrames = video.frames
        }
        let frames = (imageFrames ?? []) + (videoFrames ?? [])
        guard !pixelParts.isEmpty, !frames.isEmpty else { return (nil, nil, nil) }

        let embedded = languageModel.model.embedTokens(tokens)
        let (visionHidden, _) = visionModel(concatenated(pixelParts), gridTHW: frames)
        let merged = MultimodalEmbedding.mergeInputIDsWithImageFeatures(
            inputIDs: tokens, inputEmbeds: embedded,
            imageFeatures: visionHidden.asType(embedded.dtype),
            imageTokenID: config.imageTokenIndex, videoTokenID: config.videoTokenIndex)

        let (positionIDs, delta) = Qwen3VLLanguage.getRopeIndex(
            inputIds: tokens, imageGridTHW: imageFrames, videoGridTHW: videoFrames,
            spatialMergeSize: config.spatialMergeSize, imageTokenId: config.imageTokenIndex,
            videoTokenId: config.videoTokenIndex,
            visionStartTokenId: config.visionStartTokenID)
        return (merged, positionIDs, delta)
    }

    /// Moves the checkpoint onto the module tree.
    ///
    /// Checkpoint keys nest the language model inside `model.`, and the vision tower is
    /// stored as `visual` either alongside it or inside it.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var renamed: [String: MLXArray] = [:]
        renamed.reserveCapacity(weights.count)
        for (rawKey, value) in weights where !rawKey.hasPrefix("mtp.") {
            var key = rawKey
            if key.hasPrefix("model.language_model.visual") {
                key = key.replacingOccurrences(
                    of: "model.language_model.visual", with: "vision_tower")
            } else if key.hasPrefix("model.visual") {
                key = key.replacingOccurrences(of: "model.visual", with: "vision_tower")
            } else if key.hasPrefix("model.language_model") {
                key = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if key.hasPrefix("model.") {
                key = "language_model." + key
            } else if key.hasPrefix("lm_head") {
                key = "language_model." + key
            }
            renamed[key] = value
        }

        var languageWeights: [String: MLXArray] = [:]
        var visionWeights: [String: MLXArray] = [:]
        for (key, value) in renamed {
            if key.hasPrefix("vision_tower.") {
                visionWeights[key] = value
            } else if key.hasPrefix("language_model.") {
                languageWeights[String(key.dropFirst("language_model.".count))] = value
            } else {
                languageWeights[key] = value
            }
        }

        var sanitized = languageModel.sanitize(weights: languageWeights)
            .reduce(into: [String: MLXArray]()) { result, entry in
                result["language_model." + entry.key] = entry.value
            }
        for (key, value) in visionModel?.sanitize(weights: visionWeights) ?? visionWeights {
            sanitized[key] = value
        }
        return sanitized
    }
}

extension Qwen4Exp {
    public var toolCallFormat: ToolCallFormat? { .qwen35 }
    public var reasoningConfig: ReasoningConfig? { QwenReasoningProtocol.tagged }
}

extension Qwen4Exp: ContinuationStateCopying {
    public func copyContinuationState(_ state: LMOutput.State) throws -> LMOutput.State {
        var result = state
        result[Self.ropeDeltaKey] = state[Self.ropeDeltaKey].map { $0[.ellipsis] }
        return result
    }
}
