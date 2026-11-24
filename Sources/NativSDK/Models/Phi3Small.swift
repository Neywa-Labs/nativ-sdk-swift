import Foundation
import MLX
import MLXNN

/// The interleaved, clamped GEGELU activation.
///
/// `x` carries the gate on its even lanes and the linear branch on its odd lanes, so the
/// two are recovered with a stride of two rather than by splitting in half. Both branches
/// are clamped to `limit` -- the gate from above only -- with infinities passed through
/// unchanged.
func phi3SmallGEGELU(_ x: MLXArray, limit: Float) -> MLXArray {
    let gate = x[.ellipsis, .stride(by: 2)]
    let linear = x[.ellipsis, .stride(from: 1, by: 2)]

    let clampedGate = which(isInf(gate), gate, clip(gate, max: limit))
    let clampedLinear = which(isInf(linear), linear, clip(linear, min: -limit, max: limit))

    return (clampedGate * sigmoid(1.702 * clampedGate)) * (clampedLinear + 1.0)
}

class Phi3SmallAttention: Module {

    let heads: Int
    let kvHeads: Int
    let queriesPerKV: Int
    let headDim: Int
    let scale: Float

    let blockSparse: Bool
    let blockSize: Int
    let localBlocks: Int
    let vertStride: Int

    @ModuleInfo(key: "query_key_value") var qkv: Linear
    @ModuleInfo(key: "dense") var dense: Linear

    let rope: RoPE

    init(_ args: Phi3SmallConfiguration, layerIndex: Int) {
        let dim = args.hiddenSize
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.queriesPerKV = args.attentionHeads / args.kvHeads
        self.headDim = args.hiddenSize / args.attentionHeads

        let normFactor =
            args.mupUseScaling
            ? Float(headDim) / args.mupAttnMultiplier : sqrt(Float(headDim))
        self.scale = 1.0 / normFactor

        _qkv.wrappedValue = Linear(
            dim, (heads + 2 * kvHeads) * headDim, bias: true)
        _dense.wrappedValue = Linear(dim, dim, bias: true)

        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: args.ropeEmbeddingBase,
            scale: args.ropePositionScale)

        self.blockSparse = layerIndex % args.denseAttentionEveryNLayers == 0
        if blockSparse {
            precondition(
                args.blocksparseBlockSize == 32 || args.blocksparseBlockSize == 64,
                "phi3small supports block sizes 32 and 64, got \(args.blocksparseBlockSize)")
        }
        self.blockSize = args.blocksparseBlockSize
        self.localBlocks = args.blocksparseNumLocalBlocks
        self.vertStride = args.blocksparseVertStride
    }

    /// The dense expansion of the block-sparse pattern, shaped
    /// `(kvHeads, queriesPerKV, queryLength, keyLength)`.
    ///
    /// Query blocks are aligned to the *end* of the key axis, so a decode step scores
    /// against the pattern its absolute position implies rather than block zero's.
    func blockSparseMask(queryLength: Int, keyLength: Int) -> MLXArray {
        let queries = MLXArray((keyLength - queryLength) ..< keyLength).floorDivide(blockSize)
            .reshaped(1, queryLength, 1)
        let keys = MLXArray(0 ..< keyLength).floorDivide(blockSize).reshaped(1, 1, keyLength)
        let head = MLXArray(1 ... heads).reshaped(heads, 1, 1)
        let vertical = (keys + head) % vertStride .== 0
        return logicalAnd(
            queries .>= keys,
            logicalOr(queries - keys .< localBlocks, vertical)
        )
        .reshaped(kvHeads, queriesPerKV, queryLength, keyLength)
    }

    func blockSparseAttention(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let B = queries.dim(0)
        let L = queries.dim(2)

        let q = (scale * queries).reshaped(B, kvHeads, queriesPerKV, L, -1)
        let k = keys.expandedDimensions(axis: 2)
        let v = values.expandedDimensions(axis: 2)

        var scores = matmul(q, k.swappedAxes(-1, -2))
        scores = applyAttentionMask(scores: scores, mask: mask)

        let pattern = blockSparseMask(queryLength: L, keyLength: keys.dim(-2))
        scores =
            scores
            + which(
                pattern, MLXArray(0).asType(scores.dtype),
                MLXArray(-Float.infinity).asType(scores.dtype))
        scores = softmax(scores, axis: -1, precise: true)

        return matmul(scores, v).reshaped(B, heads, L, -1)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let fused = qkv(x).reshaped(B, L, kvHeads, queriesPerKV + 2, headDim)
        var queries =
            fused[.ellipsis, ..<queriesPerKV, 0...]
            .flattened(start: 2, end: 3)
            .transposed(0, 2, 1, 3)
        var keys = fused[.ellipsis, queriesPerKV, 0...].transposed(0, 2, 1, 3)
        let values = fused[.ellipsis, queriesPerKV + 1, 0...].transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        let output: MLXArray
        if blockSparse {
            let (cachedKeys, cachedValues) =
                cache?.update(keys: keys, values: values) ?? (keys, values)
            output = blockSparseAttention(
                queries: queries, keys: cachedKeys, values: cachedValues, mask: mask)
        } else {
            output = attentionWithCacheUpdate(
                queries: queries, keys: keys, values: values, cache: cache,
                scale: scale, mask: mask)
        }

        return dense(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

class Phi3SmallMLP: Module {

    let gegeluLimit: Float

    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ args: Phi3SmallConfiguration) {
        self.gegeluLimit = args.gegeluLimit
        _upProj.wrappedValue = Linear(
            args.hiddenSize, 2 * args.ffIntermediateSize, bias: true)
        _downProj.wrappedValue = Linear(
            args.ffIntermediateSize, args.hiddenSize, bias: true)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(phi3SmallGEGELU(upProj(x), limit: gegeluLimit))
    }
}

class Phi3SmallBlock: Module {

    @ModuleInfo(key: "self_attn") var selfAttention: Phi3SmallAttention
    @ModuleInfo(key: "mlp") var mlp: Phi3SmallMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: LayerNorm

    var blockSparse: Bool { selfAttention.blockSparse }

    init(_ args: Phi3SmallConfiguration, layerIndex: Int) {
        _selfAttention.wrappedValue = Phi3SmallAttention(args, layerIndex: layerIndex)
        _mlp.wrappedValue = Phi3SmallMLP(args)
        _inputLayerNorm.wrappedValue = LayerNorm(
            dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
        _postAttentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + selfAttention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class Phi3SmallModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [Phi3SmallBlock]

    @ModuleInfo(key: "final_layernorm") var finalLayerNorm: LayerNorm

    let embeddingMultiplier: Float

    public init(_ args: Phi3SmallConfiguration) {
        precondition(args.vocabularySize > 0)
        self.embeddingMultiplier = args.mupEmbeddingMultiplier
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { Phi3SmallBlock(args, layerIndex: $0) }
        _finalLayerNorm.wrappedValue = LayerNorm(
            dimensions: args.hiddenSize, eps: args.layerNormEpsilon)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        if embeddingMultiplier != 0 {
            h = embeddingMultiplier * h
        }

        let firstSparse = layers.firstIndex { $0.blockSparse }
        let firstDense = layers.firstIndex { !$0.blockSparse }
        let sparseMask =
            firstSparse.map {
                createAttentionMask(h: h, cache: cache?[$0], returnArray: true)
            } ?? .none
        let denseMask = firstDense.map { createAttentionMask(h: h, cache: cache?[$0]) } ?? .none

        for (i, layer) in layers.enumerated() {
            h = layer(
                h, mask: layer.blockSparse ? sparseMask : denseMask, cache: cache?[i])
        }

        return finalLayerNorm(h)
    }
}

public class Phi3SmallModel: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: Phi3SmallModelInner

    let widthMultiplier: Float

    public init(_ args: Phi3SmallConfiguration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: args.hiddenLayers)
        self.widthMultiplier = args.mupWidthMultiplier
        self.model = Phi3SmallModelInner(args)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model.embedTokens.asLinear(model(inputs, cache: cache))
        if widthMultiplier != 0 {
            return out / widthMultiplier
        }
        return out
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        if let capacity = configuration?.capacity {
            throw KVCacheConfigurationError.incompatibleCapacity(
                expected: capacity.maxTokens, count: model.layers.filter(\.blockSparse).count)
        }
        return try model.layers.map { layer in
            let cache = try makeAttentionKVCache(configuration: configuration)
            return layer.blockSparse ? MaterializedKVCache(cache) : cache
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { !$0.key.contains("self_attn.rotary_emb.inv_freq") }
    }
}

public struct Phi3SmallConfiguration: Decodable, Sendable {

    var modelType: String
    var hiddenSize: Int
    var denseAttentionEveryNLayers: Int
    var ffIntermediateSize: Int
    var gegeluLimit: Float
    var hiddenLayers: Int
    var attentionHeads: Int
    var layerNormEpsilon: Float
    var vocabularySize: Int
    var kvHeads: Int
    var mupAttnMultiplier: Float
    var mupUseScaling: Bool
    var mupEmbeddingMultiplier: Float
    var mupWidthMultiplier: Float
    var ropeEmbeddingBase: Float
    var ropePositionScale: Float
    var blocksparseBlockSize: Int
    var blocksparseNumLocalBlocks: Int
    var blocksparseVertStride: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case denseAttentionEveryNLayers = "dense_attention_every_n_layers"
        case ffIntermediateSize = "ff_intermediate_size"
        case gegeluLimit = "gegelu_limit"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case layerNormEpsilon = "layer_norm_epsilon"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case mupAttnMultiplier = "mup_attn_multiplier"
        case mupUseScaling = "mup_use_scaling"
        case mupEmbeddingMultiplier = "mup_embedding_multiplier"
        case mupWidthMultiplier = "mup_width_multiplier"
        case ropeEmbeddingBase = "rope_embedding_base"
        case ropePositionScale = "rope_position_scale"
        case blocksparseBlockSize = "blocksparse_block_size"
        case blocksparseNumLocalBlocks = "blocksparse_num_local_blocks"
        case blocksparseVertStride = "blocksparse_vert_stride"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "phi3small"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        denseAttentionEveryNLayers = try c.decode(
            Int.self, forKey: .denseAttentionEveryNLayers)
        ffIntermediateSize = try c.decode(Int.self, forKey: .ffIntermediateSize)
        gegeluLimit = try c.decode(Float.self, forKey: .gegeluLimit)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        layerNormEpsilon = try c.decode(Float.self, forKey: .layerNormEpsilon)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        mupAttnMultiplier = try c.decodeIfPresent(Float.self, forKey: .mupAttnMultiplier) ?? 1.0
        mupUseScaling = try c.decodeIfPresent(Bool.self, forKey: .mupUseScaling) ?? true
        mupEmbeddingMultiplier =
            try c.decodeIfPresent(Float.self, forKey: .mupEmbeddingMultiplier) ?? 10.0
        mupWidthMultiplier =
            try c.decodeIfPresent(Float.self, forKey: .mupWidthMultiplier) ?? 8.0
        ropeEmbeddingBase =
            try c.decodeIfPresent(Float.self, forKey: .ropeEmbeddingBase) ?? 1_000_000
        ropePositionScale = try c.decodeIfPresent(Float.self, forKey: .ropePositionScale) ?? 1.0
        blocksparseBlockSize =
            try c.decodeIfPresent(Int.self, forKey: .blocksparseBlockSize) ?? 64
        blocksparseNumLocalBlocks =
            try c.decodeIfPresent(Int.self, forKey: .blocksparseNumLocalBlocks) ?? 16
        blocksparseVertStride =
            try c.decodeIfPresent(Int.self, forKey: .blocksparseVertStride) ?? 8
    }
}

extension Phi3SmallModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
