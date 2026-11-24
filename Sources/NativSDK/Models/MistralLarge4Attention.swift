import Foundation
import MLX
import MLXNN

/// Stores one compressed KV history at each source layer, and one key history
/// at each full indexer. Readers advance their own logical offset.
final class MistralLarge4Cache: BaseKVCache {
    private var latents: KVCacheSimple?
    private var indexKeys: KVCacheSimple?

    init(ownsKV: Bool, ownsIndexer: Bool) {
        latents = ownsKV ? KVCacheSimple() : nil
        indexKeys = ownsIndexer ? KVCacheSimple() : nil
        super.init()
    }

    override var capabilities: CacheCapabilities { [.attention, .snapshot, .fullRewind] }
    override var isTrimmable: Bool { true }

    private func append(_ data: MLXArray, to cache: KVCacheSimple?) -> MLXArray {
        guard let cache else { return data }
        let empty = MLXArray.zeros([data.dim(0), 1, data.dim(2), 0], dtype: data.dtype)
        return cache.update(keys: data, values: empty).0
    }

    func appendLatents(_ data: MLXArray) -> MLXArray { append(data, to: latents) }
    func appendIndexerKeys(_ data: MLXArray) -> MLXArray { append(data, to: indexKeys) }

    override func innerState() -> [MLXArray] {
        [latents?.keys, indexKeys?.keys].compactMap { $0 }
    }

    override var state: [MLXArray] {
        get { [latents?.state.first, indexKeys?.state.first].compactMap { $0 } }
        set {
            var entries = newValue.makeIterator()
            for cache in [latents, indexKeys].compactMap({ $0 }) {
                guard let keys = entries.next() else {
                    cache.keys = nil
                    cache.values = nil
                    cache.offset = 0
                    continue
                }
                cache.state = [
                    keys, MLXArray.zeros([keys.dim(0), 1, keys.dim(2), 0], dtype: keys.dtype),
                ]
            }
            offset = newValue.first?.dim(2) ?? 0
        }
    }

    override var metaState: [String] {
        get { [String(offset)] }
        set { if let value = newValue.first.flatMap(Int.init) { offset = value } }
    }

    @discardableResult
    override func trim(_ n: Int) -> Int {
        let count = min(offset, n)
        latents?.trim(count)
        indexKeys?.trim(count)
        offset -= count
        return count
    }

    override func copy() -> any KVCache {
        let cache = MistralLarge4Cache(ownsKV: false, ownsIndexer: false)
        cache.latents = latents?.copy() as? KVCacheSimple
        cache.indexKeys = indexKeys?.copy() as? KVCacheSimple
        cache.offset = offset
        return cache
    }
}

final class MistralLarge4LeRoPE: Module {
    @ParameterInfo var alpha: MLXArray
    private let baseFrequencies: [Float]

    init(_ config: MistralLarge4Configuration.Text) {
        let dim = Double(config.ropeSize)
        let base = Double(config.ropeBase)
        let origin = Double(config.ropeOriginal)
        func correction(_ rotations: Float) -> Double {
            dim * log(origin / (Double(rotations) * 2 * .pi)) / (2 * log(base))
        }
        let low = max(floor(correction(config.ropeFast)), 0)
        let high = min(ceil(correction(config.ropeSlow)), dim - 1)
        let factor = config.ropeFactor
        baseFrequencies = (0 ..< config.ropeSize / 2).map { index in
            let blend = Float(min(1, max(0, (Double(index) - low) / max(high - low, 0.001))))
            return Float(pow(base, -2 * Double(index) / dim)) * ((1 - blend) + blend / factor)
        }
        _alpha.wrappedValue = MLXArray.zeros([config.ropeSize / 2])
    }

    func callAsFunction(_ positions: MLXArray, dtype: DType) -> MistralLarge4Rotation {
        let angles =
            positions.expandedDimensions(axis: -1).asType(.float32)
            * MLXArray(baseFrequencies) * exp(alpha.asType(.float32))
        return .init(cosine: cos(angles).asType(dtype), sine: sin(angles).asType(dtype))
    }
}

struct MistralLarge4Rotation {
    let cosine: MLXArray
    let sine: MLXArray

    func apply(_ x: MLXArray) -> MLXArray {
        let width = x.dim(-1)
        let even = x[.ellipsis, stride(from: 0, to: width, by: 2)]
        let odd = x[.ellipsis, stride(from: 1, to: width, by: 2)]
        // Mistral stores paired inputs adjacently and returns the two halves separately.
        return concatenated([even * cosine - odd * sine, odd * cosine + even * sine], axis: -1)
    }

    var perHead: Self {
        .init(cosine: cosine.expandedDimensions(axis: 1), sine: sine.expandedDimensions(axis: 1))
    }
}

private final class MistralLarge4SoftmaxScaling: Module {
    @ParameterInfo var scale: MLXArray
    @ParameterInfo var bias: MLXArray

    init(heads: Int) {
        _scale.wrappedValue = MLXArray.zeros([heads])
        _bias.wrappedValue = MLXArray.ones([heads])
    }

    func callAsFunction(_ q: MLXArray, positions: MLXArray) -> MLXArray {
        let positionScale = log(positions.asType(.float32) + 1).asType(q.dtype)
        let factors =
            scale.asType(q.dtype).reshaped(1, -1, 1)
            * positionScale.expandedDimensions(axis: 1) + bias.asType(q.dtype).reshaped(1, -1, 1)
        return q * factors.expandedDimensions(axis: -1)
    }
}

private final class MistralLarge4Indexer: Module {
    @ModuleInfo var wk: Linear
    @ModuleInfo(key: "wq_b") var query: Linear
    @ModuleInfo(key: "weights_proj") var weights: Linear
    @ModuleInfo(key: "q_input_norm") var queryNorm: LayerNorm
    @ModuleInfo(key: "k_norm") var keyNorm: LayerNorm
    @ModuleInfo(key: "softmax_scaling") var scaling: MistralLarge4SoftmaxScaling

    let heads: Int
    let width: Int
    let ropeWidth: Int
    let budget: Int

    init(_ config: MistralLarge4Configuration.Text, layer: Int) {
        heads = config.indexHeads
        width = config.indexSize
        ropeWidth = config.rotary[layer] ? config.ropeSize : 0
        budget = config.indexTopK
        _wk.wrappedValue = Linear(config.hiddenSize, width, bias: false)
        _query.wrappedValue = Linear(config.hiddenSize, heads * width, bias: false)
        _weights.wrappedValue = Linear(config.hiddenSize, heads, bias: false)
        _queryNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
        _keyNorm.wrappedValue = LayerNorm(dimensions: width, eps: 1e-6)
        _scaling.wrappedValue = MistralLarge4SoftmaxScaling(heads: heads)
    }

    private func rotate(_ x: MLXArray, _ rotation: MistralLarge4Rotation) -> MLXArray {
        guard ropeWidth > 0 else { return x }
        return concatenated(
            [rotation.apply(x[.ellipsis, ..<ropeWidth]), x[.ellipsis, ropeWidth...]], axis: -1)
    }

    func callAsFunction(
        _ x: MLXArray, positions: MLXArray, rotation: MistralLarge4Rotation,
        cache: MistralLarge4Cache?, causal: MLXArray, keyLength: Int
    ) -> MLXArray? {
        let key = rotate(keyNorm(wk(x)), rotation).expandedDimensions(axis: 1)
        let keys = cache?.appendIndexerKeys(key) ?? key
        guard keyLength > budget else { return nil }

        let query = self.query(queryNorm(x)).reshaped(x.dim(0), x.dim(1), heads, width)
            .transposed(0, 2, 1, 3)
        let scaled = rotate(scaling(query, positions: positions), rotation.perHead)
        let mixing = weights(x).transposed(0, 2, 1).asType(.float32) * pow(Float(heads), -0.5)
        let dots =
            scaled.asType(.float32).matmul(keys.asType(.float32).swappedAxes(-1, -2))
            * pow(Float(width), -0.5)
        let scores = (maximum(dots, 0) * mixing.expandedDimensions(axis: -1))
            .sum(axis: 1, keepDims: true)
        let masked = MLX.where(causal, scores, -Float.infinity)
        return argPartition(-masked, kth: budget - 1, axis: -1)[.ellipsis, ..<budget]
    }
}

struct MistralLarge4AttentionContext {
    let positions: MLXArray
    let rotation: MistralLarge4Rotation
    let causal: MLXArray
    var sharedKV: [Int: MLXArray] = [:]
    var selected: MLXArray?
}

final class MistralLarge4Attention: Module {
    @ModuleInfo(key: "q_proj") var query: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var latentProjection: Linear?
    @ModuleInfo(key: "kv_a_layernorm") var latentNorm: RMSNorm?
    @ModuleInfo(key: "embed_q") var embedQuery: MultiLinear
    @ModuleInfo(key: "unembed_out") var unembedOutput: MultiLinear
    @ModuleInfo(key: "softmax_scaling") private var scaling: MistralLarge4SoftmaxScaling
    @ModuleInfo private var indexer: MistralLarge4Indexer?

    private let source: Int
    private let sharedLayer: Int?
    private let heads: Int
    private let latentWidth: Int
    private let contentWidth: Int
    private let ropeWidth: Int
    private let queryWidth: Int
    private let attentionScale: Float

    init(_ config: MistralLarge4Configuration.Text, layer: Int) {
        source = config.kvSources[layer]
        sharedLayer =
            source == layer && config.kvSources.lastIndex(of: layer) != layer ? layer : nil
        heads = config.heads
        latentWidth = config.latentSize
        queryWidth = config.contentSize + config.ropeSize
        ropeWidth = config.rotary[layer] ? config.ropeSize : 0
        contentWidth = queryWidth - ropeWidth
        attentionScale = pow(Float(queryWidth), -0.5)
        _query.wrappedValue = Linear(config.hiddenSize, heads * queryWidth, bias: false)
        _output.wrappedValue = Linear(heads * config.valueSize, config.hiddenSize, bias: false)
        if source == layer {
            _latentProjection.wrappedValue = Linear(
                config.hiddenSize, latentWidth + ropeWidth, bias: false)
            _latentNorm.wrappedValue = RMSNorm(dimensions: latentWidth, eps: config.eps)
        }
        _embedQuery.wrappedValue = MultiLinear(
            inputDimensions: contentWidth, outputDimensions: latentWidth, numHeads: heads)
        _unembedOutput.wrappedValue = MultiLinear(
            inputDimensions: latentWidth, outputDimensions: config.valueSize, numHeads: heads)
        _scaling.wrappedValue = MistralLarge4SoftmaxScaling(heads: heads)
        if config.fullIndexers[layer] {
            _indexer.wrappedValue = MistralLarge4Indexer(config, layer: layer)
        }
    }

    func callAsFunction(
        _ x: MLXArray, cache: MistralLarge4Cache?, context: inout MistralLarge4AttentionContext
    ) -> MLXArray {
        let (batch, length) = (x.dim(0), x.dim(1))
        let rotation = context.rotation.perHead
        var q = query(x).reshaped(batch, length, heads, queryWidth).transposed(0, 2, 1, 3)
        if ropeWidth > 0 {
            q = concatenated(
                [q[.ellipsis, ..<contentWidth], rotation.apply(q[.ellipsis, contentWidth...])],
                axis: -1)
        }
        q = scaling(q, positions: context.positions)

        var history: MLXArray
        if let latentProjection, let latentNorm {
            let projected = latentProjection(x).expandedDimensions(axis: 1)
            let latent = latentNorm(projected[.ellipsis, ..<latentWidth])
            let data =
                ropeWidth == 0
                ? latent
                : concatenated(
                    [latent, rotation.apply(projected[.ellipsis, latentWidth...])], axis: -1)
            history = cache?.appendLatents(data) ?? data
            if let sharedLayer { context.sharedKV[sharedLayer] = history }
        } else {
            history = context.sharedKV[source]!
        }

        var mask = context.causal
        if let indexer {
            context.selected = indexer(
                x, positions: context.positions, rotation: context.rotation, cache: cache,
                causal: mask, keyLength: history.dim(2))
        }
        if let selected = context.selected {
            if length == 1 {
                let indices = sorted(selected, axis: -1)
                history = takeAlong(history, indices.transposed(0, 1, 3, 2), axis: 2)
                mask = takeAlong(mask, indices, axis: -1)
            } else {
                let sparse = putAlong(
                    MLXArray.zeros(mask.shape, dtype: .bool), selected, values: MLXArray(true),
                    axis: -1)
                mask = logicalAnd(mask, sparse)
            }
        }

        let latent = history[.ellipsis, ..<latentWidth]
        let attended: MLXArray
        if length == 1 {
            let foldedQuery = embedQuery(q[.ellipsis, ..<contentWidth]).asType(.float32)
            let key = latent.asType(.float32)
            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
            if ropeWidth > 0 {
                let positional =
                    q[.ellipsis, contentWidth...].asType(.float32).matmul(
                        history[.ellipsis, latentWidth...].asType(.float32).swappedAxes(-1, -2))
                    * attentionScale
                attentionMask = .array(MLX.where(mask, positional, -Float.infinity))
            } else {
                attentionMask = .array(mask)
            }
            let mixed = MLXFast.scaledDotProductAttention(
                queries: foldedQuery, keys: key, values: key, scale: attentionScale,
                mask: attentionMask)
            attended = unembedOutput(mixed.asType(x.dtype))
        } else {
            var keys = embedQuery(latent, transpose: false)
            if ropeWidth > 0 {
                let positional = broadcast(
                    history[.ellipsis, latentWidth...],
                    to: [batch, heads, history.dim(2), ropeWidth])
                keys = concatenated([keys, positional], axis: -1)
            }
            let values = unembedOutput(latent)
            attended = MLXFast.scaledDotProductAttention(
                queries: q, keys: keys, values: values, scale: attentionScale, mask: .array(mask))
        }
        cache?.offset += length
        return output(attended.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}
