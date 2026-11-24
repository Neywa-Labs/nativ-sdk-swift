import Foundation
import MLX
import MLXNN

/// Either a dense linear or a fixed-sparsity one, sharing a checkpoint key.
protocol Z1TLinear: Module, UnaryLayer {}

extension Linear: Z1TLinear {}

/// `y[o] = sum_k weight[o, k] * x[indices[o, k]] + bias[o]`.
///
/// The fan-in table is a loaded int parameter, not a trained one, so it comes from the
/// checkpoint alongside the weights.
class Z1TSparseLinear: Module, Z1TLinear {

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    @ParameterInfo(key: "indices") var indices: MLXArray

    /// Ceiling on the transient `[batch, chunk, out, k]` gather, so a long prefill stays in
    /// memory. Splitting along the sequence axis is exact.
    private static let chunkElements = 8_000_000

    init(inputDimensions: Int, outputDimensions: Int, fanIn: Int) {
        _weight.wrappedValue = MLXArray.zeros([outputDimensions, fanIn])
        _bias.wrappedValue = MLXArray.zeros([outputDimensions])
        _indices.wrappedValue = MLXArray.zeros([outputDimensions, fanIn], dtype: .int32)
        super.init()
    }

    private func gathered(_ x: MLXArray) -> MLXArray {
        (take(x, indices.asType(.int32), axis: -1) * weight).sum(axis: -1) + bias
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let outputs = weight.dim(0)
        guard x.ndim == 3, x.dim(1) * outputs > Self.chunkElements else {
            return gathered(x)
        }
        let step = Swift.max(1, Self.chunkElements / outputs)
        var parts: [MLXArray] = []
        var start = 0
        while start < x.dim(1) {
            let end = Swift.min(start + step, x.dim(1))
            parts.append(gathered(x[0..., start ..< end]))
            start = end
        }
        return concatenated(parts, axis: 1)
    }
}

func z1tLinear(inputDimensions: Int, outputDimensions: Int, fanIn: Int?) -> Z1TLinear {
    let k = fanIn.map { Swift.max(1, Swift.min($0, inputDimensions)) } ?? inputDimensions
    if k >= inputDimensions {
        return Linear(inputDimensions, outputDimensions)
    }
    return Z1TSparseLinear(
        inputDimensions: inputDimensions, outputDimensions: outputDimensions, fanIn: k)
}

/// `weight * tanh(alpha * x) + bias`, in place of a normalization.
class Z1TDynamicTanh: Module, UnaryLayer {

    @ParameterInfo(key: "alpha") var alpha: MLXArray
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(dimensions: Int, alpha alphaInit: Float) {
        _alpha.wrappedValue = MLXArray([alphaInit])
        _weight.wrappedValue = MLXArray.ones([dimensions])
        _bias.wrappedValue = MLXArray.zeros([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        weight * tanh(alpha * x) + bias
    }
}

/// Attention-free token mixing: a causal depthwise convolution over `aft_ksize` taps plus a
/// causal cumulative pool over everything before it.
///
/// There are no keys or values to cache. What carries between calls is the pool's running
/// sums and the convolution's `ksize - 1` tail, which is what makes a decode step cost the
/// same at any context length.
class Z1TAFTConv: Module {

    let dimensions: Int
    let heads: Int
    let kernelSize: Int
    let tanhLinear: Bool

    @ModuleInfo(key: "qkv_proj") var qkvProj: Z1TLinear
    @ModuleInfo(key: "out_proj") var outProj: Z1TLinear
    @ParameterInfo(key: "w_conv") var wConv: MLXArray

    init(_ args: Z1TConfiguration) {
        precondition(
            args.hiddenSize % args.aftHeads == 0,
            "hidden_size must divide by aft_heads")
        self.dimensions = args.hiddenSize
        self.heads = args.aftHeads
        self.kernelSize = args.aftKernelSize
        self.tanhLinear = args.tanhLinear

        let fan = args.fanIn(site: .attention)
        _qkvProj.wrappedValue = z1tLinear(
            inputDimensions: dimensions, outputDimensions: 2 * dimensions + heads,
            fanIn: fan)
        _outProj.wrappedValue = z1tLinear(
            inputDimensions: dimensions, outputDimensions: dimensions, fanIn: fan)
        _wConv.wrappedValue = MLXArray.zeros([heads, kernelSize])
        super.init()
    }

    private struct State {
        var pooledValues: MLXArray
        var pooledKeys: MLXArray
        var windowValues: MLXArray
        var windowKeys: MLXArray
    }

    private func emptyState(batch: Int, dtype: DType) -> State {
        let headDim = dimensions / heads
        return State(
            pooledValues: MLXArray.zeros([batch, heads, headDim], dtype: dtype),
            pooledKeys: MLXArray.zeros([batch, heads], dtype: dtype),
            windowValues: MLXArray.zeros(
                [batch, kernelSize - 1, heads, headDim], dtype: dtype),
            windowKeys: MLXArray.zeros([batch, kernelSize - 1, heads], dtype: dtype))
    }

    private func state(from cache: ArraysCache?, batch: Int, dtype: DType) -> State {
        guard let cache, let pooledValues = cache[0], let pooledKeys = cache[1],
            let windowValues = cache[2], let windowKeys = cache[3]
        else {
            return emptyState(batch: batch, dtype: dtype)
        }
        return State(
            pooledValues: pooledValues, pooledKeys: pooledKeys,
            windowValues: windowValues, windowKeys: windowKeys)
    }

    private func store(_ state: State, into cache: ArraysCache?, advancing positions: Int) {
        guard let cache else { return }
        cache[0] = state.pooledValues
        cache[1] = state.pooledKeys
        if kernelSize > 1 {
            cache[2] = state.windowValues
            cache[3] = state.windowKeys
        }
        cache.offset += positions
        eval(cache.innerState())
    }

    func callAsFunction(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        if let cache, x.dim(1) == 1 {
            return decodeStep(x, cache: cache)
        }

        let (batch, length) = (x.dim(0), x.dim(1))
        let headDim = dimensions / heads

        var mixed = qkvProj(x)
        if tanhLinear { mixed = tanh(mixed) }
        let queries = mixed[.ellipsis, ..<dimensions].reshaped(batch, length, heads, headDim)
        let values = mixed[.ellipsis, dimensions ..< (2 * dimensions)]
            .reshaped(batch, length, heads, headDim)
        let keys = exp(tanh(mixed[.ellipsis, (2 * dimensions)...]))

        let kernel = exp(wConv) - 1
        let keyedValues = keys[.ellipsis, .newAxis] * values

        var carried = state(from: cache, batch: batch, dtype: x.dtype)

        let extendedValues = concatenated([carried.windowValues, keyedValues], axis: 1)
        let extendedKeys = concatenated([carried.windowKeys, keys], axis: 1)

        var convolvedValues = MLXArray.zeros(
            [batch, length, heads, headDim], dtype: x.dtype)
        var convolvedKeys = MLXArray.zeros([batch, length, heads], dtype: x.dtype)
        for tap in 0 ..< kernelSize {
            convolvedValues =
                convolvedValues
                + kernel[0..., tap].reshaped(1, 1, heads, 1)
                * extendedValues[0..., tap ..< (tap + length)]
            convolvedKeys =
                convolvedKeys
                + kernel[0..., tap].reshaped(1, 1, heads)
                * extendedKeys[0..., tap ..< (tap + length)]
        }

        let pooledValues =
            carried.pooledValues.expandedDimensions(axis: 1) + cumsum(keyedValues, axis: 1)
        let pooledKeys =
            carried.pooledKeys.expandedDimensions(axis: 1) + cumsum(keys, axis: 1)

        let numerator = convolvedValues + pooledValues
        let denominator = convolvedKeys + pooledKeys
        let context = numerator / (denominator[.ellipsis, .newAxis] + 1e-6)
        let mixedOut = (tanh(queries) * context).reshaped(batch, length, dimensions)

        var out = outProj(mixedOut)
        if tanhLinear { out = tanh(out) }

        carried.pooledValues = carried.pooledValues + keyedValues.sum(axis: 1)
        carried.pooledKeys = carried.pooledKeys + keys.sum(axis: 1)
        if kernelSize > 1 {
            carried.windowValues = extendedValues[0..., (1 - kernelSize)...]
            carried.windowKeys = extendedKeys[0..., (1 - kernelSize)...]
        }
        store(carried, into: cache, advancing: length)
        return out
    }

    /// One token, with no length-one time axis anywhere.
    ///
    /// Keeping the tensors time-axis-free is load-bearing rather than tidy: MLX miscompiles
    /// some fused kernels over a length-one sequence axis, which upstream notes at the same
    /// place.
    private func decodeStep(_ x: MLXArray, cache: ArraysCache) -> MLXArray {
        let batch = x.dim(0)
        let headDim = dimensions / heads

        var mixed = qkvProj(x[0..., 0])
        if tanhLinear { mixed = tanh(mixed) }
        let queries = mixed[0..., ..<dimensions].reshaped(batch, heads, headDim)
        let values = mixed[0..., dimensions ..< (2 * dimensions)]
            .reshaped(batch, heads, headDim)
        let keys = exp(tanh(mixed[0..., (2 * dimensions)...]))
        let keyedValues = keys[.ellipsis, .newAxis] * values
        let kernel = exp(wConv) - 1

        var carried = state(from: cache, batch: batch, dtype: x.dtype)

        var numerator = carried.pooledValues + keyedValues
        var denominator = carried.pooledKeys + keys
        for tap in 0 ..< kernelSize {
            let tapValues =
                tap < kernelSize - 1 ? carried.windowValues[0..., tap] : keyedValues
            let tapKeys = tap < kernelSize - 1 ? carried.windowKeys[0..., tap] : keys
            numerator = numerator + kernel[0..., tap].reshaped(1, heads, 1) * tapValues
            denominator = denominator + kernel[0..., tap].reshaped(1, heads) * tapKeys
        }
        let context = numerator / (denominator[.ellipsis, .newAxis] + 1e-6)
        let mixedOut = (tanh(queries) * context).reshaped(batch, dimensions)

        var out = outProj(mixedOut)
        if tanhLinear { out = tanh(out) }

        carried.pooledValues = carried.pooledValues + keyedValues
        carried.pooledKeys = carried.pooledKeys + keys
        if kernelSize > 1 {
            carried.windowValues = concatenated(
                [
                    carried.windowValues[0..., 1...],
                    keyedValues.expandedDimensions(axis: 1),
                ], axis: 1)
            carried.windowKeys = concatenated(
                [carried.windowKeys[0..., 1...], keys.expandedDimensions(axis: 1)],
                axis: 1)
        }
        store(carried, into: cache, advancing: 1)
        return out[0..., .newAxis]
    }
}

class Z1TMLP: Module, UnaryLayer {

    @ModuleInfo(key: "proj1") var proj1: Z1TLinear
    @ModuleInfo(key: "proj2") var proj2: Z1TLinear

    let tanhLinear: Bool
    let tanhActivation: Bool

    init(_ args: Z1TConfiguration) {
        let hidden = Int(Float(args.hiddenSize) * 4)
        let fan = args.fanIn(site: .mlp)
        _proj1.wrappedValue = z1tLinear(
            inputDimensions: args.hiddenSize, outputDimensions: hidden, fanIn: fan)
        _proj2.wrappedValue = z1tLinear(
            inputDimensions: hidden, outputDimensions: args.hiddenSize, fanIn: fan)
        self.tanhLinear = args.tanhLinear
        self.tanhActivation = args.tanhMLP
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = proj1(x)
        if tanhLinear { h = tanh(h) }
        h = tanhActivation ? tanh(h) : silu(h)
        var out = proj2(h)
        if tanhLinear { out = tanh(out) }
        return out
    }
}

class Z1TBlock: Module {

    @ModuleInfo(key: "norm1") var norm1: Z1TDynamicTanh
    @ModuleInfo(key: "attn") var attention: Z1TAFTConv
    @ModuleInfo(key: "norm2") var norm2: Z1TDynamicTanh
    @ModuleInfo(key: "mlp") var mlp: Z1TMLP

    init(_ args: Z1TConfiguration) {
        _norm1.wrappedValue = Z1TDynamicTanh(
            dimensions: args.hiddenSize, alpha: args.dynamicTanhAlpha)
        _attention.wrappedValue = Z1TAFTConv(args)
        _norm2.wrappedValue = Z1TDynamicTanh(
            dimensions: args.hiddenSize, alpha: args.dynamicTanhAlpha)
        _mlp.wrappedValue = Z1TMLP(args)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let h = x + attention(norm1(x), cache: cache)
        return h + mlp(norm2(h))
    }
}

public class Z1TModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [Z1TBlock]

    @ModuleInfo(key: "norm") var norm: Z1TDynamicTanh

    /// Additive sinusoidal positions, initialized to the closed form and then loaded from
    /// the checkpoint, which is where a trained table would come from.
    @ParameterInfo(key: "pe") var positionalEncoding: MLXArray

    public init(_ args: Z1TConfiguration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { _ in Z1TBlock(args) }
        _norm.wrappedValue = Z1TDynamicTanh(
            dimensions: args.hiddenSize, alpha: args.dynamicTanhAlpha)
        _positionalEncoding.wrappedValue = Self.sinusoidalTable(
            dimensions: args.hiddenSize, length: args.maxPositionEmbeddings)
        super.init()
    }

    static func sinusoidalTable(dimensions: Int, length: Int, omega: Float = 1e4)
        -> MLXArray
    {
        let half = dimensions / 2
        let frequencies = exp(
            -log(omega) * MLXArray(0 ..< half).asType(.float32) / Float(half))
        let angles =
            MLXArray(0 ..< length).asType(.float32)[0..., .newAxis]
            * frequencies[.newAxis]
        return stacked([sin(angles), cos(angles)], axis: -1).reshaped(length, dimensions)
    }

    /// The table's rows for `length` positions from `offset`, zero-extended past its end.
    func positions(offset: Int, length: Int, dtype: DType) -> MLXArray {
        let available = Swift.max(0, Swift.min(length, positionalEncoding.dim(0) - offset))
        guard available > 0 else {
            return MLXArray.zeros(
                [length, positionalEncoding.dim(1)], dtype: dtype)
        }
        var rows = positionalEncoding[offset ..< (offset + available)]
        if available < length {
            rows = concatenated(
                [
                    rows,
                    MLXArray.zeros(
                        [length - available, positionalEncoding.dim(1)],
                        dtype: positionalEncoding.dtype),
                ], axis: 0)
        }
        return rows.asType(dtype)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var x = embedTokens(inputs)
        let offset = cache?.first?.offset ?? 0
        x = x + positions(offset: offset, length: x.dim(1), dtype: x.dtype)

        for (i, layer) in layers.enumerated() {
            x = layer(x, cache: cache?[i] as? ArraysCache)
        }
        return norm(x)
    }
}

public class Z1TModel: Module, LLMModel {

    public let vocabularySize: Int

    public let model: Z1TModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ args: Z1TConfiguration) {
        self.vocabularySize = args.vocabularySize
        self.model = Z1TModelInner(args)
        _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        let prefixes = [
            ("embedding.", "model.embed_tokens."), ("blocks.", "model.layers."),
            ("norm.", "model.norm."), ("clf.", "lm_head."),
        ]
        for (key, value) in weights {
            var normalized = key == "pe.pe" ? "model.pe" : key
            for (source, destination) in prefixes where key.hasPrefix(source) {
                normalized = destination + key.dropFirst(source.count)
                break
            }
            sanitized[normalized] = value
        }
        return sanitized
    }

    /// Four array slots per layer: the pool's two running sums and the convolution's two
    /// tails. Nothing here is a key or a value.
    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        model.layers.map { _ in ArraysCache(size: 4) }
    }
}

public struct Z1TConfiguration: Decodable, Sendable {

    public enum LinearSite {
        case attention
        case mlp
    }

    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var hiddenLayers: Int
    var maxPositionEmbeddings: Int
    var aftKind: String
    var aftHeads: Int
    var aftKernelSize: Int
    var dynamicTanhAlpha: Float
    var linearFanIn: Int?
    var attentionFanIn: Int?
    var mlpFanIn: Int?
    var tanhLinear: Bool
    var tanhMLP: Bool
    var tieWordEmbeddings: Bool

    /// A per-site override falls back to the shared fan-in; `nil` means dense.
    func fanIn(site: LinearSite) -> Int? {
        switch site {
        case .attention: return attentionFanIn ?? linearFanIn
        case .mlp: return mlpFanIn ?? linearFanIn
        }
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case maxPositionEmbeddings = "max_position_embeddings"
        case aftKind = "aft_kind"
        case aftHeads = "aft_heads"
        case aftKernelSize = "aft_ksize"
        case dynamicTanhAlpha = "dyt_alpha"
        case linearFanIn = "linear_fan_in"
        case attentionFanIn = "attn_fan_in"
        case mlpFanIn = "mlp_fan_in"
        case tanhLinear = "tanh_linear"
        case tanhMLP = "tanh_mlp"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "z1t"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 50257
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 12288
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 4
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 256
        aftKind = try c.decodeIfPresent(String.self, forKey: .aftKind) ?? "conv"
        aftHeads = try c.decodeIfPresent(Int.self, forKey: .aftHeads) ?? 4
        aftKernelSize = try c.decodeIfPresent(Int.self, forKey: .aftKernelSize) ?? 4
        dynamicTanhAlpha = try c.decodeIfPresent(Float.self, forKey: .dynamicTanhAlpha) ?? 0.5
        linearFanIn = try c.decodeIfPresent(Int.self, forKey: .linearFanIn)
        attentionFanIn = try c.decodeIfPresent(Int.self, forKey: .attentionFanIn)
        mlpFanIn = try c.decodeIfPresent(Int.self, forKey: .mlpFanIn)
        tanhLinear = try c.decodeIfPresent(Bool.self, forKey: .tanhLinear) ?? false
        tanhMLP = try c.decodeIfPresent(Bool.self, forKey: .tanhMLP) ?? false
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

extension Z1TModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
