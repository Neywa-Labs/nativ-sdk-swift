import Foundation
import MLX
import MLXNN

/// Unweighted RMS normalisation with a dtype-dependent epsilon.
///
/// The reference calls `F.rms_norm` with `eps=None`, which resolves to
/// `finfo(dtype).eps` -- far larger than a typical 1e-6, and different per dtype. The
/// epsilon is therefore a property of the activation dtype, not of the config.
func limiteRMSNorm(_ x: MLXArray) -> MLXArray {
    let eps: Float
    switch x.dtype {
    case .bfloat16: eps = 0.0078125
    case .float16: eps = 0.0009765625
    default: eps = 1.1920929e-07
    }
    return MLXFast.rmsNorm(x, weight: MLXArray.ones([x.dim(-1)], dtype: x.dtype), eps: eps)
}

/// Scales a projection by a learned scalar, rounding the scalar to the weight's dtype
/// first.
///
/// The reference multiplies a 0-dim float32 tensor by a bfloat16 matrix, which torch
/// promotes as a wrapped scalar: the scalar is rounded to bfloat16 before the product.
/// Doing the product in float32 instead gives different numbers.
func limiteFold(_ scale: MLXArray, _ weight: MLXArray) -> MLXArray {
    scale.asType(weight.dtype) * weight
}

class LimiteMuddMixer: Module {
    @ParameterInfo(key: "dense1") var dense1: MLXArray
    @ParameterInfo(key: "dense2") var dense2: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    @ParameterInfo(key: "dense2_mlp") var dense2MLP: MLXArray?
    @ParameterInfo(key: "bias_mlp") var biasMLP: MLXArray?

    let usesRWay: Bool

    init(_ config: LimiteConfiguration) {
        let (layers, taps, inter) = (
            config.numHiddenLayers, config.muddTaps, config.muddInter
        )
        self.usesRWay = config.muddMLP
        _dense1.wrappedValue = MLXArray.zeros([inter, config.hiddenSize])
        _dense2.wrappedValue = MLXArray.zeros([layers, taps, inter])
        _bias.wrappedValue = MLXArray.zeros([layers, taps])
        if usesRWay {
            _dense2MLP.wrappedValue = MLXArray.zeros([layers, taps, inter])
            _biasMLP.wrappedValue = MLXArray.zeros([layers, taps])
        }
    }

    func combine(_ values: [MLXArray], current: MLXArray, layer: Int, rWay: Bool = false)
        -> MLXArray
    {
        let count = values.count
        let inner = gelu(
            limiteRMSNorm(current).matmul(dense1.asType(current.dtype).transposed()))
        let secondDense = rWay ? (dense2MLP ?? dense2) : dense2
        let secondBias = rWay ? (biasMLP ?? bias) : bias
        var w = inner.matmul(
            secondDense[layer, ..<count].asType(inner.dtype).transposed())
        w = w + secondBias[layer, ..<count].asType(w.dtype)

        // Left to right, in the activation dtype. A stacked contraction is
        // mathematically identical and numerically is not.
        var out = w[.ellipsis, 0 ..< 1].asType(values[0].dtype) * values[0]
        for index in 1 ..< count {
            out =
                out + w[.ellipsis, index ..< (index + 1)].asType(values[index].dtype)
                * values[index]
        }
        return out
    }
}

class LimiteRotary: Module {
    let _frequencies: MLXArray

    init(_ config: LimiteConfiguration) {
        let pairs = config.ropeNPairs
        var freq = pow(
            MLXArray(1.0 / config.ropeBaseLocal),
            MLXArray.linspace(0, 1, count: pairs).asType(.float32))
        freq = repeated(freq, count: 2, axis: 0)
        let tail = config.headDimensions - 2 * pairs
        self._frequencies =
            tail > 0
            ? concatenated([freq, MLXArray.zeros([tail])], axis: 0) : freq
    }

    func callAsFunction(_ q: MLXArray, _ k: MLXArray, positions: MLXArray)
        -> (MLXArray, MLXArray)
    {
        let headDim = _frequencies.dim(0)
        let theta =
            positions.asType(.float32).reshaped(-1, 1) * _frequencies.reshaped(1, -1)
        let cos = MLX.cos(theta).asType(.bfloat16).reshaped(-1, 1, headDim)
        var sin = MLX.sin(theta).asType(.bfloat16).reshaped(-1, 1, headDim)
        // The odd lanes carry the negated sine, which with the pairwise flip below gives
        // (a, b) -> (a cos + b sin, b cos - a sin).
        let lane = MLX.where(
            MLXArray(0 ..< headDim) % 2 .== 1, MLXArray(Float(-1)), MLXArray(Float(1))
        ).asType(sin.dtype)
        sin = sin * lane.reshaped(1, 1, headDim)
        return (Self.rotate(q, cos, sin), Self.rotate(k, cos, sin))
    }

    static func rotate(_ x: MLXArray, _ cos: MLXArray, _ sin: MLXArray) -> MLXArray {
        let shape = x.shape
        let paired = x.reshaped(shape.dropLast() + [shape[shape.count - 1] / 2, 2])
        let flipped = MLX.flipped(paired, axis: -1).reshaped(shape)
        return cos.asType(x.dtype) * x + sin.asType(x.dtype) * flipped
    }
}

/// A causal mask, optionally also bounded below by a sliding window.
///
/// The lower bound keeps `k >= q - window`, inclusive of the query token, so a window of
/// `w` admits `w + 1` keys.
func limiteCausalWindowMask(
    queryLength: Int, offset: Int, window: Int?, dtype: DType
) -> MLXArray {
    let queryIndex = MLXArray(offset ..< (offset + queryLength)).reshaped(-1, 1)
    let keyIndex = MLXArray(0 ..< (offset + queryLength)).reshaped(1, -1)
    var keep = keyIndex .<= queryIndex
    if let window {
        keep = keep .&& (keyIndex .>= (queryIndex - window))
    }
    return MLX.where(keep, MLXArray(0, dtype: dtype), MLXArray(-Float.infinity, dtype: dtype))
}

class LimiteAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let groups: Int
    let scale: Float
    let veDimensions: Int
    let veStoredHeads: Int
    let veGateScale: Float
    let attnGateChannels: Int
    let attnGateScale: Float
    let xsaEps: Float

    let usesVE: Bool
    let usesXSA: Bool
    let isGlobal: Bool
    let usesRope: Bool
    let window: Int?

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear
    @ParameterInfo(key: "qkv_scale") var qkvScale: MLXArray
    @ParameterInfo(key: "o_scale") var oScale: MLXArray
    @ParameterInfo(key: "xsa_alpha") var xsaAlpha: MLXArray
    @ParameterInfo(key: "ve_gate") var veGate: MLXArray?
    @ParameterInfo(key: "attn_gate") var attnGate: MLXArray?

    let rotary: LimiteRotary

    init(_ config: LimiteConfiguration, layer: Int, rotary: LimiteRotary) {
        self.heads = config.numAttentionHeads
        self.kvHeads = config.numKeyValueHeads
        self.headDimensions = config.headDimensions
        self.groups = heads / kvHeads
        self.scale = config.attentionSoftmaxScale
        self.veDimensions = config.veDimensions
        self.veStoredHeads = config.veStoredHeads
        self.veGateScale = config.veGateScale
        self.attnGateChannels = config.attnGateChannels
        self.attnGateScale = config.attnGateScale
        self.xsaEps = config.xsaNormalizeEps

        self.usesVE = config.veLayers.contains(layer)
        self.usesXSA = config.xsa && config.xsaLayers.contains(layer)
        self.isGlobal = config.globalLayers.contains(layer)
        self.usesRope = !(config.globalNope && isGlobal)
        let bound = isGlobal ? config.globalWindow : config.slidingWindow
        self.window = bound < 0 ? nil : bound
        self.rotary = rotary

        let hidden = config.hiddenSize
        let kvWidth = kvHeads * headDimensions
        _wq.wrappedValue = Linear(hidden, hidden, bias: false)
        _wk.wrappedValue = Linear(hidden, kvWidth, bias: false)
        _wv.wrappedValue = Linear(hidden, kvWidth, bias: false)
        _wo.wrappedValue = Linear(hidden, hidden, bias: false)
        _qkvScale.wrappedValue = MLXArray(Float(0))
        _oScale.wrappedValue = MLXArray(Float(0))
        _xsaAlpha.wrappedValue = MLXArray.zeros([heads])
        if usesVE {
            _veGate.wrappedValue = MLXArray.zeros([veStoredHeads, config.veGateChannels])
        }
        if attnGateChannels > 0 {
            _attnGate.wrappedValue = MLXArray.zeros([heads, attnGateChannels])
        }
    }

    /// Mixes the token's stored value embedding into the attention values through a
    /// sigmoid gate. Both attention and the KV cache consume the adjusted values.
    private func valueEmbeddings(_ attnIn: MLXArray, _ v: MLXArray, _ veRows: MLXArray)
        -> MLXArray
    {
        guard let veGate else { return v }
        let (B, L) = (attnIn.dim(0), attnIn.dim(1))
        var ve = veRows.asType(v.dtype).reshaped(B, L, veStoredHeads, veDimensions)
        if veDimensions < headDimensions {
            ve = padded(
                ve,
                widths: [
                    .init((0, 0)), .init((0, 0)), .init((0, 0)),
                    .init((0, headDimensions - veDimensions)),
                ])
        }
        var gateWeight = veGate.asType(attnIn.dtype)
        if veStoredHeads > kvHeads {
            ve = ve[0..., 0..., ..<kvHeads]
            gateWeight = gateWeight[..<kvHeads]
        }
        let channels = gateWeight.dim(-1)
        let gate =
            veGateScale
            * sigmoid(attnIn[.ellipsis, ..<channels].matmul(gateWeight.transposed()))
        return v + expandedDimensions(gate, axis: -1) * ve
    }

    func callAsFunction(
        _ attnIn: MLXArray, veRows: MLXArray, positions: MLXArray, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (attnIn.dim(0), attnIn.dim(1))
        var q = attnIn.matmul(limiteFold(qkvScale, wq.weight).transposed())
        var k = attnIn.matmul(limiteFold(qkvScale, wk.weight).transposed())
        var v = attnIn.matmul(limiteFold(qkvScale, wv.weight).transposed())
        q = q.reshaped(B, L, heads, headDimensions)
        k = k.reshaped(B, L, kvHeads, headDimensions)
        v = v.reshaped(B, L, kvHeads, headDimensions)

        if usesVE {
            v = valueEmbeddings(attnIn, v, veRows)
        }

        q = limiteRMSNorm(q)
        k = limiteRMSNorm(k)
        if usesRope {
            let (rq, rk) = rotary(
                q.reshaped(B * L, heads, headDimensions),
                k.reshaped(B * L, kvHeads, headDimensions),
                positions: positions.reshaped(-1))
            q = rq.reshaped(B, L, heads, headDimensions)
            k = rk.reshaped(B, L, kvHeads, headDimensions)
        }

        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        v = v.transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        var keys = k
        var values = v
        if let cache {
            (keys, values) = cache.update(keys: k, values: v)
        }
        let mask = limiteCausalWindowMask(
            queryLength: L, offset: offset, window: window, dtype: q.dtype)
        var y = MLXFast.scaledDotProductAttention(
            queries: q, keys: keys, values: values, scale: scale, mask: .array(mask)
        )
        .transposed(0, 2, 1, 3)

        if usesXSA {
            // Removes each head's output component along its own value direction, by a
            // learned per-head amount bounded through tanh.
            let recent = expandKV(values.transposed(0, 2, 1, 3)[0..., (-L)...])
            let asFloat = recent.asType(.float32)
            let normalized =
                asFloat
                / maximum(
                    MLX.sqrt((asFloat * asFloat).sum(axis: -1, keepDims: true)),
                    MLXArray(xsaEps))
            let projection = (y.asType(.float32) * normalized).sum(axis: -1, keepDims: true)
            let alpha = MLX.tanh(xsaAlpha.asType(.float32)).reshaped(1, 1, -1, 1)
            y = y - (alpha * projection * normalized).asType(y.dtype)
        }

        if let attnGate {
            let gate =
                attnGateScale
                * sigmoid(
                    attnIn[.ellipsis, ..<attnGateChannels].matmul(
                        attnGate.asType(attnIn.dtype).transposed()))
            y = y * expandedDimensions(gate, axis: -1).asType(y.dtype)
        }

        y = y.reshaped(B, L, heads * headDimensions)
        return y.matmul(limiteFold(oScale, wo.weight).transposed())
    }

    private func expandKV(_ x: MLXArray) -> MLXArray {
        x.dim(2) < heads ? repeated(x, count: groups, axis: 2) : x
    }
}

class LimiteMLP: Module, UnaryLayer {
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "gate_proj") var gate: Linear?

    let usesSwiGLU: Bool

    init(_ config: LimiteConfiguration) {
        self.usesSwiGLU = config.mlpType == "swiglu"
        _up.wrappedValue = Linear(config.hiddenSize, config.intermediateSize, bias: false)
        _down.wrappedValue = Linear(config.intermediateSize, config.hiddenSize, bias: false)
        if usesSwiGLU {
            _gate.wrappedValue = Linear(
                config.hiddenSize, config.intermediateSize, bias: false)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = up(x)
        // The alternative is a squared ReLU, which is why the gate is absent there.
        let activated =
            if let gate { silu(gate(x)) * projected } else { square(relu(projected)) }
        return down(activated)
    }
}

class LimiteDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: LimiteAttention
    @ModuleInfo(key: "mlp") var mlp: LimiteMLP
    @ParameterInfo(key: "resid_lambda_attn") var residLambdaAttn: MLXArray
    @ParameterInfo(key: "post_lambda_attn") var postLambdaAttn: MLXArray
    @ParameterInfo(key: "resid_lambda_mlp") var residLambdaMLP: MLXArray
    @ParameterInfo(key: "post_lambda_mlp") var postLambdaMLP: MLXArray

    init(_ config: LimiteConfiguration, layer: Int, rotary: LimiteRotary) {
        _attention.wrappedValue = LimiteAttention(config, layer: layer, rotary: rotary)
        _mlp.wrappedValue = LimiteMLP(config)
        for scalar in [_residLambdaAttn, _postLambdaAttn, _residLambdaMLP, _postLambdaMLP] {
            scalar.wrappedValue = MLXArray(Float(0))
        }
    }

    /// Both residual branches are learned blends rather than plain additions, which is why
    /// `residualBase` is supplied separately: a tapped layer mixes a different history
    /// into it than the one the attention reads.
    func callAsFunction(
        _ x: MLXArray, attnIn: MLXArray, residualBase: MLXArray, veRows: MLXArray,
        positions: MLXArray, cache: KVCache?
    ) -> MLXArray {
        let attended = attention(attnIn, veRows: veRows, positions: positions, cache: cache)
        let mixed =
            residLambdaAttn.asType(x.dtype) * residualBase
            + postLambdaAttn.asType(x.dtype) * attended
        let projected = mlp(limiteRMSNorm(mixed))
        return residLambdaMLP.asType(mixed.dtype) * mixed
            + postLambdaMLP.asType(mixed.dtype) * projected
    }
}

public class LimiteModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "value_embeds") var valueEmbeds: Embedding
    @ModuleInfo(key: "mudd") var mudd: LimiteMuddMixer

    let layers: [LimiteDecoderLayer]
    let tapIndices: [Int: [Int]]
    let retained: Set<Int>
    let finalSoftcap: Float

    public init(_ config: LimiteConfiguration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _valueEmbeds.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize,
            dimensions: config.veStoredHeads * config.veDimensions)
        let rotary = LimiteRotary(config)
        self.layers = (0 ..< config.numHiddenLayers).map {
            LimiteDecoderLayer(config, layer: $0, rotary: rotary)
        }
        _mudd.wrappedValue = LimiteMuddMixer(config)
        self.tapIndices = config.muddTapIndices
        self.retained = Set(config.muddTapIndices.values.flatMap { $0 })
        self.finalSoftcap = config.finalSoftcap
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var x = limiteRMSNorm(embedTokens(inputs))
        let veRows = valueEmbeds(inputs)

        let offset = cache?.first?.offset ?? 0
        let positions = broadcast(
            MLXArray(offset ..< (offset + inputs.dim(1))), to: inputs.shape)

        // A tapped layer reads a learned combination of earlier depths rather than the
        // running residual, so those depths are kept as they are produced.
        var history: [Int: MLXArray] = tapIndices.isEmpty ? [:] : [0: x]
        for (index, layer) in layers.enumerated() {
            let attnIn: MLXArray
            let residualBase: MLXArray
            if let taps = tapIndices[index] {
                let values = taps.compactMap { history[$0] }
                attnIn = limiteRMSNorm(mudd.combine(values, current: x, layer: index))
                residualBase =
                    mudd.usesRWay
                    ? mudd.combine(values, current: x, layer: index, rWay: true) : x
            } else {
                attnIn = limiteRMSNorm(x)
                residualBase = x
            }
            x = layer(
                x, attnIn: attnIn, residualBase: residualBase, veRows: veRows,
                positions: positions, cache: cache?[index])
            if retained.contains(index + 1) {
                history[index + 1] = x
            }
        }

        if finalSoftcap > 0 {
            x = finalSoftcap * MLX.tanh(x / finalSoftcap)
        }
        return limiteRMSNorm(x)
    }
}

public class LimiteModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let model: LimiteModelInner

    let softcapA: Float
    let softcapB: Float
    let softcapC: Float

    public init(_ config: LimiteConfiguration) {
        // Upstream refuses each of these rather than running something it cannot reproduce,
        // and a silent acceptance here would answer with the wrong numbers instead.
        precondition(
            !config.ropePerLayer,
            "rope_per_layer needs a per-layer frequency table; the released checkpoint "
                + "declares false")
        precondition(
            ["swiglu", "relu2"].contains(config.mlpType),
            "unsupported mlp_type \(config.mlpType)")
        if let cap = config.softcapLogits, !cap.isEmpty {
            let kind = cap["kind"].flatMap { value -> String? in
                if case .string(let name) = value { return name }
                return nil
            }
            precondition(kind == "sigmoid", "only a sigmoid softcap_logits is supported")
        }
        if let declared = config.muddLayers {
            precondition(
                Set(declared) == Set(config.muddTapIndices.keys),
                "mudd_tap_idx covers \(config.muddTapIndices.keys.sorted()) but "
                    + "mudd_layers is \(declared.sorted())")
        }
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.numHiddenLayers).map { _ in config.numKeyValueHeads }
        self.model = LimiteModelInner(config)
        let cap = config.softcapLogits ?? [:]
        self.softcapA = cap["a"]?.asFloat() ?? 0
        self.softcapB = cap["b"]?.asFloat() ?? 0
        self.softcapC = cap["c"]?.asFloat() ?? 1
    }

    /// The head is a bounded sigmoid over the tied embedding, not a bare projection, so
    /// the result lies in (0, a) rather than being an unbounded logit.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let h = model(inputs, cache: cache)
        let raw = h.matmul(model.embedTokens.weight.asType(h.dtype).transposed())
        return softcapA * sigmoid((raw.asType(.float32) + softcapB) / softcapC)
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        kvHeads.map { _ in MaterializedKVCache(KVCacheSimple()) }
    }

    /// A checkpoint of this family writes the backbone at bare `model.`; the tree keeps it
    /// one level down. The learned scalars stay rank 0: a shape-(1,) scalar would promote
    /// the weight fold to float32 and diverge from the reference.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("language_model.") {
                sanitized[String(key.dropFirst("language_model.".count))] = value
            } else {
                sanitized[key] = value
            }
        }
        return sanitized
    }
}

public struct LimiteConfiguration: Codable, Sendable {
    public let modelType: String
    private let _hiddenSize: Int?
    public var hiddenSize: Int { _hiddenSize ?? 1280 }
    private let _intermediateSize: Int?
    public var intermediateSize: Int { _intermediateSize ?? 3328 }
    private let _numHiddenLayers: Int?
    public var numHiddenLayers: Int { _numHiddenLayers ?? 48 }
    private let _numAttentionHeads: Int?
    public var numAttentionHeads: Int { _numAttentionHeads ?? 10 }
    private let _numKeyValueHeads: Int?
    public var numKeyValueHeads: Int { _numKeyValueHeads ?? 2 }
    private let _headDimensions: Int?
    public var headDimensions: Int { _headDimensions ?? 128 }
    private let _vocabularySize: Int?
    public var vocabularySize: Int { _vocabularySize ?? 151_680 }
    private let _tieWordEmbeddings: Bool?
    public var tieWordEmbeddings: Bool { _tieWordEmbeddings ?? true }
    private let _mlpType: String?
    public var mlpType: String { _mlpType ?? "swiglu" }

    private let _attentionSoftmaxScale: Float?
    public var attentionSoftmaxScale: Float { _attentionSoftmaxScale ?? 0.1 }
    private let _attnGateChannels: Int?
    public var attnGateChannels: Int { _attnGateChannels ?? 128 }
    private let _attnGateScale: Float?
    public var attnGateScale: Float { _attnGateScale ?? 2.0 }

    private let _veDimensions: Int?
    public var veDimensions: Int { _veDimensions ?? 128 }
    private let _veGateChannels: Int?
    public var veGateChannels: Int { _veGateChannels ?? 12 }
    private let _veGateScale: Float?
    public var veGateScale: Float { _veGateScale ?? 2.0 }
    private let _veStoredHeads: Int?
    public var veStoredHeads: Int { _veStoredHeads ?? 2 }
    private let _veLayers: [Int]?
    public var veLayers: Set<Int> { Set(_veLayers ?? []) }

    private let _xsa: Bool?
    public var xsa: Bool { _xsa ?? true }
    private let _xsaLayers: [Int]?
    public var xsaLayers: Set<Int> { Set(_xsaLayers ?? []) }
    private let _xsaNormalizeEps: Float?
    public var xsaNormalizeEps: Float { _xsaNormalizeEps ?? 1e-4 }

    private let _globalLayers: [Int]?
    public var globalLayers: Set<Int> { Set(_globalLayers ?? []) }
    private let _globalNope: Bool?
    public var globalNope: Bool { _globalNope ?? true }
    private let _globalWindow: Int?
    public var globalWindow: Int { _globalWindow ?? -1 }
    private let _slidingWindow: Int?
    public var slidingWindow: Int { _slidingWindow ?? 1024 }

    private let _ropeBaseLocal: Float?
    public var ropeBaseLocal: Float { _ropeBaseLocal ?? 1024.0 }
    private let _ropeNPairs: Int?
    public var ropeNPairs: Int { _ropeNPairs ?? 32 }
    private let _ropePerLayer: Bool?
    public var ropePerLayer: Bool { _ropePerLayer ?? false }

    private let _muddMLP: Bool?
    public var muddMLP: Bool { _muddMLP ?? true }
    private let _muddTaps: Int?
    public var muddTaps: Int { _muddTaps ?? 3 }
    private let _muddInter: Int?
    public var muddInter: Int { _muddInter ?? 32 }
    /// Keyed by layer index as a string in the config, which is how JSON carries an
    /// integer-keyed map.
    private let _muddTapIdx: [String: [Int]]?
    public var muddTapIndices: [Int: [Int]] {
        Dictionary(
            uniqueKeysWithValues: (_muddTapIdx ?? [:]).compactMap { key, value in
                Int(key).map { ($0, value) }
            })
    }

    /// Declared alongside `mudd_tap_idx`; upstream refuses a config where the two
    /// disagree, since one of them would then be describing layers the other does not.
    private let _muddLayers: [Int]?
    public var muddLayers: [Int]? { _muddLayers?.isEmpty == true ? nil : _muddLayers }

    private let _finalSoftcap: Float?
    public var finalSoftcap: Float { _finalSoftcap ?? 0 }
    public let softcapLogits: [String: StringOrNumber]?
    private let _eosTokenId: IntOrIntArray?
    public var eosTokenId: [Int]? { _eosTokenId?.values }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case _hiddenSize = "hidden_size"
        case _intermediateSize = "intermediate_size"
        case _numHiddenLayers = "num_hidden_layers"
        case _numAttentionHeads = "num_attention_heads"
        case _numKeyValueHeads = "num_key_value_heads"
        case _headDimensions = "head_dim"
        case _vocabularySize = "vocab_size"
        case _tieWordEmbeddings = "tie_word_embeddings"
        case _mlpType = "mlp_type"
        case _attentionSoftmaxScale = "attention_softmax_scale"
        case _attnGateChannels = "attn_gate_channels"
        case _attnGateScale = "attn_gate_scale"
        case _veDimensions = "ve_dim"
        case _veGateChannels = "ve_gate_channels"
        case _veGateScale = "ve_gate_scale"
        case _veStoredHeads = "ve_stored_heads"
        case _veLayers = "ve_layers"
        case _xsa = "xsa"
        case _xsaLayers = "xsa_layers"
        case _xsaNormalizeEps = "xsa_normalize_eps"
        case _globalLayers = "global_layers"
        case _globalNope = "global_nope"
        case _globalWindow = "global_window"
        case _slidingWindow = "sliding_window"
        case _ropeBaseLocal = "rope_base_local"
        case _ropeNPairs = "rope_n_pairs"
        case _ropePerLayer = "rope_per_layer"
        case _muddMLP = "mudd_mlp"
        case _muddTaps = "mudd_taps"
        case _muddInter = "mudd_inter"
        case _muddTapIdx = "mudd_tap_idx"
        case _muddLayers = "mudd_layers"
        case _finalSoftcap = "final_softcap"
        case softcapLogits = "softcap_logits"
        case _eosTokenId = "eos_token_id"
    }
}

extension LimiteModel: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}
