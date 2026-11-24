import Foundation
import MLX
import MLXNN

private let rwkv7WKVKernel = MLXFast.metalKernel(
    name: "wkv7_kernel",
    inputNames: ["r", "w", "k", "v", "a", "b", "state_in", "T"],
    outputNames: ["y", "state_out"],
    source: """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / H;
            auto h_idx = n % H;
            constexpr int n_per_t = D / 32;

            auto r_ = r + b_idx * T * H * D + h_idx * D;
            auto w_ = w + b_idx * T * H * D + h_idx * D;
            auto k_ = k + b_idx * T * H * D + h_idx * D;
            auto v_ = v + b_idx * T * H * D + h_idx * D;
            auto a_ = a + b_idx * T * H * D + h_idx * D;
            auto b_ = b + b_idx * T * H * D + h_idx * D;
            y += b_idx * T * H * D + h_idx * D;

            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;

            auto i_state = state_in  + (n * D + dv_idx) * D;
            auto o_state = state_out + (n * D + dv_idx) * D;

            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }

            for (int t = 0; t < T; ++t) {
              float sa = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                sa += state[i] * a_[s_idx];
                state[i] = state[i] * w_[s_idx];
              }
              sa = simd_sum(sa);

              float out = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] + k_[s_idx] * v_[dv_idx] + sa * b_[s_idx];
                out += state[i] * r_[s_idx];
              }
              out = simd_sum(out);
              if (thread_index_in_simdgroup == 0) {
                y[dv_idx] = static_cast<InT>(out);
              }

              r_ += H * D;
              w_ += H * D;
              k_ += H * D;
              v_ += H * D;
              a_ += H * D;
              b_ += H * D;
              y  += H * D;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<InT>(state[i]);
            }
        """)

/// One step of the WKV-7 recurrence, written in MLX.
///
/// `state <- state * w + v k^T + (state a) b^T`, then `y = state r`. This is the reference
/// the Metal kernel has to reproduce.
///
/// Diverged: the two outer products are broadcast multiplies, where upstream writes them as
/// `v[..., None] @ k[..., None, :]`. On Metal, `matmul` with an inner dimension of one
/// rounds its operands to about 1e-3 relative -- measured here against the same product
/// computed by broadcasting -- which is a thousand times the float32 noise floor and made
/// this reference disagree with the kernel it is supposed to check. Every genuine
/// contraction below is still a `matmul`.
func rwkv7Step(
    r: MLXArray, w: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray, b: MLXArray,
    state: MLXArray
) -> (MLXArray, MLXArray) {
    let sa = matmul(state, a[.ellipsis, .newAxis])
    let updated =
        state * w[0..., 0..., .newAxis, 0...]
        + v[.ellipsis, .newAxis] * k[.ellipsis, .newAxis, 0...]
        + sa * b[.ellipsis, .newAxis, 0...]
    return (matmul(updated, r[.ellipsis, .newAxis]), updated)
}

/// The WKV-7 recurrence over a whole sequence, on the Metal kernel where one is available.
///
/// - Parameters shaped `[batch, positions, heads, head_dim]`, state `[batch, heads,
///   head_dim, head_dim]`.
/// - Returns: the outputs, shaped like `r`, and the state after the last position.
func rwkv7WKV(
    r: MLXArray, w: MLXArray, k: MLXArray, v: MLXArray, a: MLXArray, b: MLXArray,
    state: MLXArray, forceReference: Bool = false
) -> (MLXArray, MLXArray) {
    let positions = r.dim(1)

    if !forceReference, Device.defaultDevice().deviceType == .gpu {
        let outputs = rwkv7WKVKernel(
            [r, w, k, v, a, b, state, positions],
            template: [("InT", r.dtype), ("H", r.dim(2)), ("D", r.dim(3))],
            grid: (32, r.dim(3), r.dim(0) * r.dim(2)),
            threadGroup: (32, 4, 1),
            outputShapes: [r.shape, state.shape],
            outputDTypes: [r.dtype, r.dtype])
        return (outputs[0], outputs[1])
    }

    var carried = state
    var steps: [MLXArray] = []
    steps.reserveCapacity(positions)
    for t in 0 ..< positions {
        let (y, next) = rwkv7Step(
            r: r[0..., t], w: w[0..., t], k: k[0..., t], v: v[0..., t],
            a: a[0..., t], b: b[0..., t], state: carried)
        steps.append(y)
        carried = next
    }
    return (stacked(steps, axis: 1).asType(r.dtype).squeezed(axis: -1), carried)
}

/// The previous position's input, per position: the state for a single step, and the
/// sequence shifted right by one otherwise.
func rwkv7TokenShift(_ x: MLXArray, state: MLXArray?) -> MLXArray {
    let previous = state ?? MLXArray.zeros([x.dim(0), 1, x.dim(2)], dtype: x.dtype)
    if x.dim(1) == 1 {
        return previous
    }
    return concatenated([previous, x[0..., ..<(x.dim(1) - 1)]], axis: 1)
}

/// A per-head scale and shift around an unparameterized layer norm.
class RWKV7LayerNormPerHead: Module {

    @ModuleInfo var weight: MLXArray
    @ModuleInfo var bias: MLXArray

    let eps: Float

    init(headDimensions: Int, heads: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.zeros([heads, headDimensions])
        _bias.wrappedValue = MLXArray.zeros([heads, headDimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        weight
            * MLXFast.layerNorm(
                x, weight: MLXArray.mlxNone, bias: MLXArray.mlxNone, eps: eps) + bias
    }
}

/// A down-up projection pair with an activation between them.
///
/// The activation retains a Module slot so `lora.0` and `lora.2` stay aligned
/// during strict loading and partial quantization.
class RWKV7LoRA: Module {

    @ModuleInfo(key: "lora") var lora: [UnaryLayer]

    init(
        inputDimensions: Int, outputDimensions: Int, lowRankDimensions: Int, bias: Bool = true,
        activation: String?
    ) {
        let middle: UnaryLayer
        switch activation {
        case "sigmoid": middle = Sigmoid()
        case "tanh": middle = Tanh()
        case "relu": middle = ReLU()
        case nil: middle = Identity()
        default: fatalError("unsupported rwkv7 lora activation \(activation ?? "nil")")
        }
        _lora.wrappedValue = [
            Linear(inputDimensions, lowRankDimensions, bias: false),
            SequentialActivation(middle),
            Linear(lowRankDimensions, outputDimensions, bias: bias),
        ]
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        lora[2](lora[1](lora[0](x)))
    }
}

class RWKV7ChannelMixing: Module {

    @ModuleInfo(key: "key") var key: Linear
    @ModuleInfo(key: "value") var value: Linear

    @ParameterInfo(key: "x_k") var xK: MLXArray

    init(_ args: RWKV7Configuration) {
        _key.wrappedValue = Linear(args.hiddenSize, args.intermediateSize, bias: false)
        _value.wrappedValue = Linear(args.intermediateSize, args.hiddenSize, bias: false)
        _xK.wrappedValue = MLXArray.zeros([args.hiddenSize])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: ArraysCache?) -> MLXArray {
        let shifted = rwkv7TokenShift(x, state: cache?[2])
        let mixed = x + (shifted - x) * xK
        cache?[2] = x[0..., (x.dim(1) - 1)...]
        return value(square(relu(key(mixed))))
    }
}

class RWKV7TimeMixing: Module {

    let layerIndex: Int
    let heads: Int
    let headDim: Int

    @ParameterInfo(key: "x_r") var xR: MLXArray
    @ParameterInfo(key: "x_w") var xW: MLXArray
    @ParameterInfo(key: "x_k") var xK: MLXArray
    @ParameterInfo(key: "x_v") var xV: MLXArray
    @ParameterInfo(key: "x_a") var xA: MLXArray
    @ParameterInfo(key: "x_g") var xG: MLXArray

    @ParameterInfo(key: "k_k") var kK: MLXArray
    @ParameterInfo(key: "k_a") var kA: MLXArray
    @ParameterInfo(key: "r_k") var rK: MLXArray

    @ModuleInfo(key: "r_proj") var rProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "g_norm") var gNorm: RWKV7LayerNormPerHead

    @ModuleInfo(key: "w_lora") var wLoRA: RWKV7LoRA
    @ModuleInfo(key: "v_lora") var vLoRA: RWKV7LoRA?
    @ModuleInfo(key: "a_lora") var aLoRA: RWKV7LoRA
    @ModuleInfo(key: "g_lora") var gLoRA: RWKV7LoRA

    init(_ args: RWKV7Configuration, layerIndex: Int) {
        self.layerIndex = layerIndex
        self.headDim = args.headDimensions
        self.heads = args.hiddenSize / args.headDimensions

        for parameter in [_xR, _xW, _xK, _xV, _xA, _xG] {
            parameter.wrappedValue = MLXArray.zeros([1, 1, args.hiddenSize])
        }
        _kK.wrappedValue = MLXArray.zeros([heads, headDim])
        _kA.wrappedValue = MLXArray.zeros([heads, headDim])
        _rK.wrappedValue = MLXArray.zeros([heads, headDim])

        _rProj.wrappedValue = Linear(args.hiddenSize, args.hiddenSize, bias: false)
        _kProj.wrappedValue = Linear(args.hiddenSize, args.hiddenSize, bias: false)
        _vProj.wrappedValue = Linear(args.hiddenSize, args.hiddenSize, bias: false)
        _oProj.wrappedValue = Linear(args.hiddenSize, args.hiddenSize, bias: false)

        _gNorm.wrappedValue = RWKV7LayerNormPerHead(
            headDimensions: headDim, heads: heads, eps: 64e-5)

        _wLoRA.wrappedValue = RWKV7LoRA(
            inputDimensions: args.hiddenSize, outputDimensions: args.hiddenSize,
            lowRankDimensions: args.decayLowRankDimensions, activation: "tanh")
        if layerIndex > 0 {
            _vLoRA.wrappedValue = RWKV7LoRA(
                inputDimensions: args.hiddenSize, outputDimensions: args.hiddenSize,
                lowRankDimensions: args.valueLowRankDimensions, activation: nil)
        }
        _aLoRA.wrappedValue = RWKV7LoRA(
            inputDimensions: args.hiddenSize, outputDimensions: args.hiddenSize,
            lowRankDimensions: args.aLowRankDimensions, activation: nil)
        _gLoRA.wrappedValue = RWKV7LoRA(
            inputDimensions: args.hiddenSize, outputDimensions: args.hiddenSize,
            lowRankDimensions: args.gateLowRankDimensions, bias: false,
            activation: "sigmoid")

        super.init()
    }

    func callAsFunction(_ x: MLXArray, valueFirst: MLXArray?, cache: ArraysCache?) -> (
        MLXArray, MLXArray
    ) {
        let (B, L, D) = (x.dim(0), x.dim(1), x.dim(2))

        let shifted = rwkv7TokenShift(x, state: cache?[0])
        let delta = shifted - x

        let key0 = kProj(x + delta * xK).reshaped(B, L, heads, headDim)
        var value = vProj(x + delta * xV).reshaped(B, L, heads, headDim)
        let receptance = rProj(x + delta * xR).reshaped(B, L, heads, headDim)
        let iclr = sigmoid(aLoRA(x + delta * xA)).reshaped(B, L, heads, headDim)
        let gate = gLoRA(x + delta * xG)

        let carriedValueFirst: MLXArray
        if let vLoRA {
            guard let valueFirst else {
                fatalError("layer \(layerIndex) has a v_lora but received no v_first")
            }
            let blend = sigmoid(vLoRA(x + delta * xV)).reshaped(B, L, heads, headDim)
            value = value + (valueFirst - value) * blend
            carriedValueFirst = valueFirst
        } else {
            carriedValueFirst = value
        }

        var decay = sigmoid(wLoRA(x + delta * xW).reshaped(B, L, heads, headDim))
            .asType(.float32)
        decay = exp(-0.606531 * decay).asType(receptance.dtype)

        let scaled = key0 * kK
        let kk = scaled / maximum(sqrt(scaled.square().sum(axis: -1, keepDims: true)), 1e-7)
        let key = key0 * (1 + (iclr - 1) * kA)

        let state =
            cache?[1]
            ?? MLXArray.zeros([B, heads, headDim, headDim], dtype: receptance.dtype)
        let (wkv, nextState) = rwkv7WKV(
            r: receptance, w: decay, k: key, v: value, a: -kk, b: kk * iclr, state: state)

        var out = gNorm(wkv.reshaped(B, L, heads, headDim))
        out =
            (out + (receptance * key * rK).sum(axis: -1, keepDims: true) * value)
            .reshaped(B, L, D)

        cache?[0] = x[0..., (L - 1)...]
        cache?[1] = nextState

        return (oProj(out * gate), carriedValueFirst)
    }
}

class RWKV7Layer: Module {

    let layerIndex: Int

    @ModuleInfo(key: "pre_norm") var preNorm: LayerNorm?
    @ModuleInfo(key: "attn") var timeMixing: RWKV7TimeMixing
    @ModuleInfo(key: "ffn") var channelMixing: RWKV7ChannelMixing
    @ModuleInfo(key: "attn_norm") var attentionNorm: LayerNorm
    @ModuleInfo(key: "ffn_norm") var feedForwardNorm: LayerNorm

    init(_ args: RWKV7Configuration, layerIndex: Int) {
        self.layerIndex = layerIndex
        if layerIndex == 0 {
            _preNorm.wrappedValue = LayerNorm(
                dimensions: args.hiddenSize, eps: args.normEps)
        }
        _timeMixing.wrappedValue = RWKV7TimeMixing(args, layerIndex: layerIndex)
        _channelMixing.wrappedValue = RWKV7ChannelMixing(args)
        _attentionNorm.wrappedValue = LayerNorm(
            dimensions: args.hiddenSize, eps: args.normEps)
        _feedForwardNorm.wrappedValue = LayerNorm(
            dimensions: args.hiddenSize, eps: args.normEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, valueFirst: MLXArray?, cache: ArraysCache?) -> (
        MLXArray, MLXArray
    ) {
        var input = x
        if let preNorm {
            input = preNorm(input)
        }
        let (mixed, carried) = timeMixing(
            attentionNorm(input), valueFirst: valueFirst, cache: cache)
        let h = input + mixed
        return (h + channelMixing(feedForwardNorm(h), cache: cache), carried)
    }
}

public class RWKV7ModelInner: Module {

    @ModuleInfo(key: "embeddings") var embeddings: Embedding

    let layers: [RWKV7Layer]
    let norm: LayerNorm

    public init(_ args: RWKV7Configuration) {
        precondition(args.vocabularySize > 0)
        _embeddings.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { RWKV7Layer(args, layerIndex: $0) }
        self.norm = LayerNorm(dimensions: args.hiddenSize, eps: args.normEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embeddings(inputs)
        var valueFirst: MLXArray? = nil

        for (i, layer) in layers.enumerated() {
            let (next, carried) = layer(
                h, valueFirst: valueFirst, cache: cache?[i] as? ArraysCache)
            h = next
            valueFirst = carried
        }

        return norm(h)
    }
}

public class RWKV7Model: Module, LLMModel {

    public let vocabularySize: Int

    public let model: RWKV7ModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: RWKV7Configuration

    public init(_ args: RWKV7Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.model = RWKV7ModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(hidden)
        }
        return model.embeddings.asLinear(hidden)
    }

    /// Three slots per layer: the time-mixing token shift, the recurrent state, and the
    /// channel-mixing token shift.
    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        model.layers.map { _ in ArraysCache(size: 3) }
    }

    /// `k_k`, `k_a` and the per-head norm's parameters ship flat.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let heads = configuration.hiddenSize / configuration.headDimensions
        var sanitized = weights
        for (key, value) in weights
        where key.contains("k_k") || key.contains("k_a") || key.contains("g_norm") {
            sanitized[key] = value.reshaped(heads, configuration.headDimensions)
        }
        return sanitized
    }
}

public struct RWKV7Configuration: Decodable, Sendable {

    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var normEps: Float
    var headDimensions: Int
    var hiddenLayers: Int
    var aLowRankDimensions: Int
    var valueLowRankDimensions: Int
    var gateLowRankDimensions: Int
    var decayLowRankDimensions: Int
    var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case normEps = "norm_eps"
        case headDimensions = "head_dim"
        case hiddenLayers = "num_hidden_layers"
        case aLowRankDimensions = "a_low_rank_dim"
        case valueLowRankDimensions = "v_low_rank_dim"
        case gateLowRankDimensions = "gate_low_rank_dim"
        case decayLowRankDimensions = "decay_low_rank_dim"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "rwkv7"
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        normEps = try c.decode(Float.self, forKey: .normEps)
        headDimensions = try c.decode(Int.self, forKey: .headDimensions)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        aLowRankDimensions = try c.decode(Int.self, forKey: .aLowRankDimensions)
        valueLowRankDimensions = try c.decode(Int.self, forKey: .valueLowRankDimensions)
        gateLowRankDimensions = try c.decode(Int.self, forKey: .gateLowRankDimensions)
        decayLowRankDimensions = try c.decode(Int.self, forKey: .decayLowRankDimensions)
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

extension RWKV7Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
