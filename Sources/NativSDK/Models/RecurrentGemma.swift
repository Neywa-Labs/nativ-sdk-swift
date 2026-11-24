import Foundation
import MLX
import MLXNN

/// Gemma's `1 + weight` normalization, with the default epsilon the recurrent layers use.
class RecurrentGemmaRMSNorm: Module, UnaryLayer {

    let eps: Float
    @ModuleInfo var weight: MLXArray

    init(dimensions: Int, eps: Float = 1e-5) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dimensions])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: 1.0 + weight, eps: eps)
    }
}

/// The linear recurrence `h_t = a_t * h_{t-1} + x_t`, returning the whole trajectory and
/// the final state.
///
/// A single position with no incoming state is the identity, which is what makes the first
/// token of a fresh sequence special.
func recurrentGemmaScan(x: MLXArray, a: MLXArray, state: MLXArray?) -> (
    MLXArray, MLXArray
) {
    if x.dim(1) == 1 {
        guard let state else { return (x, x[0..., 0]) }
        let y = a * state[0..., .newAxis] + x
        return (y, y[0..., -1])
    }

    var h = state ?? MLXArray.zeros([x.dim(0), x.dim(2)], dtype: x.dtype)
    var steps: [MLXArray] = []
    steps.reserveCapacity(x.dim(1))
    for t in 0 ..< x.dim(1) {
        h = a[0..., t] * h + x[0..., t]
        steps.append(h)
    }
    return (stacked(steps, axis: 1), h)
}

/// A depthwise convolution over time, carrying the last `kernelSize - 1` inputs forward.
class RecurrentGemmaConv1d: Module {

    @ModuleInfo var weight: MLXArray
    @ModuleInfo var bias: MLXArray

    let kernelSize: Int

    init(channels: Int, kernelSize: Int) {
        self.kernelSize = kernelSize
        _weight.wrappedValue = MLXArray.zeros([channels, kernelSize, 1])
        _bias.wrappedValue = MLXArray.zeros([channels])
        super.init()
    }

    func callAsFunction(_ x: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        let groups = weight.dim(0)
        let history: MLXArray
        if let state {
            history = concatenated([state, x], axis: 1)
        } else {
            history = padded(
                x,
                widths: [
                    IntOrPair((0, 0)), IntOrPair((kernelSize - 1, 0)), IntOrPair((0, 0)),
                ])
        }
        let y = conv1d(history, weight, groups: groups) + bias
        return (y, history[0..., (1 - kernelSize)..., 0...])
    }
}

/// A Real-Gated Linear Recurrent Unit.
///
/// Both gates are block diagonal: `[heads, head_dim, head_dim]` weights applied per head,
/// not one dense `[width, width]`.
class RecurrentGemmaRGLRU: Module {

    let width: Int
    let heads: Int
    let headDim: Int

    @ParameterInfo(key: "recurrent_param") var recurrentParam: MLXArray
    @ParameterInfo(key: "input_gate_weight") var inputGateWeight: MLXArray
    @ParameterInfo(key: "input_gate_bias") var inputGateBias: MLXArray
    @ParameterInfo(key: "recurrent_gate_weight") var recurrentGateWeight: MLXArray
    @ParameterInfo(key: "recurrent_gate_bias") var recurrentGateBias: MLXArray

    init(width: Int, heads: Int) {
        self.width = width
        self.heads = heads
        self.headDim = width / heads

        _recurrentParam.wrappedValue = MLXArray.zeros([width])
        _inputGateWeight.wrappedValue = MLXArray.zeros([heads, headDim, headDim])
        _inputGateBias.wrappedValue = MLXArray.zeros([heads, headDim])
        _recurrentGateWeight.wrappedValue = MLXArray.zeros([heads, headDim, headDim])
        _recurrentGateBias.wrappedValue = MLXArray.zeros([heads, headDim])
        super.init()
    }

    private func blockLinear(_ x: MLXArray, _ w: MLXArray, _ b: MLXArray) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let h = x.reshaped(B, L, heads, headDim).swappedAxes(1, 2)
        return sigmoid((matmul(h, w).swappedAxes(1, 2) + b).flattened(start: 2, end: 3))
    }

    func callAsFunction(_ x: MLXArray, state: MLXArray?) -> (MLXArray, MLXArray) {
        let inputGate = blockLinear(x, inputGateWeight, inputGateBias)
        let recurrentGate = blockLinear(x, recurrentGateWeight, recurrentGateBias)

        let logA = -8.0 * recurrentGate * softplus(recurrentParam)
        let a = exp(logA)
        let aSquared = exp(2 * logA)

        var multiplier = sqrt(1 - aSquared)
        if state == nil {
            let positions = MLXArray(0 ..< x.dim(1)).reshaped(1, x.dim(1), 1)
            multiplier = which(positions .== 0, MLXArray(Float(1)), multiplier)
        }

        return recurrentGemmaScan(
            x: x * inputGate * multiplier.asType(x.dtype), a: a, state: state)
    }
}

/// The two shapes a Griffin temporal block can take.
protocol RecurrentGemmaTemporalBlock: Module {
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray
}

class RecurrentGemmaRecurrentBlock: Module, RecurrentGemmaTemporalBlock {

    @ModuleInfo(key: "linear_y") var linearY: Linear
    @ModuleInfo(key: "linear_x") var linearX: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    @ModuleInfo(key: "conv_1d") var conv: RecurrentGemmaConv1d
    @ModuleInfo(key: "rg_lru") var rgLRU: RecurrentGemmaRGLRU

    init(width: Int, heads: Int, lruWidth: Int, convolutionWidth: Int) {
        _linearY.wrappedValue = Linear(width, lruWidth)
        _linearX.wrappedValue = Linear(width, lruWidth)
        _linearOut.wrappedValue = Linear(lruWidth, width)
        _conv.wrappedValue = RecurrentGemmaConv1d(
            channels: lruWidth, kernelSize: convolutionWidth)
        _rgLRU.wrappedValue = RecurrentGemmaRGLRU(width: lruWidth, heads: heads)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let gate = geluApproximate(linearY(x))

        let slots = cache as? ArraysCache
        var hidden = linearX(x)
        let (convolved, convolutionState) = conv(hidden, state: slots?[0])
        slots?[0] = convolutionState
        let (recurred, recurrentState) = rgLRU(convolved, state: slots?[1])
        slots?[1] = recurrentState
        hidden = recurred

        return linearOut(hidden * gate)
    }
}

/// Windowed multi-query attention: one key head and one value head, whatever
/// `num_key_value_heads` says, and a rotary over half the head dimension.
class RecurrentGemmaLocalAttentionBlock: Module, RecurrentGemmaTemporalBlock {

    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    init(width: Int, heads: Int) {
        self.heads = heads
        self.headDim = width / heads
        self.scale = pow(Float(width / heads), -0.5)

        _wq.wrappedValue = Linear(width, width, bias: false)
        _wk.wrappedValue = Linear(width, headDim, bias: false)
        _wv.wrappedValue = Linear(width, headDim, bias: false)
        _wo.wrappedValue = Linear(width, width, bias: true)

        self.rope = RoPE(dimensions: headDim / 2, traditional: false)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, 1, -1).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, 1, -1).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache,
            scale: scale, mask: mask)

        return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

class RecurrentGemmaMLPBlock: Module, UnaryLayer {

    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(width: Int, expandedWidth: Int) {
        let half = expandedWidth / 2
        _upProj.wrappedValue = Linear(width, half)
        _gateProj.wrappedValue = Linear(width, half)
        _downProj.wrappedValue = Linear(half, width)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(geluApproximate(gateProj(x)) * upProj(x))
    }
}

class RecurrentGemmaResidualBlock: Module {

    @ModuleInfo(key: "temporal_pre_norm") var temporalPreNorm: RecurrentGemmaRMSNorm
    @ModuleInfo(key: "temporal_block") var temporalBlock: RecurrentGemmaTemporalBlock
    @ModuleInfo(key: "channel_pre_norm") var channelPreNorm: RecurrentGemmaRMSNorm
    @ModuleInfo(key: "mlp_block") var mlpBlock: RecurrentGemmaMLPBlock

    let isRecurrent: Bool

    init(_ args: RecurrentGemmaConfiguration, blockType: String) {
        self.isRecurrent = blockType == "recurrent"

        _temporalPreNorm.wrappedValue = RecurrentGemmaRMSNorm(dimensions: args.hiddenSize)
        _channelPreNorm.wrappedValue = RecurrentGemmaRMSNorm(dimensions: args.hiddenSize)

        if isRecurrent {
            _temporalBlock.wrappedValue = RecurrentGemmaRecurrentBlock(
                width: args.hiddenSize, heads: args.attentionHeads,
                lruWidth: args.hiddenSize, convolutionWidth: args.convolutionWidth)
        } else {
            _temporalBlock.wrappedValue = RecurrentGemmaLocalAttentionBlock(
                width: args.hiddenSize, heads: args.attentionHeads)
        }

        _mlpBlock.wrappedValue = RecurrentGemmaMLPBlock(
            width: args.hiddenSize, expandedWidth: args.intermediateSize)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let residual = temporalBlock(temporalPreNorm(x), mask: mask, cache: cache) + x
        return mlpBlock(channelPreNorm(residual)) + residual
    }
}

public class RecurrentGemmaModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [RecurrentGemmaResidualBlock]

    @ModuleInfo(key: "final_norm") var finalNorm: RecurrentGemmaRMSNorm

    let scaleByRootDimension: Bool
    let windowSize: Int

    /// Cache index the windowed mask is built from: the first `attention` entry in the
    /// `block_types` cycle, which is also the first attention layer.
    let attentionIndex: Int

    public init(_ args: RecurrentGemmaConfiguration) {
        precondition(args.vocabularySize > 0)
        precondition(
            args.blockTypes.contains("attention"),
            "block_types names no attention layer, so there is no window to mask with")

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map {
            RecurrentGemmaResidualBlock(
                args, blockType: args.blockTypes[$0 % args.blockTypes.count])
        }
        _finalNorm.wrappedValue = RecurrentGemmaRMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.scaleByRootDimension = args.scaleEmbeddingsByRootDimension
        self.windowSize = args.attentionWindowSize
        self.attentionIndex = args.blockTypes.firstIndex(of: "attention") ?? 0
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        if scaleByRootDimension {
            h = h * sqrt(Float(h.dim(-1)))
        }

        let mask = createAttentionMask(
            h: h, cache: cache?[attentionIndex], windowSize: windowSize)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return finalNorm(h)
    }
}

public class RecurrentGemmaModel: Module, LLMModel {

    public let vocabularySize: Int

    public let model: RecurrentGemmaModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: RecurrentGemmaConfiguration

    public init(_ args: RecurrentGemmaConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.model = RecurrentGemmaModelInner(args)
        _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        var logits = lmHead.map { $0(hidden) } ?? model.embedTokens.asLinear(hidden)
        if configuration.logitsSoftCap != 0 {
            logits =
                tanh(logits / configuration.logitsSoftCap) * configuration.logitsSoftCap
        }
        return logits
    }

    /// Two array slots for a recurrent layer -- the convolution's tail and the recurrent
    /// state -- and a windowed key/value cache for an attention layer.
    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        model.layers.map { layer in
            layer.isRecurrent
                ? ArraysCache(size: 2)
                : RotatingKVCache(maxSize: configuration.attentionWindowSize, keep: 0)
        }
    }

    /// A checkpoint stores the depthwise convolution as `[channels, 1, kernel]`; this tree
    /// wants `[channels, kernel, 1]`. Dropping the untied head is how upstream signals a
    /// tied embedding, since the configuration has no flag for it.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        if weights["lm_head.weight"] == nil, lmHead != nil {
            update(modules: ModuleChildren(values: ["lm_head": .none]))
        }
        var sanitized = weights
        for (key, value) in weights where key.contains("conv_1d.weight") {
            if value.dim(-1) != 1 {
                sanitized[key] = value.movedAxis(source: 2, destination: 1)
            }
        }
        return sanitized
    }
}

public struct RecurrentGemmaConfiguration: Decodable, Sendable {

    var modelType: String
    var attentionBias: Bool
    var convolutionWidth: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var logitsSoftCap: Float
    var attentionHeads: Int
    var hiddenLayers: Int
    var kvHeads: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var attentionWindowSize: Int
    var vocabularySize: Int
    var scaleEmbeddingsByRootDimension: Bool
    var blockTypes: [String]

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case attentionBias = "attention_bias"
        case convolutionWidth = "conv1d_width"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case logitsSoftCap = "logits_soft_cap"
        case attentionHeads = "num_attention_heads"
        case hiddenLayers = "num_hidden_layers"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case attentionWindowSize = "attention_window_size"
        case vocabularySize = "vocab_size"
        case scaleEmbeddingsByRootDimension = "embeddings_scale_by_sqrt_dim"
        case blockTypes = "block_types"
        case underscoreBlockTypes = "_block_types"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType) ?? "recurrent_gemma"
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        convolutionWidth = try c.decode(Int.self, forKey: .convolutionWidth)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        logitsSoftCap = try c.decode(Float.self, forKey: .logitsSoftCap)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        attentionWindowSize = try c.decode(Int.self, forKey: .attentionWindowSize)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        scaleEmbeddingsByRootDimension =
            try c.decodeIfPresent(Bool.self, forKey: .scaleEmbeddingsByRootDimension) ?? true

        blockTypes =
            try c.decodeIfPresent([String].self, forKey: .blockTypes)
            ?? c.decodeIfPresent([String].self, forKey: .underscoreBlockTypes)
            ?? ["recurrent", "recurrent", "attention"]
    }
}

extension RecurrentGemmaModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
