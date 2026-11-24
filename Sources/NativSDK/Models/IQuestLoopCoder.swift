import Foundation
import MLX
import MLXNN

/// The per-head mix between the global and the local attention.
///
/// `weight` is `[heads, head_dim]` and `bias` is `[heads]`, so the result is
/// `[batch, heads, positions, 1]` and broadcasts across the head dimension.
class IQuestLoopCoderGate: Module {

    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray

    init(heads: Int, headDimensions: Int) {
        _weight.wrappedValue = MLXArray.zeros([heads, headDimensions])
        _bias.wrappedValue = MLXArray.zeros([heads])
    }

    func callAsFunction(_ queries: MLXArray) -> MLXArray {
        let contracted = matmul(
            queries, weight.expandedDimensions(axis: 1).swappedAxes(-1, -2))
        return sigmoid(contracted + bias.reshaped(-1, 1, 1))
    }
}

/// Projections and rotary only: the loop drives the attention itself, and applies
/// `o_proj` after mixing, so there is no single forward here.
class IQuestLoopCoderAttention: Module {

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer

    init(_ args: IQuestLoopCoderConfiguration) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDim = args.headDimensions
        self.scale = pow(Float(args.headDimensions), -0.5)

        _wq.wrappedValue = Linear(
            args.hiddenSize, heads * headDim, bias: args.attentionBias)
        _wk.wrappedValue = Linear(
            args.hiddenSize, kvHeads * headDim, bias: args.attentionBias)
        _wv.wrappedValue = Linear(
            args.hiddenSize, kvHeads * headDim, bias: args.attentionBias)
        _wo.wrappedValue = Linear(
            heads * headDim, args.hiddenSize, bias: args.attentionBias)

        self.rope = initializeRope(
            dims: headDim, base: args.ropeTheta, traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)
    }

    func projected(_ x: MLXArray, offset: Int) -> (
        queries: MLXArray, keys: MLXArray, values: MLXArray
    ) {
        let (B, L) = (x.dim(0), x.dim(1))
        let queries = wq(x).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        let keys = wk(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        return (
            applyRotaryPosition(rope, to: queries, offset: .scalar(offset)),
            applyRotaryPosition(rope, to: keys, offset: .scalar(offset)),
            values
        )
    }

    func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
    }
}

class IQuestLoopCoderBlock: Module {

    @ModuleInfo(key: "self_attn") var attention: IQuestLoopCoderAttention
    @ModuleInfo(key: "mlp") var mlp: SwiGLUMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: IQuestLoopCoderConfiguration) {
        _attention.wrappedValue = IQuestLoopCoderAttention(args)
        _mlp.wrappedValue = SwiGLUMLP(
            inputDimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize,
            bias: args.mlpBias)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }
}

public class IQuestLoopCoderModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [IQuestLoopCoderBlock]
    let norm: RMSNorm

    @ModuleInfo(key: "gate_projections") var gateProjections: [IQuestLoopCoderGate]

    let windowSize: Int

    public init(_ args: IQuestLoopCoderConfiguration) {
        precondition(args.vocabularySize > 0)
        precondition(args.loopCount == 2, "only loop_num=2 is supported")

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { _ in IQuestLoopCoderBlock(args) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _gateProjections.wrappedValue = (0 ..< args.hiddenLayers).map { _ in
            IQuestLoopCoderGate(
                heads: args.attentionHeads, headDimensions: args.headDimensions)
        }
        self.windowSize = args.loopWindowSize
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let (B, L) = (h.dim(0), h.dim(1))
        let count = layers.count

        let globalMask = createAttentionMask(h: h, cache: cache?[0])
        let windowMask = createAttentionMask(
            h: h, cache: cache?[count], windowSize: windowSize)

        var recorded: [(keys: MLXArray, values: MLXArray)] = []
        recorded.reserveCapacity(count)

        for (i, layer) in layers.enumerated() {
            let entry = cache?[i]
            let normalized = layer.inputLayerNorm(h)
            let projected = layer.attention.projected(
                normalized, offset: entry?.offset ?? 0)
            var keys = projected.keys
            var values = projected.values
            if let entry {
                (keys, values) = entry.update(keys: keys, values: values)
            }
            recorded.append((keys, values))

            let attended = layer.attention.attend(
                queries: projected.queries, keys: keys, values: values, mask: globalMask)
            h = h + layer.attention.wo(attended.transposed(0, 2, 1, 3).reshaped(B, L, -1))
            h = h + layer.mlp(layer.postAttentionLayerNorm(h))
        }

        for (i, layer) in layers.enumerated() {
            let entry = cache?[count + i]
            let normalized = layer.inputLayerNorm(h)
            let projected = layer.attention.projected(
                normalized, offset: entry?.offset ?? 0)
            let queries = projected.queries

            let gate = gateProjections[i](queries)

            let global = layer.attention.attend(
                queries: queries, keys: recorded[i].keys, values: recorded[i].values,
                mask: globalMask)

            var localKeys = projected.keys
            var localValues = projected.values
            if let entry {
                (localKeys, localValues) = entry.update(
                    keys: localKeys, values: localValues)
            }
            let local = layer.attention.attend(
                queries: queries, keys: localKeys, values: localValues, mask: windowMask)

            let mixed = gate * global + (1 - gate) * local
            h = h + layer.attention.wo(mixed.transposed(0, 2, 1, 3).reshaped(B, L, -1))
            h = h + layer.mlp(layer.postAttentionLayerNorm(h))
        }

        return norm(h)
    }
}

public class IQuestLoopCoderModel: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: IQuestLoopCoderModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: IQuestLoopCoderConfiguration

    public init(_ args: IQuestLoopCoderConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.kvHeads, count: 2 * args.hiddenLayers)
        self.model = IQuestLoopCoderModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try model.layers.map { _ in
            MaterializedKVCache(try makeAttentionKVCache(configuration: configuration))
        }
            + model.layers.map { _ in
                try makeSlidingWindowKVCache(
                    configuration: configuration, window: self.configuration.loopWindowSize)
            }
    }
}

public struct IQuestLoopCoderConfiguration: Decodable, Sendable {

    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var headDimensions: Int
    var kvHeads: Int
    var maxPositionEmbeddings: Int
    var attentionBias: Bool
    var mlpBias: Bool
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool
    var loopCount: Int
    var loopWindowSize: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case headDimensions = "head_dim"
        case kvHeads = "num_key_value_heads"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
        case loopCount = "loop_num"
        case loopWindowSize = "loop_window_size"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType) ?? "iquestloopcoder"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        headDimensions = try c.decode(Int.self, forKey: .headDimensions)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        loopCount = try c.decodeIfPresent(Int.self, forKey: .loopCount) ?? 2
        loopWindowSize = try c.decodeIfPresent(Int.self, forKey: .loopWindowSize) ?? 64
    }
}

extension IQuestLoopCoderModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
