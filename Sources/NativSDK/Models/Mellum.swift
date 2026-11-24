import Foundation
import MLX
import MLXNN

class MellumAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ args: MellumConfiguration, layerType: String) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDimensions = args.headDimensions
        self.scale = pow(Float(args.headDimensions), -0.5)

        _wq.wrappedValue = Linear(
            args.hiddenSize, heads * headDimensions, bias: false)
        _wk.wrappedValue = Linear(args.hiddenSize, kvHeads * headDimensions, bias: false)
        _wv.wrappedValue = Linear(args.hiddenSize, kvHeads * headDimensions, bias: false)
        _wo.wrappedValue = Linear(heads * headDimensions, args.hiddenSize, bias: false)

        // Per head, before the transpose.
        _qNorm.wrappedValue = RMSNorm(dimensions: headDimensions, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDimensions, eps: args.rmsNormEps)

        self.rope = args.rope(forLayerType: layerType)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = qNorm(wq(x).reshaped(B, L, heads, -1)).transposed(0, 2, 1, 3)
        var keys = kNorm(wk(x).reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

class MellumSparseMoeBlock: Module, UnaryLayer {
    let topK: Int
    let normalizesTopK: Bool

    @ModuleInfo var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(_ args: MellumConfiguration) {
        self.topK = args.expertsPerToken
        self.normalizesTopK = args.normalizeTopKProbabilities

        _gate.wrappedValue = Linear(args.hiddenSize, args.experts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
            numExperts: args.experts)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = softmax(gate(x), axis: -1, precise: true)
        // Ascending partition, then the tail: the largest, without negating.
        let indices = MLX.argPartition(gates, kth: gates.dim(-1) - topK, axis: -1)[
            .ellipsis, (-topK)...]
        var scores = takeAlong(gates, indices, axis: -1)
        if normalizesTopK {
            // Softmax over everything, then renormalize the selection -- between
            // qwen2_moe, which leaves the mass short of one, and phixtral, which softmaxes
            // only the selection.
            scores = scores / scores.sum(axis: -1, keepDims: true)
        }
        return (switchMLP(x, indices) * scores[.ellipsis, .newAxis]).sum(axis: -2)
    }
}

class MellumBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: MellumAttention
    @ModuleInfo(key: "mlp") var mlp: MellumSparseMoeBlock

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let usesSliding: Bool

    init(_ args: MellumConfiguration, layerType: String) {
        self.usesSliding = layerType == "sliding_attention"
        _attention.wrappedValue = MellumAttention(args, layerType: layerType)
        _mlp.wrappedValue = MellumSparseMoeBlock(args)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class MellumModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [MellumBlock]
    let norm: RMSNorm

    let configuration: MellumConfiguration

    public init(_ args: MellumConfiguration) {
        precondition(args.vocabularySize > 0)
        self.configuration = args
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = args.layerTypes.map { MellumBlock(args, layerType: $0) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let firstFull = layers.firstIndex { !$0.usesSliding }
        let firstSliding = layers.firstIndex { $0.usesSliding }
        let fullMask =
            firstFull.map { createAttentionMask(h: h, cache: cache?[$0]) } ?? .none
        let slidingMask =
            firstSliding.map {
                createAttentionMask(
                    h: h, cache: cache?[$0], windowSize: configuration.slidingWindow)
            } ?? .none

        for (i, layer) in layers.enumerated() {
            h = layer(
                h, mask: layer.usesSliding ? slidingMask : fullMask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class MellumModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: MellumModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: MellumConfiguration

    public init(_ args: MellumConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = args.layerTypes.map { _ in args.kvHeads }
        self.model = MellumModelInner(args)
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
        try model.layers.map { layer in
            try makeHybridAttentionKVCache(
                configuration: configuration,
                slidingWindow: self.configuration.slidingWindow,
                usesSlidingWindow: layer.usesSliding)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        if sanitized["model.layers.0.mlp.experts.0.up_proj.weight"] != nil {
            for layer in 0 ..< configuration.layerTypes.count {
                let prefix = "model.layers.\(layer)"
                for projection in ["up_proj", "down_proj", "gate_proj"] {
                    var stacked: [MLXArray] = []
                    for expert in 0 ..< configuration.experts {
                        let key = "\(prefix).mlp.experts.\(expert).\(projection).weight"
                        guard let value = sanitized.removeValue(forKey: key) else { break }
                        stacked.append(value)
                    }
                    guard stacked.count == configuration.experts else { continue }
                    sanitized["\(prefix).mlp.switch_mlp.\(projection).weight"] =
                        MLX.stacked(stacked)
                }
            }
        }

        if configuration.tieWordEmbeddings {
            sanitized = sanitized.filter { $0.key != "lm_head.weight" }
        }
        return sanitized
    }
}

public struct MellumConfiguration: Decodable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var experts: Int
    var expertsPerToken: Int
    var moeIntermediateSize: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var headDimensions: Int
    var tieWordEmbeddings: Bool
    var maxPositionEmbeddings: Int
    var normalizeTopKProbabilities: Bool
    var slidingWindow: Int
    var layerTypes: [String]

    /// One entry per layer *type*, not per layer.
    var ropeParameters: [String: [String: StringOrNumber]]

    /// The rope a layer of this type gets.
    ///
    /// A "default" or "linear" kind means no scaling dictionary at all; anything else keeps
    /// the parameters and hands them over as the scaling, with the kind copied to `type`
    /// where `initializeRope` looks for it.
    func rope(forLayerType layerType: String) -> RoPELayer {
        let parameters = ropeParameters[layerType] ?? [:]

        var base: Float = 10000
        switch parameters["rope_theta"] {
        case .float(let value): base = value
        case .int(let value): base = Float(value)
        default: break
        }

        var kind = "default"
        if case .string(let name) = parameters["rope_type"] { kind = name }

        if kind == "default" || kind == "linear" {
            return initializeRope(
                dims: headDimensions, base: base, traditional: false, scalingConfig: nil,
                maxPositionEmbeddings: nil)
        }

        var scaling = parameters
        scaling["type"] = .string(kind)
        return initializeRope(
            dims: headDimensions, base: base, traditional: false, scalingConfig: scaling,
            maxPositionEmbeddings: maxPositionEmbeddings)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case experts = "num_experts"
        case expertsPerToken = "num_experts_per_tok"
        case moeIntermediateSize = "moe_intermediate_size"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case headDimensions = "head_dim"
        case tieWordEmbeddings = "tie_word_embeddings"
        case maxPositionEmbeddings = "max_position_embeddings"
        case normalizeTopKProbabilities = "norm_topk_prob"
        case slidingWindow = "sliding_window"
        case layerTypes = "layer_types"
        case ropeParameters = "rope_parameters"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "mellum"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        experts = try c.decode(Int.self, forKey: .experts)
        expertsPerToken = try c.decode(Int.self, forKey: .expertsPerToken)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDimensions = try c.decode(Int.self, forKey: .headDimensions)
        tieWordEmbeddings = try c.decode(Bool.self, forKey: .tieWordEmbeddings)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        normalizeTopKProbabilities = try c.decode(
            Bool.self, forKey: .normalizeTopKProbabilities)
        slidingWindow = try c.decode(Int.self, forKey: .slidingWindow)
        layerTypes = try c.decode([String].self, forKey: .layerTypes)
        ropeParameters =
            try c.decodeIfPresent(
                [String: [String: StringOrNumber]].self, forKey: .ropeParameters) ?? [:]
    }
}

extension MellumModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
