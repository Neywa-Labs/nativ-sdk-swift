import Foundation
import MLX
import MLXNN

/// LayerNorm whose stored weight is one less than the scale it applies.
///
/// Subclassing rather than wrapping keeps the parameter key at `weight`, which is what the
/// checkpoint carries.
public class NemotronLayerNorm1P: LayerNorm {
    public override func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: weight.map { $0 + 1 }, bias: bias, eps: eps)
    }
}

class NemotronAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE

    /// How many of each head's dimensions rotate, from `partial_rotary_factor`.
    let rotaryDimensions: Int

    init(_ args: NemotronConfiguration) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDimensions = args.resolvedHeadDimensions
        self.scale = pow(Float(args.resolvedHeadDimensions), -0.5)

        _wq.wrappedValue = Linear(
            args.hiddenSize, heads * headDimensions, bias: args.attentionBias)
        _wk.wrappedValue = Linear(
            args.hiddenSize, kvHeads * headDimensions, bias: args.attentionBias)
        _wv.wrappedValue = Linear(
            args.hiddenSize, kvHeads * headDimensions, bias: args.attentionBias)
        _wo.wrappedValue = Linear(
            heads * headDimensions, args.hiddenSize, bias: args.attentionBias)

        var ropeScale: Float = 1
        if let scaling = args.ropeScaling, let factor = scaling["factor"]?.asFloat() {
            ropeScale = 1 / factor
        }

        self.rotaryDimensions = Int(args.partialRotaryFactor * Float(headDimensions))
        self.rope = RoPE(
            dimensions: rotaryDimensions, traditional: args.ropeTraditional,
            base: args.ropeTheta, scale: ropeScale)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
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

class NemotronMLP: Module, UnaryLayer {
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear

    init(_ args: NemotronConfiguration) {
        _up.wrappedValue = Linear(
            args.hiddenSize, args.intermediateSize, bias: args.mlpBias)
        _down.wrappedValue = Linear(
            args.intermediateSize, args.hiddenSize, bias: args.mlpBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Ungated: squared ReLU on one projection, not silu(gate) * up. There is no
        // gate_proj in the checkpoint to bind.
        let activated = relu(up(x))
        return down(activated * activated)
    }
}

class NemotronBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: NemotronAttention
    let mlp: NemotronMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: NemotronLayerNorm1P
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: NemotronLayerNorm1P

    init(_ args: NemotronConfiguration) {
        _attention.wrappedValue = NemotronAttention(args)
        self.mlp = NemotronMLP(args)
        _inputLayerNorm.wrappedValue = NemotronLayerNorm1P(
            dimensions: args.hiddenSize, eps: args.normEps)
        _postAttentionLayerNorm.wrappedValue = NemotronLayerNorm1P(
            dimensions: args.hiddenSize, eps: args.normEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class NemotronModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [NemotronBlock]
    let norm: NemotronLayerNorm1P

    public init(_ args: NemotronConfiguration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { _ in NemotronBlock(args) }
        self.norm = NemotronLayerNorm1P(dimensions: args.hiddenSize, eps: args.normEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class NemotronModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: NemotronModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: NemotronConfiguration

    public init(_ args: NemotronConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = NemotronModelInner(args)
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

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        guard configuration.tieWordEmbeddings else { return weights }
        return weights.filter { $0.key != "lm_head.weight" }
    }
}

public struct NemotronConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenActivation: String
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var normEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var headDimensions: Int?
    var maxPositionEmbeddings: Int?
    var attentionBias: Bool
    var mlpBias: Bool
    var partialRotaryFactor: Float
    var ropeTheta: Float
    var ropeTraditional: Bool
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool

    var resolvedHeadDimensions: Int {
        headDimensions ?? (hiddenSize / attentionHeads)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenActivation = "hidden_act"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case normEps = "norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case headDimensions = "head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case partialRotaryFactor = "partial_rotary_factor"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "nemotron"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenActivation =
            try c.decodeIfPresent(String.self, forKey: .hiddenActivation)
            ?? "relu2"
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        normEps = try c.decode(Float.self, forKey: .normEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        headDimensions = try c.decodeIfPresent(Int.self, forKey: .headDimensions)
        maxPositionEmbeddings = try c.decodeIfPresent(
            Int.self, forKey: .maxPositionEmbeddings)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        partialRotaryFactor =
            try c.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.5
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        // Upstream raises on both of these rather than falling back to default positions,
        // and accepts the factor under either `type` or `rope_type`.
        if let scaling = ropeScaling {
            guard scaling["factor"] != nil else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: c.codingPath,
                        debugDescription: "rope_scaling must contain 'factor'"))
            }
            let kind = scaling["type"] ?? scaling["rope_type"]
            guard let kind else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: c.codingPath,
                        debugDescription:
                            "rope_scaling must contain either 'type' or 'rope_type'"))
            }
            guard case .string("linear") = kind else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: c.codingPath,
                        debugDescription: "rope_scaling 'type' supports only 'linear'"))
            }
        }
    }
}

extension NemotronModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
