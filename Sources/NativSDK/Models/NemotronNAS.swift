import Foundation
import MLX
import MLXNN

/// The two shapes a `self_attn` slot can take.
protocol NemotronNASAttentionLayer: Module {
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray

    /// Whether the slot reads and writes the cache entry it is given.
    var attends: Bool { get }
}

/// A whole subblock collapsed to one linear map, which is how the search prunes without
/// changing the residual width. Serves both the attention and the feed-forward slot.
class NemotronNASLinearSubblock: Module, NemotronNASAttentionLayer, UnaryLayer {

    @ModuleInfo(key: "linear") var linear: Linear

    var attends: Bool { false }

    init(hiddenSize: Int, bias: Bool) {
        _linear.wrappedValue = Linear(hiddenSize, hiddenSize, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        linear(x)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        linear(x)
    }
}

class NemotronNASAttention: Module, NemotronNASAttentionLayer {

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer

    var attends: Bool { true }

    init(_ args: NemotronNASConfiguration, headsInGroup: Int) {
        let dim = args.hiddenSize
        self.heads = args.attentionHeads
        self.kvHeads = args.attentionHeads / headsInGroup
        self.headDim = dim / args.attentionHeads
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(dim, heads * headDim, bias: args.attentionBias)
        _wk.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        _wv.wrappedValue = Linear(dim, kvHeads * headDim, bias: args.attentionBias)
        _wo.wrappedValue = Linear(heads * headDim, dim, bias: args.attentionBias)

        self.rope = initializeRope(
            dims: headDim, base: args.ropeTheta, traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x).reshaped(B, L, heads, headDim).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache,
            scale: scale, mask: mask)

        return wo(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

class NemotronNASMLP: Module, UnaryLayer {

    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    let activation: @Sendable (MLXArray) -> MLXArray

    init(_ args: NemotronNASConfiguration, multiplier: Double) {
        let hidden = NemotronNASConfiguration.intermediateSize(
            multiplier: multiplier, hiddenSize: args.hiddenSize)
        _gateProj.wrappedValue = Linear(args.hiddenSize, hidden, bias: args.mlpBias)
        _upProj.wrappedValue = Linear(args.hiddenSize, hidden, bias: args.mlpBias)
        _downProj.wrappedValue = Linear(hidden, args.hiddenSize, bias: args.mlpBias)
        self.activation = args.activation
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(activation(gateProj(x)) * upProj(x))
    }
}

class NemotronNASBlock: Module {

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm?
    @ModuleInfo(key: "self_attn") var attention: NemotronNASAttentionLayer?
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm?
    @ModuleInfo(key: "mlp") var mlp: UnaryLayer?

    /// Whether this layer consumes a cache entry, which is true for a linear replacement
    /// as well -- it takes the entry and leaves it untouched.
    var usesCacheEntry: Bool { attention?.attends == true }

    let kvHeads: Int

    init(_ args: NemotronNASConfiguration, layerIndex: Int) {
        let block = args.blockConfigurations[layerIndex]

        if !block.attention.noOp {
            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            if block.attention.replaceWithLinear {
                _attention.wrappedValue = NemotronNASLinearSubblock(
                    hiddenSize: args.hiddenSize, bias: args.attentionBias)
                self.kvHeads = args.attentionHeads
            } else {
                let headsInGroup = block.attention.headsInGroup ?? 1
                _attention.wrappedValue = NemotronNASAttention(
                    args, headsInGroup: headsInGroup)
                self.kvHeads = args.attentionHeads / headsInGroup
            }
        } else {
            self.kvHeads = args.attentionHeads
        }

        if !block.ffn.noOp {
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            if block.ffn.replaceWithLinear {
                _mlp.wrappedValue = NemotronNASLinearSubblock(
                    hiddenSize: args.hiddenSize, bias: args.mlpBias)
            } else {
                _mlp.wrappedValue = NemotronNASMLP(
                    args, multiplier: block.ffn.multiplier ?? 0)
            }
        }
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        var h = x
        if let attention, let inputLayerNorm {
            h = h + attention(inputLayerNorm(h), mask: mask, cache: cache)
        }
        if let mlp, let postAttentionLayerNorm {
            h = h + mlp(postAttentionLayerNorm(h))
        }
        return h
    }
}

public class NemotronNASModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [NemotronNASBlock]
    let norm: RMSNorm

    public init(_ args: NemotronNASConfiguration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        let layers = (0 ..< args.hiddenLayers).map {
            NemotronNASBlock(args, layerIndex: $0)
        }
        self.layers = layers
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)

        var cacheIndex = 0
        for layer in layers {
            var entry: KVCache? = nil
            if layer.usesCacheEntry {
                entry = cache?[cacheIndex]
                cacheIndex += 1
            }
            h = layer(h, mask: mask, cache: entry)
        }

        return norm(h)
    }
}

public class NemotronNASModel: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: NemotronNASModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let tieWordEmbeddings: Bool

    public init(_ args: NemotronNASConfiguration) {
        self.vocabularySize = args.vocabularySize
        self.tieWordEmbeddings = args.tieWordEmbeddings
        self.model = NemotronNASModelInner(args)
        self.kvHeads = model.layers.filter(\.usesCacheEntry).map(\.kvHeads)
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

    /// One entry per cache-consuming layer, not per layer.
    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try model.layers.filter(\.usesCacheEntry).map { _ in
            try makeAttentionKVCache(configuration: configuration)
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        guard tieWordEmbeddings else { return weights }
        return weights.filter { $0.key != "lm_head.weight" }
    }
}

public struct NemotronNASConfiguration: Decodable, Sendable {

    public struct AttentionBlock: Decodable, Sendable {
        var noOp: Bool
        var replaceWithLinear: Bool
        var headsInGroup: Int?

        enum CodingKeys: String, CodingKey {
            case noOp = "no_op"
            case replaceWithLinear = "replace_with_linear"
            case headsInGroup = "n_heads_in_group"
        }

        init(noOp: Bool, replaceWithLinear: Bool, headsInGroup: Int?) {
            self.noOp = noOp
            self.replaceWithLinear = replaceWithLinear
            self.headsInGroup = headsInGroup
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            noOp = try c.decodeIfPresent(Bool.self, forKey: .noOp) ?? false
            replaceWithLinear =
                try c.decodeIfPresent(Bool.self, forKey: .replaceWithLinear) ?? false
            let group = try c.decodeIfPresent(Int.self, forKey: .headsInGroup)
            headsInGroup = (noOp || replaceWithLinear) ? nil : group
        }
    }

    public struct FeedForwardBlock: Decodable, Sendable {
        var noOp: Bool
        var replaceWithLinear: Bool
        var multiplier: Double?

        enum CodingKeys: String, CodingKey {
            case noOp = "no_op"
            case replaceWithLinear = "replace_with_linear"
            case multiplier = "ffn_mult"
        }

        init(noOp: Bool, replaceWithLinear: Bool, multiplier: Double?) {
            self.noOp = noOp
            self.replaceWithLinear = replaceWithLinear
            self.multiplier = multiplier
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            noOp = try c.decodeIfPresent(Bool.self, forKey: .noOp) ?? false
            replaceWithLinear =
                try c.decodeIfPresent(Bool.self, forKey: .replaceWithLinear) ?? false
            let mult = try c.decodeIfPresent(Double.self, forKey: .multiplier)
            multiplier = (noOp || replaceWithLinear) ? nil : mult
        }
    }

    public struct BlockConfiguration: Decodable, Sendable {
        var attention: AttentionBlock
        var ffn: FeedForwardBlock

        enum CodingKeys: String, CodingKey {
            case attention
            case ffn
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            attention =
                try c.decodeIfPresent(AttentionBlock.self, forKey: .attention)
                ?? AttentionBlock(noOp: false, replaceWithLinear: false, headsInGroup: nil)
            ffn =
                try c.decodeIfPresent(FeedForwardBlock.self, forKey: .ffn)
                ?? FeedForwardBlock(noOp: false, replaceWithLinear: false, multiplier: nil)
        }
    }

    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var blockConfigurations: [BlockConfiguration]
    var hiddenAct: String
    var attentionBias: Bool
    var mlpBias: Bool
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var maxPositionEmbeddings: Int
    var tieWordEmbeddings: Bool

    /// The activation named by `hidden_act`.
    ///
    /// `gelu_new` and `gelu_fast` both select the tanh approximation -- `gelu_fast` does
    /// *not* mean the sigmoid form here.
    var activation: @Sendable (MLXArray) -> MLXArray {
        switch hiddenAct {
        case "relu": return relu
        case "gelu": return gelu
        case "gelu_new", "gelu_fast": return geluApproximate
        default: return silu
        }
    }

    /// `ffn_mult` to a feed-forward width: two thirds of the product, truncated, then
    /// rounded *up* to a multiple of 256.
    public static func intermediateSize(multiplier: Double, hiddenSize: Int) -> Int {
        let rounded = (multiplier * 1e6).rounded(.toNearestOrEven) / 1e6
        let size = Int(2 * rounded * Double(hiddenSize) / 3)
        let step = 256
        return size % step == 0 ? size : size + step - (size % step)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case blockConfigurations = "block_configs"
        case hiddenAct = "hidden_act"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case maxPositionEmbeddings = "max_position_embeddings"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "nemotron-nas"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 8192
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 80
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 64
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 128_256
        blockConfigurations =
            try c.decodeIfPresent([BlockConfiguration].self, forKey: .blockConfigurations)
            ?? []
        hiddenAct = try c.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 500_000
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        precondition(
            blockConfigurations.count == hiddenLayers,
            "block_configs has \(blockConfigurations.count) entries for "
                + "\(hiddenLayers) layers")
    }
}

extension NemotronNASModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
