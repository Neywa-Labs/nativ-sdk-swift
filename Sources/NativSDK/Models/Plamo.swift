import Foundation
import MLX
import MLXNN

class PlamoAttention: Module {
    let queryHeads: Int
    let kvHeads: Int
    let headDimensions: Int
    let sharedHeads: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "rotary_emb") var rope: RoPE

    init(_ config: PlamoConfiguration) {
        self.queryHeads = config.attentionHeads
        self.headDimensions = config.hiddenSize / config.attentionHeads
        self.sharedHeads = config.sharedHeads
        self.kvHeads = config.resolvedKVHeads
        self.scale = pow(Float(headDimensions), -0.5)

        _wq.wrappedValue = Linear(
            config.hiddenSize, queryHeads * headDimensions, bias: false)
        _wk.wrappedValue = Linear(config.hiddenSize, kvHeads * headDimensions, bias: false)
        _wv.wrappedValue = Linear(config.hiddenSize, kvHeads * headDimensions, bias: false)
        _wo.wrappedValue = Linear(
            queryHeads * headDimensions, config.hiddenSize, bias: false)

        _rope.wrappedValue = RoPE(
            dimensions: headDimensions, traditional: config.ropeTraditional,
            base: config.ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x).reshaped(B, L, queryHeads, -1).transposed(0, 2, 1, 3)
        var keys = wk(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        var values = wv(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        // Tiled along the head axis, so the shared set repeats as a block. A head-wise
        // repeat would give the same shape and a different pairing.
        keys = tiled(keys, repetitions: [1, sharedHeads, 1, 1])
        values = tiled(values, repetitions: [1, sharedHeads, 1, 1])

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

class PlamoBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: PlamoAttention
    let mlp: SwiGLUMLP
    let norm: RMSNorm

    init(_ config: PlamoConfiguration) {
        _attention.wrappedValue = PlamoAttention(config)
        self.mlp = SwiGLUMLP(
            inputDimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        // One norm per layer, feeding both branches.
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = norm(x)
        return x + attention(h, mask: mask, cache: cache) + mlp(h)
    }
}

/// Holds the layer list, which is why a published key reads `layers.layers.N`.
public class PlamoDecoder: Module {
    let layers: [PlamoBlock]

    public init(_ config: PlamoConfiguration) {
        self.layers = (0 ..< config.hiddenLayers).map { _ in PlamoBlock(config) }
    }
}

public class PlamoModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: PlamoDecoder
    let norm: RMSNorm

    public init(_ config: PlamoConfiguration) {
        precondition(config.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = PlamoDecoder(config)
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class PlamoModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: PlamoModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: PlamoConfiguration) {
        self.vocabularySize = config.vocabularySize
        // The cache holds the untiled heads; the tiling happens after it is read.
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.resolvedKVHeads }
        self.model = PlamoModelInner(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }
}

public struct PlamoConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var sharedHeads: Int
    var ropeTheta: Float
    var ropeTraditional: Bool

    /// How many key-value heads are stored before tiling.
    ///
    /// Rounded *up*, so a head count that does not divide by the sharing factor keeps an
    /// extra head rather than losing one -- and the tiled result can then be wider than the
    /// query head count, which upstream also allows.
    var resolvedKVHeads: Int {
        Int((Double(attentionHeads) / Double(sharedHeads)).rounded(.up))
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case sharedHeads = "n_shared_head"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "plamo"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        sharedHeads = try c.decodeIfPresent(Int.self, forKey: .sharedHeads) ?? 8
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
    }
}

extension PlamoModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers.layers
    }
}
