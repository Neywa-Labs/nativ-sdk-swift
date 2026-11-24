import Foundation
import MLX
import MLXNN

class QwenAttention: Module {
    let heads: Int
    let headDimensions: Int
    let scale: Float

    @ModuleInfo(key: "c_attn") var cAttn: Linear
    @ModuleInfo(key: "c_proj") var cProj: Linear

    @ModuleInfo(key: "rotary_emb") var rope: RoPE

    init(_ config: QwenConfiguration) {
        self.heads = config.attentionHeads
        // The rotary width and the scale come from the even split of hidden_size, while the
        // projection width comes from kv_channels. They coincide on every published config
        // and upstream keeps them separate, so this does too.
        self.headDimensions = config.hiddenSize / config.attentionHeads
        self.scale = pow(Float(headDimensions), -0.5)

        let projectionSize = config.kvChannels * config.attentionHeads
        _cAttn.wrappedValue = Linear(config.hiddenSize, projectionSize * 3, bias: true)
        _cProj.wrappedValue = Linear(config.hiddenSize, projectionSize, bias: !config.noBias)

        _rope.wrappedValue = RoPE(dimensions: headDimensions, traditional: false)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        // Split per role, not per head: the three thirds of `c_attn` are all the queries,
        // then all the keys, then all the values. gpt_neox interleaves instead.
        let parts = split(cAttn(x), parts: 3, axis: -1)
        let (B, L) = (parts[0].dim(0), parts[0].dim(1))

        var queries = parts[0].reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        var keys = parts[1].reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        let values = parts[2].reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return cProj(output)
    }
}

class QwenMLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "c_proj") var cProj: Linear

    init(_ config: QwenConfiguration) {
        // Half the configured width, and the halving is upstream's, not a typo here.
        let hidden = config.intermediateSize / 2
        _w1.wrappedValue = Linear(config.hiddenSize, hidden, bias: !config.noBias)
        _w2.wrappedValue = Linear(config.hiddenSize, hidden, bias: !config.noBias)
        _cProj.wrappedValue = Linear(hidden, config.hiddenSize, bias: !config.noBias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // `swiglu(a2, a1)` upstream, so the gate is w2 and the linear branch is w1 -- the
        // reverse of the usual reading of these names. Swapping them loads identically and
        // changes every activation.
        cProj(silu(w2(x)) * w1(x))
    }
}

class QwenBlock: Module {
    @ModuleInfo(key: "ln_1") var ln1: RMSNorm
    @ModuleInfo(key: "attn") var attention: QwenAttention
    @ModuleInfo(key: "ln_2") var ln2: RMSNorm
    let mlp: QwenMLP

    init(_ config: QwenConfiguration) {
        _ln1.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
        _attention.wrappedValue = QwenAttention(config)
        _ln2.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
        self.mlp = QwenMLP(config)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(ln1(x), mask: mask, cache: cache)
        return h + mlp(ln2(h))
    }
}

public class QwenTransformer: Module {
    @ModuleInfo(key: "wte") var wte: Embedding

    let h: [QwenBlock]

    @ModuleInfo(key: "ln_f") var lnF: RMSNorm

    public init(_ config: QwenConfiguration) {
        precondition(config.vocabularySize > 0)
        _wte.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.h = (0 ..< config.hiddenLayers).map { _ in QwenBlock(config) }
        _lnF.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEpsilon)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden = wte(inputs)
        let mask = createAttentionMask(h: hidden, cache: cache?.first)

        for (i, layer) in h.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[i])
        }

        return lnF(hidden)
    }
}

public class QwenLanguageModel: Module {
    public let transformer: QwenTransformer

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(_ config: QwenConfiguration) {
        self.transformer = QwenTransformer(config)
        _lmHead.wrappedValue = Linear(
            config.hiddenSize, config.vocabularySize, bias: !config.noBias)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(transformer(inputs, cache: cache))
    }
}

public class QwenModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: QwenLanguageModel

    public init(_ config: QwenConfiguration) {
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        _languageModel.wrappedValue = QwenLanguageModel(config)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            var name = key
            if name.hasPrefix("language_model.") {
                name = String(name.dropFirst("language_model.".count))
            }
            sanitized["language_model." + name] = value
        }
        return sanitized
    }
}

public struct QwenConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var attentionHeads: Int
    var hiddenLayers: Int
    var kvChannels: Int
    var maxPositionEmbeddings: Int
    var layerNormEpsilon: Float
    var intermediateSize: Int
    var noBias: Bool
    var vocabularySize: Int
    var kvHeads: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case attentionHeads = "num_attention_heads"
        case hiddenLayers = "num_hidden_layers"
        case kvChannels = "kv_channels"
        case maxPositionEmbeddings = "max_position_embeddings"
        case layerNormEpsilon = "layer_norm_epsilon"
        case intermediateSize = "intermediate_size"
        case noBias = "no_bias"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "qwen"
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 24
        kvChannels = try c.decodeIfPresent(Int.self, forKey: .kvChannels) ?? 128
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 8192
        layerNormEpsilon = try c.decodeIfPresent(Float.self, forKey: .layerNormEpsilon) ?? 1e-6
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 11008
        noBias = try c.decodeIfPresent(Bool.self, forKey: .noBias) ?? true
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 151_936
        // Filled in by upstream's __post_init__ from the head count. This model has no
        // grouped-query path -- its forward reshapes keys and values with the full head
        // count -- so the value only sizes the caches.
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads
    }
}

extension QwenModel: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.transformer.h
    }
}
