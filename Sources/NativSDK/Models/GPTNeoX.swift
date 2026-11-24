import Foundation
import MLX
import MLXNN

class GPTNeoXAttention: Module {
    let heads: Int
    let headDimensions: Int
    let scale: Float

    @ModuleInfo(key: "query_key_value") var qkv: Linear
    @ModuleInfo var dense: Linear

    let rope: RoPE

    /// How many of each head's dimensions rotate, from `rotary_pct`.
    let rotaryDimensions: Int

    init(_ config: GPTNeoXConfiguration) {
        self.heads = config.attentionHeads
        self.headDimensions = config.hiddenSize / config.attentionHeads
        self.scale = pow(Float(headDimensions), -0.5)

        _qkv.wrappedValue = Linear(config.hiddenSize, 3 * config.hiddenSize, bias: true)
        _dense.wrappedValue = Linear(config.hiddenSize, config.hiddenSize, bias: true)

        self.rotaryDimensions = Int(Float(headDimensions) * config.rotaryPct)
        self.rope = RoPE(
            dimensions: rotaryDimensions, traditional: false,
            base: Float(config.rotaryEmbeddingBase))
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        // Reshape to per-head thirds *before* splitting, which is what makes the split
        // separate the three roles rather than cut across the heads.
        let fused = qkv(x).reshaped(B, L, heads, 3 * headDimensions)
        let parts = split(fused, parts: 3, axis: -1).map { $0.transposed(0, 2, 1, 3) }

        var queries = parts[0]
        var keys = parts[1]
        let values = parts[2]

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return dense(output)
    }
}

class GPTNeoXMLP: Module, UnaryLayer {
    @ModuleInfo(key: "dense_h_to_4h") var up: Linear
    @ModuleInfo(key: "dense_4h_to_h") var down: Linear

    init(_ config: GPTNeoXConfiguration) {
        _up.wrappedValue = Linear(config.hiddenSize, 4 * config.hiddenSize)
        _down.wrappedValue = Linear(4 * config.hiddenSize, config.hiddenSize)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(geluApproximate(up(x)))
    }
}

class GPTNeoXBlock: Module {
    @ModuleInfo(key: "attention") var attention: GPTNeoXAttention
    let mlp: GPTNeoXMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: LayerNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: LayerNorm

    let useParallelResidual: Bool

    init(_ config: GPTNeoXConfiguration) {
        _attention.wrappedValue = GPTNeoXAttention(config)
        self.mlp = GPTNeoXMLP(config)
        _inputLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        _postAttentionLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
        self.useParallelResidual = config.useParallelResidual
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        // Both arrangements keep both norms, unlike stablelm: parallel feeds the second one
        // the block's *input*, sequential feeds it the attention's output. Same weights
        // either way, so the wrong one answers plausibly.
        let attended = attention(inputLayerNorm(x), mask: mask, cache: cache)
        if useParallelResidual {
            return attended + mlp(postAttentionLayerNorm(x)) + x
        }
        let h = x + attended
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class GPTNeoXModelInner: Module {
    @ModuleInfo(key: "embed_in") var embedIn: Embedding
    @ModuleInfo(key: "embed_out") var embedOut: Linear

    let h: [GPTNeoXBlock]

    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    public init(_ config: GPTNeoXConfiguration) {
        precondition(config.vocabularySize > 0)
        _embedIn.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        _embedOut.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        self.h = (0 ..< config.hiddenLayers).map { _ in GPTNeoXBlock(config) }
        _finalLayerNorm.wrappedValue = LayerNorm(
            dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var hidden = embedIn(inputs)
        let mask = createAttentionMask(h: hidden, cache: cache?.first)

        for (i, layer) in h.enumerated() {
            hidden = layer(hidden, mask: mask, cache: cache?[i])
        }

        return embedOut(finalLayerNorm(hidden))
    }
}

public class GPTNeoXLanguageModel: Module {
    public let model: GPTNeoXModelInner

    public init(_ config: GPTNeoXConfiguration) {
        self.model = GPTNeoXModelInner(config)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model(inputs, cache: cache)
    }
}

public class GPTNeoXModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: GPTNeoXLanguageModel

    public init(_ config: GPTNeoXConfiguration) {
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        _languageModel.wrappedValue = GPTNeoXLanguageModel(config)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    /// Flattens the published `gpt_neox.` nesting and drops the buffers that are not
    /// parameters.
    ///
    /// `attention.bias` and `attention.masked_bias` are the causal mask the original
    /// implementation kept as buffers, and `rotary_emb.inv_freq` is recomputed from
    /// `rotary_emb_base`; binding any of them would fail on a tree that has no such module.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let dropped = [
            ".attention.bias", ".attention.masked_bias", ".attention.rotary_emb.inv_freq",
        ]

        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            if dropped.contains(where: key.hasSuffix) {
                continue
            }
            var name = key
            if name.hasPrefix("language_model.") {
                name = String(name.dropFirst("language_model.".count))
            }
            if !name.hasPrefix("model.") {
                name = "model." + name
            }
            name = name.replacingOccurrences(of: ".gpt_neox.layers.", with: ".h.")
            name = name.replacingOccurrences(of: ".gpt_neox.", with: ".")
            sanitized["language_model." + name] = value
        }
        return sanitized
    }
}

public struct GPTNeoXConfiguration: Codable, Sendable {
    var modelType: String
    var maxPositionEmbeddings: Int
    var hiddenSize: Int
    var attentionHeads: Int
    var hiddenLayers: Int
    var layerNormEps: Float
    var vocabularySize: Int
    var rotaryEmbeddingBase: Int
    var rotaryPct: Float
    var useParallelResidual: Bool
    var kvHeads: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case maxPositionEmbeddings = "max_position_embeddings"
        case hiddenSize = "hidden_size"
        case attentionHeads = "num_attention_heads"
        case hiddenLayers = "num_hidden_layers"
        case layerNormEps = "layer_norm_eps"
        case vocabularySize = "vocab_size"
        case rotaryEmbeddingBase = "rotary_emb_base"
        case rotaryPct = "rotary_pct"
        case useParallelResidual = "use_parallel_residual"
        case kvHeads = "num_key_value_heads"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gpt_neox"
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        layerNormEps = try c.decode(Float.self, forKey: .layerNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        rotaryEmbeddingBase = try c.decode(Int.self, forKey: .rotaryEmbeddingBase)
        rotaryPct = try c.decode(Float.self, forKey: .rotaryPct)
        useParallelResidual =
            try c.decodeIfPresent(Bool.self, forKey: .useParallelResidual) ?? true
        // Filled in by upstream's __post_init__ from the head count, so a config that omits
        // it is multi-head attention rather than one key-value head.
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? attentionHeads

        guard hiddenSize % attentionHeads == 0 else {
            throw DecodingError.dataCorrupted(
                .init(
                    codingPath: c.codingPath,
                    debugDescription:
                        "hidden_size \(hiddenSize) must be divisible by num_attention_heads "
                        + "\(attentionHeads)"))
        }
    }
}

extension GPTNeoXModel: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.h
    }
}
