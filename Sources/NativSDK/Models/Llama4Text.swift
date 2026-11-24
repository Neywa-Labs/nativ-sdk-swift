import Foundation
import MLX
import MLXNN

class Llama4TextAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let scale: Float
    let useQKNorm: Bool
    let rmsNormEps: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPE?

    init(_ args: Llama4TextConfiguration, useRoPE: Bool) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDimensions = args.headDim
        self.scale = pow(Float(args.headDim), -0.5)
        self.useQKNorm = args.useQKNorm
        self.rmsNormEps = args.rmsNormEps

        _wq.wrappedValue = Linear(args.hiddenSize, heads * headDimensions, bias: false)
        _wk.wrappedValue = Linear(args.hiddenSize, kvHeads * headDimensions, bias: false)
        _wv.wrappedValue = Linear(args.hiddenSize, kvHeads * headDimensions, bias: false)
        _wo.wrappedValue = Linear(heads * headDimensions, args.hiddenSize, bias: false)

        self.rope =
            useRoPE
            ? RoPE(dimensions: headDimensions, traditional: true, base: args.ropeTheta) : nil
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x).reshaped(B, L, heads, -1)
        var keys = wk(x).reshaped(B, L, kvHeads, -1)

        // Normalized on the head-last layout, before the transpose, and with no learned
        // weight -- upstream calls mx.fast.rms_norm(q, None, eps) here.
        if useQKNorm {
            queries = MLXFast.rmsNorm(queries, weight: MLXArray.mlxNone, eps: rmsNormEps)
            keys = MLXFast.rmsNorm(keys, weight: MLXArray.mlxNone, eps: rmsNormEps)
        }

        queries = queries.transposed(0, 2, 1, 3)
        keys = keys.transposed(0, 2, 1, 3)
        let values = wv(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        if let rope {
            let offset = cache?.ropeOffset
            queries = applyRotaryPosition(rope, to: queries, offset: offset)
            keys = applyRotaryPosition(rope, to: keys, offset: offset)
        }

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

class Llama4TextMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    @ModuleInfo(key: "up_proj") var up: Linear

    init(dimensions: Int, hiddenDimensions: Int) {
        _gate.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
        _down.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
        _up.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        down(silu(gate(x)) * up(x))
    }
}

class Llama4TextBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: Llama4TextAttention
    @ModuleInfo(key: "feed_forward") var feedForward: Llama4TextMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: Llama4TextConfiguration, useRoPE: Bool) {
        _attention.wrappedValue = Llama4TextAttention(args, useRoPE: useRoPE)
        _feedForward.wrappedValue = Llama4TextMLP(
            dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSizeMLP)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + feedForward(postAttentionLayerNorm(h))
    }
}

public class Llama4TextModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [Llama4TextBlock]
    let norm: RMSNorm

    public init(_ args: Llama4TextConfiguration) {
        precondition(args.vocabularySize > 0)

        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)

        // `no_rope_layers[i]` is truthy for the layers that *do* rope, despite the name.
        // Llama 4 interleaves a few NoPE layers, and reading the flag the other way round
        // costs nothing at load time and every position after it.
        self.layers = (0 ..< args.hiddenLayers).map { index in
            Llama4TextBlock(args, useRoPE: args.usesRoPE(atLayer: index))
        }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
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

public class Llama4TextLanguageModel: Module {
    public let model: Llama4TextModelInner

    @ModuleInfo(key: "output") var output: Linear?

    let configuration: Llama4TextConfiguration

    public init(_ args: Llama4TextConfiguration) {
        self.configuration = args
        self.model = Llama4TextModelInner(args)
        if !args.tieWordEmbeddings {
            _output.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let output {
            return output(out)
        }
        return model.embedTokens.asLinear(out)
    }
}

public class Llama4TextModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: Llama4TextLanguageModel

    let configuration: Llama4TextConfiguration

    public init(_ args: Llama4TextConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        _languageModel.wrappedValue = Llama4TextLanguageModel(args)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            var name = key
            // The checkpoint head maps to the output projection inside language_model.
            if name.hasPrefix("lm_head.") {
                name = "output." + name.dropFirst("lm_head.".count)
            }
            if !name.hasPrefix("language_model.") {
                name = "language_model." + name
            }
            sanitized[name] = value
        }
        if configuration.tieWordEmbeddings {
            sanitized = sanitized.filter { !$0.key.hasPrefix("language_model.output.") }
        }
        return sanitized
    }
}

public struct Llama4TextConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var attentionHeads: Int
    var hiddenLayers: Int
    var vocabularySize: Int
    var intermediateSize: Int
    var intermediateSizeMLP: Int
    var kvHeads: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var headDim: Int
    var tieWordEmbeddings: Bool
    var noRoPELayers: [Int]
    var useQKNorm: Bool

    /// Whether layer `index` applies rotary positions.
    ///
    /// A checkpoint whose `no_rope_layers` is shorter than the stack is taken to rope the
    /// layers it does not mention, which is how a config that omits the tail behaves
    /// upstream by virtue of the list it ships being full length.
    func usesRoPE(atLayer index: Int) -> Bool {
        index < noRoPELayers.count ? noRoPELayers[index] != 0 : true
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case attentionHeads = "num_attention_heads"
        case hiddenLayers = "num_hidden_layers"
        case vocabularySize = "vocab_size"
        case intermediateSize = "intermediate_size"
        case intermediateSizeMLP = "intermediate_size_mlp"
        case kvHeads = "num_key_value_heads"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case headDim = "head_dim"
        case tieWordEmbeddings = "tie_word_embeddings"
        case noRoPELayers = "no_rope_layers"
        case useQKNorm = "use_qk_norm"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "llama4_text"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        intermediateSizeMLP = try c.decode(Int.self, forKey: .intermediateSizeMLP)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        headDim = try c.decode(Int.self, forKey: .headDim)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        noRoPELayers = try c.decodeIfPresent([Int].self, forKey: .noRoPELayers) ?? []
        useQKNorm = try c.decodeIfPresent(Bool.self, forKey: .useQKNorm) ?? false
    }
}

extension Llama4TextModel: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.layers
    }
}
