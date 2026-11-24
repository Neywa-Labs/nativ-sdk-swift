import Foundation
import MLX
import MLXNN

class GPT2Attention: Module {
    let heads: Int
    let headDimensions: Int
    let scale: Float

    @ModuleInfo(key: "c_attn") var cAttn: Linear
    @ModuleInfo(key: "c_proj") var cProj: Linear

    init(_ args: GPT2Configuration) {
        precondition(
            args.embeddingDimensions % args.heads == 0,
            "n_embd must be divisible by n_head")
        self.heads = args.heads
        self.headDimensions = args.embeddingDimensions / args.heads
        self.scale = pow(Float(headDimensions), -0.5)

        _cAttn.wrappedValue = Linear(
            args.embeddingDimensions, 3 * args.embeddingDimensions, bias: true)
        _cProj.wrappedValue = Linear(
            args.embeddingDimensions, args.embeddingDimensions, bias: true)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let qkv = split(cAttn(x), parts: 3, axis: -1)
        let queries = qkv[0].reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        let keys = qkv[1].reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        let values = qkv[2].reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return cProj(output)
    }
}

class GPT2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "c_fc") var cFC: Linear
    @ModuleInfo(key: "c_proj") var cProj: Linear

    init(_ args: GPT2Configuration) {
        _cFC.wrappedValue = Linear(args.embeddingDimensions, 4 * args.embeddingDimensions)
        _cProj.wrappedValue = Linear(4 * args.embeddingDimensions, args.embeddingDimensions)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        cProj(geluApproximate(cFC(x)))
    }
}

class GPT2Block: Module {
    @ModuleInfo(key: "attn") var attention: GPT2Attention
    let mlp: GPT2MLP

    @ModuleInfo(key: "ln_1") var ln1: LayerNorm
    @ModuleInfo(key: "ln_2") var ln2: LayerNorm

    init(_ args: GPT2Configuration) {
        _attention.wrappedValue = GPT2Attention(args)
        self.mlp = GPT2MLP(args)
        _ln1.wrappedValue = LayerNorm(
            dimensions: args.embeddingDimensions, eps: args.layerNormEpsilon)
        _ln2.wrappedValue = LayerNorm(
            dimensions: args.embeddingDimensions, eps: args.layerNormEpsilon)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(ln1(x), mask: mask, cache: cache)
        return h + mlp(ln2(h))
    }
}

public class GPT2ModelInner: Module {
    @ModuleInfo(key: "wte") var tokenEmbedding: Embedding
    @ModuleInfo(key: "wpe") var positionEmbedding: Embedding

    let h: [GPT2Block]

    @ModuleInfo(key: "ln_f") var lnF: LayerNorm

    public init(_ args: GPT2Configuration) {
        precondition(args.vocabularySize > 0)

        _tokenEmbedding.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.embeddingDimensions)
        _positionEmbedding.wrappedValue = Embedding(
            embeddingCount: args.positions, dimensions: args.embeddingDimensions)
        self.h = (0 ..< args.layers).map { _ in GPT2Block(args) }
        _lnF.wrappedValue = LayerNorm(
            dimensions: args.embeddingDimensions, eps: args.layerNormEpsilon)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        let L = inputs.dim(-1)

        // Positions are looked up, not rotated, so a warm cache has to contribute its
        // offset here: decoding token n with an empty position index would embed every
        // generated token as if it were the first.
        let offset = cache?.first?.offset ?? 0
        var hiddenStates = tokenEmbedding(inputs)
        hiddenStates = hiddenStates + positionEmbedding(MLXArray(offset ..< offset + L))

        let mask = createAttentionMask(h: hiddenStates, cache: cache?.first)

        for (i, layer) in h.enumerated() {
            hiddenStates = layer(hiddenStates, mask: mask, cache: cache?[i])
        }

        return lnF(hiddenStates)
    }
}

public class GPT2Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "language_model") var languageModel: GPT2LanguageModel

    public init(_ args: GPT2Configuration) {
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.layers).map { _ in args.kvHeads }
        _languageModel.wrappedValue = GPT2LanguageModel(args)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        languageModel(inputs, cache: cache)
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        languageModel.sanitize(weights: weights)
    }
}

public class GPT2LanguageModel: Module {
    public let model: GPT2ModelInner
    let configuration: GPT2Configuration

    public init(_ args: GPT2Configuration) {
        self.configuration = args
        self.model = GPT2ModelInner(args)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model.tokenEmbedding.asLinear(model(inputs, cache: cache))
    }

    /// Undo HuggingFace's `Conv1D` layout and nest the result.
    ///
    /// GPT-2's projections were written as `Conv1D`, which holds `[in, out]` where a
    /// `Linear` holds `[out, in]`. Loading them untransposed does not fail -- the attention
    /// output projection is square in `n_embd` and the MLP's two are transposes of each
    /// other's shape -- so the model runs and answers nonsense. `attn.bias` is the causal
    /// mask the original implementation stored as a buffer, and is dropped rather than
    /// bound.
    ///
    /// The non-square `c_attn` projection distinguishes Conv1D from converted Linear
    /// layout. Square projections follow the same decision.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let transposed: Set<String> = [
            "attn.c_attn.weight", "attn.c_proj.weight",
            "mlp.c_fc.weight", "mlp.c_proj.weight",
        ]
        let fused = 3 * configuration.embeddingDimensions
        let isConv1DLayout =
            weights.first { key, value in
                key.hasSuffix("attn.c_attn.weight") && value.ndim == 2
            }
            .map { _, value in value.dim(0) != fused } ?? true

        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            var name = key
            if name.hasPrefix("language_model.") {
                name = String(name.dropFirst("language_model.".count))
            }
            if name.hasPrefix("model.") {
                name = String(name.dropFirst("model.".count))
            }
            if name.hasSuffix(".attn.bias") || name.hasSuffix(".attn.masked_bias") {
                continue
            }

            var array = value
            if isConv1DLayout,
                let suffix = transposed.first(where: { name.hasSuffix($0) }), !suffix.isEmpty,
                array.ndim == 2
            {
                array = array.transposed(1, 0)
            }
            sanitized["language_model.model." + name] = array
        }
        return sanitized
    }
}

public struct GPT2Configuration: Codable, Sendable {
    var modelType: String
    var contextSize: Int
    var embeddingDimensions: Int
    var heads: Int
    var layers: Int
    var positions: Int
    var layerNormEpsilon: Float
    var vocabularySize: Int
    var kvHeads: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case contextSize = "n_ctx"
        case embeddingDimensions = "n_embd"
        case heads = "n_head"
        case layers = "n_layer"
        case positions = "n_positions"
        case layerNormEpsilon = "layer_norm_epsilon"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "gpt2"
        embeddingDimensions = try c.decode(Int.self, forKey: .embeddingDimensions)
        heads = try c.decode(Int.self, forKey: .heads)
        layers = try c.decode(Int.self, forKey: .layers)
        positions = try c.decode(Int.self, forKey: .positions)
        contextSize = try c.decodeIfPresent(Int.self, forKey: .contextSize) ?? positions
        layerNormEpsilon = try c.decode(Float.self, forKey: .layerNormEpsilon)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        // Upstream fills this in __post_init__ from n_head, so a config that omits it is
        // multi-head attention, not one key-value head.
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? heads
    }
}

extension GPT2Model: LoRAModel {
    public var loraLayers: [Module] {
        languageModel.model.h
    }
}
