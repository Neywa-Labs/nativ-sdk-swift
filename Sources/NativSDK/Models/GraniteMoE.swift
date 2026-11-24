import Foundation
import MLX
import MLXNN

class GraniteMoEAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int

    /// Not `pow(headDim, -0.5)`: Granite gives the scale outright.
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer

    init(_ args: GraniteMoEConfiguration) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDimensions = args.hiddenSize / args.attentionHeads
        self.scale = args.attentionMultiplier

        _wq.wrappedValue = Linear(
            args.hiddenSize, heads * headDimensions, bias: args.attentionBias)
        _wk.wrappedValue = Linear(
            args.hiddenSize, kvHeads * headDimensions, bias: args.attentionBias)
        _wv.wrappedValue = Linear(
            args.hiddenSize, kvHeads * headDimensions, bias: args.attentionBias)
        _wo.wrappedValue = Linear(
            heads * headDimensions, args.hiddenSize, bias: args.attentionBias)

        self.rope = initializeRope(
            dims: headDimensions,
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)
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

class GraniteMoETopKGating: Module {
    let topK: Int
    @ModuleInfo var layer: Linear

    init(inputSize: Int, experts: Int, topK: Int) {
        self.topK = topK
        _layer.wrappedValue = Linear(inputSize, experts, bias: false)
    }

    /// Selects the largest `topK` logits and softmaxes those, so the kept weights sum to
    /// one -- the same order phixtral uses and the opposite of qwen2_moe.
    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, gates: MLXArray) {
        let logits = layer(x)
        // Ascending partition, then the tail: the largest, without negating first.
        let indices = MLX.argPartition(logits, kth: logits.dim(-1) - topK, axis: -1)[
            .ellipsis, (-topK)...]
        let gates = softmax(takeAlong(logits, indices, axis: -1).asType(.float32), axis: -1)
        return (indices, gates)
    }
}

class GraniteMoEBlockSparse: Module {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    let router: GraniteMoETopKGating

    init(_ args: GraniteMoEConfiguration) {
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.intermediateSize,
            numExperts: args.localExperts)
        self.router = GraniteMoETopKGating(
            inputSize: args.hiddenSize, experts: args.localExperts,
            topK: args.expertsPerToken)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, gates) = router(x)
        let y = switchMLP(x, indices)
        return (y * gates[.ellipsis, .newAxis]).sum(axis: -2).asType(y.dtype)
    }
}

class GraniteMoEBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: GraniteMoEAttention
    @ModuleInfo(key: "block_sparse_moe") var moe: GraniteMoEBlockSparse

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let residualMultiplier: Float

    init(_ args: GraniteMoEConfiguration) {
        _attention.wrappedValue = GraniteMoEAttention(args)
        _moe.wrappedValue = GraniteMoEBlockSparse(args)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self.residualMultiplier = args.residualMultiplier
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h =
            x + attention(inputLayerNorm(x), mask: mask, cache: cache) * residualMultiplier
        return h + moe(postAttentionLayerNorm(h)) * residualMultiplier
    }
}

public class GraniteMoEModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [GraniteMoEBlock]
    let norm: RMSNorm
    let embeddingMultiplier: Float

    public init(_ args: GraniteMoEConfiguration) {
        precondition(args.vocabularySize > 0)
        self.embeddingMultiplier = args.embeddingMultiplier
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { _ in GraniteMoEBlock(args) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs) * embeddingMultiplier
        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class GraniteMoEModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: GraniteMoEModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: GraniteMoEConfiguration

    public init(_ args: GraniteMoEConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = GraniteMoEModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let hidden = model(inputs, cache: cache)
        let out = lmHead?(hidden) ?? model.embedTokens.asLinear(hidden)
        // Divides, not multiplies.
        return out / configuration.logitsScaling
    }

    /// Splits the fused expert projection and renames the pair.
    ///
    /// Granite ships `input_linear` holding gate and up stacked on the output axis of an
    /// already-per-expert tensor, so this is a split rather than the stack the other
    /// mixtures here need. Gate is the first half; taking them the other way round binds
    /// the same shapes and swaps the two branches of every expert.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights

        if sanitized["model.layers.0.block_sparse_moe.input_linear.weight"] != nil {
            for layer in 0 ..< configuration.hiddenLayers {
                let prefix = "model.layers.\(layer).block_sparse_moe"
                if let fused = sanitized.removeValue(forKey: "\(prefix).input_linear.weight") {
                    let halves = split(fused, parts: 2, axis: 1)
                    sanitized["\(prefix).switch_mlp.gate_proj.weight"] = halves[0]
                    sanitized["\(prefix).switch_mlp.up_proj.weight"] = halves[1]
                }
                if let down = sanitized.removeValue(forKey: "\(prefix).output_linear.weight") {
                    sanitized["\(prefix).switch_mlp.down_proj.weight"] = down
                }
            }
        }

        if configuration.tieWordEmbeddings {
            sanitized = sanitized.filter { $0.key != "lm_head.weight" }
        }
        return sanitized
    }
}

public struct GraniteMoEConfiguration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var logitsScaling: Float
    var attentionMultiplier: Float
    var embeddingMultiplier: Float
    var residualMultiplier: Float
    var maxPositionEmbeddings: Int
    var kvHeads: Int
    var attentionBias: Bool
    var ropeTheta: Float
    var localExperts: Int
    var expertsPerToken: Int
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case logitsScaling = "logits_scaling"
        case attentionMultiplier = "attention_multiplier"
        case embeddingMultiplier = "embedding_multiplier"
        case residualMultiplier = "residual_multiplier"
        case maxPositionEmbeddings = "max_position_embeddings"
        case kvHeads = "num_key_value_heads"
        case attentionBias = "attention_bias"
        case ropeTheta = "rope_theta"
        case localExperts = "num_local_experts"
        case expertsPerToken = "num_experts_per_tok"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "granitemoe"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        // Required, all four of them: a default here would be a silently wrong model.
        logitsScaling = try c.decode(Float.self, forKey: .logitsScaling)
        attentionMultiplier = try c.decode(Float.self, forKey: .attentionMultiplier)
        embeddingMultiplier = try c.decode(Float.self, forKey: .embeddingMultiplier)
        residualMultiplier = try c.decode(Float.self, forKey: .residualMultiplier)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        attentionBias = try c.decode(Bool.self, forKey: .attentionBias)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        localExperts = try c.decode(Int.self, forKey: .localExperts)
        expertsPerToken = try c.decode(Int.self, forKey: .expertsPerToken)
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }
}

extension GraniteMoEModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
