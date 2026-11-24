import Foundation
import MLX
import MLXNN

class YoutuLLMAttention: Module {
    let heads: Int
    let qkNopeHeadDimensions: Int
    let qkRopeHeadDimensions: Int
    let valueHeadDimensions: Int
    let kvLoRARank: Int
    let queryHeadDimensions: Int
    let scale: Float

    /// Whether the query path goes through a low-rank bottleneck.
    let usesQueryLoRA: Bool

    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qA: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qANorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qB: Linear?

    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvA: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvANorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvB: Linear

    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer

    init(_ config: YoutuLLMConfiguration) {
        self.heads = config.attentionHeads
        self.qkNopeHeadDimensions = config.qkNopeHeadDimensions
        self.qkRopeHeadDimensions = config.qkRopeHeadDimensions
        self.valueHeadDimensions = config.valueHeadDimensions
        self.kvLoRARank = config.kvLoRARank
        self.queryHeadDimensions =
            config.qkNopeHeadDimensions + config.qkRopeHeadDimensions
        self.scale = pow(Float(queryHeadDimensions), -0.5)
        self.usesQueryLoRA = config.qLoRARank != nil

        if let rank = config.qLoRARank {
            _qA.wrappedValue = Linear(
                config.hiddenSize, rank, bias: config.attentionBias)
            _qANorm.wrappedValue = RMSNorm(dimensions: rank, eps: config.rmsNormEps)
            _qB.wrappedValue = Linear(rank, heads * queryHeadDimensions, bias: false)
        } else {
            _qProj.wrappedValue = Linear(
                config.hiddenSize, heads * queryHeadDimensions, bias: false)
        }

        _kvA.wrappedValue = Linear(
            config.hiddenSize, config.kvLoRARank + config.qkRopeHeadDimensions,
            bias: config.attentionBias)
        _kvANorm.wrappedValue = RMSNorm(
            dimensions: config.kvLoRARank, eps: config.rmsNormEps)
        _kvB.wrappedValue = Linear(
            config.kvLoRARank,
            heads * (config.qkNopeHeadDimensions + config.valueHeadDimensions),
            bias: false)

        _wo.wrappedValue = Linear(
            heads * config.valueHeadDimensions, config.hiddenSize,
            bias: config.attentionBias)

        self.rope = initializeRope(
            dims: config.qkRopeHeadDimensions,
            base: config.ropeTheta,
            traditional: config.ropeTraditional,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let projected: MLXArray =
            if let qA, let qANorm, let qB {
                qB(qANorm(qA(x)))
            } else {
                qProj!(x)
            }
        let q = projected.reshaped(B, L, heads, queryHeadDimensions).transposed(0, 2, 1, 3)
        let qNope = q[0..., 0..., 0..., ..<qkNopeHeadDimensions]
        var qRope = q[0..., 0..., 0..., qkNopeHeadDimensions...]

        let latent = kvA(x)
        let compressed = latent[0..., 0..., ..<kvLoRARank]
        var kRope = latent[0..., 0..., kvLoRARank...]
            .reshaped(B, L, 1, qkRopeHeadDimensions).transposed(0, 2, 1, 3)

        let kv = kvB(kvANorm(compressed)).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        let kNope = kv[0..., 0..., 0..., ..<qkNopeHeadDimensions]
        var values = kv[0..., 0..., 0..., qkNopeHeadDimensions...]

        let offset = cache?.ropeOffset
        qRope = applyRotaryPosition(rope, to: qRope, offset: offset)
        kRope = applyRotaryPosition(rope, to: kRope, offset: offset)
        // The single rotary head is repeated across every head before the concatenation,
        // so the cache stores full-width keys rather than the shared slice.
        kRope = MLX.repeated(kRope, count: heads, axis: 1)

        var keys = concatenated([kNope, kRope], axis: -1)
        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        let queries = concatenated([qNope, qRope], axis: -1)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

class YoutuLLMBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: YoutuLLMAttention
    let mlp: SwiGLUMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ config: YoutuLLMConfiguration) {
        _attention.wrappedValue = YoutuLLMAttention(config)
        self.mlp = SwiGLUMLP(
            inputDimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize,
            bias: config.mlpBias)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class YoutuLLMModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [YoutuLLMBlock]
    let norm: RMSNorm

    public init(_ config: YoutuLLMConfiguration) {
        precondition(config.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map { _ in YoutuLLMBlock(config) }
        self.norm = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
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

public class YoutuLLMModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: YoutuLLMModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: YoutuLLMConfiguration

    public init(_ config: YoutuLLMConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.attentionHeads }
        self.model = YoutuLLMModelInner(config)
        if !config.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(
                config.hiddenSize, config.vocabularySize, bias: false)
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

public struct YoutuLLMConfiguration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var kvLoRARank: Int
    /// Absent means no query bottleneck: a single `q_proj` instead of the a/b pair.
    var qLoRARank: Int?
    var qkRopeHeadDimensions: Int
    var valueHeadDimensions: Int
    var qkNopeHeadDimensions: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeTraditional: Bool
    var ropeScaling: [String: StringOrNumber]?
    var attentionBias: Bool
    var mlpBias: Bool
    var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case kvLoRARank = "kv_lora_rank"
        case qLoRARank = "q_lora_rank"
        case qkRopeHeadDimensions = "qk_rope_head_dim"
        case valueHeadDimensions = "v_head_dim"
        case qkNopeHeadDimensions = "qk_nope_head_dim"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
        case mlpBias = "mlp_bias"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "youtu_llm"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 128_256
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 2048
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 6144
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 16
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 16
        kvLoRARank = try c.decodeIfPresent(Int.self, forKey: .kvLoRARank) ?? 512
        // An explicit null selects direct queries; an absent field uses the default rank.
        qLoRARank =
            c.contains(.qLoRARank)
            ? try c.decodeIfPresent(Int.self, forKey: .qLoRARank)
            : 1536
        qkRopeHeadDimensions =
            try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDimensions) ?? 64
        valueHeadDimensions =
            try c.decodeIfPresent(Int.self, forKey: .valueHeadDimensions) ?? 128
        qkNopeHeadDimensions =
            try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDimensions) ?? 128
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_600_000
        // Traditional by default here, unlike most of the zoo.
        ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? true
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        mlpBias = try c.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? true
    }
}

extension YoutuLLMModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
