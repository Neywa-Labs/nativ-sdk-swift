import Foundation
import MLX
import MLXNN

class MiniCPM3Attention: Module {
    let heads: Int
    let qkNopeHeadDimensions: Int
    let qkRopeHeadDimensions: Int
    let valueHeadDimensions: Int
    let kvLoRARank: Int
    let softmaxScale: Float

    @ModuleInfo(key: "q_a_proj") var qA: Linear
    @ModuleInfo(key: "q_a_layernorm") var qANorm: RMSNorm
    @ModuleInfo(key: "q_b_proj") var qB: Linear

    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvA: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvANorm: RMSNorm
    @ModuleInfo(key: "kv_b_proj") var kvB: Linear

    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: SuScaledRoPE

    init(_ args: MiniCPM3Configuration) {
        self.heads = args.attentionHeads
        self.qkNopeHeadDimensions = args.qkNopeHeadDimensions
        self.qkRopeHeadDimensions = args.qkRopeHeadDimensions
        self.valueHeadDimensions = args.hiddenSize / args.attentionHeads
        self.kvLoRARank = args.kvLoRARank

        let queryHeadDimensions = args.qkNopeHeadDimensions + args.qkRopeHeadDimensions
        self.softmaxScale = pow(Float(queryHeadDimensions), -0.5)

        _qA.wrappedValue = Linear(args.hiddenSize, args.qLoRARank, bias: args.attentionBias)
        _qANorm.wrappedValue = RMSNorm(dimensions: args.qLoRARank)
        _qB.wrappedValue = Linear(args.qLoRARank, heads * queryHeadDimensions, bias: false)

        // One projection produces the compressed key-value latent *and* the shared rotary
        // key, which is why its width is the rank plus the rotary slice.
        _kvA.wrappedValue = Linear(
            args.hiddenSize, args.kvLoRARank + args.qkRopeHeadDimensions,
            bias: args.attentionBias)
        _kvANorm.wrappedValue = RMSNorm(dimensions: args.kvLoRARank)
        _kvB.wrappedValue = Linear(
            args.kvLoRARank, heads * (qkNopeHeadDimensions + valueHeadDimensions),
            bias: false)

        _wo.wrappedValue = Linear(
            heads * valueHeadDimensions, args.hiddenSize, bias: args.attentionBias)

        // `StringOrNumber` decodes an integral JSON number as `.int`, so matching only
        // `.float` here silently kept the defaults: `original_max_position_embeddings` is
        // written `32768`, which left this at 4096 and made the window factor 8 rather
        // than 1. SuScaledRoPE turns that into a 1.118x scale on the rotary half of every
        // query and key -- a 25% inflation of its share of each attention score, over 62
        // layers. The accessors coerce across the numeric cases; the raw patterns do not.
        let scaling = args.ropeScaling ?? [:]
        self.rope = SuScaledRoPE(
            dimensions: args.qkRopeHeadDimensions,
            base: args.ropeTheta,
            maxPositionEmbeddings: args.maxPositionEmbeddings,
            originalMaxPositionEmbeddings: scaling["original_max_position_embeddings"]?
                .asFloat().map { Int($0) } ?? 4096,
            shortFactor: scaling["short_factor"]?.asFloats() ?? [1.0],
            longFactor: scaling["long_factor"]?.asFloats() ?? [1.0])
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let q = qB(qANorm(qA(x))).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        let qNope = q[0..., 0..., 0..., ..<qkNopeHeadDimensions]
        var qPE = q[0..., 0..., 0..., qkNopeHeadDimensions...]

        let latent = kvA(x)
        let compressed = latent[0..., 0..., ..<kvLoRARank]
        var kPE = latent[0..., 0..., kvLoRARank...]
            .reshaped(B, L, 1, qkRopeHeadDimensions).transposed(0, 2, 1, 3)

        let kv = kvB(kvANorm(compressed)).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        let kNope = kv[0..., 0..., 0..., ..<qkNopeHeadDimensions]
        let values = kv[0..., 0..., 0..., qkNopeHeadDimensions...]

        let offset = cache?.ropeOffset
        qPE = applyRotaryPosition(rope, to: qPE, offset: offset)
        kPE = applyRotaryPosition(rope, to: kPE, offset: offset)

        // The rotary key is shared across heads -- one head's worth is projected and then
        // broadcast, which is the "MQA" in `kv_a_proj_with_mqa`.
        let kPEBroadcast = broadcast(kPE, to: [B, heads, L, qkRopeHeadDimensions])

        let queries = concatenated([qNope, qPE], axis: -1)
        let keys = concatenated([kNope, kPEBroadcast], axis: -1)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: softmaxScale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return wo(output)
    }
}

class MiniCPM3Block: Module {
    @ModuleInfo(key: "self_attn") var attention: MiniCPM3Attention
    let mlp: SwiGLUMLP

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    /// `scale_depth / sqrt(layers)`, applied to both residual branches.
    let residualScale: Float

    init(_ args: MiniCPM3Configuration) {
        _attention.wrappedValue = MiniCPM3Attention(args)
        self.mlp = SwiGLUMLP(
            inputDimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        self.residualScale = args.scaleDepth / sqrt(Float(args.hiddenLayers))
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache) * residualScale
        return h + mlp(postAttentionLayerNorm(h)) * residualScale
    }
}

public class MiniCPM3ModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [MiniCPM3Block]
    let norm: RMSNorm

    public init(_ args: MiniCPM3Configuration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { _ in MiniCPM3Block(args) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        // Token lookups are unscaled; scale_emb applies only to external embeddings.
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class MiniCPM3Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: MiniCPM3ModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: MiniCPM3Configuration

    public init(_ args: MiniCPM3Configuration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.attentionHeads }
        self.model = MiniCPM3ModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        guard let lmHead else {
            // Width scaling applies only to the separate output projection.
            return model.embedTokens.asLinear(out)
        }
        return lmHead(
            out / (Float(configuration.hiddenSize) / Float(configuration.baseModelDimensions)))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights
        if configuration.tieWordEmbeddings {
            sanitized = sanitized.filter { !$0.key.hasPrefix("lm_head.") }
        }
        return sanitized
    }
}

public struct MiniCPM3Configuration: Codable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var baseModelDimensions: Int
    var hiddenLayers: Int
    var intermediateSize: Int
    var attentionHeads: Int
    var rmsNormEps: Float
    var vocabularySize: Int
    var kvHeads: Int
    var qLoRARank: Int
    var qkNopeHeadDimensions: Int
    var qkRopeHeadDimensions: Int
    var kvLoRARank: Int
    var scaleDepth: Float
    var scaleEmbedding: Float
    var maxPositionEmbeddings: Int
    var attentionBias: Bool
    var ropeTheta: Float
    var ropeTraditional: Bool
    var ropeScaling: [String: StringOrNumber]?
    var tieWordEmbeddings: Bool

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case baseModelDimensions = "dim_model_base"
        case hiddenLayers = "num_hidden_layers"
        case intermediateSize = "intermediate_size"
        case attentionHeads = "num_attention_heads"
        case rmsNormEps = "rms_norm_eps"
        case vocabularySize = "vocab_size"
        case kvHeads = "num_key_value_heads"
        case qLoRARank = "q_lora_rank"
        case qkNopeHeadDimensions = "qk_nope_head_dim"
        case qkRopeHeadDimensions = "qk_rope_head_dim"
        case kvLoRARank = "kv_lora_rank"
        case scaleDepth = "scale_depth"
        case scaleEmbedding = "scale_emb"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionBias = "attention_bias"
        case ropeTheta = "rope_theta"
        case ropeTraditional = "rope_traditional"
        case ropeScaling = "rope_scaling"
        case tieWordEmbeddings = "tie_word_embeddings"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "minicpm3"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        baseModelDimensions = try c.decode(Int.self, forKey: .baseModelDimensions)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        qLoRARank = try c.decode(Int.self, forKey: .qLoRARank)
        qkNopeHeadDimensions = try c.decode(Int.self, forKey: .qkNopeHeadDimensions)
        qkRopeHeadDimensions = try c.decode(Int.self, forKey: .qkRopeHeadDimensions)
        kvLoRARank = try c.decode(Int.self, forKey: .kvLoRARank)
        scaleDepth = try c.decode(Float.self, forKey: .scaleDepth)
        scaleEmbedding = try c.decode(Float.self, forKey: .scaleEmbedding)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
        ropeTraditional = try c.decodeIfPresent(Bool.self, forKey: .ropeTraditional) ?? false
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
    }
}

extension MiniCPM3Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
