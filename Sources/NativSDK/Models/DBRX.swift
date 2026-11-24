import Foundation
import MLX
import MLXNN

class DBRXAttention: Module {

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let modelDim: Int
    let scale: Float
    let clipQKV: Float?

    @ModuleInfo(key: "Wqkv") var wqkv: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    let rope: RoPE

    init(_ args: DBRXConfiguration) {
        self.heads = args.heads
        self.kvHeads = args.attention.kvHeads
        self.modelDim = args.modelDimensions
        self.headDim = args.modelDimensions / args.heads
        self.scale = pow(Float(headDim), -0.5)
        self.clipQKV = args.attention.clipQKV

        _wqkv.wrappedValue = Linear(
            args.modelDimensions, (kvHeads * 2 + heads) * headDim, bias: false)
        _outProj.wrappedValue = Linear(
            args.modelDimensions, args.modelDimensions, bias: false)

        self.rope = RoPE(
            dimensions: headDim, traditional: false, base: args.attention.ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var fused = wqkv(x)
        if let clipQKV {
            fused = clip(fused, min: -clipQKV, max: clipQKV)
        }
        let parts = split(
            fused, indices: [modelDim, modelDim + headDim * kvHeads], axis: -1)

        var queries = parts[0].reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        var keys = parts[1].reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        let values = parts[2].reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values, cache: cache,
            scale: scale, mask: mask)

        return outProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

/// Normalizes, attends, adds the residual, and normalizes again.
///
/// Returns both halves: the residual the block adds back and the normalized value its
/// feed-forward consumes.
class DBRXNormAttnNorm: Module {

    @ModuleInfo(key: "norm_1") var norm1: LayerNorm
    @ModuleInfo(key: "norm_2") var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attention: DBRXAttention

    init(_ args: DBRXConfiguration) {
        _norm1.wrappedValue = LayerNorm(dimensions: args.modelDimensions, bias: false)
        _norm2.wrappedValue = LayerNorm(dimensions: args.modelDimensions, bias: false)
        _attention.wrappedValue = DBRXAttention(args)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> (residual: MLXArray, normalized: MLXArray) {
        let residual = attention(norm1(x), mask: mask, cache: cache) + x
        return (residual, norm2(residual))
    }
}

class DBRXRouter: Module {
    @ModuleInfo(key: "layer") var layer: Linear

    init(modelDimensions: Int, experts: Int) {
        _layer.wrappedValue = Linear(modelDimensions, experts, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        layer(x)
    }
}

class DBRXSparseMoeBlock: Module {

    let topK: Int

    @ModuleInfo(key: "router") var router: DBRXRouter
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

    init(_ args: DBRXConfiguration) {
        self.topK = args.feedForward.topK
        _router.wrappedValue = DBRXRouter(
            modelDimensions: args.modelDimensions, experts: args.feedForward.experts)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.modelDimensions, hiddenDims: args.feedForward.hiddenSize,
            numExperts: args.feedForward.experts)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = softmax(router(x).asType(.float32), axis: -1)

        let indices = stopGradient(
            MLX.argPartition(gates, kth: gates.dim(-1) - topK, axis: -1)[
                .ellipsis, (-topK)...])
        var scores = takeAlong(gates, indices, axis: -1)
        scores = scores / abs(scores).sum(axis: -1, keepDims: true)

        return (switchMLP(x, indices) * scores.asType(x.dtype)[.ellipsis, .newAxis])
            .sum(axis: -2)
    }
}

class DBRXBlock: Module {

    @ModuleInfo(key: "norm_attn_norm") var normAttentionNorm: DBRXNormAttnNorm
    @ModuleInfo(key: "ffn") var ffn: DBRXSparseMoeBlock

    init(_ args: DBRXConfiguration) {
        _normAttentionNorm.wrappedValue = DBRXNormAttnNorm(args)
        _ffn.wrappedValue = DBRXSparseMoeBlock(args)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (residual, normalized) = normAttentionNorm(x, mask: mask, cache: cache)
        return ffn(normalized) + residual
    }
}

public class DBRXTransformer: Module {

    @ModuleInfo(key: "wte") var wte: Embedding

    let blocks: [DBRXBlock]

    @ModuleInfo(key: "norm_f") var normF: LayerNorm

    public init(_ args: DBRXConfiguration) {
        precondition(args.vocabularySize > 0)
        _wte.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.modelDimensions)
        self.blocks = (0 ..< args.layers).map { _ in DBRXBlock(args) }
        _normF.wrappedValue = LayerNorm(dimensions: args.modelDimensions, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = wte(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)

        for (i, block) in blocks.enumerated() {
            h = block(h, mask: mask, cache: cache?[i])
        }

        return normF(h)
    }
}

public class DBRXModel: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]

    @ModuleInfo(key: "transformer") public var transformer: DBRXTransformer
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let configuration: DBRXConfiguration

    public init(_ args: DBRXConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = Array(repeating: args.attention.kvHeads, count: args.layers)
        _transformer.wrappedValue = DBRXTransformer(args)
        _lmHead.wrappedValue = Linear(
            args.modelDimensions, args.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(transformer(inputs, cache: cache))
    }

    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try transformer.blocks.map { _ in try makeAttentionKVCache(configuration: configuration) }
    }

    /// Splits each stacked expert tensor into the layout ``SwitchGLU`` expects.
    ///
    /// A checkpoint stores all experts of a projection in one `experts.mlp.<name>` tensor of
    /// shape `[experts * ffn_hidden_size, d_model]`, with no `.weight` suffix. `w2` is stored
    /// transposed relative to the other two.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        let projections = ["w1": "gate_proj", "v1": "up_proj", "w2": "down_proj"]
        let experts = configuration.feedForward.experts
        var sanitized: [String: MLXArray] = [:]

        for (key, value) in weights {
            guard let marker = key.range(of: ".experts.mlp.") else {
                sanitized[key] = value
                continue
            }

            var name = String(key[marker.upperBound...])
            var suffix = ".weight"
            if let dot = name.firstIndex(of: ".") {
                suffix = String(name[dot...])
                name = String(name[name.startIndex ..< dot])
            }

            guard let projection = projections[name] else {
                sanitized[key] = value
                continue
            }

            var stacked = value.reshaped(experts, -1, value.dim(-1))
            if name == "w2" {
                stacked = stacked.swappedAxes(-1, -2)
            }
            let prefix = String(key[key.startIndex ..< marker.lowerBound])
            sanitized["\(prefix).switch_mlp.\(projection)\(suffix)"] = stacked
        }

        return sanitized
    }
}

public struct DBRXConfiguration: Decodable, Sendable {

    /// The `attn_config` sub-dictionary.
    public struct AttentionConfiguration: Decodable, Sendable {
        var kvHeads: Int
        var clipQKV: Float?
        var ropeTheta: Float

        enum CodingKeys: String, CodingKey {
            case kvHeads = "kv_n_heads"
            case clipQKV = "clip_qkv"
            case ropeTheta = "rope_theta"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            kvHeads = try c.decode(Int.self, forKey: .kvHeads)
            clipQKV = try c.decodeIfPresent(Float.self, forKey: .clipQKV)
            ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10_000
        }
    }

    /// The `ffn_config` sub-dictionary.
    public struct FeedForwardConfiguration: Decodable, Sendable {
        var hiddenSize: Int
        var experts: Int
        var topK: Int

        enum CodingKeys: String, CodingKey {
            case hiddenSize = "ffn_hidden_size"
            case experts = "moe_num_experts"
            case topK = "moe_top_k"
        }
    }

    var modelType: String
    var vocabularySize: Int
    var modelDimensions: Int
    var layers: Int
    var heads: Int
    var attention: AttentionConfiguration
    var feedForward: FeedForwardConfiguration

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case modelDimensions = "d_model"
        case layers = "n_layers"
        case heads = "n_heads"
        case attention = "attn_config"
        case feedForward = "ffn_config"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "dbrx"
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        modelDimensions = try c.decode(Int.self, forKey: .modelDimensions)
        layers = try c.decode(Int.self, forKey: .layers)
        heads = try c.decode(Int.self, forKey: .heads)
        attention = try c.decode(AttentionConfiguration.self, forKey: .attention)
        feedForward = try c.decode(FeedForwardConfiguration.self, forKey: .feedForward)
    }
}

extension DBRXModel: LoRAModel {
    public var loraLayers: [Module] {
        transformer.blocks
    }
}
