import Foundation
import MLX
import MLXNN

/// Quantizes to 8 bits and back at a learned scale, in float32.
///
/// The clamp is asymmetric -- `[-128, 127]`, as int8 is -- so it is not a no-op even for
/// values that round to the boundary.
func afm7FakeEightBitQuantization(_ x: MLXArray, scale: MLXArray) -> MLXArray {
    let wide = x.asType(.float32)
    let levels = clip(round(wide / scale), min: -128, max: 127)
    return (levels * scale).asType(x.dtype)
}

class AFM7Attention: Module {

    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    @ParameterInfo(key: "quant_key_scale") var keyScale: MLXArray
    @ParameterInfo(key: "quant_value_scale") var valueScale: MLXArray

    let rope: RoPE

    init(_ args: AFM7Configuration) {
        self.heads = args.heads
        self.kvHeads = args.kvHeads
        self.headDim = args.hiddenDimensions / args.heads
        self.scale = pow(Float(headDim), -0.5)

        _qkvProj.wrappedValue = Linear(
            args.hiddenDimensions, (heads + 2 * kvHeads) * headDim, bias: false)
        _outProj.wrappedValue = Linear(
            args.hiddenDimensions, args.hiddenDimensions, bias: false)

        _qNorm.wrappedValue = RMSNorm(dimensions: headDim)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim)

        _keyScale.wrappedValue = MLXArray(Float(1))
        _valueScale.wrappedValue = MLXArray(Float(1))

        self.rope = RoPE(dimensions: headDim, traditional: true, base: args.ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        let queryEnd = heads * headDim
        let parts = split(
            qkvProj(x), indices: [queryEnd, queryEnd + kvHeads * headDim], axis: -1)

        var queries = parts[0].reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        var keys = parts[1].reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        var values = parts[2].reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        let offset = cache?.offset ?? 0
        queries = qNorm(rope(queries, offset: offset))
        keys = kNorm(rope(keys, offset: offset))

        keys = afm7FakeEightBitQuantization(keys, scale: keyScale)
        values = afm7FakeEightBitQuantization(values, scale: valueScale)

        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)

        return outProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

/// Queries only. Keys and values arrive from the last ordinary layer's cache, and the
/// rotary offset tracks the logical position even when the cache evicts older tokens.
class AFM7KVReuseAttention: Module {

    let heads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm

    let rope: RoPE

    init(_ args: AFM7Configuration) {
        self.heads = args.heads
        self.headDim = args.hiddenDimensions / args.heads
        self.scale = pow(Float(headDim), -0.5)

        _wq.wrappedValue = Linear(
            args.hiddenDimensions, args.hiddenDimensions, bias: false)
        _outProj.wrappedValue = Linear(
            args.hiddenDimensions, args.hiddenDimensions, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim)

        self.rope = RoPE(dimensions: headDim, traditional: true, base: args.ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray, keys: MLXArray, values: MLXArray, offset: Int,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var queries = wq(x).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        queries = qNorm(rope(queries, offset: offset))

        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)

        return outProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

class AFM7Block: Module {

    @ModuleInfo(key: "self_attn") var attention: AFM7Attention
    @ModuleInfo(key: "mlp") var mlp: SwiGLUMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: AFM7Configuration) {
        _attention.wrappedValue = AFM7Attention(args)
        _mlp.wrappedValue = SwiGLUMLP(
            inputDimensions: args.hiddenDimensions,
            hiddenDimensions: args.feedForwardDimensions)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenDimensions, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenDimensions, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

class AFM7KVReuseBlock: Module {

    @ModuleInfo(key: "self_attn") var attention: AFM7KVReuseAttention
    @ModuleInfo(key: "mlp") var mlp: SwiGLUMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: AFM7Configuration) {
        _attention.wrappedValue = AFM7KVReuseAttention(args)
        _mlp.wrappedValue = SwiGLUMLP(
            inputDimensions: args.hiddenDimensions,
            hiddenDimensions: args.feedForwardDimensions)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenDimensions, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenDimensions, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, keys: MLXArray, values: MLXArray, offset: Int,
        mask: MLXFast.ScaledDotProductAttentionMaskMode
    ) -> MLXArray {
        let h =
            x + attention(inputLayerNorm(x), keys: keys, values: values, offset: offset, mask: mask)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class AFM7ModelInner: Module {

    @ModuleInfo(key: "embedding") var embedding: Embedding

    let layers: [AFM7Block]

    @ModuleInfo(key: "kv_reuse_layers") var kvReuseLayers: [AFM7KVReuseBlock]

    @ModuleInfo(key: "output_norm") var outputNorm: RMSNorm

    public init(_ args: AFM7Configuration) {
        precondition(args.vocabularySize > 0)
        precondition(
            args.layerCount > args.kvReuseLayerCount,
            "num_kv_reuse_layers must leave at least one ordinary layer to reuse from")

        _embedding.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenDimensions)
        self.layers = (0 ..< (args.layerCount - args.kvReuseLayerCount)).map { _ in
            AFM7Block(args)
        }
        _kvReuseLayers.wrappedValue = (0 ..< args.kvReuseLayerCount).map { _ in
            AFM7KVReuseBlock(args)
        }
        _outputNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenDimensions, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedding(inputs)

        let capture = cache?[layers.count - 1] ?? KVCacheSimple()
        let offset = capture.offset
        let mask = createAttentionMask(h: h, cache: cache?[0])

        for (i, layer) in layers.enumerated() {
            let entry: KVCache? = i == layers.count - 1 ? capture : cache?[i]
            h = layer(h, mask: mask, cache: entry)
        }

        if !kvReuseLayers.isEmpty {
            let recorded = capture.state
            precondition(
                recorded.count == 2, "the reuse layers have no keys to attend to")
            for layer in kvReuseLayers {
                h = layer(h, keys: recorded[0], values: recorded[1], offset: offset, mask: mask)
            }
        }

        return outputNorm(h)
    }
}

public class AFM7Model: Module, LLMModel, KVCacheDimensionProvider {

    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: AFM7ModelInner

    public init(_ args: AFM7Configuration) {
        self.vocabularySize = args.vocabularySize
        self.model = AFM7ModelInner(args)
        self.kvHeads = Array(repeating: args.kvHeads, count: model.layers.count)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model.embedding.asLinear(model(inputs, cache: cache))
    }

    /// One entry per *ordinary* layer. The reuse layers share the last one's.
    public func newCache(configuration: KVCacheConfiguration?) throws -> [KVCache] {
        try model.layers.map { _ in
            MaterializedKVCache(try makeAttentionKVCache(configuration: configuration))
        }
    }
}

public struct AFM7Configuration: Decodable, Sendable {

    var modelType: String
    var vocabularySize: Int
    var hiddenDimensions: Int
    var layerCount: Int
    var kvReuseLayerCount: Int
    var heads: Int
    var kvHeads: Int
    var hiddenDimensionScaleFactor: Double
    var ropeTheta: Float
    var rmsNormEps: Float

    /// `int(hidden_dim * hidden_dim_scale_factor)`, truncated.
    var feedForwardDimensions: Int {
        Int(Double(hiddenDimensions) * hiddenDimensionScaleFactor)
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenDimensions = "hidden_dim"
        case layerCount = "num_layers"
        case kvReuseLayerCount = "num_kv_reuse_layers"
        case heads = "num_heads"
        case kvHeads = "num_kv_heads"
        case hiddenDimensionScaleFactor = "hidden_dim_scale_factor"
        case ropeTheta = "rope_theta"
        case rmsNormEps = "rms_norm_eps"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "afm7"
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenDimensions = try c.decode(Int.self, forKey: .hiddenDimensions)
        layerCount = try c.decode(Int.self, forKey: .layerCount)
        kvReuseLayerCount = try c.decode(Int.self, forKey: .kvReuseLayerCount)
        heads = try c.decode(Int.self, forKey: .heads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        hiddenDimensionScaleFactor =
            try c.decodeIfPresent(Double.self, forKey: .hiddenDimensionScaleFactor) ?? 3.25
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 50_000
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-5
    }
}

extension AFM7Model: LoRAModel {
    public var loraLayers: [Module] {
        model.layers + model.kvReuseLayers
    }
}
