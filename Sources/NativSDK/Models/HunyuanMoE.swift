import Foundation
import MLX
import MLXNN

/// Rotary positions whose base is pre-scaled once by `alpha ** (dims / (dims - 2))`.
///
/// Unlike the dynamic NTK rope internlm uses, nothing is recomputed per call -- the alpha is
/// a fixed property of the checkpoint, folded into the base at construction.
public class HunyuanMoEDynamicNTKAlphaRoPE: Module, OffsetLayer, ArrayOffsetLayer {
    let dimensions: Int
    let _freqs: MLXArray

    public init(dimensions: Int, base: Float = 10000, scalingAlpha: Float = 1) {
        precondition(dimensions % 2 == 0, "Dimensions must be even")
        self.dimensions = dimensions
        let scaled = base * pow(scalingAlpha, Float(dimensions) / Float(dimensions - 2))
        self._freqs = pow(scaled, arange(0, dimensions, step: 2, dtype: .float32) / dimensions)
    }

    public func callAsFunction(_ x: MLXArray, offset: Int = 0) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: dimensions, traditional: false, base: nil, scale: 1.0,
            offset: offset, freqs: _freqs)
    }

    public func callAsFunction(_ x: MLXArray, offset: MLXArray) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: dimensions, traditional: false, base: nil, scale: 1.0,
            offset: offset, freqs: _freqs)
    }
}

class HunyuanMoEAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let scale: Float
    let usesQKNorm: Bool

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear?
    @ModuleInfo(key: "v_proj") var wv: Linear?
    @ModuleInfo(key: "o_proj") var wo: Linear

    @ModuleInfo(key: "query_layernorm") var queryNorm: RMSNorm?
    @ModuleInfo(key: "key_layernorm") var keyNorm: RMSNorm?

    let rope: HunyuanMoEDynamicNTKAlphaRoPE

    init(_ args: HunyuanMoEConfiguration, ownsKeyValueProjections: Bool) {
        self.heads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.headDimensions = args.hiddenSize / args.attentionHeads
        self.scale = pow(Float(headDimensions), -0.5)
        self.usesQKNorm = args.useQKNorm

        _wq.wrappedValue = Linear(
            args.hiddenSize, heads * headDimensions, bias: args.attentionBias)
        // Absent on a consumer layer: there is no such weight in the checkpoint.
        if ownsKeyValueProjections {
            _wk.wrappedValue = Linear(
                args.hiddenSize, kvHeads * headDimensions, bias: args.attentionBias)
            _wv.wrappedValue = Linear(
                args.hiddenSize, kvHeads * headDimensions, bias: args.attentionBias)
        }
        _wo.wrappedValue = Linear(
            heads * headDimensions, args.hiddenSize, bias: args.attentionBias)

        if args.useQKNorm {
            _queryNorm.wrappedValue = RMSNorm(
                dimensions: headDimensions, eps: args.rmsNormEps)
            _keyNorm.wrappedValue = RMSNorm(dimensions: headDimensions, eps: args.rmsNormEps)
        }

        self.rope = HunyuanMoEDynamicNTKAlphaRoPE(
            dimensions: headDimensions, base: args.ropeTheta,
            scalingAlpha: args.ropeScalingAlpha)
    }

    /// - Parameter sharedKeyValues: a producer layer's raw projections, when this layer is
    ///   a consumer. Returned either way, so the caller can pass it along.
    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        sharedKeyValues: (MLXArray, MLXArray)?
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (B, L) = (x.dim(0), x.dim(1))

        var keyProjection: MLXArray
        var valueProjection: MLXArray
        let carried: (MLXArray, MLXArray)
        if let sharedKeyValues {
            (keyProjection, valueProjection) = sharedKeyValues
            carried = sharedKeyValues
        } else {
            keyProjection = wk!(x)
            valueProjection = wv!(x)
            carried = (keyProjection, valueProjection)
        }

        var queries = wq(x).reshaped(B, L, heads, -1).transposed(0, 2, 1, 3)
        var keys = keyProjection.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        let values = valueProjection.reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)

        // Each layer ropes the shared projections itself, at its own offset.
        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        // After the rotation, not before -- the reverse would normalize away the positions.
        if let queryNorm, let keyNorm {
            queries = queryNorm(queries)
            keys = keyNorm(keys)
        }

        let output = attentionWithCacheUpdate(
            queries: queries, keys: keys, values: values,
            cache: cache, scale: scale, mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return (wo(output), carried)
    }
}

class HunyuanMoEGate: Module {
    @ModuleInfo var wg: Linear

    init(dimensions: Int, experts: Int) {
        _wg.wrappedValue = Linear(dimensions, experts, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { wg(x) }
}

class HunyuanMoESparseBlock: Module, UnaryLayer {
    let topK: Int
    let usesSharedMLP: Bool

    let gate: HunyuanMoEGate
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_mlp") var sharedMLP: SwiGLUMLP?

    init(_ args: HunyuanMoEConfiguration, layerIndex: Int) {
        self.topK = args.moeTopK(atLayer: layerIndex)
        self.usesSharedMLP = args.useMixedMLPMoE

        let expertWidth = args.moeIntermediateSize(atLayer: layerIndex)
        self.gate = HunyuanMoEGate(dimensions: args.hiddenSize, experts: args.experts)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: expertWidth, numExperts: args.experts)

        if args.useMixedMLPMoE {
            let shared = args.sharedExperts(atLayer: layerIndex)
            _sharedMLP.wrappedValue = SwiGLUMLP(
                inputDimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize(atLayer: layerIndex) * shared)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gates = softmax(gate(x), axis: -1, precise: true)
        let indices = MLX.stopGradient(
            MLX.argPartition(-gates, kth: topK - 1, axis: -1)
        )[.ellipsis, ..<topK]
        let scores = takeAlong(gates, indices, axis: -1)

        var y = (switchMLP(x, indices) * scores[.ellipsis, .newAxis]).sum(axis: -2)
        if let sharedMLP {
            y = y + sharedMLP(x)
        }
        return y
    }
}

class HunyuanMoEBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: HunyuanMoEAttention
    @ModuleInfo(key: "mlp") var mlp: UnaryLayer

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    /// Whether this layer owns the key and value projections the next few will share.
    let producesKeyValues: Bool

    init(_ args: HunyuanMoEConfiguration, layerIndex: Int) {
        self.producesKeyValues = args.producesKeyValues(atLayer: layerIndex)
        _attention.wrappedValue = HunyuanMoEAttention(
            args, ownsKeyValueProjections: producesKeyValues)
        _mlp.wrappedValue =
            args.experts != 1
            ? HunyuanMoESparseBlock(args, layerIndex: layerIndex)
            : SwiGLUMLP(
                inputDimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize(atLayer: layerIndex))
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?,
        sharedKeyValues: (MLXArray, MLXArray)?
    ) -> (MLXArray, (MLXArray, MLXArray)) {
        let (attended, carried) = attention(
            inputLayerNorm(x), mask: mask, cache: cache, sharedKeyValues: sharedKeyValues)
        let h = x + attended
        return (h + mlp(postAttentionLayerNorm(h)), carried)
    }
}

public class HunyuanMoEModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [HunyuanMoEBlock]
    let norm: RMSNorm

    public init(_ args: HunyuanMoEConfiguration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.hiddenLayers).map { HunyuanMoEBlock(args, layerIndex: $0) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)
        let mask = createAttentionMask(h: h, cache: cache?.first)

        // Reset at each producer, so a run of consumers reads the producer immediately
        // above them rather than the first one in the stack.
        var shared: (MLXArray, MLXArray)?
        for (i, layer) in layers.enumerated() {
            if layer.producesKeyValues {
                shared = nil
            }
            (h, shared) = layer(
                h, mask: mask, cache: cache?[i], sharedKeyValues: shared)
        }

        return norm(h)
    }
}

public class HunyuanMoEModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: HunyuanMoEModelInner

    public init(_ args: HunyuanMoEConfiguration) {
        self.vocabularySize = args.vocabularySize
        // Every layer keeps its own cache, consumer or not: what CLA shares is the
        // projections, and each layer ropes and caches them itself.
        self.kvHeads = (0 ..< args.hiddenLayers).map { _ in args.kvHeads }
        self.model = HunyuanMoEModelInner(args)
    }

    /// Always the tied embedding -- this family has no `lm_head` in its tree.
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        model.embedTokens.asLinear(model(inputs, cache: cache))
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights.filter { $0.key != "lm_head.weight" && !$0.key.contains("rotary_emb.inv_freq") }
    }
}

public struct HunyuanMoEConfiguration: Decodable, Sendable {
    var modelType: String
    var hiddenSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var vocabularySize: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var attentionBias: Bool
    var useQKNorm: Bool
    var useCLA: Bool
    var claShareFactor: Int
    var experts: Int
    var useMixedMLPMoE: Bool
    var tieWordEmbeddings: Bool
    var ropeScalingAlpha: Float

    /// Fields that may be a single value or one value per layer.
    private var intermediateSizes: [Int]
    private var moeIntermediateSizes: [Int]
    private var moeTopKs: [Int]
    private var sharedExpertCounts: [Int]

    /// Whether layer `index` owns the key and value projections.
    ///
    /// Without CLA every layer owns its own. With it, every `cla_share_factor`-th does and
    /// the rest have no such weight at all.
    func producesKeyValues(atLayer index: Int) -> Bool {
        !useCLA || index % claShareFactor == 0
    }

    func intermediateSize(atLayer index: Int) -> Int { Self.at(intermediateSizes, index) }
    func moeIntermediateSize(atLayer index: Int) -> Int {
        Self.at(moeIntermediateSizes, index)
    }
    func moeTopK(atLayer index: Int) -> Int { Self.at(moeTopKs, index) }
    func sharedExperts(atLayer index: Int) -> Int {
        Self.at(sharedExpertCounts, index)
    }

    private static func at(_ values: [Int], _ index: Int) -> Int {
        guard !values.isEmpty else { return 0 }
        return values.count == 1 ? values[0] : values[min(index, values.count - 1)]
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case vocabularySize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case attentionBias = "attention_bias"
        case useQKNorm = "use_qk_norm"
        case useCLA = "use_cla"
        case claShareFactor = "cla_share_factor"
        case experts = "num_experts"
        case useMixedMLPMoE = "use_mixed_mlp_moe"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeScaling = "rope_scaling"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case moeTopK = "moe_topk"
        case sharedExpert = "num_shared_expert"
    }

    /// Accepts a shared scalar or per-layer values.
    private static func values(
        _ c: KeyedDecodingContainer<CodingKeys>, _ key: CodingKeys, default fallback: [Int]
    ) throws -> [Int] {
        if let scalar = try? c.decodeIfPresent(Int.self, forKey: key) { return [scalar] }
        if let list = try? c.decodeIfPresent([Int].self, forKey: key), !list.isEmpty {
            return list
        }
        return fallback
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "hunyuan"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        attentionBias = try c.decode(Bool.self, forKey: .attentionBias)
        useQKNorm = try c.decode(Bool.self, forKey: .useQKNorm)
        useCLA = try c.decode(Bool.self, forKey: .useCLA)
        claShareFactor = try c.decodeIfPresent(Int.self, forKey: .claShareFactor) ?? 2
        experts = try c.decode(Int.self, forKey: .experts)
        useMixedMLPMoE = try c.decode(Bool.self, forKey: .useMixedMLPMoE)
        tieWordEmbeddings = try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false

        intermediateSizes = try Self.values(c, .intermediateSize, default: [0])
        moeIntermediateSizes = try Self.values(c, .moeIntermediateSize, default: intermediateSizes)
        moeTopKs = try Self.values(c, .moeTopK, default: [1])
        sharedExpertCounts = try Self.values(c, .sharedExpert, default: [0])

        // A supplied scaling configuration must identify its type and factor.
        let scaling =
            try c.decodeIfPresent([String: StringOrNumber].self, forKey: .ropeScaling) ?? [:]
        func number(_ key: String) -> Float? {
            switch scaling[key] {
            case .float(let value): value
            case .int(let value): Float(value)
            default: nil
            }
        }
        if scaling.isEmpty {
            ropeScalingAlpha = 1
        } else {
            guard scaling["factor"] != nil, scaling["type"] != nil else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: c.codingPath,
                        debugDescription: "rope_scaling must contain 'factor' and 'type'"))
            }
            guard let alpha = number("alpha") else {
                throw DecodingError.dataCorrupted(
                    .init(
                        codingPath: c.codingPath,
                        debugDescription:
                            "rope_scaling must carry 'alpha'; this family folds it into the "
                            + "rotary base and cannot default it"))
            }
            ropeScalingAlpha = alpha
        }
    }
}

extension HunyuanMoEModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
