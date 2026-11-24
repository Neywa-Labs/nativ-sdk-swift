import Foundation
import MLX
import MLXNN

class DeepseekAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDimensions: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var wq: Linear
    @ModuleInfo(key: "k_proj") var wk: Linear
    @ModuleInfo(key: "v_proj") var wv: Linear
    @ModuleInfo(key: "o_proj") var wo: Linear

    let rope: RoPELayer

    init(_ config: DeepseekConfiguration) {
        self.heads = config.attentionHeads
        self.kvHeads = config.kvHeads
        self.headDimensions = config.hiddenSize / config.attentionHeads
        self.scale = pow(Float(headDimensions), -0.5)

        _wq.wrappedValue = Linear(
            config.hiddenSize, heads * headDimensions, bias: config.attentionBias)
        _wk.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDimensions, bias: config.attentionBias)
        _wv.wrappedValue = Linear(
            config.hiddenSize, kvHeads * headDimensions, bias: config.attentionBias)
        _wo.wrappedValue = Linear(
            heads * headDimensions, config.hiddenSize, bias: config.attentionBias)

        self.rope = initializeRope(
            dims: headDimensions,
            base: config.ropeTheta,
            traditional: false,
            scalingConfig: config.ropeScaling,
            maxPositionEmbeddings: config.maxPositionEmbeddings)
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

/// The router, whose weight is a bare parameter rather than a `Linear`.
///
/// A `Linear` here would put the checkpoint's `mlp.gate.weight` one level too deep.
class DeepseekMoEGate: Module {
    let topK: Int

    @ParameterInfo(key: "weight") var weight: MLXArray

    init(_ config: DeepseekConfiguration) {
        self.topK = config.expertsPerToken ?? 1
        _weight.wrappedValue = MLXArray.zeros(
            [config.routedExperts ?? 0, config.hiddenSize])
    }

    /// Softmaxes every logit and *then* selects, so the kept weights sum to less than one
    /// -- qwen2_moe's order, not phixtral's.
    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        let gates = matmul(x, weight.transposed())
        let scores = softmax(gates, axis: -1, precise: true)
        let indices = MLX.stopGradient(
            MLX.argPartition(-scores, kth: topK - 1, axis: -1)
        )[.ellipsis, ..<topK]
        return (indices, takeAlong(scores, indices, axis: -1))
    }
}

class DeepseekMoE: Module, UnaryLayer {
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    let gate: DeepseekMoEGate
    @ModuleInfo(key: "shared_experts") var sharedExperts: SwiGLUMLP?

    init(_ config: DeepseekConfiguration) {
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: config.hiddenSize, hiddenDims: config.moeIntermediateSize,
            numExperts: config.routedExperts ?? 0)
        self.gate = DeepseekMoEGate(config)

        // The shared experts are one MLP whose width is a *multiple* of the routed width,
        // not one MLP per shared expert.
        if let shared = config.sharedExperts {
            _sharedExperts.wrappedValue = SwiGLUMLP(
                inputDimensions: config.hiddenSize,
                hiddenDimensions: config.moeIntermediateSize * shared)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = gate(x)
        var y = (switchMLP(x, indices) * scores[.ellipsis, .newAxis]).sum(axis: -2)
        if let sharedExperts {
            y = y + sharedExperts(x)
        }
        return y
    }
}

class DeepseekBlock: Module {
    @ModuleInfo(key: "self_attn") var attention: DeepseekAttention
    @ModuleInfo(key: "mlp") var mlp: UnaryLayer

    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    /// Recorded so a test can see which kind this layer got without reflection.
    let isMixture: Bool

    init(_ config: DeepseekConfiguration, layerIndex: Int) {
        _attention.wrappedValue = DeepseekAttention(config)
        self.isMixture = config.isMixtureLayer(layerIndex)
        _mlp.wrappedValue =
            isMixture
            ? DeepseekMoE(config)
            : SwiGLUMLP(
                inputDimensions: config.hiddenSize,
                hiddenDimensions: config.intermediateSize)
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

public class DeepseekModelInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [DeepseekBlock]
    let norm: RMSNorm

    public init(_ config: DeepseekConfiguration) {
        precondition(config.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.hiddenLayers).map {
            DeepseekBlock(config, layerIndex: $0)
        }
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

public class DeepseekModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    public let model: DeepseekModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let configuration: DeepseekConfiguration

    public init(_ config: DeepseekConfiguration) {
        self.configuration = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.hiddenLayers).map { _ in config.kvHeads }
        self.model = DeepseekModelInner(config)
        _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    /// Stacks each mixture layer's separately shipped experts, and only those layers.
    ///
    /// A dense layer has no `mlp.experts.*` keys at all, so the loop asks each layer rather
    /// than assuming the stack is uniform.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        guard let experts = configuration.routedExperts else { return weights }

        var sanitized = weights
        for layer in 0 ..< configuration.hiddenLayers
        where configuration.isMixtureLayer(layer) {
            let prefix = "model.layers.\(layer)"
            for projection in ["gate_proj", "down_proj", "up_proj"] {
                for field in ["weight", "scales", "biases"] {
                    guard sanitized["\(prefix).mlp.experts.0.\(projection).\(field)"] != nil
                    else { continue }

                    var stacked: [MLXArray] = []
                    for expert in 0 ..< experts {
                        let key = "\(prefix).mlp.experts.\(expert).\(projection).\(field)"
                        guard let value = sanitized.removeValue(forKey: key) else { break }
                        stacked.append(value)
                    }
                    guard stacked.count == experts else { continue }
                    sanitized["\(prefix).mlp.switch_mlp.\(projection).\(field)"] =
                        MLX.stacked(stacked)
                }
            }
        }
        return sanitized
    }
}

public struct DeepseekConfiguration: Codable, Sendable {
    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var intermediateSize: Int
    var moeIntermediateSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var sharedExperts: Int?
    var routedExperts: Int?
    var expertsPerToken: Int?
    var moeLayerFrequency: Int
    var firstKDenseReplace: Int
    var maxPositionEmbeddings: Int
    var rmsNormEps: Float
    var ropeTheta: Float
    var ropeScaling: [String: StringOrNumber]?
    var attentionBias: Bool

    /// Whether layer `index` gets a mixture rather than a dense MLP.
    ///
    /// Three conditions, all of which must hold: the model has routed experts, the layer is
    /// at or past `first_k_dense_replace`, and its index falls on `moe_layer_freq`. A
    /// checkpoint's keys differ per layer accordingly.
    func isMixtureLayer(_ index: Int) -> Bool {
        routedExperts != nil && index >= firstKDenseReplace
            && index % moeLayerFrequency == 0
    }

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case moeIntermediateSize = "moe_intermediate_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case sharedExperts = "n_shared_experts"
        case routedExperts = "n_routed_experts"
        case expertsPerToken = "num_experts_per_tok"
        case moeLayerFrequency = "moe_layer_freq"
        case firstKDenseReplace = "first_k_dense_replace"
        case maxPositionEmbeddings = "max_position_embeddings"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case ropeScaling = "rope_scaling"
        case attentionBias = "attention_bias"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "deepseek"
        vocabularySize = try c.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 102_400
        hiddenSize = try c.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        intermediateSize = try c.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 11008
        moeIntermediateSize =
            try c.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 1407
        hiddenLayers = try c.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 30
        attentionHeads = try c.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
        kvHeads = try c.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 32
        sharedExperts = try c.decodeIfPresent(Int.self, forKey: .sharedExperts)
        routedExperts = try c.decodeIfPresent(Int.self, forKey: .routedExperts)
        expertsPerToken = try c.decodeIfPresent(Int.self, forKey: .expertsPerToken)
        moeLayerFrequency = try c.decodeIfPresent(Int.self, forKey: .moeLayerFrequency) ?? 1
        firstKDenseReplace = try c.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        maxPositionEmbeddings =
            try c.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 2048
        rmsNormEps = try c.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        ropeTheta = try c.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
    }
}

extension DeepseekModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
