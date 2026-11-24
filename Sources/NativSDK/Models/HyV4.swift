import Foundation
import MLX
import MLXNN

/// Collapses the multi-copy residual into one stream and returns the coefficients that
/// fold a block's output back into it.
class HyV4IdentityHyperConnectionPre: Module {
    let multiplier: Int
    let eps: Float
    let magnitude: Float
    let normEps: Float

    @ParameterInfo(key: "hc_fn") var fn: MLXArray
    @ParameterInfo(key: "hc_base") var base: MLXArray
    @ParameterInfo(key: "hc_scale") var scale: MLXArray

    init(_ config: HyV4Configuration) {
        self.multiplier = config.hcMultiplier
        self.eps = config.hcEps
        self.magnitude = config.hcMagnitude
        self.normEps = config.rmsNormEps
        _fn.wrappedValue = MLXArray.zeros(
            [2 * multiplier, multiplier * config.hiddenSize], dtype: .float32)
        _base.wrappedValue = MLXArray.zeros([2 * multiplier], dtype: .float32)
        _scale.wrappedValue = MLXArray.ones([2], dtype: .float32)
    }

    func callAsFunction(_ x: MLXArray) -> (collapsed: MLXArray, post: MLXArray) {
        let residual = x.asType(.float32)
        let normalized = MLXFast.rmsNorm(
            residual.flattened(start: -2, end: -1), weight: MLXArray.mlxNone, eps: normEps)
        let mixes = normalized.matmul(fn.T)
        let pre =
            sigmoid(mixes[.ellipsis, ..<multiplier] * scale[0] + base[..<multiplier]) + eps
        let post =
            magnitude
            * sigmoid(
                mixes[.ellipsis, multiplier...] * scale[1] + base[multiplier...])
        let collapsed = (pre[.ellipsis, .newAxis] * residual).sum(axis: 2).asType(x.dtype)
        return (collapsed, post)
    }
}

class HyV4IdentityHyperConnection: Module {
    @ModuleInfo(key: "hc_pre") var pre: HyV4IdentityHyperConnectionPre

    init(_ config: HyV4Configuration) {
        _pre.wrappedValue = HyV4IdentityHyperConnectionPre(config)
    }

    func callAsFunction(_ x: MLXArray) -> (collapsed: MLXArray, post: MLXArray) {
        pre(x)
    }
}

/// Folds a block's output back into the multi-copy residual. Unlike the Sinkhorn variant
/// there is no cross-copy term: each copy receives the same output, scaled per copy.
func hyV4HyperConnectionExpand(_ x: MLXArray, residual: MLXArray, post: MLXArray)
    -> MLXArray
{
    let y =
        residual.asType(.float32)
        + post[.ellipsis, .newAxis] * x[0..., 0..., .newAxis, 0...].asType(.float32)
    return y.asType(residual.dtype)
}

/// Collapses the multi-copy residual once more at the end of the stack.
class HyV4IdentityHyperHead: Module {
    let multiplier: Int
    let eps: Float
    let normEps: Float

    @ParameterInfo(key: "hc_head_fn") var fn: MLXArray
    @ParameterInfo(key: "hc_head_base") var base: MLXArray
    @ParameterInfo(key: "hc_head_scale") var scale: MLXArray

    init(_ config: HyV4Configuration) {
        self.multiplier = config.hcMultiplier
        self.eps = config.hcEps
        self.normEps = config.rmsNormEps
        _fn.wrappedValue = MLXArray.zeros(
            [multiplier, multiplier * config.hiddenSize], dtype: .float32)
        _base.wrappedValue = MLXArray.zeros([multiplier], dtype: .float32)
        _scale.wrappedValue = MLXArray.ones([1], dtype: .float32)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual = x.asType(.float32)
        let normalized = MLXFast.rmsNorm(
            residual.flattened(start: -2, end: -1), weight: MLXArray.mlxNone, eps: normEps)
        let mixes = normalized.matmul(fn.T)
        let pre = sigmoid(mixes * scale + base) + eps
        return (pre[.ellipsis, .newAxis] * residual).sum(axis: 2).asType(x.dtype)
    }
}

/// deepseek_v32's attention with a learned sink and a per-head output gate.
class HyV4Attention: DeepseekV32Attention {
    @ModuleInfo(key: "linear_gate") var linearGate: Linear
    @ParameterInfo(key: "learnable_sink_param") var sinkParameter: MLXArray

    let gateHeads: Int
    let gateHeadDimensions: Int

    init(_ config: HyV4Configuration, layer: Int) {
        let usesOwnIndexer = config.indexerTypes[layer] != "shared"
        self.gateHeads = config.attentionHeads
        self.gateHeadDimensions = config.vHeadDim
        _linearGate.wrappedValue = Linear(
            config.hiddenSize, gateHeads * gateHeadDimensions, bias: false)
        _sinkParameter.wrappedValue = MLXArray.zeros([gateHeads], dtype: .float32)
        super.init(config.deepseekConfiguration, forceFullIndexer: usesOwnIndexer)
    }

    override var allowsAbsorbedPrefill: Bool { false }

    override func attentionSinks(dtype: DType) -> MLXArray? {
        sinkParameter.asType(dtype)
    }

    override func gateAttentionOutput(_ output: MLXArray, input x: MLXArray) -> MLXArray {
        let gate = sigmoid(
            linearGate(x).reshaped(x.dim(0), x.dim(1), gateHeads, gateHeadDimensions)
        )
        .transposed(0, 2, 1, 3)
        return output * gate.asType(output.dtype)
    }
}

class HyV4DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: HyV4Attention
    @ModuleInfo(key: "mlp") var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm
    @ModuleInfo(key: "hc_attn_layer") var hcAttention: HyV4IdentityHyperConnection
    @ModuleInfo(key: "hc_mlp_layer") var hcMLP: HyV4IdentityHyperConnection

    init(_ config: HyV4Configuration, layer: Int) {
        _attention.wrappedValue = HyV4Attention(config, layer: layer)
        let dense = config.deepseekConfiguration
        _mlp.wrappedValue =
            layer < config.firstKDenseReplace
            ? DeepseekV32MLP(dense) : DeepseekV32MoE(dense)
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _hcAttention.wrappedValue = HyV4IdentityHyperConnection(config)
        _hcMLP.wrappedValue = HyV4IdentityHyperConnection(config)
    }

    /// Both branches read a collapsed view of the multi-copy residual and write back
    /// through the coefficients that view produced.
    func callAsFunction(
        _ h: MLXArray, mask: MLXArray?, cache: CacheList?, previousTopKIndices: MLXArray?
    ) -> (MLXArray, MLXArray?) {
        var residual = h
        var (x, post) = hcAttention(h)
        let (attended, topKIndices) = attention(
            inputLayerNorm(x), mask: mask, cache: cache,
            previousTopKIndices: previousTopKIndices)
        var out = hyV4HyperConnectionExpand(attended, residual: residual, post: post)

        residual = out
        (x, post) = hcMLP(out)
        let projected = mlp(postAttentionLayerNorm(x))
        out = hyV4HyperConnectionExpand(projected, residual: residual, post: post)
        return (out, topKIndices)
    }
}

public class HyV4Inner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "hc_head") var hcHead: HyV4IdentityHyperHead
    @ModuleInfo var norm: RMSNorm

    let layers: [HyV4DecoderLayer]
    let multiplier: Int

    public init(_ config: HyV4Configuration) {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        self.layers = (0 ..< config.numHiddenLayers).map { HyV4DecoderLayer(config, layer: $0) }
        _hcHead.wrappedValue = HyV4IdentityHyperHead(config)
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        self.multiplier = config.hcMultiplier
    }

    public func callAsFunction(
        _ inputs: MLXArray, cache: [CacheList]? = nil, inputsEmbeds: MLXArray? = nil
    ) -> MLXArray {
        let embedded = inputsEmbeds ?? embedTokens(inputs)
        var h = broadcast(
            embedded[0..., 0..., .newAxis, 0...],
            to: [embedded.dim(0), embedded.dim(1), multiplier, embedded.dim(2)])
        h = contiguous(h)

        let mask: MLXArray? = createAttentionMask(
            h: h[0..., 0..., 0, 0...], cache: cache?.first.map { [$0[0]] })

        var previousTopKIndices: MLXArray?
        for (index, layer) in layers.enumerated() {
            (h, previousTopKIndices) = layer(
                h, mask: mask, cache: cache?[index],
                previousTopKIndices: previousTopKIndices)
        }
        return norm(hcHead(h))
    }
}

public class HyV4Model: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]
    public let model: HyV4Inner

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let config: HyV4Configuration

    public init(_ config: HyV4Configuration) {
        self.config = config
        self.vocabularySize = config.vocabularySize
        self.kvHeads = (0 ..< config.numHiddenLayers).map { _ in 1 }
        self.model = HyV4Inner(config)
        _lmHead.wrappedValue = Linear(
            config.hiddenSize, config.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache?.compactMap { $0 as? CacheList }))
    }

    /// Two caches per layer: the latent with its positional key, and the indexer's own
    /// narrow one. A shared-indexer layer still gets both, because the pair is what the
    /// attention indexes into.
    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        model.layers.map { _ in
            CacheList(MaterializedKVCache(KVCacheSimple()), MaterializedKVCache(KVCacheSimple()))
        }
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        for (key, value) in weights {
            let trimmed =
                key.hasPrefix("language_model.")
                ? String(key.dropFirst("language_model.".count)) : key
            if trimmed.contains("rotary_emb.inv_freq") { continue }
            sanitized[trimmed] = value
        }
        // Source MXFP8 weights store four bytes per packed word.
        for layer in 0 ..< config.numHiddenLayers {
            let mlp = "model.layers.\(layer).mlp"
            for name in ["gate_up_proj", "down_proj"] {
                let source = "\(mlp).experts.\(name)"
                if let weight = sanitized.removeValue(forKey: source),
                    let scales = sanitized.removeValue(forKey: source + "_scale")
                {
                    sanitized["\(mlp).switch_mlp.\(name).weight"] = weight.view(dtype: .uint32)
                    sanitized["\(mlp).switch_mlp.\(name).scales"] = scales
                }
            }
            for suffix in ["weight", "scales", "biases"] {
                if let fused = sanitized.removeValue(
                    forKey: "\(mlp).switch_mlp.gate_up_proj.\(suffix)")
                {
                    let parts = split(fused, parts: 2, axis: 1)
                    sanitized["\(mlp).switch_mlp.gate_proj.\(suffix)"] = parts[0]
                    sanitized["\(mlp).switch_mlp.up_proj.\(suffix)"] = parts[1]
                }
            }
            let attention = "model.layers.\(layer).self_attn"
            if let scales = sanitized.removeValue(forKey: "\(attention).kv_b_proj.weight_scale"),
                let packed = sanitized.removeValue(forKey: "\(attention).kv_b_proj.weight")
            {
                let dense = dequantized(
                    packed.view(dtype: .uint32), scales: scales, biases: nil,
                    groupSize: 32, bits: 8, mode: .mxfp8
                )
                .reshaped(
                    config.attentionHeads,
                    config.deepseekConfiguration.qkNopeHeadDim + config.vHeadDim, -1)
                let split = config.deepseekConfiguration.qkNopeHeadDim
                for (name, tensor) in [
                    ("embed_q", contiguous(dense[0..., ..<split, 0...].swappedAxes(-1, -2))),
                    ("unembed_out", contiguous(dense[0..., split..., 0...])),
                ] {
                    let (weight, scale, _) = MLX.quantized(
                        tensor, groupSize: 32, bits: 8, mode: .mxfp8)
                    sanitized["\(attention).\(name).weight"] = weight
                    sanitized["\(attention).\(name).scales"] = scale
                }
            }
        }
        for key in Array(sanitized.keys) where key.hasSuffix(".weight_scale") {
            let weightKey = String(key.dropLast("_scale".count))
            if let weight = sanitized[weightKey] {
                sanitized[weightKey] = weight.view(dtype: .uint32)
                sanitized[String(weightKey.dropLast("weight".count)) + "scales"] =
                    sanitized.removeValue(forKey: key)
            }
        }
        DeepseekV32Model.stackExperts(weights: &sanitized, config: config.deepseekConfiguration)
        DeepseekV32Model.splitFusedKV(weights: &sanitized, config: config.deepseekConfiguration)
        return sanitized
    }
}

public struct HyV4Configuration: Decodable, Sendable {
    /// The shared half of the config, decoded from the same flat object. hy_v4 is
    /// deepseek_v32's stack plus the hyper-connection and the indexer pattern, and holding
    /// that config rather than mirroring its thirty fields keeps the two views from
    /// disagreeing about the model they describe.
    public let deepseekConfiguration: DeepseekV32Configuration

    private let _indexerTypes: [String]?
    private let _hcMultiplier: Int?
    public var hcMultiplier: Int { _hcMultiplier ?? 4 }
    private let _hcEps: Float?
    public var hcEps: Float { _hcEps ?? 1e-6 }
    private let _hcMagnitude: Float?
    public var hcMagnitude: Float { _hcMagnitude ?? 2.0 }
    private let _eosTokenId: IntOrIntArray?
    public var eosTokenId: [Int]? { _eosTokenId?.values }

    public var modelType: String { deepseekConfiguration.modelType }
    public var vocabularySize: Int { deepseekConfiguration.vocabularySize }
    public var hiddenSize: Int { deepseekConfiguration.hiddenSize }
    public var intermediateSize: Int { deepseekConfiguration.intermediateSize }
    public var numHiddenLayers: Int { deepseekConfiguration.hiddenLayers }
    public var attentionHeads: Int { deepseekConfiguration.attentionHeads }
    public var vHeadDim: Int { deepseekConfiguration.vHeadDim }
    public var firstKDenseReplace: Int { deepseekConfiguration.firstKDenseReplace }
    public var rmsNormEps: Float { deepseekConfiguration.rmsNormEps }

    /// One entry per layer, either "full" or "shared". Absent, the pattern is derived:
    /// the first two layers and then every fourth run their own indexer, and the rest
    /// reuse the selection the preceding full layer made.
    public var indexerTypes: [String] {
        if let declared = _indexerTypes, !declared.isEmpty { return declared }
        return (0 ..< numHiddenLayers).map { $0 < 2 || $0 % 4 == 1 ? "full" : "shared" }
    }

    enum CodingKeys: String, CodingKey {
        case _indexerTypes = "indexer_types"
        case _hcMultiplier = "hc_mult"
        case _hcEps = "hc_eps"
        case _hcMagnitude = "hc_magnitude"
        case _eosTokenId = "eos_token_id"
    }

    public init(from decoder: Swift.Decoder) throws {
        deepseekConfiguration = try DeepseekV32Configuration(from: decoder)
        let container = try decoder.container(keyedBy: CodingKeys.self)
        _indexerTypes = try container.decodeIfPresent([String].self, forKey: ._indexerTypes)
        _hcMultiplier = try container.decodeIfPresent(Int.self, forKey: ._hcMultiplier)
        _hcEps = try container.decodeIfPresent(Float.self, forKey: ._hcEps)
        _hcMagnitude = try container.decodeIfPresent(Float.self, forKey: ._hcMagnitude)
        _eosTokenId = try container.decodeIfPresent(IntOrIntArray.self, forKey: ._eosTokenId)
        try validate()
    }

    /// One entry per layer, and only the two kinds the stack implements.
    public func validate() throws {
        guard indexerTypes.count == numHiddenLayers else {
            throw HyV4ConfigurationError.indexerTypeCount(
                indexerTypes.count, numHiddenLayers)
        }
        let unsupported = Set(indexerTypes).subtracting(["full", "shared"])
        guard unsupported.isEmpty else {
            throw HyV4ConfigurationError.unsupportedIndexerTypes(unsupported.sorted())
        }
    }
}

public enum HyV4ConfigurationError: LocalizedError {
    case indexerTypeCount(Int, Int)
    case unsupportedIndexerTypes([String])

    public var errorDescription: String? {
        switch self {
        case .indexerTypeCount(let given, let layers):
            "indexer_types must have one entry per hidden layer, got \(given) for \(layers)"
        case .unsupportedIndexerTypes(let kinds):
            "unsupported hy_v4 indexer types: \(kinds.joined(separator: ", "))"
        }
    }
}

extension HyV4Model: LoRAModel {
    public var loraLayers: [Module] { model.layers }
}
