import Foundation
import MLX
import MLXNN

class LongcatFlashMLA: Module {

    let heads: Int
    let qkRopeHeadDim: Int
    let qkNopeHeadDim: Int
    let kvLoraRank: Int
    let qLoraRank: Int?
    let vHeadDim: Int
    let qkHeadDim: Int
    let scale: Float

    let queryLoraScale: Float?
    let keyValueLoraScale: Float?

    @ModuleInfo(key: "q_proj") var qProj: Linear?
    @ModuleInfo(key: "q_a_proj") var qAProj: Linear?
    @ModuleInfo(key: "q_a_layernorm") var qALayerNorm: RMSNorm?
    @ModuleInfo(key: "q_b_proj") var qBProj: Linear?

    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProj: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm

    @ModuleInfo(key: "embed_q") var embedQ: Module
    @ModuleInfo(key: "unembed_out") var unembedOut: Module

    @ModuleInfo(key: "o_proj") var oProj: Linear

    let rope: RoPELayer

    init(_ args: LongcatFlashConfiguration) {
        self.heads = args.attentionHeads
        self.qkRopeHeadDim = args.qkRopeHeadDim
        self.qkNopeHeadDim = args.qkNopeHeadDim
        self.kvLoraRank = args.kvLoraRank
        self.qLoraRank = args.qLoraRank
        self.vHeadDim = args.vHeadDim
        self.qkHeadDim = args.qkNopeHeadDim + args.qkRopeHeadDim

        var scale = pow(Float(qkHeadDim), -0.5)
        if let scaling = args.ropeScaling,
            let mscaleAllDim = scaling["mscale_all_dim"]?.asFloat(), mscaleAllDim != 0,
            let factor = scaling["factor"]?.asFloat(), factor > 1
        {
            let adjustment = 0.1 * mscaleAllDim * log(factor) + 1
            scale = scale * adjustment * adjustment
        }
        self.scale = scale

        if let qLoraRank = args.qLoraRank {
            _qAProj.wrappedValue = Linear(
                args.hiddenSize, qLoraRank, bias: args.attentionBias)
            _qALayerNorm.wrappedValue = RMSNorm(dimensions: qLoraRank)
            _qBProj.wrappedValue = Linear(qLoraRank, heads * qkHeadDim, bias: false)
        } else {
            _qProj.wrappedValue = Linear(
                args.hiddenSize, heads * qkHeadDim, bias: false)
        }

        _kvAProj.wrappedValue = Linear(
            args.hiddenSize, args.kvLoraRank + args.qkRopeHeadDim,
            bias: args.attentionBias)
        _kvALayerNorm.wrappedValue = RMSNorm(dimensions: args.kvLoraRank)

        _embedQ.wrappedValue = MultiLinear(
            inputDimensions: args.qkNopeHeadDim, outputDimensions: args.kvLoraRank,
            numHeads: heads)
        _unembedOut.wrappedValue = MultiLinear(
            inputDimensions: args.kvLoraRank, outputDimensions: args.vHeadDim,
            numHeads: heads)

        _oProj.wrappedValue = Linear(
            heads * args.vHeadDim, args.hiddenSize, bias: args.attentionBias)

        self.queryLoraScale =
            args.scaleQueryLora && args.qLoraRank != nil
            ? sqrt(Float(args.hiddenSize) / Float(args.qLoraRank!)) : nil
        self.keyValueLoraScale =
            args.scaleKeyValueLora
            ? sqrt(Float(args.hiddenSize) / Float(args.kvLoraRank)) : nil

        self.rope = initializeRope(
            dims: args.qkRopeHeadDim, base: args.ropeTheta, traditional: true,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings)

        super.init()
    }

    private func multiLinear(_ module: Module, _ x: MLXArray, transpose: Bool = true)
        -> MLXArray
    {
        if let linear = module as? MultiLinear {
            return linear(x, transpose: transpose)
        }
        if let quantized = module as? QuantizedMultiLinear {
            return quantized(x, transpose: transpose)
        }
        fatalError("embed_q and unembed_out must be MultiLinear or QuantizedMultiLinear")
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))

        var q: MLXArray
        if let qBProj, let qALayerNorm, let qAProj {
            q = qBProj(qALayerNorm(qAProj(x)))
        } else {
            q = qProj!(x)
        }
        q = q.reshaped(B, L, heads, qkHeadDim).transposed(0, 2, 1, 3)
        if let queryLoraScale {
            q = q * queryLoraScale
        }

        let qSplit = split(q, indices: [qkNopeHeadDim], axis: -1)
        var qNope = qSplit[0]
        var qPE = qSplit[1]

        let compressed = kvAProj(x)
        let kvSplit = split(compressed, indices: [kvLoraRank], axis: -1)
        var kPE = kvSplit[1].reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)
        var kvLatent = kvALayerNorm(kvSplit[0])
        if let keyValueLoraScale {
            kvLatent = kvLatent * keyValueLoraScale
        }

        let offset = cache?.offset ?? 0
        qPE = rope(qPE, offset: offset)
        kPE = rope(kPE, offset: offset)

        kvLatent = kvLatent.expandedDimensions(axis: 1)
        if let cache {
            (kvLatent, kPE) = cache.update(keys: kvLatent, values: kPE)
        }

        var peScores = (qPE * scale).matmul(kPE.swappedAxes(-1, -2))
        if let mask {
            peScores = MLX.where(
                mask, peScores,
                MLXArray(-Float.greatestFiniteMagnitude, dtype: peScores.dtype))
        }

        let absorbed =
            L == 1
            || L
                <= maxAbsorbedQueries(
                    kvLoraRank: kvLoraRank, qkNopeHeadDim: qkNopeHeadDim,
                    vHeadDim: vHeadDim, cacheLength: latentLength(kvLatent))

        let keys: MLXArray
        let values: MLXArray
        if absorbed {
            qNope = multiLinear(embedQ, qNope)
            keys = kvLatent
            values = kvLatent
        } else {
            keys = multiLinear(embedQ, kvLatent, transpose: false)
            values = multiLinear(unembedOut, kvLatent)
        }

        var output = MLXFast.scaledDotProductAttention(
            queries: qNope, keys: keys, values: values, scale: scale,
            mask: .array(peScores))
        if absorbed {
            output = multiLinear(unembedOut, output)
        }

        return oProj(output.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

class LongcatFlashRouter: Module {

    let topK: Int
    let routedScalingFactor: Float
    let normalizeTopK: Bool

    @ModuleInfo(key: "classifier") var classifier: Linear
    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    init(_ args: LongcatFlashConfiguration) {
        self.topK = args.moeTopK
        self.routedScalingFactor = args.routedScalingFactor
        self.normalizeTopK = args.normalizeTopKProbabilities

        let outputs = args.routedExperts + args.zeroExperts
        _classifier.wrappedValue = Linear(
            args.hiddenSize, outputs, bias: args.routerBias)
        _correctionBias.wrappedValue = MLXArray.zeros([outputs])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> (indices: MLXArray, scores: MLXArray) {
        groupedExpertSelection(
            gates: classifier(x),
            correctionBias: correctionBias,
            topK: topK,
            groups: 1,
            topGroups: 1,
            routedScalingFactor: routedScalingFactor,
            normalizeTopK: normalizeTopK,
            scoring: .softmax)
    }
}

class LongcatFlashMoE: Module, UnaryLayer {

    let routedExperts: Int

    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "router") var router: LongcatFlashRouter

    init(_ args: LongcatFlashConfiguration) {
        precondition(
            args.zeroExpertType == "identity",
            "only identity zero-experts are implemented, got \(args.zeroExpertType)")
        self.routedExperts = args.routedExperts
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.expertFeedForwardSize,
            numExperts: args.routedExperts)
        _router.wrappedValue = LongcatFlashRouter(args)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = router(x)

        let isIdentity = indices .>= routedExperts
        let routedIndices = MLX.where(isIdentity, MLXArray(0), indices)
        let routedScores = MLX.where(isIdentity, MLXArray(Float(0)), scores)

        let routed = switchMLP(x, routedIndices)
        let combined = (routed * routedScores[.ellipsis, .newAxis]).sum(axis: -2)

        let identityWeight = MLX.where(isIdentity, scores, MLXArray(Float(0)))
            .sum(axis: -1, keepDims: true)

        return combined + x * identityWeight
    }
}

class LongcatFlashBlock: Module {

    @ModuleInfo(key: "mlp") var mixture: LongcatFlashMoE
    @ModuleInfo(key: "self_attn") var attentions: [LongcatFlashMLA]
    @ModuleInfo(key: "mlps") var feedForwards: [SwiGLUMLP]
    @ModuleInfo(key: "input_layernorm") var inputLayerNorms: [RMSNorm]
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorms: [RMSNorm]

    init(_ args: LongcatFlashConfiguration) {
        _mixture.wrappedValue = LongcatFlashMoE(args)
        _attentions.wrappedValue = (0 ..< 2).map { _ in LongcatFlashMLA(args) }
        _feedForwards.wrappedValue = (0 ..< 2).map { _ in
            SwiGLUMLP(
                inputDimensions: args.hiddenSize,
                hiddenDimensions: args.feedForwardSize)
        }
        _inputLayerNorms.wrappedValue = (0 ..< 2).map { _ in
            RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }
        _postAttentionLayerNorms.wrappedValue = (0 ..< 2).map { _ in
            RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: CacheList?) -> MLXArray {
        var h = x
        var shortcut: MLXArray? = nil

        for i in 0 ..< 2 {
            var residual = h
            h = residual + attentions[i](inputLayerNorms[i](h), mask: mask, cache: cache?[i])

            residual = h
            let normalized = postAttentionLayerNorms[i](h)

            if i == 0 {
                shortcut = mixture(normalized)
            }

            h = residual + feedForwards[i](normalized)

            if i == 1, let shortcut {
                h = h + shortcut
            }
        }

        return h
    }
}

public class LongcatFlashModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [LongcatFlashBlock]
    let norm: RMSNorm

    public init(_ args: LongcatFlashConfiguration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        self.layers = (0 ..< args.layerCount).map { _ in LongcatFlashBlock(args) }
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let first = (cache?.first as? CacheList)?[0]
        let mask: MLXArray? = createAttentionMask(h: h, cache: first.map { [$0] })

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: mask, cache: cache?[i] as? CacheList)
        }

        return norm(h)
    }
}

public class LongcatFlashModel: Module, LLMModel {

    public let vocabularySize: Int

    public let model: LongcatFlashModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let configuration: LongcatFlashConfiguration

    public init(_ args: LongcatFlashConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.model = LongcatFlashModelInner(args)
        _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        lmHead(model(inputs, cache: cache))
    }

    /// Two caches per layer, one per sub-layer.
    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        model.layers.map { _ in
            CacheList(MaterializedKVCache(KVCacheSimple()), MaterializedKVCache(KVCacheSimple()))
        }
    }

    /// Stacks the per-expert weights and splits each sub-layer's fused `kv_b_proj` into the
    /// absorbed `embed_q` and `unembed_out`, requantizing when the source was quantized.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        Self.sanitize(weights: weights, configuration: configuration)
    }

    /// Static so the n-gram variant can reuse it without constructing a second model --
    /// which would allocate every expert weight over again.
    static func sanitize(
        weights: [String: MLXArray], configuration: LongcatFlashConfiguration
    ) -> [String: MLXArray] {
        var sanitized = weights.filter { !$0.key.hasPrefix("model.mtp") }

        for layer in 0 ..< configuration.layerCount {
            let prefix = "model.layers.\(layer)"
            for projection in ["gate_proj", "down_proj", "up_proj"] {
                for kind in ["weight", "scales", "biases"] {
                    let keys = (0 ..< configuration.routedExperts).map {
                        "\(prefix).mlp.experts.\($0).\(projection).\(kind)"
                    }
                    guard let first = keys.first, sanitized[first] != nil else { continue }
                    let stacked = keys.compactMap { sanitized.removeValue(forKey: $0) }
                    guard stacked.count == configuration.routedExperts else { continue }
                    sanitized["\(prefix).mlp.switch_mlp.\(projection).\(kind)"] =
                        MLX.stacked(stacked)
                }
            }

            for sublayer in 0 ..< 2 {
                splitFusedKeyValue(
                    prefix: "\(prefix).self_attn.\(sublayer)", weights: &sanitized,
                    configuration: configuration)
            }
        }

        return sanitized
    }

    private static func splitFusedKeyValue(
        prefix: String, weights: inout [String: MLXArray],
        configuration: LongcatFlashConfiguration
    ) {
        let headDim = configuration.qkNopeHeadDim + configuration.vHeadDim
        guard var fused = weights.removeValue(forKey: "\(prefix).kv_b_proj.weight")
        else { return }

        var bits: Int?
        var groupSize: Int?
        if let scales = weights.removeValue(forKey: "\(prefix).kv_b_proj.scales") {
            let biases = weights.removeValue(forKey: "\(prefix).kv_b_proj.biases")
            bits = (fused.dim(-1) * 32) / configuration.kvLoraRank
            groupSize = configuration.kvLoraRank / scales.dim(-1)
            fused = dequantized(
                fused, scales: scales, biases: biases, groupSize: groupSize!, bits: bits!)
        }

        let reshaped = fused.reshaped(configuration.attentionHeads, headDim, -1)
        var embedQ = contiguous(
            reshaped[0..., ..<configuration.qkNopeHeadDim, 0...].swappedAxes(-1, -2))
        var unembedOut = contiguous(reshaped[0..., configuration.qkNopeHeadDim..., 0...])

        if let bits, let groupSize {
            let (qWeight, qScales, qBiases) = MLX.quantized(
                embedQ, groupSize: groupSize, bits: bits)
            let (vWeight, vScales, vBiases) = MLX.quantized(
                unembedOut, groupSize: groupSize, bits: bits)
            weights["\(prefix).embed_q.scales"] = qScales
            weights["\(prefix).embed_q.biases"] = qBiases
            weights["\(prefix).unembed_out.scales"] = vScales
            weights["\(prefix).unembed_out.biases"] = vBiases
            embedQ = qWeight
            unembedOut = vWeight
        }
        weights["\(prefix).embed_q.weight"] = embedQ
        weights["\(prefix).unembed_out.weight"] = unembedOut
    }
}

public struct LongcatFlashConfiguration: Decodable, Sendable {

    var modelType: String
    var attentionMethod: String
    var zeroExpertType: String
    var hiddenSize: Int
    var feedForwardSize: Int
    var moeTopK: Int
    var expertFeedForwardSize: Int
    var routedExperts: Int
    var zeroExperts: Int
    var layerCount: Int
    var vocabularySize: Int
    var maxPositionEmbeddings: Int
    var attentionHeads: Int
    var kvLoraRank: Int
    var qLoraRank: Int?
    var qkRopeHeadDim: Int
    var qkNopeHeadDim: Int
    var vHeadDim: Int
    var routedScalingFactor: Float
    var rmsNormEps: Float
    var ropeTheta: Float
    var scaleQueryLora: Bool
    var scaleKeyValueLora: Bool
    var attentionBias: Bool
    var normalizeTopKProbabilities: Bool
    var routerBias: Bool
    var ropeScaling: [String: StringOrNumber]?

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case attentionMethod = "attention_method"
        case zeroExpertType = "zero_expert_type"
        case hiddenSize = "hidden_size"
        case feedForwardSize = "ffn_hidden_size"
        case moeTopK = "moe_topk"
        case expertFeedForwardSize = "expert_ffn_hidden_size"
        case routedExperts = "n_routed_experts"
        case zeroExperts = "zero_expert_num"
        case layerCount = "num_layers"
        case vocabularySize = "vocab_size"
        case maxPositionEmbeddings = "max_position_embeddings"
        case attentionHeads = "num_attention_heads"
        case kvLoraRank = "kv_lora_rank"
        case qLoraRank = "q_lora_rank"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case vHeadDim = "v_head_dim"
        case routedScalingFactor = "routed_scaling_factor"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case scaleQueryLora = "mla_scale_q_lora"
        case scaleKeyValueLora = "mla_scale_kv_lora"
        case attentionBias = "attention_bias"
        case normalizeTopKProbabilities = "norm_topk_prob"
        case routerBias = "router_bias"
        case ropeScaling = "rope_scaling"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType =
            try c.decodeIfPresent(String.self, forKey: .modelType) ?? "longcat_flash"
        attentionMethod =
            try c.decodeIfPresent(String.self, forKey: .attentionMethod) ?? "MLA"
        zeroExpertType =
            try c.decodeIfPresent(String.self, forKey: .zeroExpertType) ?? "identity"
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        feedForwardSize = try c.decode(Int.self, forKey: .feedForwardSize)
        moeTopK = try c.decode(Int.self, forKey: .moeTopK)
        expertFeedForwardSize = try c.decode(Int.self, forKey: .expertFeedForwardSize)
        routedExperts = try c.decode(Int.self, forKey: .routedExperts)
        zeroExperts = try c.decode(Int.self, forKey: .zeroExperts)
        layerCount = try c.decode(Int.self, forKey: .layerCount)
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        maxPositionEmbeddings = try c.decode(Int.self, forKey: .maxPositionEmbeddings)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        qLoraRank = try c.decodeIfPresent(Int.self, forKey: .qLoraRank)
        qkRopeHeadDim = try c.decode(Int.self, forKey: .qkRopeHeadDim)
        qkNopeHeadDim = try c.decode(Int.self, forKey: .qkNopeHeadDim)
        vHeadDim = try c.decode(Int.self, forKey: .vHeadDim)
        routedScalingFactor = try c.decode(Float.self, forKey: .routedScalingFactor)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        ropeTheta = try c.decode(Float.self, forKey: .ropeTheta)
        scaleQueryLora = try c.decode(Bool.self, forKey: .scaleQueryLora)
        scaleKeyValueLora = try c.decode(Bool.self, forKey: .scaleKeyValueLora)
        attentionBias = try c.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        normalizeTopKProbabilities =
            try c.decodeIfPresent(Bool.self, forKey: .normalizeTopKProbabilities) ?? false
        routerBias = try c.decodeIfPresent(Bool.self, forKey: .routerBias) ?? false
        ropeScaling = try c.decodeIfPresent(
            [String: StringOrNumber].self, forKey: .ropeScaling)
    }
}

extension LongcatFlashModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
