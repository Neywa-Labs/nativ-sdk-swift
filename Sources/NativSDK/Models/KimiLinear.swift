import Foundation
import MLX
import MLXNN

/// A depthwise convolution over the last few positions, carrying its tail forward.
class KimiShortConv1d: Module {

    let kernelSize: Int

    @ModuleInfo(key: "conv") var conv: Conv1d

    init(channels: Int, kernelSize: Int) {
        self.kernelSize = kernelSize
        _conv.wrappedValue = Conv1d(
            inputChannels: channels, outputChannels: channels, kernelSize: kernelSize,
            padding: 0, groups: channels, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, state: MLXArray?, mask: MLXArray?) -> (
        MLXArray, MLXArray
    ) {
        var input = x
        if let mask {
            input = MLX.where(mask[.ellipsis, .newAxis], input, MLXArray(0))
        }

        let tail =
            state
            ?? MLXArray.zeros([x.dim(0), kernelSize - 1, x.dim(-1)], dtype: x.dtype)
        let history = concatenated([tail, input], axis: 1)
        let out = silu(conv(history))
        return (out, contiguous(history[0..., (1 - kernelSize)..., 0...]))
    }
}

/// The two shapes the `self_attn` slot can take.
///
/// One property, not two with the same key: two `@ModuleInfo` sharing a key publish only
/// the first, and the other kind's weights then load from nothing.
protocol KimiAttentionLayer: Module {
    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray
}

class KimiDeltaAttention: Module, KimiAttentionLayer {

    let heads: Int
    let headDim: Int
    let convKernel: Int
    let projectionDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear

    @ModuleInfo(key: "q_conv") var qConv: KimiShortConv1d
    @ModuleInfo(key: "k_conv") var kConv: KimiShortConv1d
    @ModuleInfo(key: "v_conv") var vConv: KimiShortConv1d

    @ModuleInfo(key: "f_a_proj") var fAProj: Linear
    @ModuleInfo(key: "f_b_proj") var fBProj: Linear
    @ModuleInfo(key: "b_proj") var bProj: Linear
    @ModuleInfo(key: "g_a_proj") var gAProj: Linear
    @ModuleInfo(key: "g_b_proj") var gBProj: Linear

    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray

    @ModuleInfo(key: "o_norm") var oNorm: RMSNorm
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ args: KimiLinearConfiguration) {
        self.heads = args.linearAttention.heads
        self.headDim = args.linearAttention.headDimensions
        self.convKernel = args.linearAttention.shortConvolutionKernel
        self.projectionDim = heads * headDim
        self.scale = pow(Float(headDim), -0.5)

        let hidden = args.hiddenSize
        _qProj.wrappedValue = Linear(hidden, projectionDim, bias: false)
        _kProj.wrappedValue = Linear(hidden, projectionDim, bias: false)
        _vProj.wrappedValue = Linear(hidden, projectionDim, bias: false)

        _qConv.wrappedValue = KimiShortConv1d(
            channels: projectionDim, kernelSize: convKernel)
        _kConv.wrappedValue = KimiShortConv1d(
            channels: projectionDim, kernelSize: convKernel)
        _vConv.wrappedValue = KimiShortConv1d(
            channels: projectionDim, kernelSize: convKernel)

        _fAProj.wrappedValue = Linear(hidden, headDim, bias: false)
        _fBProj.wrappedValue = Linear(headDim, projectionDim, bias: false)
        _bProj.wrappedValue = Linear(hidden, heads, bias: false)
        _gAProj.wrappedValue = Linear(hidden, headDim, bias: false)
        _gBProj.wrappedValue = Linear(headDim, projectionDim, bias: false)

        _aLog.wrappedValue = log(
            MLXRandom.uniform(low: 1.0, high: 16.0, [heads]))
        _dtBias.wrappedValue = MLXArray.zeros([projectionDim])

        _oNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
        _oProj.wrappedValue = Linear(projectionDim, hidden, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        let (B, T) = (x.dim(0), x.dim(1))
        let slots = cache as? ArraysCache

        let (qOut, qState) = qConv(qProj(x), state: slots?[0], mask: mask)
        let (kOut, kState) = kConv(kProj(x), state: slots?[1], mask: mask)
        let (vOut, vState) = vConv(vProj(x), state: slots?[2], mask: mask)
        slots?[0] = qState
        slots?[1] = kState
        slots?[2] = vState

        var q = qOut.reshaped(B, T, heads, headDim)
        var k = kOut.reshaped(B, T, heads, headDim)
        let v = vOut.reshaped(B, T, heads, headDim)

        q = (scale * scale) * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        k = scale * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

        let a = fBProj(fAProj(x)).reshaped(B, T, heads, headDim)
        let b = bProj(x).reshaped(B, T, heads)

        let (recurred, state) = gatedDeltaUpdate(
            q: q, k: k, v: v, a: a, b: b,
            aLog: aLog.reshaped(heads, 1),
            dtBias: dtBias.reshaped(heads, headDim),
            state: slots?[3], mask: mask)
        slots?[3] = state
        slots?.advance(T)

        let gate = gBProj(gAProj(x)).reshaped(B, T, heads, headDim)
        let gated = oNorm(recurred.reshaped(B, T, heads, headDim)) * sigmoid(gate)
        return oProj(gated.reshaped(B, T, -1))
    }
}

/// Latent MLA with *no* rotary: the positional halves are attended to unrotated.
class KimiMLAAttention: Module, KimiAttentionLayer {

    let heads: Int
    let qkNopeHeadDim: Int
    let qkRopeHeadDim: Int
    let queryHeadDim: Int
    let vHeadDim: Int
    let kvLoraRank: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "kv_a_proj_with_mqa") var kvAProj: Linear
    @ModuleInfo(key: "kv_a_layernorm") var kvALayerNorm: RMSNorm
    @ModuleInfo(key: "embed_q") var embedQ: Module
    @ModuleInfo(key: "unembed_out") var unembedOut: Module
    @ModuleInfo(key: "o_proj") var oProj: Linear

    init(_ args: KimiLinearConfiguration) {
        self.heads = args.attentionHeads
        self.qkNopeHeadDim = args.qkNopeHeadDim ?? args.headDimensions
        self.qkRopeHeadDim = args.qkRopeHeadDim ?? 0
        self.queryHeadDim = qkNopeHeadDim + qkRopeHeadDim
        self.vHeadDim = args.vHeadDim ?? args.headDimensions
        self.kvLoraRank = args.kvLoraRank
        self.scale = pow(Float(queryHeadDim), -0.5)

        _qProj.wrappedValue = Linear(
            args.hiddenSize, heads * queryHeadDim, bias: false)
        _kvAProj.wrappedValue = Linear(
            args.hiddenSize, args.kvLoraRank + qkRopeHeadDim, bias: false)
        _kvALayerNorm.wrappedValue = RMSNorm(
            dimensions: args.kvLoraRank, eps: args.rmsNormEps)
        _embedQ.wrappedValue = MultiLinear(
            inputDimensions: qkNopeHeadDim, outputDimensions: args.kvLoraRank,
            numHeads: heads)
        _unembedOut.wrappedValue = MultiLinear(
            inputDimensions: args.kvLoraRank, outputDimensions: vHeadDim,
            numHeads: heads)
        _oProj.wrappedValue = Linear(heads * vHeadDim, args.hiddenSize, bias: false)
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

        let q = qProj(x).reshaped(B, L, heads, queryHeadDim).transposed(0, 2, 1, 3)
        let qSplit = split(q, indices: [qkNopeHeadDim], axis: -1)
        var qNope = qSplit[0]
        let qPE = qSplit[1]

        let compressed = kvAProj(x)
        let kvSplit = split(compressed, indices: [kvLoraRank], axis: -1)
        var kPE = kvSplit[1].reshaped(B, L, 1, qkRopeHeadDim).transposed(0, 2, 1, 3)
        var kvLatent = kvALayerNorm(kvSplit[0]).expandedDimensions(axis: 1)

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

class KimiSparseMoE: Module, UnaryLayer {

    let topK: Int
    let groups: Int
    let topGroups: Int
    let routedScalingFactor: Float
    let renormalize: Bool
    let scoring: ExpertScoring

    @ModuleInfo(key: "gate") var gate: Linear
    @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU
    @ModuleInfo(key: "shared_experts") var sharedExperts: SwiGLUMLP?

    @ParameterInfo(key: "e_score_correction_bias") var correctionBias: MLXArray

    init(_ args: KimiLinearConfiguration) {
        self.topK = args.expertsPerToken
        self.groups = args.expertGroups
        self.topGroups = args.topGroups
        self.routedScalingFactor = args.routedScalingFactor
        self.renormalize = args.renormalize
        self.scoring = args.routerActivation == "softmax" ? .softmax : .sigmoid

        _gate.wrappedValue = Linear(args.hiddenSize, args.experts, bias: false)
        _switchMLP.wrappedValue = SwitchGLU(
            inputDims: args.hiddenSize, hiddenDims: args.moeIntermediateSize,
            numExperts: args.experts)
        _correctionBias.wrappedValue = MLXArray.zeros([args.experts])

        if args.sharedExperts > 0 {
            _sharedExperts.wrappedValue = SwiGLUMLP(
                inputDimensions: args.hiddenSize,
                hiddenDimensions: args.moeIntermediateSize * args.sharedExperts)
        }
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (indices, scores) = groupedExpertSelection(
            gates: gate(x),
            correctionBias: correctionBias,
            topK: topK,
            groups: groups,
            topGroups: topGroups,
            routedScalingFactor: routedScalingFactor,
            normalizeTopK: renormalize,
            scoring: scoring,
            inFloat32: false)

        var out = (switchMLP(x, indices) * scores[.ellipsis, .newAxis]).sum(axis: -2)
        if let sharedExperts {
            out = out + sharedExperts(x)
        }
        return out
    }
}

class KimiLinearBlock: Module {

    @ModuleInfo(key: "self_attn") var attention: KimiAttentionLayer
    @ModuleInfo(key: "mlp") var mlp: UnaryLayer
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    let isLinear: Bool

    init(_ args: KimiLinearConfiguration, layerIndex: Int) {
        self.isLinear = args.linearAttention.kdaLayers.contains(layerIndex + 1)

        _attention.wrappedValue =
            isLinear ? KimiDeltaAttention(args) : KimiMLAAttention(args)

        let isMixture =
            args.experts > 0 && layerIndex >= args.firstDenseLayers
            && layerIndex % args.moeLayerFrequency == 0
        _mlp.wrappedValue =
            isMixture
            ? KimiSparseMoE(args)
            : SwiGLUMLP(
                inputDimensions: args.hiddenSize,
                hiddenDimensions: args.intermediateSize)

        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray?, cache: KVCache?) -> MLXArray {
        let h = x + attention(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

public class KimiLinearModelInner: Module {

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding

    let layers: [KimiLinearBlock]
    let norm: RMSNorm

    let recurrentIndex: Int
    let attentionIndex: Int

    public init(_ args: KimiLinearConfiguration) {
        precondition(args.vocabularySize > 0)
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
        let layers = (0 ..< args.hiddenLayers).map {
            KimiLinearBlock(args, layerIndex: $0)
        }
        self.layers = layers
        self.norm = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

        self.recurrentIndex = (args.linearAttention.kdaLayers.min() ?? 1) - 1
        self.attentionIndex = layers.firstIndex { !$0.isLinear } ?? 0
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]? = nil) -> MLXArray {
        var h = embedTokens(inputs)

        let recurrentMask =
            (cache?[min(recurrentIndex, (cache?.count ?? 1) - 1)] as? ArraysCache)?
            .makeMask(N: h.dim(1))
        let attentionCache = cache?[min(attentionIndex, (cache?.count ?? 1) - 1)]
        let attentionMask: MLXArray? = createAttentionMask(
            h: h, cache: attentionCache.map { [$0] })

        for (i, layer) in layers.enumerated() {
            h = layer(
                h, mask: layer.isLinear ? recurrentMask : attentionMask, cache: cache?[i])
        }

        return norm(h)
    }
}

public class KimiLinearModel: Module, LLMModel {

    public let vocabularySize: Int

    public let model: KimiLinearModelInner

    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    let configuration: KimiLinearConfiguration

    public init(_ args: KimiLinearConfiguration) {
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.model = KimiLinearModelInner(args)
        if !args.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(args.hiddenSize, args.vocabularySize, bias: false)
        }
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let out = model(inputs, cache: cache)
        if let lmHead {
            return lmHead(out)
        }
        return model.embedTokens.asLinear(out)
    }

    /// Four array slots for a delta layer -- three convolution tails and the recurrent
    /// state -- and an ordinary key/value cache for a latent one.
    public func newCache(configuration cacheConfiguration: KVCacheConfiguration?) throws
        -> [KVCache]
    {
        model.layers.map { layer -> KVCache in
            layer.isLinear ? ArraysCache(size: 4) : MaterializedKVCache(KVCacheSimple())
        }
    }

    /// The checkpoint spells the mixture `block_sparse_moe` with `w1`/`w2`/`w3` experts,
    /// the convolutions `*_conv1d`, and keeps `kv_b_proj` fused.
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized = weights.filter { !$0.key.hasPrefix("model.mtp") }
        if configuration.tieWordEmbeddings {
            sanitized.removeValue(forKey: "lm_head.weight")
        }

        for (index, layer) in model.layers.enumerated() {
            let prefix = "model.layers.\(index)"

            if layer.mlp is KimiSparseMoE {
                let source = "\(prefix).block_sparse_moe"
                for (from, to) in [
                    ("w1", "gate_proj"), ("w2", "down_proj"), ("w3", "up_proj"),
                ] {
                    let keys = (0 ..< configuration.experts).map {
                        "\(source).experts.\($0).\(from).weight"
                    }
                    guard let first = keys.first, sanitized[first] != nil else { continue }
                    let stacked = keys.compactMap { sanitized.removeValue(forKey: $0) }
                    guard stacked.count == configuration.experts else { continue }
                    sanitized["\(prefix).mlp.switch_mlp.\(to).weight"] =
                        MLX.stacked(stacked)
                }

                for name in ["gate_proj", "up_proj", "down_proj"] {
                    if let value = sanitized.removeValue(
                        forKey: "\(source).shared_experts.\(name).weight")
                    {
                        sanitized["\(prefix).mlp.shared_experts.\(name).weight"] = value
                    }
                }
                if let value = sanitized.removeValue(forKey: "\(source).gate.weight") {
                    sanitized["\(prefix).mlp.gate.weight"] = value
                }
                if let value = sanitized.removeValue(
                    forKey: "\(source).gate.e_score_correction_bias")
                {
                    sanitized["\(prefix).mlp.e_score_correction_bias"] = value
                }
            }

            let attention = "\(prefix).self_attn"
            if layer.isLinear {
                for (from, to) in [
                    ("q_conv1d", "q_conv"), ("k_conv1d", "k_conv"),
                    ("v_conv1d", "v_conv"),
                ] {
                    if var value = sanitized.removeValue(
                        forKey: "\(attention).\(from).weight")
                    {
                        if value.ndim == 3 {
                            value = value.movedAxis(source: 2, destination: 1)
                        }
                        sanitized["\(attention).\(to).conv.weight"] = value
                    }
                }
                for name in ["dt_bias", "A_log"] {
                    if let value = sanitized["\(attention).\(name)"], value.ndim > 1 {
                        sanitized["\(attention).\(name)"] = value.reshaped(-1)
                    }
                }
            } else {
                splitFusedKeyValue(prefix: attention, weights: &sanitized)
            }
        }

        return sanitized
    }

    private func splitFusedKeyValue(prefix: String, weights: inout [String: MLXArray]) {
        let qkNope = configuration.qkNopeHeadDim ?? configuration.headDimensions
        let vHead = configuration.vHeadDim ?? configuration.headDimensions
        let headDim = qkNope + vHead

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
        var embedQ = contiguous(reshaped[0..., ..<qkNope, 0...].swappedAxes(-1, -2))
        var unembedOut = contiguous(reshaped[0..., qkNope..., 0...])

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

public struct KimiLinearConfiguration: Decodable, Sendable {

    /// The `linear_attn_config` sub-dictionary.
    public struct LinearAttentionConfiguration: Decodable, Sendable {
        var heads: Int
        var headDimensions: Int
        var kdaLayers: [Int]
        var shortConvolutionKernel: Int

        enum CodingKeys: String, CodingKey {
            case heads = "num_heads"
            case headDimensions = "head_dim"
            case kdaLayers = "kda_layers"
            case shortConvolutionKernel = "short_conv_kernel_size"
        }

        public init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            heads = try c.decode(Int.self, forKey: .heads)
            headDimensions = try c.decode(Int.self, forKey: .headDimensions)
            kdaLayers = try c.decode([Int].self, forKey: .kdaLayers)
            shortConvolutionKernel =
                try c.decodeIfPresent(Int.self, forKey: .shortConvolutionKernel) ?? 4
        }
    }

    var modelType: String
    var vocabularySize: Int
    var hiddenSize: Int
    var hiddenLayers: Int
    var attentionHeads: Int
    var kvHeads: Int
    var intermediateSize: Int
    var headDimensions: Int
    var rmsNormEps: Float
    var linearAttention: LinearAttentionConfiguration
    var experts: Int
    var moeIntermediateSize: Int
    var kvLoraRank: Int
    var tieWordEmbeddings: Bool
    var qkNopeHeadDim: Int?
    var qkRopeHeadDim: Int?
    var vHeadDim: Int?
    var expertsPerToken: Int
    var sharedExperts: Int
    var routerActivation: String
    var renormalize: Bool
    var routedScalingFactor: Float
    var firstDenseLayers: Int
    var moeLayerFrequency: Int
    var expertGroups: Int
    var topGroups: Int

    enum CodingKeys: String, CodingKey {
        case modelType = "model_type"
        case vocabularySize = "vocab_size"
        case hiddenSize = "hidden_size"
        case hiddenLayers = "num_hidden_layers"
        case attentionHeads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case intermediateSize = "intermediate_size"
        case headDimensions = "head_dim"
        case rmsNormEps = "rms_norm_eps"
        case linearAttention = "linear_attn_config"
        case experts = "num_experts"
        case moeIntermediateSize = "moe_intermediate_size"
        case kvLoraRank = "kv_lora_rank"
        case tieWordEmbeddings = "tie_word_embeddings"
        case qkNopeHeadDim = "qk_nope_head_dim"
        case qkRopeHeadDim = "qk_rope_head_dim"
        case vHeadDim = "v_head_dim"
        case expertsPerToken = "num_experts_per_token"
        case sharedExperts = "num_shared_experts"
        case routerActivation = "moe_router_activation_func"
        case renormalize = "moe_renormalize"
        case routedScalingFactor = "routed_scaling_factor"
        case firstDenseLayers = "first_k_dense_replace"
        case moeLayerFrequency = "moe_layer_freq"
        case expertGroups = "num_expert_group"
        case topGroups = "topk_group"
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        modelType = try c.decodeIfPresent(String.self, forKey: .modelType) ?? "kimi_linear"
        vocabularySize = try c.decode(Int.self, forKey: .vocabularySize)
        hiddenSize = try c.decode(Int.self, forKey: .hiddenSize)
        hiddenLayers = try c.decode(Int.self, forKey: .hiddenLayers)
        attentionHeads = try c.decode(Int.self, forKey: .attentionHeads)
        kvHeads = try c.decode(Int.self, forKey: .kvHeads)
        intermediateSize = try c.decode(Int.self, forKey: .intermediateSize)
        headDimensions = try c.decode(Int.self, forKey: .headDimensions)
        rmsNormEps = try c.decode(Float.self, forKey: .rmsNormEps)
        linearAttention = try c.decode(
            LinearAttentionConfiguration.self, forKey: .linearAttention)
        experts = try c.decode(Int.self, forKey: .experts)
        moeIntermediateSize = try c.decode(Int.self, forKey: .moeIntermediateSize)
        kvLoraRank = try c.decode(Int.self, forKey: .kvLoraRank)
        tieWordEmbeddings =
            try c.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        qkNopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkNopeHeadDim)
        qkRopeHeadDim = try c.decodeIfPresent(Int.self, forKey: .qkRopeHeadDim)
        vHeadDim = try c.decodeIfPresent(Int.self, forKey: .vHeadDim)
        expertsPerToken = try c.decodeIfPresent(Int.self, forKey: .expertsPerToken) ?? 1
        sharedExperts = try c.decodeIfPresent(Int.self, forKey: .sharedExperts) ?? 0
        routerActivation =
            try c.decodeIfPresent(String.self, forKey: .routerActivation) ?? "sigmoid"
        renormalize = try c.decodeIfPresent(Bool.self, forKey: .renormalize) ?? true
        routedScalingFactor =
            try c.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 1
        firstDenseLayers = try c.decodeIfPresent(Int.self, forKey: .firstDenseLayers) ?? 0
        moeLayerFrequency = try c.decodeIfPresent(Int.self, forKey: .moeLayerFrequency) ?? 1
        expertGroups = try c.decodeIfPresent(Int.self, forKey: .expertGroups) ?? 1
        topGroups = try c.decodeIfPresent(Int.self, forKey: .topGroups) ?? 1
    }
}

extension KimiLinearModel: LoRAModel {
    public var loraLayers: [Module] {
        model.layers
    }
}
