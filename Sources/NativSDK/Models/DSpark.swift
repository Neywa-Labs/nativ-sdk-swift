import CoreFoundation
import Foundation
import MLX
import MLXNN

/// Detect target-bound drafts before dispatching their advertised backbone type.
/// These checkpoints use the target tokenizer.
func isTargetBoundDraftConfiguration(_ data: Data) throws -> Bool {
    guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
        return false
    }
    return ["qwen3_5_mtp", "dspark", "qwen3_dspark", "gemma4_dspark", "gemma4_dflash"].contains(
        root["model_type"] as? String ?? "")
        || root["dflash_config"] != nil
        || (root["architectures"] as? [String] ?? []).contains {
            $0.contains("DSpark") || $0.contains("DFlash")
        }
}

struct DSparkConfiguration: DFlashDraftConfiguration {
    let strategy: SpeculativeGenerationStatistics.Strategy
    let requiresBF16Target: Bool
    let sampleFromAnchor: Bool
    let hiddenSize: Int
    let intermediateSize: Int
    let layerCount: Int
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let normEps: Float
    let vocabularySize: Int
    let targetLayerCount: Int
    let explicitTargetLayerCount: Bool
    let targetLayers: [Int]
    let proposalCount: Int
    let defaultProposalCount: Int
    let maskToken: Int
    let ropeTheta: Float
    let traditionalRoPE: Bool
    let markovRank: Int
    let confidenceHead: Bool
    let layerTypes: [String]
    let slidingWindow: Int?
    let gemmaConfiguration: Gemma4TextConfiguration?
    let finalLogitSoftcapping: Float?

    init(data: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            (root["architectures"] as? [String] ?? []).contains(where: {
                [
                    "Lfm2DSparkDraftModel", "Qwen3DSparkModel", "DSparkDraftModel",
                    "DFlashDraftModel", "Gemma4DFlashDraftModel", "Gemma4DSparkModel",
                ].contains($0)
            })
                || ["dspark", "qwen3_dspark", "gemma4_dspark", "gemma4_dflash"].contains(
                    root["model_type"] as? String ?? "")
        else {
            throw ModelExecutionError.invalidOptions("unsupported target-bound draft architecture")
        }
        let architectures = root["architectures"] as? [String] ?? []
        let dflash = root["dflash_config"] as? [String: Any] ?? [:]
        requiresBF16Target = architectures.contains("Lfm2DSparkDraftModel")
        let gemma =
            architectures.contains("Gemma4DSparkModel")
            || root["model_type"] as? String == "gemma4_dspark"
        strategy =
            architectures.contains(where: { $0.contains("DFlash") })
                || root["model_type"] as? String == "gemma4_dflash" ? .dflash : .dspark
        if gemma {
            var config = root
            config["num_kv_shared_layers"] = root["num_kv_shared_layers"] ?? 0
            config["hidden_size_per_layer_input"] = root["hidden_size_per_layer_input"] ?? 0
            config["use_double_wide_mlp"] = root["use_double_wide_mlp"] ?? false
            config["num_global_key_value_heads"] = root["num_global_key_value_heads"] ?? 1
            config["attention_k_eq_v"] = root["attention_k_eq_v"] ?? true
            config["rope_parameters"] = (root["rope_parameters"] as? [String: Any] ?? [:])
                .filter { $0.value is [String: Any] }
            gemmaConfiguration = try JSONDecoder().decode(
                Gemma4TextConfiguration.self, from: JSONSerialization.data(withJSONObject: config))
        } else {
            gemmaConfiguration = nil
        }
        finalLogitSoftcapping =
            root["final_logit_softcapping"] is NSNull
            ? nil
            : (root["final_logit_softcapping"] as? NSNumber)?.floatValue ?? (gemma ? 30 : nil)

        sampleFromAnchor =
            strategy == .dspark
            && (dflash["sample_from_anchor"] as? Bool ?? root["sample_from_anchor"] as? Bool ?? true)
        func present(_ value: Any?) -> Bool { value != nil && !(value is NSNull) }
        // Reject malformed values rather than silently substituting a default.
        for values in [root, dflash] {
            for key in [
                "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
                "num_key_value_heads", "head_dim", "vocab_size", "num_target_layers", "block_size",
                "mask_token_id", "markov_rank", "runtime_block_size", "proposal_length",
                "sliding_window",
            ] where values[key] != nil && (key != "sliding_window" || present(values[key])) {
                guard let number = values[key] as? NSNumber,
                    CFGetTypeID(number) != CFBooleanGetTypeID(),
                    number.doubleValue == Double(number.intValue)
                else { throw ModelExecutionError.invalidOptions("DSpark requires integer \(key)") }
            }
            for key in [
                "rope_is_neox_style", "enable_confidence_head", "confidence_head_with_markov",
                "is_causal", "causal", "sample_from_anchor", "attention_bias",
                "attention_sink_bias",
                "dflash_query_causal", "dspark_bonus_anchor", "tie_word_embeddings",
                "attention_k_eq_v", "enable_moe_block", "use_double_wide_mlp", "rope_traditional",
            ] where values[key] != nil {
                guard let number = values[key] as? NSNumber,
                    CFGetTypeID(number) == CFBooleanGetTypeID()
                else {
                    throw ModelExecutionError.invalidOptions("DSpark requires boolean \(key)")
                }
            }
        }
        var values = root
        for key in [
            "markov_rank", "markov_head_type", "enable_confidence_head",
            "confidence_head_with_markov",
            "runtime_block_size", "block_size_policy", "draft_window_size",
            "dflash_initial_block_size",
        ] {
            if let value = dflash[key] { values[key] = value }
        }
        func integer(_ key: String, in values: [String: Any], default fallback: Int? = nil) throws
            -> Int
        {
            guard let value = values[key] as? Int ?? fallback else {
                throw ModelExecutionError.invalidOptions("DSpark requires integer \(key)")
            }
            return value
        }
        hiddenSize = try integer("hidden_size", in: root)
        intermediateSize = try integer("intermediate_size", in: root)
        layerCount = try integer("num_hidden_layers", in: root)
        heads = try integer("num_attention_heads", in: root)
        kvHeads = try integer("num_key_value_heads", in: root)
        headDim = try integer("head_dim", in: root)
        vocabularySize = try integer("vocab_size", in: root)
        explicitTargetLayerCount =
            dflash["num_target_layers"] != nil || root["num_target_layers"] != nil
        let capture =
            dflash["target_layer_ids"] as? [Int] ?? root["target_layer_ids"] as? [Int] ?? []
        targetLayerCount = try integer(
            "num_target_layers", in: dflash,
            default: root["num_target_layers"] as? Int ?? capture.max().map { $0 + 1 })
        proposalCount =
            try integer("block_size", in: root, default: dflash["block_size"] as? Int)
            - (strategy == .dflash || (root["dspark_bonus_anchor"] as? Bool ?? false) ? 1 : 0)
        let runtimeWidth = try integer(
            "runtime_block_size", in: values, default: min(8, proposalCount + 1))
        defaultProposalCount = runtimeWidth - 1
        maskToken = try integer(
            "mask_token_id", in: dflash,
            default: root["mask_token_id"] as? Int
                ?? (gemma || root["model_type"] as? String == "gemma4_dflash" ? 4 : nil))
        markovRank = strategy == .dspark ? try integer("markov_rank", in: values) : 0
        normEps = (root["rms_norm_eps"] as? NSNumber)?.floatValue ?? (gemma ? 1e-6 : 1e-5)
        let ropeParameters = root["rope_parameters"] as? [String: Any] ?? [:]
        ropeTheta =
            (root["rope_theta"] as? NSNumber)?.floatValue
            ?? (ropeParameters["rope_theta"] as? NSNumber)?.floatValue ?? 1e7
        traditionalRoPE = !(root["rope_is_neox_style"] as? Bool ?? true)
        confidenceHead = values["enable_confidence_head"] as? Bool ?? false
        layerTypes =
            root["layer_types"] as? [String]
            ?? Array(repeating: "full_attention", count: max(0, layerCount))
        slidingWindow =
            present(root["sliding_window"]) ? try integer("sliding_window", in: root) : nil
        targetLayers =
            dflash["target_layer_ids"] as? [Int] ?? root["target_layer_ids"] as? [Int] ?? []
        guard
            [
                hiddenSize, intermediateSize, layerCount, heads, kvHeads, headDim,
                vocabularySize, targetLayerCount, proposalCount,
            ].allSatisfy({ $0 > 0 }),
            heads % kvHeads == 0, strategy != .dspark || markovRank > 0,
            normEps.isFinite, normEps > 0, ropeTheta.isFinite, ropeTheta > 0,
            !targetLayers.isEmpty, targetLayers == Array(Set(targetLayers)).sorted(),
            targetLayers.allSatisfy({ (0 ..< targetLayerCount).contains($0) }),
            (0 ..< vocabularySize).contains(maskToken),
            (1 ... proposalCount).contains(defaultProposalCount),
            ["qwen3", "dspark", "qwen3_dspark", "gemma4_text", "gemma4_dspark", "gemma4_dflash"]
                .contains(root["model_type"] as? String ?? "qwen3"),
            dflash["projector_type"] == nil || (dflash["projector_type"] as? String) == "dspark",
            (values["block_size_policy"] as? String ?? "fixed") == "fixed",
            values["draft_window_size"] == nil, values["dflash_initial_block_size"] == nil,
            root["proposal_length"] == nil || (root["proposal_length"] as? Int) == proposalCount,
            !requiresBF16Target
                || (!present(root["quantization"]) && !present(root["quantization_config"])),
            gemma || (root["hidden_act"] as? String ?? "silu") == "silu",
            (values["markov_head_type"] as? String ?? "vanilla") == "vanilla",
            values["confidence_head_with_markov"] as? Bool ?? true,
            layerTypes.count == layerCount,
            layerTypes.allSatisfy({
                $0 == "full_attention"
                    || ((strategy == .dflash || gemma) && $0 == "sliding_attention")
            }),
            !layerTypes.contains("sliding_attention") || (slidingWindow.map { $0 > 1 } ?? false),
            !(root["is_causal"] as? Bool ?? false),
            !(dflash["causal"] as? Bool ?? false),
            !(root["dflash_query_causal"] as? Bool ?? false),
            !requiresBF16Target || sampleFromAnchor,
            !present(root["rope_scaling"]),
            gemma || ropeParameters.isEmpty
                || ((ropeParameters["rope_type"] as? String ?? "default") == "default"
                    && Set(ropeParameters.keys).isSubset(of: ["rope_type", "rope_theta"])),
            finalLogitSoftcapping.map({ $0.isFinite && $0 > 0 }) ?? true,
            dflash["swa_window_size"] == nil,
            !(root["attention_sink_bias"] as? Bool ?? false),
            !(dflash["attention_sink_bias"] as? Bool ?? false),
            !requiresBF16Target || (root["tie_word_embeddings"] as? Bool ?? true),
            !(root["attention_bias"] as? Bool ?? false)
        else {
            throw ModelExecutionError.invalidOptions(
                "unsupported DFlash/DSpark configuration; requires a supported fixed-width Qwen or Gemma4 draft backbone"
            )
        }
        if let value = root["final_logit_softcapping"], !(value is NSNull) {
            guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID()
            else {
                throw ModelExecutionError.invalidOptions(
                    "draft logit soft cap must be a positive number")
            }
        }
        if let config = gemmaConfiguration {
            guard config.numKVSharedLayers == 0, config.hiddenSizePerLayerInput == 0,
                !config.enableMoEBlock, !config.useDoubleWideMLP,
                config.globalHeadDim > 0, (config.globalKVHeads ?? kvHeads) > 0,
                heads % (config.globalKVHeads ?? kvHeads) == 0,
                (root["hidden_activation"] as? String ?? "gelu_pytorch_tanh")
                    == "gelu_pytorch_tanh",
                !(root["tie_word_embeddings"] as? Bool ?? false)
            else { throw ModelExecutionError.invalidOptions("unsupported Gemma4 DSpark backbone") }
            for (type, parameters) in config.ropeParameters {
                let ropeType = parameters["rope_type"] ?? .string("default")
                let theta = parameters["rope_theta"]?.asFloat() ?? 10000
                let factor = parameters["partial_rotary_factor"]?.asFloat() ?? 1
                guard case .string(let name) = ropeType,
                    ["default", "proportional"].contains(name),
                    ["full_attention", "sliding_attention"].contains(type),
                    theta.isFinite, theta > 0, factor.isFinite, factor > 0, factor <= 1,
                    Int(Float(type == "full_attention" ? config.globalHeadDim : headDim) * factor)
                        >= 2
                else {
                    throw ModelExecutionError.invalidOptions(
                        "unsupported Gemma4 draft rotary configuration")
                }
            }
        }
    }

    func validate(target: any DFlashTarget) throws {
        guard target.draftHiddenSize == hiddenSize,
            explicitTargetLayerCount
                ? target.draftLayerCount == targetLayerCount
                : targetLayers.allSatisfy({ $0 < target.draftLayerCount }),
            target.vocabularySize == vocabularySize,
            requiresBF16Target
                ? target is LFM2Model
                : (target is Qwen3Model || target is Qwen35TextModel || target is Qwen35
                    || target is any Gemma4DraftTarget || target is Gemma4TextModel),
            gemmaConfiguration == nil || target is any Gemma4DraftTarget
                || target is Gemma4TextModel,
            !requiresBF16Target
                || target.parameters().flattened().allSatisfy({ $0.1.dtype == .bfloat16 })
        else {
            throw ModelExecutionError.invalidOptions(
                "draft requires matching target hidden size, layers and vocabulary; LFM2 DSpark additionally requires an unquantized BF16 LFM2 target"
            )
        }
    }

}

private final class DSparkAttention: Module {
    let config: DSparkConfiguration
    let slidingWindow: Int?
    let headDim: Int
    let kvHeads: Int
    let scale: Float
    let gemmaRoPE: RoPELayer?
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear?
    @ModuleInfo(key: "o_proj") var o: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    init(_ config: DSparkConfiguration, index: Int) {
        self.config = config
        slidingWindow = config.layerTypes[index] == "sliding_attention" ? config.slidingWindow : nil
        let gemma = config.gemmaConfiguration
        let sharedKV = gemma?.attentionKEqV == true && slidingWindow == nil
        headDim = gemma != nil && slidingWindow == nil ? gemma!.globalHeadDim : config.headDim
        kvHeads = sharedKV ? (gemma!.globalKVHeads ?? config.kvHeads) : config.kvHeads
        scale = gemma != nil ? 1 : pow(Float(headDim), -0.5)
        if let gemma {
            let params = gemma.ropeParameters[config.layerTypes[index]]
            gemmaRoPE = initializeRope(
                dims: headDim, base: params?["rope_theta"]?.asFloat() ?? 10000,
                traditional: gemma.ropeTraditional, scalingConfig: params,
                maxPositionEmbeddings: gemma.maxPositionEmbeddings)
        } else {
            gemmaRoPE = nil
        }
        _q.wrappedValue = Linear(config.hiddenSize, config.heads * headDim, bias: false)
        _k.wrappedValue = Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _v.wrappedValue = sharedKV ? nil : Linear(config.hiddenSize, kvHeads * headDim, bias: false)
        _o.wrappedValue = Linear(config.heads * headDim, config.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
    }
    func callAsFunction(
        _ x: MLXArray, context: MLXArray, rope: RoPE, cache: KVCache,
        projected: (MLXArray, MLXArray)?
    ) -> MLXArray {
        let rope: RoPELayer = gemmaRoPE ?? rope
        let batch = x.dim(0)
        let length = x.dim(1)
        var context = context
        var projected = projected
        var count = context.dim(1)
        if let slidingWindow, count > slidingWindow - 1 {
            let skip = count - (slidingWindow - 1)
            context = context[0..., skip...]
            projected = projected.map { ($0.0[0..., skip...], $0.1[0..., skip...]) }
            count -= skip
            (cache as! BaseKVCache).offset += skip
        }
        var queries = qNorm(q(x).reshaped(batch, length, config.heads, headDim)).transposed(
            0, 2, 1, 3)
        let rawContextKeys = projected?.0 ?? k(context)
        let rawProposalKeys = k(x)
        var contextKeys = kNorm(
            rawContextKeys.reshaped(batch, count, kvHeads, headDim)
        ).transposed(0, 2, 1, 3)
        let contextValues = (projected?.1 ?? v.map { $0(context) } ?? rawContextKeys).reshaped(
            batch, count, kvHeads, headDim
        ).transposed(0, 2, 1, 3)
        var proposalKeys = kNorm(rawProposalKeys.reshaped(batch, length, kvHeads, headDim))
            .transposed(0, 2, 1, 3)
        let proposalValues = (v.map { $0(x) } ?? rawProposalKeys).reshaped(
            batch, length, kvHeads, headDim
        )
        .transposed(0, 2, 1, 3)
        queries = rope(queries, offset: cache.offset + count)
        contextKeys = rope(contextKeys, offset: cache.offset)
        proposalKeys = rope(proposalKeys, offset: cache.offset + count)
        let (keys, values) = cache.update(keys: contextKeys, values: contextValues)
        // Proposals attend bidirectionally, but only committed target context persists.
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: concatenated([keys, proposalKeys], axis: 2),
            values: concatenated([values, proposalValues], axis: 2),
            scale: scale, mask: .none)
        return o(output.transposed(0, 2, 1, 3).reshaped(batch, length, -1))
    }
}

private final class DSparkLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: DSparkAttention
    let mlp: Qwen3MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFF: RMSNorm?
    @ModuleInfo(key: "post_feedforward_layernorm") var postFF: RMSNorm?
    @ParameterInfo(key: "layer_scalar") var layerScalar: MLXArray?
    init(_ config: DSparkConfiguration, index: Int) {
        if config.gemmaConfiguration != nil {
            _preFF.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
            _postFF.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
            _layerScalar.wrappedValue = MLXArray.ones([1])
        }
        _attention.wrappedValue = DSparkAttention(config, index: index)
        mlp = Qwen3MLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
    }
    func callAsFunction(
        _ x: MLXArray, context: MLXArray, rope: RoPE, cache: KVCache,
        projected: (MLXArray, MLXArray)?
    ) -> MLXArray {
        if let preFF, let postFF, let layerScalar {
            let h =
                x
                + postNorm(
                    attention(
                        inputNorm(x), context: context, rope: rope, cache: cache,
                        projected: projected))
            let ff = preFF(h)
            return (h + postFF(mlp.down(geluApproximate(mlp.gate(ff)) * mlp.up(ff)))) * layerScalar
        }
        let h =
            x
            + attention(
                inputNorm(x), context: context, rope: rope, cache: cache, projected: projected)
        return h + mlp(postNorm(h))
    }
}

private final class DSparkMarkovHead: Module {
    @ModuleInfo(key: "markov_w1") var embedding: Embedding
    @ModuleInfo(key: "markov_w2") var projection: Linear
    init(_ config: DSparkConfiguration) {
        _embedding.wrappedValue = Embedding(
            embeddingCount: config.vocabularySize, dimensions: config.markovRank)
        _projection.wrappedValue = Linear(config.markovRank, config.vocabularySize, bias: false)
    }
    func callAsFunction(_ token: MLXArray) -> MLXArray { projection(embedding(token)) }
}

private final class DSparkConfidenceHead: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    init(_ config: DSparkConfiguration) {
        _proj.wrappedValue = Linear(config.hiddenSize + config.markovRank, 1)
    }
}

private final class DSparkContextProjection { var weight: MLXArray? }

final class DSparkModel: Module, DFlashDrafter {
    let config: DSparkConfiguration
    @ModuleInfo(key: "fc") var fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    private let layers: [DSparkLayer]
    let norm: RMSNorm
    @ModuleInfo(key: "embed_tokens") private var embedTokens: Embedding?
    @ModuleInfo(key: "lm_head") private var lmHead: Linear?
    @ModuleInfo(key: "markov_head") private var markov: DSparkMarkovHead?
    @ModuleInfo(key: "confidence_head") private var confidence: DSparkConfidenceHead?
    let rope: RoPE
    private let contextProjection = DSparkContextProjection()

    init(_ config: DSparkConfiguration) {
        self.config = config
        _fc.wrappedValue = Linear(
            config.hiddenSize * config.targetLayers.count, config.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        layers = (0 ..< config.layerCount).map { DSparkLayer(config, index: $0) }
        norm = RMSNorm(dimensions: config.hiddenSize, eps: config.normEps)
        if config.gemmaConfiguration != nil {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
        _markov.wrappedValue = config.strategy == .dspark ? DSparkMarkovHead(config) : nil
        _confidence.wrappedValue = config.confidenceHead ? DSparkConfidenceHead(config) : nil
        rope = RoPE(
            dimensions: config.headDim, traditional: config.traditionalRoPE, base: config.ropeTheta)
    }
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var normalized: [String: MLXArray] = [:]
        for (key, value) in weights {
            let name = key.hasPrefix("model.") ? String(key.dropFirst(6)) : key
            // Preserve a colliding key as an unknown parameter so strict loading
            // throws instead of trapping or silently overwriting a weight.
            normalized[normalized[name] == nil ? name : "__duplicate." + key] = value
        }
        if normalized["embed_tokens.weight"] != nil, embedTokens == nil {
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: config.vocabularySize, dimensions: config.hiddenSize)
        }
        if normalized["lm_head.weight"] != nil, lmHead == nil {
            _lmHead.wrappedValue = Linear(config.hiddenSize, config.vocabularySize, bias: false)
        }
        return normalized
    }
    func prepareContextProjection() {
        contextProjection.weight = nil
        guard config.gemmaConfiguration == nil,
            !layers.contains(where: {
                $0.attention.k is QuantizedLinear || $0.attention.v is QuantizedLinear
            })
        else { return }
        contextProjection.weight = concatenated(
            layers.flatMap { [$0.attention.k.weight, $0.attention.v!.weight] }, axis: 0)
        eval(contextProjection.weight!)
    }
    func newCache() -> [KVCache] {
        config.layerTypes.map { type in
            type == "sliding_attention"
                ? RotatingKVCache(maxSize: config.slidingWindow! - 1) : KVCacheSimple()
        }
    }
    func propose(
        seed: MLXArray, features: MLXArray, count: Int,
        target: any DFlashTarget, cache: [KVCache],
        sampler: any LogitSampler = ArgMaxSampler()
    ) throws -> MLXArray {
        try Task.checkCancellation()
        let skipAnchor = !config.sampleFromAnchor
        let input = concatenated(
            [
                seed,
                MLXArray.full(
                    [count - (skipAnchor ? 0 : 1)], values: MLXArray(Int32(config.maskToken)),
                    dtype: .int32),
            ], axis: 0)
        let embedded = (embedTokens ?? target.draftEmbedding)(input[.newAxis])
        let scale =
            config.gemmaConfiguration != nil
            ? Float(config.hiddenSize).squareRoot()
            : (embedTokens == nil ? target.draftEmbeddingScale : 1)
        var hidden = (embedded * scale).asType(hiddenNorm.weight.dtype)
        let context = hiddenNorm(fc(features.asType(hiddenNorm.weight.dtype)))
        let projections = contextProjection.weight.map {
            matmul(context, $0.T).split(parts: layers.count * 2, axis: -1)
        }
        for index in layers.indices {
            try Task.checkCancellation()
            hidden = layers[index](
                hidden, context: context, rope: rope, cache: cache[index],
                projected: projections.map { ($0[2 * index], $0[2 * index + 1]) })
        }
        let normalized = norm(hidden)
        var logits =
            lmHead.map { $0(normalized) } ?? embedTokens.map { $0.asLinear(normalized) }
            ?? target.draftReadout(normalized)
        if let cap = config.finalLogitSoftcapping {
            logits = tanh(logits / cap) * cap
        }
        var previous = seed
        var tokens: [MLXArray] = []
        for index in 0 ..< count {
            let base = logits[0, index + (skipAnchor ? 1 : 0), 0...]
            previous = sampler.sample(logits: markov.map { base + $0(previous)[0] } ?? base)
                .asType(.int32).reshaped(1)
            tokens.append(previous)
            asyncEval(previous)
        }
        return concatenated(tokens, axis: 0)
    }
}
