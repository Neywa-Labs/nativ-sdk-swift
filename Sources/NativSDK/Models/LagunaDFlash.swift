import CoreFoundation
import Foundation
import MLX
import MLXNN

struct LagunaDFlashConfiguration: DFlashDraftConfiguration {
    let backbone: LagunaConfiguration
    let targetLayers: [Int]
    let targetLayerCount: Int
    let proposalCount: Int
    let maskToken: Int
    var strategy: SpeculativeGenerationStatistics.Strategy { .dflash }
    var defaultProposalCount: Int { proposalCount }

    static func matches(_ data: Data) throws -> Bool {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return false
        }
        return root["model_type"] as? String == "laguna" && root["dflash_config"] != nil
    }

    init(data: Data) throws {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            root["model_type"] as? String == "laguna",
            let draft = root["dflash_config"] as? [String: Any]
        else { throw ModelExecutionError.invalidOptions("Laguna DFlash requires dflash_config") }
        func invalid(_ field: String) -> ModelExecutionError {
            .invalidOptions("unsupported Laguna DFlash configuration: \(field)")
        }
        func integer(_ key: String, in values: [String: Any]) throws -> Int {
            guard let n = values[key] as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID(),
                n.doubleValue.isFinite, n.doubleValue == Double(n.intValue)
            else { throw invalid(key) }
            return n.intValue
        }
        func indices(_ key: String, in values: [String: Any]) throws -> [Int] {
            guard let list = values[key] as? [Any] else { throw invalid(key) }
            return try list.map { try integer(key, in: [key: $0]) }
        }
        for key in [
            "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "head_dim", "vocab_size", "draft_vocab_size",
            "max_position_embeddings", "sliding_window",
        ] {
            guard try integer(key, in: root) > 0 else { throw invalid(key) }
        }
        let layers = try integer("num_hidden_layers", in: root)
        guard let types = root["layer_types"] as? [String], types.count == layers,
            types.allSatisfy({ $0 == "sliding_attention" }),
            root["gating"] as? String == "per-head",
            let causal = draft["causal"] as? NSNumber,
            CFGetTypeID(causal) == CFBooleanGetTypeID(), causal.boolValue,
            (root["hidden_act"] as? String ?? "silu") == "silu",
            (root["num_experts"] as? Int ?? 0) == 0
        else {
            throw invalid(
                "requires causal sliding attention, per-head gating and a dense SwiGLU draft")
        }
        if let heads = root["num_attention_heads_per_layer"] {
            let values = try indices("heads", in: ["heads": heads])
            guard values.count == layers,
                values.allSatisfy({ $0 == root["num_attention_heads"] as? Int })
            else { throw invalid("num_attention_heads_per_layer") }
        }
        backbone = try JSONDecoder().decode(LagunaConfiguration.self, from: data)
        targetLayerCount = try integer("num_target_layers", in: draft)
        let blockSize = try integer("block_size", in: draft)
        guard blockSize > 1 else { throw invalid("block_size") }
        proposalCount = blockSize - 1
        maskToken = try integer("mask_token_id", in: draft)
        targetLayers = try indices("target_layer_ids", in: draft)
        let auxiliary = try indices("eagle_aux_hidden_state_layer_ids", in: root)
        let window = backbone.slidingWindow!
        guard targetLayerCount > 0, window > 1,
            backbone.attentionHeads % backbone.kvHeads == 0, backbone.headDim % 2 == 0,
            backbone.rmsNormEps.isFinite, backbone.rmsNormEps > 0,
            backbone.ropeTheta.isFinite, backbone.ropeTheta > 0,
            !backbone.attentionBias, !backbone.qkvBias, !backbone.swaAttentionSinkEnabled,
            try integer("draft_vocab_size", in: root) == backbone.vocabularySize,
            (0 ..< backbone.vocabularySize).contains(maskToken),
            targetLayers.count == layers, targetLayers == Array(Set(targetLayers)).sorted(),
            targetLayers.allSatisfy({ (0 ..< targetLayerCount).contains($0) }),
            auxiliary == targetLayers.map({ $0 + 1 })
        else { throw invalid("geometry, vocabulary or target layers") }
        if root["sliding_windows"] != nil {
            guard
                try indices("sliding_windows", in: root) == Array(repeating: window, count: layers)
            else { throw invalid("sliding_windows") }
        }
        // Published Laguna drafts use full, unscaled RoPE. Reject unsupported
        // variants instead of silently applying the target's different SWA RoPE.
        let rope = backbone.ropeParameters(for: "sliding_attention")
        guard case .string("default") = rope["rope_type"],
            (rope["partial_rotary_factor"]?.asFloat() ?? 1) == 1,
            Set(rope.keys).isSubset(of: ["rope_type", "rope_theta", "partial_rotary_factor"]),
            let theta = rope["rope_theta"]?.asFloat(), theta.isFinite, theta > 0
        else { throw invalid("rotary position encoding") }
    }

    func validate(target: any DFlashTarget) throws {
        guard target is Laguna, target.draftHiddenSize == backbone.hiddenSize,
            target.draftLayerCount == targetLayerCount,
            target.vocabularySize == backbone.vocabularySize
        else {
            throw ModelExecutionError.invalidOptions(
                "Laguna DFlash requires a Laguna target with matching hidden size, layer count and vocabulary"
            )
        }
    }
}

private final class LagunaDFlashAttention: Module {
    let config: LagunaConfiguration
    @ModuleInfo(key: "qkv_proj") var qkv: Linear
    @ModuleInfo(key: "o_proj") var output: Linear
    @ModuleInfo(key: "g_proj") var gate: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: LagunaConfiguration) {
        self.config = config
        _qkv.wrappedValue = Linear(
            config.hiddenSize, (config.attentionHeads + 2 * config.kvHeads) * config.headDim,
            bias: false)
        _output.wrappedValue = Linear(
            config.attentionHeads * config.headDim, config.hiddenSize, bias: false)
        _gate.wrappedValue = Linear(config.hiddenSize, config.attentionHeads, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray, rope: RoPELayer, cache: KVCache)
        -> MLXArray
    {
        let batch = x.dim(0)
        let length = x.dim(1)
        var context = context
        let window = config.slidingWindow!
        if context.dim(1) >= window {
            let skip = context.dim(1) - (window - 1)
            context = context[0..., skip..., 0...]
            (cache as! BaseKVCache).offset += skip
        }
        let count = context.dim(1)
        let qWidth = config.attentionHeads * config.headDim
        let kvWidth = config.kvHeads * config.headDim
        let block = qkv(x)
        let projectedContext = qkv(context)
        func heads(_ value: MLXArray, _ n: Int) -> MLXArray {
            value.reshaped(batch, -1, n, config.headDim)
        }
        var queries = qNorm(heads(block[.ellipsis, ..<qWidth], config.attentionHeads)).transposed(
            0, 2, 1, 3)
        var contextKeys = kNorm(
            heads(projectedContext[.ellipsis, qWidth ..< (qWidth + kvWidth)], config.kvHeads)
        ).transposed(0, 2, 1, 3)
        var proposalKeys = kNorm(
            heads(block[.ellipsis, qWidth ..< (qWidth + kvWidth)], config.kvHeads)
        ).transposed(0, 2, 1, 3)
        let contextValues = heads(
            projectedContext[.ellipsis, (qWidth + kvWidth)...], config.kvHeads
        ).transposed(0, 2, 1, 3)
        let proposalValues = heads(block[.ellipsis, (qWidth + kvWidth)...], config.kvHeads)
            .transposed(0, 2, 1, 3)
        queries = rope(queries, offset: cache.offset + count)
        contextKeys = rope(contextKeys, offset: cache.offset)
        proposalKeys = rope(proposalKeys, offset: cache.offset + count)
        let (keys, values) = cache.update(keys: contextKeys, values: contextValues)
        // All retained context is visible; proposals can only see earlier proposals.
        let query = MLXArray(0 ..< length)[.newAxis, 0...].T
        let key = MLXArray(0 ..< (keys.dim(2) + length))[.newAxis]
        let mask = key .<= query + keys.dim(2)
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: concatenated([keys, proposalKeys], axis: 2),
            values: concatenated([values, proposalValues], axis: 2),
            scale: pow(Float(config.headDim), -0.5), mask: .array(mask)
        )
        .transposed(0, 2, 1, 3)
        let gates = softplus(gate(x).asType(.float32)).asType(attended.dtype)
        return output((attended * gates[.ellipsis, .newAxis]).reshaped(batch, length, -1))
    }
}

private final class LagunaDFlashLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: LagunaDFlashAttention
    let mlp: Qwen3MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    init(_ config: LagunaConfiguration) {
        _attention.wrappedValue = LagunaDFlashAttention(config)
        mlp = Qwen3MLP(dimensions: config.hiddenSize, hiddenDimensions: config.intermediateSize)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }
    func callAsFunction(_ x: MLXArray, context: MLXArray, rope: RoPELayer, cache: KVCache)
        -> MLXArray
    {
        let h = x + attention(inputNorm(x), context: context, rope: rope, cache: cache)
        return h + mlp(postNorm(h))
    }
}

final class LagunaDFlashModel: Module, DFlashDrafter {
    let config: LagunaDFlashConfiguration
    private let layers: [LagunaDFlashLayer]
    @ModuleInfo(key: "aux_hidden_norms") var auxiliaryNorms: [RMSNorm]
    let fc: Linear
    @ModuleInfo(key: "hidden_norm") var hiddenNorm: RMSNorm
    let norm: RMSNorm
    let rope: RoPELayer

    init(_ config: LagunaDFlashConfiguration) {
        self.config = config
        let b = config.backbone
        layers = (0 ..< b.hiddenLayers).map { _ in LagunaDFlashLayer(b) }
        _auxiliaryNorms.wrappedValue = config.targetLayers.map { _ in
            RMSNorm(dimensions: b.hiddenSize, eps: b.rmsNormEps)
        }
        fc = Linear(b.hiddenSize * config.targetLayers.count, b.hiddenSize, bias: false)
        _hiddenNorm.wrappedValue = RMSNorm(dimensions: b.hiddenSize, eps: b.rmsNormEps)
        norm = RMSNorm(dimensions: b.hiddenSize, eps: b.rmsNormEps)
        rope = initializeRope(
            dims: b.headDim, base: b.ropeTheta, traditional: false,
            scalingConfig: b.ropeParameters(for: "sliding_attention"),
            maxPositionEmbeddings: b.maxPositionEmbeddings)
    }

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var normalized: [String: MLXArray] = [:]
        for (key, value) in weights {
            let name = key.hasPrefix("model.") ? String(key.dropFirst(6)) : key
            normalized[normalized[name] == nil ? name : "__duplicate." + key] = value
        }
        // Fuse all corresponding quantization tensors as well as the weights.
        // Incomplete or colliding layouts remain unknown keys for strict loading.
        for index in layers.indices {
            let prefix = "layers.\(index).self_attn."
            for suffix in ["weight", "scales", "biases"] {
                let keys = ["q_proj", "k_proj", "v_proj"].map { prefix + $0 + "." + suffix }
                let destination = prefix + "qkv_proj." + suffix
                if normalized[destination] == nil, keys.allSatisfy({ normalized[$0] != nil }) {
                    normalized[destination] = concatenated(
                        keys.map { normalized.removeValue(forKey: $0)! }, axis: 0)
                }
            }
        }
        return normalized
    }

    func newCache() -> [KVCache] {
        layers.map { _ in RotatingKVCache(maxSize: config.backbone.slidingWindow! - 1) }
    }

    func propose(
        seed: MLXArray, features: MLXArray, count: Int,
        target: any DFlashTarget, cache: [KVCache], sampler: any LogitSampler = ArgMaxSampler()
    ) throws -> MLXArray {
        try Task.checkCancellation()
        guard (1 ... config.proposalCount).contains(count), cache.count == layers.count,
            features.dim(-1) == config.backbone.hiddenSize * config.targetLayers.count
        else { throw ModelExecutionError.invalidOptions("invalid Laguna DFlash proposal inputs") }
        let tokens = concatenated([
            seed, MLXArray.full([count], values: MLXArray(Int32(config.maskToken)), dtype: .int32),
        ])
        var hidden = (target.draftEmbedding(tokens[.newAxis]) * target.draftEmbeddingScale)
            .asType(hiddenNorm.weight.dtype)
        let slices = features.asType(hiddenNorm.weight.dtype).split(
            parts: auxiliaryNorms.count, axis: -1)
        let context = hiddenNorm(
            fc(concatenated(zip(auxiliaryNorms, slices).map { $0($1) }, axis: -1)))
        for index in layers.indices {
            try Task.checkCancellation()
            hidden = layers[index](hidden, context: context, rope: rope, cache: cache[index])
        }
        let logits = target.draftReadout(norm(hidden))
        return concatenated(
            (1 ... count).map { index in
                sampler.sample(logits: logits[0, index, 0...]).asType(.int32).reshaped(1)
            })
    }
}
